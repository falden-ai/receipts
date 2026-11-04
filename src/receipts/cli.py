"""The `receipts` command and the composition root. Each lane's `register(sub)` adds its subcommands, each setting
`func` to a handler that returns the exit code; the integrator wires a lane in here when it lands, with the real
implementations behind the contracts it codes against.

- core: `prices`, `spend`, `spend arm`, `claims` and `run`. `run` runs on CoreWiring: the Meta client for intake and
  planning, a router that sends each arm's attacker model to its client, the attack lane's attackers, the surface
  lane's stage-2 attacker, the surface lane's surrogate fixer (SK3: LoopSurrogateBuilder on the same router, which
  then also serves its fixed muse-spark-1.3; `run --no-surrogates` leaves it out, so fix-validity catches stay
  s_sat_inconclusive), the core lane's refuter (SK2: Refuter on the Meta client, muse-wonder-1.0, on every blocker;
  `run --no-refuter` leaves it out, so blockers stay as SK1 and SK3 left them), AT4 (DifferentialAttacker) on the
  differential lens when `run --lenses` names it (the default, regression and correctness, plans Tier A as before
  and wires no differential attacker), the surface lane's JSONL ledger for
  each arm, and for each target a run phase of its own (`run_phase`: a new run-phase sandbox and a fresh runner
  mapping, released with the sandbox's cleanup once the target is done). No two targets run on one sandbox, so
  `run --jobs N` runs N targets at once, up to RUN_MAX_JOBS.
- surface: `replay`, `ledger rollup`, `ticket` (one blocker drafted as a Linear ticket; `--post` files it), `check
  post` (a Receipts result as a GitHub check) and `receipt html` (a static receipt page for one target).
- agentloop: `fix` (FX1: fix one confirmed blocker with Spark and try a fresh attack on the combined patch), on
  FixWiring: the fixer's client (`fix_client`: SparkFixer's muse-spark-1.3 routed to the Meta client), a run phase of
  its own (`run_phase`, as for a target of `receipts run`) and the fresh attack (FixReAttack: the `receipts run`
  pipeline on the combined patch, with the fixer's fresh seeds, on new seams and a new run phase, and with no plan
  caps, so every planned attack runs: REATTACK_PLAN_CAPS).
- eval: `eval` and its subcommands: the Tier A data, R24, the label runs, the harness check, the sample, the pilot and
  score. `eval r24` and `eval score` run on the eval lane's Wiring: each sandbox they need is a new
  `run_phase_sandbox()` and each runner mapping a fresh `run_phase_runners()`.

Nothing that needs a key, the network or Docker is built until a run starts, so `--help` and every command that calls
no model work with no key set. Secrets and endpoints come only from the environment, by exact name, when a run builds
its clients: FALDEN_META_API_KEY (the Meta client), RECEIPTS_GLIMMER_URL (the Glimmer client, built only when an arm
attacks with muse-glimmer-30b) and RECEIPTS_SPEND_FILE (the spend meter the Meta client holds every call to).
`ticket --post` reads RECEIPTS_LINEAR_API_KEY. `check post --post` reads RECEIPTS_GITHUB_APP_ID,
RECEIPTS_GITHUB_APP_INSTALLATION_ID and RECEIPTS_GITHUB_APP_KEY_FILE.

The run phase's sandbox is the attack lane's DockerSandbox under gVisor runsc, never runc, with the run phase's flags
(contracts.RUN_PHASE_LIMITS, its default) and, for a Tier A target, the image's own /testbed as the workspace.
`run_phase_sandbox` and `run_phase_runners` build a new sandbox and a fresh runner mapping on every call, so no two
callers share one; `run_phase` puts one of each together for a target of `receipts run`.

Two integrator calls of 2026-10-28, for the founder to confirm: the run phase stays on runsc=True with no runc
fallback, so a host without gVisor fails a run instead of running it under another runtime; and `receipts run`
runs at most RUN_MAX_JOBS targets at once, 16: the eval VM's 16 sandboxes and the spec's Meta concurrency of 16 per
run.
"""

from __future__ import annotations

import argparse
import functools
import json
import re
import sys
import threading
from collections.abc import Callable, Sequence
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, Final

from receipts import __version__
from receipts.agentloop import fix_command
from receipts.agentloop.attacker import AttackTrace, LoopAttacker
from receipts.agentloop.fixloop import FIX_CAP_MICRO_USD, FIXER_MODEL, FixStageError, ReAttackResult
from receipts.agentloop.surrogate import SURROGATE_MODEL, LoopSurrogateBuilder
from receipts.attack import (
    CorrectnessAttacker,
    DifferentialAttacker,
    DjangoRunner,
    DockerSandbox,
    PytestRunner,
    RegressionAttacker,
    SympyRunner,
)
from receipts.attack.correctness import MAX_TOKENS as CORRECTNESS_MAX_TOKENS
from receipts.contracts import (
    Arm,
    Attack,
    Attacker,
    AttackOutcome,
    Finding,
    Ledger,
    LedgerEntry,
    Lens,
    ModelClient,
    ModelRequest,
    ModelResponse,
    RunnerKind,
    Sandbox,
    StageCost,
    Status,
    SuiteRunner,
    Target,
    Tier,
    UncheckedReason,
)
from receipts.core import cli_commands as core_commands
from receipts.core.glimmer_client import GLIMMER_MODEL, GlimmerClient, GlimmerConfigError
from receipts.core.meta_client import Hold, MetaClient, MissingKeyError, SpendCapError, SpendMeter, shared_meter
from receipts.core.pipeline import (
    TARGET_CAP_MICRO_USD,
    RunPhase,
    Seams,
    TargetSummary,
    ledger_entry,
    run_target,
    summary_path,
)
from receipts.core.planner import TIER_A_LENSES, PlanCaps, TemperatureMode
from receipts.core.refuter import Refuter
from receipts.core.routing import ModelRouter
from receipts.eval import cli as eval_commands
from receipts.surface import check_command, receipt_command, ticket_command
from receipts.surface import cli_commands as surface_commands
from receipts.surface.ledger import JsonlLedger
from receipts.surface.text import visible

if TYPE_CHECKING:
    Subparsers = argparse._SubParsersAction[argparse.ArgumentParser]

# The attack lane's settings for the pilot: AT3 has the spec's 15 minutes, a minute of it kept for its judgment call,
# and each AT2 stage-1 candidate run has 60 s.
REGRESSION_WALL_S: Final = 900.0
REGRESSION_JUDGE_RESERVE_S: Final = 60.0
CORRECTNESS_RUN_TIMEOUT_S: Final = 60.0
# The lenses `receipts run --lenses` may name: those an attacker is wired for here.
RUN_LENSES: Final = (Lens.REGRESSION, Lens.CORRECTNESS, Lens.DIFFERENTIAL)
TRACES_DIR: Final = "traces"  # under the ledger directory, apart from the arms' ledgers
_ARM_NAME: Final = re.compile(r"[a-z0-9][a-z0-9_-]{0,31}")  # contracts.Arm.name
# `receipts run` runs at most this many targets at once, each on a run phase of its own (run_phase): the eval VM's 16
# sandboxes and the spec's Meta concurrency of 16 per run (SPEC.md, parameters).
RUN_MAX_JOBS: Final = 16
# `receipts fix`'s fresh attack: its ledgers, traces and summaries go here, under the fix's --out-dir.
REATTACK_DIR: Final = "reattack"
# The fresh attack's temperature mode. ReAttack is handed the target and the arm, not the run file's row, so it takes
# the planner's spec mode: each claim's first attack at 0.0 and the rest at 0.8.
REATTACK_TEMPERATURES: Final = TemperatureMode.SPEC
# The fresh attack's plan caps: none (REG-616, the founder's of 2026-10-31). `receipts run` keeps at most 4 lenses and
# 3 attacks per lens and lists the rest unchecked (budget), which could leave out every attack on the claim whose
# blocker the fix addresses (all 3 on pandas #64677); the fresh attack runs every attack its plan holds, bounded only
# by its dollar caps (fixloop.REATTACK_CAP_MICRO_USD and FIX_CAP_MICRO_USD).
REATTACK_PLAN_CAPS: Final = PlanCaps(max_lenses=None, max_attacks_per_lens=None)
# FAL-635: only the fix's fresh attack gets more output room. Exploration and all other callers keep their limits.
REATTACK_STAGE1_MAX_TOKENS: Final = 32_000
REATTACK_WRITE_MAX_TOKENS: Final = 16_384


def run_phase_sandbox() -> DockerSandbox:
    """A new run-phase sandbox on every call, never a shared one: gVisor runsc, never runc, so a host without it fails
    the run instead of running it under another runtime; the run phase's flags; and no workspace path, so a Tier A
    target runs in its image's /testbed."""
    return DockerSandbox(runsc=True)


def run_phase_runners() -> dict[RunnerKind, SuiteRunner]:
    """A fresh runner for each RunnerKind on every call. The sandbox binds each one to the workspace root."""
    return {RunnerKind.PYTEST: PytestRunner(), RunnerKind.DJANGO: DjangoRunner(), RunnerKind.SYMPY: SympyRunner()}


def run_phase() -> RunPhase:
    """One target's run phase for `receipts run`, new on every call: a new run-phase sandbox, a fresh runner mapping,
    and the sandbox's cleanup as its release, so a container an attacker left open and the trees the sandbox copied
    out of the image go when the target is done, not when the run ends."""
    sandbox = run_phase_sandbox()
    return RunPhase(sandbox=sandbox, runners=run_phase_runners(), release=sandbox.cleanup)


class ExistingTestsOnly:
    """The regression lens's first attacker: AT3, for the plan's one attack per target that reruns the existing tests
    (contracts.Attack.existing_tests). Any other regression attack gets an empty outcome here, so the pipeline goes on
    to the lens's model attackers, which leave AT3's attack alone. AT3 itself is unchanged."""

    lens = Lens.REGRESSION

    def __init__(self, suite: Attacker) -> None:
        self.suite = suite

    def attack(
        self,
        target: Target,
        attack: Attack,
        *,
        arm: Arm,
        client: ModelClient,
        sandbox: Sandbox,
        runner: SuiteRunner,
    ) -> AttackOutcome:
        if not attack.existing_tests:
            return AttackOutcome(attack_id=attack.id)
        return self.suite.attack(target, attack, arm=arm, client=client, sandbox=sandbox, runner=runner)


class CoreWiring:
    """`receipts run`'s wiring, one per parser: the seams, built once a run's arms are read, and each arm's ledger.
    Each stage-2 attack's trace goes to <ledger dir>/traces/<arm>.jsonl, beside the arms' ledgers and never among them,
    so a directory rollup never reads a trace as an arm. `surrogates` is False for `run --no-surrogates`, and
    `refuter` False for `run --no-refuter`. `lenses` are `run --lenses`: a Tier A target's lenses, and the
    differential attacker is wired only when they name it."""

    def __init__(
        self,
        *,
        surrogates: bool = True,
        refuter: bool = True,
        lenses: Sequence[Lens] = TIER_A_LENSES,
        budget: _FreshAttackMeter | None = None,
        stage_one_max_tokens: int = CORRECTNESS_MAX_TOKENS,
        write_max_tokens: int | None = None,
    ) -> None:
        self._lock = threading.Lock()
        self._ledger_dir: Path | None = None
        self.surrogates = surrogates
        self.refuter = refuter
        self.lenses = tuple(lenses)
        self.budget = budget
        self.stage_one_max_tokens = stage_one_max_tokens
        self.write_max_tokens = write_max_tokens

    def seams(self, arms: Sequence[Arm]) -> Seams:
        """The run's seams. The Meta client reads FALDEN_META_API_KEY here, and the Glimmer client RECEIPTS_GLIMMER_URL
        only when an arm attacks with muse-glimmer-30b; either one missing stops the run before anything runs. Seams'
        policy, case names and caps keep their defaults: a repro names its own cases (Repro.cases). The surrogate
        fixer, unless turned off, runs on the attacker router, with SURROGATE_MODEL routed to the Meta client."""
        meta = MetaClient() if self.budget is None else MetaClient(meter=self.budget)
        routes: dict[str, ModelClient] = {}
        for arm in arms:
            if arm.attacker_model not in routes:
                if arm.attacker_model == GLIMMER_MODEL:
                    glimmer = GlimmerClient()
                    routes[arm.attacker_model] = (
                        glimmer if self.budget is None else _FreeBudgetClient(glimmer, self.budget)
                    )
                else:
                    routes[arm.attacker_model] = meta
        if self.surrogates:
            routes.setdefault(SURROGATE_MODEL, meta)
        router = ModelRouter(routes)
        attackers: dict[Lens, tuple[Attacker, ...]] = {
            # AT3 for the plan's existing-tests attack; stage 1, then stage 2 on the regression lens for the rest.
            Lens.REGRESSION: (
                ExistingTestsOnly(
                    RegressionAttacker(wall_s=REGRESSION_WALL_S, judge_reserve_s=REGRESSION_JUDGE_RESERVE_S)
                ),
                CorrectnessAttacker(
                    lens=Lens.REGRESSION,
                    run_timeout_s=CORRECTNESS_RUN_TIMEOUT_S,
                    max_tokens=self.stage_one_max_tokens,
                ),
                LoopAttacker(lens=Lens.REGRESSION, on_trace=self.trace, write_max_tokens=self.write_max_tokens),
            ),
            Lens.CORRECTNESS: (
                CorrectnessAttacker(run_timeout_s=CORRECTNESS_RUN_TIMEOUT_S, max_tokens=self.stage_one_max_tokens),
                LoopAttacker(on_trace=self.trace, write_max_tokens=self.write_max_tokens),
            ),
        }
        if Lens.DIFFERENTIAL in self.lenses:
            attackers[Lens.DIFFERENTIAL] = (DifferentialAttacker(),)
        return Seams(
            intake_client=meta,
            planner_client=meta,
            attacker_client=router,
            surrogates=(
                LoopSurrogateBuilder(router, remaining_micro_usd=self.budget.remaining)
                if self.budget is not None
                else LoopSurrogateBuilder(router)
            )
            if self.surrogates
            else None,
            refuter=Refuter(meta) if self.refuter else None,
            run_phase=run_phase,
            attackers=attackers,
            tier_a_lenses=self.lenses,
            target_caps=(
                TARGET_CAP_MICRO_USD if self.budget is None else {tier: self.budget.cap_micro_usd for tier in Tier}
            ),
        )

    def ledger(self, path: Path) -> Ledger:
        """An arm's ledger, created when missing; its directory is where the traces go."""
        with self._lock:
            self._ledger_dir = path.parent
        return JsonlLedger(path)

    def trace(self, trace: AttackTrace) -> None:
        """Append one stage-2 attack's trace to its arm's trace file. LoopAttacker logs anything this raises and the
        attack goes on."""
        if _ARM_NAME.fullmatch(trace.arm) is None:
            raise ValueError(f"a trace names {trace.arm!r}, which is not an arm name")
        line = trace.model_dump_json() + "\n"
        with self._lock:
            if self._ledger_dir is None:
                raise RuntimeError(f"no arm ledger is open, so the trace of {trace.attack_id} has nowhere to go")
            folder = self._ledger_dir / TRACES_DIR
            folder.mkdir(parents=True, exist_ok=True)
            with (folder / f"{trace.arm}.jsonl").open("a", encoding="utf-8") as out:
                out.write(line)


def fix_client() -> ModelClient:
    """SparkFixer's client for `receipts fix`: a router that serves only the fixer's model, muse-spark-1.3, from the
    Meta client, which reads FALDEN_META_API_KEY here, after the fix command has checked its blocker. The fixer's own
    calls, including retries, are hard-capped at FIX_CAP_MICRO_USD, counting settled spend plus open holds."""
    return ModelRouter({FIXER_MODEL: MetaClient(meter=_FixBudgetMeter(shared_meter(), FIX_CAP_MICRO_USD))})


class _RecordingLedger:
    """An arm's ledger that also keeps every entry this fresh attack appends, so its findings are this run's alone,
    never an earlier fix's in the same file."""

    def __init__(self, ledger: Ledger) -> None:
        self.ledger = ledger
        self.appended: list[LedgerEntry] = []
        self._lock = threading.Lock()

    def append(self, entry: LedgerEntry) -> None:
        self.ledger.append(entry)
        with self._lock:
            self.appended.append(entry)

    def entries(self, target_id: str | None = None) -> Sequence[LedgerEntry]:
        return self.ledger.entries(target_id)

    def costs(self) -> list[StageCost]:
        with self._lock:
            return [cost for entry in self.appended for cost in entry.finding.costs]

    def findings(self) -> tuple[Finding, ...]:
        """Each finding in its latest state, in the order it was first recorded."""
        latest: dict[str, Finding] = {}
        with self._lock:
            for entry in self.appended:
                latest[entry.finding.id] = entry.finding
        return tuple(latest.values())


class _FixBudgetMeter(SpendMeter):
    """Hard-cap a fix stage's settled spend plus open holds, on top of the global meter, for every request attempt."""

    def __init__(self, parent: SpendMeter, budget_micro_usd: int, *, stage_name: str = "fix") -> None:
        if budget_micro_usd <= 0:
            raise ValueError(f"a {stage_name} needs a positive budget")
        self.parent = parent
        self.stage_name = stage_name
        self.cap_micro_usd = budget_micro_usd
        self._budget_lock = threading.Lock()
        self._budget_spent = 0
        self._budget_holds: dict[str, int] = {}
        self.refused = False

    def remaining(self) -> int:
        with self._budget_lock:
            return max(0, self.cap_micro_usd - self._budget_spent - sum(self._budget_holds.values()))

    @property
    def spent_micro_usd(self) -> int:
        with self._budget_lock:
            return self._budget_spent

    @property
    def held_micro_usd(self) -> int:
        with self._budget_lock:
            return sum(self._budget_holds.values())

    def reserve(self, worst_micro_usd: int, *, stage: str, model: str) -> Hold:
        with self._budget_lock:
            if worst_micro_usd < 0:
                raise ValueError("a worst case is never negative")
            if self._budget_spent + sum(self._budget_holds.values()) + worst_micro_usd > self.cap_micro_usd:
                self.refused = True
                raise SpendCapError(f"the request's worst case exceeds the {self.stage_name}'s remaining budget")
            try:
                hold = self.parent.reserve(worst_micro_usd, stage=stage, model=model)
            except SpendCapError:
                self.refused = True
                raise
            self._budget_holds[hold.id] = worst_micro_usd
            return hold

    def settle(self, hold: Hold, micro_usd: int, *, estimated: bool) -> None:
        try:
            self.parent.settle(hold, micro_usd, estimated=estimated)
        finally:
            with self._budget_lock:
                self._budget_holds.pop(hold.id)
                self._budget_spent += max(0, micro_usd)


class _FreshAttackMeter(_FixBudgetMeter):
    """One fresh attack's allowance, including every retry."""

    def __init__(self, parent: SpendMeter, budget_micro_usd: int) -> None:
        super().__init__(parent, budget_micro_usd, stage_name="fresh attack")


class _FreeBudgetClient:
    """Glimmer bills zero, so its worst-case reservation and settlement are both zero."""

    def __init__(self, client: ModelClient, meter: _FreshAttackMeter) -> None:
        self.client, self.meter = client, meter

    def complete(self, request: ModelRequest) -> ModelResponse:
        hold = self.meter.reserve(0, stage=request.stage, model=request.model)
        try:
            return self.client.complete(request)
        finally:
            self.meter.settle(hold, 0, estimated=False)


def _fresh_attack_error(exc: Exception) -> str:
    """Persist a cause without retaining model output, repository text or URLs."""
    if isinstance(exc, MissingKeyError):
        message = "FALDEN_META_API_KEY is missing or invalid"
    elif isinstance(exc, GlimmerConfigError):
        message = "RECEIPTS_GLIMMER_URL or Glimmer credentials are missing or invalid"
    elif isinstance(exc, SpendCapError):
        message = "a model request was refused by the spend meter"
    else:
        message = "fresh attack could not run"
    return f"{type(exc).__name__}: {message}"


class FixReAttack:
    """`receipts fix`'s fresh attack (fixloop.ReAttack), one per fix, built from the fix's --out-dir. Each call runs
    the `receipts run` pipeline (run_target) on the target it is handed, the combined patch, with no plan caps
    (REATTACK_PLAN_CAPS: every planned attack runs), stage-1 output at 32,000 tokens and write output at 16,384,
    and the arm it is
    handed, which carries the fixer's fresh seeds: new seams from a CoreWiring of its own (the run's default lenses,
    surrogates and refuter, and a new run phase from `run_phase`), the arm's ledger at
    <out-dir>/reattack/<finding sha256 prefix>-<patch sha256 prefix>/<arm>.jsonl, with traces and summaries
    in that same directory.

    The result is the findings this call recorded, each in its latest state, and the whole spend: the summary's
    costs (intake, planning and any cost no finding carries) plus every recorded finding's costs. When the pipeline
    raises, the same total from the failed summary and the findings recorded so far goes out on a FixStageError,
    empty when nothing was spent, so the fix's budget is always established."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir / REATTACK_DIR
        self._finding_id: str | None = None

    def validate(self, arm: Arm, *, finding_id: str | None = None) -> None:
        try:
            records = self.out_dir.parent / "fixes.jsonl"
            if finding_id is not None and records.exists():
                try:
                    with records.open(encoding="utf-8") as stream:
                        for number, line in enumerate(stream, start=1):
                            if not line.strip():
                                continue
                            try:
                                record = json.loads(line)
                            except ValueError as exc:
                                raise FixStageError(
                                    f"fixes.jsonl line {number} is not readable JSON", costs=()
                                ) from exc
                            if not isinstance(record, dict):
                                raise FixStageError(f"fixes.jsonl line {number} is not a JSON object", costs=())
                            if record.get("finding_id") == finding_id:
                                raise FixStageError(
                                    "a fix record already exists for this finding; use a fresh --out-dir", costs=()
                                )
                except (OSError, UnicodeError) as exc:
                    raise FixStageError("fixes.jsonl could not be read as UTF-8", costs=()) from exc
            CoreWiring().seams([arm])
            self._finding_id = finding_id
        except Exception as exc:
            print(visible(str(exc)), file=sys.stderr)
            if isinstance(exc, FixStageError):
                raise
            raise FixStageError(f"fresh attack preflight failed: {_fresh_attack_error(exc)}", costs=()) from exc

    def __call__(self, target: Target, arm: Arm, *, budget_micro_usd: int) -> ReAttackResult:
        budget = _FreshAttackMeter(shared_meter(), budget_micro_usd)
        wiring = CoreWiring(
            budget=budget,
            stage_one_max_tokens=REATTACK_STAGE1_MAX_TOKENS,
            write_max_tokens=REATTACK_WRITE_MAX_TOKENS,
        )
        finding_key = sha256((self._finding_id or target.id).encode("utf-8")).hexdigest()[:12]
        patch_key = sha256(target.patch.encode("utf-8")).hexdigest()[:12]
        out_dir = self.out_dir / f"{finding_key}-{patch_key}"
        try:
            out_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError as exc:
            print(visible(str(exc)), file=sys.stderr)
            raise FixStageError(
                "FileExistsError: fresh attack output already exists; use a fresh --out-dir",
                costs=(),
                out_dir=str(out_dir),
            ) from exc
        except OSError as exc:
            print(visible(str(exc)), file=sys.stderr)
            raise FixStageError(
                f"{type(exc).__name__}: fresh attack output could not be created", costs=(), out_dir=str(out_dir)
            ) from exc
        failed: list[TargetSummary] = []
        recording: _RecordingLedger | None = None
        try:
            recording = _RecordingLedger(wiring.ledger(out_dir / f"{arm.name}.jsonl"))
            seams = wiring.seams([arm])
            summary = run_target(
                target,
                temperatures=REATTACK_TEMPERATURES,
                arms=[arm],
                seams=seams,
                ledgers={arm.name: recording},
                on_failure=failed.append,
                plan_caps=REATTACK_PLAN_CAPS,
            )
        except Exception as exc:
            print(visible(str(exc)), file=sys.stderr)
            costs = [*(failed[0].costs if failed else ()), *(recording.costs() if recording is not None else ())]
            unassigned = max(0, budget.spent_micro_usd - sum(cost.micro_usd for cost in costs))
            if unassigned:
                cost = StageCost(stage="fresh-attack-unassigned", micro_usd=unassigned)
                costs.append(cost)
                if failed:
                    failed[0] = failed[0].model_copy(update={"costs": (*failed[0].costs, cost)})
            if failed:
                self._write(out_dir, failed[0], costs)
            raise FixStageError(
                f"the fresh attack failed: {_fresh_attack_error(exc)}",
                costs=costs,
                reason="budget" if budget.refused else "runtime",
                out_dir=str(out_dir),
            ) from exc
        if budget.refused:
            recording.append(
                ledger_entry(
                    Finding(
                        id=f"{target.id}/fresh-budget/refused",
                        target_id=target.id,
                        claim_id="fresh-budget",
                        lens=Lens.CORRECTNESS,
                        status=Status.UNCHECKED,
                        unchecked_reason=UncheckedReason.BUDGET,
                        detail="a model request's worst case did not fit the fresh attack budget",
                    ),
                    seams.now(),
                )
            )
            summary = summary.model_copy(
                update={
                    "arms": tuple(
                        item.model_copy(
                            update={
                                "findings": item.findings + 1,
                                "statuses": {
                                    **item.statuses,
                                    Status.UNCHECKED: item.statuses.get(Status.UNCHECKED, 0) + 1,
                                },
                            }
                        )
                        if item.arm == arm.name
                        else item
                        for item in summary.arms
                    )
                }
            )
        costs = [*summary.costs, *recording.costs()]
        unassigned = max(0, budget.spent_micro_usd - sum(cost.micro_usd for cost in costs))
        if unassigned:
            cost = StageCost(stage="fresh-attack-unassigned", micro_usd=unassigned)
            costs.append(cost)
            summary = summary.model_copy(update={"costs": (*summary.costs, cost)})
        self._write(out_dir, summary, costs)
        return ReAttackResult(
            findings=recording.findings(), micro_usd=sum(cost.micro_usd for cost in costs), out_dir=str(out_dir)
        )

    def _write(self, out_dir: Path, summary: TargetSummary, costs: Sequence[StageCost]) -> None:
        try:
            path = summary_path(out_dir / "summaries", summary.target_id)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(summary.model_dump_json(indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            print(visible(str(exc)), file=sys.stderr)
            raise FixStageError(
                f"{type(exc).__name__}: the fresh attack's summary could not be written",
                costs=costs,
                out_dir=str(out_dir),
            ) from exc


def at_most_max_jobs(run: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    """`receipts run` behind its --jobs cap: above RUN_MAX_JOBS it refuses before it reads a file or a key or builds
    anything; otherwise the core lane's handler runs as it is."""
    jobs: int = args.jobs
    if jobs > RUN_MAX_JOBS:
        print(
            f"receipts run: --jobs {jobs} is refused: at most {RUN_MAX_JOBS} targets run at once, each on a sandbox "
            "of its own (the eval VM's 16 sandboxes)",
            file=sys.stderr,
        )
        return 2
    return run(args)


def with_surrogate_choice(
    wiring: CoreWiring, run: Callable[[argparse.Namespace], int], args: argparse.Namespace
) -> int:
    """`receipts run` with --no-surrogates read into the wiring before its seams are built."""
    wiring.surrogates = not args.no_surrogates
    return run(args)


def with_refuter_choice(wiring: CoreWiring, run: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    """`receipts run` with --no-refuter read into the wiring before its seams are built."""
    wiring.refuter = not args.no_refuter
    return run(args)


def with_lens_choice(wiring: CoreWiring, run: Callable[[argparse.Namespace], int], args: argparse.Namespace) -> int:
    """`receipts run` with --lenses read into the wiring before its seams are built."""
    wiring.lenses = tuple(args.lenses)
    return run(args)


def lens_list(text: str) -> tuple[Lens, ...]:
    """`--lenses`: a comma list of lenses, each one an attacker is wired for (RUN_LENSES), each once."""
    try:
        lenses = tuple(Lens(part.strip()) for part in text.split(",") if part.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(str(exc)) from None
    unwired = [lens.value for lens in lenses if lens not in RUN_LENSES]
    if unwired:
        raise argparse.ArgumentTypeError(f"no attacker is wired for {', '.join(unwired)}")
    if not lenses or len(set(lenses)) != len(lenses):
        raise argparse.ArgumentTypeError("name at least one lens, each once")
    return lenses


def register_core(sub: Subparsers) -> None:
    """The core lane's commands, with `run` on a CoreWiring of its own, behind the --jobs cap, and with
    --no-surrogates, --no-refuter and --lenses."""
    wiring = CoreWiring()
    core_commands.register(sub, wiring=core_commands.RunWiring(seams=wiring.seams, ledger=wiring.ledger))
    run = sub.choices["run"]
    run.add_argument(
        "--no-surrogates",
        action="store_true",
        help="build no surrogate fixes (SK3): every fix-validity or behavior catch stays s_sat_inconclusive",
    )
    run.add_argument(
        "--no-refuter",
        action="store_true",
        help="run no refuter (SK2): every blocker stays as SK1 and SK3 left it",
    )
    run.add_argument(
        "--lenses",
        type=lens_list,
        default=TIER_A_LENSES,
        help="the lenses a Tier A target is planned with, as a comma list (default regression,correctness); "
        "differential adds AT4",
    )
    lenses = functools.partial(with_lens_choice, wiring, run.get_default("func"))
    refuter = functools.partial(with_refuter_choice, wiring, lenses)
    handler = functools.partial(with_surrogate_choice, wiring, refuter)
    run.set_defaults(func=functools.partial(at_most_max_jobs, handler))


LANE_COMMANDS: tuple[Callable[[Subparsers], None], ...] = (
    register_core,
    surface_commands.register,
    ticket_command.register,
    check_command.register,
    receipt_command.register,
    functools.partial(
        fix_command.register,
        wiring=fix_command.FixWiring(client=fix_client, run_phase=run_phase, reattack=FixReAttack),
    ),
    functools.partial(
        eval_commands.register, wiring=eval_commands.Wiring(sandbox=run_phase_sandbox, runners=run_phase_runners)
    ),
)


def _version(_: argparse.Namespace) -> int:
    print(__version__)
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="receipts", description="An adversarial verifier for AI-agent pull requests.")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("version", help="print the version").set_defaults(func=_version)
    for register in LANE_COMMANDS:
        register(sub)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler: Callable[[argparse.Namespace], int] = args.func
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
