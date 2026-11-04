"""The pipeline runner: targets through intake, one plan, then every arm's attackers, the skeptic's code gates and that
arm's ledger, over the contracts' seams only. The integrator wires the real sandbox, runners, attackers, clients and
ledgers in receipts.cli; the core lane's tests drive it with fakes.

For one target (run_target):

1. Intake (receipts.core.intake) takes the claims: Tier A's templates, or the agent's own claims by extraction.
2. The planner (receipts.core.planner) makes one plan, shared by every arm, so arms differ only in their attacker model
   (REG-561). A target whose runner kind is not wired gets no plan: each claim is unchecked, no_env.
3. For each arm, in order, for each planned attack: the lens's attackers run in order (Mapping[Lens,
   Sequence[Attacker]]), and a later one runs only if the earlier ones left no candidate that failed its first patch
   run on a counted failure (the skeptic's screen). A repro of several test cases is split into one candidate per
   case first (TargetSkeptic.split). Every candidate then gets the skeptic's verdict (SK1, with SK3's surrogates),
   except that at most TOP_N_PER_CLAIM candidates of one claim that failed their first patch run, counted in order
   across every attack of that claim in the arm, get the full reruns; the later ones are unchecked, budget. A
   candidate that did not fail its first patch run has its status from the screen, with no rerun, and never takes a
   slot. An attacker that raises is recorded as unchecked, runtime, with what its raised
   ModelCallError says it spent, and the run goes on. A divergence the differential attacker (AT4) could not make a
   blocker (AttackOutcome.suspected, decision D3) is recorded as suspected, with its reason in detail and no repro,
   so it is never a blocker and never a catch.
4. SK2 (receipts.core.refuter, Seams.refuter): every blocker the skeptic returns goes to the refuter before it is
   recorded, and is recorded once, in its final state: a blocker still, or suspected when the refuter refutes it or
   no refutation could be read. The refuter's call is outside every arm's cap, like the surrogates: it is one call
   per blocker, and a blocker it cannot check is suspected anyway. Its cost goes on that finding (stage "refuter").
   With no refuter wired, a blocker stays as SK1 and SK3 left it, and the summary says no refuter ran.
5. Every finding goes to the arm's ledger as it is made. Its wall_seconds are the target's shared seconds (intake and
   planning, which every arm needs) plus the seconds since that arm started, so an arm's times never include another
   arm's, and a run with one arm reads from the target's start.

Each arm is held to the per-target cap (spec parameters, R23: $1.00 for Tiers A and B, $3.00 for C and live), counting
the target's shared intake and plan costs plus the arm's own attacks; an attack the cap leaves out is unchecked,
budget. The surrogate fixes (SK3, Seams.surrogates) are outside every arm's cap: a claim's surrogates are built once,
by the skeptic every arm shares, under the target's own surrogate budget (contracts.SURROGATE_TARGET_CAP_MICRO_USD,
kept by the skeptic), so an arm over its cap still gets them and both arms judge a candidate by the same ones. Their
cost goes on the finding whose candidate built them (stage "surrogate"), so a ledger rollup counts it with the rest.

Finding ids are <target>/<claim>/<third>: the third part is the attack id and the item's number for each candidate
and unchecked item an attack produced ('correctness-1.1', 'correctness-1.2'), the attack id alone for an attack that
produced nothing (not_reproduced) or could not run, and the lens or the planned attack's id for what the plan left
out. Every metered call's cost is recorded exactly once: an attack's model costs on the first finding of that attack,
the intake and plan costs on the target's summary file, never on a finding; the skeptic adds only its own seconds.

A target that raises mid-run still gets a summary, marked failed with the error, holding every cost spent so far that
no finding carries: intake and planning, and an attack's costs whose finding was never recorded. A resumed run reruns
a failed target and carries the failed attempt forward: its costs into the new summary's costs, once, and the attempt
itself into earlier_attempts.

Every finding says why it has its status in Finding.detail; Finding.refutation holds only SK2's refutation, by a
different model than the attackers (contract v3). The summary lists each workspace the skeptic prepared, with how its
patch applied (skeptic_workspaces, REG-569), so the receipt can say when a patch applied only with --reject or with
fuzz, and names the refuter model (refuter_model), None when no refuter ran.

Each target runs on a run phase (RunPhase): the sandbox and the runner mapping that its skeptic and every attacker in
every arm use. With Seams.run_phase, a factory, every target gets one of its own, built when the target starts and
released when it is done or has failed. A target claims its run phase before it does anything else, so a target
handed a sandbox, a runner mapping or a runner that another running target holds stops with RunPhaseSharedError
before it spends anything: no two targets ever run on one at once. Seams can instead hold one sandbox and one runner
mapping that every target shares, which runs one target at a time only.

run_targets runs a whole run file, `jobs` targets at a time (more than one needs Seams.run_phase), writing each
target's summary to <summary dir>/<target id>.json once the target is done or has failed. Each arm's ledger takes one
append at a time, so the targets running at once never interleave a line, and on_done is called by one worker at a
time. The model clients are shared by every target: the Meta client's spend meter holds every call's worst case
against the cap, calls in flight included, under a lock (receipts.core.meta_client).
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType
from typing import Final, Literal, Self

from pydantic import Field, ValidationError, model_validator

from receipts import __version__
from receipts.contracts import (
    Arm,
    Attack,
    Attacker,
    AttackOutcome,
    AttackPlan,
    Claim,
    Divergence,
    Finding,
    Frozen,
    Ledger,
    LedgerEntry,
    Lens,
    ModelCallError,
    ModelClient,
    PatchMethod,
    RunnerKind,
    Sandbox,
    Side,
    StageCost,
    Status,
    SuiteRunner,
    SurrogateBuilder,
    Target,
    Tier,
    Unchecked,
    UncheckedReason,
)
from receipts.core.cases import CaseNamer, expected_cases
from receipts.core.intake import DroppedClaim, Extraction, intake
from receipts.core.planner import (
    DEFAULT_LENS,
    PLAN_CAPS,
    TIER_A_LENSES,
    PlanCaps,
    PlanResult,
    Skipped,
    TemperatureMode,
    plan_attacks,
)
from receipts.core.refuter import Refuter, unread
from receipts.core.skeptic import MAX_CASES_PER_REPRO, SKEPTIC_STAGE, RerunPolicy, Screened, TargetSkeptic

log = logging.getLogger(__name__)

# Spec parameters, per-target cap (R23), in micro-dollars.
TARGET_CAP_MICRO_USD: Final[Mapping[Tier, int]] = MappingProxyType(
    {Tier.A: 1_000_000, Tier.B: 1_000_000, Tier.C: 3_000_000, Tier.LIVE: 3_000_000}
)
TOP_N_PER_CLAIM: Final = 5  # the candidates of one claim, in one arm, that get the skeptic's full reruns (SK2)
SUMMARY_VERSION: Final = 1
MAX_ERROR_CHARS: Final = 500
MAX_DETAIL_CHARS: Final = 2_500  # a suspected divergence's detail (AT4)


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class RunPhase:
    """The sandbox and the runner for each RunnerKind that one target runs on: its skeptic's, and every attacker's in
    every arm. `release`, when given, is called once, when the target is done with them (done or failed), so what the
    sandbox still holds goes with the target, not at the end of the run."""

    sandbox: Sandbox
    runners: Mapping[RunnerKind, SuiteRunner]
    release: Callable[[], None] | None = None


class RunPhaseSharedError(RuntimeError):
    """A target was handed a sandbox, a runner mapping or a runner that another running target holds."""


@dataclass(frozen=True, kw_only=True)
class Seams:
    """The real implementations the pipeline runs on, every one behind a contracts interface. `attacker_client` serves
    every arm's attacker model (a receipts.core.routing.ModelRouter, for one). `case_names` names the cases of a repro
    whose Repro.cases is empty, the only repros it is asked about; the default needs no wiring.

    The run phase comes one of two ways. `run_phase` is a factory, called once for each target when it starts, that
    builds a new sandbox and runner mapping on every call; that target runs on what it returns, and no other target
    may run on it at once. Or `sandbox` and `runners`, one of each, which every target shares, so a run on them runs
    one target at a time (check_jobs).

    `surrogates` builds the plausible surrogate fixes a fix-validity or behavior candidate is judged by (SK3), under
    the target's own surrogate budget and outside the arms' caps; with none, such a candidate is s_sat_inconclusive.

    `refuter` is SK2: every blocker the skeptic returns goes to it before it is recorded, outside the arms' caps;
    with none, blockers stay as SK1 and SK3 left them.

    `tier_a_lenses` are the lenses a Tier A target's plan may use (planner.run_lenses), of those an attacker is wired
    for: regression and correctness unless the run adds one (`receipts run --lenses`)."""

    intake_client: ModelClient
    planner_client: ModelClient
    attacker_client: ModelClient
    attackers: Mapping[Lens, Sequence[Attacker]]
    run_phase: Callable[[], RunPhase] | None = None
    sandbox: Sandbox | None = None
    runners: Mapping[RunnerKind, SuiteRunner] | None = None
    policy: RerunPolicy = field(default_factory=RerunPolicy)
    case_names: CaseNamer = expected_cases
    surrogates: SurrogateBuilder | None = None
    refuter: Refuter | None = None
    target_caps: Mapping[Tier, int] = TARGET_CAP_MICRO_USD
    tier_a_lenses: tuple[Lens, ...] = TIER_A_LENSES
    clock: Callable[[], float] = time.monotonic
    now: Callable[[], datetime] = _utcnow

    def __post_init__(self) -> None:
        shared = (self.sandbox is not None, self.runners is not None)
        if self.run_phase is not None and any(shared):
            raise ValueError("the seams take a run_phase factory or one shared sandbox and runner mapping, not both")
        if self.run_phase is None and not all(shared):
            raise ValueError("the seams need a run_phase factory, or one sandbox and one runner mapping")


class _Holders:
    """Which running target holds each sandbox, runner mapping and runner. A target claims its run phase before it
    starts and gives it back once it is done, so no two targets run on one at once, in any run in this process. An
    object is alive while a target holds it, so its id names it alone."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._held: dict[int, str] = {}

    def claim(self, target_id: str, phase: RunPhase) -> tuple[int, ...]:
        parts = {id(phase.sandbox): "sandbox", id(phase.runners): "runner mapping"}
        for kind, runner in phase.runners.items():
            parts.setdefault(id(runner), f"{kind.value} runner")
        with self._lock:
            for key, what in parts.items():
                holder = self._held.get(key)
                if holder is not None:
                    raise RunPhaseSharedError(
                        f"target {target_id} was handed the {what} that target {holder} is running on: each target "
                        "runs on its own, so Seams.run_phase builds a new sandbox and runner mapping on every call"
                    )
            self._held.update(dict.fromkeys(parts, target_id))
        return tuple(parts)

    def give_back(self, keys: Iterable[int]) -> None:
        with self._lock:
            for key in keys:
                self._held.pop(key, None)


_HOLDERS: Final = _Holders()


class RunItem(Frozen):
    """One line of a run file: a target and its attack prompts' temperature mode."""

    target: Target
    temperatures: TemperatureMode


class ArmSummary(Frozen):
    arm: str
    attacker_model: str
    findings: int = Field(ge=0)
    statuses: dict[Status, int]
    blockers: tuple[str, ...] = ()
    wall_seconds: float = Field(ge=0)  # this arm's own seconds


class PreparedWorkspace(Frozen):
    """A workspace the skeptic prepared for the target and how its patch applied (REG-569): git apply, git apply with
    --reject, or GNU patch with fuzz, the SWE-bench harness's order; None for a side with no patch (base)."""

    side: Side
    patch_method: PatchMethod | None = None
    image_digest: str = Field(min_length=1)


class EarlierAttempt(Frozen):
    """A failed run of this target before the one its summary describes. What it spent that no finding carries is in
    the summary's costs, so it is counted once."""

    started_at: str
    error: str = Field(min_length=1)


class TargetSummary(Frozen):
    """What one target's run did beyond its findings, which are in each arm's ledger. Its costs are every cost of the
    target that no finding carries, each once: the shared calls (intake and planning), and, for a failed run or one
    that follows failed ones, the costs those runs spent that no finding records."""

    v: Literal[1] = SUMMARY_VERSION
    receipts_version: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    tier: Tier
    temperatures: TemperatureMode
    status: Literal["done", "failed"] = "done"
    error: str = ""  # why a failed run stopped
    started_at: str
    wall_seconds: float = Field(ge=0)
    claims: tuple[Claim, ...] = ()
    dropped: tuple[DroppedClaim, ...] = ()
    intake_unchecked: tuple[Unchecked, ...] = ()
    plan: AttackPlan
    planner_model: str | None = None
    planner_fallback: bool = False
    planner_fallback_reason: str = ""
    planner_notes: tuple[str, ...] = ()
    costs: tuple[StageCost, ...] = ()
    arms: tuple[ArmSummary, ...] = ()
    earlier_attempts: tuple[EarlierAttempt, ...] = ()
    skeptic_workspaces: tuple[PreparedWorkspace, ...] = ()  # in the order the skeptic prepared them
    refuter_model: str | None = None  # SK2's model when a refuter was wired; None means no refuter ran

    @model_validator(mode="after")
    def _status_has_its_error(self) -> Self:
        if (self.status == "failed") != bool(self.error):
            raise ValueError("a failed run names its error, and only a failed run has one")
        return self


def ledger_entry(finding: Finding, at: datetime) -> LedgerEntry:
    return LedgerEntry(at=_stamp(at), receipts_version=__version__, finding=finding)


def run_target(
    target: Target,
    *,
    temperatures: TemperatureMode,
    arms: Sequence[Arm],
    seams: Seams,
    ledgers: Mapping[str, Ledger],
    previous: TargetSummary | None = None,
    on_failure: Callable[[TargetSummary], None] | None = None,
    plan_caps: PlanCaps = PLAN_CAPS,
) -> TargetSummary:
    """One target through intake, one plan and every arm (module docstring). Each finding is appended to its arm's
    ledger as soon as it is made; the summary is returned for the caller to write. `previous` is this target's failed
    summary from an earlier run, carried forward. When the run raises, its summary marked failed goes to
    `on_failure` before the exception goes on. `plan_caps` are the plan's caps: only `receipts fix`'s fresh attack
    plans with others (none, REG-616); run_targets never passes them, so `receipts run` keeps the default."""
    if not arms:
        raise ValueError("a run needs at least one arm")
    missing = [arm.name for arm in arms if arm.name not in ledgers]
    if missing:
        raise ValueError(f"no ledger for arm(s): {', '.join(missing)}")
    if previous is not None and (previous.target_id != target.id or previous.status != "failed"):
        raise ValueError("only a failed summary of this same target is carried into a new run")
    run = _TargetRun(
        target,
        temperatures=temperatures,
        arms=arms,
        seams=seams,
        ledgers=ledgers,
        previous=previous,
        plan_caps=plan_caps,
    )
    try:
        return run.run()
    except BaseException as exc:
        failed = run.summary(failure=exc)
        log.error("target %s failed mid-run: %s", target.id, failed.error)
        if on_failure is not None:
            try:
                on_failure(failed)
            except Exception as unrecorded:  # the failure stays the run's error; this one is logged with the costs
                log.error(
                    "target %s: the failed summary could not be recorded (%r); its costs: %s",
                    target.id,
                    unrecorded,
                    failed.model_dump_json(include={"costs"}),
                )
        raise


class _TargetRun:
    """One target's run, keeping what it has done so far, so a run that raises can still say what it spent."""

    def __init__(
        self,
        target: Target,
        *,
        temperatures: TemperatureMode,
        arms: Sequence[Arm],
        seams: Seams,
        ledgers: Mapping[str, Ledger],
        previous: TargetSummary | None,
        plan_caps: PlanCaps,
    ) -> None:
        self.target, self.temperatures, self.arms, self.seams, self.ledgers = target, temperatures, arms, seams, ledgers
        self.previous, self.plan_caps = previous, plan_caps
        self.started_at, self.started = seams.now(), seams.clock()
        self.extraction: Extraction | None = None
        self.plan_costs: list[StageCost] = []  # each planning call's cost as it is spent, kept if planning raises
        self.planned: PlanResult | None = None
        self.skeptic: TargetSkeptic | None = None
        self.arm_runs: list[_ArmRun] = []

    def run(self) -> TargetSummary:
        """The target on its own run phase, claimed before anything is spent and released once it is done. A target
        refused for a run phase another target holds never releases it: the other target is running on it."""
        phase = self._phase()
        held = _HOLDERS.claim(self.target.id, phase)
        try:
            try:
                return self._run(phase)
            finally:
                self._release(phase)
        finally:
            _HOLDERS.give_back(held)

    def _phase(self) -> RunPhase:
        seams = self.seams
        if seams.run_phase is None:
            assert seams.sandbox is not None and seams.runners is not None  # Seams refuses anything else
            return RunPhase(sandbox=seams.sandbox, runners=seams.runners)
        phase = seams.run_phase()
        if not isinstance(phase, RunPhase):
            raise TypeError(f"Seams.run_phase returned a {type(phase).__name__}, not a RunPhase")
        return phase

    def _release(self, phase: RunPhase) -> None:
        if phase.release is None:
            return
        try:
            phase.release()
        except Exception as exc:  # the target's findings are on record; a release that fails is logged, never fatal
            log.error("target %s: its run phase could not be released: %r", self.target.id, exc)

    def _run(self, phase: RunPhase) -> TargetSummary:
        target, seams = self.target, self.seams
        extraction = self.extraction = intake(target, seams.intake_client)
        runner = phase.runners.get(target.test_runner)
        if runner is None:
            detail = f"no {target.test_runner.value} runner is wired, so nothing can run"
            skipped = tuple(
                Skipped(claim.id, DEFAULT_LENS[claim.type], UncheckedReason.NO_ENV, detail)
                for claim in extraction.claims
            )
            planned = PlanResult(
                plan=AttackPlan(target_id=target.id, attacks=(), unchecked=tuple(s.unchecked() for s in skipped)),
                skipped=skipped,
            )
        else:
            wired = [lens for lens, chain in seams.attackers.items() if chain]  # the lenses an attacker can take
            if target.tier is not Tier.A:  # AT4 is gated on Tier A (R8): no other tier gets the differential lens
                wired = [lens for lens in wired if lens is not Lens.DIFFERENTIAL]
            planned = plan_attacks(
                target,
                extraction.claims,
                seams.planner_client,
                temperatures=self.temperatures,
                lenses=wired,
                tier_a=seams.tier_a_lenses,
                costs=self.plan_costs,
                caps=self.plan_caps,
            )
        self.planned = planned
        shared_micro_usd = sum(cost.micro_usd for cost in (*extraction.costs, *planned.costs))
        shared_seconds = max(seams.clock() - self.started, 0.0)  # every arm needs these: each arm's times start here
        claims = {claim.id: claim for claim in extraction.claims}
        skeptic = self.skeptic = (
            TargetSkeptic(
                target,
                sandbox=phase.sandbox,
                runner=runner,
                policy=seams.policy,
                case_names=seams.case_names,
                surrogates=seams.surrogates,
            )
            if runner is not None
            else None
        )
        try:
            for arm in self.arms:
                arm_run = _ArmRun(
                    target,
                    arm,
                    seams=seams,
                    sandbox=phase.sandbox,
                    runner=runner,
                    skeptic=skeptic,
                    ledger=self.ledgers[arm.name],
                    claims=claims,
                    offset=shared_seconds,
                    spent=shared_micro_usd,
                )
                self.arm_runs.append(arm_run)
                arm_run.run(planned)
        finally:
            if skeptic is not None:
                skeptic.close()
        return self.summary()

    def summary(self, failure: BaseException | None = None) -> TargetSummary:
        """The target's summary as it stands: done, or failed with `failure` and every cost so far that no finding
        carries."""
        previous, extraction, planned = self.previous, self.extraction, self.planned
        costs: list[StageCost] = list(previous.costs) if previous is not None else []
        if extraction is not None:
            costs += extraction.costs
        costs += planned.costs if planned is not None else self.plan_costs  # a planner that raised spent these
        costs += [cost for arm_run in self.arm_runs for cost in arm_run.pending]  # spent, on no finding
        earlier = () if previous is None else (*previous.earlier_attempts, _attempt_of(previous))
        return TargetSummary(
            receipts_version=__version__,
            target_id=self.target.id,
            tier=self.target.tier,
            temperatures=self.temperatures,
            status="done" if failure is None else "failed",
            error="" if failure is None else _error_text(failure),
            started_at=_stamp(self.started_at),
            wall_seconds=max(self.seams.clock() - self.started, 0.0),
            claims=extraction.claims if extraction is not None else (),
            dropped=extraction.dropped if extraction is not None else (),
            intake_unchecked=extraction.unchecked if extraction is not None else (),
            plan=planned.plan if planned is not None else AttackPlan(target_id=self.target.id, attacks=()),
            planner_model=planned.model if planned is not None else None,
            planner_fallback=planned.fallback if planned is not None else False,
            planner_fallback_reason=planned.fallback_reason if planned is not None else "",
            planner_notes=planned.notes if planned is not None else (),
            costs=tuple(costs),
            arms=tuple(arm_run.summary() for arm_run in self.arm_runs),
            earlier_attempts=earlier,
            skeptic_workspaces=tuple(
                PreparedWorkspace(side=w.side, patch_method=w.patch_method, image_digest=w.image_digest)
                for w in (self.skeptic.prepared if self.skeptic is not None else ())
            ),
            refuter_model=self.seams.refuter.model if self.seams.refuter is not None else None,
        )


def _attempt_of(summary: TargetSummary) -> EarlierAttempt:
    return EarlierAttempt(started_at=summary.started_at, error=summary.error)


def _error_text(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}".encode("utf-8", "replace").decode("utf-8")
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 3] + "..."


class _ArmRun:
    """One arm's pass over a target's plan, on the target's own sandbox and runner."""

    def __init__(
        self,
        target: Target,
        arm: Arm,
        *,
        seams: Seams,
        sandbox: Sandbox,
        runner: SuiteRunner | None,
        skeptic: TargetSkeptic | None,
        ledger: Ledger,
        claims: Mapping[str, Claim],
        offset: float,
        spent: int,
    ) -> None:
        self.target, self.arm, self.seams, self.sandbox = target, arm, seams, sandbox
        self.runner, self.skeptic, self.ledger, self.claims = runner, skeptic, ledger, claims
        self.offset = offset  # the target's shared seconds, before any arm started
        self.arm_started = seams.clock()
        self.spent = spent  # the target's shared costs, then this arm's attacks
        self.cap = seams.target_caps.get(target.tier)
        self.statuses: Counter[Status] = Counter()
        self.blockers: list[str] = []
        self.pending: list[StageCost] = []  # what this arm's attackers spent that no recorded finding carries yet
        self.judged: Counter[str] = Counter()  # claim id: its candidates judged so far in this arm (TOP_N_PER_CLAIM)
        self.ended: float | None = None

    def run(self, planned: PlanResult) -> ArmSummary:
        for item in planned.skipped:
            third = item.attack_id or item.lens.value
            self._unchecked(item.claim_id, item.lens, third, item.reason, item.detail)
        for attack in planned.plan.attacks:
            self._attack(attack)
        self.ended = self.seams.clock()
        return self.summary()

    def summary(self) -> ArmSummary:
        """The arm so far: its own seconds end where it ended, or now for an arm a failure stopped."""
        ended = self.ended if self.ended is not None else self.seams.clock()
        return ArmSummary(
            arm=self.arm.name,
            attacker_model=self.arm.attacker_model,
            findings=sum(self.statuses.values()),
            statuses=dict(self.statuses),
            blockers=tuple(self.blockers),
            wall_seconds=max(ended - self.arm_started, 0.0),
        )

    def _wall(self) -> float:
        return self.offset + max(self.seams.clock() - self.arm_started, 0.0)

    def _record(self, finding: Finding, *, carries_pending: bool = False) -> None:
        """Append the finding to the arm's ledger; the attack costs it carries count as recorded only once it is."""
        self.ledger.append(ledger_entry(finding, self.seams.now()))
        if carries_pending:
            self.pending.clear()
        self.statuses[finding.status] += 1
        if finding.status is Status.BLOCKER:
            self.blockers.append(finding.id)

    def _unchecked(
        self,
        claim_id: str,
        lens: Lens,
        third: str,
        reason: UncheckedReason,
        detail: str,
        *,
        carries_pending: bool = False,
    ) -> None:
        self._record(
            Finding(
                id=f"{self.target.id}/{claim_id}/{third}",
                target_id=self.target.id,
                claim_id=claim_id,
                lens=lens,
                status=Status.UNCHECKED,
                unchecked_reason=reason,
                detail=detail or None,
                costs=tuple(self.pending) if carries_pending else (),
                wall_seconds=self._wall(),
            ),
            carries_pending=carries_pending,
        )

    def _over_cap(self) -> bool:
        return self.cap is not None and self.spent >= self.cap

    def _attack(self, attack: Attack) -> None:
        claim = self.claims[attack.claim_id]
        chain = self.seams.attackers.get(attack.lens, ())
        if self.runner is None or self.skeptic is None:
            self._unchecked(claim.id, attack.lens, attack.id, UncheckedReason.NO_ENV, "no runner is wired")
            return
        if not chain:
            detail = f"no attacker is wired for the {attack.lens.value} lens"
            self._unchecked(claim.id, attack.lens, attack.id, UncheckedReason.RUNTIME, detail)
            return
        if self._over_cap():
            self._unchecked(claim.id, attack.lens, attack.id, UncheckedReason.BUDGET, self._cap_detail())
            return
        unchecked: list[Unchecked] = []
        screened: list[Screened] = []
        suspected: list[Divergence] = []
        for number, attacker in enumerate(chain):
            if number and self._over_cap():
                unchecked.append(Unchecked(what=attack.id, reason=UncheckedReason.BUDGET, detail=self._cap_detail()))
                break
            outcome = self._run_attacker(attacker, attack)
            self.pending.extend(outcome.costs)  # spent: on the attack's first finding, or the summary if none is made
            unchecked.extend(outcome.unchecked)
            suspected.extend(outcome.suspected)
            self.spent += sum(cost.micro_usd for cost in outcome.costs)
            for repro in outcome.repros:
                parts, left = self.skeptic.split(repro)  # one candidate per test case (receipts.core.skeptic)
                screened.extend(self.skeptic.screen(part, attack) for part in parts)
                if left:
                    named = ", ".join(repr(case[:200]) for case in left[:5])
                    detail = f"over the cap of {MAX_CASES_PER_REPRO} test cases judged per repro: {named}"
                    unchecked.append(Unchecked(what=repro.id, reason=UncheckedReason.BUDGET, detail=detail))
            if any(candidate.failed_first_patch_run for candidate in screened):
                break  # a candidate failed its first patch run: no need for the lens's next attacker
        if not screened and not unchecked and not suspected:
            detail = "the attack produced no candidate repro"
            self._record(
                Finding(
                    id=f"{self.target.id}/{claim.id}/{attack.id}",
                    target_id=self.target.id,
                    claim_id=claim.id,
                    lens=attack.lens,
                    status=Status.NOT_REPRODUCED,
                    detail=detail,
                    costs=tuple(self.pending),
                    wall_seconds=self._wall(),
                ),
                carries_pending=True,
            )
            return
        chosen = self._top_n(claim.id, screened)
        first = True  # the attack's costs go on its first finding only
        for number, candidate in enumerate(screened, start=1):
            third = f"{attack.id}.{number}"
            if number - 1 not in chosen:
                self._record(self._over_top_n(candidate, claim, third, carries_pending=first), carries_pending=first)
                first = False
                continue
            # Surrogates the judgment builds, and SK2's refutation, are outside the arm's cap (module docstring):
            # never added to self.spent.
            finding = self.skeptic.judge(
                candidate,
                claim=claim,
                finding_id=f"{self.target.id}/{claim.id}/{third}",
                costs=tuple(self.pending) if first else (),
                wall=self._wall,
            )
            self._record(self._refute(finding, claim), carries_pending=first)
            first = False
        for number, divergence in enumerate(suspected, start=len(screened) + 1):
            # D3: a divergence AT4's intent gate did not make a blocker. It has no repro and is never a catch.
            self._record(
                Finding(
                    id=f"{self.target.id}/{claim.id}/{attack.id}.{number}",
                    target_id=self.target.id,
                    claim_id=claim.id,
                    lens=attack.lens,
                    status=Status.SUSPECTED,
                    detail=suspected_detail(divergence),
                    costs=tuple(self.pending) if first else (),
                    wall_seconds=self._wall(),
                ),
                carries_pending=first,
            )
            first = False
        for number, item in enumerate(unchecked, start=len(screened) + len(suspected) + 1):
            detail = item.detail or f"the attacker could not check {item.what}"
            third = f"{attack.id}.{number}"
            self._unchecked(claim.id, attack.lens, third, item.reason, detail, carries_pending=first)
            first = False

    def _top_n(self, claim_id: str, screened: Sequence[Screened]) -> frozenset[int]:
        """The places, in `screened`, of the candidates that are judged. Top-n bounds the reruns, so only a candidate
        that failed its first patch run (one judge reruns) takes a slot of the claim's TOP_N_PER_CLAIM in this arm,
        across its attacks, in their order; one that did not has its status from the screen and is always judged."""
        failed = [n for n, candidate in enumerate(screened) if candidate.failed_first_patch_run]
        room = max(TOP_N_PER_CLAIM - self.judged[claim_id], 0)
        taken = failed[:room]
        self.judged[claim_id] += len(taken)
        screened_only = (n for n, candidate in enumerate(screened) if not candidate.failed_first_patch_run)
        return frozenset((*taken, *screened_only))

    def _over_top_n(self, candidate: Screened, claim: Claim, third: str, *, carries_pending: bool) -> Finding:
        """A first-run failure past the claim's TOP_N_PER_CLAIM: unchecked, budget, with what the screen found kept on
        it: its repro, its first patch run, and the screen's seconds as the skeptic's cost."""
        screen = StageCost(stage=SKEPTIC_STAGE, seconds=max(candidate.seconds, 0.0))
        return Finding(
            id=f"{self.target.id}/{claim.id}/{third}",
            target_id=self.target.id,
            claim_id=claim.id,
            lens=candidate.attack.lens,
            status=Status.UNCHECKED,
            kind=candidate.repro.kind,
            unchecked_reason=UncheckedReason.BUDGET,
            repro=candidate.repro,
            runs=candidate.runs,
            quote=claim.span.text,
            detail=f"over the top-n of {TOP_N_PER_CLAIM} candidates judged per claim",
            costs=(*self.pending, screen) if carries_pending else (screen,),
            wall_seconds=self._wall(),
        )

    def _refute(self, finding: Finding, claim: Claim) -> Finding:
        """A blocker after SK2 (Seams.refuter), outside the arm's cap; any other finding, or any finding with no
        refuter wired, as it is. A refuter never raises (receipts.core.refuter); one that does leaves the blocker
        suspected, never a crash of the run."""
        refuter = self.seams.refuter
        if refuter is None or finding.status is not Status.BLOCKER:
            return finding
        try:
            return refuter.refute(self.target, claim, finding, wall=self._wall)
        except Exception as exc:
            log.exception("refuter raised on %s", finding.id)
            return unread(finding, f"the refuter raised {type(exc).__name__}", wall=self._wall)

    def _run_attacker(self, attacker: Attacker, attack: Attack) -> AttackOutcome:
        assert self.runner is not None
        try:
            outcome = attacker.attack(
                self.target,
                attack,
                arm=self.arm,
                client=self.seams.attacker_client,
                sandbox=self.sandbox,
                runner=self.runner,
            )
        except Exception as exc:  # an attacker never raises (contracts.Attacker); one that does is unchecked, not fatal
            log.exception("attacker for %s raised on %s/%s", attack.lens.value, self.target.id, attack.id)
            detail = f"the {attack.lens.value} attacker raised {type(exc).__name__}"
            # A ModelCallError says what its call spent: that still counts, once.
            spent = (exc.cost,) if isinstance(exc, ModelCallError) and exc.cost is not None else ()
            return AttackOutcome(
                attack_id=attack.id,
                costs=spent,
                unchecked=(Unchecked(what=attack.id, reason=UncheckedReason.RUNTIME, detail=detail),),
            )
        if outcome.attack_id != attack.id:
            detail = f"the attacker answered for attack {outcome.attack_id[:64]!r}, not {attack.id}"
            return AttackOutcome(
                attack_id=attack.id,
                costs=outcome.costs,  # what it spent still counts
                unchecked=(Unchecked(what=attack.id, reason=UncheckedReason.RUNTIME, detail=detail),),
            )
        return outcome

    def _cap_detail(self) -> str:
        assert self.cap is not None
        return (
            f"arm {self.arm.name} reached the per-target cap of {self.cap:,} micro-dollars "
            f"({self.spent:,} spent on this target)"
        )


def suspected_detail(divergence: Divergence) -> str:
    """A suspected divergence's finding detail: why it is no blocker, the example, and what each side printed."""
    text = (
        f"suspected divergence (AT4), not a blocker: {divergence.reason[:600]}; "
        f"example: {divergence.example[:800]!r}; base printed {divergence.base_output[:300]!r}; "
        f"patch printed {divergence.patch_output[:300]!r}"
    )
    return text if len(text) <= MAX_DETAIL_CHARS else text[: MAX_DETAIL_CHARS - 3] + "..."


# ---------------------------------------------------------------- a whole run


class _SerialLedger:
    """A ledger appended to by one thread at a time, for targets running in parallel."""

    def __init__(self, ledger: Ledger) -> None:
        self._ledger = ledger
        self._lock = threading.Lock()

    def append(self, entry: LedgerEntry) -> None:
        with self._lock:
            self._ledger.append(entry)

    def entries(self, target_id: str | None = None) -> Sequence[LedgerEntry]:
        with self._lock:
            return self._ledger.entries(target_id)


@dataclass(frozen=True)
class RunReport:
    done: tuple[TargetSummary, ...]
    skipped: tuple[str, ...]  # targets a resumed run found done already
    failed: tuple[tuple[str, str], ...]  # (target id, what went wrong)
    rerun: tuple[str, ...] = ()  # targets a resumed run ran again after a failed run


def summary_path(summary_dir: Path, target_id: str) -> Path:
    return summary_dir / f"{target_id}.json"


def read_summary(path: Path) -> TargetSummary:
    """A target's summary file; one that cannot be read or validated is refused by its path."""
    try:
        return TargetSummary.model_validate_json(path.read_bytes())
    except (OSError, ValidationError) as exc:
        raise ValueError(f"the summary {path} cannot be read: {type(exc).__name__}") from None


def check_run(items: Sequence[RunItem], arms: Sequence[Arm]) -> None:
    """Refuse a run file or arms list the run cannot use: empty, or with a target id or an arm name twice."""
    if not items:
        raise ValueError("the run file has no targets")
    if not arms:
        raise ValueError("the arms file has no arms")
    for what, names in (("target id", [i.target.id for i in items]), ("arm name", [a.name for a in arms])):
        twice = sorted(name for name, count in Counter(names).items() if count > 1)
        if twice:
            raise ValueError(f"a {what} appears twice: {', '.join(twice)}")


def check_jobs(jobs: int, seams: Seams) -> None:
    """Refuse a jobs count the seams cannot run: below 1, or above 1 on one sandbox and runner mapping that every
    target shares. Targets run at once only on run phases of their own (Seams.run_phase)."""
    if jobs < 1:
        raise ValueError("jobs is at least 1")
    if jobs > 1 and seams.run_phase is None:
        raise ValueError(
            f"{jobs} targets at once need a run phase of each target's own (Seams.run_phase), and these seams give "
            "every target one shared sandbox and runner mapping, so they run one target at a time"
        )


def run_targets(
    items: Sequence[RunItem],
    *,
    arms: Sequence[Arm],
    seams: Seams,
    ledgers: Mapping[str, Ledger],
    summary_dir: Path,
    jobs: int = 1,
    resume: bool = False,
    on_done: Callable[[TargetSummary], None] | None = None,
) -> RunReport:
    """Every target of a run file through run_target, `jobs` at a time, each summary written when its target is done
    or has failed. With `resume`, a target whose summary says done is skipped and one whose summary says failed runs
    again, carrying the failed run forward; without it, an existing summary is refused before anything runs, so a run
    never doubles a ledger by accident. A target that raises is reported and the others go on. More than one job needs
    Seams.run_phase (check_jobs); each arm's ledger takes one append at a time, and on_done is called by one worker at
    a time."""
    check_run(items, arms)
    check_jobs(jobs, seams)
    existing = [i.target.id for i in items if summary_path(summary_dir, i.target.id).exists()]
    if existing and not resume:
        raise FileExistsError(
            f"{len(existing)} target(s) already have a summary in {summary_dir} (first: {existing[0]}); "
            "pass resume to skip the done ones and run the failed ones again"
        )
    found = {target_id: read_summary(summary_path(summary_dir, target_id)) for target_id in existing}
    done_already = [target_id for target_id, summary in found.items() if summary.status == "done"]
    previous = {target_id: summary for target_id, summary in found.items() if summary.status == "failed"}
    summary_dir.mkdir(parents=True, exist_ok=True)
    serial: dict[str, Ledger] = {name: _SerialLedger(ledger) for name, ledger in ledgers.items()}
    skip = set(done_already)
    todo = [i for i in items if i.target.id not in skip]
    reporting = threading.Lock()  # on_done is called by one worker at a time, so the lines it prints never interleave

    def write(summary: TargetSummary) -> None:
        _write_atomically(summary_path(summary_dir, summary.target_id), summary.model_dump_json(indent=2) + "\n")

    def one(item: RunItem) -> TargetSummary:
        summary = run_target(
            item.target,
            temperatures=item.temperatures,
            arms=arms,
            seams=seams,
            ledgers=serial,
            previous=previous.get(item.target.id),
            on_failure=write,
        )
        try:
            write(summary)
        except Exception:  # the target ran, but its record did not land: say what it spent, then report it failed
            log.error("target %s: its summary could not be written; its costs: %s", summary.target_id, summary.costs)
            raise
        if on_done is not None:
            with reporting:
                on_done(summary)
        return summary

    done: list[TargetSummary] = []
    failed: list[tuple[str, str]] = []
    with ThreadPoolExecutor(max_workers=jobs) as pool:
        futures = [(item.target.id, pool.submit(one, item)) for item in todo]
        for target_id, future in futures:
            try:
                done.append(future.result())
            except Exception as exc:  # one target's failure is reported; the run's other targets go on
                log.error("target %s failed: %r", target_id, exc, exc_info=exc)
                failed.append((target_id, f"{type(exc).__name__}: {exc}"))
    return RunReport(
        done=tuple(done),
        skipped=tuple(done_already),
        failed=tuple(failed),
        rerun=tuple(previous),
    )


def _write_atomically(path: Path, text: str) -> None:
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f".{path.name}.", suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as out:
            out.write(text)
            out.flush()
            os.fsync(out.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
