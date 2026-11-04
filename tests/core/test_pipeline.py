"""The pipeline runner and 'receipts run', over fakes for every seam: clients, sandbox, runner, attackers and ledgers.
Nothing here calls a model, a container or the network."""

from __future__ import annotations

import contextlib
import itertools
import json
import re
import secrets
import threading
import time
from collections.abc import Iterable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, ClassVar, NoReturn

import httpx
import pytest

from receipts import cli
from receipts.contracts import (
    Arm,
    Attack,
    AttackOutcome,
    CaseResult,
    CatchKind,
    ClaimType,
    Divergence,
    FailureType,
    Finding,
    LedgerEntry,
    Lens,
    Message,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    Outcome,
    PatchMethod,
    Repro,
    RunnerKind,
    RunResult,
    Side,
    StageCost,
    Status,
    Surrogate,
    Target,
    Tier,
    UncheckedReason,
    Usage,
    Workspace,
)
from receipts.core.cli_commands import RunWiring, register
from receipts.core.meta_client import MetaClient, SpendMeter, TokenGovernor
from receipts.core.pipeline import (
    TARGET_CAP_MICRO_USD,
    TOP_N_PER_CLAIM,
    RunItem,
    RunPhase,
    RunPhaseSharedError,
    Seams,
    TargetSummary,
    check_jobs,
    read_summary,
    run_target,
    run_targets,
    summary_path,
)
from receipts.core.planner import FALLBACK_MODEL, PLAN_STAGE, PLANNER_MODEL, PlanCaps, TemperatureMode
from receipts.core.refuter import NO_REFUTATION, REFUTER_MODEL, REFUTER_STAGE, Refuter
from receipts.core.routing import ModelRouter
from receipts.core.skeptic import SKEPTIC_STAGE
from tests.core.test_intake import SECURITY_PATH

SPARK, GLIMMER = "muse-spark-1.3", "muse-glimmer-30b"
ARMS = (Arm(name="spark", attacker_model=SPARK), Arm(name="glimmer", attacker_model=GLIMMER))
DIGEST = "sha256:" + "e" * 64
PATCH = """\
diff --git a/src/widgets/render.py b/src/widgets/render.py
--- a/src/widgets/render.py
+++ b/src/widgets/render.py
@@ -1,2 +1,2 @@ def render(items):
-    return "".join(items)
+    return ",".join(items)
"""


def tier_a(name: str, **changes: object) -> Target:
    fields: dict[str, object] = {
        "id": f"verified~octo__widgets-{name}",
        "tier": Tier.A,
        "repo": "octo/widgets",
        "base_sha": "a" * 40,
        "patch": PATCH,
        "issue": "render crashes on an empty list",
    }
    fields.update(changes)
    return Target.model_validate(fields)


class Costs:
    """Every cost a fake spends, each with its own micro-dollar amount, so a test can find where each one landed."""

    def __init__(self) -> None:
        self.issued: list[StageCost] = []
        self._amount = itertools.count(101)
        self._lock = threading.Lock()

    def spend(self, stage: str, model: str | None) -> StageCost:
        with self._lock:
            cost = StageCost(stage=stage, model=model, micro_usd=next(self._amount), seconds=0.25)
            self.issued.append(cost)
        return cost


PLAN_LENS = {"fixed_x": "correctness", "nothing_else_changed": "regression", "tests_pass": "regression"}


def plan_text(prompt: str) -> str:
    """The plan a planner fake answers with: every claim the prompt names gets the attacks it asks for, on its type's
    lens. `prompt` is the plan request's user message (the claims and the diff)."""
    claims = re.findall(r'<claim id="([^"]+)" type="([^"]+)" attacks="(\d)">', prompt)
    return json.dumps(
        {
            "claims": [
                {"claim_id": cid, "attacks": [{"lens": PLAN_LENS[ctype], "hypothesis": "h"}] * int(k)}
                for cid, ctype, k in claims
            ]
        }
    )


class Models:
    """The intake and planner clients: Spark extraction for a target with agent text, and a Wonder plan that gives
    each claim three attacks on its type's lens."""

    def __init__(self, costs: Costs, *, planner_fails: bool = False, wonder_invalid: bool = False) -> None:
        self.costs = costs
        self.planner_fails = planner_fails
        self.wonder_invalid = wonder_invalid  # Wonder answers, but with a plan that fails validation
        self.requests: list[ModelRequest] = []
        self._lock = threading.Lock()

    def complete(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self.requests.append(request)
        if request.stage == "intake":
            text = json.dumps(
                {"claims": [{"type": "tests_pass", "source": "pr_body", "quote": "All tests pass.", "text": "t"}]}
            )
        else:
            if self.planner_fails:
                raise ModelCallError(
                    "down", UncheckedReason.RUNTIME, cost=self.costs.spend(request.stage, request.model)
                )
            text = plan_text(request.messages[1].content)  # the claims and the diff, whatever a retry appends
            if self.wonder_invalid and request.model == PLANNER_MODEL:
                text = '{"claims": "not a list"}'
        return ModelResponse(
            model=request.model,
            text=text,
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=10),
            cost=self.costs.spend(request.stage, request.model),
        )


class AttackerModels:
    """The router's two servers, which the fake attackers never call: arms differ only in the model they name."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("the fake attackers make no model call")


Behavior = str  # 'blocker' fails on the patch and passes on base; 'passes' passes; 'import' fails to import


def repro(attack: Attack, behavior: Behavior, number: int) -> Repro:
    kind = CatchKind.FIX_VALIDITY if attack.lens is Lens.CORRECTNESS else CatchKind.REGRESSION
    path = f"tests/test_{behavior}_{attack.id.replace('-', '_')}_{number}.py"
    return Repro(
        id=f"{attack.id}-r{number}",
        attack_id=attack.id,
        claim_id=attack.claim_id,
        lens=attack.lens,
        kind=kind,
        files={path: f"# {behavior}\ndef test_case():\n    assert True\n"},
        command=("python", "-m", "pytest", path),
        quote="render crashes on an empty list" if kind is CatchKind.FIX_VALIDITY else "",  # the issue, verbatim
    )


@dataclass
class FakeAttacker:
    """Answers each attack id with the scripted candidates (or raises), spending one cost per call."""

    lens: Lens
    name: str
    costs: Costs
    script: Mapping[str, Sequence[Behavior] | BaseException]
    calls: list[tuple[str, str, str, float, object]] = field(default_factory=list)
    got: list[tuple[str, object, object]] = field(default_factory=list)  # (target id, sandbox, runner) per call

    def attack(
        self, target: Target, attack: Attack, *, arm: Arm, client: Any, sandbox: Any, runner: Any
    ) -> AttackOutcome:
        self.calls.append((target.id, arm.name, attack.id, attack.temperature, client))
        self.got.append((target.id, sandbox, runner))
        step = self.script.get(attack.id, ())
        if isinstance(step, BaseException):
            raise step
        cost = self.costs.spend(f"attack:{self.name}", arm.attacker_model)
        return AttackOutcome(
            attack_id=attack.id,
            repros=tuple(repro(attack, behavior, n) for n, behavior in enumerate(step, start=1)),
            costs=(cost,),
        )


class Runner:
    kind = RunnerKind.PYTEST

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["python", "-m", "pytest", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return ()


class FakeSandbox:
    """Runs a repro by the behavior written in its test file."""

    def __init__(self) -> None:
        self.runs: list[tuple[str, Side, int]] = []
        self.prepared: list[tuple[str, Side]] = []
        self.closed = 0
        self._lock = threading.Lock()

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        with self._lock:
            self.prepared.append((target.id, side))
        return Workspace(id=f"{target.id}-{side.value}", side=side, root="/testbed", image_digest=DIGEST)

    def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
        raise AssertionError("not used")

    def run_test(self, workspace: Workspace, tests: Sequence[str], **options: Any) -> RunResult:
        files: Mapping[str, str] = options["files"]
        seed: int = options["seed"]
        assert options["fresh"] is True
        behavior = next(iter(files.values())).split("\n", 1)[0].removeprefix("# ")
        with self._lock:
            self.runs.append((behavior, workspace.side, seed))
        common: dict[str, Any] = {"side": workspace.side, "seed": seed, "fresh_container": True, "duration_s": 1.0}
        if behavior == "blocker" and workspace.side is Side.PATCH:
            return RunResult(outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION, **common)
        if behavior == "import":
            return RunResult(outcome=Outcome.FAIL, failure_type=FailureType.IMPORT, **common)
        return RunResult(outcome=Outcome.PASS, **common)

    def close(self, workspace: Workspace) -> None:
        with self._lock:
            self.closed += 1


class MemoryLedger:
    def __init__(self) -> None:
        self.items: list[LedgerEntry] = []

    def append(self, entry: LedgerEntry) -> None:
        self.items.append(entry)

    def entries(self, target_id: str | None = None) -> Sequence[LedgerEntry]:
        return [e for e in self.items if target_id is None or e.finding.target_id == target_id]


@dataclass
class World:
    costs: Costs
    models: Models
    sandbox: FakeSandbox
    regression: FakeAttacker
    stage1: FakeAttacker
    stage2: FakeAttacker
    router: ModelRouter
    seams: Seams
    ledgers: dict[str, MemoryLedger]


Script = dict[str, Sequence[Behavior] | BaseException]
REGRESSION_SCRIPT: Script = {
    "regression-1": ["blocker"],
    "regression-2": ["passes"],
    "regression-3": [],
}
STAGE1_SCRIPT: Script = {
    "correctness-1": ["passes"],  # fails nothing on its first patch run, so stage 2 runs
    "correctness-2": ["blocker"],  # fails its first patch run, so stage 2 never runs
    "correctness-3": RuntimeError("stage 1 broke"),
}
STAGE2_SCRIPT: Script = {"correctness-1": ["blocker"], "correctness-3": ["import"]}


def world(**seam_changes: Any) -> World:
    costs = Costs()
    models = Models(
        costs,
        planner_fails=seam_changes.pop("planner_fails", False),
        wonder_invalid=seam_changes.pop("wonder_invalid", False),
    )
    sandbox = FakeSandbox()
    regression = FakeAttacker(Lens.REGRESSION, "at3", costs, seam_changes.pop("regression_script", REGRESSION_SCRIPT))
    stage1 = FakeAttacker(Lens.CORRECTNESS, "at2-stage1", costs, seam_changes.pop("stage1_script", STAGE1_SCRIPT))
    stage2 = FakeAttacker(Lens.CORRECTNESS, "at2-stage2", costs, seam_changes.pop("stage2_script", STAGE2_SCRIPT))
    router = ModelRouter({SPARK: AttackerModels(), GLIMMER: AttackerModels()})
    ticks = itertools.count()
    options: dict[str, Any] = {
        "intake_client": models,
        "planner_client": models,
        "attacker_client": router,
        "sandbox": sandbox,
        "runners": {RunnerKind.PYTEST: Runner()},
        "attackers": {Lens.REGRESSION: [regression], Lens.CORRECTNESS: [stage1, stage2]},
        "clock": lambda: float(next(ticks)),
    }
    if "run_phase" in seam_changes:  # each target's own sandbox and runner, from the factory, and no shared one
        del options["sandbox"], options["runners"]
    options.update(seam_changes)
    ledgers = {arm.name: MemoryLedger() for arm in ARMS}
    return World(costs, models, sandbox, regression, stage1, stage2, router, Seams(**options), ledgers)


EXPECTED = {
    "nothing_else_changed/regression-1.1": Status.BLOCKER,
    "nothing_else_changed/regression-2.1": Status.NOT_REPRODUCED,
    "nothing_else_changed/regression-3": Status.NOT_REPRODUCED,  # the attack produced nothing
    "fixed_x/correctness-1.1": Status.NOT_REPRODUCED,
    "fixed_x/correctness-1.2": Status.S_SAT_INCONCLUSIVE,  # stage 2's candidate: no surrogate fix yet
    "fixed_x/correctness-2.1": Status.S_SAT_INCONCLUSIVE,
    "fixed_x/correctness-3.1": Status.REFUTED,  # stage 2's import failure
    "fixed_x/correctness-3.2": Status.UNCHECKED,  # stage 1 raised
}


def findings_of(ledger: MemoryLedger, target: Target) -> dict[str, Status]:
    prefix = target.id + "/"
    return {e.finding.id.removeprefix(prefix): e.finding.status for e in ledger.entries(target.id)}


def assert_every_cost_once(costs: Costs, ledgers: Iterable[MemoryLedger], summaries: Iterable[TargetSummary]) -> None:
    """Every cost the fakes issued appears exactly once across the run's records: the ledgers' findings (the
    skeptic's own seconds aside) and the targets' summaries. Each issued cost has its own micro-dollar amount."""
    recorded = [
        c.micro_usd for ledger in ledgers for e in ledger.items for c in e.finding.costs if c.stage != SKEPTIC_STAGE
    ]
    recorded += [c.micro_usd for summary in summaries for c in summary.costs]
    assert sorted(recorded) == sorted(c.micro_usd for c in costs.issued)


def test_a_two_arm_run_on_two_targets_shares_one_plan_and_keeps_every_cost_once() -> None:
    w = world()
    targets = [tier_a("1"), tier_a("2")]
    summaries = [
        run_target(t, temperatures=mode, arms=ARMS, seams=w.seams, ledgers=w.ledgers)
        for t, mode in zip(targets, (TemperatureMode.SPEC, TemperatureMode.META_DEFAULT), strict=True)
    ]

    for arm in ARMS:
        for t in targets:
            assert findings_of(w.ledgers[arm.name], t) == EXPECTED, (arm.name, t.id)
    # One plan per target, made once and shared by both arms; intake made no call for Tier A.
    assert [r.stage for r in w.models.requests] == ["plan", "plan"]
    assert {r.model for r in w.models.requests} == {PLANNER_MODEL}
    for summary in summaries:
        assert len(summary.plan.attacks) == 6
        assert [a.name for a in ARMS] == [s.arm for s in summary.arms]
    # Each arm's attackers got that arm and the one router; the plan's temperatures came from the run file.
    calls = w.regression.calls + w.stage1.calls + w.stage2.calls
    assert {c[1] for c in calls} == {"spark", "glimmer"}
    assert all(c[4] is w.router for c in calls)
    assert {c[3] for c in calls if c[0] == targets[1].id} == {1.0}
    # Stage 2 ran only where stage 1 left no candidate that failed its first patch run.
    assert sorted({c[2] for c in w.stage2.calls}) == ["correctness-1", "correctness-3"]
    assert len(w.stage2.calls) == 2 * 2 * 2

    # Every finding has its wall time, and every metered call's cost is recorded exactly once.
    recorded: list[StageCost] = []
    for ledger in w.ledgers.values():
        for entry in ledger.items:
            assert entry.finding.wall_seconds is not None
            recorded += [c for c in entry.finding.costs if c.stage != SKEPTIC_STAGE]
    for summary in summaries:
        recorded += summary.costs  # the plan's cost, on the target's summary and never on a finding
    assert sorted(c.micro_usd for c in recorded) == sorted(c.micro_usd for c in w.costs.issued)
    assert all(
        c.model is None and c.micro_usd == 0
        for e in w.ledgers["spark"].items
        for c in e.finding.costs
        if c.stage == SKEPTIC_STAGE
    )

    blocker = next(e.finding for e in w.ledgers["glimmer"].items if e.finding.status is Status.BLOCKER)
    assert blocker.replay is not None and blocker.kind is CatchKind.REGRESSION
    assert summaries[0].arms[0].blockers == (f"{targets[0].id}/nothing_else_changed/regression-1.1",)
    assert w.sandbox.closed == 2 * 2  # each target's patch and base workspaces, closed once


def test_targets_run_in_parallel_write_their_summaries(tmp_path: Path) -> None:
    w = world(run_phase=Phases())  # more than one job: each target on its own run phase
    items = [RunItem(target=tier_a(str(n)), temperatures=TemperatureMode.SPEC) for n in range(4)]
    report = run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path / "targets", jobs=3)
    assert (len(report.done), report.failed, report.skipped) == (4, (), ())
    for item in items:
        summary = TargetSummary.model_validate_json(summary_path(tmp_path / "targets", item.target.id).read_text())
        assert summary.target_id == item.target.id and len(summary.arms) == 2
        assert findings_of(w.ledgers["spark"], item.target) == EXPECTED

    with pytest.raises(FileExistsError, match="already have a summary"):
        run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path / "targets")
    again = run_targets(
        items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path / "targets", resume=True
    )
    assert (again.done, len(again.skipped)) == ((), 4)


def test_one_target_failing_leaves_the_others_running(tmp_path: Path) -> None:
    w = world()

    class Fragile(MemoryLedger):
        def append(self, entry: LedgerEntry) -> None:
            if entry.finding.target_id.endswith("-bad"):
                raise OSError("disk full")
            super().append(entry)

    ledgers = {"spark": Fragile(), "glimmer": Fragile()}
    items = [RunItem(target=tier_a(n), temperatures=TemperatureMode.SPEC) for n in ("ok", "bad")]
    report = run_targets(items, arms=ARMS, seams=w.seams, ledgers=ledgers, summary_dir=tmp_path)
    assert [s.target_id for s in report.done] == [items[0].target.id]
    assert report.failed == ((items[1].target.id, "OSError: disk full"),)
    # The failed target has its summary, marked failed, with every cost no finding records: its plan, and the
    # first attack's costs, whose finding never reached the ledger (round 3, call d).
    failed = read_summary(summary_path(tmp_path, items[1].target.id))
    assert (failed.status, failed.error) == ("failed", "OSError: disk full")
    assert [c.stage for c in failed.costs] == [PLAN_STAGE, "attack:at2-stage1", "attack:at2-stage2"]
    assert_every_cost_once(w.costs, ledgers.values(), [report.done[0], failed])


def test_the_per_target_cap_leaves_later_attacks_unchecked() -> None:
    # The plan costs 101 and each attacker call about as much: a cap of 400 lets the arm's first few attacks run.
    w = world(target_caps={Tier.A: 400})
    run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    statuses = [e.finding for e in w.ledgers["spark"].items]
    budget = [f for f in statuses if f.unchecked_reason is UncheckedReason.BUDGET]
    assert budget and all("per-target cap of 400" in (f.detail or "") for f in budget)
    assert TARGET_CAP_MICRO_USD[Tier.A] == 1_000_000 and TARGET_CAP_MICRO_USD[Tier.LIVE] == 3_000_000


def test_a_target_whose_runner_is_not_wired_makes_no_plan_and_is_unchecked() -> None:
    w = world()
    django = tier_a("1", test_runner=RunnerKind.DJANGO)
    summary = run_target(django, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers)
    assert w.models.requests == []  # nothing could run, so nothing was spent planning
    assert summary.plan.attacks == ()
    findings = [e.finding for e in w.ledgers["spark"].items]
    assert {(f.claim_id, f.unchecked_reason) for f in findings} == {
        ("fixed_x", UncheckedReason.NO_ENV),
        ("nothing_else_changed", UncheckedReason.NO_ENV),
    }


def test_a_planner_failure_is_unchecked_in_every_arm_and_its_cost_is_on_the_summary() -> None:
    w = world(planner_fails=True)
    summary = run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers)
    assert [c.micro_usd for c in summary.costs] == [c.micro_usd for c in w.costs.issued]  # Wonder's and Spark's
    for arm in ARMS:
        ids = {e.finding.id.split("/", 1)[1]: e.finding.unchecked_reason for e in w.ledgers[arm.name].items}
        assert ids == {
            "fixed_x/correctness": UncheckedReason.RUNTIME,
            "nothing_else_changed/regression": UncheckedReason.RUNTIME,
        }
        assert all(e.finding.costs == () for e in w.ledgers[arm.name].items)


def test_a_target_with_no_agent_text_has_no_claims_and_says_why() -> None:
    w = world()
    live = Target(id="pr-9", tier=Tier.LIVE, repo="octo/widgets", base_sha="b" * 40, patch=PATCH)
    summary = run_target(live, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers)
    assert summary.claims == () and w.models.requests == []
    (unchecked,) = summary.intake_unchecked
    assert unchecked.reason is UncheckedReason.NO_INPUT
    assert all(ledger.items == [] for ledger in w.ledgers.values())


def test_a_live_target_is_planned_from_its_extracted_claims() -> None:
    w = world()
    live = Target(
        id="pr-9", tier=Tier.LIVE, repo="octo/widgets", base_sha="b" * 40, patch=PATCH, body="All tests pass."
    )
    summary = run_target(live, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    assert [r.stage for r in w.models.requests] == ["intake", "plan"]
    assert [c.type for c in summary.claims] == [ClaimType.TESTS_PASS]
    assert {a.claim_id for a in summary.plan.attacks} == {summary.claims[0].id}
    assert len(summary.costs) == 2  # intake and plan, here only
    on_findings = [c for e in w.ledgers["spark"].items for c in e.finding.costs if c.stage != SKEPTIC_STAGE]
    assert sorted(c.micro_usd for c in (*summary.costs, *on_findings)) == sorted(c.micro_usd for c in w.costs.issued)
    assert not {c.stage for c in on_findings} & {"intake", "plan"}


def test_an_attacker_that_answers_for_another_attack_is_unchecked_and_its_cost_kept() -> None:
    w = world()

    class Confused(FakeAttacker):
        def attack(self, target: Target, attack: Attack, **options: Any) -> AttackOutcome:
            return AttackOutcome(attack_id="someone-else", costs=(self.costs.spend("attack:confused", SPARK),))

    confused = Confused(Lens.REGRESSION, "confused", w.costs, {})
    seams = Seams(**{**w.seams.__dict__, "attackers": {Lens.REGRESSION: [confused]}})
    run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=seams, ledgers=w.ledgers)
    findings = [e.finding for e in w.ledgers["spark"].items]
    # Only regression is wired, so all six attacks are regression ones and the per-lens cap leaves three unchecked.
    answered = [f for f in findings if "someone-else" in (f.detail or "")]
    assert len(answered) == 3 and {f.unchecked_reason for f in answered} == {UncheckedReason.RUNTIME}
    assert sum(f.unchecked_reason is UncheckedReason.BUDGET for f in findings) == 3
    # What the confused attacker spent still counts: every attack cost (all but the plan's) is on a finding.
    assert sum(c.micro_usd for f in findings for c in f.costs) == sum(c.micro_usd for c in w.costs.issued[1:])


# REG-616: a risky patch gives both Tier A claims five masks, 5 correctness and 5 regression attacks. Under the default
# caps the four past 3 per lens are unchecked (budget); only `receipts fix`'s fresh attack plans with none.
OVER_THE_CAPS = [
    "fixed_x/correctness-4",
    "fixed_x/correctness-5",
    "nothing_else_changed/regression-4",
    "nothing_else_changed/regression-5",
]


def over_the_caps(target: Target, findings: Iterable[Finding]) -> list[str]:
    return sorted(
        {f.id.removeprefix(f"{target.id}/") for f in findings if f.unchecked_reason is UncheckedReason.BUDGET}
    )


def test_run_target_keeps_the_default_plan_caps_unless_its_caller_plans_with_none() -> None:
    risky = tier_a("1", patch=SECURITY_PATH)
    capped, uncapped = world(), world()
    default = run_target(
        risky, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=capped.seams, ledgers=capped.ledgers
    )
    assert len(default.plan.attacks) == 6
    assert over_the_caps(risky, (e.finding for e in capped.ledgers["spark"].items)) == OVER_THE_CAPS
    assert not any(note.startswith("planned with") for note in default.planner_notes)
    every = run_target(
        risky,
        temperatures=TemperatureMode.SPEC,
        arms=ARMS[:1],
        seams=uncapped.seams,
        ledgers=uncapped.ledgers,
        plan_caps=PlanCaps(max_lenses=None, max_attacks_per_lens=None),
    )
    assert len(every.plan.attacks) == 10 and every.plan.unchecked == ()
    assert over_the_caps(risky, (e.finding for e in uncapped.ledgers["spark"].items)) == []
    assert "planned with no lens cap and no per-lens cap, not the default: 10 of 10 attacks kept" in every.planner_notes


# ---------------------------------------------------------------- receipts run


class FileLedger(MemoryLedger):
    """A ledger kept in memory and mirrored to its file, the way the integrator's JSONL ledger is opened by path."""

    opened: ClassVar[dict[Path, FileLedger]] = {}

    def __init__(self, path: Path) -> None:
        super().__init__()
        self.path = path
        FileLedger.opened[path] = self

    def append(self, entry: LedgerEntry) -> None:
        super().append(entry)
        with self.path.open("a") as out:
            out.write(entry.model_dump_json() + "\n")


@pytest.fixture
def receipts_run(monkeypatch: pytest.MonkeyPatch) -> tuple[Any, World]:
    w = world()
    wiring = RunWiring(seams=lambda arms: w.seams, ledger=FileLedger)
    monkeypatch.setattr(cli, "LANE_COMMANDS", (lambda sub: register(sub, wiring=wiring),))
    return (lambda *argv: cli.main(list(argv))), w


def write_run_files(tmp_path: Path, targets: Sequence[Target]) -> tuple[Path, Path]:
    run_file = tmp_path / "pilot.jsonl"
    modes = itertools.cycle(["spec", "meta_default"])
    run_file.write_text(
        "".join(json.dumps({"target": t.model_dump(mode="json"), "temperatures": next(modes)}) + "\n" for t in targets)
    )
    arms_file = tmp_path / "arms.json"
    arms_file.write_text(json.dumps([a.model_dump(mode="json") for a in ARMS]))
    return run_file, arms_file


def test_receipts_run_drives_two_arms_over_a_run_file(
    receipts_run: tuple[Any, World], tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    run, _ = receipts_run
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1"), tier_a("2")])
    ledger_dir = tmp_path / "ledgers"
    assert run("run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(ledger_dir)) == 0
    out = capsys.readouterr().out
    assert "2 of 2 targets done" in out
    for arm in ARMS:
        lines = (ledger_dir / f"{arm.name}.jsonl").read_text().splitlines()
        assert len(lines) == 2 * len(EXPECTED)
        assert all(LedgerEntry.model_validate_json(line).v == 2 for line in lines)
    summaries = sorted((ledger_dir / "targets").glob("*.json"))
    assert [p.stem for p in summaries] == [tier_a("1").id, tier_a("2").id]
    modes = [TargetSummary.model_validate_json(p.read_text()).temperatures for p in summaries]
    assert modes == [TemperatureMode.SPEC, TemperatureMode.META_DEFAULT]

    # A second run over the same directory is refused, unless it resumes.
    assert run("run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(ledger_dir)) == 2
    assert "already have a summary" in capsys.readouterr().err
    assert (
        run("run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(ledger_dir), "--resume")
        == 0
    )
    assert "2 skipped" in capsys.readouterr().out


def test_receipts_run_still_plans_under_the_default_caps(receipts_run: tuple[Any, World], tmp_path: Path) -> None:
    run, _ = receipts_run
    risky = tier_a("1", patch=SECURITY_PATH)
    run_file, arms_file = write_run_files(tmp_path, [risky])
    ledger_dir = tmp_path / "ledgers"
    assert run("run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(ledger_dir)) == 0
    for arm in ARMS:
        lines = (ledger_dir / f"{arm.name}.jsonl").read_text().splitlines()
        assert over_the_caps(risky, (LedgerEntry.model_validate_json(line).finding for line in lines)) == OVER_THE_CAPS
    summary = TargetSummary.model_validate_json((ledger_dir / "targets" / f"{risky.id}.json").read_text())
    assert len(summary.plan.attacks) == 6
    assert not any(note.startswith("planned with") for note in summary.planner_notes)


@pytest.mark.parametrize(
    ("run_text", "arms_text", "message"),
    [
        ("", None, "no targets"),
        ("{not json}\n", None, "line 1"),
        ('{"target": {}, "temperatures": "spec"}\n', None, "line 1"),
        (None, "[]", "no arms"),
        (None, '[{"name": "spark", "attacker_model": "muse-spark-1.3-contributor"}]', "arms file"),
        (None, '[{"name": "a"}, {"name": "a"}]', "appears twice"),
    ],
    ids=["empty-run-file", "bad-json", "bad-target", "no-arms", "contributor-arm", "duplicate-arm"],
)
def test_receipts_run_refuses_files_it_cannot_use(
    receipts_run: tuple[Any, World],
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    run_text: str | None,
    arms_text: str | None,
    message: str,
) -> None:
    run, w = receipts_run
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1")])
    if run_text is not None:
        run_file.write_text(run_text)
    if arms_text is not None:
        arms_file.write_text(arms_text)
    assert run("run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(tmp_path / "l")) == 2
    assert message in capsys.readouterr().err
    assert w.models.requests == [] and not (tmp_path / "l").exists()


def test_receipts_run_without_wiring_says_so(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "LANE_COMMANDS", (register,))
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1")])
    code = cli.main(["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(tmp_path / "l")])
    assert code == 2
    assert "not wired" in capsys.readouterr().err


def test_receipts_run_reports_wiring_that_cannot_start_and_runs_nothing(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    def no_key(arms: Sequence[Arm]) -> Seams:
        raise ModelCallError("no Meta API key: set FALDEN_META_API_KEY", UncheckedReason.RUNTIME)

    wiring = RunWiring(seams=no_key, ledger=FileLedger)
    monkeypatch.setattr(cli, "LANE_COMMANDS", (lambda sub: register(sub, wiring=wiring),))
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1")])
    code = cli.main(["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(tmp_path / "l")])
    assert code == 2
    assert "could not start: ModelCallError: no Meta API key" in capsys.readouterr().err
    assert not (tmp_path / "l").exists()


# ---------------------------------------------------------------- round 3: the integrator's calls and the test gaps


def test_the_next_attacker_runs_only_when_no_candidate_failed_its_first_patch_run() -> None:
    # AT2 stage 1 gives 3 candidates. One that fails its first patch run on a counted failure is enough to stop the
    # escalation; candidates that pass, or fail on an import error, are not.
    w = world(
        stage1_script={
            "correctness-1": ["passes", "blocker", "passes"],  # one failed its first run: stage 2 stays out
            "correctness-2": ["passes", "import", "passes"],  # an import failure never counts: stage 2 runs
            "correctness-3": ["blocker", "blocker", "blocker"],
        },
        stage2_script={"correctness-1": ["blocker"], "correctness-2": ["blocker"], "correctness-3": ["blocker"]},
    )
    target = tier_a("1")
    run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    assert sorted({call[2] for call in w.stage2.calls}) == ["correctness-2"]
    found = findings_of(w.ledgers["spark"], target)
    assert [found[f"fixed_x/correctness-1.{n}"] for n in (1, 2, 3)] == [
        Status.NOT_REPRODUCED,
        Status.S_SAT_INCONCLUSIVE,
        Status.NOT_REPRODUCED,
    ]
    assert found["fixed_x/correctness-2.4"] is Status.S_SAT_INCONCLUSIVE  # stage 2's candidate, after stage 1's 3
    # The claim's first-run failures before correctness-3 are 2 (1.2 and 2.4), so its 3 are judged too, within the top-n
    # of 5; the candidates that passed their first run took no slot.
    assert [found[f"fixed_x/correctness-3.{n}"] for n in (1, 2, 3)] == [Status.S_SAT_INCONCLUSIVE] * 3


@pytest.mark.parametrize(("cap", "attacks_run"), [(101, False), (102, True)], ids=["at-the-cap", "under-the-cap"])
def test_the_per_target_cap_counts_the_shared_plan_cost_and_binds_when_spend_reaches_it(
    cap: int, attacks_run: bool
) -> None:
    # The plan's call costs 101 micro-dollars, the first cost issued. A cap of 101 is reached by the plan alone, so no
    # attacker runs at all; at 102 the first attack still runs.
    w = world(target_caps={Tier.A: cap})
    run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    assert w.costs.issued[0].stage == PLAN_STAGE and w.costs.issued[0].micro_usd == 101
    calls = w.regression.calls + w.stage1.calls + w.stage2.calls
    assert bool(calls) is attacks_run
    if not attacks_run:
        findings = [e.finding for e in w.ledgers["spark"].items]
        assert len(findings) == 6
        assert all(f.unchecked_reason is UncheckedReason.BUDGET for f in findings)
        assert all("(101 spent on this target)" in (f.detail or "") for f in findings)


def test_each_arm_s_wall_seconds_start_from_its_own_start() -> None:
    # Planning takes 7 s and the spark arm's attacks 1,000 s each; the glimmer arm's take 1 s. The glimmer arm's
    # findings read 7 s of shared planning plus its own seconds, never the spark arm's.
    now = {"t": 0.0}

    class Timed(Models):
        def complete(self, request: ModelRequest) -> ModelResponse:
            now["t"] += 7.0
            return super().complete(request)

    class Slow(FakeAttacker):
        def attack(
            self, target: Target, attack: Attack, *, arm: Arm, client: Any, sandbox: Any, runner: Any
        ) -> AttackOutcome:
            now["t"] += 1_000.0 if arm.name == "spark" else 1.0
            return super().attack(target, attack, arm=arm, client=client, sandbox=sandbox, runner=runner)

    w = world(clock=lambda: now["t"])
    slow = Slow(Lens.REGRESSION, "slow", w.costs, REGRESSION_SCRIPT)
    timed = Timed(w.costs)
    seams = replace(w.seams, planner_client=timed, attackers={Lens.REGRESSION: [slow]})
    summary = run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS, seams=seams, ledgers=w.ledgers)

    glimmer = [e.finding.wall_seconds for e in w.ledgers["glimmer"].items]
    spark = [e.finding.wall_seconds for e in w.ledgers["spark"].items]
    assert glimmer and all(wall is not None and 7.0 <= wall <= 7.0 + 3.0 for wall in glimmer)
    assert max(wall or 0.0 for wall in spark) >= 3_000.0
    assert [arm.wall_seconds for arm in summary.arms] == [3_000.0, 3.0]  # each arm's own seconds
    assert summary.wall_seconds == 7.0 + 3_000.0 + 3.0


class AttackerCrash(BaseException):
    """A failure that is not an Exception, as SystemExit is not, so the attacker guard does not take it and the target
    stops mid-run."""


def test_a_target_that_raises_mid_run_records_what_it_spent_and_resume_reruns_it(tmp_path: Path) -> None:
    # The spark arm records correctness-1 and regression-1, then correctness-2's stage 1 attacker spends and its
    # stage 2 attacker crashes: the target stops with the plan's and that attacker's costs on no finding.
    w = world(stage2_script={**STAGE2_SCRIPT, "correctness-2": AttackerCrash("killed mid-attack")})
    w.stage1.script = {**STAGE1_SCRIPT, "correctness-2": ["passes"]}
    items = [RunItem(target=tier_a("1"), temperatures=TemperatureMode.SPEC)]
    with pytest.raises(AttackerCrash):
        run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path)

    failed = read_summary(summary_path(tmp_path, items[0].target.id))
    assert (failed.status, failed.error) == ("failed", "AttackerCrash: killed mid-attack")
    assert [c.stage for c in failed.costs] == [PLAN_STAGE, "attack:at2-stage1"]  # intake (Tier A) made no call
    assert [a.arm for a in failed.arms] == ["spark"] and failed.arms[0].findings == 3
    assert len(failed.plan.attacks) == 6 and failed.claims
    assert_every_cost_once(w.costs, w.ledgers.values(), [failed])

    # Resume: the failed target runs again, carrying the failed attempt forward, and every cost is still on record
    # exactly once across both attempts' findings and the one summary left.
    w.stage2.script = STAGE2_SCRIPT
    w.stage1.script = STAGE1_SCRIPT
    report = run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path, resume=True)
    assert (report.rerun, report.skipped, report.failed) == ((items[0].target.id,), (), ())
    final = read_summary(summary_path(tmp_path, items[0].target.id))
    assert final.status == "done" and final.error == ""
    assert [(a.started_at, a.error) for a in final.earlier_attempts] == [(failed.started_at, failed.error)]
    assert final.costs[:2] == failed.costs  # the failed attempt's costs, carried once
    assert_every_cost_once(w.costs, w.ledgers.values(), [final])

    again = run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path, resume=True)
    assert (again.done, again.skipped, again.rerun) == ((), (items[0].target.id,), ())


def test_an_attacker_that_raises_a_model_call_error_keeps_its_cost_once() -> None:
    spent = StageCost(stage="attack:at3", model=SPARK, micro_usd=7_777)
    w = world(
        regression_script={
            **REGRESSION_SCRIPT,
            "regression-1": ModelCallError("cap", UncheckedReason.BUDGET, cost=spent),
        }
    )
    target = tier_a("1")
    run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    (finding,) = [e.finding for e in w.ledgers["spark"].items if e.finding.id.endswith("/regression-1.1")]
    assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.RUNTIME)
    assert finding.costs == (spent,)
    on_record = [c for e in w.ledgers["spark"].items for c in e.finding.costs if c.micro_usd == 7_777]
    assert len(on_record) == 1


def test_a_plan_spark_makes_after_wonder_fails_validation_twice_is_named_and_costed_once() -> None:
    # REG-570 end to end: Wonder answers twice with a plan that fails validation, one Spark call makes the plan, and
    # the summary (what the receipt reads) names the fallback; the three calls' costs are on it, once each.
    w = world(wonder_invalid=True)
    target = tier_a("1")
    summary = run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers)
    assert [(r.stage, r.model) for r in w.models.requests] == [
        (PLAN_STAGE, PLANNER_MODEL),
        (PLAN_STAGE, PLANNER_MODEL),
        (PLAN_STAGE, FALLBACK_MODEL),
    ]
    assert (summary.planner_model, summary.planner_fallback) == (FALLBACK_MODEL, True)
    assert "failed schema validation twice" in summary.planner_fallback_reason
    assert "REG-570" in summary.planner_fallback_reason
    assert summary.plan.unchecked[0].what == f"{PLAN_STAGE}:{PLANNER_MODEL}"
    assert [(c.stage, c.model) for c in summary.costs] == [
        (PLAN_STAGE, PLANNER_MODEL),
        (PLAN_STAGE, PLANNER_MODEL),
        (PLAN_STAGE, FALLBACK_MODEL),
    ]
    assert len(summary.plan.attacks) == 6
    assert not any(c.stage == PLAN_STAGE for e in w.ledgers["spark"].items for c in e.finding.costs)
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])


def test_a_summary_s_status_and_error_go_together() -> None:
    w = world()
    summary = run_target(
        tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers
    )
    with pytest.raises(ValueError, match="names its error"):
        TargetSummary.model_validate({**summary.model_dump(), "status": "failed"})
    with pytest.raises(ValueError, match="names its error"):
        TargetSummary.model_validate({**summary.model_dump(), "error": "x"})
    with pytest.raises(ValueError, match="only a failed summary"):
        run_target(
            tier_a("1"),
            temperatures=TemperatureMode.SPEC,
            arms=ARMS[:1],
            seams=w.seams,
            ledgers=w.ledgers,
            previous=summary,
        )


def test_receipts_run_resume_reruns_a_failed_target_and_names_a_planner_fallback(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = world(wonder_invalid=True)
    broken = {"on": True}

    class Brittle(FileLedger):
        def append(self, entry: LedgerEntry) -> None:
            if broken["on"]:
                raise OSError("disk full")
            super().append(entry)

    wiring = RunWiring(seams=lambda arms: w.seams, ledger=Brittle)
    monkeypatch.setattr(cli, "LANE_COMMANDS", (lambda sub: register(sub, wiring=wiring),))
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1")])
    argv = ["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(tmp_path / "l")]

    assert cli.main(argv) == 1
    assert "failed: OSError: disk full" in capsys.readouterr().err
    path = summary_path(tmp_path / "l" / "targets", tier_a("1").id)
    assert read_summary(path).status == "failed"

    assert cli.main(argv) == 2  # a summary is there: refused without --resume
    capsys.readouterr()
    broken["on"] = False
    assert cli.main([*argv, "--resume"]) == 0
    out = capsys.readouterr().out
    assert "1 of 1 targets done, 0 skipped, 0 failed, 1 run again after a failed run" in out
    assert f"planned by {FALLBACK_MODEL} (fallback)" in out
    assert read_summary(path).status == "done"


def test_a_planner_that_raises_after_spending_leaves_its_costs_on_the_failed_summary(tmp_path: Path) -> None:
    # Wonder's plan fails validation twice (both calls spent), then the client breaks on the Spark call with an
    # error that is no ModelCallError: planning raises, and the two Wonder calls' costs are still on record.
    w = world(wonder_invalid=True)

    class BreaksOnSpark(Models):
        def complete(self, request: ModelRequest) -> ModelResponse:
            if request.model == FALLBACK_MODEL:
                raise RuntimeError("client bug")
            return super().complete(request)

    seams = replace(w.seams, planner_client=BreaksOnSpark(w.costs, wonder_invalid=True))
    items = [RunItem(target=tier_a("1"), temperatures=TemperatureMode.SPEC)]
    report = run_targets(items, arms=ARMS, seams=seams, ledgers=w.ledgers, summary_dir=tmp_path)
    assert report.failed == ((items[0].target.id, "RuntimeError: client bug"),)
    failed = read_summary(summary_path(tmp_path, items[0].target.id))
    assert [(c.stage, c.model) for c in failed.costs] == [(PLAN_STAGE, PLANNER_MODEL), (PLAN_STAGE, PLANNER_MODEL)]
    assert failed.plan.attacks == () and failed.arms == ()
    assert_every_cost_once(w.costs, w.ledgers.values(), [failed])


# ---------------------------------------------------------------- round 4: contract v3 and the round-3 test gaps


def test_the_pipeline_s_own_findings_say_why_in_detail_never_as_a_refutation() -> None:
    w = world()
    target = tier_a("1")
    run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    findings = {e.finding.id.removeprefix(target.id + "/"): e.finding for e in w.ledgers["spark"].items}
    nothing, raised = findings["nothing_else_changed/regression-3"], findings["fixed_x/correctness-3.2"]
    assert (nothing.status, nothing.detail) == (Status.NOT_REPRODUCED, "the attack produced no candidate repro")
    assert (raised.status, raised.detail) == (Status.UNCHECKED, "the correctness attacker raised RuntimeError")
    assert all(f.refutation is None for f in findings.values())

    capped = world(target_caps={Tier.A: 400})
    run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=capped.seams, ledgers=capped.ledgers)
    budget = [e.finding for e in capped.ledgers["spark"].items if e.finding.unchecked_reason is UncheckedReason.BUDGET]
    assert budget and all("per-target cap of 400" in (f.detail or "") and f.refutation is None for f in budget)


class AppliedSandbox(FakeSandbox):
    """Prepares the patch side as if the patch applied only with GNU patch's fuzz, the base side with no patch."""

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        workspace = super().prepare(target, side, patch=patch)
        method = PatchMethod.GNU_PATCH_FUZZ if side is Side.PATCH else None
        return workspace.model_copy(update={"patch_method": method})


def test_the_summary_records_how_each_skeptic_workspace_s_patch_applied(tmp_path: Path) -> None:
    w = world(sandbox=AppliedSandbox())
    items = [RunItem(target=tier_a("1"), temperatures=TemperatureMode.SPEC)]
    run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path)
    summary = read_summary(summary_path(tmp_path, items[0].target.id))
    assert [(p.side, p.patch_method, p.image_digest) for p in summary.skeptic_workspaces] == [
        (Side.PATCH, PatchMethod.GNU_PATCH_FUZZ, DIGEST),
        (Side.BASE, None, DIGEST),
    ]
    raw = json.loads(summary_path(tmp_path, items[0].target.id).read_text())
    assert raw["skeptic_workspaces"][0] == {"side": "patch", "patch_method": "gnu_patch_fuzz", "image_digest": DIGEST}
    # A summary written before this field existed still reads, with no workspaces listed.
    del raw["skeptic_workspaces"]
    assert TargetSummary.model_validate(raw).skeptic_workspaces == ()


def test_the_run_line_says_when_a_patch_applied_only_with_fuzz(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = world(sandbox=AppliedSandbox())
    wiring = RunWiring(seams=lambda arms: w.seams, ledger=FileLedger)
    monkeypatch.setattr(cli, "LANE_COMMANDS", (lambda sub: register(sub, wiring=wiring),))
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1")])
    argv = ["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(tmp_path / "l")]
    assert cli.main(argv) == 0
    assert "; the agent's patch applied only with GNU patch --fuzz=5 (" in capsys.readouterr().out


def test_a_failed_summary_keeps_the_workspaces_prepared_before_the_failure(tmp_path: Path) -> None:
    w = world(sandbox=AppliedSandbox(), stage2_script={**STAGE2_SCRIPT, "correctness-1": AttackerCrash("killed")})
    items = [RunItem(target=tier_a("1"), temperatures=TemperatureMode.SPEC)]
    with pytest.raises(AttackerCrash):
        run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path)
    failed = read_summary(summary_path(tmp_path, items[0].target.id))
    assert failed.status == "failed"
    assert [(p.side, p.patch_method) for p in failed.skeptic_workspaces] == [(Side.PATCH, PatchMethod.GNU_PATCH_FUZZ)]


def test_receipts_run_refuses_an_existing_summary_before_it_builds_anything(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The command's own refusal (round-3 review, CL2): with a summary already there and no --resume, it builds no
    # seams and opens no ledger; it does not lean on run_targets refusing later, after the run's clients exist.
    w = world()
    built: list[str] = []

    def seams(arms: Sequence[Arm]) -> Seams:
        built.append("seams")
        return w.seams

    def ledger(path: Path) -> FileLedger:
        built.append(path.name)
        return FileLedger(path)

    monkeypatch.setattr(
        cli, "LANE_COMMANDS", (lambda sub: register(sub, wiring=RunWiring(seams=seams, ledger=ledger)),)
    )
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1")])
    ledger_dir = tmp_path / "l"
    summary_path(ledger_dir / "targets", tier_a("1").id).parent.mkdir(parents=True)
    summary_path(ledger_dir / "targets", tier_a("1").id).write_text("{}")
    code = cli.main(["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(ledger_dir)])
    assert code == 2
    err = capsys.readouterr().err
    assert "already have a summary" in err and "or use a new --ledger-dir" in err
    assert built == [] and not any(ledger_dir.glob("*.jsonl"))


# ---------------------------------------------------------------- round 6: --jobs above 1, a run phase per target


class ConcurrentUse(AssertionError):
    """A fake sandbox or ledger was used the way one not shown to be thread-safe must never be."""


class Meeting:
    """The first `parties` sandbox prepares of a run, across every sandbox, wait here for one another, each inside its
    own sandbox, so a test sees that many targets in flight at the same moment."""

    def __init__(self, parties: int, timeout_s: float = 5.0) -> None:
        self.barrier = threading.Barrier(parties, timeout=timeout_s)
        self._left = parties
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            if self._left == 0:
                return
            self._left -= 1
        self.barrier.wait()


class OwnSandbox(FakeSandbox):
    """A fake sandbox for one target that takes one call at a time, the way a sandbox not shown to be thread-safe may
    be used. A call while another call is in it, a call for a second target or a call after its release is a
    violation: it goes on the run's shared list and raises. Its first prepare waits at the run's meeting while inside
    the sandbox, so a second target sharing it would find it busy."""

    def __init__(self, violations: list[str], meeting: Meeting | None = None, *, one_target: bool = True) -> None:
        super().__init__()
        self.violations, self.meeting, self.one_target = violations, meeting, one_target
        self.targets: list[str] = []
        self.releases: list[tuple[int, int]] = []  # (workspaces prepared, workspaces closed) at each release
        self._busy = threading.Lock()

    @contextmanager
    def one_call(self, what: str, target_id: str | None = None) -> Iterator[None]:
        if not self._busy.acquire(blocking=False):
            self.violate(f"{what} while another call was in the sandbox")
        try:
            if self.releases:
                self.violate(f"{what} after the sandbox was released")
            if target_id is not None and target_id not in self.targets:
                self.targets.append(target_id)
                if self.one_target and len(self.targets) > 1:
                    self.violate(f"{what} for {target_id}, a second target")
            yield
        finally:
            self._busy.release()

    def violate(self, why: str) -> NoReturn:
        self.violations.append(why)
        raise ConcurrentUse(why)

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        with self.one_call("prepare", target.id):
            if self.meeting is not None and not self.prepared:
                self.meeting.wait()
            return super().prepare(target, side, patch=patch)

    def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
        with self.one_call("run"):
            return super().run(workspace, command, **options)

    def run_test(self, workspace: Workspace, tests: Sequence[str], **options: Any) -> RunResult:
        with self.one_call("run_test"):
            return super().run_test(workspace, tests, **options)

    def close(self, workspace: Workspace) -> None:
        with self.one_call("close"):
            super().close(workspace)

    def release(self) -> None:
        with self.one_call("release"):
            self.releases.append((len(self.prepared), self.closed))


class Phases:
    """Seams.run_phase for a test: a new OwnSandbox and a fresh runner on every call, with the sandbox's release as
    the run phase's, each one kept."""

    def __init__(self, meeting: Meeting | None = None) -> None:
        self.meeting = meeting
        self.violations: list[str] = []
        self.made: list[RunPhase] = []
        self._lock = threading.Lock()

    def __call__(self) -> RunPhase:
        sandbox = OwnSandbox(self.violations, self.meeting)
        phase = RunPhase(sandbox=sandbox, runners={RunnerKind.PYTEST: Runner()}, release=sandbox.release)
        with self._lock:
            self.made.append(phase)
        return phase

    def by_target(self) -> dict[str, tuple[OwnSandbox, RunPhase]]:
        """Each target's run phase, by the one target its sandbox served."""
        found: dict[str, tuple[OwnSandbox, RunPhase]] = {}
        for phase in self.made:
            assert isinstance(phase.sandbox, OwnSandbox)
            (target_id,) = phase.sandbox.targets
            assert target_id not in found
            found[target_id] = (phase.sandbox, phase)
        return found


def eight_targets() -> list[RunItem]:
    return [RunItem(target=tier_a(str(n)), temperatures=TemperatureMode.SPEC) for n in range(1, 9)]


def test_jobs_above_one_run_each_target_on_its_own_sandbox_and_runner(tmp_path: Path) -> None:
    # Eight targets, four at once: the meeting needs four targets inside their own sandboxes at the same moment.
    phases = Phases(Meeting(4))
    w = world(run_phase=phases)
    items = eight_targets()
    report = run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path, jobs=4)
    assert (len(report.done), report.failed, phases.violations) == (8, (), [])
    assert phases.meeting is not None and not phases.meeting.barrier.broken
    # The factory ran once per target, and each sandbox served that one target, in both arms.
    own = phases.by_target()
    assert len(phases.made) == 8 and set(own) == {item.target.id for item in items}
    handed = [*w.regression.got, *w.stage1.got, *w.stage2.got]
    assert len(handed) == 8 * len(ARMS) * 8  # every attacker call: 3 regression, 3 stage 1 (one raises), 2 stage 2
    for target_id, sandbox, runner in handed:
        mine, phase = own[target_id]
        assert sandbox is mine and runner is phase.runners[RunnerKind.PYTEST], target_id
    # The skeptic ran on the same sandbox, and each sandbox was released once, after its target closed both of its
    # workspaces (patch and base).
    for target_id, (sandbox, _) in own.items():
        assert {t for t, _ in sandbox.prepared} == {target_id} and sandbox.runs
        assert sandbox.releases == [(2, 2)], target_id
    for arm in ARMS:
        for item in items:
            assert findings_of(w.ledgers[arm.name], item.target) == EXPECTED
    assert_every_cost_once(w.costs, w.ledgers.values(), report.done)


class HeldOpen(OwnSandbox):
    """Holds its first prepare open, inside the sandbox, until the test lets it go, so one target is mid-run on it
    while another starts."""

    def __init__(self, violations: list[str]) -> None:
        super().__init__(violations, one_target=False)
        self.inside, self.go = threading.Event(), threading.Event()

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        with self.one_call("prepare", target.id):
            if not self.inside.is_set():
                self.inside.set()
                self.go.wait(timeout=5.0)
            return FakeSandbox.prepare(self, target, side, patch=patch)


def test_a_run_phase_another_running_target_holds_stops_the_second_target_before_it_spends() -> None:
    # A broken factory hands its one run phase to every caller. The first target is mid-run on it, inside the sandbox;
    # the second target is refused before it plans, spends or touches the sandbox, and the first goes on.
    violations: list[str] = []
    held = HeldOpen(violations)
    shared = RunPhase(sandbox=held, runners={RunnerKind.PYTEST: Runner()})
    w = world(run_phase=lambda: shared)
    first, second = tier_a("1"), tier_a("2")
    results: list[TargetSummary] = []

    def run_first() -> None:
        results.append(
            run_target(first, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers)
        )

    running = threading.Thread(target=run_first)
    running.start()
    failed: list[TargetSummary] = []
    try:
        assert held.inside.wait(timeout=5.0)
        spent_before = len(w.costs.issued)
        said = f"target {second.id} was handed the sandbox that target {first.id} is running on"
        with pytest.raises(RunPhaseSharedError, match=re.escape(said)):
            run_target(
                second,
                temperatures=TemperatureMode.SPEC,
                arms=ARMS,
                seams=w.seams,
                ledgers=w.ledgers,
                on_failure=failed.append,
            )
        assert len(w.costs.issued) == spent_before  # the second target made no model call
    finally:
        held.go.set()
        running.join(timeout=10.0)
    (summary,) = failed
    assert (summary.status, summary.costs, summary.plan.attacks, summary.arms) == ("failed", (), (), ())
    assert summary.error.startswith("RunPhaseSharedError: ")
    assert not any(e.finding.target_id == second.id for ledger in w.ledgers.values() for e in ledger.items)
    assert [s.status for s in results] == ["done"] and violations == []
    assert {t for t, _ in held.prepared} == {first.id}
    # Done, the first target holds nothing: a later target may run on the same run phase, one at a time.
    assert run_target(
        second, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers
    ).status == ("done")
    assert violations == []


def test_more_than_one_job_on_one_shared_sandbox_is_refused_before_anything_runs(tmp_path: Path) -> None:
    w = world()  # one sandbox and one runner mapping for every target
    items = eight_targets()[:2]
    said = r"2 targets at once need a run phase of each target's own \(Seams\.run_phase\)"
    with pytest.raises(ValueError, match=said):
        run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path / "targets", jobs=2)
    assert w.models.requests == [] and w.sandbox.prepared == [] and not (tmp_path / "targets").exists()
    with pytest.raises(ValueError, match="jobs is at least 1"):
        check_jobs(0, w.seams)
    check_jobs(1, w.seams)  # one target at a time is what shared seams run
    check_jobs(16, world(run_phase=Phases()).seams)
    report = run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path / "targets")
    assert len(report.done) == 2


def test_the_seams_take_a_run_phase_factory_or_one_shared_sandbox_and_runners_never_both() -> None:
    w = world()
    with pytest.raises(ValueError, match="not both"):
        replace(w.seams, run_phase=Phases())
    with pytest.raises(ValueError, match="need a run_phase factory, or one sandbox and one runner mapping"):
        replace(w.seams, sandbox=None)
    with pytest.raises(ValueError, match="need a run_phase factory, or one sandbox and one runner mapping"):
        replace(w.seams, runners=None)
    phased = world(run_phase=Phases()).seams
    assert phased.sandbox is None and phased.runners is None and phased.run_phase is not None


def test_a_factory_that_returns_no_run_phase_fails_its_target_before_anything_is_spent(tmp_path: Path) -> None:
    w = world(run_phase=lambda: FakeSandbox())
    items = eight_targets()[:1]
    report = run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path)
    assert report.failed == ((items[0].target.id, "TypeError: Seams.run_phase returned a FakeSandbox, not a RunPhase"),)
    assert w.models.requests == [] and read_summary(summary_path(tmp_path, items[0].target.id)).costs == ()


def test_a_target_that_fails_still_releases_its_run_phase_and_a_failed_release_is_only_logged(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    phases = Phases()
    w = world(run_phase=phases, stage2_script={**STAGE2_SCRIPT, "correctness-1": AttackerCrash("killed")})
    items = eight_targets()[:1]
    with pytest.raises(AttackerCrash):
        run_targets(items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path)
    ((sandbox, _),) = phases.by_target().values()
    assert len(sandbox.releases) == 1  # released once, after the crash, with whatever the skeptic had closed

    def broken() -> None:
        raise OSError("no such container")

    breaks = world(
        run_phase=lambda: RunPhase(sandbox=FakeSandbox(), runners={RunnerKind.PYTEST: Runner()}, release=broken)
    )
    report = run_targets(items, arms=ARMS, seams=breaks.seams, ledgers=breaks.ledgers, summary_dir=tmp_path / "b")
    assert (len(report.done), report.failed) == (1, ())
    assert "its run phase could not be released: OSError('no such container')" in caplog.text


class OneAppendAtATime(MemoryLedger):
    """An arm's ledger that refuses an append made while another append is still writing, as two JSONL writers at
    once would interleave their lines. Each append takes a little time, as one with an fsync does."""

    def __init__(self, overlaps: list[str]) -> None:
        super().__init__()
        self.overlaps = overlaps
        self._busy = threading.Lock()

    def append(self, entry: LedgerEntry) -> None:
        if not self._busy.acquire(blocking=False):
            self.overlaps.append(entry.finding.id)
            raise ConcurrentUse(f"{entry.finding.id} was appended while another append was writing")
        try:
            time.sleep(0.002)
            super().append(entry)
        finally:
            self._busy.release()


def test_each_arm_s_ledger_takes_one_append_at_a_time_and_keeps_every_finding_with_jobs(tmp_path: Path) -> None:
    overlaps: list[str] = []
    phases = Phases(Meeting(4))
    w = world(run_phase=phases)
    ledgers = {arm.name: OneAppendAtATime(overlaps) for arm in ARMS}
    items = eight_targets()
    report = run_targets(items, arms=ARMS, seams=w.seams, ledgers=ledgers, summary_dir=tmp_path, jobs=4)
    assert (len(report.done), report.failed, overlaps) == (8, (), [])
    assert phases.meeting is not None and not phases.meeting.barrier.broken
    # Each target's findings, in each arm, are the ones one target alone makes, in the same order, each once.
    alone = world()
    run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS, seams=alone.seams, ledgers=alone.ledgers)
    order = {arm.name: [e.finding.id.split("/", 1)[1] for e in alone.ledgers[arm.name].items] for arm in ARMS}
    for arm in ARMS:
        assert len(ledgers[arm.name].items) == 8 * len(EXPECTED)
        for item in items:
            made = [e.finding.id.split("/", 1)[1] for e in ledgers[arm.name].entries(item.target.id)]
            assert made == order[arm.name], (arm.name, item.target.id)
    assert_every_cost_once(w.costs, ledgers.values(), report.done)


def test_on_done_is_called_by_one_worker_at_a_time(tmp_path: Path) -> None:
    # The first call waits up to half a second for any other call to start alongside it: four targets finish at about
    # the same moment (the meeting starts them together), so a second call would start within it.
    started = threading.Event()
    overlaps: list[str] = []
    busy = threading.Lock()
    called: list[str] = []

    def on_done(summary: TargetSummary) -> None:
        if not busy.acquire(blocking=False):
            overlaps.append(summary.target_id)
            started.set()
            return
        try:
            if not called:
                started.wait(timeout=0.5)
            called.append(summary.target_id)
        finally:
            busy.release()

    w = world(run_phase=Phases(Meeting(4)))
    items = eight_targets()
    report = run_targets(
        items, arms=ARMS, seams=w.seams, ledgers=w.ledgers, summary_dir=tmp_path, jobs=4, on_done=on_done
    )
    assert (len(report.done), overlaps) == (8, [])
    assert sorted(called) == sorted(item.target.id for item in items)


def test_receipts_run_with_jobs_runs_each_target_on_its_own_run_phase(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    phases = Phases(Meeting(3))
    w = world(run_phase=phases)
    wiring = RunWiring(seams=lambda arms: w.seams, ledger=FileLedger)
    monkeypatch.setattr(cli, "LANE_COMMANDS", (lambda sub: register(sub, wiring=wiring),))
    targets = [tier_a(str(n)) for n in range(1, 7)]
    run_file, arms_file = write_run_files(tmp_path, targets)
    ledger_dir = tmp_path / "ledgers"
    argv = ["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(ledger_dir)]
    assert cli.main([*argv, "--jobs", "3"]) == 0
    out = capsys.readouterr().out
    assert "6 of 6 targets done, 0 skipped, 0 failed" in out
    assert (len(phases.made), phases.violations) == (6, [])
    assert phases.meeting is not None and not phases.meeting.barrier.broken
    assert sorted(line.split(": ", 1)[0] for line in out.splitlines() if line.startswith("verified~")) == sorted(
        t.id for t in targets
    )
    for arm in ARMS:
        entries = [
            LedgerEntry.model_validate_json(line)
            for line in (ledger_dir / f"{arm.name}.jsonl").read_text().splitlines()
        ]
        assert len(entries) == 6 * len(EXPECTED)
        for t in targets:
            prefix = t.id + "/"
            assert {
                e.finding.id.removeprefix(prefix): e.finding.status for e in entries if e.finding.target_id == t.id
            } == (EXPECTED)


def test_receipts_run_refuses_jobs_above_one_on_shared_seams_before_it_creates_a_ledger(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    w = world()
    opened: list[Path] = []

    def ledger(path: Path) -> FileLedger:
        opened.append(path)
        return FileLedger(path)

    monkeypatch.setattr(
        cli, "LANE_COMMANDS", (lambda sub: register(sub, wiring=RunWiring(seams=lambda arms: w.seams, ledger=ledger)),)
    )
    run_file, arms_file = write_run_files(tmp_path, [tier_a("1"), tier_a("2")])
    ledger_dir = tmp_path / "ledgers"
    argv = ["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(ledger_dir)]
    assert cli.main([*argv, "--jobs", "2"]) == 2
    err = capsys.readouterr().err
    assert "receipts run: --jobs 2 is refused: 2 targets at once need a run phase of each target's own" in err
    assert (opened, ledger_dir.exists(), w.models.requests, w.sandbox.prepared) == ([], False, [], [])
    assert cli.main(argv) == 0  # the same seams, one target at a time
    assert "2 of 2 targets done" in capsys.readouterr().out


# ---------------------------------------------------------------- round 6: the spend meter with targets at once

META_ARMS = (Arm(name="spark", attacker_model="muse-spark-1.3"), Arm(name="spark12", attacker_model="muse-spark-1.2"))


class FakeMetaAPI:
    """Meta's Chat Completions API behind httpx.MockTransport, safe across threads: a plan answers a plan request (the
    one plan_text gives) and 'ok' anything else. With `bounds`, each answer reports a usage of exactly the call's token
    bounds (every request byte an input token, the whole output budget), so the call costs exactly its hold; without
    it, 100 tokens in and 50 out. With `meet`, a plan request waits there for others (at most its timeout)."""

    def __init__(self, *, bounds: bool = False, meet: threading.Barrier | None = None) -> None:
        self.bounds, self.meet = bounds, meet
        self.bodies: list[dict[str, Any]] = []
        self._lock = threading.Lock()

    def handler(self, request: httpx.Request) -> httpx.Response:
        body: dict[str, Any] = json.loads(request.content)
        with self._lock:
            self.bodies.append(body)
        if body["prompt_cache_key"] == f"receipts-{PLAN_STAGE}":
            if self.meet is not None:
                with contextlib.suppress(threading.BrokenBarrierError):
                    self.meet.wait()
            content = plan_text(body["messages"][1]["content"])
        else:
            content = "ok"
        prompt, output = (len(request.content), body["max_completion_tokens"]) if self.bounds else (100, 50)
        return httpx.Response(
            200,
            json={
                "id": f"chatcmpl-{secrets.token_hex(4)}",
                "object": "chat.completion",
                "created": 1_793_000_000,
                "model": body["model"],
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": content, "refusal": None},
                        "finish_reason": "stop",
                        "logprobs": None,
                    }
                ],
                "usage": {
                    "prompt_tokens": prompt,
                    "completion_tokens": output,
                    "total_tokens": prompt + output,
                    "prompt_tokens_details": {"cached_tokens": 0},
                    "completion_tokens_details": {"reasoning_tokens": 0},
                },
            },
        )


@dataclass
class ModelAttacker(FakeAttacker):
    """A fake attacker that makes one model call per attack through the router it is handed, as the real ones do, and
    reports that call's own cost."""

    def attack(
        self, target: Target, attack: Attack, *, arm: Arm, client: Any, sandbox: Any, runner: Any
    ) -> AttackOutcome:
        self.calls.append((target.id, arm.name, attack.id, attack.temperature, client))
        self.got.append((target.id, sandbox, runner))
        step = self.script.get(attack.id, ())
        if isinstance(step, BaseException):
            raise step
        response = client.complete(
            ModelRequest(
                stage=f"attack:{self.name}",
                model=arm.attacker_model,
                messages=(Message(role="user", content=f"attack {attack.id} on {target.id}"),),
                temperature=0.0,
                max_tokens=64,
            )
        )
        assert response.cost is not None
        return AttackOutcome(
            attack_id=attack.id,
            repros=tuple(repro(attack, behavior, n) for n, behavior in enumerate(step, start=1)),
            costs=(response.cost,),
        )


def metered(tmp_path: Path, api: FakeMetaAPI, *, cap: int | None = None) -> tuple[MetaClient, SpendMeter]:
    """A Meta client on `api`, holding every call against its own spend file, armed the way an operator arms one."""
    meter = SpendMeter(tmp_path / "spend" / "spend.jsonl", **({} if cap is None else {"cap_micro_usd": cap}))
    meter.arm(credit_confirmed=True)
    client = MetaClient(
        api_key=f"stand-in-{secrets.token_hex(8)}",  # never a real key
        meter=meter,
        governor=TokenGovernor(),
        transport=httpx.MockTransport(api.handler),
    )
    return client, meter


def metered_seams(client: MetaClient, phases: Phases, *, attackers_call: bool = True) -> Seams:
    kind = ModelAttacker if attackers_call else FakeAttacker
    costs = Costs()  # only the attackers that make no model call spend from it
    return Seams(
        intake_client=client,
        planner_client=client,
        attacker_client=ModelRouter({arm.attacker_model: client for arm in META_ARMS}),
        attackers={
            Lens.REGRESSION: [kind(Lens.REGRESSION, "at3", costs, REGRESSION_SCRIPT)],
            Lens.CORRECTNESS: [
                kind(Lens.CORRECTNESS, "at2-stage1", costs, STAGE1_SCRIPT),
                kind(Lens.CORRECTNESS, "at2-stage2", costs, STAGE2_SCRIPT),
            ],
        },
        run_phase=phases,
    )


def spend_records(meter: SpendMeter) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    records = [json.loads(line) for line in meter.path.read_text(encoding="utf-8").splitlines()]
    return [r for r in records if r["kind"] == "hold"], [r for r in records if r["kind"] == "settle"]


def recorded_micro_usd(ledgers: Mapping[str, MemoryLedger], summaries: Iterable[TargetSummary]) -> int:
    on_findings = sum(c.micro_usd for ledger in ledgers.values() for e in ledger.items for c in e.finding.costs)
    return on_findings + sum(c.micro_usd for summary in summaries for c in summary.costs)


def test_the_spend_meter_and_the_records_agree_to_the_micro_dollar_with_targets_at_once(tmp_path: Path) -> None:
    # Eight targets, four at once, every plan and every attack a call through one Meta client and its one meter.
    api = FakeMetaAPI()
    client, meter = metered(tmp_path, api)
    phases = Phases(Meeting(4))
    ledgers = {arm.name: MemoryLedger() for arm in META_ARMS}
    items = eight_targets()
    report = run_targets(
        items, arms=META_ARMS, seams=metered_seams(client, phases), ledgers=ledgers, summary_dir=tmp_path / "t", jobs=4
    )
    assert (len(report.done), report.failed, phases.violations) == (8, (), [])
    for arm in META_ARMS:
        for item in items:
            assert findings_of(ledgers[arm.name], item.target) == EXPECTED
    # One plan call per target and 7 attacker calls per arm: each is one hold and one settle in the spend file.
    holds, settles = spend_records(meter)
    assert len(api.bodies) == len(holds) == len(settles) == 8 * (1 + 7 * len(META_ARMS))
    assert {s["hold"] for s in settles} == {h["id"] for h in holds}
    # Read back by a meter of its own, nothing is still held, and what was spent is what the records carry.
    later = SpendMeter(meter.path)
    assert later.held_micro_usd == 0
    assert later.spent_micro_usd == sum(s["micro_usd"] for s in settles) == recorded_micro_usd(ledgers, report.done)
    assert later.spent_micro_usd > 0


def test_targets_at_once_never_carry_spend_past_the_cap(tmp_path: Path) -> None:
    # Each answer reports exactly its call's token bounds, so a plan call costs exactly its hold, H. With the cap at
    # 2.5 H, at most two plan calls fit, in flight or spent. Four targets ask at the same moment: the API holds each
    # plan request until four have arrived, or half a second, so no call settles before the others have asked.
    probe = FakeMetaAPI(bounds=True)
    probe_client, probe_meter = metered(tmp_path / "probe", probe)
    one = eight_targets()[:1]
    run_targets(
        one,
        arms=META_ARMS,
        seams=metered_seams(probe_client, Phases(), attackers_call=False),
        ledgers={arm.name: MemoryLedger() for arm in META_ARMS},
        summary_dir=tmp_path / "probe" / "t",
    )
    (hold,), _ = spend_records(probe_meter)
    h = int(hold["micro_usd"])
    cap = 2 * h + h // 2

    api = FakeMetaAPI(bounds=True, meet=threading.Barrier(4, timeout=0.5))
    client, meter = metered(tmp_path, api, cap=cap)
    ledgers = {arm.name: MemoryLedger() for arm in META_ARMS}
    items = eight_targets()
    seams = metered_seams(client, Phases(), attackers_call=False)
    report = run_targets(items, arms=META_ARMS, seams=seams, ledgers=ledgers, summary_dir=tmp_path / "t", jobs=4)
    assert (len(report.done), report.failed) == (8, ())
    later = SpendMeter(meter.path)
    assert (later.spent_micro_usd, later.held_micro_usd) == (2 * h, 0)
    assert later.spent_micro_usd <= cap
    # The two plan calls' costs are on their targets' summaries; these attackers make no model call.
    assert sum(c.micro_usd for summary in report.done for c in summary.costs) == 2 * h
    # Two targets were planned; every claim of the other six is unchecked for the budget, in both arms.
    planned = [s.target_id for s in report.done if s.plan.attacks]
    assert len(planned) == 2
    for summary in report.done:
        if summary.target_id in planned:
            continue
        assert {u.reason for u in summary.plan.unchecked} == {UncheckedReason.BUDGET}
        for arm in META_ARMS:
            found = [e.finding for e in ledgers[arm.name].entries(summary.target_id)]
            assert found and {(f.status, f.unchecked_reason) for f in found} == {
                (Status.UNCHECKED, UncheckedReason.BUDGET)
            }


# ---------------------------------------------------------------- fix round 5: one test case per candidate


@dataclass
class SeveralCases:
    """A regression attacker whose one repro is a file of `cases` passing-on-base, failing-on-patch test functions."""

    cases: int
    lens: Lens = Lens.REGRESSION

    def attack(
        self, target: Target, attack: Attack, *, arm: Arm, client: Any, sandbox: Any, runner: Any
    ) -> AttackOutcome:
        path = "tests/test_receipts_several.py"
        body = "# blocker\n" + "".join(f"def test_{n}():\n    assert True\n" for n in range(self.cases))
        repro = Repro(
            id=f"{attack.id}-r",
            attack_id=attack.id,
            claim_id=attack.claim_id,
            lens=attack.lens,
            kind=CatchKind.REGRESSION,
            files={path: body},
            command=("python", "-m", "pytest", path),
        )
        return AttackOutcome(attack_id=attack.id, repros=(repro,))


def test_a_repro_of_several_cases_is_judged_one_finding_per_case_with_stable_ids() -> None:
    def run_once(cases: int) -> dict[str, Finding]:
        w = world(attackers={Lens.REGRESSION: [SeveralCases(cases)]})
        target = tier_a("1")
        run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
        prefix = f"{target.id}/fixed_x/regression-1."
        return {
            e.finding.id.removeprefix(prefix): e.finding
            for e in w.ledgers["spark"].items
            if e.finding.id.startswith(prefix)
        }

    two = run_once(2)
    assert {
        k: (f.status, f.repro.id if f.repro else None, f.repro.cases if f.repro else None) for k, f in two.items()
    } == {
        "1": (Status.BLOCKER, "regression-1-r.c1", ("tests/test_receipts_several.py::test_0",)),
        "2": (Status.BLOCKER, "regression-1-r.c2", ("tests/test_receipts_several.py::test_1",)),
    }
    assert [f.id for f in run_once(2).values()] == [f.id for f in two.values()]  # the same ids on every run
    five = run_once(5)
    assert [five[k].status for k in ("1", "2", "3")] == [Status.BLOCKER] * 3
    over = five["4"]
    assert (over.status, over.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.BUDGET)
    assert "over the cap of 3 test cases judged per repro" in (over.detail or "")
    assert "test_receipts_several.py::test_3" in (over.detail or "") and "test_4" in (over.detail or "")


# ---------------------------------------------------------------- SK3's surrogates


class Surrogates:
    """A surrogate builder that makes one plausible surrogate per claim, at a cost the fakes' Costs issue."""

    def __init__(self, costs: Costs, *, micro_usd: int | None = None) -> None:
        self.costs = costs
        self.micro_usd = micro_usd
        self.calls: list[tuple[str, str]] = []
        self.ks: list[int] = []

    def build(self, target: Target, claim: Any, *, sandbox: Any, runner: Any, k: int) -> tuple[Surrogate, ...]:
        self.calls.append((target.id, claim.id))
        self.ks.append(k)
        cost = self.costs.spend("surrogate", SPARK)
        if self.micro_usd is not None:
            self.costs.issued[-1] = cost = cost.model_copy(update={"micro_usd": self.micro_usd})
        return (
            Surrogate(id=f"{claim.id}.s1", claim_id=claim.id, patch=PATCH, plausible=True, why="ok", costs=(cost,)),
        )


def test_the_seams_surrogates_turn_fix_validity_catches_into_blockers_built_once_per_claim_and_target() -> None:
    builder = Surrogates(Costs())
    w = world(surrogates=builder)
    builder.costs = w.costs  # the world's own, so every cost is checked once
    targets = [tier_a("1"), tier_a("2")]
    summaries = [
        run_target(t, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers) for t in targets
    ]
    expected = {**EXPECTED, "fixed_x/correctness-1.2": Status.BLOCKER, "fixed_x/correctness-2.1": Status.BLOCKER}
    for arm in ARMS:
        for t in targets:
            assert findings_of(w.ledgers[arm.name], t) == expected, (arm.name, t.id)
    # Two candidates of one claim, on two arms: one build per target.
    assert builder.calls == [(t.id, "fixed_x") for t in targets]
    assert_every_cost_once(w.costs, w.ledgers.values(), summaries)


def test_surrogate_spend_stays_out_of_the_arm_s_cap_and_on_the_finding_that_built_them() -> None:
    builder = Surrogates(Costs(), micro_usd=2_000_000)  # over the $1.00 arm cap alone, under the surrogate budget
    w = world(surrogates=builder)
    builder.costs = w.costs
    summary = run_target(
        tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers
    )
    items = [e.finding for e in w.ledgers["spark"].items]
    assert not [f for f in items if f.unchecked_reason is UncheckedReason.BUDGET]  # no attack left out for it
    carried = [f for f in items if any(c.stage == "surrogate" for c in f.costs)]
    assert [sum(c.micro_usd for c in f.costs if c.stage == "surrogate") for f in carried] == [2_000_000]
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])  # a rollup of the ledger counts it


def _over_cap_world() -> tuple[World, Surrogates, Seams]:
    # The plan costs 101, under a cap of 102, so each arm's first attack runs; its own cost puts the arm over its cap
    # before its candidate is judged.
    builder = Surrogates(Costs())
    w = world(surrogates=builder, target_caps={Tier.A: 102}, stage1_script={"correctness-1": ["blocker"]})
    builder.costs = w.costs
    return w, builder, replace(w.seams, attackers={Lens.CORRECTNESS: [w.stage1, w.stage2]})


def test_an_arm_over_its_own_cap_still_builds_all_3_surrogates() -> None:
    w, builder, seams = _over_cap_world()
    run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=seams, ledgers=w.ledgers)
    first = next(e.finding for e in w.ledgers["spark"].items if e.finding.id.endswith("/fixed_x/correctness-1.1"))
    assert builder.ks == [3]
    assert first.status is Status.BLOCKER, first.detail
    later = [e.finding for e in w.ledgers["spark"].items if e.finding.unchecked_reason is UncheckedReason.BUDGET]
    assert later  # the arm is over its cap: its later attacks are left out, its surrogates are not


def test_both_arms_judge_a_candidate_by_the_same_surrogates_even_with_arm_1_over_its_cap() -> None:
    w, builder, seams = _over_cap_world()
    run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS, seams=seams, ledgers=w.ledgers)
    found = {
        arm.name: next(
            e.finding for e in w.ledgers[arm.name].items if e.finding.id.endswith("/fixed_x/correctness-1.1")
        )
        for arm in ARMS
    }
    spark, glimmer = found["spark"], found["glimmer"]
    assert spark.status is glimmer.status is Status.BLOCKER
    assert spark.detail == glimmer.detail
    surrogate_runs = [
        [(r.side, r.outcome, r.seed) for r in f.runs if r.side is Side.SURROGATE] for f in (spark, glimmer)
    ]
    assert surrogate_runs[0] == surrogate_runs[1] and len(surrogate_runs[0]) == 5
    assert builder.ks == [3]  # built once, for both arms


# ---------------------------------------------------------------- SK2: the refuter, and top-n per claim


class RefuterModel:
    """SK2's model: answers every refuter call with `category` (refuted unless it is "none"), at a cost the fakes'
    Costs issue, or raises `fail`."""

    def __init__(self, costs: Costs, *, category: str = "test_bug", fail: BaseException | None = None) -> None:
        self.costs = costs
        self.category = category
        self.fail = fail
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        if self.fail is not None:
            raise self.fail
        text = json.dumps(
            {"evidence": "tests/test_x.py:1 asserts it", "category": self.category, "refuted": self.category != "none"}
        )
        return ModelResponse(
            model=request.model,
            text=text,
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=10),
            cost=self.costs.spend(request.stage, request.model),
        )


def refuting_world(category: str = "test_bug", **changes: Any) -> tuple[World, RefuterModel]:
    model = RefuterModel(Costs(), category=category)
    w = world(refuter=Refuter(model), **changes)
    model.costs = w.costs  # the world's own, so every cost is checked once
    return w, model


def one_entry_per_finding(ledger: MemoryLedger) -> None:
    ids = [e.finding.id for e in ledger.items]
    assert len(ids) == len(set(ids))  # each finding recorded once, in its final state


def test_a_blocker_the_refuter_refutes_is_recorded_once_as_suspected_and_its_cost_counted_once() -> None:
    w, model = refuting_world()
    target = tier_a("1")
    summary = run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=w.seams, ledgers=w.ledgers)
    caught = f"{target.id}/nothing_else_changed/regression-1.1"
    for arm in ARMS:
        assert findings_of(w.ledgers[arm.name], target) == {
            **EXPECTED,
            "nothing_else_changed/regression-1.1": Status.SUSPECTED,
        }
        one_entry_per_finding(w.ledgers[arm.name])
        finding = next(e.finding for e in w.ledgers[arm.name].items if e.finding.id == caught)
        assert finding.refutation == "test_bug: tests/test_x.py:1 asserts it"
        assert [c.stage for c in finding.costs].count(REFUTER_STAGE) == 1
    assert len(model.requests) == 2  # one blocker per arm, one call each
    assert all(a.blockers == () for a in summary.arms)
    assert summary.refuter_model == REFUTER_MODEL
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])


def test_a_blocker_the_refuter_does_not_refute_stays_a_blocker_with_no_refutation() -> None:
    w, model = refuting_world("none")
    target = tier_a("1")
    summary = run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    assert findings_of(w.ledgers["spark"], target) == EXPECTED
    caught = next(e.finding for e in w.ledgers["spark"].items if e.finding.status is Status.BLOCKER)
    assert caught.refutation is None
    assert (caught.detail or "").startswith(NO_REFUTATION + "tests/test_x.py:1 asserts it")
    assert summary.arms[0].blockers == (caught.id,)
    assert len(model.requests) == 1
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])


def test_only_blockers_reach_the_refuter() -> None:
    builder = Surrogates(Costs())
    w, model = refuting_world("none", surrogates=builder)
    builder.costs = w.costs
    target = tier_a("1")
    summary = run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    blockers = [e.finding for e in w.ledgers["spark"].items if e.finding.status is Status.BLOCKER]
    assert len(blockers) == 3 and len(model.requests) == 3
    asked = [r.messages[1].content for r in model.requests]
    for finding, content in zip(blockers, asked, strict=True):
        assert finding.repro is not None and next(iter(finding.repro.files)) in content
    others = [e.finding for e in w.ledgers["spark"].items if e.finding.status is not Status.BLOCKER]
    assert others and all(REFUTER_STAGE not in {c.stage for c in f.costs} for f in others)
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])


def test_a_failed_refuter_call_leaves_the_blocker_suspected_and_keeps_its_cost_once() -> None:
    w, model = refuting_world()
    model.fail = ModelCallError("down", UncheckedReason.RUNTIME, cost=w.costs.spend(REFUTER_STAGE, REFUTER_MODEL))
    target = tier_a("1")
    summary = run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    caught = next(e.finding for e in w.ledgers["spark"].items if e.finding.id.endswith("/regression-1.1"))
    assert (caught.status, caught.refutation) == (Status.SUSPECTED, None)
    assert (caught.detail or "").startswith("SK2: no refutation could be read (the muse-wonder-1.0 call failed")
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])


def test_a_refuter_that_raises_leaves_the_blocker_suspected_and_the_run_goes_on() -> None:
    w, model = refuting_world()
    model.fail = RuntimeError("a bug in the client")
    target = tier_a("1")
    run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=w.seams, ledgers=w.ledgers)
    assert findings_of(w.ledgers["spark"], target) == {
        **EXPECTED,
        "nothing_else_changed/regression-1.1": Status.SUSPECTED,
    }
    caught = next(e.finding for e in w.ledgers["spark"].items if e.finding.id.endswith("/regression-1.1"))
    assert (caught.detail or "").startswith("SK2: no refutation could be read (the refuter raised RuntimeError)")


def test_the_refuter_runs_on_a_blocker_even_with_the_arm_over_its_cap() -> None:
    w, _, seams = _over_cap_world()
    model = RefuterModel(w.costs)
    seams = replace(seams, refuter=Refuter(model))
    summary = run_target(tier_a("1"), temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=seams, ledgers=w.ledgers)
    first = next(e.finding for e in w.ledgers["spark"].items if e.finding.id.endswith("/fixed_x/correctness-1.1"))
    assert first.status is Status.SUSPECTED and (first.refutation or "").startswith("test_bug: ")
    assert len(model.requests) == 1
    assert [e.finding for e in w.ledgers["spark"].items if e.finding.unchecked_reason is UncheckedReason.BUDGET]
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])


def test_with_no_refuter_blockers_stay_and_the_summary_and_run_line_say_no_refuter_ran(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    for refuted, w in ((False, world()), (True, refuting_world()[0])):
        wiring = RunWiring(seams=lambda arms, w=w: w.seams, ledger=FileLedger)
        monkeypatch.setattr(cli, "LANE_COMMANDS", (lambda sub, wiring=wiring: register(sub, wiring=wiring),))
        out_dir = tmp_path / str(refuted)
        out_dir.mkdir()
        run_file, arms_file = write_run_files(out_dir, [tier_a("1")])
        argv = ["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(out_dir / "l")]
        assert cli.main(argv) == 0
        out = capsys.readouterr().out
        summary = read_summary(summary_path(out_dir / "l" / "targets", tier_a("1").id))
        assert (summary.refuter_model is not None) is refuted
        assert ("no refuter ran (SK2)" in out) is not refuted
        if not refuted:
            assert all(len(a.blockers) == 1 for a in summary.arms)  # as SK1 left them


def test_top_n_reruns_5_first_run_failures_per_claim_across_its_attacks_and_passes_take_no_slot() -> None:
    w = world(
        regression_script={},
        stage1_script={
            "correctness-1": ["passes", "blocker", "blocker", "blocker"],
            "correctness-2": ["blocker", "passes", "blocker", "blocker", "blocker"],
            "correctness-3": [],
        },
    )
    seams = replace(w.seams, attackers={Lens.REGRESSION: [w.regression], Lens.CORRECTNESS: [w.stage1]})
    target = tier_a("1")
    summary = run_target(target, temperatures=TemperatureMode.SPEC, arms=ARMS, seams=seams, ledgers=w.ledgers)
    for arm in ARMS:  # each arm has its own room
        found = {
            e.finding.id.removeprefix(f"{target.id}/fixed_x/"): e.finding
            for e in w.ledgers[arm.name].items
            if e.finding.claim_id == "fixed_x"
        }
        assert {k: f.status for k, f in found.items()} == {
            "correctness-1.1": Status.NOT_REPRODUCED,  # passed its first run: its screen status, no slot
            "correctness-1.2": Status.S_SAT_INCONCLUSIVE,
            "correctness-1.3": Status.S_SAT_INCONCLUSIVE,
            "correctness-1.4": Status.S_SAT_INCONCLUSIVE,
            # 3 slots taken by the first attack: 2 left, for the first two first-run failures of this one.
            "correctness-2.1": Status.S_SAT_INCONCLUSIVE,
            "correctness-2.2": Status.NOT_REPRODUCED,  # past the slots' order, still its screen status
            "correctness-2.3": Status.S_SAT_INCONCLUSIVE,
            "correctness-2.4": Status.UNCHECKED,
            "correctness-2.5": Status.UNCHECKED,
            "correctness-3": Status.NOT_REPRODUCED,
        }, arm.name
        for over in (found["correctness-2.4"], found["correctness-2.5"]):
            assert over.unchecked_reason is UncheckedReason.BUDGET
            assert over.detail == "over the top-n of 5 candidates judged per claim"
            # What the screen found stays on it: the repro, its failing first patch run, the screen's own seconds.
            assert over.repro is not None and over.kind is CatchKind.FIX_VALIDITY and over.quote
            assert [(r.side, r.seed, r.outcome) for r in over.runs] == [(Side.PATCH, 0, Outcome.FAIL)]
            assert [c.stage for c in over.costs if c.stage == SKEPTIC_STAGE] == [SKEPTIC_STAGE]
        assert found["correctness-2.1"].costs  # the attack's costs, on its first finding
    # Only the 5 judged first-run failures were rerun: 4 more patch runs and 5 base runs each, per arm.
    reruns = [run for run in w.sandbox.runs if run[2] > 0 or run[1] is Side.BASE]
    assert len(reruns) == 2 * 5 * (4 + 5)
    assert TOP_N_PER_CLAIM == 5
    assert_every_cost_once(w.costs, w.ledgers.values(), [summary])


@dataclass
class DivergingAttacker:
    """AT4's shape on the regression lens: regression-1 gives a blocker candidate and a suspected divergence,
    regression-2 a suspected divergence only, and regression-3 nothing."""

    costs: Costs
    lens: Lens = Lens.REGRESSION

    def attack(
        self, target: Target, attack: Attack, *, arm: Arm, client: Any, sandbox: Any, runner: Any
    ) -> AttackOutcome:
        if attack.id == "regression-3":
            return AttackOutcome(attack_id=attack.id)
        divergence = Divergence(
            attack_id=attack.id,
            example="from widgets.render import render\nprint(render(['a', 'b']))\n",
            base_output="ab\n",
            patch_output="a,b\n",
            reason="the gate found the change intended: the issue asks for commas",
        )
        repros = (repro(attack, "blocker", 1),) if attack.id == "regression-1" else ()
        cost = self.costs.spend("at4_gate", arm.attacker_model)
        return AttackOutcome(attack_id=attack.id, repros=repros, costs=(cost,), suspected=(divergence,))


def test_a_suspected_divergence_reaches_the_ledger_as_suspected_and_is_never_a_blocker_or_a_catch() -> None:
    w = world()
    seams = replace(w.seams, attackers={Lens.REGRESSION: [DivergingAttacker(w.costs)], Lens.CORRECTNESS: [w.stage1]})
    t = tier_a("1")
    summary = run_target(t, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=seams, ledgers=w.ledgers)
    found = findings_of(w.ledgers["spark"], t)
    assert found["nothing_else_changed/regression-1.1"] is Status.BLOCKER  # the candidate, judged as ever
    assert found["nothing_else_changed/regression-1.2"] is Status.SUSPECTED  # numbered after the candidates
    assert found["nothing_else_changed/regression-2.1"] is Status.SUSPECTED
    assert found["nothing_else_changed/regression-3"] is Status.NOT_REPRODUCED
    suspected = [e.finding for e in w.ledgers["spark"].items if e.finding.status is Status.SUSPECTED]
    for finding in suspected:
        assert finding.repro is None and finding.kind is None and finding.refutation is None
        assert finding.detail is not None and "the issue asks for commas" in finding.detail
        assert "base printed 'ab\\n'" in finding.detail and "patch printed 'a,b\\n'" in finding.detail
    # The attack's cost is on its first finding once: regression-2's only finding is the suspected one.
    regression_two = next(f for f in suspected if f.id.endswith("regression-2.1"))
    assert [c.stage for c in regression_two.costs] == ["at4_gate"]
    assert summary.arms[0].blockers == (f"{t.id}/nothing_else_changed/regression-1.1",)
    assert summary.arms[0].statuses[Status.SUSPECTED] == 2

    # The score counts no suspected divergence: with the blocker's replay passing on gold, one catch, from it alone.
    from receipts.eval.labels import Label
    from receipts.eval.score import GoldReplay, score_arm

    replayed: list[str] = []

    def replay(target: Target, r: Repro) -> GoldReplay:
        replayed.append(r.id)
        return GoldReplay(outcome=Outcome.PASS)

    score = score_arm(ARMS[0], w.ledgers["spark"].items, {t.id: t}, {t.id: Label.BAD}, replay)
    assert (score.blockers, score.catches, score.catch_findings) == (1, 1, 1)
    assert replayed == ["regression-1-r1"] and score.statuses["suspected"] == 2
    only = [e for e in w.ledgers["spark"].items if e.finding.status is not Status.BLOCKER]
    none = score_arm(ARMS[0], only, {t.id: t}, {t.id: Label.BAD}, replay)
    assert (none.blockers, none.catches, none.catch_findings, none.refutations_by_category) == (0, 0, 0, {})


def test_the_differential_lens_is_planned_for_tier_a_only_even_when_wired() -> None:
    w = world()
    at4 = FakeAttacker(Lens.DIFFERENTIAL, "at4", w.costs, {})
    added = (Lens.REGRESSION, Lens.CORRECTNESS, Lens.DIFFERENTIAL)
    chains = {Lens.REGRESSION: [w.regression], Lens.CORRECTNESS: [w.stage1], Lens.DIFFERENTIAL: [at4]}
    seams = replace(w.seams, attackers=chains, tier_a_lenses=added)
    for t in (tier_a("a"), tier_a("b", tier=Tier.B, body="All tests pass.")):
        run_target(t, temperatures=TemperatureMode.SPEC, arms=ARMS[:1], seams=seams, ledgers=w.ledgers)
    lens_lists = [r.messages[0].content for r in w.models.requests if r.stage == PLAN_STAGE]
    assert len(lens_lists) == 2
    assert "- differential:" in lens_lists[0]  # Tier A, with --lenses adding it
    assert "- differential:" not in lens_lists[1]  # Tier B: AT4 is gated on Tier A (R8)
