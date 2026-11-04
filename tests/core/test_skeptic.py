"""SK1 against a scripted sandbox: the gates in order, 5 fresh runs a side with their seeds and the repro's own cases,
and every status the spec names, each fed the input that must produce it."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from receipts.contracts import (
    Attack,
    CaseResult,
    CatchKind,
    Claim,
    ClaimType,
    ContextMask,
    FailureType,
    Lens,
    Outcome,
    PatchMethod,
    Repro,
    RunnerKind,
    RunResult,
    Side,
    Span,
    SpanSource,
    StageCost,
    Status,
    Target,
    Tier,
    UncheckedReason,
    Workspace,
)
from receipts.core.skeptic import SKEPTIC_STAGE, RerunPolicy, TargetSkeptic

DIGEST = "sha256:" + "d" * 64
PATCH = """\
diff --git a/src/widgets/render.py b/src/widgets/render.py
--- a/src/widgets/render.py
+++ b/src/widgets/render.py
@@ -1,2 +1,2 @@ def render(items):
-    return "".join(items)
+    return ",".join(items)
"""
TEST = "from widgets.render import render\n\n\ndef test_joins():\n    assert render(['a', 'b']) == 'ab'\n"
QUOTE = "render crashes on an empty list"  # the target's issue, which a fix-validity repro quotes


def target(**changes: object) -> Target:
    fields: dict[str, object] = {
        "id": "verified~octo__widgets-1",
        "tier": Tier.A,
        "repo": "octo/widgets",
        "base_sha": "a" * 40,
        "patch": PATCH,
        "issue": "render crashes on an empty list",
    }
    fields.update(changes)
    return Target.model_validate(fields)


CLAIM = Claim(
    id="nothing_else_changed",
    type=ClaimType.NOTHING_ELSE_CHANGED,
    text="nothing else changed",
    span=Span(source=SpanSource.TEMPLATE, text="The change alters no behavior beyond what fixing the issue requires."),
)
ATTACK = Attack(
    id="regression-1",
    claim_id=CLAIM.id,
    lens=Lens.REGRESSION,
    hypothesis="joining changed",
    mask=ContextMask.DIFF,
    temperature=0.0,
)


def repro(kind: CatchKind = CatchKind.REGRESSION, **changes: Any) -> Repro:
    fields: dict[str, Any] = {
        "id": "r1",
        "attack_id": ATTACK.id,
        "claim_id": CLAIM.id,
        "lens": Lens.REGRESSION,
        "kind": kind,
        "files": {"tests/test_receipts_repro.py": TEST},
        "command": ("python", "-m", "pytest"),
    }
    fields.update(changes)
    return Repro.model_validate(fields)


def result(side: Side, seed: int, outcome: Outcome, failure: FailureType | None = None, **extra: Any) -> RunResult:
    fields: dict[str, Any] = {
        "side": side,
        "outcome": outcome,
        "failure_type": failure,
        "seed": seed,
        "fresh_container": True,
        "duration_s": 2.0,
        "image_digest": DIGEST,
    }
    fields.update(extra)
    return RunResult.model_validate(fields)


Script = Callable[[Side, int], RunResult | Exception]


def regression_bug(side: Side, seed: int) -> RunResult:
    """Fails on the patch on an assertion every time, passes on base every time: a regression catch."""
    if side is Side.PATCH:
        return result(side, seed, Outcome.FAIL, FailureType.ASSERTION)
    return result(side, seed, Outcome.PASS)


class Runner:
    kind = RunnerKind.PYTEST

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["python", "-m", "pytest", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return ()


@dataclass
class FakeSandbox:
    script: Script
    prepare_errors: Mapping[Side, Exception] = field(default_factory=dict)
    calls: list[dict[str, Any]] = field(default_factory=list)
    prepared: list[Side] = field(default_factory=list)
    attempts: list[Side] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        self.attempts.append(side)
        if side in self.prepare_errors:
            raise self.prepare_errors[side]
        self.prepared.append(side)
        return Workspace(id=f"ws-{side.value}", side=side, root="/testbed", image_digest=DIGEST)

    def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
        raise AssertionError("SK1 runs tests through run_test only")

    def run_test(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: Any,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
        expect: Sequence[str] | None = None,
    ) -> RunResult:
        assert seed is not None
        self.calls.append(
            {
                "side": workspace.side,
                "tests": tuple(tests),
                "seed": seed,
                "fresh": fresh,
                "expect": tuple(expect or ()),
                "files": dict(files or {}),
                "runner": runner.kind,
                "changed": tuple(changed_files),
                "timeout": timeout_s,
            }
        )
        outcome = self.script(workspace.side, seed)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome

    def close(self, workspace: Workspace) -> None:
        self.closed.append(workspace.id)


def check(script: Script, candidate: Repro | None = None, **sandbox: Any) -> tuple[Any, FakeSandbox]:
    box = FakeSandbox(script, **sandbox)
    skeptic = TargetSkeptic(target(), sandbox=box, runner=Runner())
    screened = skeptic.screen(candidate or repro(), ATTACK)
    cost = StageCost(stage="attack", model="muse-spark-1.3", micro_usd=99)
    finding = skeptic.judge(
        screened, claim=CLAIM, finding_id=f"{target().id}/{CLAIM.id}/{ATTACK.id}.1", costs=(cost,), wall=lambda: 12.5
    )
    skeptic.close()
    return finding, box


def test_a_regression_that_fails_5_of_5_on_the_patch_and_passes_5_of_5_on_base_is_a_blocker() -> None:
    finding, box = check(regression_bug)
    assert finding.status is Status.BLOCKER
    assert (finding.kind, finding.repro, finding.quote) == (CatchKind.REGRESSION, repro(), CLAIM.span.text)
    assert [(c["side"], c["seed"]) for c in box.calls] == [(Side.PATCH, s) for s in range(5)] + [
        (Side.BASE, s) for s in range(5)
    ]
    for call in box.calls:  # every run fresh, through the target's runner, expecting the repro's own case
        assert call["fresh"] is True
        assert call["tests"] == ("tests/test_receipts_repro.py",)
        assert call["expect"] == ("tests/test_receipts_repro.py::test_joins",)
        assert call["files"] == repro().files
        assert (call["runner"], call["changed"]) == (RunnerKind.PYTEST, ("src/widgets/render.py",))
    assert len(finding.runs) == 10
    assert finding.replay is not None and finding.replay.image_digest == DIGEST and finding.replay.seed == 0
    assert finding.wall_seconds == 12.5
    attack_cost, skeptic_cost = finding.costs
    assert attack_cost.micro_usd == 99  # passed through once
    assert (skeptic_cost.stage, skeptic_cost.model, skeptic_cost.micro_usd) == (SKEPTIC_STAGE, None, 0)
    assert sorted(box.prepared) == [Side.BASE, Side.PATCH]  # never gold: Tier A's gold is for scoring only
    assert sorted(box.closed) == ["ws-base", "ws-patch"]


def test_a_crash_in_changed_code_counts_like_an_assertion() -> None:
    def crash(side: Side, seed: int) -> RunResult:
        if side is Side.PATCH:
            return result(side, seed, Outcome.FAIL, FailureType.CRASH_IN_CHANGED)
        return result(side, seed, Outcome.PASS)

    assert check(crash)[0].status is Status.BLOCKER


@pytest.mark.parametrize(
    "failure",
    [
        FailureType.IMPORT,
        FailureType.NAME,
        FailureType.FIXTURE,
        FailureType.TIMEOUT,
        FailureType.INFRA,
        FailureType.OTHER,
        FailureType.CRASH_ELSEWHERE,
    ],
)
def test_a_failure_that_does_not_count_is_refuted_after_one_run(failure: FailureType) -> None:
    finding, box = check(lambda side, seed: result(side, seed, Outcome.FAIL, failure))
    assert finding.status is Status.REFUTED
    assert "failure-type gate" in (finding.detail or "") and failure.value in (finding.detail or "")
    assert len(box.calls) == 1 and len(finding.runs) == 1


@pytest.mark.parametrize("outcome", [Outcome.PASS, Outcome.SKIPPED])
def test_a_candidate_that_passes_or_is_skipped_on_the_patch_is_not_reproduced(outcome: Outcome) -> None:
    finding, box = check(lambda side, seed: result(side, seed, outcome))
    assert finding.status is Status.NOT_REPRODUCED
    assert len(box.calls) == 1


def test_a_patch_failure_in_only_some_fresh_runs_is_flaky() -> None:
    def sometimes(side: Side, seed: int) -> RunResult:
        if side is Side.PATCH and seed in (0, 1, 3):
            return result(side, seed, Outcome.FAIL, FailureType.ASSERTION)
        return result(side, seed, Outcome.PASS)

    finding, box = check(sometimes)
    assert finding.status is Status.FLAKY
    assert "3 of 5" in (finding.detail or "")
    assert [c["side"] for c in box.calls] == [Side.PATCH] * 5  # no base run for a candidate already flaky


def test_mixed_failure_types_on_the_patch_are_flaky() -> None:
    def mixed(side: Side, seed: int) -> RunResult:
        return result(side, seed, Outcome.FAIL, FailureType.ASSERTION if seed != 2 else FailureType.IMPORT)

    assert check(mixed)[0].status is Status.FLAKY


def test_a_regression_candidate_that_fails_on_base_too_is_refuted_and_mixed_base_is_flaky() -> None:
    finding, _ = check(lambda side, seed: result(side, seed, Outcome.FAIL, FailureType.ASSERTION))
    assert finding.status is Status.REFUTED
    assert "fails on base too" in (finding.detail or "")

    def base_sometimes(side: Side, seed: int) -> RunResult:
        if side is Side.PATCH or seed == 4:
            return result(side, seed, Outcome.FAIL, FailureType.ASSERTION)
        return result(side, seed, Outcome.PASS)

    flaky, _ = check(base_sometimes)
    assert flaky.status is Status.FLAKY
    assert "4 of 5" in (flaky.detail or "")


@pytest.mark.parametrize("kind", [CatchKind.FIX_VALIDITY, CatchKind.BEHAVIOR])
def test_a_fix_validity_catch_without_a_surrogate_is_inconclusive_never_a_blocker(kind: CatchKind) -> None:
    finding, box = check(regression_bug, repro(kind, quote=QUOTE))
    assert finding.status is Status.S_SAT_INCONCLUSIVE
    assert len(box.calls) == 10
    assert Side.GOLD not in box.prepared and Side.SURROGATE not in box.prepared


def test_a_candidate_with_no_workspace_is_unchecked_no_env_and_the_build_is_not_retried() -> None:
    box = FakeSandbox(regression_bug, prepare_errors={Side.PATCH: RuntimeError("image pull failed")})
    skeptic = TargetSkeptic(target(), sandbox=box, runner=Runner())
    for number in (1, 2):
        screened = skeptic.screen(repro(), ATTACK)
        finding = skeptic.judge(
            screened, claim=CLAIM, finding_id=f"{target().id}/{CLAIM.id}/x.{number}", wall=lambda: 1.0
        )
        assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.NO_ENV)
        assert finding.replay is None  # no workspace, so nothing to replay
    assert box.calls == []
    assert box.attempts == [Side.PATCH]  # one failed build, never retried for the next candidate

    base_down = FakeSandbox(regression_bug, prepare_errors={Side.BASE: RuntimeError("no base image")})
    finding, _ = check(regression_bug, **{"prepare_errors": base_down.prepare_errors})
    assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.NO_ENV)


@pytest.mark.parametrize(
    ("bad", "reason"),
    [
        (RuntimeError("docker died"), UncheckedReason.RUNTIME),
        ("error", UncheckedReason.RUNTIME),
        ("timeout", UncheckedReason.TIMEOUT),
        ("stale", UncheckedReason.RUNTIME),
        ("wrong-side", UncheckedReason.RUNTIME),
    ],
    ids=["raises", "error", "timeout", "not-fresh", "wrong-side"],
)
def test_a_run_with_no_readable_result_leaves_the_candidate_unchecked(bad: object, reason: UncheckedReason) -> None:
    def script(side: Side, seed: int) -> RunResult | Exception:
        if side is Side.PATCH and seed == 2:  # the third patch run goes wrong
            if isinstance(bad, Exception):
                return bad
            if bad == "error":
                return result(side, seed, Outcome.ERROR)
            if bad == "timeout":
                return result(side, seed, Outcome.TIMEOUT)
            if bad == "stale":
                return result(side, seed, Outcome.FAIL, FailureType.ASSERTION, fresh_container=False)
            return result(Side.BASE, seed, Outcome.FAIL, FailureType.ASSERTION)
        return regression_bug(side, seed)

    finding, box = check(script)
    assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, reason)
    assert len(box.calls) == 3


def test_the_static_gates_refute_before_any_sandbox_work() -> None:
    patched = "from unittest.mock import patch\n\n@patch('widgets.render.render')\ndef test_x(m):\n    assert True\n"
    mocked = repro(files={"tests/test_receipts_repro.py": patched})
    no_case = repro(files={"tests/test_receipts_repro.py": "def helper():\n    pass\n"})
    other_claim = repro(claim_id="fixed_x")
    for candidate, words in ((mocked, "mocks_changed"), (no_case, "no test case"), (other_claim, "another claim")):
        finding, box = check(regression_bug, candidate)
        assert finding.status is Status.REFUTED
        assert words in (finding.detail or "")
        assert (box.calls, box.prepared) == ([], [])


def test_only_a_counted_first_patch_failure_passes_the_screen() -> None:
    box = FakeSandbox(lambda side, seed: result(side, seed, Outcome.FAIL, FailureType.ASSERTION))
    skeptic = TargetSkeptic(target(), sandbox=box, runner=Runner())
    assert skeptic.screen(repro(), ATTACK).failed_first_patch_run
    other = FakeSandbox(lambda side, seed: result(side, seed, Outcome.FAIL, FailureType.IMPORT))
    assert not TargetSkeptic(target(), sandbox=other, runner=Runner()).screen(repro(), ATTACK).failed_first_patch_run


def test_a_finding_the_contract_refuses_as_a_blocker_is_suspected(monkeypatch: pytest.MonkeyPatch) -> None:
    # A plant on the last guard: with no replay key the contract refuses a blocker, so the skeptic never makes one.
    monkeypatch.setattr(TargetSkeptic, "_replay", lambda self, screened: None)
    finding, _ = check(regression_bug)
    assert finding.status is Status.SUSPECTED
    assert "not a blocker" in (finding.detail or "")


def test_the_policy_never_runs_fewer_than_five_times() -> None:
    with pytest.raises(ValueError, match="at least 5 runs"):
        RerunPolicy(runs=4)
    with pytest.raises(ValueError, match="positive"):
        RerunPolicy(timeout_s=0)


def test_the_integrator_s_case_namer_sets_what_each_run_expects() -> None:
    box = FakeSandbox(regression_bug)

    def namer(candidate: Repro, kind: RunnerKind) -> tuple[str, ...]:
        return ("runner::own-id",)

    skeptic = TargetSkeptic(target(), sandbox=box, runner=Runner(), case_names=namer)
    screened = skeptic.screen(repro(), ATTACK)
    skeptic.judge(screened, claim=CLAIM, finding_id=f"{target().id}/{CLAIM.id}/x.1", wall=lambda: 1.0)
    assert {call["expect"] for call in box.calls} == {("runner::own-id",)}


@pytest.mark.parametrize(
    ("patch_sha", "expected"),
    [
        (
            "d27e2f320150e3b064f3256a8bada7478e89da53",
            "d27e2f320150e3b064f3256a8bada7478e89da53",
        ),
        (None, "a763bfdee62aaaabfed201657168e133373f7f1f4ae980b436ffc360ec6e2153"),
    ],
    ids=["commit-sha", "patch-sha256"],
)
def test_recorded_replay_patch_key_preserves_supplied_revision_or_patch_hash(
    patch_sha: str | None, expected: str
) -> None:
    box = FakeSandbox(regression_bug)
    skeptic = TargetSkeptic(target(patch_sha=patch_sha), sandbox=box, runner=Runner())
    try:
        finding = skeptic.judge(
            skeptic.screen(repro(), ATTACK), claim=CLAIM, finding_id=f"{target().id}/{CLAIM.id}/x.1", wall=lambda: 1.0
        )
        assert finding.status is Status.BLOCKER
        assert finding.replay is not None
        assert finding.replay.patch_sha == expected
    finally:
        skeptic.close()


def test_the_replay_key_s_harness_hash_covers_the_target_s_test_command() -> None:
    # The repository's own test command decides how a repro runs, so two commands never share a replay key.
    def harness(command: str) -> str:
        box = FakeSandbox(regression_bug)
        skeptic = TargetSkeptic(target(test_command=command), sandbox=box, runner=Runner())
        finding = skeptic.judge(
            skeptic.screen(repro(), ATTACK), claim=CLAIM, finding_id=f"{target().id}/{CLAIM.id}/x.1", wall=lambda: 1.0
        )
        assert finding.replay is not None
        return finding.replay.harness_hash

    assert harness("pytest -rA") == harness("pytest -rA")
    assert harness("pytest -rA") != harness("pytest -rA -p no:randomly")
    assert harness("") != harness("pytest -rA")


# ---------------------------------------------------------------- contract v3: detail, a run's problem, cases (round 4)


@pytest.mark.parametrize(
    ("script", "status", "words"),
    [
        (lambda side, seed: result(side, seed, Outcome.PASS), Status.NOT_REPRODUCED, "passes on the patch"),
        (lambda side, seed: result(side, seed, Outcome.FAIL, FailureType.ASSERTION), Status.REFUTED, "base too"),
        (lambda side, seed: result(side, seed, Outcome.ERROR), Status.UNCHECKED, "could not be read"),
        (
            lambda side, seed: result(side, seed, Outcome.FAIL, FailureType.ASSERTION if seed else FailureType.IMPORT),
            Status.REFUTED,
            "failure-type gate",
        ),
    ],
    ids=["not-reproduced", "refuted", "unchecked", "gate"],
)
def test_why_a_finding_has_its_status_is_its_detail_never_a_refutation(
    script: Script, status: Status, words: str
) -> None:
    finding, _ = check(script)
    assert finding.status is status
    assert words in (finding.detail or "") and finding.refutation is None


def test_a_blocker_has_neither_a_detail_nor_a_refutation() -> None:
    finding, _ = check(regression_bug)
    assert (finding.status, finding.detail, finding.refutation) == (Status.BLOCKER, None, None)


@pytest.mark.parametrize(
    ("outcome", "failure", "reason"),
    [
        (Outcome.ERROR, FailureType.INFRA, UncheckedReason.RUNTIME),
        (Outcome.ERROR, None, UncheckedReason.RUNTIME),
        (Outcome.TIMEOUT, FailureType.TIMEOUT, UncheckedReason.TIMEOUT),
    ],
    ids=["infra", "untyped", "timeout"],
)
def test_a_run_s_own_problem_is_carried_into_the_unchecked_finding_s_detail(
    outcome: Outcome, failure: FailureType | None, reason: UncheckedReason
) -> None:
    problem = "expected case tests/test_receipts_repro.py::test_joins never ran: the run wrote no report"
    finding, _ = check(lambda side, seed: result(side, seed, outcome, failure, problem=problem))
    assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, reason)
    assert problem in (finding.detail or "")


@pytest.mark.parametrize(
    "failure",
    [FailureType.IMPORT, FailureType.NAME, FailureType.FIXTURE, FailureType.CRASH_ELSEWHERE, FailureType.OTHER],
)
def test_an_error_the_runner_typed_is_a_failure_that_never_counts_not_an_unread_run(failure: FailureType) -> None:
    # The attack lane reports every failure that does not count as ERROR with its type, and v3 types a case that
    # never ran because its module failed to import as IMPORT: read and typed, so the gate refutes it.
    problem = "expected case tests/test_receipts_repro.py::test_joins never ran: its module raised ImportError"
    finding, box = check(lambda side, seed: result(side, seed, Outcome.ERROR, failure, problem=problem))
    assert finding.status is Status.REFUTED
    assert f"failure-type gate: fails on the patch with {failure.value}" in (finding.detail or "")
    assert problem in (finding.detail or "") and len(box.calls) == 1

    def import_error_on_base(side: Side, seed: int) -> RunResult:
        if side is Side.PATCH:
            return result(side, seed, Outcome.FAIL, FailureType.ASSERTION)
        return result(side, seed, Outcome.ERROR, failure)

    on_base, _ = check(import_error_on_base)
    assert on_base.status is Status.REFUTED and "fails on base too" in (on_base.detail or "")


def test_a_repro_s_cases_are_what_each_run_runs_and_expects() -> None:
    case = "tests/test_receipts_repro.py::test_joins"
    finding, box = check(regression_bug, repro(cases=(case,)))
    assert finding.status is Status.BLOCKER
    assert {(call["tests"], call["expect"]) for call in box.calls} == {((case,), (case,))}


def test_the_skeptic_keeps_every_workspace_it_prepared_with_how_its_patch_applied() -> None:
    class Applied(FakeSandbox):
        def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
            workspace = super().prepare(target, side, patch=patch)
            method = PatchMethod.GIT_APPLY_REJECT if side is Side.PATCH else None
            return workspace.model_copy(update={"patch_method": method})

    box = Applied(regression_bug)
    skeptic = TargetSkeptic(target(), sandbox=box, runner=Runner())
    skeptic.judge(
        skeptic.screen(repro(), ATTACK), claim=CLAIM, finding_id=f"{target().id}/{CLAIM.id}/x.1", wall=lambda: 1.0
    )
    skeptic.close()  # closing keeps the record
    assert [(w.side, w.patch_method) for w in skeptic.prepared] == [
        (Side.PATCH, PatchMethod.GIT_APPLY_REJECT),
        (Side.BASE, None),
    ]


# ---------------------------------------------------------------- fix round 5: the quote, base's failure, one case each

TWO_CASES = TEST + "\n\ndef test_empty():\n    assert render([]) == ''\n"


@pytest.mark.parametrize("kind", [CatchKind.FIX_VALIDITY, CatchKind.BEHAVIOR])
@pytest.mark.parametrize(
    ("quote", "why"),
    [
        ("", "quotes no sentence"),
        ("render crashes", "under 3 words"),
        ("render fails on an empty list", "not in the issue verbatim"),
        ("render crashes on a list", "not in the issue verbatim"),  # a sentence it does not say
        ("der crashes on an empty", "not in the issue verbatim"),  # 'render' cut mid-word: no word boundary
        ("render crashes on an empty li", "not in the issue verbatim"),
    ],
)
def test_a_fix_validity_repro_that_does_not_quote_the_issue_is_refuted_before_any_run(
    kind: CatchKind, quote: str, why: str
) -> None:
    finding, box = check(regression_bug, repro(kind, quote=quote))
    assert finding.status is Status.REFUTED and "SK2 quote check" in (finding.detail or "") and why in finding.detail
    assert box.calls == [] and box.prepared == []


def test_a_quote_stands_when_only_its_spacing_differs_and_a_regression_repro_quotes_nothing() -> None:
    spaced = "render  crashes\n on an\tempty list"
    assert check(regression_bug, repro(CatchKind.FIX_VALIDITY, quote=spaced))[0].status is Status.S_SAT_INCONCLUSIVE
    agent = target(issue="", body="Fixes it: the widget\nnow renders an empty list.")
    box = FakeSandbox(regression_bug)
    skeptic = TargetSkeptic(agent, sandbox=box, runner=Runner())
    screened = skeptic.screen(repro(CatchKind.BEHAVIOR, quote="the widget now renders an empty list"), ATTACK)
    assert screened.stop is None  # the agent's own text holds it: on to the reruns
    assert check(regression_bug, repro(quote=""))[0].status is Status.BLOCKER


def patch_fails_base(outcome: Outcome, failure: FailureType | None) -> Script:
    def script(side: Side, seed: int) -> RunResult:
        if side is Side.PATCH:
            return result(side, seed, Outcome.FAIL, FailureType.ASSERTION)
        return result(side, seed, outcome, failure)

    return script


@pytest.mark.parametrize("outcome", [Outcome.ERROR, Outcome.FAIL])
def test_a_fix_validity_repro_with_an_import_error_on_base_is_refuted(outcome: Outcome) -> None:
    # Pilot, pylint-8898 Glimmer 1.1: it imported a function only the agent's patch adds, so base failed to import it.
    finding, box = check(patch_fails_base(outcome, FailureType.IMPORT), repro(CatchKind.FIX_VALIDITY, quote=QUOTE))
    assert finding.status is Status.REFUTED and len(box.calls) == 10
    assert "fails on base with an import error in 5 of 5 runs" in (finding.detail or "")
    # An adds_behavior repro fails on base however it may: the behavior is not there yet.
    behavior = repro(CatchKind.BEHAVIOR, quote=QUOTE)
    assert check(patch_fails_base(outcome, FailureType.IMPORT), behavior)[0].status is Status.S_SAT_INCONCLUSIVE


@pytest.mark.parametrize(
    ("outcome", "failure"),
    [
        (Outcome.ERROR, FailureType.CRASH_ELSEWHERE),  # a crash in a file the change did not touch
        (Outcome.ERROR, FailureType.NAME),
        (Outcome.ERROR, FailureType.FIXTURE),
        (Outcome.SKIPPED, None),
        (Outcome.FAIL, FailureType.ASSERTION),  # the bug, on base: what a fix-validity catch looks like
    ],
)
def test_only_an_import_error_on_base_refutes_a_fix_validity_repro(
    outcome: Outcome, failure: FailureType | None
) -> None:
    finding, _ = check(patch_fails_base(outcome, failure), repro(CatchKind.FIX_VALIDITY, quote=QUOTE))
    assert finding.status is Status.S_SAT_INCONCLUSIVE, finding.detail


def test_a_quote_comes_from_the_issue_when_there_is_one_and_else_from_the_title_or_body() -> None:
    def screened(quote: str, **changes: object) -> str | None:
        skeptic = TargetSkeptic(target(**changes), sandbox=FakeSandbox(regression_bug), runner=Runner())
        stop = skeptic.screen(repro(CatchKind.FIX_VALIDITY, quote=quote), ATTACK).stop
        return None if stop is None else stop.detail

    agent = {"session_summary": "I made render return an empty string.", "commit_messages": "Make render total."}
    for quote in ("render return an empty string", "Make render total"):
        why = screened(quote, **agent)  # the issue exists, so the agent's own words never stand for it
        assert why is not None and "not in the issue verbatim" in why
        why = screened(quote, issue="", **agent)  # nor without one: never commits or the session summary
        assert why is not None and "the pull request's title or body" in why
    assert screened("render crashes\non  an empty list") is None  # spacing aside, verbatim from the issue
    why = screened("the widget now renders it", body="Fixes it: the widget now renders it.")  # the issue rules
    assert why is not None and "not in the issue verbatim" in why
    pr = {"issue": "", "title": "Render an empty list", "body": "The widget now renders an empty list."}
    assert screened("widget now renders an empty", **pr) is None
    assert screened("Render an empty list", **pr) is None
    why = screened("render crashes on an empty list", issue="", title="t", body="b")
    assert why is not None and "the pull request's title or body" in why


def test_a_repro_of_several_cases_splits_into_one_repro_per_case_in_order() -> None:
    skeptic = TargetSkeptic(target(), sandbox=FakeSandbox(regression_bug), runner=Runner())
    several = repro(CatchKind.FIX_VALIDITY, quote=QUOTE, files={"tests/test_receipts_repro.py": TWO_CASES})
    parts, left = skeptic.split(several)
    assert left == ()
    assert [(p.id, p.cases) for p in parts] == [
        ("r1.c1", ("tests/test_receipts_repro.py::test_joins",)),
        ("r1.c2", ("tests/test_receipts_repro.py::test_empty",)),
    ]
    for part in parts:  # everything else is the repro's own
        assert part.model_dump(exclude={"id", "cases"}) == several.model_dump(exclude={"id", "cases"})
    assert skeptic.split(several) == (parts, ())  # the same every time
    one = repro()
    assert skeptic.split(one) == ((one,), ())  # one case: left as it is
    named = repro(cases=("a::x", "a::y", "a::z", "a::w", "a::v"))  # the repro's own cases, in its order
    parts, left = skeptic.split(named)
    assert [p.cases for p in parts] == [("a::x",), ("a::y",), ("a::z",)] and left == ("a::w", "a::v")


def test_each_case_of_a_split_repro_runs_alone() -> None:
    box = FakeSandbox(regression_bug)
    skeptic = TargetSkeptic(target(), sandbox=box, runner=Runner())
    parts, _ = skeptic.split(repro(files={"tests/test_receipts_repro.py": TWO_CASES}))
    for part in parts:
        skeptic.judge(
            skeptic.screen(part, ATTACK), claim=CLAIM, finding_id=f"{target().id}/{CLAIM.id}/y", wall=lambda: 1.0
        )
    assert {call["expect"] for call in box.calls} == {
        ("tests/test_receipts_repro.py::test_joins",),
        ("tests/test_receipts_repro.py::test_empty",),
    }
