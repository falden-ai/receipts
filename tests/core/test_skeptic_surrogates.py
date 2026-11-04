"""SK3's surrogate oracle in the skeptic, against a scripted sandbox and a scripted surrogate builder: every status a
fix-validity or behavior candidate can get from its claim's surrogates, each fed the input that must produce it, and
the build made once per claim, shared by its candidates and by every arm."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any

import pytest

from receipts.contracts import (
    SURROGATE_CAP_MICRO_USD,
    SURROGATE_TARGET_CAP_MICRO_USD,
    Attack,
    CaseResult,
    CatchKind,
    Claim,
    ClaimType,
    ContextMask,
    FailureType,
    Finding,
    Lens,
    Outcome,
    Repro,
    RunnerKind,
    RunResult,
    Sandbox,
    Side,
    Span,
    SpanSource,
    StageCost,
    Status,
    SuiteRunner,
    Surrogate,
    Target,
    Tier,
    UncheckedReason,
    Workspace,
)
from receipts.core.skeptic import TargetSkeptic

DIGEST = "sha256:" + "d" * 64
PATCH = """\
diff --git a/src/widgets/render.py b/src/widgets/render.py
--- a/src/widgets/render.py
+++ b/src/widgets/render.py
@@ -1,2 +1,2 @@ def render(items):
-    return "".join(items)
+    return ",".join(items)
"""
TEST = "from widgets.render import render\n\n\ndef test_empty():\n    assert render([]) == ''\n"
QUOTE = "render crashes on an empty list"
TARGET = Target(
    id="verified~octo__widgets-1",
    tier=Tier.A,
    repo="octo/widgets",
    base_sha="a" * 40,
    patch=PATCH,
    issue=QUOTE,
)
CLAIM = Claim(id="fixed_x", type=ClaimType.FIXED_X, text="fixed", span=Span(source=SpanSource.ISSUE, text=QUOTE))
ATTACK = Attack(
    id="correctness-1", claim_id=CLAIM.id, lens=Lens.CORRECTNESS, hypothesis="h", mask=ContextMask.DIFF, temperature=0
)


def repro(n: int = 1, kind: CatchKind = CatchKind.FIX_VALIDITY) -> Repro:
    return Repro(
        id=f"r{n}",
        attack_id=ATTACK.id,
        claim_id=CLAIM.id,
        lens=Lens.CORRECTNESS,
        kind=kind,
        files={f"tests/test_receipts_repro_{n}.py": TEST},
        command=("python", "-m", "pytest"),
        quote=QUOTE,
    )


def surrogate(n: int, *, plausible: bool = True, micro_usd: int = 0) -> Surrogate:
    return Surrogate(
        id=f"fixed_x.s{n}",
        claim_id=CLAIM.id,
        patch=f"diff --git a/src/widgets/render.py b/src/widgets/render.py\n# surrogate {n}\n",
        plausible=plausible,
        why="plausible" if plausible else f"surrogate {n} broke an existing test",
        costs=(StageCost(stage="surrogate", model="muse-spark-1.3", micro_usd=micro_usd),) if micro_usd else (),
    )


def result(side: Side, seed: int, outcome: Outcome, failure: FailureType | None = None) -> RunResult:
    return RunResult(side=side, outcome=outcome, failure_type=failure, seed=seed, fresh_container=True, duration_s=1.0)


# (side, seed, the workspace's surrogate number or 0) -> the run
Script = Callable[[Side, int, int], RunResult | Exception]


def fails_on_patch(on_surrogate: Callable[[int, int], Outcome]) -> Script:
    """Fails on the patch every run, passes on base every run, and on surrogate n's run `seed` as told."""

    def script(side: Side, seed: int, n: int) -> RunResult:
        if side is Side.PATCH:
            return result(side, seed, Outcome.FAIL, FailureType.ASSERTION)
        if side is Side.BASE:
            return result(side, seed, Outcome.PASS)
        outcome = on_surrogate(n, seed)
        return result(side, seed, outcome, FailureType.ASSERTION if outcome is Outcome.FAIL else None)

    return script


class Runner:
    kind = RunnerKind.PYTEST

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["python", "-m", "pytest", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return ()


@dataclass
class Box:
    script: Script
    prepare_error_on_surrogate: bool = False
    patches: dict[str, int] = field(default_factory=dict)  # workspace id -> surrogate number
    calls: list[tuple[Side, int, int]] = field(default_factory=list)
    prepared: list[tuple[Side, str | None]] = field(default_factory=list)
    closed: list[str] = field(default_factory=list)

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        if side is Side.SURROGATE and self.prepare_error_on_surrogate:
            raise RuntimeError("no surrogate image")
        self.prepared.append((side, patch))
        n = int(patch.rsplit("surrogate ", 1)[1]) if patch else 0
        wid = f"ws-{side.value}-{len(self.prepared)}"
        self.patches[wid] = n
        return Workspace(id=wid, side=side, root="/testbed", image_digest=DIGEST)

    def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
        raise AssertionError("the skeptic runs tests through run_test only")

    def run_test(self, workspace: Workspace, tests: Sequence[str], **options: Any) -> RunResult:
        seed: int = options["seed"]
        assert options["fresh"] is True
        n = self.patches[workspace.id]
        self.calls.append((workspace.side, seed, n))
        out = self.script(workspace.side, seed, n)
        if isinstance(out, Exception):
            raise out
        return out

    def close(self, workspace: Workspace) -> None:
        self.closed.append(workspace.id)


@dataclass
class Builder:
    made: Sequence[Surrogate]
    calls: list[tuple[str, str, int]] = field(default_factory=list)

    def build(
        self, target: Target, claim: Claim, *, sandbox: Sandbox, runner: SuiteRunner, k: int
    ) -> tuple[Surrogate, ...]:
        self.calls.append((target.id, claim.id, k))
        return tuple(self.made)


def judge(skeptic: TargetSkeptic, candidate: Repro | None = None, number: int = 1, claim: Claim = CLAIM) -> Finding:
    screened = skeptic.screen(candidate or repro(), ATTACK)
    return skeptic.judge(
        screened, claim=claim, finding_id=f"{TARGET.id}/{claim.id}/{ATTACK.id}.{number}", wall=lambda: 3.0
    )


def check(script: Script, builder: Builder | None, **box: Any) -> tuple[Finding, Box]:
    sandbox = Box(script, **box)
    skeptic = TargetSkeptic(TARGET, sandbox=sandbox, runner=Runner(), surrogates=builder)
    finding = judge(skeptic)
    skeptic.close()
    return finding, sandbox


def surrogate_runs(finding: Finding) -> list[RunResult]:
    return [r for r in finding.runs if r.side is Side.SURROGATE]


@pytest.mark.parametrize("kind", [CatchKind.FIX_VALIDITY, CatchKind.BEHAVIOR])
def test_a_catch_that_passes_5_of_5_on_a_plausible_surrogate_is_a_blocker(kind: CatchKind) -> None:
    builder = Builder([surrogate(1)])
    sandbox = Box(fails_on_patch(lambda n, seed: Outcome.PASS))
    skeptic = TargetSkeptic(TARGET, sandbox=sandbox, runner=Runner(), surrogates=builder)
    finding = judge(skeptic, repro(kind=kind))
    assert finding.status is Status.BLOCKER, finding.detail
    assert Finding.model_validate(finding.model_dump()) == finding
    assert [r.seed for r in surrogate_runs(finding)] == [0, 1, 2, 3, 4]
    assert all(r.outcome is Outcome.PASS for r in surrogate_runs(finding))
    assert "fixed_x.s1" in (finding.detail or "")
    assert (Side.SURROGATE, surrogate(1).patch) in sandbox.prepared  # base with the surrogate's patch
    assert builder.calls == [(TARGET.id, CLAIM.id, 3)]
    skeptic.close()
    assert len(sandbox.closed) == 3  # patch, base and the surrogate's workspace
    assert Side.SURROGATE in {w.side for w in skeptic.prepared}


def test_with_no_builder_wired_it_stays_inconclusive_and_says_so() -> None:
    finding, sandbox = check(fails_on_patch(lambda n, seed: Outcome.PASS), None)
    assert finding.status is Status.S_SAT_INCONCLUSIVE
    assert "no surrogate builder is wired" in (finding.detail or "")
    assert all(side is not Side.SURROGATE for side, _ in sandbox.prepared)


def test_with_only_implausible_surrogates_it_stays_inconclusive_with_each_reason() -> None:
    builder = Builder([surrogate(1, plausible=False), surrogate(2, plausible=False)])
    finding, sandbox = check(fails_on_patch(lambda n, seed: Outcome.PASS), builder)
    assert finding.status is Status.S_SAT_INCONCLUSIVE
    detail = finding.detail or ""
    assert "none of the 2 built is plausible" in detail
    assert "fixed_x.s1: surrogate 1 broke an existing test" in detail
    assert "fixed_x.s2: surrogate 2 broke an existing test" in detail
    assert surrogate_runs(finding) == [] and all(side is not Side.SURROGATE for side, _ in sandbox.prepared)


def test_with_no_surrogate_built_it_stays_inconclusive() -> None:
    finding, _ = check(fails_on_patch(lambda n, seed: Outcome.PASS), Builder([]))
    assert finding.status is Status.S_SAT_INCONCLUSIVE
    assert "none was built" in (finding.detail or "")


def test_a_catch_every_plausible_surrogate_fails_every_run_is_possibly_toxic() -> None:
    builder = Builder([surrogate(1), surrogate(2, plausible=False), surrogate(3)])
    finding, sandbox = check(fails_on_patch(lambda n, seed: Outcome.FAIL), builder)
    assert finding.status is Status.POSSIBLY_TOXIC
    assert "all 2 plausible surrogates" in (finding.detail or "")
    assert [(n, seed) for side, seed, n in sandbox.calls if side is Side.SURROGATE] == [
        (n, s) for n in (1, 3) for s in range(5)
    ]
    assert len(surrogate_runs(finding)) == 10


def test_the_first_surrogate_fails_it_and_the_second_passes_it_so_it_is_a_blocker_on_the_second() -> None:
    builder = Builder([surrogate(1), surrogate(2)])
    finding, sandbox = check(fails_on_patch(lambda n, seed: Outcome.PASS if n == 2 else Outcome.FAIL), builder)
    assert finding.status is Status.BLOCKER, finding.detail
    assert "fixed_x.s2" in (finding.detail or "")
    assert [r.outcome for r in surrogate_runs(finding)] == [Outcome.PASS] * 5  # surrogate 2's runs only
    assert [n for side, _, n in sandbox.calls if side is Side.SURROGATE] == [1] * 5 + [2] * 5


def test_a_surrogate_that_passes_only_some_runs_is_flaky_unless_another_passes_5_of_5() -> None:
    sometimes = fails_on_patch(lambda n, seed: Outcome.PASS if n == 2 or seed < 3 else Outcome.FAIL)
    flaky, _ = check(sometimes, Builder([surrogate(1)]))
    assert flaky.status is Status.FLAKY
    assert "fixed_x.s1 in 3 of 5" in (flaky.detail or "")
    later, _ = check(sometimes, Builder([surrogate(1), surrogate(2)]))
    assert later.status is Status.BLOCKER


def test_a_surrogate_run_that_cannot_be_read_leaves_the_candidate_unchecked() -> None:
    def script(side: Side, seed: int, n: int) -> RunResult | Exception:
        if side is Side.SURROGATE and seed == 2:
            return result(side, seed, Outcome.TIMEOUT, FailureType.TIMEOUT)
        return fails_on_patch(lambda n, seed: Outcome.PASS)(side, seed, n)

    finding, _ = check(script, Builder([surrogate(1)]))
    assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.TIMEOUT)

    raised, _ = check(
        lambda side, seed, n: (
            RuntimeError("docker died")
            if side is Side.SURROGATE
            else fails_on_patch(lambda n, seed: Outcome.PASS)(side, seed, n)
        ),
        Builder([surrogate(1)]),
    )
    assert (raised.status, raised.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.RUNTIME)

    no_env, _ = check(
        fails_on_patch(lambda n, seed: Outcome.PASS), Builder([surrogate(1)]), prepare_error_on_surrogate=True
    )
    assert (no_env.status, no_env.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.NO_ENV)


def test_two_candidates_of_one_claim_on_two_arms_build_the_surrogates_once_and_charge_them_once() -> None:
    builder = Builder([surrogate(1, micro_usd=7_000)])
    sandbox = Box(fails_on_patch(lambda n, seed: Outcome.PASS))
    skeptic = TargetSkeptic(TARGET, sandbox=sandbox, runner=Runner(), surrogates=builder)
    findings = [judge(skeptic, repro(n), number=n) for _arm in ("spark", "glimmer") for n in (1, 2)]
    assert [f.status for f in findings] == [Status.BLOCKER] * 4
    assert len(builder.calls) == 1
    charged = [sum(c.micro_usd for c in f.costs if c.stage == "surrogate") for f in findings]
    assert charged == [7_000, 0, 0, 0]  # on the finding whose candidate built them, only
    assert skeptic.surrogate_spent == 7_000
    assert sum(side is Side.SURROGATE for side, _ in sandbox.prepared) == 1  # its workspace kept and shared
    skeptic.close()


def test_another_claim_gets_its_own_build_and_a_regression_candidate_none() -> None:
    builder = Builder([surrogate(1)])
    sandbox = Box(fails_on_patch(lambda n, seed: Outcome.PASS))
    skeptic = TargetSkeptic(TARGET, sandbox=sandbox, runner=Runner(), surrogates=builder)
    regression = Repro.model_validate({**repro().model_dump(), "kind": CatchKind.REGRESSION, "quote": ""})
    assert judge(skeptic, regression).status is Status.BLOCKER  # passes on base: no surrogate needed
    assert builder.calls == []
    behavior = Claim(id="adds_behavior", type=ClaimType.ADDS_BEHAVIOR, text="adds", span=CLAIM.span)
    judge(skeptic, repro())
    other = Repro.model_validate({**repro().model_dump(), "claim_id": behavior.id})
    other_attack = ATTACK.model_copy(update={"claim_id": behavior.id})
    skeptic.judge(
        skeptic.screen(other, other_attack), claim=behavior, finding_id=f"{TARGET.id}/{behavior.id}/x", wall=lambda: 1
    )
    assert [c[1] for c in builder.calls] == ["fixed_x", "adds_behavior"]
    skeptic.close()


def test_an_import_error_on_base_still_refutes_before_any_surrogate_is_built() -> None:
    def script(side: Side, seed: int, n: int) -> RunResult:
        if side is Side.PATCH:
            return result(side, seed, Outcome.FAIL, FailureType.ASSERTION)
        return result(side, seed, Outcome.FAIL, FailureType.IMPORT)

    builder = Builder([surrogate(1)])
    finding, _ = check(script, builder)
    assert finding.status is Status.REFUTED
    assert builder.calls == []


def test_a_builder_that_raises_builds_nothing_and_never_crashes_the_run() -> None:
    class Raising:
        def build(self, target: Target, claim: Claim, **_: Any) -> tuple[Surrogate, ...]:
            raise RuntimeError("builder broke")

    sandbox = Box(fails_on_patch(lambda n, seed: Outcome.PASS))
    skeptic = TargetSkeptic(TARGET, sandbox=sandbox, runner=Runner(), surrogates=Raising())  # type: ignore[arg-type]
    finding = judge(skeptic)
    assert finding.status is Status.S_SAT_INCONCLUSIVE and "none was built" in (finding.detail or "")


def test_a_plausible_surrogate_needs_a_patch() -> None:
    with pytest.raises(ValueError, match="a plausible surrogate has a patch"):
        Surrogate(id="s", claim_id="fixed_x", patch=" ", plausible=True, why="ok")
    assert Surrogate(id="s", claim_id="fixed_x", patch="", plausible=False, why="gave up").patch == ""


def test_a_surrogate_with_a_run_that_has_no_result_is_passed_over_for_the_next() -> None:
    def script(on_two: Outcome) -> Script:
        def run(side: Side, seed: int, n: int) -> RunResult:
            if side is Side.SURROGATE and n == 1 and seed == 1:
                return result(side, seed, Outcome.TIMEOUT, FailureType.TIMEOUT)
            return fails_on_patch(lambda n, seed: Outcome.PASS if n == 1 else on_two)(side, seed, n)

        return run

    passes, sandbox = check(script(Outcome.PASS), Builder([surrogate(1), surrogate(2)]))
    assert passes.status is Status.BLOCKER and "fixed_x.s2" in (passes.detail or "")
    assert [n for side, _, n in sandbox.calls if side is Side.SURROGATE] == [1, 1] + [2] * 5
    # No surrogate passes it 5 of 5 and one had a run with no result: unchecked, never toxic.
    fails, _ = check(script(Outcome.FAIL), Builder([surrogate(1), surrogate(2)]))
    assert (fails.status, fails.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.TIMEOUT)
    assert "1 of 2 plausible surrogates had a run with no result" in (fails.detail or "")


@dataclass
class Charging:
    """A builder that makes k plausible surrogates of whatever claim it is asked for, each spending `each`."""

    each: int = SURROGATE_CAP_MICRO_USD
    calls: list[tuple[str, int]] = field(default_factory=list)

    def build(
        self, target: Target, claim: Claim, *, sandbox: Sandbox, runner: SuiteRunner, k: int
    ) -> tuple[Surrogate, ...]:
        self.calls.append((claim.id, k))
        return tuple(
            surrogate(n, micro_usd=self.each).model_copy(update={"id": f"{claim.id}.s{n}", "claim_id": claim.id})
            for n in range(1, k + 1)
        )


def judged_on(skeptic: TargetSkeptic, claim: Claim, n: int = 1) -> Finding:
    attack = ATTACK.model_copy(update={"claim_id": claim.id})
    candidate = Repro.model_validate({**repro(n).model_dump(), "claim_id": claim.id})
    return skeptic.judge(
        skeptic.screen(candidate, attack), claim=claim, finding_id=f"{TARGET.id}/{claim.id}/a.{n}", wall=lambda: 1.0
    )


def claim_of(n: int) -> Claim:
    return Claim(id=f"fixed_x:{n:010d}", type=ClaimType.FIXED_X, text="fixed", span=CLAIM.span)


def test_the_target_s_surrogate_budget_limits_k_across_claims_and_a_fourth_full_claim_gets_none() -> None:
    builder = Charging()
    sandbox = Box(fails_on_patch(lambda n, seed: Outcome.PASS))
    skeptic = TargetSkeptic(TARGET, sandbox=sandbox, runner=Runner(), surrogates=builder)
    assert SURROGATE_TARGET_CAP_MICRO_USD == 3 * 3 * SURROGATE_CAP_MICRO_USD
    for n in (1, 2, 3):  # three full builds spend the whole budget
        assert judged_on(skeptic, claim_of(n)).status is Status.BLOCKER
    assert skeptic.surrogate_spent == SURROGATE_TARGET_CAP_MICRO_USD
    fourth = judged_on(skeptic, claim_of(4))
    assert fourth.status is Status.S_SAT_INCONCLUSIVE
    assert "over the target's surrogate budget, no surrogate built" in (fourth.detail or "")
    assert builder.calls == [(claim_of(n).id, 3) for n in (1, 2, 3)]  # the fourth claim's builder is never called
    # A claim already built is still judged by its surrogates, at no cost, with the budget spent.
    again = judged_on(skeptic, claim_of(2), n=2)
    assert again.status is Status.BLOCKER and len(builder.calls) == 3
    assert sum(c.micro_usd for c in again.costs if c.stage == "surrogate") == 0
    skeptic.close()


@pytest.mark.parametrize(
    ("budget", "k"),
    [
        (0, 0),
        (SURROGATE_CAP_MICRO_USD - 1, 0),
        (SURROGATE_CAP_MICRO_USD, 1),
        (2 * SURROGATE_CAP_MICRO_USD - 1, 1),
        (2 * SURROGATE_CAP_MICRO_USD, 2),
        (SURROGATE_TARGET_CAP_MICRO_USD, 3),
    ],
)
def test_a_build_makes_only_as_many_surrogates_as_the_budget_left_covers_and_is_never_redone(
    budget: int, k: int
) -> None:
    builder = Charging(each=1)
    sandbox = Box(fails_on_patch(lambda n, seed: Outcome.FAIL))
    skeptic = TargetSkeptic(TARGET, sandbox=sandbox, runner=Runner(), surrogates=builder, surrogate_budget=budget)
    first = judged_on(skeptic, CLAIM)
    if k == 0:
        assert builder.calls == [] and first.status is Status.S_SAT_INCONCLUSIVE
        assert "over the target's surrogate budget, no surrogate built" in (first.detail or "")
        return
    assert builder.calls == [(CLAIM.id, k)]
    assert first.status is Status.POSSIBLY_TOXIC and f"all {k} plausible surrogate" in (first.detail or "")
    # Another candidate, or another arm: the claim's build is kept, never redone, however few it made.
    again = judged_on(skeptic, CLAIM, n=2)
    assert builder.calls == [(CLAIM.id, k)] and f"all {k} plausible surrogate" in (again.detail or "")
    skeptic.close()
