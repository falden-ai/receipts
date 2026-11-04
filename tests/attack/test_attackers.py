"""AT3 (regression) and AT2 stage 1 (correctness) with a fake model client and a fake sandbox: no model, no Docker.

The teeth: a candidate needs a counted failure on the patch (and, for AT3, a pass on base); the model's judgment can
only take candidates away; an agent's edit to an existing test never hides a regression; a repro never overwrites a
repository file; every cost is kept once; and neither attacker raises for a model or sandbox failure.
"""

from __future__ import annotations

import json
import threading
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pytest

from receipts.attack.classify import SuiteRun, classify_suite, with_expect
from receipts.attack.correctness import (
    ONE_CASE,
    REGRESSION_RULE,
    SCOPE_RULE,
    CorrectnessAttacker,
    kind_for,
    placed,
)
from receipts.attack.regression import RegressionAttacker
from receipts.attack.runners import DjangoRunner, PytestRunner, SympyRunner
from receipts.attack.sandbox import SandboxError
from receipts.contracts import (
    Arm,
    Attack,
    AttackOutcome,
    CaseResult,
    CatchKind,
    ContextMask,
    FailureType,
    Lens,
    ModelCallError,
    ModelRequest,
    ModelResponse,
    Outcome,
    RunnerKind,
    RunResult,
    Side,
    StageCost,
    SuiteRunner,
    Target,
    Tier,
    UncheckedReason,
    Usage,
    Workspace,
)

DIGEST = "sha256:" + "c" * 64
ARM = Arm(name="spark", attacker_model="muse-spark-1.3", step_limit=20, seed=7)
TREE = {
    "src/pkg/__init__.py": "",
    "src/pkg/parser.py": "def parse(x):\n    return x\n",
    "tests/test_parser.py": "from pkg.parser import parse\n\n\ndef test_a():\n    assert parse(1) == 1\n",
}
PATCH = (
    "diff --git a/src/pkg/parser.py b/src/pkg/parser.py\n--- a/src/pkg/parser.py\n+++ b/src/pkg/parser.py\n"
    "@@ -1,2 +1,2 @@\n def parse(x):\n-    return x\n+    return x + 1\n"
    "diff --git a/tests/test_parser.py b/tests/test_parser.py\n--- a/tests/test_parser.py\n+++ b/tests/test_parser.py\n"
    "@@ -4,2 +4,2 @@\n def test_a():\n-    assert parse(1) == 1\n+    assert parse(1) == 2\n"
)


def target(patch: str = PATCH) -> Target:
    return Target(
        id="entry~pkg-1", tier=Tier.A, repo="o/pkg", base_sha="a" * 40, patch=patch, issue="parse must return x"
    )


def attack(claim: str = "nothing_else_changed", mask: ContextMask = ContextMask.DIFF, **kw: object) -> Attack:
    fields: dict[str, object] = {
        "id": "a1",
        "claim_id": claim,
        "lens": Lens.REGRESSION,
        "hypothesis": "the change breaks parse",
        "mask": mask,
        "temperature": 0.8,
        **kw,
    }
    return Attack.model_validate(fields)


def response(payload: object, *, micro_usd: int = 100, finish: str = "stop") -> ModelResponse:
    text = payload if isinstance(payload, str) else json.dumps(payload)
    return ModelResponse(
        model="muse-spark-1.3",
        text=text,
        finish_reason=finish,
        usage=Usage(input_tokens=10, output_tokens=5),
        cost=StageCost(stage="x", model="muse-spark-1.3", input_tokens=10, output_tokens=5, micro_usd=micro_usd),
    )


class FakeClient:
    def __init__(self, *answers: ModelResponse | ModelCallError) -> None:
        self.answers = list(answers)
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        answer = self.answers.pop(0)
        if isinstance(answer, ModelCallError):
            raise answer
        return answer


def run(
    side: Side, outcome: Outcome = Outcome.PASS, ftype: FailureType | None = None, seconds: float = 1.5
) -> RunResult:
    return RunResult(
        side=side,
        outcome=outcome,
        failure_type=ftype,
        exit_code=0 if outcome is Outcome.PASS else 1,
        fresh_container=False,
        duration_s=seconds,
        image_digest=DIGEST,
    )


def suite(side: Side, *cases: CaseResult, problem: str | None = None) -> SuiteRun:
    bad = any(c.outcome is not Outcome.PASS for c in cases)
    result = (
        run(side, Outcome.ERROR, FailureType.INFRA)
        if problem
        else run(side, Outcome.FAIL if bad else Outcome.PASS, FailureType.ASSERTION if bad else None)
    )
    return SuiteRun(result, () if problem else cases, problem)


def ok(name: str, file: str | None = None) -> CaseResult:
    return CaseResult(name=name, outcome=Outcome.PASS, file=file)


def failed(
    name: str,
    ftype: FailureType = FailureType.ASSERTION,
    message: str = "AssertionError: 2 != 1",
    *,
    file: str | None = None,
) -> CaseResult:
    outcome = Outcome.FAIL if ftype in (FailureType.ASSERTION, FailureType.CRASH_IN_CHANGED) else Outcome.ERROR
    return CaseResult(name=name, outcome=outcome, failure_type=ftype, message=message, file=file)


class FakeClock:
    def __init__(self, now: float = 100.0) -> None:
        self.now = now

    def __call__(self) -> float:
        return self.now


class FakeSandbox:
    """A SuiteSandbox: scripted per-side suite runs and run_test results, the tree from a dict, every call recorded.
    With a clock, each suite run moves it on by the run's duration."""

    def __init__(
        self,
        tree: Mapping[str, str] = TREE,
        *,
        suites: Mapping[Side, SuiteRun] | None = None,
        run_test_result: Callable[[Mapping[str, str], Sequence[str]], RunResult] | None = None,
        fail_prepare: bool = False,
        fail_runs: bool = False,
        clock: FakeClock | None = None,
    ) -> None:
        self.clock = clock
        self.tree = dict(tree)
        self.suites = dict(suites or {})
        self.run_test_result = run_test_result
        self.fail_prepare = fail_prepare
        self.fail_runs = fail_runs
        self.prepared: list[tuple[Side, str | None]] = []
        self.closed: list[str] = []
        self.suite_calls: list[dict[str, object]] = []
        self.test_calls: list[dict[str, object]] = []

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        if self.fail_prepare:
            raise SandboxError("the image is gone")
        self.prepared.append((side, patch))
        return Workspace(id=f"ws-{side.value}-{len(self.prepared)}", side=side, root="/testbed", image_digest=DIGEST)

    def run(self, workspace: Workspace, command: Sequence[str], **kw: object) -> RunResult:
        raise AssertionError("the attackers use run_test and run_cases")

    def run_test(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: SuiteRunner,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
        expect: Sequence[str] | None = None,
    ) -> RunResult:
        if self.fail_runs:
            raise SandboxError("docker exec failed")
        self.test_calls.append(
            {"workspace": workspace.id, "tests": list(tests), "files": dict(files or {}), "seed": seed,
             "expect": list(expect or []), "timeout": timeout_s, "changed": list(changed_files)}
        )  # fmt: skip
        assert self.run_test_result is not None
        return self.run_test_result(files or {}, tests)

    def run_cases(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: SuiteRunner,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
    ) -> SuiteRun:
        if self.fail_runs:
            raise SandboxError("docker exec failed")
        self.suite_calls.append(
            {"workspace": workspace.id, "side": workspace.side, "tests": list(tests), "files": dict(files or {}),
             "seed": seed, "changed": list(changed_files), "timeout": timeout_s}
        )  # fmt: skip
        run = self.suites[workspace.side]
        if self.clock is not None:
            self.clock.now += run.result.duration_s
        return run

    def close(self, workspace: Workspace) -> None:
        self.closed.append(workspace.id)

    def tree_files(self, workspace: Workspace) -> tuple[str, ...]:
        return tuple(sorted(self.tree))

    def read_text(self, workspace: Workspace, path: str, *, max_bytes: int = 4 * 1024 * 1024) -> str:
        if path not in self.tree:
            raise SandboxError(f"{path} is not a file in the workspace")
        return self.tree[path]


class PlainSandbox:
    """A contracts.Sandbox with no per-case results and no tree."""

    def __init__(self) -> None:
        self.prepared = 0

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        self.prepared += 1
        return Workspace(id="ws", side=side, root="/testbed", image_digest=DIGEST)

    def run(self, workspace: Workspace, command: Sequence[str], **kw: object) -> RunResult:
        raise AssertionError

    def run_test(self, workspace: Workspace, tests: Sequence[str], **kw: object) -> RunResult:
        raise AssertionError

    def close(self, workspace: Workspace) -> None:
        pass


VERDICT_KEEP = {"verdicts": [{"test": "tests/test_parser.py::test_a", "intended": False, "reason": "unasked"}]}
VERDICT_DROP = {"verdicts": [{"test": "tests/test_parser.py::test_a", "intended": True, "reason": "the issue asked"}]}


def at3_sandbox(**kw: object) -> FakeSandbox:
    suites = {
        Side.BASE: suite(Side.BASE, ok("tests/test_parser.py::test_a")),
        Side.PATCH: suite(Side.PATCH, failed("tests/test_parser.py::test_a")),
    }
    return FakeSandbox(suites=suites, **kw)  # type: ignore[arg-type]


# -- AT3 --------------------------------------------------------------------------------------------------------------


def test_at3_finds_a_regression_and_runs_the_existing_test_as_it_was_on_base() -> None:
    sandbox, client = at3_sandbox(), FakeClient(response(VERDICT_KEEP))
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    (repro,) = outcome.repros
    assert (repro.kind, repro.lens, repro.claim_id, repro.attack_id) == (
        CatchKind.REGRESSION,
        Lens.REGRESSION,
        "nothing_else_changed",
        "a1",
    )
    assert repro.command == ("tests/test_parser.py::test_a",)
    assert repro.cases == ("tests/test_parser.py::test_a",)  # contract v3: the one case it runs, in the runner's id
    assert repro.files == {"tests/test_parser.py": TREE["tests/test_parser.py"]}  # the base version, not the agent's
    base_call, patch_call = sandbox.suite_calls
    assert (base_call["side"], base_call["tests"], base_call["files"]) == (Side.BASE, ["tests/test_parser.py"], {})
    # PLANT for the edited test: the agent changed the test to expect its new value. The patch run writes the base
    # test back, so the regression shows; run with the agent's version, the case would pass and hide it.
    assert patch_call["files"] == {"tests/test_parser.py": TREE["tests/test_parser.py"]}
    assert base_call["changed"] == ["src/pkg/parser.py", "tests/test_parser.py"] and base_call["seed"] == 7
    assert sorted(sandbox.closed) == ["ws-base-1", "ws-patch-2"]
    # Every cost once: the judgment's, and each suite run's compute seconds.
    assert [c.micro_usd for c in outcome.costs if c.model] == [100]
    assert [c.seconds for c in outcome.costs if c.stage == "at3_suite"] == [1.5, 1.5]
    (request,) = client.requests
    assert (request.model, request.temperature, request.stage) == ("muse-spark-1.3", 0.8, "at3_intent")
    assert (
        "tests/test_parser.py::test_a" in request.messages[1].content
        and "parse must return x" in request.messages[1].content
    )


def test_at3_the_model_can_only_take_candidates_away() -> None:
    outcome = RegressionAttacker().attack(
        target(),
        attack(),
        arm=ARM,
        client=FakeClient(response(VERDICT_DROP)),
        sandbox=at3_sandbox(),
        runner=PytestRunner(),
    )
    assert outcome.repros == () and outcome.unchecked == ()
    # A verdict for a test that is not a candidate adds nothing.
    extra = {"verdicts": [*VERDICT_KEEP["verdicts"], {"test": "tests/test_other.py::test_z", "intended": False}]}
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(response(extra)), sandbox=at3_sandbox(), runner=PytestRunner()
    )
    assert [r.command for r in outcome.repros] == [("tests/test_parser.py::test_a",)]


def test_at3_a_failed_judgment_takes_nothing_away_and_keeps_its_cost() -> None:
    spent = StageCost(stage="at3_intent", model="muse-spark-1.3", micro_usd=40)
    error = ModelCallError("rate limited", UncheckedReason.BUDGET, cost=spent)
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(error), sandbox=at3_sandbox(), runner=PytestRunner()
    )
    assert len(outcome.repros) == 1
    (unchecked,) = outcome.unchecked
    assert unchecked.reason is UncheckedReason.BUDGET and "no candidate was taken away" in unchecked.detail
    assert spent in outcome.costs


def test_at3_garbage_twice_is_unparsed_and_both_calls_are_paid() -> None:
    client = FakeClient(response("not json"), response({"verdicts": "nope"}))
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=client, sandbox=at3_sandbox(), runner=PytestRunner()
    )
    assert len(outcome.repros) == 1 and outcome.unchecked[0].reason is UncheckedReason.UNPARSED
    assert len([c for c in outcome.costs if c.model]) == 2
    assert "did not match" in client.requests[1].messages[-1].content  # the retry carried the error back


@pytest.mark.parametrize(
    ("base", "patch"),
    [
        (ok("t::a"), failed("t::a", FailureType.CRASH_ELSEWHERE)),  # not a counted failure
        (ok("t::a"), failed("t::a", FailureType.IMPORT)),
        (failed("t::a"), failed("t::a")),  # already failing on base
        (CaseResult(name="t::a", outcome=Outcome.SKIPPED), failed("t::a")),  # did not run on base
        # Contract v3.1: a case that failed as expected on base did not pass there (only R24's gold check counts it).
        (CaseResult(name="t::a", outcome=Outcome.SKIPPED, expected_failure=True), failed("t::a")),
        (ok("t::a"), ok("t::a")),
        (ok("t::a"), failed("t::b")),  # a case base never ran
    ],
)
def test_at3_only_a_counted_failure_on_a_case_that_passed_on_base_is_a_candidate(
    base: CaseResult, patch: CaseResult
) -> None:
    sandbox = FakeSandbox(suites={Side.BASE: suite(Side.BASE, base), Side.PATCH: suite(Side.PATCH, patch)})
    client = FakeClient()
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.repros == () and client.requests == []  # no candidate, so no model call


def test_at3_a_crash_in_changed_code_is_a_candidate() -> None:
    patch = failed("tests/test_parser.py::test_a", FailureType.CRASH_IN_CHANGED, "TypeError: boom")
    sandbox = FakeSandbox(
        suites={Side.BASE: suite(Side.BASE, ok("tests/test_parser.py::test_a")), Side.PATCH: suite(Side.PATCH, patch)}
    )
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(response(VERDICT_KEEP)), sandbox=sandbox, runner=PytestRunner()
    )
    assert len(outcome.repros) == 1


def test_at3_needs_per_case_results() -> None:
    sandbox = PlainSandbox()
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.unchecked[0].reason is UncheckedReason.NO_ENV and sandbox.prepared == 0


@pytest.mark.parametrize(
    ("sandbox", "patch", "reason"),
    [
        (FakeSandbox(fail_prepare=True), PATCH, UncheckedReason.NO_ENV),
        (FakeSandbox(tree={"src/pkg/parser.py": ""}), PATCH.split("diff --git a/tests")[0], UncheckedReason.NO_INPUT),
        (FakeSandbox(), "diff --git a/x b/x\n", UncheckedReason.NO_INPUT),  # changes no readable file
    ],
)
def test_at3_unchecked_reasons(sandbox: FakeSandbox, patch: str, reason: UncheckedReason) -> None:
    outcome = RegressionAttacker().attack(
        target(patch), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.repros == () and outcome.unchecked[0].reason is reason


def test_at3_an_unreadable_run_is_unchecked_and_a_timeout_says_so() -> None:
    sandbox = FakeSandbox(suites={Side.BASE: suite(Side.BASE, problem="the run wrote no report")})
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.unchecked[0].reason is UncheckedReason.RUNTIME and "no report" in outcome.unchecked[0].detail
    timed = SuiteRun(run(Side.BASE, Outcome.TIMEOUT, FailureType.TIMEOUT), (), "killed for time")
    sandbox = FakeSandbox(suites={Side.BASE: timed})
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.unchecked[0].reason is UncheckedReason.TIMEOUT


def test_at3_never_raises_for_a_sandbox_failure_and_closes_its_workspaces() -> None:
    sandbox = FakeSandbox(fail_runs=True)
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert (
        outcome.unchecked[0].reason is UncheckedReason.RUNTIME and "docker exec failed" in outcome.unchecked[0].detail
    )
    assert sandbox.closed == ["ws-base-1"]


def test_at3_lists_the_test_files_past_its_cap() -> None:
    tree = {**TREE, "tests/test_more.py": "import pkg.parser\n", "tests/test_also.py": "from pkg import parser\n"}
    sandbox = FakeSandbox(tree, suites=at3_sandbox().suites)
    outcome = RegressionAttacker(max_test_files=1).attack(
        target(), attack(), arm=ARM, client=FakeClient(response(VERDICT_KEEP)), sandbox=sandbox, runner=PytestRunner()
    )
    budget = [u for u in outcome.unchecked if u.reason is UncheckedReason.BUDGET]
    assert len(budget) == 1 and "tests/test_also.py" in budget[0].detail and "tests/test_more.py" in budget[0].detail
    assert sandbox.suite_calls[0]["tests"] == ["tests/test_parser.py"]


def test_at3_keeps_the_overflow_note_when_a_run_cannot_be_read() -> None:
    tree = {**TREE, "tests/test_more.py": "import pkg.parser\n"}
    sandbox = FakeSandbox(tree, suites={Side.BASE: suite(Side.BASE, problem="the run wrote no report")})
    outcome = RegressionAttacker(max_test_files=1).attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert [u.reason for u in outcome.unchecked] == [UncheckedReason.BUDGET, UncheckedReason.RUNTIME]


def test_at3_names_sympy_and_django_repros_in_their_runners_forms() -> None:
    tree = {"sympy/core/numbers.py": "", "sympy/core/tests/test_numbers.py": "def test_x():\n    assert True\n"}
    patch = (
        "diff --git a/sympy/core/numbers.py b/sympy/core/numbers.py\n"
        "--- a/sympy/core/numbers.py\n+++ b/sympy/core/numbers.py\n"
    )
    numbers = "sympy/core/tests/test_numbers.py"
    sandbox = FakeSandbox(
        tree,
        suites={
            Side.BASE: suite(Side.BASE, ok("test_x", numbers)),
            Side.PATCH: suite(Side.PATCH, failed("test_x", file=numbers)),
        },
    )
    verdict = {"verdicts": [{"test": f"{numbers}::test_x", "intended": False}]}
    outcome = RegressionAttacker().attack(
        target(patch), attack(), arm=ARM, client=FakeClient(response(verdict)), sandbox=sandbox, runner=SympyRunner()
    )
    assert outcome.repros[0].command == ("sympy/core/tests/test_numbers.py::test_x",) == outcome.repros[0].cases
    dj_tree = {
        "django/db/models/query.py": "",
        "tests/queries/tests.py": "from django.db.models.query import Q\nfrom django.test import TestCase\n\n\n"
        "class T(TestCase):\n    def test_q(self):\n        pass\n",
    }
    dj_patch = "diff --git a/django/db/models/query.py b/django/db/models/query.py\n"
    name = "queries.tests.T.test_q"
    sandbox = FakeSandbox(
        dj_tree, suites={Side.BASE: suite(Side.BASE, ok(name)), Side.PATCH: suite(Side.PATCH, failed(name))}
    )
    verdict = {"verdicts": [{"test": name, "intended": False}]}
    outcome = RegressionAttacker().attack(
        target(dj_patch),
        attack(),
        arm=ARM,
        client=FakeClient(response(verdict)),
        sandbox=sandbox,
        runner=DjangoRunner(),
    )
    assert sandbox.suite_calls[0]["tests"] == ["queries.tests"]
    assert outcome.repros[0].command == (name,) == outcome.repros[0].cases
    assert list(outcome.repros[0].files) == ["tests/queries/tests.py"]


_SY_A, _SY_B = "sympy/core/tests/test_a.py", "sympy/core/tests/test_b.py"
_SY_TREE = {
    "sympy/core/f.py": "def f():\n    return 1\n",
    _SY_A: "from sympy.core.f import f\n\n\ndef test_x():\n    assert f() == 1\n",
    _SY_B: "from sympy.core.f import f\n\n\ndef test_x():\n    assert f() + 0 == 1\n",
}
_SY_PATCH = "diff --git a/sympy/core/f.py b/sympy/core/f.py\n--- a/sympy/core/f.py\n+++ b/sympy/core/f.py\n"


def _not_intended() -> FakeClient:
    return FakeClient(response({"verdicts": []}))


def test_at3_finds_a_sympy_regression_in_its_own_file_when_two_files_name_the_test() -> None:
    # PLANT (review blocker): cases were keyed by name alone, and the repro went to the first selected file that
    # defines test_x. test_b.py's regression became a repro of test_a.py's test_x, which passes, and was lost.
    sandbox = FakeSandbox(
        _SY_TREE,
        suites={
            Side.BASE: suite(Side.BASE, ok("test_x", _SY_A), ok("test_x", _SY_B)),
            Side.PATCH: suite(Side.PATCH, ok("test_x", _SY_A), failed("test_x", file=_SY_B)),
        },
    )
    client = _not_intended()
    outcome = RegressionAttacker().attack(
        target(_SY_PATCH), attack(), arm=ARM, client=client, sandbox=sandbox, runner=SympyRunner()
    )
    (repro,) = outcome.repros
    assert repro.command == (f"{_SY_B}::test_x",) and repro.files == {_SY_B: _SY_TREE[_SY_B]}
    assert f"Test: {_SY_B}::test_x" in client.requests[0].messages[1].content  # the judgment sees which file


def test_at3_a_case_failing_on_base_in_one_file_is_not_a_candidate_for_its_namesake() -> None:
    # PLANT (review blocker): test_x passed on base in test_a.py, so test_b.py's test_x, failing on base and on the
    # patch alike, was a false candidate.
    sandbox = FakeSandbox(
        _SY_TREE,
        suites={
            Side.BASE: suite(Side.BASE, ok("test_x", _SY_A), failed("test_x", file=_SY_B)),
            Side.PATCH: suite(Side.PATCH, ok("test_x", _SY_A), failed("test_x", file=_SY_B)),
        },
    )
    client = FakeClient()
    outcome = RegressionAttacker().attack(
        target(_SY_PATCH), attack(), arm=ARM, client=client, sandbox=sandbox, runner=SympyRunner()
    )
    assert outcome.repros == () and client.requests == []


_SELECTED = json.loads((Path(__file__).parent / "fixtures" / "runners" / "sympy_13974_selected_tests.json").read_text())


def _duplicated_names() -> list[tuple[str, list[str]]]:
    files: dict[str, list[str]] = _SELECTED["functions"]
    by_name: dict[str, list[str]] = {}
    for path in _SELECTED["selected"]:
        for name in files[path]:
            by_name.setdefault(name, []).append(path)
    return sorted((name, paths) for name, paths in by_name.items() if len(paths) > 1)


def test_the_six_names_the_13974_selection_repeats_are_six() -> None:
    # The test functions of the 7 files AT3 selects in the sympy__sympy-13974 image's own tree (names only).
    assert [name for name, _ in _duplicated_names()] == [
        "test_anticommutator", "test_commutator", "test_dagger", "test_eval_trace", "test_innerproduct",
        "test_tensorproduct",
    ]  # fmt: skip


@pytest.mark.parametrize(("name", "paths"), _duplicated_names())
def test_at3_attributes_each_repeated_13974_name_to_the_file_that_regressed(name: str, paths: list[str]) -> None:
    # Every case of every selected file passes on base; on the patch only the last file defining `name` fails it.
    # Each file imports the changed module, as the real ones do (selection rule 3), then defines its real names.
    tree = {"sympy/physics/quantum/tensorproduct.py": ""}
    for path in _SELECTED["selected"]:
        defs = "".join(f"def {fn}():\n    pass\n\n\n" for fn in _SELECTED["functions"][path])
        tree[path] = f"from sympy.physics.quantum.tensorproduct import TensorProduct\n\n\n{defs}"
    every = [(path, fn) for path in _SELECTED["selected"] for fn in _SELECTED["functions"][path]]
    culprit = paths[-1]
    base = suite(Side.BASE, *(ok(fn, path) for path, fn in every))
    patch = suite(
        Side.PATCH, *(failed(fn, file=path) if (path, fn) == (culprit, name) else ok(fn, path) for path, fn in every)
    )
    sandbox = FakeSandbox(tree, suites={Side.BASE: base, Side.PATCH: patch})
    change = "sympy/physics/quantum/tensorproduct.py"
    outcome = RegressionAttacker().attack(
        target(f"diff --git a/{change} b/{change}\n"),
        attack(),
        arm=ARM,
        client=_not_intended(),
        sandbox=sandbox,
        runner=SympyRunner(),
    )
    assert sandbox.suite_calls[0]["tests"] == _SELECTED["selected"]
    (repro,) = outcome.repros
    assert repro.command == (f"{culprit}::{name}",) and list(repro.files) == [culprit]


def test_at3_lists_a_case_that_passed_on_base_and_was_infra_on_the_patch() -> None:
    # PLANT (the review's hole, in AT3): a case that passed on base and hit a sandbox-caused error on the patch
    # (typed INFRA) was dropped with no record, so a regression the sandbox hid looked like no regression.
    sandbox = FakeSandbox(
        suites={
            Side.BASE: suite(Side.BASE, ok("tests/test_parser.py::test_a")),
            Side.PATCH: suite(
                Side.PATCH,
                failed(
                    "tests/test_parser.py::test_a", FailureType.INFRA, "OSError: [Errno 101] Network is unreachable"
                ),
            ),
        }
    )
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    (item,) = outcome.unchecked
    assert (item.what, item.reason) == ("a1:tests/test_parser.py::test_a", UncheckedReason.RUNTIME)
    assert "Network is unreachable" in item.detail and outcome.repros == ()


def test_at3_shares_the_attackers_wall_between_its_two_runs() -> None:
    # PLANT (integrator call c): each run got 900 s, so one AT3 attack could take 30 minutes and more; the spec gives
    # an attacker 15 minutes of wall time. The base run gets half of what is left for the runs, the patch run the rest.
    clock = FakeClock()
    sandbox = at3_sandbox(clock=clock)
    sandbox.suites[Side.BASE] = suite(Side.BASE, ok("tests/test_parser.py::test_a"))
    RegressionAttacker(wall_s=900.0, judge_reserve_s=60.0, clock=clock).attack(
        target(), attack(), arm=ARM, client=FakeClient(response(VERDICT_KEEP)), sandbox=sandbox, runner=PytestRunner()
    )
    base_call, patch_call = sandbox.suite_calls
    assert base_call["timeout"] == pytest.approx(420.0)  # (900 - 60) / 2
    assert patch_call["timeout"] == pytest.approx(900.0 - 60.0 - 1.5)  # what is left after the 1.5 s base run


def test_at3_a_run_that_does_not_fit_the_wall_is_unchecked_for_time() -> None:
    clock = FakeClock()
    slow_base = SuiteRun(run(Side.BASE, Outcome.TIMEOUT, FailureType.TIMEOUT, seconds=420.0), (), "killed for time")
    sandbox = FakeSandbox(suites={Side.BASE: slow_base}, clock=clock)
    outcome = RegressionAttacker(clock=clock).attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    (item,) = outcome.unchecked
    assert item.reason is UncheckedReason.TIMEOUT and "420 s share of the attacker's 900 s wall" in item.detail
    # A base run that used up the wall leaves the patch run no time: unchecked for time, and never started.
    clock = FakeClock()
    sandbox = at3_sandbox(clock=clock)
    sandbox.suites[Side.BASE] = SuiteRun(run(Side.BASE, seconds=890.0), (ok("tests/test_parser.py::test_a"),))
    outcome = RegressionAttacker(clock=clock).attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert [u.reason for u in outcome.unchecked] == [UncheckedReason.TIMEOUT] and len(sandbox.suite_calls) == 1
    with pytest.raises(ValueError, match="must exceed"):
        RegressionAttacker(wall_s=60.0, judge_reserve_s=60.0)


_REC = Path(__file__).parent / "fixtures" / "runners"


def _recorded_patch_run(runner: SuiteRunner, stream: str, name: str) -> SuiteRun:
    """A patch-side suite run read from a real recording by the runner's own parser."""
    rec = json.loads((_REC / f"{name}.json").read_text())
    coarse = RunResult(
        side=Side.PATCH, outcome=Outcome.ERROR, exit_code=int(rec["exit_code"]), fresh_container=False, duration_s=1.0
    )
    return classify_suite(
        coarse, runner=runner, output=str(rec[stream]), report=None, changed_files=[], timed_out=False
    )


def _imports_changed(module: str) -> str:
    return f"from {module} import x  # noqa: F401\n"


@pytest.mark.parametrize("runner_name", ["pytest", "django", "sympy"])
def test_at3_lists_each_case_a_module_that_stopped_importing_on_the_patch_dropped(runner_name: str) -> None:
    # PLANT (round-4 item 5): a test module that imports on base and fails to import on the patch reports one case for
    # the module and none of its own, so every case that passed on base vanished from the patch run with no record.
    # Each is listed as unchecked (runtime) with the import error; an import error is never a candidate. The patch
    # runs are real recordings read by each runner's own parser; the base runs name the same modules' cases.
    if runner_name == "pytest":
        runner: SuiteRunner = PytestRunner()
        report = (_JUNIT / "collect_import.xml").read_text()
        coarse = RunResult(side=Side.PATCH, outcome=Outcome.ERROR, exit_code=2, fresh_container=False, duration_s=1.0)
        patch_run = classify_suite(coarse, runner=runner, output="", report=report, changed_files=[], timed_out=False)
        tree, changed = {"pkg/mod.py": "", "tests/test_import.py": _imports_changed("pkg.mod")}, "pkg/mod.py"
        lost = ["tests/test_import.py::test_a", "tests/test_import.py::TestK::test_b"]
        base = suite(Side.BASE, *(ok(c) for c in lost))
        module, why = "tests/test_import.py", "collection failure: ModuleNotFoundError: No module named"
    elif runner_name == "django":
        runner = DjangoRunner()
        patch_run = _recorded_patch_run(runner, "stderr", "unittest_failed_imports_py311")
        changed = "django/utils/text.py"
        tree = {
            changed: "",
            **{f"tests/{a}/tests.py": _imports_changed("django.utils.text") for a in ("brokena", "brokenb", "fine")},
        }
        lost = ["brokena.tests.T.test_x", "brokenb.tests.T.test_y"]
        base = suite(Side.BASE, *(ok(c) for c in lost), ok("fine.tests.T.test_ok"))  # fine still runs on the patch
        module, why = "brokena.tests", "ModuleNotFoundError: No module named 'receipts_nowhere'"
    else:
        runner = SympyRunner()
        patch_run = _recorded_patch_run(runner, "stdout", "sympy_modules")
        d, changed = "sympy/utilities/tests/test_receipts_", "sympy/utilities/misc.py"
        tree = {
            changed: "",
            **{
                f"{d}{m}.py": _imports_changed("sympy.utilities.misc")
                for m in ("a_ok", "b_import_error", "c_module_error", "d_module_error_first")
            },
        }
        lost = [f"{d}b_import_error.py::test_never_runs", f"{d}d_module_error_first.py::test_never_defined"]
        base = suite(
            Side.BASE,
            ok("test_ok", f"{d}a_ok.py"),
            ok("test_never_runs", f"{d}b_import_error.py"),
            ok("test_defined_before_the_error", f"{d}c_module_error.py"),  # it ran on the patch before the raise
            ok("test_never_defined", f"{d}d_module_error_first.py"),
        )
        module, why = f"{d}b_import_error.py", "ModuleNotFoundError: No module named 'receipts_no_such_module'"
    assert patch_run.problem is None  # the patch run itself was read
    sandbox = FakeSandbox(tree, suites={Side.BASE: base, Side.PATCH: patch_run})
    client = FakeClient()
    outcome = RegressionAttacker().attack(
        target(f"diff --git a/{changed} b/{changed}\n"),
        attack(),
        arm=ARM,
        client=client,
        sandbox=sandbox,
        runner=runner,
    )
    assert outcome.repros == () and client.requests == []  # no candidate, so no model call
    listed = {u.what.removeprefix("a1:"): u for u in outcome.unchecked}
    assert sorted(listed) == sorted(lost)  # each lost case once, and nothing for the cases that still ran
    assert all(u.reason is UncheckedReason.RUNTIME for u in listed.values())
    first = listed[lost[0]].detail
    assert (
        first.startswith(f"passed on base; on the patch its test module {module} failed to import: ") and why in first
    )


def test_at3_lists_a_case_that_passed_on_base_and_never_ran_on_the_patch_for_no_reason_it_gives() -> None:
    sandbox = FakeSandbox(
        suites={
            Side.BASE: suite(Side.BASE, ok("tests/test_parser.py::test_a"), ok("tests/test_parser.py::test_b")),
            Side.PATCH: suite(Side.PATCH, ok("tests/test_parser.py::test_a")),
        }
    )
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    (item,) = outcome.unchecked
    assert (item.what, item.reason) == ("a1:tests/test_parser.py::test_b", UncheckedReason.RUNTIME)
    assert item.detail == "passed on base; it never ran on the patch, which reports no import error for its test module"


def test_at3_a_case_that_failed_as_expected_on_base_and_never_ran_on_the_patch_is_no_gap_in_the_check() -> None:
    # Contract v3.1: a case that failed as expected on base did not pass there, so nothing was lost when the patch
    # never ran it, and the receipt must not say it "passed on base".
    # PLANT (the round-5 attack review's X2, regression.py's missing_on_patch counting a base expected failure as a
    # pass): it listed this case as unchecked, "passed on base; it never ran on the patch".
    xfail = CaseResult(name="tests/test_parser.py::test_b", outcome=Outcome.SKIPPED, expected_failure=True)
    sandbox = FakeSandbox(
        suites={
            Side.BASE: suite(Side.BASE, ok("tests/test_parser.py::test_a"), xfail),
            Side.PATCH: suite(Side.PATCH, ok("tests/test_parser.py::test_a")),
        }
    )
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.unchecked == () and outcome.repros == ()


def test_at3_says_which_class_hook_kept_a_case_from_running_on_the_patch() -> None:
    # A setUpClass that errors on the patch (recorded under Python 3.11: an assert in it) keeps its class's tests from
    # running. A fixture error is rule-rejected, so each such case is unchecked, and the detail names the hook.
    patch_run = _recorded_patch_run(DjangoRunner(), "stderr", "unittest_assert_hooks_py311")
    changed = "django/utils/text.py"
    tree = {changed: "", "tests/asserthooks/tests.py": _imports_changed("django.utils.text")}
    lost = "asserthooks.tests.SetUpClassAsserts.test_never_runs"
    base = suite(Side.BASE, ok(lost), ok("asserthooks.tests.A.test_ok"))
    sandbox = FakeSandbox(tree, suites={Side.BASE: base, Side.PATCH: patch_run})
    outcome = RegressionAttacker().attack(
        target(f"diff --git a/{changed} b/{changed}\n"), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox,
        runner=DjangoRunner(),
    )  # fmt: skip
    (item,) = outcome.unchecked
    assert (item.what, item.reason) == (f"a1:{lost}", UncheckedReason.RUNTIME)
    assert item.detail == (
        "passed on base; it never ran on the patch, where asserthooks.tests.SetUpClassAsserts.setUpClass errored: "
        "AssertionError: setUpClass assert"
    )


def test_at3_a_base_run_that_reported_no_case_checked_nothing_and_says_so() -> None:
    # A base run that was read but reported no case (pytest collected nothing): no candidate can come of it, and the
    # receipt says nothing was checked rather than nothing.
    empty = SuiteRun(
        run(Side.BASE, Outcome.ERROR, FailureType.INFRA).model_copy(update={"problem": "the run reported no case"}), ()
    )
    sandbox = FakeSandbox(suites={Side.BASE: empty})
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    (item,) = outcome.unchecked
    assert (item.reason, item.detail) == (
        UncheckedReason.RUNTIME,
        "the base run checked nothing: the run reported no case",
    )
    assert len(sandbox.suite_calls) == 1 and [s for s, _ in sandbox.prepared] == [Side.BASE]  # no patch run


class _JumpClock:
    """The attack starts at 100 s; every later reading is 900 s on, as if choosing the tests took the whole wall."""

    def __init__(self) -> None:
        self.calls = 0

    def __call__(self) -> float:
        self.calls += 1
        return 100.0 if self.calls == 1 else 1000.0


def test_at3_with_no_time_left_before_the_base_run_is_unchecked_for_time_and_runs_nothing() -> None:
    # The round-3 review's surviving mutant (regression.py:190, the floor before the base run): without it a spent
    # wall reached run_cases with no time, which the real sandbox refuses (runtime), not the timeout this is.
    sandbox = at3_sandbox()
    outcome = RegressionAttacker(clock=_JumpClock()).attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    (item,) = outcome.unchecked
    assert (item.reason, item.detail) == (UncheckedReason.TIMEOUT, "no time was left in the attacker's 900 s wall")
    assert sandbox.suite_calls == []


@pytest.mark.parametrize(
    "base_case",
    [
        failed("tests/test_parser.py::test_a"),
        CaseResult(name="tests/test_parser.py::test_a", outcome=Outcome.SKIPPED),
        CaseResult(name="tests/test_parser.py::test_a", outcome=Outcome.SKIPPED, expected_failure=True),
    ],
)
def test_at3_an_infra_case_that_did_not_pass_on_base_is_no_gap_in_the_check(base_case: CaseResult) -> None:
    # The round-3 review's surviving mutant (regression.py:324, the base-pass condition): a case that failed or
    # skipped on base was never going to be a regression, so its infra error on the patch is not listed.
    infra = failed("tests/test_parser.py::test_a", FailureType.INFRA, "OSError: [Errno 101] Network is unreachable")
    sandbox = FakeSandbox(suites={Side.BASE: suite(Side.BASE, base_case), Side.PATCH: suite(Side.PATCH, infra)})
    outcome = RegressionAttacker().attack(
        target(), attack(), arm=ARM, client=FakeClient(), sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.unchecked == () and outcome.repros == ()


# -- AT2 stage 1 ------------------------------------------------------------------------------------------------------

PASSING = "from pkg.parser import parse\n\n\ndef test_ok():\n    assert parse(1) == 2\n"
FAILING = "from pkg.parser import parse\n\n\ndef test_bad():\n    assert parse(1) == 1\n"
BROKEN_IMPORT = "import nothing_here\n\n\ndef test_import():\n    assert True\n"


def scripted(
    outcomes: Mapping[str, tuple[Outcome, FailureType | None]],
) -> Callable[[Mapping[str, str], Sequence[str]], RunResult]:
    def result(files: Mapping[str, str], tests: Sequence[str]) -> RunResult:
        (code,) = files.values()
        outcome, ftype = outcomes[code]
        return run(Side.PATCH, outcome, ftype, seconds=2.0)

    return result


def candidates(*items: tuple[str, str]) -> ModelResponse:
    return response({"tests": [{"path": path, "code": code, "why": "x"} for path, code in items]})


def test_at2_keeps_only_candidates_that_fail_on_the_patch_with_a_counted_failure() -> None:
    sandbox = FakeSandbox(
        run_test_result=scripted(
            {
                FAILING: (Outcome.FAIL, FailureType.ASSERTION),
                PASSING: (Outcome.PASS, None),
                BROKEN_IMPORT: (Outcome.ERROR, FailureType.IMPORT),
            }
        )
    )
    client = FakeClient(
        candidates(("tests/test_a.py", FAILING), ("tests/test_b.py", PASSING), ("tests/test_c.py", BROKEN_IMPORT))
    )
    outcome = CorrectnessAttacker().attack(
        target(),
        attack("fixed_x", lens=Lens.CORRECTNESS),
        arm=ARM,
        client=client,
        sandbox=sandbox,
        runner=PytestRunner(),
    )
    (repro,) = outcome.repros
    assert (repro.kind, repro.lens) == (CatchKind.FIX_VALIDITY, Lens.CORRECTNESS)
    assert list(repro.files) == ["tests/test_receipts_a1_0.py"]  # renamed to a name the attacker owns
    assert repro.command == ("tests/test_receipts_a1_0.py::test_bad",)
    # PLANT (contract v3, Repro.cases): the repro named no cases, so its reruns had to rebuild them from its files.
    assert repro.cases == ("tests/test_receipts_a1_0.py::test_bad",)
    (first,) = [c for c in sandbox.test_calls if "tests/test_receipts_a1_0.py" in c["files"]]  # runs overlap
    assert first["tests"] == list(repro.command) and first["expect"] == list(repro.cases)  # it ran as its repro
    assert (
        first["files"] == {"tests/test_receipts_a1_0.py": FAILING} and first["seed"] == 7 and first["timeout"] == 60.0
    )
    assert (
        len(sandbox.test_calls) == 3 and sandbox.prepared == [(Side.PATCH, None)] and sandbox.closed == ["ws-patch-1"]
    )
    assert [c.seconds for c in outcome.costs if c.stage == "at2_runs"] == [2.0, 2.0, 2.0]
    (request,) = client.requests
    assert (request.stage, request.model, request.temperature) == ("at2_stage1", "muse-spark-1.3", 0.8)


def test_at2_a_candidate_never_overwrites_a_repository_file_or_leaves_its_test_directory() -> None:
    existing = set(TREE) | {"tests/test_receipts_a1_0.py"}
    assert (
        placed(RunnerKind.PYTEST, "tests/test_parser.py", "test_receipts_a1_1.py", existing)
        == "tests/test_receipts_a1_1.py"
    )
    assert placed(RunnerKind.PYTEST, "tests/x.py", "test_receipts_a1_0.py", existing) is None  # the owned name exists
    for bad in ("/etc/x.py", "../x.py", "tests/../../x.py", ".git/hooks/x.py", "tests/x.txt"):
        assert placed(RunnerKind.PYTEST, bad, "test_receipts_a1_2.py", existing) is None, bad
    django_tree = {"tests/model_fields/tests.py"}
    assert placed(RunnerKind.DJANGO, "tests/model_fields/test_new.py", "test_receipts_a_0.py", django_tree) == (
        "tests/model_fields/test_receipts_a_0.py"
    )
    assert placed(RunnerKind.DJANGO, "tests/no_such_app/t.py", "test_receipts_a_0.py", django_tree) is None
    assert placed(RunnerKind.DJANGO, "django/t.py", "test_receipts_a_0.py", django_tree) is None
    assert placed(RunnerKind.SYMPY, "sympy/core/tests/t.py", "test_receipts_a_0.py", set()) == (
        "sympy/core/tests/test_receipts_a_0.py"
    )
    assert placed(RunnerKind.SYMPY, "sympy/core/t.py", "test_receipts_a_0.py", set()) is None


def test_at2_skips_a_candidate_that_does_not_parse_or_defines_no_test() -> None:
    sandbox = FakeSandbox(run_test_result=scripted({}))
    client = FakeClient(candidates(("tests/t.py", "def test_x(:\n"), ("tests/u.py", "x = 1\n"), ("/abs/t.py", FAILING)))
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.repros == () and sandbox.test_calls == []
    assert [u.what for u in outcome.unchecked] == ["a1#0", "a1#1", "a1#2"]
    assert all(u.reason is UncheckedReason.UNPARSED for u in outcome.unchecked)


@pytest.mark.parametrize(
    ("claim", "kind"),
    [
        ("fixed_x", CatchKind.FIX_VALIDITY),
        ("adds_behavior:0123456789", CatchKind.BEHAVIOR),
        ("nothing_else_changed", CatchKind.REGRESSION),
        ("tests_pass:0123456789", CatchKind.REGRESSION),
    ],
)
def test_at2_a_repros_kind_follows_its_claim(claim: str, kind: CatchKind) -> None:
    assert kind_for(claim) is kind


@pytest.mark.parametrize(
    ("answer", "reason"),
    [
        (
            ModelCallError("down", UncheckedReason.RUNTIME, cost=StageCost(stage="at2_stage1", micro_usd=9)),
            UncheckedReason.RUNTIME,
        ),
        (response({"tests": []}, finish="length"), UncheckedReason.UNPARSED),
        (response("", finish="stop"), UncheckedReason.UNPARSED),  # an empty answer is truncated (RT1)
    ],
)
def test_at2_a_model_failure_is_unchecked_never_raised(
    answer: ModelResponse | ModelCallError, reason: UncheckedReason
) -> None:
    sandbox = FakeSandbox(run_test_result=scripted({}))
    outcome = CorrectnessAttacker().attack(
        target(),
        attack(lens=Lens.CORRECTNESS),
        arm=ARM,
        client=FakeClient(answer),
        sandbox=sandbox,
        runner=PytestRunner(),
    )
    assert outcome.repros == () and outcome.unchecked[0].reason is reason and len(outcome.costs) == 1
    assert sandbox.closed == ["ws-patch-1"]


def test_at2_never_raises_for_a_sandbox_failure() -> None:
    sandbox = FakeSandbox(fail_runs=True)
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=FakeClient(candidates(("tests/t.py", FAILING))),
        sandbox=sandbox, runner=PytestRunner(),
    )  # fmt: skip
    assert outcome.unchecked[0].reason is UncheckedReason.RUNTIME and len(outcome.costs) == 1
    assert sandbox.closed == ["ws-patch-1"]
    no_env = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=FakeClient(), sandbox=FakeSandbox(fail_prepare=True),
        runner=PytestRunner(),
    )  # fmt: skip
    assert no_env.unchecked[0].reason is UncheckedReason.NO_ENV


@pytest.mark.parametrize(
    ("mask", "present", "absent"),
    [
        (ContextMask.DIFF, ["Diff:", "+    return x + 1"], ["changed functions"]),
        (ContextMask.DIFF_FUNCTIONS, ["changed functions", "def parse(x):"], []),
        (ContextMask.DIFF_TESTS, ["Existing tests over the changed files", "def test_a():"], []),
        (ContextMask.CLAIM_ONLY, ["work from the claim"], ["Diff:"]),
        (ContextMask.CLAIM_SPEC, ["terse specification"], ["Diff:"]),
    ],
)
def test_at2_the_mask_decides_the_code_context(mask: ContextMask, present: list[str], absent: list[str]) -> None:
    client = FakeClient(candidates(("tests/t.py", PASSING)))
    sandbox = FakeSandbox(run_test_result=scripted({PASSING: (Outcome.PASS, None)}))
    CorrectnessAttacker().attack(
        target(),
        attack(mask=mask, lens=Lens.CORRECTNESS),
        arm=ARM,
        client=client,
        sandbox=sandbox,
        runner=PytestRunner(),
    )
    prompt = client.requests[0].messages[1].content
    assert (
        "src/pkg/parser.py" in prompt and "the change breaks parse" in prompt
    )  # every mask names files and hypothesis
    for text in present:
        assert text in prompt, text
    for text in absent:
        assert text not in prompt, text


def test_at2_django_and_sympy_candidates_run_in_their_runners_forms() -> None:
    dj_tree = {"tests/model_fields/tests.py": "", "django/db/models/fields/__init__.py": ""}
    code = (
        "from django.test import SimpleTestCase\n\n\n"
        "class T(SimpleTestCase):\n    def test_x(self):\n        self.assertEqual(1, 2)\n"
    )
    sandbox = FakeSandbox(dj_tree, run_test_result=scripted({code: (Outcome.FAIL, FailureType.ASSERTION)}))
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM,
        client=FakeClient(candidates(("tests/model_fields/t.py", code))),
        sandbox=sandbox, runner=DjangoRunner(),
    )  # fmt: skip
    label = "model_fields.test_receipts_a1_0.T.test_x"
    assert sandbox.test_calls[0]["tests"] == [label] == sandbox.test_calls[0]["expect"]
    assert outcome.repros[0].command == (label,) == outcome.repros[0].cases
    sy_code = "def test_y():\n    assert 1 == 2\n"
    sandbox = FakeSandbox({}, run_test_result=scripted({sy_code: (Outcome.FAIL, FailureType.ASSERTION)}))
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM,
        client=FakeClient(candidates(("sympy/core/tests/t.py", sy_code))),
        sandbox=sandbox, runner=SympyRunner(),
    )  # fmt: skip
    assert sandbox.test_calls[0]["tests"] == ["sympy/core/tests/test_receipts_a1_0.py::test_y"]
    # sympy names a case by its function and its file: path::name answers for that file's case only.
    assert sandbox.test_calls[0]["expect"] == ["sympy/core/tests/test_receipts_a1_0.py::test_y"]
    assert outcome.repros[0].command == ("sympy/core/tests/test_receipts_a1_0.py::test_y",) == outcome.repros[0].cases


def test_at2_runs_its_candidates_at_once() -> None:
    # Each run waits for the other two; run one after another, the first would time out on the barrier.
    barrier = threading.Barrier(3, timeout=10)

    def together(files: Mapping[str, str], tests: Sequence[str]) -> RunResult:
        barrier.wait()
        return run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION)

    codes = [FAILING.replace("test_bad", f"test_bad_{i}") for i in range(3)]
    sandbox = FakeSandbox(run_test_result=together)
    client = FakeClient(candidates(*[(f"tests/t{i}.py", code) for i, code in enumerate(codes)]))
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    # Results come back in candidate order, whichever run finished first.
    assert [r.command for r in outcome.repros] == [(f"tests/test_receipts_a1_{i}.py::test_bad_{i}",) for i in range(3)]


@pytest.mark.parametrize("ftype", [FailureType.INFRA, None])
def test_at2_a_candidate_whose_run_could_not_be_read_is_unchecked_never_dropped(ftype: FailureType | None) -> None:
    # PLANT (review blocker): a candidate whose run came back ERROR with failure type INFRA (an expected case that
    # never ran, as with the sympy -k bug) or none disappeared with no record.
    def unreadable(files: Mapping[str, str], tests: Sequence[str]) -> RunResult:
        return RunResult(
            side=Side.PATCH, outcome=Outcome.ERROR, failure_type=ftype, exit_code=1, fresh_container=False,
            duration_s=2.0, stdout_tail="tests finished: 0 passed, 1 failed, in 0.01 seconds\nDO *NOT* COMMIT!",
        )  # fmt: skip

    sandbox = FakeSandbox(run_test_result=unreadable)
    client = FakeClient(candidates(("tests/test_a.py", FAILING), ("tests/test_b.py", PASSING)))
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome.repros == ()
    assert [(u.what, u.reason) for u in outcome.unchecked] == [
        ("a1#0", UncheckedReason.RUNTIME),
        ("a1#1", UncheckedReason.RUNTIME),
    ]
    # A sandbox that says nothing (no problem on the result) still leaves the exit code and the tails as the detail.
    assert "exit 1" in outcome.unchecked[0].detail and "DO *NOT* COMMIT!" in outcome.unchecked[0].detail


def test_at2_an_unreadable_candidates_detail_is_the_runs_own_problem() -> None:
    # PLANT (contract v3, RunResult.problem): the detail was rebuilt from the exit code and the output's tails,
    # whatever the sandbox knew about why the run could not be read.
    why = "expected case tests/test_receipts_a1_0.py::test_bad never ran, and the run reports no import error for its"

    def unreadable(files: Mapping[str, str], tests: Sequence[str]) -> RunResult:
        return RunResult(
            side=Side.PATCH, outcome=Outcome.ERROR, failure_type=FailureType.INFRA, exit_code=1, fresh_container=False,
            duration_s=2.0, stdout_tail="noise from the run", problem=why,
        )  # fmt: skip

    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=FakeClient(candidates(("tests/test_a.py", FAILING))),
        sandbox=FakeSandbox(run_test_result=unreadable), runner=PytestRunner(),
    )  # fmt: skip
    (item,) = outcome.unchecked
    assert (item.what, item.reason, item.detail) == ("a1#0", UncheckedReason.RUNTIME, f"{why} (exit 1)")


def test_at2_drops_a_candidate_that_failed_its_own_way_without_an_unchecked_item() -> None:
    # A candidate's own failure (an import error, a crash outside the change) is a result, not a gap in the check.
    sandbox = FakeSandbox(run_test_result=scripted({BROKEN_IMPORT: (Outcome.ERROR, FailureType.IMPORT)}))
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=FakeClient(candidates(("tests/t.py", BROKEN_IMPORT))),
        sandbox=sandbox, runner=PytestRunner(),
    )  # fmt: skip
    assert outcome.repros == () and outcome.unchecked == ()


_JUNIT = Path(__file__).parent / "fixtures" / "junit"


def test_at2_a_candidate_whose_own_file_does_not_import_is_dropped_as_its_import_error() -> None:
    # The same through the real classifier: pytest's recorded collection error (fixtures/junit/collect_import.xml),
    # for the candidate's own file. PLANT (the round-3 review's note): run_test answered a case that never ran ERROR
    # typed INFRA, so this candidate was listed as unchecked (runtime), and the test above scripted a result that
    # never occurred. Contract v3 types it IMPORT, and an import error is the candidate's own failure.
    owned = "tests/test_receipts_a1_0.py"
    report = (_JUNIT / "collect_import.xml").read_text().replace("tests/test_import.py", owned)
    report = report.replace('name="tests.test_import"', 'name="tests.test_receipts_a1_0"')

    def classified(files: Mapping[str, str], tests: Sequence[str]) -> RunResult:
        coarse = RunResult(side=Side.PATCH, outcome=Outcome.ERROR, exit_code=2, fresh_container=False, duration_s=1.0)
        run = classify_suite(coarse, runner=PytestRunner(), output="", report=report, changed_files=[], timed_out=False)
        return with_expect(run, tests)

    sandbox = FakeSandbox(run_test_result=classified)
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=FakeClient(candidates(("tests/t.py", BROKEN_IMPORT))),
        sandbox=sandbox, runner=PytestRunner(),
    )  # fmt: skip
    assert sandbox.test_calls[0]["expect"] == [f"{owned}::test_import"]
    assert outcome.repros == () and outcome.unchecked == ()


def test_at2_a_candidate_that_defines_a_test_twice_runs_it_once() -> None:
    # Python keeps the later definition, so the file has one case of that name. PLANT: the case ids repeated it, and
    # a repro naming a case twice is refused (contract v3), which threw the whole attack away as a runtime error.
    twice = FAILING + "\n\ndef test_bad():\n    assert parse(1) == 1\n"
    sandbox = FakeSandbox(run_test_result=scripted({twice: (Outcome.FAIL, FailureType.ASSERTION)}))
    outcome = CorrectnessAttacker().attack(
        target(), attack(lens=Lens.CORRECTNESS), arm=ARM, client=FakeClient(candidates(("tests/t.py", twice))),
        sandbox=sandbox, runner=PytestRunner(),
    )  # fmt: skip
    (repro,) = outcome.repros
    assert repro.cases == ("tests/test_receipts_a1_0.py::test_bad",) and outcome.unchecked == ()


def test_at2_at_most_three_candidates() -> None:
    four = candidates(*[(f"tests/t{i}.py", PASSING) for i in range(4)])
    sandbox = FakeSandbox(run_test_result=scripted({PASSING: (Outcome.PASS, None)}))
    outcome = CorrectnessAttacker().attack(
        target(),
        attack(lens=Lens.CORRECTNESS),
        arm=ARM,
        client=FakeClient(four, four),
        sandbox=sandbox,
        runner=PytestRunner(),
    )
    assert outcome.unchecked[0].reason is UncheckedReason.UNPARSED and sandbox.test_calls == []


def test_the_attackers_declare_their_lenses() -> None:
    assert RegressionAttacker.lens is Lens.REGRESSION and CorrectnessAttacker().lens is Lens.CORRECTNESS
    assert CorrectnessAttacker(lens=Lens.REGRESSION).lens is Lens.REGRESSION  # the regression lens's model attacks
    with pytest.raises(ValueError, match="correctness or the regression lens"):
        CorrectnessAttacker(lens=Lens.DIFFERENTIAL)


# -- pilot replays (10/28): Django placement --------------------------------------------------------------------------

DJ_PATCH = (
    "diff --git a/django/utils/datastructures.py b/django/utils/datastructures.py\n"
    "--- a/django/utils/datastructures.py\n+++ b/django/utils/datastructures.py\n"
    "@@ -1,2 +1,2 @@\n class OrderedSet:\n-    pass\n+    __reversed__ = None\n"
)
DJ_TREE = {
    "django/utils/datastructures.py": "class OrderedSet:\n    pass\n",
    "tests/basic/tests.py": "",
    "tests/basic/models.py": "",
    "tests/lookup/tests.py": "",
    "tests/utils_tests/test_datastructures.py": "from django.utils.datastructures import OrderedSet\n",
}
DJ_CODE = (
    "from django.test import SimpleTestCase\n\n\n"
    "class T(SimpleTestCase):\n    def test_x(self):\n        self.assertEqual(1, 2)\n"
)


def _dj_attack(path: str) -> tuple[FakeSandbox, AttackOutcome]:
    sandbox = FakeSandbox(DJ_TREE, run_test_result=scripted({DJ_CODE: (Outcome.FAIL, FailureType.ASSERTION)}))
    outcome = CorrectnessAttacker().attack(
        target(DJ_PATCH), attack("fixed_x", lens=Lens.CORRECTNESS), arm=ARM,
        client=FakeClient(candidates((path, DJ_CODE))), sandbox=sandbox, runner=DjangoRunner(),
    )  # fmt: skip
    return sandbox, outcome


def test_at2_a_django_candidate_below_a_test_module_of_the_same_name_runs_in_the_app() -> None:
    # Glimmer, django-14351: tests/basic/tests/<file>.py, where tests/basic/tests.py is a module, ran under the label
    # basic.tests.test_receipts_correctness_2_0... and never imported, so the item was unchecked (runtime).
    sandbox, outcome = _dj_attack("tests/basic/tests/test_related_in_or.py")
    assert sandbox.test_calls[0]["tests"] == ["basic.test_receipts_a1_0.T.test_x"]
    assert [list(r.files) for r in outcome.repros] == [["tests/basic/test_receipts_a1_0.py"]]
    assert outcome.unchecked == ()


def test_at2_a_django_candidate_for_an_app_that_does_not_exist_runs_where_the_tests_over_the_change_are() -> None:
    # Spark, django-14089: 'tests/ordered_set/test_reversed_empty.py' (and two more) was refused as "not a new test
    # file the runner collects"; OrderedSet's existing tests are in tests/utils_tests.
    sandbox, outcome = _dj_attack("tests/ordered_set/test_reversed_empty.py")
    assert sandbox.test_calls[0]["tests"] == ["utils_tests.test_receipts_a1_0.T.test_x"]
    assert [list(r.files) for r in outcome.repros] == [["tests/utils_tests/test_receipts_a1_0.py"]]
    assert outcome.unchecked == ()


def test_placed_keeps_a_django_package_below_the_app_and_needs_a_home_that_is_an_app() -> None:
    tree = {"tests/m/tests.py", "tests/m/sub/__init__.py", "tests/m/sub/tests.py"}
    assert (
        placed(RunnerKind.DJANGO, "tests/m/sub/t.py", "test_receipts_a_0.py", tree)
        == "tests/m/sub/test_receipts_a_0.py"
    )
    assert (
        placed(RunnerKind.DJANGO, "tests/m/nope/t.py", "test_receipts_a_0.py", tree) == "tests/m/test_receipts_a_0.py"
    )
    assert placed(RunnerKind.DJANGO, "tests/x/t.py", "test_receipts_a_0.py", tree, home="tests/m") == (
        "tests/m/test_receipts_a_0.py"
    )
    for home in ("tests/none", "django", "tests/m/sub", "tests/../m"):
        assert placed(RunnerKind.DJANGO, "tests/x/t.py", "test_receipts_a_0.py", tree, home=home) is None, home
    assert placed(RunnerKind.DJANGO, "django/t.py", "test_receipts_a_0.py", tree, home="tests/m") is None


class _Reasoner(FakeClient):
    """Reasons for ``needs`` output tokens before it answers: under a smaller max_tokens the answer is cut off
    (finish_reason length) with every token spent, as Spark's were."""

    def __init__(self, needs: int, answer: ModelResponse) -> None:
        super().__init__(answer)
        self.needs = needs

    def complete(self, request: ModelRequest) -> ModelResponse:
        answer = super().complete(request)
        if request.max_tokens < self.needs:
            usage = Usage(input_tokens=10, output_tokens=request.max_tokens, reasoning_tokens=request.max_tokens - 2)
            return answer.model_copy(update={"text": "", "finish_reason": "length", "usage": usage})
        return answer


@pytest.mark.parametrize("needs", [8_001, 12_000, 16_000])
def test_at2_a_stage_one_answer_past_eight_thousand_tokens_is_not_cut_off(needs: int) -> None:
    # Spark, sphinx-9591 correctness-1 and 13 more: 8,000 output tokens, 7,968 of them reasoning, finish_reason length.
    sandbox = FakeSandbox(run_test_result=scripted({FAILING: (Outcome.FAIL, FailureType.ASSERTION)}))
    client = _Reasoner(needs, candidates(("tests/test_a.py", FAILING)))
    outcome = CorrectnessAttacker().attack(
        target(), attack("fixed_x", lens=Lens.CORRECTNESS), arm=ARM, client=client, sandbox=sandbox,
        runner=PytestRunner(),
    )  # fmt: skip
    assert len(outcome.repros) == 1 and outcome.unchecked == ()


def test_at2_an_answer_that_is_not_json_is_retried_with_where_it_broke_and_how_to_escape() -> None:
    # Glimmer, 14 stage-1 items: "not JSON: Expecting ',' delimiter: line 1 column 1158", on the retry too. The
    # answer below breaks the same way: a bare double quote inside the code string.
    broken = '{"tests": [{"path": "tests/test_a.py", "code": "def test_bad():\\n    assert parse("1") == 1\\n"}]}'
    sandbox = FakeSandbox(run_test_result=scripted({FAILING: (Outcome.FAIL, FailureType.ASSERTION)}))
    client = FakeClient(response(broken), candidates(("tests/test_a.py", FAILING)))
    outcome = CorrectnessAttacker().attack(
        target(), attack("fixed_x", lens=Lens.CORRECTNESS), arm=ARM, client=client, sandbox=sandbox,
        runner=PytestRunner(),
    )  # fmt: skip
    assert len(outcome.repros) == 1
    retry = client.requests[1].messages[-1].content
    assert "Expecting ',' delimiter" in retry
    assert 'assert parse("' in retry  # the answer's own text up to the place it broke
    assert 'write a double quote as \\"' in retry


def test_at2_a_schema_error_is_retried_without_the_escaping_note() -> None:
    sandbox = FakeSandbox(run_test_result=scripted({FAILING: (Outcome.FAIL, FailureType.ASSERTION)}))
    client = FakeClient(response({"tests": []}), candidates(("tests/test_a.py", FAILING)))
    CorrectnessAttacker().attack(
        target(), attack("fixed_x", lens=Lens.CORRECTNESS), arm=ARM, client=client, sandbox=sandbox,
        runner=PytestRunner(),
    )  # fmt: skip
    retry = client.requests[1].messages[-1].content
    assert "tests:" in retry and "double quote" not in retry


# -- fix round 5 (the 10/28 pilot): the scope rule, the quote, and the regression lens's model attacks -------------

GLIMMER_ARM = Arm(name="glimmer", attacker_model="muse-glimmer-30b", step_limit=20, seed=7)


@pytest.mark.parametrize("arm", [ARM, GLIMMER_ARM], ids=["spark", "glimmer"])
def test_at2_every_arm_s_prompt_holds_the_scope_rule_and_asks_for_a_quote(arm: Arm) -> None:
    sandbox = FakeSandbox(run_test_result=scripted({FAILING: (Outcome.FAIL, FailureType.ASSERTION)}))
    client = FakeClient(candidates(("tests/test_a.py", FAILING)))
    CorrectnessAttacker().attack(
        target(), attack("fixed_x", lens=Lens.CORRECTNESS), arm=arm, client=client, sandbox=sandbox,
        runner=PytestRunner(),
    )  # fmt: skip
    (request,) = client.requests
    assert request.model == arm.attacker_model
    user = request.messages[1].content
    assert SCOPE_RULE in user and ONE_CASE in user and REGRESSION_RULE not in user
    assert "no sibling function" in SCOPE_RULE and "exist before the change" in SCOPE_RULE
    item = request.json_schema["$defs"]["_Candidate"]  # type: ignore[index]
    assert "quote" in item["properties"] and "quote" in item["required"]


@pytest.mark.parametrize("arm", [ARM, GLIMMER_ARM], ids=["spark", "glimmer"])
def test_at2_on_the_regression_lens_asks_for_behavior_base_had(arm: Arm) -> None:
    sandbox = FakeSandbox(run_test_result=scripted({FAILING: (Outcome.FAIL, FailureType.ASSERTION)}))
    client = FakeClient(candidates(("tests/test_a.py", FAILING)))
    outcome = CorrectnessAttacker(lens=Lens.REGRESSION).attack(
        target(), attack("fixed_x"), arm=arm, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    (request,) = client.requests
    system, user = (m.content for m in request.messages)
    assert REGRESSION_RULE in user and SCOPE_RULE not in user and "before the change" in system
    # A regression-lens repro is a regression catch whatever its claim, and quotes nothing: its oracle is base.
    (repro,) = outcome.repros
    assert (repro.lens, repro.kind, repro.quote) == (Lens.REGRESSION, CatchKind.REGRESSION, "")


def test_at2_carries_each_test_s_quote_onto_its_repro() -> None:
    sandbox = FakeSandbox(run_test_result=scripted({FAILING: (Outcome.FAIL, FailureType.ASSERTION)}))
    answer = {"tests": [{"path": "tests/test_a.py", "code": FAILING, "why": "x", "quote": "parse must return x"}]}
    outcome = CorrectnessAttacker().attack(
        target(), attack("fixed_x", lens=Lens.CORRECTNESS), arm=ARM, client=FakeClient(response(answer)),
        sandbox=sandbox, runner=PytestRunner(),
    )  # fmt: skip
    (repro,) = outcome.repros
    assert repro.quote == "parse must return x"


@pytest.mark.parametrize("lens", [Lens.CORRECTNESS, Lens.REGRESSION])
def test_at2_leaves_at3_s_attack_alone_and_refuses_another_lens(lens: Lens) -> None:
    sandbox = FakeSandbox(run_test_result=scripted({}))
    client = FakeClient()
    suite = attack(existing_tests=True)
    outcome = CorrectnessAttacker(lens=lens).attack(
        target(), suite, arm=ARM, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    assert outcome == AttackOutcome(attack_id="a1") and client.requests == [] and sandbox.prepared == []
    other = attack(lens=Lens.CORRECTNESS if lens is Lens.REGRESSION else Lens.REGRESSION)
    refused = CorrectnessAttacker(lens=lens).attack(
        target(), other, arm=ARM, client=client, sandbox=sandbox, runner=PytestRunner()
    )
    (item,) = refused.unchecked
    assert item.reason is UncheckedReason.RUNTIME and f"takes the {lens.value} lens" in item.detail
    assert client.requests == []
