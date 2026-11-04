"""SK1 on AT3's regression repros (round 4, the blocker from the round-3 review): an existing test file at its base
content, with Repro.cases naming the one case that regressed. Each file holds that case beside siblings that would
each refute the whole file: a skipped and an expected-to-fail test, a mock of the changed function, a call count, a
process, a socket, and a module's __file__. The skeptic expects and runs only the named case and reads only what it
uses, so a real regression reaches BLOCKER through 5 + 5 runs under each runner's own case ids.

The scripted sandbox runs what `tests` selects, the way each runner selects (a pytest node id, a Django label, sympy's
'path::name'), reports every case it ran under the runner's own name, and applies `expect` the way the attack lane's
reduce_cases does: a name matches exactly, a sympy 'path::name' matches that file's case, a skipped expected case
makes the run SKIPPED, and an expected case that never ran makes it an ERROR. It answers the skeptic's base-content
check (receipts.core.skeptic.SAME_AS_BASE) from the base tree it was given."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
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
    Finding,
    Lens,
    Outcome,
    Repro,
    RunnerKind,
    RunResult,
    Side,
    Span,
    SpanSource,
    Status,
    Target,
    Tier,
    UncheckedReason,
    Workspace,
)
from receipts.core.skeptic import SAME_AS_BASE, TargetSkeptic, base_check_command

DIGEST = "sha256:" + "a" * 64
FAIL, PASS, SKIP = Outcome.FAIL, Outcome.PASS, Outcome.SKIPPED


def patch_of(*paths: str) -> str:
    return "".join(
        f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n"
        "@@ -1,2 +1,2 @@ def receipts_join(items):\n"
        '-    return "".join(items)\n'
        '+    return ",".join(items)\n'
        for path in paths
    )


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
    hypothesis="the agent's change broke an existing test",
    mask=ContextMask.DIFF,
    temperature=0.0,
)

# ---------------------------------------------------------------- three repositories' existing test files

PYTEST_FILE = """\
import os

import pytest

from widgets import render


def test_joins():
    assert render.receipts_join(["a", "b"]) == "ab"


@pytest.mark.skip(reason="not on this platform")
def test_skipped():
    assert render.receipts_join([]) == ""


@pytest.mark.xfail(reason="a known bug", strict=True)
def test_known_bug():
    assert render.receipts_join(None) == ""


def test_mocks_the_changed_function(monkeypatch):
    monkeypatch.setattr(render, "receipts_join", lambda items: "")
    assert render.receipts_join(["x"]) == ""


def test_counts_calls(mocker):
    spy = mocker.spy(render, "helper")
    render.receipts_join(["a"])
    assert spy.call_count == 1


def test_runs_a_process():
    import subprocess

    assert subprocess.run(["true"]).returncode == 0


def test_opens_a_socket():
    import socket

    assert socket.gethostname()


def test_reads_a_module_file():
    assert os.path.exists(render.__file__)
"""

DJANGO_FILE = """\
import os
import unittest
from unittest import mock

import django
from django.test import SimpleTestCase
from django.utils import text


class TextTests(SimpleTestCase):
    def setUp(self):
        self.items = ["a", "b"]

    def test_joins(self):
        self.assertEqual(text.receipts_join(self.items), "ab")

    @unittest.skip("not on this platform")
    def test_skipped(self):
        self.assertEqual(text.receipts_join([]), "")

    @unittest.expectedFailure
    def test_known_bug(self):
        self.assertEqual(text.receipts_join(None), "")

    @mock.patch("django.utils.text.receipts_join")
    def test_mocks_the_changed_function(self, join):
        join.return_value = ""
        text.receipts_join([])
        join.assert_called_once_with([])

    def test_runs_a_process(self):
        import subprocess

        self.assertEqual(subprocess.run(["true"]).returncode, 0)

    def test_opens_a_socket(self):
        import socket

        self.assertTrue(socket.gethostname())

    def test_reads_a_module_file(self):
        self.assertTrue(os.path.dirname(django.__file__))
"""

SYMPY_FILE = """\
import os

from sympy.utilities import misc
from sympy.utilities.pytest import XFAIL, skip


def test_joins():
    assert misc.receipts_join(["a", "b"]) == "ab"


def test_skipped():
    skip("not on this platform")


@XFAIL
def test_known_bug():
    assert misc.receipts_join(None) == ""


def test_mocks_the_changed_function():
    from unittest import mock

    with mock.patch.object(misc, "receipts_join", return_value=""):
        assert misc.receipts_join([]) == ""


def test_runs_a_process():
    import subprocess

    assert subprocess.run(["true"]).returncode == 0


def test_opens_a_socket():
    import socket

    assert socket.gethostname()


def test_reads_a_module_file():
    assert os.path.exists(misc.__file__)
"""

SIBLINGS = (
    "test_skipped",
    "test_known_bug",
    "test_mocks_the_changed_function",
    "test_counts_calls",
    "test_runs_a_process",
    "test_opens_a_socket",
    "test_reads_a_module_file",
)


@dataclass(frozen=True)
class Kind:
    runner: RunnerKind
    repo: str
    changed: str
    test_path: str
    source: str
    regressed: str  # the regressed case's id, as AT3 names it (the runner's own id)
    names: Mapping[str, str]  # each test function, to the name the runner reports it under

    def target(self, *also_changed: str) -> Target:
        return Target(
            id=f"verified~{self.repo.replace('/', '__')}-7",
            tier=Tier.A,
            repo=self.repo,
            base_sha="b" * 40,
            patch=patch_of(self.changed, *also_changed),
            issue="joining a list puts commas between the items",
            test_runner=self.runner,
        )

    def repro(self, source: str | None = None, *, cases: Sequence[str] | None = None, path: str | None = None) -> Repro:
        where = path or self.test_path
        return Repro(
            id="regression-1-0-abcdef12",
            attack_id=ATTACK.id,
            claim_id=CLAIM.id,
            lens=Lens.REGRESSION,
            kind=CatchKind.REGRESSION,
            files={where: self.source if source is None else source},
            command=(self.regressed,),
            cases=tuple(cases) if cases is not None else (self.regressed,),
        )


def _functions(source: str) -> list[str]:
    return [line.split("(", 1)[0].split()[-1] for line in source.splitlines() if line.lstrip().startswith("def test")]


KINDS = {
    "pytest": Kind(
        runner=RunnerKind.PYTEST,
        repo="octo/widgets",
        changed="src/widgets/render.py",
        test_path="tests/test_render.py",
        source=PYTEST_FILE,
        regressed="tests/test_render.py::test_joins",
        names={name: f"tests/test_render.py::{name}" for name in _functions(PYTEST_FILE)},
    ),
    "django": Kind(
        runner=RunnerKind.DJANGO,
        repo="django/django",
        changed="django/utils/text.py",
        test_path="tests/utils_tests/test_text.py",
        source=DJANGO_FILE,
        regressed="utils_tests.test_text.TextTests.test_joins",
        names={name: f"utils_tests.test_text.TextTests.{name}" for name in _functions(DJANGO_FILE)},
    ),
    "sympy": Kind(
        runner=RunnerKind.SYMPY,
        repo="sympy/sympy",
        changed="sympy/utilities/misc.py",
        test_path="sympy/utilities/tests/test_misc.py",
        source=SYMPY_FILE,
        regressed="sympy/utilities/tests/test_misc.py::test_joins",  # AT3's form; the run reports 'test_joins'
        names={name: name for name in _functions(SYMPY_FILE)},
    ),
}


def outcome_of(function: str, side: Side) -> Outcome:
    """The regressed case fails on the patch and passes on base; skipped and expected-to-fail tests never run; every
    other sibling passes on both sides."""
    if function == "test_joins":
        return FAIL if side is Side.PATCH else PASS
    return SKIP if function in ("test_skipped", "test_known_bug") else PASS


class ScriptedRunner:
    def __init__(self, kind: RunnerKind) -> None:
        self.kind = kind

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["run", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        raise AssertionError("the scripted sandbox reads the cases itself")


def _matches(case: CaseResult, wanted: str) -> bool:
    """The attack lane's matcher (classify.expected): the exact name; for a case that names its file (sympy),
    'path::name'; for a pytest node id, one of its parametrizations."""
    if case.name == wanted:
        return True
    if "::" not in wanted:
        return False
    if case.file is not None:
        return wanted == f"{case.file}::{case.name}"
    return "[" not in wanted and case.name.startswith(wanted + "[")


@dataclass
class At3Sandbox:
    kind: Kind
    base_tree: dict[str, str]  # the repository's files at base, which the base-content check reads
    runs: list[dict[str, Any]] = field(default_factory=list)
    checks: list[tuple[str, ...]] = field(default_factory=list)
    prepared: list[Side] = field(default_factory=list)
    check_fails: bool = False

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        self.prepared.append(side)
        return Workspace(id=f"ws-{side.value}", side=side, root="/testbed", image_digest=DIGEST)

    def run(self, workspace: Workspace, command: Sequence[str], **options: Any) -> RunResult:
        """Only the skeptic's base-content check comes here: the script, run on the base tree's text."""
        self.checks.append(tuple(command))
        assert workspace.side is Side.BASE and list(command[:4]) == ["python", "-I", "-c", SAME_AS_BASE]
        assert options["timeout_s"] > 0 and not options.get("fresh", False)
        common: dict[str, Any] = {"side": Side.BASE, "fresh_container": False, "duration_s": 0.1}
        if self.check_fails:
            return RunResult(outcome=Outcome.ERROR, exit_code=125, problem="docker exec failed", **common)
        path, digest = command[4].removeprefix("./"), command[5]
        if path not in self.base_tree:
            return RunResult(outcome=Outcome.ERROR, exit_code=4, **common)
        text = self.base_tree[path].replace("\r\n", "\n").replace("\r", "\n")  # universal newlines, as open() reads
        same = hashlib.sha256(text.encode()).hexdigest() == digest
        return RunResult(outcome=PASS if same else Outcome.ERROR, exit_code=0 if same else 3, **common)

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
        assert fresh and expect and files is not None and runner.kind is self.kind.runner
        ran = self._selected(tests, files, workspace.side)
        self.runs.append({"side": workspace.side, "seed": seed, "tests": tuple(tests), "expect": tuple(expect)})
        common: dict[str, Any] = {"side": workspace.side, "seed": seed, "fresh_container": True, "duration_s": 1.0}
        chosen = [case for case in ran if any(_matches(case, wanted) for wanted in expect)]
        missing = [wanted for wanted in expect if not any(_matches(case, wanted) for case in chosen)]
        if missing:
            return RunResult(
                outcome=Outcome.ERROR, failure_type=FailureType.INFRA, problem=f"{missing} never ran", **common
            )
        if any(case.outcome is FAIL for case in chosen):
            return RunResult(outcome=FAIL, failure_type=FailureType.ASSERTION, **common)
        if any(case.outcome is SKIP for case in chosen):
            return RunResult(outcome=SKIP, **common)
        return RunResult(outcome=PASS, **common)

    def _selected(self, tests: Sequence[str], files: Mapping[str, str], side: Side) -> list[CaseResult]:
        """The cases a run of `tests` reports, each under the runner's own name, with its outcome on `side`."""
        everything = [
            CaseResult(
                name=self.kind.names[function],
                outcome=outcome_of(function, side),
                failure_type=FailureType.ASSERTION if outcome_of(function, side) is FAIL else None,
                file=self.kind.test_path if self.kind.runner is RunnerKind.SYMPY else None,
            )
            for function in _functions(files[self.kind.test_path])
        ]
        picked: list[CaseResult] = []
        for test in tests:
            for case in everything:
                if self.kind.runner is RunnerKind.DJANGO:
                    hit = case.name == test or case.name.startswith(test + ".")
                elif self.kind.runner is RunnerKind.SYMPY:
                    path, sep, name = test.partition("::")
                    hit = case.file == path and (not sep or case.name == name)
                else:
                    hit = test in (self.kind.test_path, case.name) or case.name.startswith(test + "[")
                if hit and case not in picked:
                    picked.append(case)
        return picked

    def close(self, workspace: Workspace) -> None:
        pass


def judge(
    kind: Kind,
    candidate: Repro,
    *,
    base_tree: dict[str, str] | None = None,
    target: Target | None = None,
    check_fails: bool = False,
) -> tuple[Finding, At3Sandbox]:
    """SK1's verdict on one candidate: the screen, then the reruns. The base tree defaults to the kind's own file."""
    tree = {kind.test_path: kind.source} if base_tree is None else base_tree
    sandbox = At3Sandbox(kind, tree, check_fails=check_fails)
    target = target or kind.target()
    skeptic = TargetSkeptic(target, sandbox=sandbox, runner=ScriptedRunner(kind.runner))
    finding = skeptic.judge(
        skeptic.screen(candidate, ATTACK),
        claim=CLAIM,
        finding_id=f"{target.id}/{CLAIM.id}/{ATTACK.id}.1",
        wall=lambda: 4.0,
    )
    skeptic.close()
    return finding, sandbox


def five_and_five(sandbox: At3Sandbox) -> list[tuple[Side, int | None]]:
    return [(run["side"], run["seed"]) for run in sandbox.runs]


FIVE_AND_FIVE = [(Side.PATCH, s) for s in range(5)] + [(Side.BASE, s) for s in range(5)]


# ---------------------------------------------------------------- 1 and 2: the named case, run and read alone


@pytest.mark.parametrize("name", list(KINDS), ids=list(KINDS))
def test_an_at3_repro_beside_siblings_that_would_each_refute_it_is_a_blocker(name: str) -> None:
    kind = KINDS[name]
    finding, box = judge(kind, kind.repro())
    assert finding.status is Status.BLOCKER, finding.detail
    assert five_and_five(box) == FIVE_AND_FIVE
    # Every run is pointed at the named case only, and expects exactly it.
    assert {run["expect"] for run in box.runs} == {(kind.regressed,)}
    assert {run["tests"] for run in box.runs} == {(kind.regressed,)}
    assert finding.refutation is None and finding.detail is None


@pytest.mark.parametrize("name", list(KINDS), ids=list(KINDS))
def test_the_same_file_with_no_cases_named_is_read_and_run_whole(name: str) -> None:
    # Repro.cases empty means every case the file defines (contract v3): the siblings' violations refute it, as before.
    kind = KINDS[name]
    finding, box = judge(kind, kind.repro(cases=()))
    assert finding.status is Status.REFUTED
    assert "SK1 fake-test rules" in (finding.detail or "")
    assert box.runs == []


@pytest.mark.parametrize("name", list(KINDS), ids=list(KINDS))
@pytest.mark.parametrize("sibling", ["test_skipped", "test_known_bug"])
def test_a_skipped_or_expected_to_fail_sibling_never_makes_the_base_run_skipped(name: str, sibling: str) -> None:
    # Only the named case is expected, so a sibling that never runs cannot turn a base pass into a skip. Here the
    # file holds the regressed case and that one sibling, so no fake-test rule is in play.
    kind = KINDS[name]
    source = _keep(kind.source, ["test_joins", sibling])
    finding, box = judge(kind, kind.repro(source), base_tree={kind.test_path: source})
    assert finding.status is Status.BLOCKER, finding.detail
    assert five_and_five(box) == FIVE_AND_FIVE


FAKE_SIBLINGS = [
    (name, sibling) for name, kind in KINDS.items() for sibling in SIBLINGS[2:] if sibling in _functions(kind.source)
]


@pytest.mark.parametrize(("name", "sibling"), FAKE_SIBLINGS, ids=[f"{n}-{s}" for n, s in FAKE_SIBLINGS])
def test_a_sibling_s_fake_test_violation_does_not_refute_the_named_case(name: str, sibling: str) -> None:
    kind = KINDS[name]
    source = _keep(kind.source, ["test_joins", sibling])
    finding, box = judge(kind, kind.repro(source), base_tree={kind.test_path: source})
    assert finding.status is Status.BLOCKER, finding.detail
    assert five_and_five(box) == FIVE_AND_FIVE
    # Named instead, the sibling is read, and refuted.
    named = kind.names[sibling] if kind.runner is not RunnerKind.SYMPY else f"{kind.test_path}::{sibling}"
    own, _ = judge(kind, kind.repro(source, cases=[named]), base_tree={kind.test_path: source})
    assert own.status is Status.REFUTED and "SK1 fake-test rules" in (own.detail or ""), own.detail


def test_a_bare_sympy_name_runs_as_path_and_name_and_expects_the_bare_name() -> None:
    kind = KINDS["sympy"]
    finding, box = judge(kind, kind.repro(cases=["test_joins"]))
    assert finding.status is Status.BLOCKER, finding.detail
    assert {run["tests"] for run in box.runs} == {(f"{kind.test_path}::test_joins",)}
    assert {run["expect"] for run in box.runs} == {("test_joins",)}


def test_a_django_case_in_swe_bench_s_form_runs_as_its_label() -> None:
    kind = KINDS["django"]
    finding, box = judge(kind, kind.repro(cases=["test_joins (utils_tests.test_text.TextTests)"]))
    assert finding.status is Status.BLOCKER, finding.detail
    assert {(run["tests"], run["expect"]) for run in box.runs} == {((kind.regressed,), (kind.regressed,))}


@pytest.mark.parametrize(
    ("name", "case"),
    [
        ("pytest", "tests/test_other.py::test_joins"),  # a file the repro does not hold
        ("pytest", "tests/test_render.py::test_never_written"),
        ("django", "other_app.tests.TextTests.test_joins"),
        ("sympy", "test_never_written"),
        ("sympy", "sympy/core/tests/test_basic.py::test_joins"),
    ],
)
def test_a_case_none_of_the_repro_s_files_defines_is_refuted_before_any_run(name: str, case: str) -> None:
    kind = KINDS[name]
    finding, box = judge(kind, kind.repro(_keep(kind.source, ["test_joins"]), cases=[case]))
    if "test_never_written" in case and kind.runner is RunnerKind.PYTEST:
        # A pytest node id names its file; a function the file does not define is read with the whole file, and the
        # run, which never reports it, cannot pass.
        assert finding.status is Status.UNCHECKED and box.runs and box.runs[0]["expect"] == (case,)
        return
    assert finding.status is Status.REFUTED
    assert "names cases none of its files defines" in (finding.detail or "")
    assert (box.runs, box.prepared) == ([], [])


# ---------------------------------------------------------------- what a named case uses is still read

USES = {
    "its-own-mock": "from unittest import mock\n\nfrom widgets import render\n\n\n"
    "def test_joins():\n    with mock.patch.object(render, 'receipts_join', return_value='ab'):\n"
    "        assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "setUp": "from unittest import TestCase, mock\n\nfrom widgets import render\n\n\nclass JoinTests(TestCase):\n"
    "    def setUp(self):\n        mock.patch('widgets.render.receipts_join').start()\n\n"
    "    def test_joins(self):\n        assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "setUpClass": "import subprocess\nfrom unittest import TestCase\n\nfrom widgets import render\n\n\n"
    "class JoinTests(TestCase):\n    @classmethod\n    def setUpClass(cls):\n        subprocess.run(['true'])\n\n"
    "    def test_joins(self):\n        assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "tearDown": "from unittest import TestCase\n\nfrom widgets import render\n\n\nclass JoinTests(TestCase):\n"
    "    def tearDown(self):\n        assert render.__file__\n\n"
    "    def test_joins(self):\n        assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "tearDownClass": "from unittest import TestCase\n\nfrom widgets import render\n\n\nclass JoinTests(TestCase):\n"
    "    @classmethod\n    def tearDownClass(cls):\n        open('src/widgets/render.py').read()\n\n"
    "    def test_joins(self):\n        assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "a-base-class-setUp": "from unittest import TestCase, mock\n\nfrom widgets import render\n\n\n"
    "class Base(TestCase):\n    def setUp(self):\n        mock.patch('widgets.render.receipts_join').start()\n\n\n"
    "class JoinTests(Base):\n    def test_joins(self):\n        assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "a-class-attribute": "from unittest import TestCase\n\nfrom widgets import render\n\n\nclass JoinTests(TestCase):\n"
    "    source = open('src/widgets/render.py').read()\n\n"
    "    def test_joins(self):\n        assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "a-requested-fixture": "import pytest\n\nfrom widgets import render\n\n\n@pytest.fixture\n"
    "def joined(monkeypatch):\n    monkeypatch.setattr(render, 'receipts_join', lambda items: 'ab')\n\n\n"
    "def test_joins(joined):\n    assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "a-fixture-of-a-fixture": "import pytest\n\nfrom widgets import render\n\n\n@pytest.fixture\n"
    "def patched(monkeypatch):\n    monkeypatch.setattr(render, 'receipts_join', lambda items: 'ab')\n\n\n"
    "@pytest.fixture\ndef joined(patched):\n    return 'ab'\n\n\n"
    "def test_joins(joined):\n    assert render.receipts_join(['a', 'b']) == joined\n",
    "usefixtures": "import pytest\n\nfrom widgets import render\n\n\n@pytest.fixture\n"
    "def patched(monkeypatch):\n    monkeypatch.setattr(render, 'receipts_join', lambda items: 'ab')\n\n\n"
    "@pytest.mark.usefixtures('patched')\ndef test_joins():\n    assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "an-autouse-fixture": "import pytest\n\nfrom widgets import render\n\n\n@pytest.fixture(autouse=True)\n"
    "def patched(monkeypatch):\n    monkeypatch.setattr(render, 'receipts_join', lambda items: 'ab')\n\n\n"
    "def test_joins():\n    assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "a-module-hook": "import subprocess\n\nfrom widgets import render\n\n\n"
    "def setup_module(module):\n    subprocess.run(['true'])\n\n\n"
    "def test_joins():\n    assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "a-helper-it-calls": "from widgets import render\n\n\n"
    "def _source():\n    return open('src/widgets/render.py').read()\n\n\n"
    "def test_joins():\n    assert ',' not in _source()\n",
    "a-helper-of-a-helper": "from unittest import mock\n\nfrom widgets import render\n\n\n"
    "def _patch():\n    mock.patch('widgets.render.receipts_join').start()\n\n\n"
    "def _arrange():\n    _patch()\n\n\n"
    "def test_joins():\n    _arrange()\n    assert render.receipts_join(['a', 'b']) == 'ab'\n",
    "a-module-constant": "from widgets import render\n\nSOURCE = open('src/widgets/render.py').read()\n\n\n"
    "def test_joins():\n    assert ',' not in SOURCE\n",
}


@pytest.mark.parametrize("source", list(USES.values()), ids=list(USES))
def test_what_the_named_case_uses_is_read_even_in_the_repository_s_own_file(source: str) -> None:
    # The named case is test_joins, in its class when it has one; the file is at its base content, so only what the
    # case uses counts, and each of these uses a fake-test pattern.
    kind = KINDS["pytest"]
    case = "tests/test_render.py::JoinTests::test_joins" if "class JoinTests" in source else kind.regressed
    finding, box = judge(kind, kind.repro(source, cases=[case]), base_tree={kind.test_path: source})
    assert finding.status is Status.REFUTED, finding.detail
    assert "SK1 fake-test rules" in (finding.detail or "")
    assert box.runs == []


# ---------------------------------------------------------------- 3: the repository's own test, at its base content


def with_module_code(source: str, code: str) -> str:
    """The file with module-level code the named case does not use, after its imports."""
    head, sep, rest = source.partition("\n\n\n")
    return f"{head}\n{code}{sep}{rest}"


@pytest.mark.parametrize("name", list(KINDS), ids=list(KINDS))
def test_module_level_code_the_named_case_does_not_use_counts_only_in_what_the_repro_wrote(name: str) -> None:
    # Django's admin_scripts tests import socket and subprocess at module level. In the repository's own file, at its
    # base content, that is not the repro's doing: one check in the base workspace says so, and the case runs.
    kind = KINDS[name]
    source = with_module_code(_keep(kind.source, ["test_joins"]), "import socket\nimport subprocess\n")
    own, box = judge(kind, kind.repro(source), base_tree={kind.test_path: source})
    assert own.status is Status.BLOCKER, own.detail
    assert len(box.checks) == 1 and box.checks[0][4:] == (
        f"./{kind.test_path}",
        hashlib.sha256(source.encode()).hexdigest(),
    )
    # The same text where the base tree differs (the repro wrote it): its module-level imports are read, and refuted.
    written, box = judge(kind, kind.repro(source), base_tree={kind.test_path: _keep(kind.source, ["test_joins"])})
    assert written.status is Status.REFUTED and "imports socket" in (written.detail or ""), written.detail
    assert box.runs == [] and len(box.checks) == 1
    # A file the base tree does not have at all is the repro's own writing too.
    new, _ = judge(kind, kind.repro(source), base_tree={})
    assert new.status is Status.REFUTED and "imports subprocess" in (new.detail or "")


@pytest.mark.parametrize("name", list(KINDS), ids=list(KINDS))
def test_an_edited_test_file_written_back_at_its_base_content_is_no_violation(name: str) -> None:
    # The agent edited the test file too. AT3 writes the base version back on purpose, so the agent's edit cannot
    # hide the regression: at base content that is the repository's own test, never mocks_changed.
    kind = KINDS[name]
    target = kind.target(kind.test_path)
    assert kind.test_path in target.patch
    finding, box = judge(kind, kind.repro(), target=target)
    assert finding.status is Status.BLOCKER, finding.detail
    assert five_and_five(box) == FIVE_AND_FIVE
    # Any other text at that path is the repro writing over a file the patch changed.
    altered = kind.source.replace('"ab"', '"a" "b"')
    refuted, box = judge(kind, kind.repro(altered), target=target)
    assert refuted.status is Status.REFUTED and "writes over a file the patch changed" in (refuted.detail or "")
    assert box.runs == []


def test_only_a_test_file_is_ever_the_repository_s_own() -> None:
    # Writing a changed source file back at its base content mixes the patch with base code; that stays refuted
    # whatever its text, and no base check is made for it.
    kind = KINDS["pytest"]
    source = _keep(kind.source, ["test_joins"])
    candidate = Repro.model_validate(
        {
            **kind.repro(source).model_dump(),
            "files": {kind.test_path: source, kind.changed: "def receipts_join(items):\n    return ''.join(items)\n"},
        }
    )
    finding, box = judge(
        kind, candidate, base_tree={kind.test_path: source, kind.changed: candidate.files[kind.changed]}
    )
    assert finding.status is Status.REFUTED and "writes over a file the patch changed" in (finding.detail or "")
    assert box.checks == [] and box.prepared == []


def test_a_helper_file_the_repro_brings_is_read_whole_even_at_its_base_content() -> None:
    # Only a test file the runner runs can be the repository's own: a helper module beside it is support code, read
    # whole, and its module-level import of subprocess refutes the repro with no base check made.
    kind = KINDS["pytest"]
    source = _keep(kind.source, ["test_joins"])
    helper = "import subprocess\n\n\ndef build():\n    return 1\n"
    candidate = Repro.model_validate(
        {**kind.repro(source).model_dump(), "files": {kind.test_path: source, "tests/helpers.py": helper}}
    )
    tree = {kind.test_path: source, "tests/helpers.py": helper}
    finding, box = judge(kind, candidate, base_tree=tree)
    assert finding.status is Status.REFUTED and "tests/helpers.py:1: imports subprocess" in (finding.detail or "")
    assert (box.checks, box.prepared) == ([], [])


def test_no_base_check_is_made_when_a_violation_stands_anyway() -> None:
    # The named case itself reads the changed code: refuted on its syntax, with no workspace built, though the other
    # test file it brings would need a base check (its module-level import counts only if the repro wrote it).
    kind = KINDS["pytest"]
    source = "def test_joins():\n    assert ',' not in open('src/widgets/render.py').read()\n"
    other = "import subprocess\n\n\ndef test_other():\n    pass\n"
    candidate = Repro.model_validate(
        {**kind.repro(source).model_dump(), "files": {kind.test_path: source, "tests/test_other.py": other}}
    )
    finding, box = judge(kind, candidate, base_tree={kind.test_path: source, "tests/test_other.py": other})
    assert finding.status is Status.REFUTED and "reflection" in (finding.detail or "")
    assert "imports subprocess" not in (finding.detail or "")
    assert (box.checks, box.prepared, box.runs) == ([], [], [])


def test_a_base_check_that_cannot_run_leaves_the_candidate_unchecked() -> None:
    kind = KINDS["pytest"]
    source = with_module_code(_keep(kind.source, ["test_joins"]), "import socket\n")
    finding, box = judge(kind, kind.repro(source), base_tree={kind.test_path: source}, check_fails=True)
    assert (finding.status, finding.unchecked_reason) == (Status.UNCHECKED, UncheckedReason.RUNTIME)
    assert "could not tell whether tests/test_render.py is the repository's own test" in (finding.detail or "")
    assert "docker exec failed" in (finding.detail or "")
    assert box.runs == []


def test_the_base_check_is_made_once_per_file_and_text() -> None:
    kind = KINDS["pytest"]
    source = with_module_code(_keep(kind.source, ["test_joins"]), "import socket\n")
    sandbox = At3Sandbox(kind, {kind.test_path: source})
    target = kind.target()
    skeptic = TargetSkeptic(target, sandbox=sandbox, runner=ScriptedRunner(kind.runner))
    for _ in range(2):
        screened = skeptic.screen(kind.repro(source), ATTACK)
        assert screened.failed_first_patch_run
    assert len(sandbox.checks) == 1
    skeptic.close()


def test_the_base_check_script_reads_the_file_the_way_the_attack_lane_does(tmp_path: Path) -> None:
    # The real script, in a real Python (isolated, as in the workspace): 0 for the same text, also through CRLF line
    # ends, which reading as text turns into LF; 3 for other text; 4 for no such file. A hashlib.py beside the file,
    # as a repository could ship, is never imported.
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_x.py").write_bytes(b"def test_x():\r\n    assert 1\r\n")
    (tmp_path / "hashlib.py").write_text("raise SystemExit(0)\n")
    text = "def test_x():\n    assert 1\n"

    def check(path: str, content: str) -> int:
        command = base_check_command(path, hashlib.sha256(content.encode()).hexdigest())
        assert command[0] == "python"  # the repository's own interpreter in the workspace; here, this one
        return subprocess.run([sys.executable, *command[1:]], cwd=tmp_path, capture_output=True, check=False).returncode

    assert check("tests/test_x.py", text) == 0
    assert check("tests/test_x.py", text + "\n") == 3
    assert check("tests/test_missing.py", text) == 4


# ---------------------------------------------------------------- the reviewer's real files, read as AT3 repros


def _keep(source: str, functions: Sequence[str]) -> str:
    """The file with only the named test functions (and everything that is not a test)."""
    import ast

    module = ast.parse(source)
    lines = source.splitlines(keepends=True)
    drop: list[tuple[int, int]] = []
    for node in ast.walk(module):
        if isinstance(node, ast.FunctionDef) and node.name.startswith("test") and node.name not in functions:
            start = min([node.lineno, *(d.lineno for d in node.decorator_list)])
            assert node.end_lineno is not None
            drop.append((start, node.end_lineno))
    kept = [line for number, line in enumerate(lines, start=1) if not any(a <= number <= b for a, b in drop)]
    return "".join(kept)
