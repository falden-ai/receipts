"""R24 on the run phase's real path: gold plus the test patch prepared through contracts.Sandbox, a kernel probe, then
run_test with the instance's SuiteRunner on the harness's test files, expecting FAIL_TO_PASS plus PASS_TO_PASS in the
runner's own case names. Every outcome maps to a verdict; INFRA is never a drop; no instance stops the run; each
worker has its own sandbox; a rerun replaces only INFRA, and only results that ran the pinned image merge. The sandbox
and runners are fakes."""

from __future__ import annotations

import json
import shlex
import subprocess
import sys
import threading
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from receipts import contracts
from receipts.contracts import (
    CaseResult,
    FailureType,
    Outcome,
    PatchMethod,
    RunnerKind,
    RunResult,
    Side,
    Target,
    Tier,
    Workspace,
    answers,
)
from receipts.eval import r24
from receipts.eval.r24 import (
    DERIVED_SECONDS_PER_CASE,
    DERIVED_TIMEOUT_CAP_S,
    DOCSTRINGS,
    MAX_DROP_PCT,
    GoldDrop,
    GoldRunner,
    R24Attempt,
    R24Report,
    R24Result,
    R24ResultsError,
    R24Verdict,
    case_names,
    check_instance,
    django_label,
    django_test_files,
    gold_and_tests,
    gold_target,
    merge,
    naming_before_run,
    naming_lines,
    parse_docstring_labels,
    parse_drops,
    run_check,
    run_timeout,
    suite_modules,
    verdict_of,
)
from receipts.eval.rows import Row
from receipts.eval.runsc_check import GVISOR_BANNER, RuntimeMismatchError
from receipts.eval.specs import LOG_PARSERS, RepoSpec

IMAGE = "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1@sha256:" + "e" * 64
GOLD_PATCH = "diff --git a/django/x.py b/django/x.py\n--- a/django/x.py\n+++ b/django/x.py\n@@ -1 +1 @@\n-a\n+b\n"
TEST_PATCH = (
    "diff --git a/tests/app/tests.py b/tests/app/tests.py\n--- a/tests/app/tests.py\n+++ b/tests/app/tests.py\n"
    "@@ -1 +1 @@\n-x\n+y\n"
    "diff --git a/tests/app/data.json b/tests/app/data.json\n--- a/tests/app/data.json\n+++ b/tests/app/data.json\n"
    "@@ -1 +1 @@\n-{}\n+[]\n"
)
# django__django-10097's shape: the test patch touches only .txt fixtures, so the harness ran the whole suite
TXT_ONLY = (
    "diff --git a/tests/validators/invalid_urls.txt b/tests/validators/invalid_urls.txt\n"
    "--- a/tests/validators/invalid_urls.txt\n+++ b/tests/validators/invalid_urls.txt\n@@ -1 +1,2 @@\n x\n+y\n"
)
SPEC = RepoSpec(
    repo="django/django",
    version="3.0",
    test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    runner=RunnerKind.DJANGO,
)
SPECS = {("django/django", "3.0"): SPEC}
LABELS = ("app.tests.T.test_fix", "app.tests.T.test_old", "app.tests.T.test_other")


def row(iid: str = "django__django-1", **kw: object) -> Row:
    fields: dict[str, object] = {
        "FAIL_TO_PASS": ("test_fix (app.tests.T)",),
        "PASS_TO_PASS": ("test_old (app.tests.T)", "test_other (app.tests.T)"),
        "base_commit": "a" * 40,
        "created_at": "2020-01-01T00:00:00Z",
        "difficulty": "<15 min fix",
        "environment_setup_commit": "b" * 40,
        "hints_text": "",
        "instance_id": iid,
        "patch": GOLD_PATCH,
        "problem_statement": "It breaks.",
        "repo": "django/django",
        "test_patch": TEST_PATCH,
        "version": "3.0",
    }
    return Row.model_validate(fields | kw)


def result(outcome: Outcome, side: Side = Side.GOLD, **kw: object) -> RunResult:
    failure = FailureType.ASSERTION if outcome is Outcome.FAIL else None
    defaults: dict[str, object] = {"failure_type": failure, "exit_code": 0 if outcome is Outcome.PASS else 1}
    return RunResult.model_validate(
        {"side": side, "outcome": outcome, "fresh_container": True, "duration_s": 1.5, **defaults, **kw}
    )


def error(ftype: FailureType | None) -> RunResult:
    return result(Outcome.ERROR, failure_type=ftype)


GVISOR = result(Outcome.PASS, stdout_tail=f"[    0.000000] {GVISOR_BANNER}\n[    0.4] Ready!\n")
HOST_DENIED = result(
    Outcome.ERROR, exit_code=1, stderr_tail="dmesg: read kernel buffer failed: Operation not permitted"
)
HOST_READ = result(Outcome.PASS, stdout_tail="[0.0] Linux version 6.8.0\n")


class FakeRunner:
    def __init__(self, kind: RunnerKind) -> None:
        self.kind = kind

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["run", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return ()


RUNNERS = {k: FakeRunner(k) for k in RunnerKind}


def runners() -> dict[RunnerKind, FakeRunner]:
    return RUNNERS


def resolver_output(wanted: Sequence[str], docs: Mapping[str, Sequence[str]]) -> str:
    lines = sorted((n, label) for n, w in enumerate(wanted) for label in docs.get(w, ()))
    return "\n".join([f"R24-NAMES {len(lines)}", *(f"{n}\t{label}" for n, label in lines), f"R24-END {len(lines)}"])


class FakeSandbox:
    """Answers like the run phase's sandbox: `outcomes` per instance are run_test's answers in order; the probe answers
    with gVisor's banner unless a plant says otherwise; the docstring resolver answers from `docs` (docstring line ->
    labels) unless `resolver` plants its answers; prepare can be made to raise. It refuses use from more than one
    thread, which the contract never promises to allow."""

    def __init__(
        self,
        outcomes: Mapping[str, Sequence[RunResult | Exception]] | None = None,
        *,
        probe: RunResult | Sequence[RunResult] = GVISOR,
        prepare_raises: int = 0,
        delay_s: float = 0.0,
        docs: Mapping[str, Sequence[str]] | None = None,
        resolver: Sequence[RunResult | Exception] = (),
        methods: Mapping[str, PatchMethod] | None = None,
    ) -> None:
        self.outcomes = {k: list(v) for k, v in (outcomes or {}).items()}
        self.probes_answers = [probe] if isinstance(probe, RunResult) else list(probe)
        self.prepare_raises = prepare_raises
        self.delay_s = delay_s
        self.docs = dict(docs or {})
        self.resolver_answers = list(resolver)
        self.methods = dict(methods or {})  # instance -> how prepare applied its patch
        self.resolutions: list[tuple[list[str], list[str]]] = []
        self.prepared: list[tuple[str, Side, str | None]] = []
        self.tests: list[dict[str, object]] = []
        self.probes = 0
        self.closed: list[str] = []
        self.opened: list[str] = []
        self.threads: set[int] = set()

    def _enter(self) -> None:
        self.threads.add(threading.get_ident())
        if len(self.threads) > 1:
            raise AssertionError("one sandbox used from two threads")

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        self._enter()
        time.sleep(self.delay_s)
        if self.prepare_raises:
            self.prepare_raises -= 1
            raise RuntimeError("docker create: no space left on device")
        self.prepared.append((target.id, side, patch))
        method = None if patch is None else self.methods.get(target.id.split("~", 1)[1], PatchMethod.GIT_APPLY)
        ws = Workspace(
            id=f"ws-{len(self.prepared)}", side=side, root="/testbed", image_digest="sha256:" + "e" * 64,
            patch_method=method,
        )  # fmt: skip
        self.opened.append(ws.id)
        return ws

    def run(
        self,
        workspace: Workspace,
        command: Sequence[str],
        *,
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
    ) -> RunResult:
        self._enter()
        assert fresh
        if list(command[:3]) == ["python", "-c", DOCSTRINGS]:
            wanted, files = json.loads(command[3]), json.loads(command[4])
            self.resolutions.append((wanted, files))
            if self.resolver_answers:
                answer = self.resolver_answers.pop(0)
                if isinstance(answer, Exception):
                    raise answer
                return answer
            return result(Outcome.PASS, stdout_tail=resolver_output(wanted, self.docs))
        assert list(command) == ["dmesg"]
        self.probes += 1
        return self.probes_answers.pop(0) if len(self.probes_answers) > 1 else self.probes_answers[0]

    def run_test(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: FakeRunner,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
        expect: Sequence[str] | None = None,
    ) -> RunResult:
        self._enter()
        iid = self.prepared[-1][0].split("~", 1)[1]
        self.tests.append(
            {"tests": tuple(tests), "kind": runner.kind, "changed": tuple(changed_files), "fresh": fresh,
             "expect": tuple(expect or ()), "timeout": timeout_s}
        )  # fmt: skip
        answers = self.outcomes.get(iid) or [result(Outcome.PASS)]
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer

    def close(self, workspace: Workspace) -> None:
        self.closed.append(workspace.id)


def check(sandbox: FakeSandbox, iid: str = "django__django-1", r: Row | None = None, **kw: object) -> R24Result:
    return check_instance(r or row(iid), IMAGE, SPEC, sandbox, RUNNERS, run_id="t", **kw)  # type: ignore[arg-type]


def test_a_gold_that_passes_in_the_run_phase_is_kept_and_ran_the_real_path() -> None:
    sb = FakeSandbox()
    r = check(sb)
    assert r.verdict is R24Verdict.KEPT and r.kept and len(r.attempts) == 1 and r.attempts[0].gvisor is True
    [(tid, side, patch)] = sb.prepared
    assert tid == "gold~django__django-1" and side is Side.GOLD and patch == gold_and_tests(row())
    assert patch is not None and patch.startswith(GOLD_PATCH) and patch.endswith(TEST_PATCH)
    [call] = sb.tests
    assert call["tests"] == ("tests/app/tests.py",)  # the harness's test files, a .json fixture left out
    assert call["expect"] == LABELS  # the runner's own case names (contracts.CaseResult.name), not SWE-bench's
    assert call["kind"] is RunnerKind.DJANGO and call["fresh"] is True and call["changed"] == ("django/x.py",)
    assert sb.probes == 1 and sb.closed == sb.opened
    assert (r.tests_from, r.expected, r.unnamed) == ("test_patch", 3, ())


def test_the_gold_target_carries_the_runner_the_command_and_the_pinned_image() -> None:
    t = gold_target(row(), IMAGE, SPEC)
    assert (t.id, t.tier, t.image, t.test_runner, t.test_command) == (
        "gold~django__django-1",
        Tier.A,
        IMAGE,
        RunnerKind.DJANGO,
        SPEC.test_cmd,
    )
    with pytest.raises(ValueError, match="not the row's"):
        gold_target(row(), IMAGE, SPEC.model_copy(update={"version": "4.0"}))


# ---------------------------------------------------------------- SWE-bench ids in the runner's names


def test_swe_bench_ids_become_the_runners_own_case_names() -> None:
    assert django_label("test_fix (app.tests.T)") == "app.tests.T.test_fix"
    assert django_label("test_fix (app.tests.T.test_fix)") == "app.tests.T.test_fix"  # Python 3.11 and later
    assert django_label("Migration directories without an __init__.py file are loaded.") is None  # a docstring
    named, unnamed = case_names(
        RunnerKind.DJANGO,
        ["test_a (m.tests.C)", "A docstring sentence.", "test_a (m.tests.C.test_a)", "test_b (m.tests.D)"],
    )
    assert (named, unnamed) == (["m.tests.C.test_a", "m.tests.D.test_b"], ["A docstring sentence."])
    named, unnamed = case_names(
        RunnerKind.PYTEST,
        ["tests/t.py::A::test_x[1-2]", "tests/t.py::test_y", "astropy/t.py::test_z[ceci", "test_w"],
    )
    assert named == ["tests/t.py::A::test_x[1-2]", "tests/t.py::test_y"]
    assert unnamed == ["astropy/t.py::test_z[ceci", "test_w"]  # cut at a space by SWE-bench's parser; no file
    assert case_names(RunnerKind.SYMPY, ["test_x", "test_y"]) == (["test_x", "test_y"], [])
    assert suite_modules(RunnerKind.DJANGO, ["validators.tests.T.test_a", "admin.tests.A.test_b", "x.y"]) == [
        "validators.tests",
        "admin.tests",
    ]
    assert suite_modules(RunnerKind.PYTEST, ["tests/t.py::A::test_x", "tests/t.py::test_y"]) == ["tests/t.py"]
    assert suite_modules(RunnerKind.SYMPY, ["test_x"]) == []


def test_an_id_that_names_no_case_is_left_out_of_expect_and_listed() -> None:
    r = row(PASS_TO_PASS=("test_old (app.tests.T)", "Saving twice overwrites the first file."))
    sb = FakeSandbox()  # the source names no test with that docstring
    res = check(sb, r=r)
    assert res.verdict is R24Verdict.KEPT and res.unnamed == ("Saving twice overwrites the first file.",)
    assert sb.tests[0]["expect"] == ("app.tests.T.test_fix", "app.tests.T.test_old") and res.expected == 2
    assert sb.resolutions == [(["Saving twice overwrites the first file."], ["tests/app/tests.py"])]
    pytest_row = row(PASS_TO_PASS=("tests/t.py::test_x[a",))  # cut by SWE-bench's parser: no resolver for pytest
    sb = FakeSandbox()
    res = check_instance(pytest_row.model_copy(update={"FAIL_TO_PASS": ("tests/t.py::test_fix",)}), IMAGE,
                         SPEC.model_copy(update={"runner": RunnerKind.PYTEST}), sb, RUNNERS, run_id="t")  # type: ignore[arg-type]  # fmt: skip
    assert res.unnamed == ("tests/t.py::test_x[a",) and sb.resolutions == []


def test_a_docstring_id_is_named_from_the_source_and_expected() -> None:
    """76 of the pool's 102 Django instances name some tests by a docstring line (1,445 ids; 40 of them FAIL_TO_PASS),
    which the runner reports by label. The resolver reads the labels from the files the run tests."""
    doc = "Migration directories without an __init__.py file are loaded."
    r = row(FAIL_TO_PASS=(doc,), PASS_TO_PASS=("test_old (app.tests.T)", "Saving twice overwrites the first file."))
    sb = FakeSandbox(docs={doc: ["app.tests.LoaderTests.test_loading_namespace_package"]})
    res = check(sb, r=r)
    assert res.verdict is R24Verdict.KEPT and (res.resolved, res.resolver) == (1, "")
    assert sb.tests[0]["expect"] == ("app.tests.LoaderTests.test_loading_namespace_package", "app.tests.T.test_old")
    assert res.unnamed == ("Saving twice overwrites the first file.",) and res.expected == 2


def test_plant_a_docstring_label_the_run_never_reports_is_unchecked_never_kept() -> None:
    """A wrong label from the source can only be an expected case that never ran: run_test answers ERROR (INFRA)."""
    doc = "A docstring sentence."
    sb = FakeSandbox(
        {"django__django-1": [error(FailureType.INFRA)]}, docs={doc: ["app.tests.Wrong.test_x"]}
    )  # fmt: skip
    res = check(sb, r=row(FAIL_TO_PASS=(doc,)))
    assert res.verdict is R24Verdict.INFRA and len(res.attempts) == 2


def test_plant_a_resolver_that_cannot_answer_is_retried_and_its_ids_stay_unnamed() -> None:
    doc = "Saving twice overwrites the first file."
    r = row(PASS_TO_PASS=("test_old (app.tests.T)", doc))
    broken = result(Outcome.ERROR, exit_code=1, stderr_tail="python: can't open file")
    cut = result(Outcome.PASS, stdout_tail="1\tapp.tests.T.test_save\nR24-END 2")  # the head was cut off
    sb = FakeSandbox({"django__django-1": [error(FailureType.INFRA), result(Outcome.PASS)]}, resolver=[broken, cut])
    res = check(sb, r=r)
    assert res.verdict is R24Verdict.KEPT and len(sb.resolutions) == 2  # tried again in the second workspace
    assert res.unnamed == (doc,) and res.resolved == 0 and "cut off or malformed" in res.resolver


def test_a_resolver_that_raises_costs_only_the_docstring_names() -> None:
    doc = "Saving twice overwrites the first file."
    sb = FakeSandbox(resolver=[RuntimeError("docker: no such container"), RuntimeError("docker: no such container")])
    res = check(sb, r=row(PASS_TO_PASS=("test_old (app.tests.T)", doc)))
    assert res.verdict is R24Verdict.KEPT and len(res.attempts) == 1  # the run itself went ahead
    assert res.unnamed == (doc,) and "the docstring resolver raised RuntimeError" in res.resolver


def test_plant_fail_to_pass_ids_the_source_cannot_name_are_unchecked() -> None:
    """A run that expects only PASS_TO_PASS would say KEPT without checking the fix's own tests."""
    r = row(FAIL_TO_PASS=("Migration directories without an __init__.py file are loaded.",))
    sb = FakeSandbox()  # the source names none
    res = check(sb, r=r)
    assert res.verdict is R24Verdict.INFRA and "cannot check the fix's own tests" in res.attempts[-1].error
    assert sb.tests == [] and res.unnamed == r.FAIL_TO_PASS and len(res.attempts) == 1  # no retry: same answer
    sympy = RepoSpec(repo="sympy/sympy", version="1.1", test_cmd="bin/test -C --verbose", runner=RunnerKind.SYMPY)
    bad = row("sympy__sympy-1", repo="sympy/sympy", version="1.1", FAIL_TO_PASS=("not a sympy test name",))
    sb = FakeSandbox()
    res = check_instance(bad, IMAGE, sympy, sb, RUNNERS, run_id="t")  # type: ignore[arg-type]
    assert res.verdict is R24Verdict.INFRA and sb.prepared == []  # nothing to resolve: refused before any container


DJANGO_TESTS = {
    "tests/app/tests.py": '''
from django.test import SimpleTestCase, TestCase
from .base import SharedTests


class HelperMixin:
    def test_from_mixin(self):
        """Shared by the subclasses."""


class LoaderTests(HelperMixin, TestCase):
    def test_loading(self):
        """Migration directories without an __init__.py file are loaded."""

    def test_two_lines(self):
        """
        Starts on the second line.
        More text.
        """

    def helper(self):
        """Not a test."""


class MoreLoaderTests(LoaderTests):
    def test_loading(self):
        pass


class Plain:
    def test_never_loaded(self):
        """Not loaded by unittest."""


class FromElsewhere(SharedTests):
    pass
''',
    "tests/other/tests.py": '''
import unittest


class OtherTests(unittest.TestCase):
    def test_other(self):
        """Not in the files the run tests."""
''',
}


def test_the_resolver_script_reads_labels_from_the_source(tmp_path: Path) -> None:
    """DOCSTRINGS itself, run with this Python on a Django-shaped tree, as the run phase runs it at the root. The
    SWE-bench Django images run Python 3.6 to 3.11, so the script parses as 3.6 and imports only the standard
    library."""
    import ast

    tree = ast.parse(DOCSTRINGS, feature_version=(3, 6))
    imported = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names}
    assert imported == {"ast", "json", "os", "sys"} and not any(isinstance(n, ast.ImportFrom) for n in ast.walk(tree))
    assert not any(isinstance(n, ast.JoinedStr) for n in ast.walk(tree))  # no f-strings either, to be safe
    for path, text in DJANGO_TESTS.items():
        (tmp_path / path).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / path).write_text(text)
    wanted = [
        "Migration directories without an __init__.py file are loaded.",
        "Starts on the second line.",
        "Shared by the subclasses.",
        "Not a test.",
        "Not loaded by unittest.",
        "Not in the files the run tests.",
        "Defined in another file.",
    ]
    out = subprocess.run(
        [sys.executable, "-I", "-c", DOCSTRINGS, json.dumps(wanted), json.dumps(["tests/app/tests.py"])],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=True,
    ).stdout  # fmt: skip
    labels = parse_docstring_labels(out, wanted)
    assert labels == {
        wanted[0]: ("app.tests.LoaderTests.test_loading",),  # MoreLoaderTests overrides it without a docstring
        wanted[1]: ("app.tests.LoaderTests.test_two_lines", "app.tests.MoreLoaderTests.test_two_lines"),
        wanted[2]: ("app.tests.LoaderTests.test_from_mixin", "app.tests.MoreLoaderTests.test_from_mixin"),
    }
    assert django_test_files(["tests/app/tests.py", "validators.tests", "tests/app/tests.py"]) == [
        "tests/app/tests.py",
        "tests/validators/tests.py",
    ]


@pytest.mark.parametrize(
    "output",
    [
        "",
        "R24-NAMES 2\n0\tapp.tests.T.test_a\nR24-END 2",  # one line short
        "0\tapp.tests.T.test_a\nR24-END 1",  # the header was cut
        "R24-NAMES 1\n0\tapp.tests.T.test_a",  # the trailer was cut
        "R24-NAMES 1\n5\tapp.tests.T.test_a\nR24-END 1",  # an index past the wanted lines
        "R24-NAMES 1\n0\t--settings=evil\nR24-END 1",  # not a label
        "R24-NAMES 1\n0\tapp.test_a\nR24-END 1",  # a module and a name: no class
    ],
)
def test_plant_a_resolver_output_that_is_cut_or_forged_is_never_an_answer(output: str) -> None:
    assert parse_docstring_labels(output, ["A docstring sentence."]) is None


# ---------------------------------------------------------------- blocker 1: no instance stops the run


def test_plant_a_test_patch_that_touches_only_txt_files_runs_the_named_cases_modules() -> None:
    """django__django-10097: harness_test_files is empty, and the old code raised, aborting the whole R24 run. The
    harness runs the whole suite; R24 runs the modules its FAIL_TO_PASS and PASS_TO_PASS cases live in."""
    r = row(
        test_patch=TXT_ONLY,
        FAIL_TO_PASS=("test_ascii_validator (validators.tests.TestValidators)", "Named URLs should be reversible"),
        PASS_TO_PASS=("test_x (auth_tests.test_management.CreatesuperuserTests)",),
    )
    sb = FakeSandbox()
    res = check(sb, r=r)
    assert res.verdict is R24Verdict.KEPT and res.tests_from == "ids"
    assert sb.tests[0]["tests"] == ("validators.tests", "auth_tests.test_management")
    assert sb.tests[0]["expect"] == (
        "validators.tests.TestValidators.test_ascii_validator",
        "auth_tests.test_management.CreatesuperuserTests.test_x",
    )
    assert res.unnamed == ("Named URLs should be reversible",)


def test_plant_a_test_patch_with_no_test_file_and_no_module_is_infra_with_its_reason() -> None:
    sympy = RepoSpec(repo="sympy/sympy", version="1.1", test_cmd="bin/test -C --verbose", runner=RunnerKind.SYMPY)
    r = row("sympy__sympy-1", repo="sympy/sympy", version="1.1", test_patch=TXT_ONLY, FAIL_TO_PASS=("test_x",),
            PASS_TO_PASS=())  # fmt: skip
    sb = FakeSandbox()
    res = check_instance(r, IMAGE, sympy, sb, RUNNERS, run_id="t")  # type: ignore[arg-type]
    assert res.verdict is R24Verdict.INFRA and "touches no test file" in res.attempts[0].error
    assert res.tests_from == "none" and sb.prepared == []


def pool_check(tmp_path: Path, rows: Mapping[str, Row], sandbox: object, **kw: object) -> R24Report:
    ids = sorted(rows)
    factory = sandbox if callable(sandbox) else (lambda: sandbox)
    return run_check(
        rows,
        dict.fromkeys(ids, IMAGE),
        SPECS,
        ids,
        factory,  # type: ignore[arg-type]
        runners,  # type: ignore[arg-type]
        tmp_path,
        run_id="t",
        **kw,  # type: ignore[arg-type]
    )


def test_plant_one_instance_that_cannot_be_checked_never_aborts_the_run(tmp_path: Path) -> None:
    """The real pool's shape: with django__django-10097's .txt-only test patch the old code raised and lost every
    result after it, and merge then refused the base. Every instance gets a line, and the base merges."""
    rows = {
        "django__django-1": row("django__django-1"),
        "django__django-10097": row("django__django-10097", test_patch=TXT_ONLY, FAIL_TO_PASS=("Named URLs",)),
        "django__django-3": row("django__django-3"),
    }
    report = pool_check(tmp_path, rows, FakeSandbox(), workers=1)
    assert report.verdicts == {"infra": 1, "kept": 2} and report.unchecked == ("django__django-10097",)
    base = (tmp_path / "results.jsonl").read_text()
    assert len(base.splitlines()) == 3
    m = merge(base, rows, dict.fromkeys(rows, IMAGE), drops=[GoldDrop(instance_id="django__django-10097", reason="x")])
    assert m.verdicts == {"django__django-1": True, "django__django-10097": False, "django__django-3": True}


def test_plant_a_check_that_raises_is_that_instances_infra(tmp_path: Path) -> None:
    class Exploding(FakeSandbox):
        def run_test(self, workspace: Workspace, tests: Sequence[str], **kw: object) -> RunResult:  # type: ignore[override]
            if self.prepared[-1][0].endswith("-2"):
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            return super().run_test(workspace, tests, **kw)  # type: ignore[arg-type]

    class Broken(FakeSandbox):
        def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
            if target.id.endswith("-3"):
                raise KeyError("a bug in the sandbox")
            return super().prepare(target, side, patch=patch)

    rows = {i: row(i) for i in ("django__django-1", "django__django-2", "django__django-3")}
    report = pool_check(tmp_path / "a", rows, Exploding(), workers=1)
    assert (
        report.verdicts == {"infra": 1, "kept": 2}
        and len((tmp_path / "a" / "results.jsonl").read_text().splitlines()) == 3
    )

    def raising_factory() -> FakeSandbox:
        raise OSError("docker: command not found")

    report = pool_check(tmp_path / "b", rows, raising_factory, workers=1)
    assert report.verdicts == {"infra": 3}
    lines = [R24Result.model_validate_json(x) for x in (tmp_path / "b" / "results.jsonl").read_text().splitlines()]
    assert all("docker: command not found" in r.attempts[0].error for r in lines)
    report = pool_check(tmp_path / "c", rows, Broken(), workers=1)
    assert report.verdicts == {"infra": 1, "kept": 2}  # check_instance records the sandbox's raise as INFRA


def test_a_runtime_mismatch_still_stops_everything(tmp_path: Path) -> None:
    rows = {i: row(i) for i in ("django__django-1", "django__django-2")}
    with pytest.raises(RuntimeMismatchError):
        pool_check(tmp_path, rows, FakeSandbox(probe=HOST_READ), workers=1)


def test_plant_an_unwired_runner_is_refused_before_any_instance() -> None:
    with pytest.raises(ValueError, match="no django runner is wired"):
        run_check({"django__django-1": row()}, {"django__django-1": IMAGE}, SPECS, ["django__django-1"],
                  FakeSandbox, lambda: {}, Path("/nonexistent/never-written"), run_id="t")  # fmt: skip


# ---------------------------------------------------------------- blockers 2 and 3: every outcome's verdict


@pytest.mark.parametrize(
    ("answers", "verdict", "attempts"),
    [
        ([result(Outcome.FAIL)], R24Verdict.GOLD_FAILS, 1),
        ([result(Outcome.SKIPPED, exit_code=0)], R24Verdict.GOLD_SKIPS, 1),
        ([result(Outcome.TIMEOUT, exit_code=None), result(Outcome.TIMEOUT, exit_code=None)], R24Verdict.TIMEOUT, 2),
        ([result(Outcome.TIMEOUT, exit_code=None), result(Outcome.PASS)], R24Verdict.KEPT, 2),
        ([error(None), error(None)], R24Verdict.INFRA, 2),
        ([error(FailureType.INFRA), error(FailureType.INFRA)], R24Verdict.INFRA, 2),  # no report, a case missing
        ([error(None), result(Outcome.FAIL)], R24Verdict.GOLD_FAILS, 2),
        ([RuntimeError("docker exec failed"), RuntimeError("docker exec failed")], R24Verdict.INFRA, 2),
        # blocker 2: a typed ERROR is the gold failing in the run phase, one try is enough, as for FAIL
        ([error(FailureType.CRASH_ELSEWHERE)], R24Verdict.GOLD_ERRORS, 1),
        ([error(FailureType.IMPORT)], R24Verdict.GOLD_ERRORS, 1),
        ([error(FailureType.NAME)], R24Verdict.GOLD_ERRORS, 1),
        ([error(FailureType.FIXTURE)], R24Verdict.GOLD_ERRORS, 1),
        ([error(FailureType.OTHER)], R24Verdict.GOLD_ERRORS, 1),
        ([error(FailureType.INFRA), error(FailureType.IMPORT)], R24Verdict.GOLD_ERRORS, 2),
        # blocker 3: a TIMEOUT drops only when both attempts timed out
        ([error(None), result(Outcome.TIMEOUT, exit_code=None)], R24Verdict.INFRA, 2),
        ([error(FailureType.INFRA), result(Outcome.TIMEOUT, exit_code=None)], R24Verdict.INFRA, 2),
        ([RuntimeError("exec died"), result(Outcome.TIMEOUT, exit_code=None)], R24Verdict.INFRA, 2),
    ],
)
def test_plant_every_run_phase_outcome_maps_to_its_verdict(
    answers: list[RunResult | Exception], verdict: R24Verdict, attempts: int
) -> None:
    sb = FakeSandbox({"django__django-1": answers})
    r = check(sb)
    assert (r.verdict, len(r.attempts)) == (verdict, attempts)
    assert sb.closed == sb.opened  # every workspace closed, whatever happened


def test_plant_a_typed_error_is_dropped_with_its_type_and_diagnosed() -> None:
    """Blocker 2: an expected case that crashed outside the gold files (a PermissionError from a read-only root, say)
    or failed to import was INFRA, never dropped and never diagnosed."""
    seen: list[str] = []

    def diagnose(iid: str) -> str:
        seen.append(iid)
        return "harness (root): kept, so the run phase (uid 65534, read-only root) breaks it: back to the founder"

    for ftype in (FailureType.CRASH_ELSEWHERE, FailureType.IMPORT):
        r = check(FakeSandbox({"django__django-1": [error(ftype)]}), diagnose=diagnose)
        assert r.verdict is R24Verdict.GOLD_ERRORS and r.attempts[-1].failure_type == ftype.value
        assert r.diagnosis.startswith("harness (root): kept")
    assert seen == ["django__django-1", "django__django-1"]


def test_plant_a_timeout_after_another_failure_is_not_a_drop() -> None:
    """Blocker 3: [ERROR, TIMEOUT] and [prepare raises, TIMEOUT] were TIMEOUT drops after one timeout, and merge never
    lets a rerun replace a drop."""
    r = check(FakeSandbox({"django__django-1": [error(None), result(Outcome.TIMEOUT, exit_code=None)]}))
    assert r.verdict is R24Verdict.INFRA
    r = check(FakeSandbox({"django__django-1": [result(Outcome.TIMEOUT, exit_code=None)]}, prepare_raises=1))
    assert [a.outcome for a in r.attempts] == [None, Outcome.TIMEOUT] and r.verdict is R24Verdict.INFRA
    r = check(FakeSandbox({"django__django-1": [result(Outcome.TIMEOUT, exit_code=None)] * 2}))
    assert r.verdict is R24Verdict.TIMEOUT
    lone = (R24Attempt(gvisor=True, outcome=Outcome.TIMEOUT),)
    assert verdict_of(lone) is R24Verdict.INFRA  # one timeout alone is never a drop


def test_plant_a_sandbox_that_cannot_prepare_is_infra_never_a_drop() -> None:
    r = check(FakeSandbox(prepare_raises=2))
    assert r.verdict is R24Verdict.INFRA and "no space left" in r.attempts[-1].error
    r = check(FakeSandbox(prepare_raises=1))
    assert r.verdict is R24Verdict.KEPT and len(r.attempts) == 2


# ---------------------------------------------------------------- the kernel probe (the integrator's call c)


def test_plant_a_kernel_log_that_reads_and_is_not_gvisors_stops_everything() -> None:
    sb = FakeSandbox(probe=HOST_READ)
    with pytest.raises(RuntimeMismatchError, match="read a kernel log without"):
        check(sb)
    assert sb.tests == [] and sb.closed == sb.opened


@pytest.mark.parametrize(
    "probe",
    [
        HOST_DENIED,  # exit 1: a container that died right after start, or a kernel that refused: nothing was read
        result(Outcome.ERROR, exit_code=125, stderr_tail="OCI runtime create failed"),
        result(Outcome.TIMEOUT, exit_code=None),
    ],
)
def test_plant_a_probe_that_read_nothing_is_this_attempts_infra_and_retried(probe: RunResult) -> None:
    """The integrator's call: under runsc, a probe exit 1 is INFRA for that instance, never a stop for the run."""
    sb = FakeSandbox(probe=probe)
    r = check(sb)
    assert r.verdict is R24Verdict.INFRA and len(r.attempts) == 2 and "probe read no kernel log" in r.attempts[0].error
    assert sb.tests == []
    flaky = FakeSandbox(probe=[probe, GVISOR])  # the second fresh container answers
    assert check(flaky).verdict is R24Verdict.KEPT


def test_every_probe_failing_is_reported_as_such(tmp_path: Path) -> None:
    rows = {i: row(i) for i in ("django__django-1", "django__django-2")}
    report = pool_check(tmp_path, rows, FakeSandbox(probe=HOST_DENIED), workers=1)
    assert report.unchecked == report.probe_failed == ("django__django-1", "django__django-2")


def test_the_harness_diagnosis_runs_only_for_a_drop() -> None:
    seen: list[str] = []

    def diagnose(iid: str) -> str:
        seen.append(iid)
        return "harness (root): kept"

    assert check(FakeSandbox(), diagnose=diagnose).diagnosis == "" and seen == []
    dropped = check(FakeSandbox({"django__django-1": [result(Outcome.FAIL)]}), diagnose=diagnose)
    assert dropped.diagnosis == "harness (root): kept" and seen == ["django__django-1"]
    infra = check(FakeSandbox({"django__django-1": [error(None), error(None)]}), diagnose=diagnose)
    assert infra.diagnosis == "" and seen == ["django__django-1"]


# ---------------------------------------------------------------- the pool: logs, the limit, one sandbox per worker


def test_run_check_writes_each_result_and_reports_drops_and_unchecked(tmp_path: Path) -> None:
    ids = ["django__django-1", "django__django-2", "django__django-3"]
    sb = FakeSandbox({"django__django-2": [result(Outcome.FAIL)], "django__django-3": [error(None), error(None)]})
    report = pool_check(tmp_path, {i: row(i) for i in ids}, sb, workers=1)
    assert report.verdicts == {"gold_fails": 1, "infra": 1, "kept": 1}
    assert report.dropped == {"django__django-2": "gold_fails"} and report.unchecked == ("django__django-3",)
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 3
    assert (tmp_path / "dropped.tsv").read_text().startswith("django__django-2\tgold_fails")
    with pytest.raises(ValueError, match="already has results"):
        pool_check(tmp_path, {i: row(i) for i in ids}, sb, workers=1)


def test_each_worker_gets_its_own_sandbox(tmp_path: Path) -> None:
    """The integrator's call: the contract says nothing about a sandbox's thread safety, so no two threads share one.
    The fake raises if two threads touch the same sandbox; with one shared sandbox this run would fail."""
    made: list[FakeSandbox] = []

    def factory() -> FakeSandbox:
        sb = FakeSandbox(delay_s=0.02)
        made.append(sb)
        return sb

    rows = {f"django__django-{n}": row(f"django__django-{n}") for n in range(12)}
    report = pool_check(tmp_path, rows, factory, workers=4)
    assert report.verdicts == {"kept": 12}
    assert 2 <= len(made) <= 4 and all(len(sb.threads) == 1 for sb in made)
    assert sum(len(sb.prepared) for sb in made) == 12


@pytest.mark.parametrize(("dropped", "over"), [(1, False), (2, True), (25, False), (26, True)])
def test_the_drop_limit_is_more_than_a_quarter(dropped: int, over: bool) -> None:
    """Q8: exactly 25% dropped is within the limit (exit 0); one more is over it (exit 3)."""
    instances = 4 if dropped <= 2 else 100
    report = R24Report(run_id="t", instances=instances, verdicts={}, dropped={str(n): "x" for n in range(dropped)},
                       unchecked=(), max_drop_pct=MAX_DROP_PCT)  # fmt: skip
    assert report.drop_pct_over_limit is over


# ---------------------------------------------------------------- merging R24 into the manifest


IMAGES = {f"django__django-{n}": IMAGE for n in (1, 2, 3)}


def line(iid: str, verdict: R24Verdict, run_id: str = "d1") -> str:
    answers = {
        R24Verdict.KEPT: [result(Outcome.PASS)],
        R24Verdict.GOLD_FAILS: [result(Outcome.FAIL)],
        R24Verdict.GOLD_ERRORS: [error(FailureType.IMPORT)],
        R24Verdict.GOLD_SKIPS: [result(Outcome.SKIPPED)],
        R24Verdict.TIMEOUT: [result(Outcome.TIMEOUT)] * 2,
        R24Verdict.INFRA: [error(None)] * 2,
    }[verdict]
    sb = FakeSandbox({iid: answers})
    r = check(sb, iid)
    assert r.verdict is verdict
    return r.model_copy(update={"run_id": run_id}).model_dump_json()


POOL = ["django__django-1", "django__django-2", "django__django-3"]


def test_verdicts_map_to_kept_dropped_and_unchecked() -> None:
    base = "\n".join(
        [line(POOL[0], R24Verdict.KEPT), line(POOL[1], R24Verdict.GOLD_ERRORS), line(POOL[2], R24Verdict.INFRA)]
    )
    m = merge(base, POOL, IMAGES)
    assert m.verdicts == {POOL[0]: True, POOL[1]: False, POOL[2]: None}
    assert m.reasons[POOL[1]] == "gold_errors" and [i.role for i in m.inputs] == ["base"]


def test_plant_an_infra_rerun_for_one_instance_merges_into_the_base() -> None:
    """A rerun file with only the infra instance merges, so clearing one INFRA verdict never means rerunning 251."""
    base = "\n".join(
        [line(POOL[0], R24Verdict.KEPT), line(POOL[1], R24Verdict.GOLD_FAILS), line(POOL[2], R24Verdict.INFRA)]
    )
    m = merge(base, POOL, IMAGES, reruns=[line(POOL[2], R24Verdict.KEPT, run_id="d2")])
    assert m.verdicts == {POOL[0]: True, POOL[1]: False, POOL[2]: True}
    assert [(i.role, i.run_ids, i.lines) for i in m.inputs] == [("base", ("d1",), 3), ("rerun", ("d2",), 1)]


@pytest.mark.parametrize(
    ("rerun", "match"),
    [
        (line(POOL[0], R24Verdict.GOLD_FAILS, run_id="d2"), "already has a verdict"),  # a kept one never changes
        (line(POOL[1], R24Verdict.KEPT, run_id="d2"), "already has a verdict"),  # nor a dropped one
        (line("sympy__sympy-9", R24Verdict.KEPT, run_id="d2"), "not on the pool"),
    ],
)
def test_plant_a_rerun_that_would_change_a_settled_verdict_is_refused(rerun: str, match: str) -> None:
    base = "\n".join(
        [line(POOL[0], R24Verdict.KEPT), line(POOL[1], R24Verdict.GOLD_FAILS), line(POOL[2], R24Verdict.INFRA)]
    )
    with pytest.raises(R24ResultsError, match=match):
        merge(base, POOL, IMAGES, reruns=[rerun])


def test_plant_a_base_that_cannot_be_trusted_is_refused() -> None:
    good = [line(i, R24Verdict.KEPT) for i in POOL]
    with pytest.raises(R24ResultsError, match="appears twice"):
        merge("\n".join([*good, good[0]]), POOL, IMAGES)
    with pytest.raises(R24ResultsError, match="1 pool instances have no R24 verdict"):
        merge("\n".join(good[:2]), POOL, IMAGES)
    forged = R24Result.model_validate_json(line(POOL[2], R24Verdict.GOLD_FAILS)).model_copy(
        update={"verdict": R24Verdict.KEPT}
    )
    with pytest.raises(R24ResultsError, match="its runs say gold_fails"):
        merge("\n".join([*good[:2], forged.model_dump_json()]), POOL, IMAGES)
    with pytest.raises(R24ResultsError, match=r"not a receipts\.eval\.r24/2 result"):
        merge("\n".join([*good[:2], good[2].replace("receipts.eval.r24/2", "receipts.eval.r24/1")]), POOL, IMAGES)


@pytest.mark.parametrize("gvisor", [None, False])
def test_plant_a_results_line_that_passed_without_the_banner_is_never_read_as_kept(gvisor: bool | None) -> None:
    """Q1: a line whose last run passed but whose probe never showed gVisor's banner must not merge as kept."""
    good = [line(i, R24Verdict.KEPT) for i in POOL]
    kept = R24Result.model_validate_json(good[2])
    no_banner = kept.model_copy(update={"attempts": (kept.attempts[0].model_copy(update={"gvisor": gvisor}),)})
    assert verdict_of(no_banner.attempts) is R24Verdict.INFRA
    with pytest.raises(R24ResultsError, match="says kept, its runs say infra"):
        merge("\n".join([*good[:2], no_banner.model_dump_json()]), POOL, IMAGES)


def test_plant_a_result_that_ran_another_image_than_its_pin_is_refused() -> None:
    """The integrator's call: merge checks every result's image against eval/images.json."""
    good = [line(i, R24Verdict.KEPT) for i in POOL]
    other = "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1@sha256:" + "f" * 64
    swapped = R24Result.model_validate_json(good[0]).model_copy(update={"image": other})
    with pytest.raises(R24ResultsError, match=r"not its pin in eval/images\.json"):
        merge("\n".join([swapped.model_dump_json(), *good[1:]]), POOL, IMAGES)
    with pytest.raises(R24ResultsError, match="not its pin"):
        merge("\n".join(good), POOL, IMAGES | {POOL[1]: other})
    renamed = R24Result.model_validate_json(good[0]).model_copy(update={"target_id": "entry~django__django-1"})
    with pytest.raises(R24ResultsError, match="names target"):
        merge("\n".join([renamed.model_dump_json(), *good[1:]]), POOL, IMAGES)
    infra = line(POOL[2], R24Verdict.INFRA)
    rerun = R24Result.model_validate_json(line(POOL[2], R24Verdict.KEPT, run_id="d2")).model_copy(
        update={"image": other}
    )
    with pytest.raises(R24ResultsError, match="rerun 1 line 1"):
        merge("\n".join([*good[:2], infra]), POOL, IMAGES, reruns=[rerun.model_dump_json()])


def test_founder_drops_land_only_on_unchecked_instances() -> None:
    base = "\n".join(
        [line(POOL[0], R24Verdict.KEPT), line(POOL[1], R24Verdict.GOLD_FAILS), line(POOL[2], R24Verdict.INFRA)]
    )
    m = merge(base, POOL, IMAGES, drops=[GoldDrop(instance_id=POOL[2], reason="needs root (REG-565)")])
    assert m.verdicts[POOL[2]] is False and m.reasons[POOL[2]] == "founder: needs root (REG-565)"
    assert [i.role for i in m.inputs] == ["base", "drop"]
    with pytest.raises(R24ResultsError, match="only unchecked"):
        merge(base, POOL, IMAGES, drops=[GoldDrop(instance_id=POOL[0], reason="x")])
    assert parse_drops(f"# founder, 10/28\n{POOL[2]}\tneeds root\n") == [
        GoldDrop(instance_id=POOL[2], reason="needs root")
    ]
    with pytest.raises(R24ResultsError, match="TAB"):
        parse_drops(f"{POOL[2]} needs root\n")


# ---------------------------------------------------------------- round 4, item 2: an import error is the gold failing


def reduced(cases: Sequence[CaseResult], expect: Sequence[str]) -> RunResult:
    """contracts.Sandbox.run_test's rule for `expect` (contract v3), as the attack lane builds it: PASS needs every
    expected case to have run and passed; an expected case that never ran is ERROR, typed IMPORT when the run reports
    its test module's import error and INFRA otherwise; an expected case that ran and did not pass is that outcome."""
    by = {c.name: c for c in cases}
    missing = [e for e in expect if e not in by]
    if missing:
        broken = [c.name for c in cases if c.failure_type is FailureType.IMPORT]
        imported = [e for e in missing if any(e.startswith(m + ".") for m in broken)]
        if imported:
            why = f"expected case {imported[0]} never ran: its test module failed to import"
            return result(Outcome.ERROR, failure_type=FailureType.IMPORT, problem=why)
        return result(Outcome.ERROR, failure_type=FailureType.INFRA, problem=f"expected case {missing[0]} never ran")
    bad = [by[e] for e in expect if by[e].outcome is not Outcome.PASS]
    if bad:
        return result(bad[0].outcome, failure_type=bad[0].failure_type)
    return result(Outcome.PASS)


class ScriptedRunPhase(FakeSandbox):
    """A scripted run: each instance's runner reports `cases`, and run_test reduces them with the contract's rule."""

    def __init__(self, cases: Mapping[str, Sequence[CaseResult]], **kw: Any) -> None:
        super().__init__(**kw)
        self.cases = {k: list(v) for k, v in cases.items()}

    def run_test(  # type: ignore[override]
        self, workspace: Workspace, tests: Sequence[str], *, expect: Sequence[str] | None = None, **kw: Any
    ) -> RunResult:
        super().run_test(workspace, tests, expect=expect, **kw)
        return reduced(self.cases[self.prepared[-1][0].split("~", 1)[1]], expect or ())


def test_plant_an_expected_case_whose_module_failed_to_import_is_a_gold_failure_never_infra(tmp_path: Path) -> None:
    """Contract v3: an expected case that never ran because its test module failed to import is ERROR typed IMPORT.
    The harness reads that gold as unresolved and drops the instance, and so does R24: gold_errors in one attempt,
    diagnosed, never INFRA. An expected case missing for any other reason stays INFRA, retried, and a rerun may settle
    it."""
    import_error = [
        CaseResult(name="app.tests", outcome=Outcome.ERROR, failure_type=FailureType.IMPORT,
                   message="ModuleNotFoundError: No module named 'app.helpers'"),
    ]  # fmt: skip
    vanished = [CaseResult(name=label, outcome=Outcome.PASS) for label in LABELS[1:]]  # test_fix never ran, no error
    passing = [CaseResult(name=label, outcome=Outcome.PASS) for label in LABELS]
    rows = {i: row(i) for i in ("django__django-1", "django__django-2", "django__django-3")}
    sb = ScriptedRunPhase({"django__django-1": import_error, "django__django-2": vanished, "django__django-3": passing})
    seen: list[str] = []

    def diagnose(iid: str) -> str:
        seen.append(iid)
        return "harness (root): gold_fails"

    report = pool_check(tmp_path, rows, sb, workers=1, diagnose=diagnose)
    lines = [R24Result.model_validate_json(x) for x in (tmp_path / "results.jsonl").read_text().splitlines()]
    by = {r.instance_id: r for r in lines}
    imp = by["django__django-1"]
    assert (imp.verdict, len(imp.attempts), imp.attempts[0].failure_type) == (R24Verdict.GOLD_ERRORS, 1, "import")
    assert "failed to import" in imp.attempts[0].problem and imp.diagnosis == "harness (root): gold_fails"
    assert by["django__django-2"].verdict is R24Verdict.INFRA and len(by["django__django-2"].attempts) == 2
    assert "never ran" in by["django__django-2"].attempts[-1].problem and by["django__django-3"].kept
    assert report.dropped == {"django__django-1": "gold_errors"} and report.unchecked == ("django__django-2",)
    assert seen == ["django__django-1"]  # a drop is diagnosed; INFRA is not
    assert (
        tmp_path / "dropped.tsv"
    ).read_text() == "django__django-1\tgold_errors\timport\tharness (root): gold_fails\n"
    m = merge((tmp_path / "results.jsonl").read_text(), rows, dict.fromkeys(rows, IMAGE))
    assert m.verdicts == {"django__django-1": False, "django__django-2": None, "django__django-3": True}
    assert m.reasons["django__django-1"] == "gold_errors"


# ---------------------------------------------------------------- round 4, item 3: how gold and the test patch applied


def test_r24_records_how_gold_and_the_test_patch_applied(tmp_path: Path) -> None:
    """REG-569 and contract v3: prepare() applies gold plus the test patch through the harness's chain and the
    workspace names the step that applied it. R24 records it on every attempt and on the result, counts the methods
    and lists every instance a plain `git apply --verbose` could not apply."""
    rows = {i: row(i) for i in ("django__django-1", "django__django-2", "django__django-3")}
    methods = {"django__django-2": PatchMethod.GNU_PATCH_FUZZ, "django__django-3": PatchMethod.GIT_APPLY_REJECT}
    sb = FakeSandbox({"django__django-3": [error(FailureType.INFRA), result(Outcome.PASS)]}, methods=methods)
    report = pool_check(tmp_path, rows, sb, workers=1)
    by = {
        r.instance_id: r
        for r in (R24Result.model_validate_json(x) for x in (tmp_path / "results.jsonl").read_text().splitlines())
    }
    assert by["django__django-1"].patch_method is PatchMethod.GIT_APPLY
    assert by["django__django-2"].patch_method is PatchMethod.GNU_PATCH_FUZZ
    assert [a.patch_method for a in by["django__django-3"].attempts] == [PatchMethod.GIT_APPLY_REJECT] * 2
    assert report.patch_methods == {"git_apply": 1, "git_apply_reject": 1, "gnu_patch_fuzz": 1}
    assert report.patch_fallbacks == {"django__django-2": "gnu_patch_fuzz", "django__django-3": "git_apply_reject"}
    r = check(FakeSandbox(prepare_raises=1))  # an attempt that never prepared has no method; the result keeps the last
    assert [a.patch_method for a in r.attempts] == [None, PatchMethod.GIT_APPLY]
    assert r.patch_method is PatchMethod.GIT_APPLY


# ---------------------------------------------------------------- round 4, item 4: ids R24 cannot name, before the run

SPHINX = RepoSpec(
    repo="sphinx-doc/sphinx", version="3.5", test_cmd="tox --current-env -epy39 -v --", runner=RunnerKind.PYTEST
)
SPHINX_TESTS = (
    "diff --git a/tests/test_pycode_ast.py b/tests/test_pycode_ast.py\n--- a/tests/test_pycode_ast.py\n"
    "+++ b/tests/test_pycode_ast.py\n@@ -1 +1 @@\n-x\n+y\n"
)


def sphinx_row(iid: str, f2p: tuple[str, ...], p2p: tuple[str, ...]) -> Row:
    return row(
        iid, repo="sphinx-doc/sphinx", version="3.5", test_patch=SPHINX_TESTS, FAIL_TO_PASS=f2p, PASS_TO_PASS=p2p
    )


def test_every_instance_with_ids_r24_cannot_name_is_listed_before_any_instance_runs(tmp_path: Path) -> None:
    """The round-3 review's first note: sphinx-doc__sphinx-8265's only FAIL_TO_PASS id was cut at a space by SWE-bench's
    log parser, so R24 leaves it unchecked on every run, and the hint said to rerun it. R24 still judges it the same
    way (the founder decides), but now lists, before any instance runs, every instance with ids it cannot name yet,
    and marks the ones no rerun can settle."""
    ast_ids = ("tests/test_pycode_ast.py::test_unparse[a", "tests/test_pycode_ast.py::test_unparse[1]")
    cut = sphinx_row("sphinx-doc__sphinx-8265", ("tests/test_pycode_ast.py::test_unparse[(1,",), ast_ids)
    partly = sphinx_row("sphinx-doc__sphinx-9000", ("tests/test_pycode_ast.py::test_fix",),
                        ("tests/test_pycode_ast.py::test_inline[verify-:kbd:`",))  # fmt: skip
    doc = "Saving twice overwrites the first file."
    rows = {
        "django__django-1": row("django__django-1"),
        "django__django-2": row("django__django-2", PASS_TO_PASS=("test_old (app.tests.T)", doc)),
        cut.instance_id: cut,
        partly.instance_id: partly,
    }
    specs = {**SPECS, ("sphinx-doc/sphinx", "3.5"): SPHINX}

    class Watching(FakeSandbox):
        def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
            assert (tmp_path / "unnamed.json").is_file()  # the listing is written before any instance runs
            return super().prepare(target, side, patch=patch)

    namings = naming_before_run(rows, specs, rows)
    assert [n.instance_id for n in namings] == [
        "django__django-2",
        "sphinx-doc__sphinx-8265",
        "sphinx-doc__sphinx-9000",
    ]
    report = run_check(rows, dict.fromkeys(rows, IMAGE), specs, sorted(rows), Watching, runners, tmp_path,  # type: ignore[arg-type]
                       run_id="t", workers=1)  # fmt: skip
    listed = json.loads((tmp_path / "unnamed.json").read_text())
    assert [n["instance_id"] for n in listed] == [n.instance_id for n in namings]
    assert listed[1]["cut_fail_to_pass"] == ["tests/test_pycode_ast.py::test_unparse[(1,"]
    assert listed[1]["cut_pass_to_pass"] == ["tests/test_pycode_ast.py::test_unparse[a"] and listed[1]["unnameable"]
    assert listed[0]["docstring_pass_to_pass"] == [doc] and not listed[0]["unnameable"]
    assert listed[2]["cut_pass_to_pass"] and not listed[2]["unnameable"]
    text = "\n".join(naming_lines(namings))
    assert all(n.instance_id in text for n in namings) and text.startswith("before the run: 3 instances")
    stuck = next(line for line in naming_lines(namings) if "none of its FAIL_TO_PASS ids" in line)
    assert stuck.startswith("  sphinx-doc__sphinx-8265:") and "no rerun can change that" in stuck
    assert "--gold-drop" in stuck and "the founder's rule" in stuck
    # judged as before: unchecked, never a drop; and the report says no rerun can settle it
    assert report.unchecked == report.unnameable == ("sphinx-doc__sphinx-8265",) and report.dropped == {}
    assert naming_lines(()) == ["before the run: R24 names every FAIL_TO_PASS and PASS_TO_PASS id of these instances"]


def test_an_instance_whose_source_names_none_of_its_fail_to_pass_docstrings_is_marked_unnameable() -> None:
    """A Django instance whose every FAIL_TO_PASS id is a docstring line the source in the workspace does not hold
    cannot be checked by any rerun either; one whose resolver could not answer can be, so it is not marked."""
    r = row(FAIL_TO_PASS=("Migration directories without an __init__.py file are loaded.",))
    read = check(FakeSandbox(), r=r)  # the resolver read the source and found none
    assert read.verdict is R24Verdict.INFRA and read.unnameable
    broken = result(Outcome.ERROR, exit_code=1, stderr_tail="python: can't open file")
    unread = check(FakeSandbox(resolver=[broken, broken]), r=r)
    assert unread.verdict is R24Verdict.INFRA and not unread.unnameable


# ---------------------------------------------------------------- round 4, item 5: a derived run's timeout


@pytest.mark.parametrize(
    ("tests_from", "expected", "base", "timeout"),
    [
        ("test_patch", 2488, 1800, 1800),  # matplotlib__matplotlib-25122: 2,488 ids in one test file, the harness's
        ("ids", 10, 1800, 1800),  # never under --timeout
        ("ids", 1870, 1800, 3740),  # django__django-10097: 122 modules, about 1,870 expected cases
        ("ids", 5000, 1800, DERIVED_TIMEOUT_CAP_S),  # capped at 2 hours
        ("ids", 5000, 9000, 9000),  # unless --timeout itself is more
    ],
)
def test_a_derived_module_runs_timeout_scales_with_its_expected_cases(
    tests_from: str, expected: int, base: float, timeout: float
) -> None:
    assert run_timeout(tests_from, expected, base) == timeout
    assert (DERIVED_SECONDS_PER_CASE, DERIVED_TIMEOUT_CAP_S) == (2, 7200)


def test_plant_a_whole_suite_instance_gets_the_scaled_timeout_and_a_test_patch_run_keeps_the_harnesss() -> None:
    """django__django-10097's derived run (122 modules, 1,887 expected cases) had one 1800 s run_test, so a
    whole-suite instance could time out by construction under gVisor, twice, and be dropped for good."""
    many = tuple(f"test_{n} (validators.tests.TestValidators)" for n in range(1500))
    sb = FakeSandbox()
    res = check(sb, r=row(test_patch=TXT_ONLY, FAIL_TO_PASS=many[:400], PASS_TO_PASS=many[400:]))
    assert res.tests_from == "ids" and res.expected == 1500
    assert sb.tests[0]["timeout"] == 3000 and res.attempts[0].timeout_s == 3000
    plain = FakeSandbox()
    assert check(plain).attempts[0].timeout_s == 1800 and plain.tests[0]["timeout"] == 1800


# ---------------------------------------------------------------- round 4, item 6: the resolver's Mixin rule (N4)


def test_plant_a_mixin_whose_base_comes_from_elsewhere_is_never_a_loaded_test_class(tmp_path: Path) -> None:
    """N4 survived round 3: every mixin above has no base, which the base rule already skips. Here each mixin's base
    comes from another module (a name, an attribute), so without the rule that a class named *Mixin is never one
    unittest loads, its own labels would be expected, and never run."""
    source = """
from django import forms
from django.test import TestCase

from .helpers import AssertionHelpers


class ValidationMixin(AssertionHelpers):
    def test_shared(self):
        \"\"\"Shared by every validator test.\"\"\"


class FormMixin(forms.Form):
    def test_form(self):
        \"\"\"A form check.\"\"\"


class ValidatorTests(ValidationMixin, TestCase):
    pass


class FormTests(FormMixin, TestCase):
    pass
"""
    (tmp_path / "tests" / "app").mkdir(parents=True)
    (tmp_path / "tests" / "app" / "tests.py").write_text(source)
    wanted = ["Shared by every validator test.", "A form check."]
    out = subprocess.run(
        [sys.executable, "-I", "-c", DOCSTRINGS, json.dumps(wanted), json.dumps(["tests/app/tests.py"])],
        cwd=tmp_path, capture_output=True, text=True, timeout=60, check=True,
    ).stdout  # fmt: skip
    assert parse_docstring_labels(out, wanted) == {
        wanted[0]: ("app.tests.ValidatorTests.test_shared",),
        wanted[1]: ("app.tests.FormTests.test_form",),
    }


# ---------------------------------------------------------------- round 5, item 2: expected failures, as swebench

PYTEST_TESTS = (
    "diff --git a/pkg/tests/test_a.py b/pkg/tests/test_a.py\n--- a/pkg/tests/test_a.py\n+++ b/pkg/tests/test_a.py\n"
    "@@ -1 +1 @@\n-x\n+y\n"
)
FIX, OLD, MARKED = "pkg/tests/test_a.py::test_fix", "pkg/tests/test_a.py::test_old", "pkg/tests/test_a.py::test_marked"


def pytest_row(
    repo: str = "astropy/astropy", f2p: tuple[str, ...] = (FIX,), p2p: tuple[str, ...] = (OLD, MARKED)
) -> Row:
    owner, name = repo.split("/")
    return row(f"{owner}__{name}-1", repo=repo, version="5.0", test_patch=PYTEST_TESTS, FAIL_TO_PASS=f2p,
               PASS_TO_PASS=p2p)  # fmt: skip


def pytest_spec(repo: str = "astropy/astropy") -> RepoSpec:
    return RepoSpec(repo=repo, version="5.0", test_cmd="pytest -rA", runner=RunnerKind.PYTEST)


def passed(name: str, file: str | None = None) -> CaseResult:
    return CaseResult(name=name, outcome=Outcome.PASS, file=file)


def xfailed(name: str, file: str | None = None) -> CaseResult:
    return CaseResult(name=name, outcome=Outcome.SKIPPED, message="known bug", file=file, expected_failure=True)


def skipped(name: str, file: str | None = None) -> CaseResult:
    return CaseResult(name=name, outcome=Outcome.SKIPPED, message="needs a GPU", file=file)


class ReportingRunner:
    """A runner that reads its cases back from the run's output, as the run phase's runners do: here the output is the
    scripted cases as JSON. It binds to a root and names the stream it reads, as the attack lane's runners do."""

    def __init__(self, kind: RunnerKind, root: str = "/testbed") -> None:
        self.kind = kind
        self.root = root
        self.stream = {RunnerKind.PYTEST: "none", RunnerKind.DJANGO: "stderr", RunnerKind.SYMPY: "stdout"}[kind]

    def with_root(self, root: str) -> ReportingRunner:
        return ReportingRunner(self.kind, root)

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["run", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return tuple(CaseResult.model_validate(c) for c in json.loads(output))


REPORTING = {k: ReportingRunner(k) for k in RunnerKind}


def by_contract(cases: Sequence[CaseResult], expect: Sequence[str], repo: str) -> RunResult:
    """contracts.Sandbox.run_test's expect rule over the cases the runner it was given read, with the contract's own
    matching (contracts.answers, in the prepared target's repository): a missing expected case is ERROR (INFRA here),
    a typed failure is that failure, a skipped expected case is SKIPPED, else PASS."""
    chosen = [c for c in cases if any(answers(c, w, repo=repo) for w in expect)]
    missing = [w for w in expect if not any(answers(c, w, repo=repo) for c in chosen)]
    if missing:
        return result(Outcome.ERROR, failure_type=FailureType.INFRA, problem=f"expected case {missing[0]} never ran")
    typed = [c for c in chosen if c.failure_type is not None and c.outcome is not Outcome.SKIPPED]
    if typed:
        return result(typed[0].outcome, failure_type=typed[0].failure_type)
    if any(c.outcome is Outcome.SKIPPED for c in chosen):
        return result(Outcome.SKIPPED, exit_code=0)
    return result(Outcome.PASS)


class CaseRunPhase(FakeSandbox):
    """The run phase's path through the runner R24 hands run_test, as the attack lane's DockerSandbox takes it: bind the
    runner to the workspace root (keeping the bound one only when it is of the same type), read the stream it names,
    have it read the run's cases, then reduce them with the contract's expect rule. Each instance's run reports
    `cases[instance]`."""

    def __init__(self, cases: Mapping[str, Sequence[CaseResult]], **kw: Any) -> None:
        super().__init__(**kw)
        self.cases = {k: list(v) for k, v in cases.items()}
        self.bound: list[object] = []
        self.streams: list[object] = []
        self.repos: dict[str, str] = {}  # workspace -> the repository of the target it was prepared from

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        ws = super().prepare(target, side, patch=patch)
        self.repos[ws.id] = target.repo
        return ws

    def run_test(  # type: ignore[override]
        self, workspace: Workspace, tests: Sequence[str], *, runner: Any, expect: Sequence[str] | None = None, **kw: Any
    ) -> RunResult:
        super().run_test(workspace, tests, runner=runner, expect=expect, **kw)
        bound = runner.with_root(workspace.root) if callable(getattr(runner, "with_root", None)) else runner
        bound = bound if isinstance(bound, type(runner)) else runner
        self.bound.append(bound)
        self.streams.append(getattr(bound, "stream", "both"))
        output = json.dumps([c.model_dump(mode="json") for c in self.cases[self.prepared[-1][0].split("~", 1)[1]]])
        return by_contract(bound.cases(output, None, kw["changed_files"]), expect or (), self.repos[workspace.id])


def check_cases(r: Row, spec: RepoSpec, cases: Sequence[CaseResult]) -> tuple[R24Result, CaseRunPhase]:
    sb = CaseRunPhase({r.instance_id: cases})
    return check_instance(r, IMAGE, spec, sb, REPORTING, run_id="t"), sb  # type: ignore[arg-type]


def test_plant_an_expected_failure_passes_r24_where_swebench_grades_xfail_as_passing() -> None:
    """swebench 4.1.0, harness/grading.py test_passed (lines 27 to 28): a test passes when its status is PASSED or
    XFAIL, and the pytest parsers read -rA's `XFAIL <id>` line as XFAIL. So a gold whose PASS_TO_PASS or FAIL_TO_PASS
    case failed as expected is resolved there, and R24 keeps it; before, R24 read the case as SKIPPED and dropped the
    gold (gold_skips). An expected failure outside the expected cases changes nothing, and the attempt names the
    expected ones it counted."""
    unlisted = "pkg/tests/test_a.py::test_unlisted"
    p2p, sb = check_cases(pytest_row(), pytest_spec(), [passed(FIX), passed(OLD), xfailed(MARKED), xfailed(unlisted)])
    assert p2p.verdict is R24Verdict.KEPT and p2p.xfail_passes and len(p2p.attempts) == 1
    assert p2p.attempts[0].expected_failures == (MARKED,)
    assert sb.tests[0]["expect"] == (FIX, OLD, MARKED)  # the expect rule is the sandbox's, unchanged
    f2p, _ = check_cases(pytest_row(), pytest_spec(), [xfailed(FIX), passed(OLD), passed(MARKED)])
    assert f2p.verdict is R24Verdict.KEPT and f2p.attempts[0].expected_failures == (FIX,)


def test_plant_a_plain_skip_still_drops_the_gold() -> None:
    """A case that never ran is SKIPPED with no expected failure: it never passes, as before (gold_skips), and an
    expected failure beside it does not save the gold."""
    plain, _ = check_cases(pytest_row(), pytest_spec(), [passed(FIX), passed(OLD), skipped(MARKED)])
    assert plain.verdict is R24Verdict.GOLD_SKIPS and plain.attempts[0].expected_failures == ()
    both, _ = check_cases(pytest_row(), pytest_spec(), [xfailed(FIX), passed(OLD), skipped(MARKED)])
    assert both.verdict is R24Verdict.GOLD_SKIPS and both.attempts[0].expected_failures == (FIX,)
    # The round-5 review's V10: only CaseResult.expected_failure says a case failed as expected, never its words. A
    # plain skip whose reason reads like an xfail is still a plain skip, so it still drops the gold.
    worded = CaseResult(name=MARKED, outcome=Outcome.SKIPPED, message="XFAIL reason: [pytest.mark.xfail] flaky on CI")
    looks, _ = check_cases(pytest_row(), pytest_spec(), [passed(FIX), passed(OLD), worded])
    assert looks.verdict is R24Verdict.GOLD_SKIPS and looks.attempts[0].expected_failures == ()


@pytest.mark.parametrize(
    ("repo", "kept"),
    [
        ("astropy/astropy", True),  # parse_log_pytest_v2
        ("scikit-learn/scikit-learn", True),
        ("sphinx-doc/sphinx", True),
        ("matplotlib/matplotlib", True),  # parse_log_matplotlib
        ("psf/requests", True),  # parse_log_pytest_options
        ("pylint-dev/pylint", True),
        ("pytest-dev/pytest", True),  # parse_log_pytest
        ("pydata/xarray", True),
        ("pallets/flask", True),
        ("mwaskom/seaborn", False),  # parse_log_seaborn reads only FAILED and PASSED: an XFAIL is absent, so failed
    ],
)
def test_plant_each_pytest_repository_grades_an_expected_failure_as_its_swebench_parser_does(
    repo: str, kept: bool
) -> None:
    res, _ = check_cases(pytest_row(repo), pytest_spec(repo), [passed(FIX), passed(OLD), xfailed(MARKED)])
    assert res.xfail_passes is kept and res.attempts[0].expected_failures == (MARKED,)
    assert res.verdict is (R24Verdict.KEPT if kept else R24Verdict.GOLD_SKIPS)


def test_plant_an_expected_failure_still_drops_a_django_or_sympy_gold() -> None:
    """swebench's Django parser (log_parsers/python.py, parse_log_django, line 64) has no rule for unittest's
    'expected failure', and its sympy parser (parse_log_sympy, line 199) none for sympy's 'f': the test is absent from
    the status map, which test_failed (grading.py, lines 31 to 35) counts as failed, so the harness grades the gold
    unresolved. R24 keeps such a case SKIPPED, a drop, and still records it."""
    django, _ = check_cases(row(), SPEC, [passed(LABELS[0]), passed(LABELS[1]), xfailed(LABELS[2])])
    assert django.verdict is R24Verdict.GOLD_SKIPS and not django.xfail_passes
    assert django.attempts[0].expected_failures == (LABELS[2],)
    sympy = RepoSpec(repo="sympy/sympy", version="1.1", test_cmd="bin/test -C --verbose", runner=RunnerKind.SYMPY)
    tests = "diff --git a/sympy/core/tests/test_a.py b/sympy/core/tests/test_a.py\n+x\n"
    r = row("sympy__sympy-1", repo="sympy/sympy", version="1.1", test_patch=tests, FAIL_TO_PASS=("test_fix",),
            PASS_TO_PASS=("test_old",))  # fmt: skip
    home = "sympy/core/tests/test_a.py"
    res, _ = check_cases(r, sympy, [passed("test_fix", home), xfailed("test_old", home)])
    assert res.verdict is R24Verdict.GOLD_SKIPS and res.attempts[0].expected_failures == (f"{home}::test_old",)


def test_the_gold_runner_hands_the_sandbox_the_wrapped_runner_bound_and_shares_its_record() -> None:
    """The sandbox binds the runner it is given to the workspace root and reads the stream it names: the gold runner
    binds the runner it wraps, stays a gold runner (the attack lane's DockerSandbox keeps a bound runner only when it is
    of the same type), shares its record of expected failures with the bound copy and names the wrapped runner's
    stream."""
    res, sb = check_cases(pytest_row(), pytest_spec(), [passed(FIX), passed(OLD), xfailed(MARKED)])
    assert res.verdict is R24Verdict.KEPT and sb.streams == ["none"]
    [bound] = sb.bound
    assert isinstance(bound, GoldRunner) and bound.kind is RunnerKind.PYTEST
    gold_runner = GoldRunner(ReportingRunner(RunnerKind.DJANGO, "/x"), xfail_passes=True)  # type: ignore[arg-type]
    rebound = gold_runner.with_root("/testbed")
    assert rebound.root == "/testbed" and gold_runner.root == "/x" and rebound.seen is gold_runner.seen  # type: ignore[attr-defined]
    assert rebound.stream == "stderr"  # type: ignore[attr-defined]
    out = rebound.cases(json.dumps([xfailed("a.tests.T.t").model_dump(mode="json")]), None, ())
    assert [(c.outcome, c.expected_failure) for c in out] == [(Outcome.PASS, False)]
    assert out[0].message.startswith("failed as expected; graded as passing") and len(gold_runner.seen) == 1
    kept_as_is = GoldRunner(ReportingRunner(RunnerKind.DJANGO), xfail_passes=False)  # type: ignore[arg-type]
    plain = [skipped("a.tests.T.s"), xfailed("a.tests.T.t")]
    out = kept_as_is.cases(json.dumps([c.model_dump(mode="json") for c in plain]), None, ())
    assert list(out) == plain and [c.name for c in kept_as_is.seen] == ["a.tests.T.t"]


# pylint-dev__pylint-7080's and 7277's PASS_TO_PASS id, and the case R24's run in its image reported for it (its junit
# report, 2026-10-28): swebench 4.1.0's parse_log_pytest_options (log_parsers/python.py, lines 48 to 57) keeps a slash
# and what follows the last slash of a parameter that starts with one slash.
STDIN = "tests/test_self.py::TestRunTC::test_stdin[/mymodule.py]"
STDIN_CASE = "tests/test_self.py::TestRunTC::test_stdin[/testbed/tests/mymodule.py-mymodule-/testbed/tests/mymodule.py]"
SELF_TESTS = (
    "diff --git a/tests/test_self.py b/tests/test_self.py\n--- a/tests/test_self.py\n+++ b/tests/test_self.py\n"
    "@@ -1 +1 @@\n-x\n+y\n"
)


@pytest.mark.parametrize(
    ("case", "wanted", "repo", "hit"),
    [
        (passed(FIX), FIX, "astropy/astropy", True),
        (passed(f"{FIX}[1-2]"), FIX, "astropy/astropy", True),  # one parametrization of a node id
        (passed(f"{FIX}[1-2]"), f"{FIX}[3]", "astropy/astropy", False),
        (passed("test_old", "sympy/a/tests/test_b.py"), "test_old", "sympy/sympy", True),  # a bare name, any file
        (passed("test_old", "sympy/a/tests/test_b.py"), "sympy/a/tests/test_b.py::test_old", "sympy/sympy", True),
        (passed("test_old", "sympy/a/tests/test_b.py"), "sympy/a/tests/test_c.py::test_old", "sympy/sympy", False),
        (passed("app.tests.T.test_x"), "app.tests.T.test_x", "django/django", True),
        (passed("app.tests.T.test_xy"), "app.tests.T.test_x", "django/django", False),
        # pylint-dev__pylint-7080: swebench 4.1.0's pylint parser records this case under the id SWE-bench lists
        (passed(STDIN_CASE), STDIN, "pylint-dev/pylint", True),
        (passed(STDIN_CASE), STDIN, "astropy/astropy", False),
    ],
)
def test_a_case_answers_an_expected_name_as_the_contract_matches_them(
    case: CaseResult, wanted: str, repo: str, hit: bool
) -> None:
    assert answers(case, wanted, repo=repo) is hit


def test_r24s_matching_is_the_contracts_not_a_copy_of_it() -> None:
    """R24 and the attack lane matched with two copies of one rule (r24.answers, classify.expected) that could drift
    apart; R24 now uses the contract's own function."""
    assert r24.answers is contracts.answers


def pylint_row() -> Row:
    fix = "tests/test_self.py::TestRunTC::test_fix"
    return row("pylint-dev__pylint-1", repo="pylint-dev/pylint", version="2.15", test_patch=SELF_TESTS,
               FAIL_TO_PASS=(fix,), PASS_TO_PASS=(STDIN,))  # fmt: skip


def pylint_spec() -> RepoSpec:
    return RepoSpec(repo="pylint-dev/pylint", version="2.15", test_cmd="pytest -rA", runner=RunnerKind.PYTEST)


def test_plant_a_renamed_pass_to_pass_id_keeps_the_gold_and_names_its_expected_failure() -> None:
    """R24 left pylint-dev__pylint-7080 and 7277 unchecked: test_stdin[/mymodule.py] is the id swebench 4.1.0's pylint
    parser records, not the node id the run reports. R24 hands the id to run_test as it is, and the sandbox's matching
    answers it with the real case; R24's record of the expected cases that failed as expected uses that same
    matching, so the case counts there too."""
    fix = passed("tests/test_self.py::TestRunTC::test_fix")
    kept, sb = check_cases(pylint_row(), pylint_spec(), [fix, passed(STDIN_CASE)])
    assert kept.verdict is R24Verdict.KEPT and kept.unnamed == () and kept.expected == 2
    assert sb.tests[0]["expect"] == ("tests/test_self.py::TestRunTC::test_fix", STDIN)
    xfail, _ = check_cases(pylint_row(), pylint_spec(), [fix, xfailed(STDIN_CASE)])
    assert xfail.verdict is R24Verdict.KEPT and xfail.xfail_passes
    assert xfail.attempts[0].expected_failures == (STDIN_CASE,)
    # Two cases the parser records under the one id: both must pass, as the attack lane's reduce takes them.
    other = "tests/test_self.py::TestRunTC::test_stdin[/elsewhere/mymodule.py]"
    failing = CaseResult(name=other, outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION)
    both, _ = check_cases(pylint_row(), pylint_spec(), [fix, passed(STDIN_CASE), failing])
    assert both.verdict is R24Verdict.GOLD_FAILS


@pytest.mark.parametrize("repo", sorted(LOG_PARSERS))
def test_the_contracts_renames_follow_the_parser_specs_names_for_each_repository(repo: str) -> None:
    """specs.LOG_PARSERS (the eval lane's reading of MAP_REPO_TO_PARSER_PY) and contracts.answers (which cannot import
    it) must agree on which repositories rename: parse_log_pytest_options keeps a slash and the last path component
    of a parameter that starts with one slash, parse_log_matplotlib writes MouseButton.LEFT as 1, and every other
    parser records the id as the run printed it."""
    slash = passed("t.py::test_x[/a/b]")
    mouse = passed("t.py::test_x[MouseButton.LEFT]")
    parser = LOG_PARSERS[repo]
    assert answers(slash, "t.py::test_x[/b]", repo=repo) is (parser == "parse_log_pytest_options")
    assert answers(mouse, "t.py::test_x[1]", repo=repo) is (parser == "parse_log_matplotlib")


def test_the_report_lists_the_expected_failures_counted_and_the_ones_left_skipped(tmp_path: Path) -> None:
    rows = {"astropy__astropy-1": pytest_row(), "django__django-1": row()}
    sb = CaseRunPhase(
        {
            "astropy__astropy-1": [passed(FIX), passed(OLD), xfailed(MARKED)],
            "django__django-1": [passed(LABELS[0]), passed(LABELS[1]), xfailed(LABELS[2])],
        }
    )
    specs = {**SPECS, ("astropy/astropy", "5.0"): pytest_spec()}
    report = run_check(rows, dict.fromkeys(rows, IMAGE), specs, sorted(rows), lambda: sb, lambda: REPORTING,  # type: ignore[arg-type, return-value]
                       tmp_path, run_id="t", workers=1)  # fmt: skip
    assert report.xfail_counted == {"astropy__astropy-1": 1} and report.xfail_skipped == {"django__django-1": 1}
    assert report.dropped == {"django__django-1": "gold_skips"} and report.verdicts == {"gold_skips": 1, "kept": 1}
    base = (tmp_path / "results.jsonl").read_text()
    assert merge(base, rows, dict.fromkeys(rows, IMAGE)).verdicts == {
        "astropy__astropy-1": True,
        "django__django-1": False,
    }


# ---------------------------------------------------------------- round 5, item 3: the harness's locale exports

# eval/test-specs.json, from swebench 4.1.0: django/django 2.2's eval_commands, and 3.0's (3.1 and 3.2 have 3.0's)
DJANGO_22 = (
    "export LANG=en_US.UTF-8",
    "export LC_ALL=en_US.UTF-8",
    "export PYTHONIOENCODING=utf8",
    "export LANGUAGE=en_US:en",
)
DJANGO_30 = (
    "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
    "export LANG=en_US.UTF-8",
    "export LANGUAGE=en_US:en",
    "export LC_ALL=en_US.UTF-8",
)
EXPORTS_22 = ["LANG=en_US.UTF-8", "LC_ALL=en_US.UTF-8", "PYTHONIOENCODING=utf8", "LANGUAGE=en_US:en"]
EXPORTS_30 = ["LANG=en_US.UTF-8", "LANGUAGE=en_US:en", "LC_ALL=en_US.UTF-8"]


class Recording(FakeSandbox):
    """Records the test command of every target it prepares."""

    def __init__(self, **kw: Any) -> None:
        super().__init__(**kw)
        self.commands: list[str] = []

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        self.commands.append(target.test_command)
        return super().prepare(target, side, patch=patch)


@pytest.mark.parametrize(
    ("version", "evals", "exports"),
    [("2.2", DJANGO_22, EXPORTS_22), ("3.0", DJANGO_30, EXPORTS_30), ("4.0", (), [])],
)
def test_plant_r24s_gold_target_carries_the_eval_scripts_exports(
    version: str, evals: tuple[str, ...], exports: list[str]
) -> None:
    """The round-4 main review: the run phase set no locale, so Django 1.11 to 3.2 golds ran with ASCII encodings while
    the harness's eval script exports LANG and LC_ALL=en_US.UTF-8 first. The gold target's command now leads with those
    exports, as NAME=value words the run phase's runners pass to the test; the root-only locale-gen line is not
    carried, and a version that exports nothing keeps SWE-bench's command."""
    spec = SPEC.model_copy(update={"version": version, "eval_commands": evals})
    target = gold_target(row(version=version), IMAGE, spec)
    assert target.test_command == " ".join([*exports, SPEC.test_cmd])
    words = shlex.split(target.test_command)
    assert words[: len(exports)] == exports and " ".join(words[len(exports) :]) == SPEC.test_cmd
    assert "locale-gen" not in target.test_command
    sb = Recording()
    assert check_instance(row(version=version), IMAGE, spec, sb, RUNNERS, run_id="t").kept  # type: ignore[arg-type]
    assert sb.commands == [target.test_command]  # the target R24 prepares and runs is that one


# ---------------------------------------------------------------- round 6: each attempt's own expected failures


class AttemptRunPhase(CaseRunPhase):
    """CaseRunPhase whose instance reports `runs[instance][n]` on its n-th run_test: one case list per attempt."""

    def __init__(self, runs: Mapping[str, Sequence[Sequence[CaseResult]]], **kw: Any) -> None:
        super().__init__({k: v[0] for k, v in runs.items()}, **kw)
        self.runs = {k: [list(c) for c in v] for k, v in runs.items()}

    def run_test(  # type: ignore[override]
        self, workspace: Workspace, tests: Sequence[str], *, runner: Any, expect: Sequence[str] | None = None, **kw: Any
    ) -> RunResult:
        iid = self.prepared[-1][0].split("~", 1)[1]
        self.cases[iid] = self.runs[iid].pop(0)
        return super().run_test(workspace, tests, runner=runner, expect=expect, **kw)


def test_plant_each_attempt_records_its_own_expected_failures_and_the_report_reads_the_last(tmp_path: Path) -> None:
    """The round-5 review's V8 and V9. Each instance's first run misses an expected case (INFRA, so a second attempt
    runs in a fresh container) after reporting one expected failure; its second run reports two others. Each attempt
    names only its own run's expected failures (one record shared by both attempts would give the second three), and
    the report counts the last attempt's two (the first attempt's would give one), counted where swebench grades XFAIL
    as passing (astropy) and left skipped where it does not (django)."""
    astropy, django = pytest_row(), row()
    rows = {astropy.instance_id: astropy, django.instance_id: django}
    sb = AttemptRunPhase(
        {
            astropy.instance_id: [[xfailed(FIX), passed(OLD)], [passed(FIX), xfailed(OLD), xfailed(MARKED)]],
            django.instance_id: [
                [xfailed(LABELS[0]), passed(LABELS[1])],
                [passed(LABELS[0]), xfailed(LABELS[1]), xfailed(LABELS[2])],
            ],
        }
    )
    specs = {**SPECS, ("astropy/astropy", "5.0"): pytest_spec()}
    report = run_check(rows, dict.fromkeys(rows, IMAGE), specs, sorted(rows), lambda: sb, lambda: REPORTING,  # type: ignore[arg-type, return-value]
                       tmp_path, run_id="t", workers=1)  # fmt: skip
    results = {
        r.instance_id: r
        for r in (R24Result.model_validate_json(x) for x in (tmp_path / "results.jsonl").read_text().splitlines())
    }
    a, d = results[astropy.instance_id], results[django.instance_id]
    assert [x.outcome for x in a.attempts] == [Outcome.ERROR, Outcome.PASS] and a.verdict is R24Verdict.KEPT
    assert [x.expected_failures for x in a.attempts] == [(FIX,), (OLD, MARKED)]
    assert [x.outcome for x in d.attempts] == [Outcome.ERROR, Outcome.SKIPPED] and d.verdict is R24Verdict.GOLD_SKIPS
    assert [x.expected_failures for x in d.attempts] == [(LABELS[0],), (LABELS[1], LABELS[2])]
    assert report.xfail_counted == {astropy.instance_id: 2} and report.xfail_skipped == {django.instance_id: 2}


# ---------------------------------------------------------------- round 6: the founder's pre-approved drops (REG-571)

GOLD_DROPS_FILE = Path(__file__).resolve().parents[2] / "eval" / "gold-drops.tsv"


def test_the_committed_drop_file_lands_the_founders_two_pre_approved_drops_at_the_freeze() -> None:
    """REG-571: the founder pre-approved dropping sphinx-doc__sphinx-8265 and 8621 at the freeze, since SWE-bench's log
    parser cut every FAIL_TO_PASS id of both at a space, so R24 leaves both unchecked whatever a run does. The
    runbook's freeze reads eval/gold-drops.tsv: exactly those two lines, each citing REG-571. They land on an R24 base
    that left the pair unchecked; a drop on an instance R24 settled is still refused."""
    every = parse_drops(GOLD_DROPS_FILE.read_text())
    pair = ["sphinx-doc__sphinx-8265", "sphinx-doc__sphinx-8621"]
    drops = every[:2]
    assert [d.instance_id for d in drops] == pair
    assert all(d.reason.startswith("REG-571, pre-approved for the freeze: ") for d in drops)
    # The founder's later drops at the 2026-10-28 freeze, each citing its decision: REG-574 (network-only INFRA) and
    # REG-575 (an id the image's pytest names differently).
    later = {d.instance_id: d.reason.split(":", 1)[0] for d in every[2:]}
    assert later == {
        "psf__requests-1921": "REG-574",
        "psf__requests-2317": "REG-574",
        "psf__requests-2931": "REG-574",
        "psf__requests-5414": "REG-574",
        "matplotlib__matplotlib-20488": "REG-574",
        "astropy__astropy-7606": "REG-575",
    }
    unchecked = []
    for iid in pair:
        r = sphinx_row(iid, ("tests/test_pycode_ast.py::test_unparse[(1,",), ("tests/test_pycode_ast.py::test_b",))
        res = check_instance(r, IMAGE, SPHINX, FakeSandbox(), RUNNERS, run_id="d1")  # type: ignore[arg-type]
        assert res.verdict is R24Verdict.INFRA and res.unnameable
        unchecked.append(res.model_dump_json())
    pool = [POOL[0], *pair]
    base = "\n".join([line(POOL[0], R24Verdict.KEPT), *unchecked])
    m = merge(base, pool, dict.fromkeys(pool, IMAGE), drops=drops)
    assert m.verdicts == {POOL[0]: True, pair[0]: False, pair[1]: False}
    assert all(m.reasons[i].startswith("founder: REG-571, pre-approved") for i in pair)
    assert [i.role for i in m.inputs] == ["base", "drop"]
    with pytest.raises(R24ResultsError, match="only unchecked"):
        merge(base, pool, dict.fromkeys(pool, IMAGE), drops=[*drops, GoldDrop(instance_id=POOL[0], reason="x")])
