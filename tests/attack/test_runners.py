"""The suite runners: the commands they build, and the cases they read back from real recordings.

The Django and sympy recordings in fixtures/runners/ were made in the SWE-bench images (django__django-11999 and
sympy__sympy-13974) under the run phase's own isolation by fixtures/runners/record.py, and the unittest ones under
local Pythons 3.10 and 3.11 by fixtures/runners/record_unittest.py; their docstrings say what each recording holds.
sympy_13974_at3_base.json is AT3's real base run for that pilot instance, as the round-2 review captured it. The
teeth: every probe case is typed as the rules say (a crash the standard library raises for changed code counts, a
crash in the test, a setUp or setUpClass error, an import or name error and a sandbox error do not), the instance's
FAIL_TO_PASS test fails on base and passes with gold, sympy cases name their files, a test module that does not
import is a case of its own, an unexpected pass is a pass, a broken class hook never hides the rest of a run, and a run
whose lines do not add up to the runner's own summary is refused, never guessed. From round 5: every runner runs
under the environment its target's command sets (the locale SWE-bench's Django specs export), an expected failure is
a SKIPPED case that says so (contract v3.1) and never a pass, and the pytest command is one pytest 3.3.1 runs
(pytest3_xfail.json, recorded in the 11999 image's Python 3.6 by record.py).
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from receipts.attack.classify import UNEXPECTED_PASS, classify_suite, exit_agrees, reduce_cases, reduce_run
from receipts.attack.runners import (
    PYTEST_OPTIONS,
    DjangoRunner,
    PytestRunner,
    SympyRunner,
    command_environment,
    django_cases,
    django_label,
    django_label_of,
    expected_cases,
    runner_for,
    sympy_cases,
)
from receipts.contracts import CaseResult, FailureType, Outcome, RunnerKind, RunResult, Side, Target, Tier

_REC = Path(__file__).parent / "fixtures" / "runners"
ROOT = "/testbed"


def recording(name: str) -> dict[str, object]:
    data: dict[str, object] = json.loads((_REC / f"{name}.json").read_text())
    return data


def target(kind: RunnerKind = RunnerKind.PYTEST, command: str = "") -> Target:
    return Target(
        id="entry~instance-1",
        tier=Tier.A,
        repo="o/r",
        base_sha="a" * 40,
        patch="diff --git a/x b/x\n",
        test_runner=kind,
        test_command=command,
    )


def by_name(cases: tuple[CaseResult, ...]) -> dict[str, tuple[Outcome, FailureType | None, str | None]]:
    return {c.name: (c.outcome, c.failure_type, c.raising_file) for c in cases}


# -- commands -----------------------------------------------------------------------------------------------------


def test_pytest_command_puts_the_runners_options_after_the_repositorys() -> None:
    argv = PytestRunner().command(
        target(command="pytest --no-header -rA --tb=no -p no:cacheprovider"), ["t/test_a.py"], report_path="/tmp/r.xml"
    )
    assert argv[:6] == ["pytest", "--no-header", "-rA", "--tb=no", "-p", "no:cacheprovider"]
    rest = argv[6:]
    assert rest[: len(PYTEST_OPTIONS)] == list(PYTEST_OPTIONS)
    assert rest[len(PYTEST_OPTIONS) :] == ["--junitxml=/tmp/r.xml", "t/test_a.py"]
    # The repository's --tb=no comes first, so pytest's last-one-wins gives native tracebacks.
    assert argv.index("--tb=native") > argv.index("--tb=no")


def test_pytest_command_defaults_and_launchers() -> None:
    assert PytestRunner().command(target(), ["t.py"], report_path="/tmp/r.xml")[:3] == ["python", "-m", "pytest"]
    tox = PytestRunner().command(target(command="tox --current-env -epy39 -v --"), ["t.py"], report_path="/tmp/r.xml")
    assert tox[:5] == ["tox", "--current-env", "-epy39", "-v", "--"] and tox[-1] == "t.py"
    env = PytestRunner().command(target(command="PYTHONWARNINGS=ignore pytest -rA"), ["t.py"], report_path="/r")
    assert env[:3] == ["env", "PYTHONWARNINGS=ignore", "pytest"]


@pytest.mark.parametrize("bad", ["--rootdir=/", "-p", "", "a\nb"])
def test_a_test_argument_that_could_be_an_option_is_refused(bad: str) -> None:
    for runner in (PytestRunner(), DjangoRunner(), SympyRunner()):
        with pytest.raises(ValueError):
            runner.command(target(), [bad], report_path="/tmp/r.xml")


def test_a_command_that_sets_a_credential_or_proxy_is_refused() -> None:
    for command in (
        "HTTPS_PROXY=http://p:3128 pytest",
        "API_KEY=x bin/test",
        "GITHUB_TOKEN=x pytest",
        "INDEX=https://user:hunter2@pypi.invalid/simple pytest",  # a credential in the value
    ):
        with pytest.raises(ValueError, match="credential or a proxy") as info:
            PytestRunner().command(target(command=command), ["t.py"], report_path="/r")
        assert "hunter2" not in str(info.value)


def test_django_command_uses_the_settings_from_the_targets_command() -> None:
    command = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
    argv = DjangoRunner().command(target(RunnerKind.DJANGO, command), ["tests/model_fields/tests.py"], report_path="/r")
    assert argv == [
        "python", "tests/runtests.py", "--verbosity", "2", "--parallel", "1", "--settings=test_sqlite",
        "model_fields.tests",
    ]  # fmt: skip
    other = DjangoRunner().command(
        target(RunnerKind.DJANGO, "x --settings custom_settings"), ["a.tests"], report_path="/r"
    )
    assert "--settings=custom_settings" in other
    assert "--settings=test_sqlite" in DjangoRunner().command(target(RunnerKind.DJANGO), ["a.tests"], report_path="/r")
    with pytest.raises(ValueError):
        DjangoRunner().command(target(RunnerKind.DJANGO, "--settings=../evil"), ["a.tests"], report_path="/r")


@pytest.mark.parametrize(
    ("test", "label"),
    [
        ("tests/model_fields/tests.py", "model_fields.tests"),
        ("tests/model_fields/test_foo.py::FooTests::test_bar", "model_fields.test_foo.FooTests.test_bar"),
        ("model_fields.tests.GetFieldDisplayTests", "model_fields.tests.GetFieldDisplayTests"),
        # swebench 4.1.0's own mapping (harness/test_spec/python.py, get_test_directives, lines 251 to 259): a digit
        # may start a segment, a file that is not a .py keeps its ending, and only a leading tests/ goes.
        (
            "tests/migrations/test_migrations_namespace_package/0001_initial.py",
            "migrations.test_migrations_namespace_package.0001_initial",
        ),
        ("tests/template_tests/templates/inclusion-extends1.html", "template_tests.templates.inclusion-extends1.html"),
        ("tests/admin_scripts/tests/test_x.py", "admin_scripts.tests.test_x"),
        ("django/contrib/admin/tests.py", "django.contrib.admin.tests"),
    ],
)
def test_django_labels(test: str, label: str) -> None:
    assert django_label(test) == label


@pytest.mark.parametrize(
    "bad",
    ["tests/a b.py", "tests/$(id).py", "tests/x'y.py", "tests/a;b.py", "tests/-x.py", "tests/", "tests/a.py::T::m n"],
)
def test_a_label_the_harnesss_shell_would_change_or_runtests_would_read_as_an_option_is_refused(bad: str) -> None:
    with pytest.raises(ValueError, match="does not make a Django test label"):
        django_label(bad)


_ROWS = json.loads((_REC / "django_rows.json").read_text())["rows"]
# Django 3.1's command as the eval lane builds it (specs.run_phase_test_command): its exports, then its test_cmd.
SPEC_31 = (
    "LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 "
    "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
)


def test_a_django_run_gets_the_labels_the_harness_gives_runtests_for_the_test_patch_of_11141() -> None:
    # PLANT (round 5's verify:main): the label check wanted every segment to start with a letter or an underscore, so
    # the runner refused django__django-11141's tests/migrations/test_migrations_namespace_package/0001_initial.py
    # and R24 read the instance as infra on every attempt. swebench 4.1.0's get_test_directives, run on this row
    # with the pinned wheel, gives runtests.py these two labels: R24 passes the test patch's files (its pattern,
    # test_spec/python.py line 244), and the runner must make the same labels of them.
    row = _ROWS["django__django-11141"]
    files = re.findall(r"diff --git a/.* b/(.*)", row["test_patch"])
    argv = DjangoRunner().command(target(RunnerKind.DJANGO, SPEC_31), files, report_path="/tmp/r")
    assert argv[argv.index("--settings=test_sqlite") + 1 :] == [
        "migrations.test_loader",
        "migrations.test_migrations_namespace_package.0001_initial",
    ]
    assert argv[:4] == ["env", "LANG=en_US.UTF-8", "LANGUAGE=en_US:en", "LC_ALL=en_US.UTF-8"]
    assert expected_cases(RunnerKind.DJANGO, files) == argv[-2:]


def test_swebench_django_ids_become_labels() -> None:
    assert django_label_of("test_overriding_FIELD_display (model_fields.tests.GetFieldDisplayTests)") == (
        "model_fields.tests.GetFieldDisplayTests.test_overriding_FIELD_display"
    )
    # Python 3.11 and later print the method inside the parentheses too.
    assert django_label_of("test_x (app.tests.T.test_x)") == "app.tests.T.test_x"
    with pytest.raises(ValueError):
        django_label_of("A docstring sentence.")


def test_sympy_command_honors_the_targets_environment_and_keyword_filters() -> None:
    command = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
    argv = SympyRunner().command(
        target(RunnerKind.SYMPY, command), ["sympy/a/tests/test_b.py::test_c"], report_path="/r"
    )
    assert argv == [
        "env", "PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning", "python", "bin/test", "-C", "--verbose",
        "sympy/a/tests/test_b.py", "-k", "test_c",
    ]  # fmt: skip
    whole = SympyRunner().command(
        target(RunnerKind.SYMPY), ["sympy/a/tests/test_b.py", "sympy/a/tests/test_b.py::x"], report_path="/r"
    )
    assert whole == ["python", "bin/test", "-C", "--verbose", "sympy/a/tests/test_b.py"]  # a whole file: no -k


def test_sympy_command_puts_every_name_after_one_k() -> None:
    # PLANT (review blocker): one -k per name. bin/test's -k callback replaces the value on each -k, so only the last
    # name ran (fixtures/runners/sympy_k2_one_k_per_name.json). One -k takes every name up to the next option.
    tests = ["sympy/a/tests/test_b.py::test_c", "sympy/a/tests/test_b.py::test_d", "sympy/a/tests/test_e.py::test_c"]
    argv = SympyRunner().command(target(RunnerKind.SYMPY), tests, report_path="/r")
    assert argv == [
        "python", "bin/test", "-C", "--verbose", "sympy/a/tests/test_b.py", "sympy/a/tests/test_e.py",
        "-k", "test_c", "test_d",
    ]  # fmt: skip
    assert argv.count("-k") == 1 and argv[-1] == "test_d"
    for bad in ("sympy/a/tests/test_b.py::--pdb", "sympy/a/tests/test_b.py::a b", "sympy/a/tests/test_b.py::"):
        with pytest.raises(ValueError, match="sympy test function"):
            SympyRunner().command(target(RunnerKind.SYMPY), [bad], report_path="/r")


def test_a_recorded_two_name_run_reports_both_cases() -> None:
    # Recorded in the sympy__sympy-13974 image with the command SympyRunner builds for two names of one file.
    rec = recording("sympy_k2")
    two = [
        "sympy/utilities/tests/test_receipts_two.py::test_two_a",
        "sympy/utilities/tests/test_receipts_two.py::test_two_b",
    ]
    assert rec["argv"] == SympyRunner().command(target(RunnerKind.SYMPY), two, report_path="/r")
    cases = sympy_cases(str(rec["stdout"]), root=ROOT, changed_files=[])
    assert [(c.name, c.outcome) for c in cases] == [("test_two_a", Outcome.FAIL), ("test_two_b", Outcome.FAIL)]
    assert reduce_cases(cases, expect=expected_cases(RunnerKind.SYMPY, two)) == (Outcome.FAIL, FailureType.ASSERTION)
    # The same run with one -k per name ran only the last one, so a two-test repro could never be read.
    old = sympy_cases(str(recording("sympy_k2_one_k_per_name")["stdout"]), root=ROOT, changed_files=[])
    assert [c.name for c in old] == ["test_two_b"]
    assert reduce_cases(old, expect=expected_cases(RunnerKind.SYMPY, two)) == (Outcome.ERROR, FailureType.INFRA)


def test_expected_cases_match_what_each_runner_reports() -> None:
    assert expected_cases(RunnerKind.SYMPY, ["sympy/a/tests/test_b.py::test_c"]) == ["sympy/a/tests/test_b.py::test_c"]
    assert expected_cases(RunnerKind.DJANGO, ["tests/a/tests.py::T::test_x"]) == ["a.tests.T.test_x"]
    assert expected_cases(RunnerKind.PYTEST, ["t.py::test_x"]) == ["t.py::test_x"]


def test_runner_for_and_binding() -> None:
    for kind, cls in (
        (RunnerKind.PYTEST, PytestRunner),
        (RunnerKind.DJANGO, DjangoRunner),
        (RunnerKind.SYMPY, SympyRunner),
    ):
        runner = runner_for(target(kind), root="/workspace")
        assert isinstance(runner, cls) and runner.kind is kind
        assert runner.root == "/workspace" and runner.with_root("/testbed").root == "/testbed"
    assert (PytestRunner().stream, DjangoRunner().stream, SympyRunner().stream) == ("none", "stderr", "stdout")
    with pytest.raises(ValueError):
        PytestRunner("relative/root")


# -- Django, from the recordings ------------------------------------------------------------------------------------


def test_django_probe_cases_are_typed_by_the_rules() -> None:
    rec = recording("django_probe")
    cases = django_cases(str(rec["stderr"]), root=ROOT, changed_files=["django/utils/text.py"])
    got = by_name(cases)
    p = "receipts_probe.tests.ProbeTests."
    assert got[p + "test_pass"] == (Outcome.PASS, None, None)
    assert got[p + "test_docstring_pass"] == (Outcome.PASS, None, None)  # its status follows a docstring line
    assert got[p + "test_prints"] == (Outcome.PASS, None, None)  # its status follows the test's own stderr output
    assert got[p + "test_assertion"] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert got[p + "test_docstring_fail"] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert got[p + "test_fail_raises"] == (Outcome.FAIL, FailureType.ASSERTION, None)  # assertRaises: not raised
    assert got[p + "test_crash_in_changed"] == (Outcome.FAIL, FailureType.CRASH_IN_CHANGED, "django/utils/text.py")
    assert got[p + "test_crash_in_library_from_changed"] == (
        Outcome.FAIL,
        FailureType.CRASH_IN_CHANGED,
        "django/utils/text.py",
    )
    assert got[p + "test_crash_in_test"] == (Outcome.ERROR, FailureType.CRASH_ELSEWHERE, None)
    assert got[p + "test_chained"] == (Outcome.ERROR, FailureType.CRASH_ELSEWHERE, None)
    assert got[p + "test_message_imitates_a_frame"] == (Outcome.ERROR, FailureType.CRASH_ELSEWHERE, None)
    assert got[p + "test_import_error"] == (Outcome.ERROR, FailureType.IMPORT, None)
    assert got[p + "test_name_error"] == (Outcome.ERROR, FailureType.NAME, None)
    assert got[p + "test_network"] == (Outcome.ERROR, FailureType.INFRA, None)  # ENETUNREACH, really raised
    assert got[p + "test_read_only_root"] == (Outcome.ERROR, FailureType.INFRA, None)  # EROFS, really raised
    assert got[p + "test_skipped"] == (Outcome.SKIPPED, None, None)
    assert got[p + "test_expected_failure"] == (Outcome.SKIPPED, None, None)
    # An unexpected success passed: not a failure and not infra, with a message that says it was unexpected.
    assert got[p + "test_unexpected_success"] == (Outcome.PASS, None, None)
    assert {c.name: c.message for c in cases}[p + "test_unexpected_success"] == UNEXPECTED_PASS
    assert got["receipts_probe.tests.SetUpFails.test_never_runs"] == (Outcome.ERROR, FailureType.FIXTURE, None)
    assert got["receipts_probe.tests.SetUpClassFails.setUpClass"] == (Outcome.ERROR, FailureType.FIXTURE, None)
    assert len(cases) == 20 and exit_agrees(RunnerKind.DJANGO, int(str(rec["exit_code"])), cases)


def test_django_f2p_fails_on_base_and_passes_with_gold() -> None:
    f2p = "model_fields.tests.GetFieldDisplayTests.test_overriding_FIELD_display"
    changed = ["django/db/models/fields/__init__.py"]
    base = by_name(django_cases(str(recording("django_f2p_base")["stderr"]), root=ROOT, changed_files=changed))
    gold = by_name(django_cases(str(recording("django_f2p_gold")["stderr"]), root=ROOT, changed_files=changed))
    assert base[f2p] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert gold[f2p] == (Outcome.PASS, None, None) and len(gold) == 5


def test_a_missing_django_label_is_an_import_error() -> None:
    (case,) = django_cases(str(recording("django_missing_label")["stderr"]), root=ROOT, changed_files=[])
    assert (case.outcome, case.failure_type) == (Outcome.ERROR, FailureType.IMPORT)


def test_django_output_that_disagrees_with_its_summary_is_refused() -> None:
    text = str(recording("django_f2p_gold")["stderr"])
    # PLANT: a status line lost (or rewritten by the test's own output); unittest still ran 5 tests.
    lost = text.replace("test_iterator_choices (model_fields.tests.GetFieldDisplayTests) ... ok\n", "")
    with pytest.raises(ValueError, match="ran 5 tests but 4"):
        django_cases(lost, root=ROOT, changed_files=[])
    forged = text.replace("Ran 5 tests", "Ran 6 tests")
    with pytest.raises(ValueError):
        django_cases(forged, root=ROOT, changed_files=[])
    base = str(recording("django_f2p_base")["stderr"])
    with pytest.raises(ValueError, match="failures"):
        django_cases(base.replace("FAILED (failures=1)", "FAILED (failures=2)"), root=ROOT, changed_files=[])
    no_block = base.split("=" * 70)[0] + "-" * 70 + "\nRan 5 tests in 0.001s\n\nFAILED (failures=1)\n"
    with pytest.raises(ValueError, match="no FAIL block"):
        django_cases(no_block, root=ROOT, changed_files=[])


def test_empty_django_output_has_no_cases() -> None:
    assert django_cases("", root=ROOT, changed_files=[]) == ()


# -- sympy, from the recordings -------------------------------------------------------------------------------------


def test_sympy_probe_cases_are_typed_by_the_rules() -> None:
    rec = recording("sympy_probe")
    cases = sympy_cases(str(rec["stdout"]), root=ROOT, changed_files=["sympy/utilities/misc.py"])
    got = by_name(cases)
    assert got["test_pass"] == (Outcome.PASS, None, None)
    assert got["test_prints"] == (Outcome.PASS, None, None)  # its status is on the line after its own output
    assert got["test_assertion"] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert got["test_raises_did_not_raise"] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert got["test_crash_in_changed"] == (Outcome.FAIL, FailureType.CRASH_IN_CHANGED, "sympy/utilities/misc.py")
    assert got["test_crash_in_library_from_changed"] == (
        Outcome.FAIL,
        FailureType.CRASH_IN_CHANGED,
        "sympy/utilities/misc.py",
    )
    assert got["test_crash_in_test"] == (Outcome.ERROR, FailureType.CRASH_ELSEWHERE, None)
    assert got["test_chained"] == (Outcome.ERROR, FailureType.CRASH_ELSEWHERE, None)
    assert got["test_message_imitates_a_frame"] == (Outcome.ERROR, FailureType.CRASH_ELSEWHERE, None)
    assert got["test_import_error"] == (Outcome.ERROR, FailureType.IMPORT, None)
    assert got["test_name_error"] == (Outcome.ERROR, FailureType.NAME, None)
    assert got["test_network"] == (Outcome.ERROR, FailureType.INFRA, None)
    assert got["test_read_only_root"] == (Outcome.ERROR, FailureType.INFRA, None)
    assert got["test_skipped"] == (Outcome.SKIPPED, None, None)
    assert got["test_expected_failure"] == (Outcome.SKIPPED, None, None)
    assert got["test_unexpected_success"] == (Outcome.PASS, None, None)  # X: it passed, unexpectedly
    assert {c.name: c.message for c in cases}["test_unexpected_success"] == UNEXPECTED_PASS
    assert {c.file for c in cases} == {"sympy/utilities/tests/test_receipts_probe.py"}  # every case names its file
    assert len(cases) == 16 and exit_agrees(RunnerKind.SYMPY, int(str(rec["exit_code"])), cases)


def test_sympy_f2p_fails_on_base_and_passes_with_gold() -> None:
    changed = ["sympy/physics/quantum/tensorproduct.py"]
    base = by_name(sympy_cases(str(recording("sympy_f2p_base")["stdout"]), root=ROOT, changed_files=changed))
    gold = by_name(sympy_cases(str(recording("sympy_f2p_gold")["stdout"]), root=ROOT, changed_files=changed))
    assert base["test_tensor_product_simp"] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert gold["test_tensor_product_simp"] == (Outcome.PASS, None, None)
    # The image's own DeprecationWarning-as-error, raised in unchanged sympy code: not a counted failure.
    assert gold["test_tensor_product_dagger"] == (
        Outcome.ERROR,
        FailureType.CRASH_ELSEWHERE,
        "sympy/matrices/common.py",
    )


def test_a_sympy_keyword_run_reports_only_its_test() -> None:
    (case,) = sympy_cases(str(recording("sympy_probe_k")["stdout"]), root=ROOT, changed_files=[])
    assert (case.name, case.outcome) == ("test_assertion", Outcome.FAIL)


def test_sympy_output_that_disagrees_with_its_summary_is_refused() -> None:
    text = str(recording("sympy_f2p_base")["stdout"])
    # PLANT: a test whose own print put a false "ok" where its status goes; the summary still says 1 failed.
    forged = text.replace("test_tensor_product_simp F", "test_tensor_product_simp ok")
    with pytest.raises(ValueError, match="summary counts 5 passed, the progress lines 6"):
        sympy_cases(forged, root=ROOT, changed_files=[])
    lost = text.replace("test_issue_5923 ok\n", "")
    with pytest.raises(ValueError, match="passed"):
        sympy_cases(lost, root=ROOT, changed_files=[])
    no_section = text.split("_" * 80)[0] + "\n== tests finished: 5 passed, 1 failed, 1 exceptions, in 0.01 seconds ==\n"
    with pytest.raises(ValueError, match="printed 0 tracebacks for 1 exceptions and 1 failures"):
        sympy_cases(no_section, root=ROOT, changed_files=[])


def test_a_wrapped_sympy_summary_is_read_whole() -> None:
    # The probe's summary is too long for one line, so sympy wraps it; the counts must still add up.
    assert "expected to fail, \n1 expected" in str(recording("sympy_probe")["stdout"])
    assert len(sympy_cases(str(recording("sympy_probe")["stdout"]), root=ROOT, changed_files=[])) == 16


def test_ansi_colors_do_not_change_sympy_cases() -> None:
    text = str(recording("sympy_probe_k")["stdout"]).replace("test_assertion F", "test_assertion \x1b[31mF\x1b[0m")
    (case,) = sympy_cases(text, root=ROOT, changed_files=[])
    assert case.outcome is Outcome.FAIL


@pytest.mark.parametrize("version", ["310", "311"])
def test_a_test_whose_subtest_failed_takes_its_outcome_from_its_block(version: str) -> None:
    # Real unittest output (the runner Django's DiscoverRunner uses), recorded under Python 3.10 and 3.11 with a test
    # module under tests/ (paths rewritten to /testbed): before 3.11 a test whose subtest failed prints no status and
    # the next test's description follows on its line; from 3.11 each failing subtest prints an indented line.
    text = (_REC / f"unittest_subtests_py{version}.txt").read_text()
    got = by_name(django_cases(text, root=ROOT, changed_files=[]))
    qual = "subtests.tests.T"
    assert got == {
        f"{qual}.test_a_sub": (Outcome.FAIL, FailureType.ASSERTION, None),
        f"{qual}.test_b_ok": (Outcome.PASS, None, None),
        f"{qual}.test_c_sub_error": (Outcome.ERROR, FailureType.CRASH_ELSEWHERE, None),
    }


def test_a_test_that_started_and_reported_nothing_is_refused() -> None:
    text = (_REC / "unittest_subtests_py310.txt").read_text()
    # PLANT: the FAIL block of the test whose only signal is its block is lost.
    cut = text.split("=" * 70 + "\nFAIL:")[0]
    cut += "-" * 70 + "\nRan 3 tests in 0.001s\n\nFAILED (errors=1)\n"
    with pytest.raises(ValueError, match="test_a_sub started but reported no status"):
        django_cases(cut, root=ROOT, changed_files=[])


_TWO_FILES = """sympy/a/tests/test_one.py[1] 
test_same ok                                                              [OK]

sympy/b/tests/test_two.py[1] 
test_same F                                                               [FAIL]


________________________________________________________________________________
________________ sympy/b/tests/test_two.py:test_same _________________
  File "/testbed/sympy/b/tests/test_two.py", line 2, in test_same
    assert 1 == 2
AssertionError

============= tests finished: 1 passed, 1 failed, in 0.01 seconds ==============
"""


def test_two_sympy_files_can_name_a_test_the_same() -> None:
    # PLANT (keyed by name alone): the second file's test_same overwrote the first's, and the counts no longer added
    # up to sympy's summary, so the whole run read as infra. Each case also names its file (contract 65a0d21).
    cases = sympy_cases(_TWO_FILES, root=ROOT, changed_files=[])
    assert [(c.name, c.file, c.outcome) for c in cases] == [
        ("test_same", "sympy/a/tests/test_one.py", Outcome.PASS),
        ("test_same", "sympy/b/tests/test_two.py", Outcome.FAIL),
    ]


# -- sympy: the real runs this round's review could not read -------------------------------------------------------

_AT3_CHANGED = ["sympy/physics/quantum/tensorproduct.py"]


def test_the_real_at3_base_run_for_13974_reads_to_every_case() -> None:
    # AT3's base run over the 7 files it selects for this pilot instance, recorded in the image (round-2 review).
    # PLANT (review blocker): it was refused, for three reasons: the module-level exceptions' "EEE" put in front of
    # the next file's header, the summary wrapped before "in 43.02 seconds", and 60 exceptions counted against 57.
    rec = recording("sympy_13974_at3_base")
    out = str(rec["stdout"])
    assert "EEEsympy/physics/quantum/tests/test_spin.py[38]" in out and "60 exceptions, \nin 43.02 seconds" in out
    cases = sympy_cases(out, root=ROOT, changed_files=_AT3_CHANGED)
    tally: dict[tuple[Outcome, FailureType | None], int] = {}
    for c in cases:
        tally[(c.outcome, c.failure_type)] = tally.get((c.outcome, c.failure_type), 0) + 1
    # sympy's own summary: 673 passed, 61 skipped, 19 expected to fail, 60 exceptions.
    assert tally == {
        (Outcome.PASS, None): 673,
        (Outcome.SKIPPED, None): 61 + 19,
        (Outcome.ERROR, FailureType.CRASH_ELSEWHERE): 57,  # the image's DeprecationWarning-as-error, unchanged code
        (Outcome.ERROR, FailureType.IMPORT): 3,
    }
    modules = {c.name: (c.file, c.message.split(":", 1)[0]) for c in cases if c.failure_type is FailureType.IMPORT}
    assert modules == {
        f"sympy/physics/quantum/tests/test_{m}.py": (f"sympy/physics/quantum/tests/test_{m}.py", "DeprecationWarning")
        for m in ("printing", "qapply", "represent")
    }  # each module that raised at import is a case named by its own file, not the stale header sympy prints
    trace = [c.file for c in cases if c.name == "test_eval_trace"]
    assert trace == ["sympy/physics/quantum/tests/test_density.py", "sympy/physics/quantum/tests/test_tensorproduct.py"]
    assert all(c.file is not None for c in cases)
    coarse = RunResult(side=Side.BASE, outcome=Outcome.ERROR, exit_code=1, fresh_container=False, duration_s=43.0)
    run = classify_suite(
        coarse, runner=SympyRunner(ROOT), output=out, report=None, changed_files=_AT3_CHANGED, timed_out=False
    )
    assert run.problem is None and len(run.cases) == 813


def test_a_sympy_summary_wrapped_before_its_time_is_read() -> None:
    text = str(recording("sympy_f2p_base")["stdout"])
    assert "1 exceptions, in 0.01 seconds" in text
    # PLANT (review blocker): the regex wanted ", in N seconds" on one line.
    wrapped = text.replace("1 exceptions, in 0.01 seconds", "1 exceptions, \nin 0.01 seconds")
    assert len(sympy_cases(wrapped, root=ROOT, changed_files=[])) == len(sympy_cases(text, root=ROOT, changed_files=[]))


def test_test_modules_that_fail_to_import_or_raise_are_cases_named_by_their_files() -> None:
    # Recorded in the 13974 image: a clean module, one that fails to import ("[?]   Failed to import", its [FAIL]
    # wrapped to the next line), one whose own code raises after defining a test (an "E" in front of its own
    # header), and one that raises before defining any (a bare "E" line, and no header).
    rec = recording("sympy_modules")
    cases = sympy_cases(str(rec["stdout"]), root=ROOT, changed_files=[])
    d = "sympy/utilities/tests/test_receipts_"
    error, ok = (Outcome.ERROR, FailureType.IMPORT), (Outcome.PASS, None)
    assert [(c.name, c.file, (c.outcome, c.failure_type)) for c in cases] == [
        ("test_ok", f"{d}a_ok.py", ok),
        (f"{d}b_import_error.py", f"{d}b_import_error.py", error),
        (f"{d}c_module_error.py", f"{d}c_module_error.py", error),
        ("test_defined_before_the_error", f"{d}c_module_error.py", ok),
        (f"{d}d_module_error_first.py", f"{d}d_module_error_first.py", error),
    ]
    assert cases[1].message.startswith("ModuleNotFoundError") and "own code" in cases[2].message
    assert exit_agrees(RunnerKind.SYMPY, int(str(rec["exit_code"])), cases)
    # A module-level exception's traceback must name a test module: one that does not is refused, never guessed.
    text = str(rec["stdout"]).replace(
        'test_receipts_d_module_error_first.py", line 1, in <module>', 'x.py", line 1, in f'
    )
    with pytest.raises(ValueError, match="names no test module"):
        sympy_cases(text, root=ROOT, changed_files=[])


def test_a_run_whose_reporter_crashed_is_refused() -> None:
    # A first module that raises from its own code crashes sympy's reporter (it has no active file yet): the run
    # entered a file and printed no summary.
    rec = recording("sympy_reporter_crash")
    assert "AttributeError" in str(rec["stderr"])
    with pytest.raises(ValueError, match="no 'tests finished' line"):
        sympy_cases(str(rec["stdout"]), root=ROOT, changed_files=[])


def test_sympy_tracebacks_line_up_with_what_the_run_reported() -> None:
    text = str(recording("sympy_f2p_base")["stdout"])
    # PLANT: a traceback whose header names another test than the exception it stands for.
    swapped = text.replace("test_tensorproduct.py:test_tensor_product_dagger", "test_tensorproduct.py:test_other", 1)
    with pytest.raises(ValueError, match="traceback 1 names"):
        sympy_cases(swapped, root=ROOT, changed_files=[])


def test_an_unexpected_pass_is_a_pass_and_its_exit_code_agrees() -> None:
    # PLANT (review blocker): X was ERROR/OTHER, so a run whose only odd case was an X, with sympy's exit 0, was
    # "exit code 0 disagrees with the cases" and every case was thrown away.
    rec = recording("sympy_xpass_only")
    (case,) = sympy_cases(str(rec["stdout"]), root=ROOT, changed_files=[])
    assert (case.outcome, case.failure_type, case.message) == (Outcome.PASS, None, UNEXPECTED_PASS)
    coarse = RunResult(side=Side.BASE, outcome=Outcome.PASS, exit_code=0, fresh_container=False, duration_s=1.0)
    run = classify_suite(
        coarse, runner=SympyRunner(ROOT), output=str(rec["stdout"]), report=None, changed_files=[], timed_out=False
    )
    assert (run.problem, run.result.outcome) == (None, Outcome.PASS)
    for name, code in (("django_unexpected_success_only", 0), ("unittest_unexpected_success_py311", 1)):
        # Django 3.0 exits 0 on an unexpected success; unittest (and later Django) exit 1. Both agree.
        dj = recording(name)
        assert int(str(dj["exit_code"])) == code
        (only,) = django_cases(str(dj["stderr"]), root=ROOT, changed_files=[])
        assert (only.outcome, only.message) == (Outcome.PASS, UNEXPECTED_PASS) and exit_agrees(
            RunnerKind.DJANGO, code, (only,)
        )
        coarse = RunResult(side=Side.BASE, outcome=Outcome.PASS if code == 0 else Outcome.ERROR, exit_code=code,
                           fresh_container=False, duration_s=1.0)  # fmt: skip
        run = classify_suite(
            coarse, runner=DjangoRunner(ROOT), output=str(dj["stderr"]), report=None, changed_files=[], timed_out=False
        )
        assert (run.problem, run.result.outcome) == (None, Outcome.PASS), name


# -- Django: Python 3.11 hooks, and output with no newline -----------------------------------------------------------


@pytest.mark.parametrize("version", ["311", "310"])
def test_a_broken_class_or_module_hook_never_hides_the_rest_of_the_run(version: str) -> None:
    # Real unittest -v output (the result class Django's runner uses), recorded under Python 3.11 and 3.10:
    # a setUpClass that errors and one that skips, a tearDownClass and a setUpModule that error, among other tests.
    # PLANT (review blocker): from 3.11 each hook prints a progress line that "Ran 4 tests" leaves out, and the
    # whole run was refused ("unittest ran 4 tests but 8 reported a status"), so one hook hid every regression.
    rec = recording(f"unittest_hooks_py{version}")
    cases = django_cases(str(rec["stderr"]), root=ROOT, changed_files=[])
    got = by_name(cases)
    h = "hooks.tests."
    assert got[h + "A.test_ok"] == (Outcome.PASS, None, None)
    assert got[h + "B.test_fails"] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert got[h + "C.test_after_the_hooks"] == (Outcome.PASS, None, None)
    assert got[h + "TearDownClassFails.test_runs"] == (Outcome.PASS, None, None)
    for hook in (
        h + "SetUpClassFails.setUpClass",
        h + "TearDownClassFails.tearDownClass",
        "hookmodule.tests.setUpModule",
    ):
        assert got[hook] == (Outcome.ERROR, FailureType.FIXTURE, None), hook
    # From 3.11 a skipping hook has a line of its own; before, it is a bare line naming nothing.
    skips = {name for name, (outcome, _, _) in got.items() if outcome is Outcome.SKIPPED}
    assert skips == ({h + "SetUpClassSkips.setUpClass"} if version == "311" else set())
    assert exit_agrees(RunnerKind.DJANGO, int(str(rec["exit_code"])), cases)
    coarse = RunResult(side=Side.PATCH, outcome=Outcome.ERROR, exit_code=1, fresh_container=False, duration_s=1.0)
    run = classify_suite(
        coarse, runner=DjangoRunner(ROOT), output=str(rec["stderr"]), report=None, changed_files=[], timed_out=False
    )
    assert (run.problem, run.result.outcome, run.result.failure_type) == (None, Outcome.FAIL, FailureType.ASSERTION)


def test_before_311_a_hooks_bare_error_line_never_becomes_the_last_tests_status() -> None:
    # Python 3.10 prints a failing class or module hook as a bare "ERROR" line under the test before it. When that
    # test's status came at the end of its own output ("... xok"), the bare lines must stay the hooks': read as the
    # test's, the run would be refused for an ERROR with no block. (The 3.10 recording, with test_runs writing "x".)
    text = str(recording("unittest_hooks_py310")["stderr"])
    runs = "test_runs (hooks.tests.TearDownClassFails) ... ok\n"
    assert runs + "ERROR\nERROR\n" in text
    got = by_name(django_cases(text.replace(runs, runs.replace("... ok", "... xok")), root=ROOT, changed_files=[]))
    assert got["hooks.tests.TearDownClassFails.test_runs"] == (Outcome.PASS, None, None)
    assert got["hooks.tests.TearDownClassFails.tearDownClass"] == (Outcome.ERROR, FailureType.FIXTURE, None)
    # A bare "skipped" line is ambiguous there: a class-level skip, or the real status of a test whose own output
    # ended in "ok". It is read as the test's skip, which can lose a check but never make a false one.
    after = "test_after_the_hooks (hooks.tests.C) ... ok\n"
    assert after + "ERROR\nskipped 'no backend'\n" in text
    got = by_name(django_cases(text.replace(after, after.replace("... ok", "... xok")), root=ROOT, changed_files=[]))
    assert got["hooks.tests.C.test_after_the_hooks"] == (Outcome.SKIPPED, None, None)


def test_a_hook_that_reports_an_error_with_no_block_is_refused() -> None:
    text = str(recording("unittest_hooks_py311")["stderr"])
    cut = text.split("=" * 70 + "\nERROR: setUpModule")[0] + "=" * 70 + "\nFAIL:" + text.split("=" * 70 + "\nFAIL:")[1]
    cut = cut.replace("errors=3", "errors=2")
    with pytest.raises(ValueError, match="setUpModule reported ERROR with no ERROR block"):
        django_cases(cut, root=ROOT, changed_files=[])


@pytest.mark.parametrize("name", ["django_no_newline", "unittest_no_newline_py311"])
def test_a_status_after_output_with_no_newline_is_read(name: str) -> None:
    # Recorded in the Django 11999 image (Python 3.6) and under unittest on 3.11: tests that write "x" to stderr with
    # no newline, so the status lands at the end of the line ("... xok", "... xFAIL"), and one whose own output ends
    # in "ok" before unittest prints that it skipped.
    # PLANT (integrator call b): "... xok" was "started but reported no status", and the whole run was refused.
    rec = recording(name)
    text = str(rec["stderr"])
    assert "... xok\n" in text and "... xFAIL\n" in text and "... it is ok\nskipped" in text
    got = {
        c.name.rsplit(".", 1)[-1]: (c.outcome, c.failure_type) for c in django_cases(text, root=ROOT, changed_files=[])
    }
    assert got == {
        "test_a_writes_then_passes": (Outcome.PASS, None),
        "test_b_writes_then_fails": (Outcome.FAIL, FailureType.ASSERTION),
        "test_c_ends_its_output_in_ok_then_skips": (Outcome.SKIPPED, None),  # unittest's own status wins
        "test_d_after": (Outcome.PASS, None),
    }


# -- round 4: an assert in a hook, and test modules that fail to import ------------------------------------------------


@pytest.mark.parametrize("version", ["311", "310"])
def test_an_assert_in_a_class_or_module_hook_is_a_fixture_error_never_a_counted_failure(version: str) -> None:
    # Real unittest -v output, recorded under Python 3.11 and 3.10: a setUpClass, a tearDownClass and a setUpModule
    # that each fail an assert. PLANT (the round-3 review's surviving mutant): without the hook rule, an
    # AssertionError raised in a hook reads as a failed assertion, a counted failure, so a class hook's own assert
    # could make a blocker. The lane's other hook recordings raise RuntimeError, which never tested the rule.
    rec = recording(f"unittest_assert_hooks_py{version}")
    cases = django_cases(str(rec["stderr"]), root=ROOT, changed_files=["tests/asserthooks/tests.py"])
    got = by_name(cases)
    for hook in (
        "asserthooks.tests.SetUpClassAsserts.setUpClass",
        "asserthooks.tests.TearDownClassAsserts.tearDownClass",
        "asserthookmodule.tests.setUpModule",
    ):
        assert got[hook] == (Outcome.ERROR, FailureType.FIXTURE, None), hook
        assert {c.name: c.message for c in cases}[hook].startswith("AssertionError: ")
    assert got["asserthooks.tests.B.test_fails"] == (Outcome.FAIL, FailureType.ASSERTION, None)
    assert got["asserthooks.tests.A.test_ok"] == (Outcome.PASS, None, None)
    # A case its class's setUpClass kept from running is still missing (infra), and the problem says which hook.
    _, ftype, problem = reduce_run(cases, expect=["asserthooks.tests.SetUpClassAsserts.test_never_runs"])
    assert ftype is FailureType.INFRA and problem.endswith(
        "(asserthooks.tests.SetUpClassAsserts.setUpClass errored: AssertionError: setUpClass assert)"
    )


@pytest.mark.parametrize("name", ["unittest_failed_imports_py311", "unittest_failed_imports_py310"])
def test_a_test_module_that_fails_to_import_is_a_case_named_by_its_module(name: str) -> None:
    # Real unittest -v runs (the loader Django's runner uses) with labels for two modules called tests that fail to
    # import, a module that imports, a class it does not have, and an app that does not exist. unittest names each
    # failed load by the part of the label that failed, so both modules print 'tests (unittest.loader._FailedTest)'.
    # PLANT: the parser kept one of the two, so 'Ran 5 tests' met 4 statuses and the whole run was refused (infra);
    # and a failed import was named unittest.loader._FailedTest.tests, which says nothing about which module.
    text = str(recording(name)["stderr"])
    progress = [line for line in text.splitlines() if line.startswith("tests (unittest.loader._FailedTest")]
    assert len(progress) == 2 and all(line.endswith("... ERROR") for line in progress)
    cases = django_cases(text, root=ROOT, changed_files=[])
    assert [(c.name, c.outcome, c.failure_type) for c in cases] == [
        ("fine.tests.T.test_ok", Outcome.PASS, None),
        ("brokena.tests", Outcome.ERROR, FailureType.IMPORT),
        ("brokenb.tests", Outcome.ERROR, FailureType.IMPORT),
        ("unittest.loader._FailedTest.Missing", Outcome.ERROR, FailureType.OTHER),  # not an import error
        ("nosuchapp", Outcome.ERROR, FailureType.IMPORT),  # no repository code ran: named by ModuleNotFoundError
    ]
    assert {c.name: c.message for c in cases}[
        "brokena.tests"
    ] == "ModuleNotFoundError: No module named 'receipts_nowhere'"
    expect = {
        "brokena.tests.T.test_x": (FailureType.IMPORT, "its test module brokena.tests failed to import"),
        "brokenb.tests.T.test_y": (FailureType.IMPORT, "its test module brokenb.tests failed to import"),
        "nosuchapp.tests.T.test_z": (FailureType.IMPORT, "its test module nosuchapp failed to import"),
        "fine.tests.Missing.test_q": (FailureType.INFRA, "the run reports no import error for its test module"),
        "fine.tests.T.test_gone": (FailureType.INFRA, "the run reports no import error for its test module"),
    }
    for label, (ftype, why) in expect.items():
        outcome, got, problem = reduce_run(cases, expect=[label])
        assert (
            (outcome, got) == (Outcome.ERROR, ftype)
            and why in problem
            and problem.startswith(f"expected case {label} never ran")
        ), label
    assert reduce_run(cases, expect=["fine.tests.T.test_ok"]) == (Outcome.PASS, None, "")


def test_a_failed_import_is_named_only_by_a_test_module_under_tests() -> None:
    # Crafted from the 3.11 recording: the module that failed lies outside tests/ (a label such as mypkg.tests, the
    # way plain unittest would run one from the repository root). Django's runtests.py labels are relative to tests/,
    # so no label can be told for it, and the case keeps unittest's own name rather than a made-up one.
    text = str(recording("unittest_failed_imports_py311")["stderr"]).replace(
        "/testbed/tests/brokena/", "/testbed/mypkg/"
    )
    names = [c.name for c in django_cases(text, root=ROOT, changed_files=[])]
    assert "unittest.loader._FailedTest.tests" in names and "brokena.tests" not in names and "tests" not in names
    assert "brokenb.tests" in names  # the other module is still named by its own frame
    # Nor does tests/__init__.py name one (it would be the empty label): the run is still read, the name unittest's.
    top = str(recording("unittest_failed_imports_py311")["stderr"]).replace(
        "/testbed/tests/brokena/tests.py", "/testbed/tests/__init__.py"
    )
    assert "unittest.loader._FailedTest.tests" in [c.name for c in django_cases(top, root=ROOT, changed_files=[])]


def test_two_labels_in_one_module_that_failed_to_import_make_one_case() -> None:
    # The 3.11 recording with its second module's paths turned into the first's (crafted from the real run): two
    # labels of one module that failed, as unittest prints them, give one case for that module, not two.
    text = str(recording("unittest_failed_imports_py311")["stderr"]).replace("tests/brokenb/", "tests/brokena/")
    cases = django_cases(text, root=ROOT, changed_files=[])
    assert [c.name for c in cases].count("brokena.tests") == 1 and "brokenb.tests" not in [c.name for c in cases]


def test_django_drops_a_second_failed_import_of_the_same_name_so_its_cases_stay_infra() -> None:
    # Recorded in the django__django-11999 image (Django 3.0, Python 3.6): the same kind of run through
    # tests/runtests.py. Django's runner keeps one test per (class, name) (reorder_suite's OrderedSet, and unittest
    # equates two _FailedTest stand-ins of one name), so only the first module called tests that failed is reported.
    # A case of the second never ran and nothing in the run says why: infra, as contract v3 says, never a guess.
    rec = recording("django_failed_imports")
    cases = django_cases(str(rec["stderr"]), root=ROOT, changed_files=[])
    assert [(c.name, c.failure_type) for c in cases] == [
        ("receipts_fine.tests.T.test_ok", None),
        ("receipts_brokena.tests", FailureType.IMPORT),
        ("receipts_nosuchapp", FailureType.IMPORT),
    ]
    assert exit_agrees(RunnerKind.DJANGO, int(str(rec["exit_code"])), cases)
    assert reduce_cases(cases, expect=["receipts_brokena.tests.T.test_x"]) == (Outcome.ERROR, FailureType.IMPORT)
    assert reduce_cases(cases, expect=["receipts_nosuchapp.tests.T.z"]) == (Outcome.ERROR, FailureType.IMPORT)
    assert reduce_cases(cases, expect=["receipts_brokenb.tests.T.test_y"]) == (Outcome.ERROR, FailureType.INFRA)


def test_a_sympy_case_whose_module_did_not_import_answers_import_by_path_and_by_its_defining_file() -> None:
    # The 13974 recording of modules that fail to import or raise: a case of such a module never ran.
    rec = recording("sympy_modules")
    cases = sympy_cases(str(rec["stdout"]), root=ROOT, changed_files=[])
    d = "sympy/utilities/tests/test_receipts_"
    assert reduce_cases(cases, expect=[f"{d}b_import_error.py::test_never_runs"]) == (
        Outcome.ERROR, FailureType.IMPORT,
    )  # fmt: skip
    assert reduce_cases(cases, expect=[f"{d}d_module_error_first.py::test_never_defined"]) == (
        Outcome.ERROR, FailureType.IMPORT,
    )  # fmt: skip
    # SWE-bench's bare name: the module's only when the run's files say it defines it (run_test reads them).
    homes = {"test_never_runs": (f"{d}b_import_error.py",)}
    assert reduce_cases(cases, expect=["test_never_runs"], homes=homes) == (Outcome.ERROR, FailureType.IMPORT)
    assert reduce_cases(cases, expect=["test_never_runs"]) == (Outcome.ERROR, FailureType.INFRA)
    assert reduce_cases(cases, expect=[f"{d}a_ok.py::test_gone"]) == (Outcome.ERROR, FailureType.INFRA)


def test_each_runner_takes_its_own_ids_at_any_granularity_and_sympy_refuses_a_name_with_no_file() -> None:
    # Contract v3: SuiteRunner.command's tests may be the runner's own ids at any granularity.
    dj = DjangoRunner().command(
        target(RunnerKind.DJANGO),
        ["validators.tests", "validators.tests.T", "validators.tests.T.test_x"],
        report_path="/r",
    )
    assert dj[-3:] == ["validators.tests", "validators.tests.T", "validators.tests.T.test_x"]
    py = PytestRunner().command(target(), ["tests", "tests/t.py", "tests/t.py::K::test_x[1]"], report_path="/r")
    assert py[-3:] == ["tests", "tests/t.py", "tests/t.py::K::test_x[1]"]
    sy = SympyRunner().command(
        target(RunnerKind.SYMPY), ["sympy/core/tests", "sympy/a/tests/t.py::x"], report_path="/r"
    )
    assert sy[-2:] == ["sympy/core/tests", "sympy/a/tests/t.py"] and "-k" not in sy  # a directory runs whole
    # PLANT: a bare sympy name went to bin/test as a path to match, which runs whatever files contain it, or none.
    with pytest.raises(ValueError, match="names no file"):
        SympyRunner().command(target(RunnerKind.SYMPY), ["test_tensor_product_simp"], report_path="/r")


# -- round 5: a target's leading assignments are its run's environment, the same way in every runner --------------

# SWE-bench 4.1.0's Django eval_commands (harness/constants/python.py, SPECS_DJANGO; the eval lane's pinned
# eval/test-specs.json): Django 1.11 and 2.2 export these four before the tests run, and 3.0 to 3.2 these three, after
# a root-only locale-gen line that the run phase has no need of (SWE-bench's base image installs locales-all).
DJANGO_22_EXPORTS = ["LANG=en_US.UTF-8", "LC_ALL=en_US.UTF-8", "PYTHONIOENCODING=utf8", "LANGUAGE=en_US:en"]
DJANGO_30_EXPORTS = ["LANG=en_US.UTF-8", "LANGUAGE=en_US:en", "LC_ALL=en_US.UTF-8"]
SWEBENCH_COMMAND = {
    RunnerKind.PYTEST: "pytest -rA",
    RunnerKind.DJANGO: "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
    RunnerKind.SYMPY: "bin/test -C --verbose",
}
A_TEST = {
    RunnerKind.PYTEST: ["tests/test_a.py"],
    RunnerKind.DJANGO: ["tests/model_fields/tests.py"],
    RunnerKind.SYMPY: ["sympy/a/tests/test_b.py"],
}


def test_django_runs_under_the_locale_its_target_command_exports() -> None:
    # PLANT (round 4's verify:main): DjangoRunner.command kept only --settings and dropped every assignment, so the
    # pool's 55 Django 1.11 to 3.2 instances ran with ASCII encodings where the harness exports a UTF-8 locale.
    command = SWEBENCH_COMMAND[RunnerKind.DJANGO]
    for exports in (DJANGO_22_EXPORTS, DJANGO_30_EXPORTS):
        argv = DjangoRunner().command(
            target(RunnerKind.DJANGO, " ".join([*exports, command])), ["tests/model_fields/tests.py"], report_path="/r"
        )
        assert argv == [
            "env", *exports, "python", "tests/runtests.py", "--verbosity", "2", "--parallel", "1",
            "--settings=test_sqlite", "model_fields.tests",
        ]  # fmt: skip
    # The settings come from the command after its assignments, never from an assignment's value.
    tricky = DjangoRunner().command(
        target(RunnerKind.DJANGO, "X=--settings=other ./tests/runtests.py --settings custom_settings"),
        ["a.tests"],
        report_path="/r",
    )
    assert tricky[:2] == ["env", "X=--settings=other"] and "--settings=custom_settings" in tricky
    assert [a for a in tricky if a.startswith("--settings=")] == ["--settings=custom_settings"]


@pytest.mark.parametrize("kind", list(RunnerKind))
def test_every_runner_treats_leading_assignments_the_same_way(kind: RunnerKind) -> None:
    runner = runner_for(target(kind))
    command, tests = SWEBENCH_COMMAND[kind], A_TEST[kind]
    plain = runner.command(target(kind, command), tests, report_path="/r")
    assert plain[0] != "env"
    # The same command, under the environment the target's command sets, in its order (quoting as a shell reads it).
    for exports in (DJANGO_22_EXPORTS, DJANGO_30_EXPORTS, ["PYTHONWARNINGS=ignore::UserWarning,ignore::SyntaxWarning"]):
        line = " ".join(f"{e.split('=', 1)[0]}='{e.split('=', 1)[1]}'" for e in exports)
        assert runner.command(target(kind, f"{line} {command}"), tests, report_path="/r") == ["env", *exports, *plain]
    # Assignments alone keep the runner's own default command.
    default = runner.command(target(kind), tests, report_path="/r")
    assert runner.command(target(kind, "LANG=C.UTF-8"), tests, report_path="/r") == ["env", "LANG=C.UTF-8", *default]
    # A name that looks like a credential or a proxy is refused by every runner, by name, never echoing its value.
    for bad in (
        "HTTPS_PROXY=http://p:3128",
        "API_KEY=hunter2",
        "GITHUB_TOKEN=hunter2",
        "INDEX=https://user:hunter2@pypi.invalid/simple",
    ):
        with pytest.raises(ValueError, match="credential or a proxy") as info:
            runner.command(target(kind, f"{bad} {command}"), tests, report_path="/r")
        assert "hunter2" not in str(info.value)


def test_command_environment_splits_a_command_the_one_way_every_runner_uses() -> None:
    assert command_environment(target()) == ([], [])
    assert command_environment(target(command="A=1 B='x y' pytest -k 'a b'")) == (
        ["A=1", "B=x y"],
        ["pytest", "-k", "a b"],
    )
    # An option that carries an equals sign is not an assignment, and nothing after the command's first word is one.
    assert command_environment(target(command="--settings=x A=1 cmd")) == ([], ["--settings=x", "A=1", "cmd"])


# -- round 5: an expected failure says so (contract v3.1), and is still never a pass --------------------------------


def _flags(cases: tuple[CaseResult, ...]) -> dict[str, tuple[Outcome, bool]]:
    return {c.name.split(".", 2)[-1]: (c.outcome, c.expected_failure) for c in cases}


@pytest.mark.parametrize(
    "name", ["django_expected_failures", "unittest_expected_failures_py311", "unittest_expected_failures_py310"]
)
def test_a_django_expected_failure_is_flagged_and_stays_skipped(name: str) -> None:
    # Recorded in the django__django-11999 image (Django 3.1, Python 3.6) and under unittest -v on 3.11 and 3.10:
    # expectedFailure tests that fail as expected, plain, after a docstring line, after output with no newline, and a
    # whole class marked, beside an unexpected success, a skip and a pass.
    # PLANT: django_cases read "expected failure" as a plain skip, so R24 could not tell it from one.
    rec = recording(name)
    cases = django_cases(str(rec["stderr"]), root=ROOT, changed_files=[])
    skipped, passed = (Outcome.SKIPPED, True), (Outcome.PASS, False)
    assert _flags(cases) == {
        "ExpectedFailures.test_a_fails_as_expected": skipped,
        "ExpectedFailures.test_b_with_a_docstring": skipped,
        "ExpectedFailures.test_c_writes_then_fails_as_expected": skipped,
        "ExpectedFailures.test_d_passes_unexpectedly": passed,  # an unexpected success stays a pass that says so
        "ExpectedFailures.test_e_skipped": (Outcome.SKIPPED, False),  # a plain skip is not one
        "ExpectedFailures.test_f_ok": passed,
        "WholeClassExpected.test_g_fails_as_expected": skipped,
    }
    assert {c.name.rsplit(".", 1)[-1]: c.message for c in cases}["test_d_passes_unexpectedly"] == UNEXPECTED_PASS
    assert exit_agrees(RunnerKind.DJANGO, int(str(rec["exit_code"])), cases)
    # Still never a pass: a run of an expected failure is SKIPPED, as before (only R24's gold check counts it).
    label = next(c.name for c in cases if c.name.endswith("test_a_fails_as_expected"))
    assert reduce_run(cases, expect=[label]) == (Outcome.SKIPPED, None, "")
    probe = {c.name: c for c in django_cases(str(recording("django_probe")["stderr"]), root=ROOT, changed_files=[])}
    p = "receipts_probe.tests.ProbeTests."
    assert probe[p + "test_expected_failure"].expected_failure and not probe[p + "test_skipped"].expected_failure


@pytest.mark.parametrize(
    "name", ["django_imitated_status", "unittest_imitated_status_py311", "unittest_imitated_status_py310"]
)
def test_output_that_imitates_an_expected_failure_is_refused(name: str) -> None:
    # Recorded in the 11999 image and under unittest -v: a passing test whose own output is the line
    # "expected failure", which lands where unittest prints a status, before its real "ok".
    # PLANT: without unittest's own count of expected failures checked, that test read as an expected failure.
    rec = recording(name)
    assert " ... expected failure\nok\n" in str(rec["stderr"]) and int(str(rec["exit_code"])) == 0
    with pytest.raises(ValueError, match="summary counts 0 expected failures, the progress lines 1"):
        django_cases(str(rec["stderr"]), root=ROOT, changed_files=[])


@pytest.mark.parametrize("name", ["unittest_masked_expected_failure_py311", "unittest_masked_expected_failure_py310"])
def test_an_expected_failure_whose_own_output_reads_ok_is_refused(name: str) -> None:
    # Recorded under unittest -v on 3.11 and 3.10: an expectedFailure test whose own output is the line "ok", which
    # lands where unittest prints a status, before its real "expected failure". The progress line reads "ok", and
    # unittest's own summary counts one expected failure more than the progress lines show.
    # PLANT (the round-5 attack review's X1, runners.py's "!=" as "<"): a check that refused only a summary counting
    # fewer expected failures than the progress lines let this test read as a pass.
    rec = recording(name)
    assert " ... ok\nexpected failure\n" in str(rec["stderr"]) and "OK (expected failures=1)" in str(rec["stderr"])
    with pytest.raises(ValueError, match="summary counts 1 expected failures, the progress lines 0"):
        django_cases(str(rec["stderr"]), root=ROOT, changed_files=[])


def test_a_sympy_xfail_is_flagged_and_stays_skipped() -> None:
    # PLANT: sympy_cases read "f" as a plain skip.
    probe = {c.name: c for c in sympy_cases(str(recording("sympy_probe")["stdout"]), root=ROOT, changed_files=[])}
    assert (probe["test_expected_failure"].outcome, probe["test_expected_failure"].expected_failure) == (
        Outcome.SKIPPED, True,
    )  # fmt: skip
    assert not probe["test_skipped"].expected_failure and not probe["test_unexpected_success"].expected_failure
    # AT3's real base run for 13974: sympy's own summary counts 19 expected to fail and 61 skipped.
    cases = sympy_cases(str(recording("sympy_13974_at3_base")["stdout"]), root=ROOT, changed_files=_AT3_CHANGED)
    xfails = [c for c in cases if c.expected_failure]
    assert len(xfails) == 19 and all(c.outcome is Outcome.SKIPPED and c.file is not None for c in xfails)
    assert sum(1 for c in cases if c.outcome is Outcome.SKIPPED and not c.expected_failure) == 61
    one = xfails[0]
    assert reduce_run(cases, expect=[f"{one.file}::{one.name}"]) == (Outcome.SKIPPED, None, "")


def test_the_pytest_command_is_one_pytest_3_runs() -> None:
    # Recorded with pytest 3.3.1 in the 11999 image's Python 3.6 (record.py pytest5), with astropy 1.3's own command.
    # PLANT (found this round): PYTEST_OPTIONS ended in --rootdir=., an option pytest 3.3.1 does not have, so every run
    # of SWE-bench's astropy 0.1 to 1.3 instances (4 of the pool's 251) exited 2 with no report: infra on every rerun.
    rec = recording("pytest3_xfail")
    forced = rec["with_rootdir"]
    assert isinstance(forced, dict) and forced["junit"] is None and forced["exit_code"] == 2
    assert "error: unrecognized arguments: --rootdir=." in str(forced["stderr"])
    ran = rec["without_rootdir"]
    assert isinstance(ran, dict) and ran["exit_code"] == 0 and ran["junit"]
    pinned = target(command=str(rec["command"]))
    argv = PytestRunner().command(
        pinned, ["tests/receipts_xfail/test_xfail_kinds.py"], report_path="/tmp/receipts-without_rootdir.xml"
    )
    assert argv == ran["argv"]  # the runner's command is the one pytest 3.3.1 ran and reported
    assert not any(a.startswith("--rootdir") for a in PYTEST_OPTIONS)
