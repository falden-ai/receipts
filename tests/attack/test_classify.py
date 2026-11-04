"""The run classifier on recorded pytest 9.1.1 junit samples (no Docker). The samples live in fixtures/junit/, one file
per case, recorded with the pytest runner's own options by fixtures/junit/make_samples.py, which also records each
run's exit code next to it (<name>.rc); its docstring says exactly what was normalized and which samples are crafted.
One report is pytest 3.3.1's, recorded in the django__django-11999 image (fixtures/runners/pytest3_xfail.json), for
the older junit form of an xfail.

The teeth: import, name, fixture and timeout errors never count; a skipped or xfailed test is SKIPPED, never a pass,
and an xfail says so in either junit form (contract v3.1) while a skip never does, whatever its reason reads;
an assertion about a timeout is still an assertion; DID NOT RAISE and DID NOT WARN are assertions (REG-565); a crash
counts only when its deepest repository frame is a changed non-test file; and a run whose exit code disagrees with
its cases, or that wrote no report, is ERROR (infra), with or without ``expect``.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from xml.sax.saxutils import quoteattr

import pytest
from pydantic import ValidationError

from receipts.attack.classify import (
    UNEXPECTED_PASS,
    JunitParseError,
    SuiteRun,
    classify_suite,
    exit_agrees,
    import_failure_of,
    junit_cases,
    nodeid,
    reduce_cases,
    reduce_run,
    with_expect,
)
from receipts.attack.runners import PytestRunner
from receipts.contracts import (
    BLOCKER_RUNS,
    CaseResult,
    CatchKind,
    FailureType,
    Finding,
    Lens,
    Outcome,
    ReplayKey,
    Repro,
    RunnerKind,
    RunResult,
    Side,
    Status,
)

_SAMPLES = Path(__file__).parent / "fixtures" / "junit"
ROOT = "/workspace"
MOD = ["src/pkg/mod.py"]


def sample(name: str) -> str:
    return (_SAMPLES / f"{name}.xml").read_text()


def exit_code(name: str) -> int:
    return int((_SAMPLES / f"{name}.rc").read_text())


def cases_of(name: str, changed: Sequence[str] = ()) -> tuple[CaseResult, ...]:
    return junit_cases(sample(name), root=ROOT, changed_files=changed)


def only(name: str, changed: Sequence[str] = ()) -> tuple[FailureType | None, Outcome]:
    (case,) = cases_of(name, changed)
    return case.failure_type, case.outcome


# -- counted failures --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["assertion", "assertion_message", "unittest_assert"])
def test_assertions_are_counted_failures(name: str) -> None:
    assert only(name) == (FailureType.ASSERTION, Outcome.FAIL)


@pytest.mark.parametrize("name", ["assertion_timeout_attr", "assertion_timed_out_text"])
def test_an_assertion_about_a_timeout_is_still_an_assertion(name: str) -> None:
    assert only(name) == (FailureType.ASSERTION, Outcome.FAIL)


@pytest.mark.parametrize("name", ["did_not_raise", "did_not_warn"])
def test_did_not_raise_and_did_not_warn_are_assertions(name: str) -> None:
    # REG-565: pytest.raises and pytest.warns failures count as assertion failures.
    assert only(name) == (FailureType.ASSERTION, Outcome.FAIL)


def test_another_pytest_fail_is_not_counted() -> None:
    assert only("pytest_fail") == (FailureType.OTHER, Outcome.ERROR)


def test_a_crash_in_a_changed_file_counts_and_names_the_file() -> None:
    (case,) = cases_of("crash", MOD)
    assert (case.failure_type, case.outcome, case.raising_file) == (
        FailureType.CRASH_IN_CHANGED,
        Outcome.FAIL,
        "src/pkg/mod.py",
    )
    assert case.name == "tests/test_crash.py::test_crash_in_changed"


def test_the_same_crash_outside_the_change_does_not_count() -> None:
    assert only("crash", ["src/other/unrelated.py"]) == (FailureType.CRASH_ELSEWHERE, Outcome.ERROR)


def test_a_crash_the_library_raises_for_changed_code_counts() -> None:
    assert only("crash_in_library_from_changed", MOD) == (FailureType.CRASH_IN_CHANGED, Outcome.FAIL)
    assert only("crash_in_site_packages_from_changed", ["src/pkg/show.py"]) == (
        FailureType.CRASH_IN_CHANGED,
        Outcome.FAIL,
    )


def test_a_crash_in_a_dependency_called_from_the_test_does_not_count() -> None:
    # The review's multi-line repr plant: an object from the changed module, with a repr that imitates a frame, passed
    # to a dependency that raises. Native tracebacks print no reprs, and the deepest repository frame is the test.
    assert only("crash_in_site_packages_from_test", MOD) == (FailureType.CRASH_ELSEWHERE, Outcome.ERROR)


def test_a_test_callback_raising_through_changed_code_does_not_count() -> None:
    assert only("callback_raises_through_changed", MOD) == (FailureType.CRASH_ELSEWHERE, Outcome.ERROR)


def test_a_stdlib_module_with_a_changed_files_name_never_counts() -> None:
    # The review's plant: changed 'utils.py', and the standard library's email/utils.py raises.
    assert only("stdlib_same_name", ["utils.py"]) == (FailureType.CRASH_ELSEWHERE, Outcome.ERROR)


@pytest.mark.parametrize("name", ["message_imitates_a_frame", "message_imitates_a_chain"])
def test_a_message_that_imitates_a_frame_or_a_chain_is_not_a_frame(name: str) -> None:
    assert only(name, MOD) == (FailureType.CRASH_ELSEWHERE, Outcome.ERROR)


def test_a_crash_the_test_raises_from_a_changed_crash_is_the_tests_own() -> None:
    assert only("chained_in_changed", MOD) == (FailureType.CRASH_ELSEWHERE, Outcome.ERROR)


# -- never counted -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "network_in_changed",
        "dns_in_changed",
        "read_only_in_changed",
        "disk_full_in_changed",
        "home_in_changed",
        "wrapped_network_in_changed",
    ],
)
def test_errors_the_sandbox_causes_in_changed_code_are_infra(name: str) -> None:
    # PLANT (review note): each of these was fail/crash_in_changed; the patch did nothing but meet the sandbox.
    assert only(name, ["src/pkg/net.py"]) == (FailureType.INFRA, Outcome.ERROR)


def test_name_and_import_errors_are_not_counted() -> None:
    assert only("name_error") == (FailureType.NAME, Outcome.ERROR)
    assert only("import_in_call") == (FailureType.IMPORT, Outcome.ERROR)


@pytest.mark.parametrize("name", ["fixture_setup", "fixture_setup_assert", "fixture_teardown"])
def test_fixture_errors_are_not_counted(name: str) -> None:
    assert only(name) == (FailureType.FIXTURE, Outcome.ERROR)


def test_a_unittest_setup_crash_in_changed_code_is_a_fixture_error() -> None:
    assert only("unittest_setup_in_changed", MOD) == (FailureType.FIXTURE, Outcome.ERROR)


@pytest.mark.parametrize("name", ["collect_import", "collect_syntax"])
def test_collection_errors_are_import_errors(name: str) -> None:
    assert only(name) == (FailureType.IMPORT, Outcome.ERROR)


@pytest.mark.parametrize("name", ["harness_timeout", "harness_timeout_v1"])
def test_pytest_timeouts_failure_is_a_timeout(name: str) -> None:
    assert only(name) == (FailureType.TIMEOUT, Outcome.TIMEOUT)


def test_an_unknown_error_element_is_infra() -> None:
    xml = sample("assertion").replace("<failure ", "<error ").replace("</failure>", "</error>")
    (case,) = junit_cases(xml, root=ROOT, changed_files=[])
    assert (case.failure_type, case.outcome) == (FailureType.INFRA, Outcome.ERROR)


def test_a_long_style_report_fails_closed() -> None:
    # A conftest can switch the traceback style at run time; the reader then finds no native traceback and counts
    # nothing, rather than guess.
    assert only("long_style", MOD) == (FailureType.CRASH_ELSEWHERE, Outcome.ERROR)


def test_strict_xfail_makes_an_unexpected_pass_a_non_pass() -> None:
    assert only("xpass_strict") == (FailureType.OTHER, Outcome.ERROR)


# -- skipped and xfailed never pass ------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["skip_importorskip", "skip_marker", "collect_skip", "xfail"])
def test_skipped_and_xfailed_cases_are_skipped_never_pass(name: str) -> None:
    assert only(name) == (None, Outcome.SKIPPED)


def _finding(base: list[RunResult], patch: list[RunResult]) -> Finding:
    repro = Repro(
        id="r1",
        attack_id="a1",
        claim_id="nothing_else_changed",
        lens=Lens.REGRESSION,
        kind=CatchKind.REGRESSION,
        files={"tests/test_repro.py": "def test_repro():\n    assert True\n"},
        command=("tests/test_repro.py::test_repro",),
    )
    replay = ReplayKey(
        base_sha="a" * 40, patch_sha="b" * 40, harness_hash="h", seed=0, image_digest="sha256:" + "c" * 64
    )
    return Finding(
        id="entry~django__django-1/nothing_else_changed/a1",
        target_id="entry~django__django-1",
        claim_id="nothing_else_changed",
        lens=Lens.REGRESSION,
        status=Status.BLOCKER,
        kind=CatchKind.REGRESSION,
        repro=repro,
        runs=(*base, *patch),
        quote="the issue sentence",
        replay=replay,
    )


def coarse(side: Side = Side.PATCH, code: int = 0, **kw: object) -> RunResult:
    fields: dict[str, object] = {
        "side": side,
        "outcome": Outcome.PASS if code == 0 else Outcome.ERROR,
        "exit_code": code,
        "seed": 3,
        "fresh_container": True,
        "duration_s": 2.5,
        "stdout_tail": "tail",
        "image_digest": "sha256:" + "c" * 64,
        **kw,
    }
    return RunResult.model_validate(fields)


def suite(name: str, side: Side, changed: Sequence[str] = ()) -> SuiteRun:
    return classify_suite(
        coarse(side, exit_code(name)),
        runner=PytestRunner(ROOT),
        output="",
        report=sample(name),
        changed_files=changed,
        timed_out=False,
    )


def _runs(name: str, side: Side, expect: Sequence[str] | None = None) -> list[RunResult]:
    return [with_expect(suite(name, side), expect) for _ in range(BLOCKER_RUNS)]


def test_a_repro_that_skipped_on_base_cannot_make_a_regression_blocker() -> None:
    patch = _runs("assertion", Side.PATCH)
    assert _finding(_runs("pass", Side.BASE), patch).status is Status.BLOCKER  # control: a real pass on base
    for name in ("skip_importorskip", "xfail"):
        base = _runs(name, Side.BASE)
        assert all(r.outcome is Outcome.SKIPPED for r in base)
        with pytest.raises(ValidationError, match="passes on base"):
            _finding(base, patch)


# -- reducing a run ----------------------------------------------------------------------------------------------------


def test_a_clean_report_passes_and_an_empty_one_is_infra() -> None:
    assert only("pass") == (None, Outcome.PASS)
    assert reduce_cases(cases_of("empty")) == (Outcome.ERROR, FailureType.INFRA)


def test_reduce_prefers_the_counted_failure_when_a_run_mixes_results() -> None:
    assert reduce_cases(cases_of("mixed")) == (Outcome.FAIL, FailureType.ASSERTION)


def test_expect_selects_the_repro_case_and_a_missing_one_is_infra() -> None:
    cases = cases_of("mixed")
    assert reduce_cases(cases, expect=["tests/test_mixed.py::test_pass"]) == (Outcome.PASS, None)
    assert reduce_cases(cases, expect=["tests/test_mixed.py::test_assertion"]) == (Outcome.FAIL, FailureType.ASSERTION)
    # PLANT: the repro case never ran. The other case passing must not make it a pass.
    assert reduce_cases(cases, expect=["tests/test_mixed.py::test_never_ran"]) == (Outcome.ERROR, FailureType.INFRA)
    both = ["tests/test_mixed.py::test_pass", "tests/test_mixed.py::test_never_ran"]
    assert reduce_cases(cases, expect=both) == (Outcome.ERROR, FailureType.INFRA)
    with pytest.raises(ValueError):
        reduce_cases(cases, expect=[])


def test_an_expected_node_id_answers_for_its_parametrizations() -> None:
    cases = cases_of("params")
    assert [c.name for c in cases] == [
        "tests/test_params.py::test_param[1]",
        "tests/test_params.py::test_param[2]",
        "tests/test_params.py::TestK::test_method",
    ]
    assert reduce_cases(cases, expect=["tests/test_params.py::test_param"]) == (Outcome.FAIL, FailureType.ASSERTION)
    assert reduce_cases(cases, expect=["tests/test_params.py::test_param[1]"]) == (Outcome.PASS, None)
    assert reduce_cases(cases, expect=["tests/test_params.py::TestK::test_method"]) == (Outcome.PASS, None)
    assert reduce_cases(cases, expect=["tests/test_params.py::test_para"]) == (Outcome.ERROR, FailureType.INFRA)


def test_a_sympy_case_answers_by_its_file_and_its_name() -> None:
    # sympy names a case by its function and its file (contract 65a0d21); a repro's command names it path::function,
    # SWE-bench by the function alone.
    a, b = "sympy/a/tests/test_a.py", "sympy/b/tests/test_b.py"
    cases = [
        CaseResult(name="test_x", outcome=Outcome.PASS, file=a),
        CaseResult(name="test_x", outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION, file=b),
        CaseResult(name="test_y", outcome=Outcome.PASS, file=a),
    ]
    # PLANT (keyed by name alone): test_b.py's failing test_x answered for test_a.py's passing one.
    assert reduce_cases(cases, expect=[f"{a}::test_x"]) == (Outcome.PASS, None)
    assert reduce_cases(cases, expect=[f"{b}::test_x"]) == (Outcome.FAIL, FailureType.ASSERTION)
    # A repeated bare name matches every case of that name, and all of them must pass.
    assert reduce_cases(cases, expect=["test_x"]) == (Outcome.FAIL, FailureType.ASSERTION)
    assert reduce_cases(cases, expect=["test_y"]) == (Outcome.PASS, None)
    assert reduce_cases(cases, expect=[f"{b}::test_y"]) == (Outcome.ERROR, FailureType.INFRA)  # not in that file
    # A pytest node id never answers for a bare name, nor a bare name for a node id.
    node = [CaseResult(name="t.py::test_x", outcome=Outcome.PASS)]
    assert reduce_cases(node, expect=["test_x"]) == (Outcome.ERROR, FailureType.INFRA)


def test_an_unexpected_pass_is_a_pass_whichever_way_the_runner_exits() -> None:
    xpass = CaseResult(name="t", outcome=Outcome.PASS, message=UNEXPECTED_PASS)
    ok = CaseResult(name="u", outcome=Outcome.PASS)
    bad = CaseResult(name="v", outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION)
    for kind in (RunnerKind.SYMPY, RunnerKind.DJANGO):
        # sympy and Django 3.0 exit 0 on an unexpected pass; later Django exits 1. Both agree.
        assert exit_agrees(kind, 0, [xpass, ok]) and exit_agrees(kind, 1, [xpass, ok])
        assert not exit_agrees(kind, 1, [ok]) and not exit_agrees(kind, 0, [xpass, bad])
    assert reduce_cases([xpass, ok]) == (Outcome.PASS, None)


@pytest.mark.parametrize(
    ("classname", "name", "file", "want"),
    [
        ("tests.test_x", "test_a", "tests/test_x.py", "tests/test_x.py::test_a"),
        ("tests.test_x.TestK.Inner", "test_b[1-2]", "tests/test_x.py", "tests/test_x.py::TestK::Inner::test_b[1-2]"),
        ("", "tests.test_import", "tests/test_import.py", "tests/test_import.py"),
        # The file is not the classname's module, and no other testcase says what is: the classes are the trailing
        # CapWords parts, the module the rest.
        ("other.module", "test_c", "tests/test_x.py", "other/module.py::test_c"),
        ("other.test_m.TestK", "test_d", None, "other/test_m.py::TestK::test_d"),
        # No module can be told: the name stays dotted, which no SWE-bench id is.
        ("Other.Module", "test_c", "tests/test_x.py", "Other.Module::test_c"),
        ("other..module", "test_c", "tests/test_x.py", "other..module::test_c"),
    ],
)
def test_node_ids_rebuild_from_xunit1(classname: str, name: str, file: str | None, want: str) -> None:
    assert nodeid(classname, name, file) == want


# -- node ids from the module a case was collected from (round 7, R1) -------------------------------------------------

# Reports R24 captured in SWE-bench images on 2026-10-28, run at the base commit with the run phase's own
# PYTEST_OPTIONS, trimmed to some of their testcases: each kept <testcase> is the report's own text, in its order, and
# the suite's counts are recounted for what was kept.
_SWEBENCH = Path(__file__).parent / "fixtures" / "swebench"


def swebench_cases(instance: str) -> tuple[CaseResult, ...]:
    return junit_cases((_SWEBENCH / f"{instance}.xml").read_text(), root="/testbed", changed_files=())


def test_plant_a_decorated_tests_node_id_is_its_modules_not_the_decorators_file() -> None:
    """matplotlib__matplotlib-25775 (R24, unchecked: "expected cases ... never ran"): check_figures_equal's wrapper sets
    no __wrapped__, so pytest 8.3.3's junit names lib/matplotlib/testing/decorators.py as the file of every test it
    wraps, while the classname is still the test module's. pytest's own -rA line for it reads
    PASSED lib/matplotlib/tests/test_text.py::test_annotation_units[png]; the report's other testcases establish the
    module, and the case's node id is that one."""
    cases = swebench_cases("matplotlib__matplotlib-25775")
    text = "lib/matplotlib/tests/test_text.py"
    assert [c.name for c in cases] == [
        f"{text}::test_font_styles[png]",
        f"{text}::test_font_styles[svg]",
        f"{text}::test_afm_kerning",
        f"{text}::test_annotation_contains",
        f"{text}::test_titles[png]",
        f"{text}::test_annotation_units[png]",  # file="lib/matplotlib/testing/decorators.py"
        f"{text}::test_buffer_size[png]",  # the same
    ]
    wanted = [f"{text}::test_annotation_units[png]", f"{text}::test_buffer_size[png]"]
    assert reduce_cases(cases, expect=wanted, repo="matplotlib/matplotlib") == (Outcome.PASS, None)


def test_plant_an_inherited_tests_node_id_is_its_collecting_modules() -> None:
    """astropy__astropy-13236 (R24, unchecked): TestMetaTable in astropy/table/tests/test_table.py inherits its 14 tests
    from astropy/utils/tests/test_metadata.py, which pytest 7.4.0's junit names as their file. pytest's own -rA lines
    read PASSED astropy/table/tests/test_table.py::TestMetaTable::test_none and so on."""
    cases = swebench_cases("astropy__astropy-13236")
    table = "astropy/table/tests/test_table.py"
    inherited = [
        "test_none",
        "test_mapping_init[meta0]",
        "test_mapping_init[meta1]",
        "test_mapping_init[meta2]",
        "test_non_mapping_init[ceci n'est pas un meta]",
        "test_non_mapping_init[1.2]",
        "test_non_mapping_init[meta2]",
        "test_mapping_set[meta0]",
        "test_mapping_set[meta1]",
        "test_mapping_set[meta2]",
        "test_non_mapping_set[ceci n'est pas un meta]",
        "test_non_mapping_set[1.2]",
        "test_non_mapping_set[meta2]",
        "test_meta_fits_header",
    ]
    assert [c.name for c in cases] == [
        f"{table}::TestSetTableColumn::test_set_row[unmasked]",
        *(f"{table}::TestMetaTable::{name}" for name in inherited),
        f"{table}::TestUpdate::test_metadata_conflict",
    ]
    assert reduce_cases(cases, expect=[f"{table}::TestMetaTable::test_none"]) == (Outcome.PASS, None)


def _report(*testcases: tuple[str, str, str]) -> str:
    body = "".join(
        f'<testcase classname={quoteattr(c)} name={quoteattr(n)} file={quoteattr(f)} line="1" time="0.001" />'
        for c, n, f in testcases
    )
    return f'<?xml version="1.0" encoding="utf-8"?><testsuites><testsuite name="pytest">{body}</testsuite></testsuites>'


def _names(*testcases: tuple[str, str, str]) -> list[str]:
    return [c.name for c in junit_cases(_report(*testcases), root=ROOT, changed_files=())]


def test_an_inherited_class_whose_module_has_no_testcase_of_its_own() -> None:
    """pkg/tests/test_b.py holds only TestChild, whose one test it inherits from pkg/tests/test_a.py: no testcase of the
    report establishes pkg.tests.test_b, so the CapWords parts are the classes and the module is the rest."""
    assert _names(
        ("pkg.tests.test_a", "test_plain", "pkg/tests/test_a.py"),
        ("pkg.tests.test_a.TestBase", "test_shared", "pkg/tests/test_a.py"),
        ("pkg.tests.test_b.TestChild", "test_shared", "pkg/tests/test_a.py"),
    ) == [
        "pkg/tests/test_a.py::test_plain",
        "pkg/tests/test_a.py::TestBase::test_shared",
        "pkg/tests/test_b.py::TestChild::test_shared",
    ]


def test_parametrized_and_class_nested_ids_rebuild_from_their_module() -> None:
    """Parameters may hold dots and '::'; the classname never does (pytest cuts the parameters off before it writes
    the classname), so they stay whole, and every class after the module becomes its own part of the id."""
    assert _names(
        ("pkg.tests.test_c.TestOuter", "test_own", "pkg/tests/test_c.py"),
        ("pkg.tests.test_c.TestOuter.TestInner", "test_y[1-a.b::c]", "pkg/mixins.py"),
        ("pkg.tests.test_c", "test_z[png]", "pkg/testing/decorators.py"),
    ) == [
        "pkg/tests/test_c.py::TestOuter::test_own",
        "pkg/tests/test_c.py::TestOuter::TestInner::test_y[1-a.b::c]",
        "pkg/tests/test_c.py::test_z[png]",
    ]


def test_a_module_is_a_classnames_start_only_up_to_a_dot() -> None:
    """pkg.tests.test_text is a start of pkg.tests.test_text_layout in characters, not in parts: each decorated or
    inherited case takes the module of its own classname, and a module the report establishes no case for
    (pkg.tests.test_tex, a start of both in characters) falls back to the classname alone."""
    assert _names(
        ("pkg.tests.test_text", "test_own", "pkg/tests/test_text.py"),
        ("pkg.tests.test_text_layout", "test_own", "pkg/tests/test_text_layout.py"),
        ("pkg.tests.test_text", "test_wrapped[png]", "pkg/testing/decorators.py"),
        ("pkg.tests.test_text_layout.TestL", "test_inherited", "pkg/base.py"),
        ("pkg.tests.test_tex", "test_w[png]", "pkg/testing/decorators.py"),
    ) == [
        "pkg/tests/test_text.py::test_own",
        "pkg/tests/test_text_layout.py::test_own",
        "pkg/tests/test_text.py::test_wrapped[png]",
        "pkg/tests/test_text_layout.py::TestL::test_inherited",
        "pkg/tests/test_tex.py::test_w[png]",
    ]


def test_plant_a_module_two_files_establish_names_nothing_so_its_case_never_ran() -> None:
    """a/b.c.py and a/b/c.py both write their classname as a.b.c: a case collected from one of them whose file is
    elsewhere cannot be told apart, so its name stays dotted, and an expected case it might be never ran (INFRA),
    never a false pass."""
    cases = junit_cases(
        _report(
            ("a.b.c", "test_1", "a/b.c.py"), ("a.b.c", "test_2", "a/b/c.py"), ("a.b.c.TestX", "test_z", "lib/x.py")
        ),
        root=ROOT,
        changed_files=(),
    )
    assert [c.name for c in cases] == ["a/b.c.py::test_1", "a/b/c.py::test_2", "a.b.c.TestX::test_z"]
    for wanted in ("a/b/c.py::TestX::test_z", "a/b.c.py::TestX::test_z"):
        assert reduce_cases(cases, expect=[wanted]) == (Outcome.ERROR, FailureType.INFRA)


# -- ids swebench 4.1.0's log parser renames (round 7, R2) ------------------------------------------------------------

PYLINT = "pylint-dev/pylint"
# pylint-dev__pylint-7080's and 7277's PASS_TO_PASS id, as swebench 4.1.0's parse_log_pytest_options records the case
# its run reports as test_stdin[/testbed/tests/mymodule.py-mymodule-/testbed/tests/mymodule.py].
STDIN = "tests/test_self.py::TestRunTC::test_stdin[/mymodule.py]"


def test_plant_pylints_renamed_id_is_answered_by_its_real_case() -> None:
    """R24 left pylint-dev__pylint-7080 and 7277 unchecked: their PASS_TO_PASS id is the name swebench 4.1.0's pylint
    parser records, not the node id the run reports. The case keeps its node id, so a rerun by name still selects it,
    and answers the harness's id in pylint's repository only."""
    cases = swebench_cases("pylint-dev__pylint-7080")
    assert cases[0].name == (
        "tests/test_self.py::TestRunTC::test_stdin[/testbed/tests/mymodule.py-mymodule-/testbed/tests/mymodule.py]"
    )
    assert reduce_cases(cases, expect=[STDIN], repo=PYLINT) == (Outcome.PASS, None)
    for repo in ("astropy/astropy", ""):  # a parser that records ids as they are, and no repository at all
        assert reduce_cases(cases, expect=[STDIN], repo=repo) == (Outcome.ERROR, FailureType.INFRA)


def test_plant_every_case_the_harness_records_under_one_id_must_pass() -> None:
    """Two of a run's cases can rename to one id. The harness keeps whichever its log printed last; Receipts takes
    them all, as it takes sympy's repeated names, and the id passes only when every one of them passed."""
    ok = CaseResult(name="tests/test_self.py::TestRunTC::test_stdin[/a/mymodule.py]", outcome=Outcome.PASS)
    also = CaseResult(name="tests/test_self.py::TestRunTC::test_stdin[/b/mymodule.py]", outcome=Outcome.PASS)
    bad = CaseResult(
        name="tests/test_self.py::TestRunTC::test_stdin[/c/mymodule.py]",
        outcome=Outcome.FAIL,
        failure_type=FailureType.ASSERTION,
    )
    skip = CaseResult(name="tests/test_self.py::TestRunTC::test_stdin[/d/mymodule.py]", outcome=Outcome.SKIPPED)
    assert reduce_cases([ok, also], expect=[STDIN], repo=PYLINT) == (Outcome.PASS, None)
    assert reduce_cases([ok, bad], expect=[STDIN], repo=PYLINT) == (Outcome.FAIL, FailureType.ASSERTION)
    assert reduce_cases([bad, ok], expect=[STDIN], repo=PYLINT) == (Outcome.FAIL, FailureType.ASSERTION)
    assert reduce_cases([ok, skip], expect=[STDIN], repo=PYLINT) == (Outcome.SKIPPED, None)
    # One of them is also another expected id: both ids are answered, and still every case must pass.
    exact = "tests/test_self.py::TestRunTC::test_stdin[/c/mymodule.py]"
    assert reduce_cases([ok, bad], expect=[STDIN, exact], repo=PYLINT) == (Outcome.FAIL, FailureType.ASSERTION)


# -- the exit code is checked on every run -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    "name",
    [
        "pass",
        "assertion",
        "mixed",
        "crash",
        "collect_import",
        "collect_skip",
        "skip_marker",
        "xfail",
        "params",
        "empty",
    ],
)
def test_every_recorded_exit_code_agrees_with_its_report(name: str) -> None:
    assert exit_agrees(RunnerKind.PYTEST, exit_code(name), cases_of(name, MOD))


def test_a_report_that_disagrees_with_the_exit_code_is_infra_with_or_without_expect() -> None:
    def run(name: str, code: int, expect: Sequence[str] | None) -> RunResult:
        result = classify_suite(
            coarse(Side.PATCH, code), runner=PytestRunner(ROOT), output="", report=sample(name), changed_files=[],
            timed_out=False,
        )  # fmt: skip
        return with_expect(result, expect)

    passing = ["tests/test_pass.py::test_pass"]
    failing = ["tests/test_assn.py::test_assertion"]
    assert run("pass", 0, passing).outcome is Outcome.PASS
    assert run("assertion", 1, failing).outcome is Outcome.FAIL
    # PLANT (review blocker): with expect, the exit code was ignored, so a passing report from a run that exited 137,
    # 125 or 2, and a failing report from a run that exited 0, were trusted.
    for code in (137, 125, 2, 3):
        result = run("pass", code, passing)
        assert (result.outcome, result.failure_type) == (Outcome.ERROR, FailureType.INFRA), code
    assert (run("assertion", 0, failing).outcome, run("assertion", 0, None).outcome) == (Outcome.ERROR, Outcome.ERROR)


def test_a_run_that_wrote_no_report_is_infra() -> None:
    # PLANT (review blocker): every run wrote the same path, so a run that died before writing read the previous
    # run's report. Each run now has its own path, and no report at that path is infra, whatever the exit code.
    for code in (0, 1, 137):
        run = classify_suite(
            coarse(Side.BASE, code), runner=PytestRunner(ROOT), output="", report=None, changed_files=[],
            timed_out=False,
        )  # fmt: skip
        assert (run.result.outcome, run.result.failure_type, run.cases) == (Outcome.ERROR, FailureType.INFRA, ())
        assert with_expect(run, ["tests/test_pass.py::test_pass"]).outcome is Outcome.ERROR


class _ReportlessPytest:
    """A pytest-kind runner that reads cases from somewhere other than its report, here a case it makes up."""

    kind = RunnerKind.PYTEST

    def command(self, target: object, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["true"]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return (CaseResult(name="t.py::test_a", outcome=Outcome.PASS),)


def test_a_pytest_run_with_no_report_is_infra_whatever_its_runner_says() -> None:
    run = classify_suite(
        coarse(Side.BASE, 0), runner=_ReportlessPytest(), output="", report=None, changed_files=[], timed_out=False
    )
    assert (run.result.outcome, run.result.failure_type) == (Outcome.ERROR, FailureType.INFRA)
    # The reason, and the run's last words (its stdout tail here), travel on the result itself (contract v3).
    assert run.problem == run.result.problem == "the run wrote no report; the run's last line: tail"


def test_a_sandbox_timeout_stays_a_timeout() -> None:
    run = classify_suite(
        coarse(Side.PATCH, 124, outcome=Outcome.TIMEOUT, failure_type=FailureType.TIMEOUT),
        runner=PytestRunner(ROOT), output="", report=sample("assertion"), changed_files=[], timed_out=True,
    )  # fmt: skip
    assert (run.result.outcome, run.result.failure_type) == (Outcome.TIMEOUT, FailureType.TIMEOUT)
    assert with_expect(run, ["tests/test_assn.py::test_assertion"]).outcome is Outcome.TIMEOUT


def test_unreadable_output_and_reports_are_infra() -> None:
    bad = classify_suite(
        coarse(Side.PATCH, 1), runner=PytestRunner(ROOT), output="", report="<testsuites><oops", changed_files=[],
        timed_out=False,
    )  # fmt: skip
    assert bad.result.failure_type is FailureType.INFRA and bad.problem and "could not be read" in bad.problem
    big = classify_suite(
        coarse(Side.PATCH, 1), runner=PytestRunner(ROOT), output=None, report=sample("assertion"), changed_files=[],
        timed_out=False,
    )  # fmt: skip
    assert big.result.failure_type is FailureType.INFRA and big.problem and "too large" in big.problem


def test_classify_suite_keeps_everything_else_on_the_result() -> None:
    run = suite("crash", Side.PATCH, MOD)
    assert (run.result.outcome, run.result.failure_type) == (Outcome.FAIL, FailureType.CRASH_IN_CHANGED)
    assert (run.result.seed, run.result.stdout_tail, run.result.duration_s, run.result.exit_code) == (3, "tail", 2.5, 1)


def test_exit_agreement_by_runner_kind() -> None:
    ok = [CaseResult(name="a", outcome=Outcome.PASS)]
    bad = [CaseResult(name="a", outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION)]
    for kind in RunnerKind:
        assert exit_agrees(kind, 0, ok) and exit_agrees(kind, 1, bad)
        assert not exit_agrees(kind, 1, ok) and not exit_agrees(kind, 0, bad)
        assert not exit_agrees(kind, 137, ok)
    assert not exit_agrees(RunnerKind.DJANGO, 2, ok) and not exit_agrees(RunnerKind.SYMPY, 5, [])


# -- hostile reports ---------------------------------------------------------------------------------------------------


def test_a_doctype_is_refused() -> None:
    hostile = '<?xml version="1.0"?><!DOCTYPE t [<!ENTITY a "x">]><testsuites></testsuites>'
    with pytest.raises(JunitParseError, match="DOCTYPE"):
        junit_cases(hostile, root=ROOT, changed_files=[])


def test_malformed_xml_is_a_parse_error() -> None:
    with pytest.raises(JunitParseError):
        junit_cases("<testsuites><oops", root=ROOT, changed_files=[])


def test_an_oversized_report_is_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    import receipts.attack.classify as classify

    monkeypatch.setattr(classify, "MAX_REPORT_BYTES", 100)
    with pytest.raises(JunitParseError, match="larger than"):
        junit_cases(sample("pass"), root=ROOT, changed_files=[])


# -- contract v3: a run says why it could not be read, and an expected case that never ran says why ------------------


def test_every_run_that_could_not_be_read_says_why_on_its_result() -> None:
    # PLANT (contract v3, RunResult.problem): the reason lived only on SuiteRun, so a RunResult that crossed to an
    # attacker or the skeptic carried none, and an unchecked detail had to rebuild it from the output's tails.
    unread = [
        ({"output": "", "report": None}, "the run wrote no report"),
        ({"output": "", "report": "<testsuites><oops"}, "the run's report could not be read: could not parse junit"),
        ({"output": None, "report": sample("assertion")}, "the run's output was too large to read"),
    ]
    for kw, why in unread:
        run = classify_suite(coarse(Side.PATCH, 1), runner=PytestRunner(ROOT), changed_files=[], timed_out=False, **kw)
        assert run.problem is not None and run.result.problem == run.problem and run.problem.startswith(why), why
        assert with_expect(run, ["tests/test_assn.py::test_assertion"]).problem == run.problem
    crashed = coarse(Side.PATCH, 0, stderr_tail="Traceback (most recent call last):\nRuntimeError: the runner fell")
    disagree = classify_suite(
        crashed, runner=PytestRunner(ROOT), output="", report=sample("assertion"), changed_files=[], timed_out=False
    )
    assert disagree.result.problem == (
        "exit code 0 disagrees with the 1 cases the run reported; the run's last line: RuntimeError: the runner fell"
    )
    timed = classify_suite(
        coarse(Side.PATCH, 124, outcome=Outcome.TIMEOUT, failure_type=FailureType.TIMEOUT), runner=PytestRunner(ROOT),
        output="", report=None, changed_files=[], timed_out=True,
    )  # fmt: skip
    assert timed.result.problem == "the run was killed for time after 2 s"
    nothing = classify_suite(
        coarse(Side.PATCH, exit_code("empty")), runner=PytestRunner(ROOT), output="", report=sample("empty"),
        changed_files=[], timed_out=False,
    )  # fmt: skip
    assert (nothing.result.outcome, nothing.result.problem, nothing.problem) == (
        Outcome.ERROR, "the run reported no case", None,
    )  # fmt: skip
    # A run that was read and answered has no problem, with or without expect.
    read = suite("assertion", Side.PATCH)
    assert read.result.problem == "" and with_expect(read, ["tests/test_assn.py::test_assertion"]).problem == ""


def test_an_expected_case_whose_file_did_not_collect_is_an_import_error() -> None:
    # PLANT (contract v3, integrator call): an expected case that never ran was always ERROR/INFRA, so a repro whose
    # own test file failed to import read as an infra gap (retried, listed as unchecked), never as the import error
    # it is, which the rules reject and R24 counts as the gold's own failure.
    run = suite("collect_import", Side.PATCH)  # pytest's collection error for tests/test_import.py
    (module,) = run.cases
    assert module.message == "collection failure: ModuleNotFoundError: No module named 'nonexistent_module_xyz'"
    got = with_expect(run, ["tests/test_import.py::test_x"])
    assert (got.outcome, got.failure_type) == (Outcome.ERROR, FailureType.IMPORT)
    assert got.problem == (
        "expected case tests/test_import.py::test_x never ran: its test module tests/test_import.py failed to import "
        "(collection failure: ModuleNotFoundError: No module named 'nonexistent_module_xyz')"
    )
    other = with_expect(run, ["tests/test_other.py::test_y"])  # another file's case: nothing explains it
    assert (other.outcome, other.failure_type) == (Outcome.ERROR, FailureType.INFRA)
    assert other.problem == (
        "expected case tests/test_other.py::test_y never ran, and the run reports no import error for its test module"
    )
    both = with_expect(run, ["tests/test_import.py::test_x", "tests/test_other.py::test_y"])
    assert both.failure_type is FailureType.INFRA  # one gap nothing explains makes the run infra
    assert "tests/test_import.py failed to import" in both.problem and "no import error" in both.problem
    # A package's __init__.py that fails at collection stands for every file under it, and only those.
    package = CaseResult(
        name="tests/pkg/__init__.py",
        outcome=Outcome.ERROR,
        failure_type=FailureType.IMPORT,
        message="collection failure",
    )
    assert reduce_cases([package], expect=["tests/pkg/test_a.py::test_x"]) == (Outcome.ERROR, FailureType.IMPORT)
    assert reduce_cases([package], expect=["tests/pkgx/test_a.py::test_x"]) == (Outcome.ERROR, FailureType.INFRA)
    # An import error a test raised while it ran is that test's own case, never a module that did not import.
    inner = CaseResult(name="t.py::test_a", outcome=Outcome.ERROR, failure_type=FailureType.IMPORT)
    assert reduce_cases([inner], expect=["t.py::test_b"]) == (Outcome.ERROR, FailureType.INFRA)


def test_a_sympy_case_whose_module_did_not_import_is_an_import_error() -> None:
    a, b = "sympy/a/tests/test_a.py", "sympy/b/tests/test_b.py"
    module = CaseResult(
        name=a, outcome=Outcome.ERROR, failure_type=FailureType.IMPORT, message="ModuleNotFoundError: q", file=a
    )
    cases = [module, CaseResult(name="test_y", outcome=Outcome.PASS, file=b)]
    assert reduce_cases(cases, expect=[f"{a}::test_x"]) == (Outcome.ERROR, FailureType.IMPORT)
    assert reduce_cases(cases, expect=[f"{b}::test_x"]) == (Outcome.ERROR, FailureType.INFRA)
    # A bare name (SWE-bench's form) is the failed module's only when the run's files say that module defines it.
    assert reduce_cases(cases, expect=["test_x"]) == (Outcome.ERROR, FailureType.INFRA)
    assert reduce_cases(cases, expect=["test_x"], homes={"test_x": (a,)}) == (Outcome.ERROR, FailureType.IMPORT)
    assert reduce_cases(cases, expect=["test_x"], homes={"test_x": (b,)}) == (Outcome.ERROR, FailureType.INFRA)
    *_, problem = reduce_run(cases, expect=["test_x"], homes={"test_x": (a,)})
    assert problem == f"expected case test_x never ran: its test module {a} failed to import (ModuleNotFoundError: q)"
    # A test-level import error in that file is a case of its own, never the module.
    inner = CaseResult(name="test_z", outcome=Outcome.ERROR, failure_type=FailureType.IMPORT, file=a)
    assert reduce_cases([inner], expect=[f"{a}::test_x"]) == (Outcome.ERROR, FailureType.INFRA)


def test_import_failure_of_names_the_module_case_or_none() -> None:
    django = CaseResult(name="app.tests", outcome=Outcome.ERROR, failure_type=FailureType.IMPORT, message="m")
    assert import_failure_of("app.tests.T.test_x", [django]) is django
    assert import_failure_of("app.testsuite.T.test_x", [django]) is None  # a label, not a prefix of characters
    assert import_failure_of("app.tests", [django]) is None  # the module's own name answers by name, not as missing
    skipped = CaseResult(name="app.tests", outcome=Outcome.SKIPPED)
    assert import_failure_of("app.tests.T.test_x", [skipped]) is None  # only an ERROR typed IMPORT says it


# -- round 5: an xfail says so in both of pytest's junit dialects (contract v3.1), and is still never a pass --------

# Every kind of expected failure in fixtures/junit/xfail_kinds.py, a skip whose reason reads like pytest 3's xfail
# message, and a pass, as pytest's -rA summary would show them: XFAIL, XFAIL, XFAIL, SKIPPED, PASSED, XFAIL.
XFAIL_KINDS = {
    "test_marked": (Outcome.SKIPPED, True),
    "test_imperative": (Outcome.SKIPPED, True),
    "test_not_run": (Outcome.SKIPPED, True),  # an xfail with run=False: -rA prints it XFAIL too
    "test_skip_whose_reason_reads_like_an_xfail": (Outcome.SKIPPED, False),
    "test_passes": (Outcome.PASS, False),
    "test_unittest_expected_failure": (Outcome.SKIPPED, True),
}
_PYTEST3 = Path(__file__).parent / "fixtures" / "runners" / "pytest3_xfail.json"


def flags(cases: Sequence[CaseResult]) -> dict[str, tuple[Outcome, bool]]:
    return {c.name.rsplit("::", 1)[-1]: (c.outcome, c.expected_failure) for c in cases}


def test_a_pytest_xfail_is_flagged_and_stays_skipped() -> None:
    # pytest 9.1.1 (make_samples.py): <skipped type="pytest.xfail" message="<reason>"/>, the form since pytest 4.5.0.
    # PLANT: junit_cases read every <skipped> as a plain skip, so R24 could not tell an xfail from one.
    cases = cases_of("xfail_kinds")
    assert flags(cases) == XFAIL_KINDS
    by_name = {c.name.rsplit("::", 1)[-1]: c for c in cases}
    assert by_name["test_marked"].message == "known bug" and by_name["test_not_run"].message == "[NOTRUN] never run"
    assert by_name["test_unittest_expected_failure"].message == "expected failure"  # no reason: what it was
    assert cases_of("xfail")[0].expected_failure
    for name in ("skip_importorskip", "skip_marker", "collect_skip"):
        assert not any(c.expected_failure for c in cases_of(name)), name
    assert exit_agrees(RunnerKind.PYTEST, exit_code("xfail_kinds"), cases)
    # Never a pass in this lane: a run of an xfail is SKIPPED, with or without expect (only R24's gold check counts it).
    assert reduce_run(cases, expect=["tests/test_xfail_kinds.py::test_marked"]) == (Outcome.SKIPPED, None, "")
    assert with_expect(suite("xfail_kinds", Side.GOLD), ["tests/test_xfail_kinds.py::test_imperative"]).outcome is (
        Outcome.SKIPPED
    )


def test_a_pytest_3_xfail_is_flagged_in_its_older_junit_dialect() -> None:
    # Recorded with pytest 3.3.1, the version SWE-bench's astropy 0.1 to 1.3 environments pin, in the
    # django__django-11999 image's Python 3.6 under the run phase's isolation (fixtures/runners/record.py pytest5):
    # before 4.5.0 pytest writes <skipped message="expected test failure"><reason></skipped>, with no type.
    # PLANT: only type="pytest.xfail" was read as an xfail, so all four read as plain skips; and a rule that read the
    # message alone would flag the skip whose reason is "expected test failure", which pytest types pytest.skip.
    rec = json.loads(_PYTEST3.read_text())
    assert rec["pytest_version"] == "3.3.1"
    run = rec["without_rootdir"]
    assert 'skipped message="expected test failure">known bug</skipped>' in run["junit"]
    cases = junit_cases(run["junit"], root="/testbed", changed_files=[])
    assert flags(cases) == XFAIL_KINDS
    assert {c.name for c in cases} == {
        "tests/receipts_xfail/test_xfail_kinds.py::" + n for n in XFAIL_KINDS if n != "test_unittest_expected_failure"
    } | {"tests/receipts_xfail/test_xfail_kinds.py::T::test_unittest_expected_failure"}
    assert exit_agrees(RunnerKind.PYTEST, int(run["exit_code"]), cases)
    assert reduce_cases(cases) == (Outcome.SKIPPED, None)


@pytest.mark.parametrize(
    ("attributes", "xfail"),
    [
        ('type="pytest.xfail" message="r"', True),
        ('type="pytest.xfail" message=""', True),
        ('message="expected test failure"', True),  # pytest before 4.5.0
        ('type="pytest.skip" message="expected test failure"', False),  # a skip's reason is the test's to choose
        ('type="pytest.skip" message="r"', False),
        ('message="collection skipped"', False),  # a module that skipped at collection, before 4.5.0
        ('message="xfail-marked test passes unexpectedly"', False),  # never an expected failure
        ("", False),
    ],
)
def test_only_pytests_own_record_of_an_xfail_is_one(attributes: str, xfail: bool) -> None:
    xml = (
        '<testsuite><testcase classname="tests.test_a" name="test_x" file="tests/test_a.py">'
        f"<skipped {attributes}>reason</skipped></testcase></testsuite>"
    )
    (case,) = junit_cases(xml, root=ROOT, changed_files=[])
    assert (case.outcome, case.expected_failure) == (Outcome.SKIPPED, xfail)
