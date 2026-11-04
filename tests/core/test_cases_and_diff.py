"""What a patch changes, and the cases a repro expects its runner to report."""

from __future__ import annotations

import ast
import re

from receipts.contracts import CatchKind, Lens, Repro, RunnerKind
from receipts.core.cases import (
    CaseRef,
    defined_cases,
    django_label,
    expected_cases,
    locate,
    place_cases,
    repro_test_files,
)
from receipts.core.diff import changed_files, changed_names, is_test_path, parse_patch, patch_facts, unquote

PATCH = """\
diff --git a/src/widgets/render.py b/src/widgets/render.py
index 1111111..2222222 100644
--- a/src/widgets/render.py
+++ b/src/widgets/render.py
@@ -10,7 +10,8 @@ class Renderer:
     def helper(self):
         return 1

     def render(self, items):
-        if not items:
-- a removed line that starts with two dashes
+        if items is None:
+            return ""
         return "".join(items)
diff --git a/src/widgets/new.py b/src/widgets/new.py
new file mode 100644
--- /dev/null
+++ b/src/widgets/new.py
@@ -0,0 +1,2 @@
+def fresh():
+    return 2
diff --git a/tests/test_render.py b/tests/test_render.py
deleted file mode 100644
--- a/tests/test_render.py
+++ /dev/null
@@ -1,2 +0,0 @@
-def test_old():
-    assert True
"""


def test_a_patch_is_read_by_its_hunk_counts() -> None:
    render, new, removed = parse_patch(PATCH)
    assert (render.path, render.added, render.removed) == ("src/widgets/render.py", 2, 2)
    # 'Renderer' from the hunk header, 'render' as the def above the change; the line starting '--' is a removal.
    assert render.names == {"Renderer", "render"}
    assert (new.old_path, new.new_path, new.names) == (None, "src/widgets/new.py", {"fresh"})
    assert (removed.old_path, removed.new_path, removed.path) == ("tests/test_render.py", None, "tests/test_render.py")
    assert changed_files(PATCH) == ("src/widgets/render.py", "src/widgets/new.py", "tests/test_render.py")
    assert changed_names(PATCH) == {"Renderer", "render", "fresh"}  # names in test files are left out


def test_a_plain_diff_with_no_git_lines_is_read_too() -> None:
    plain = "--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n--- a/y.py\n+++ b/y.py\n@@ -1 +1 @@\n-c\n+d\n"
    assert changed_files(plain) == ("x.py", "y.py")
    assert [(c.added, c.removed) for c in parse_patch(plain)] == [(1, 1), (1, 1)]


def test_git_quoted_paths_are_read_back_as_utf8_bytes() -> None:
    # git quotes a path with a byte past ASCII, writing each byte in octal: caf\303\251.py is "caf" + e-acute + ".py".
    quoted = (
        'diff --git "a/src/caf\\303\\251.py" "b/src/caf\\303\\251.py"\n'
        '--- "a/src/caf\\303\\251.py"\n'
        '+++ "b/src/caf\\303\\251.py"\n'
        "@@ -1 +1 @@ def brew():\n-a\n+b\n"
    )
    assert changed_files(quoted) == ("src/caf\u00e9.py",)
    assert patch_facts(quoted).names_by_module == {"caf\u00e9": frozenset({"brew"})}
    # A pure rename has no ---/+++ lines, so its paths come from the 'diff --git' line alone; here one side is quoted.
    renamed = 'diff --git a/plain.py "b/r\\303\\251sum\\303\\251.py"\nsimilarity index 100%\n'
    (change,) = parse_patch(renamed)
    assert (change.old_path, change.new_path) == ("plain.py", "r\u00e9sum\u00e9.py")
    # The C escapes git writes besides octal, and bytes that are not UTF-8, which stay as surrogate escapes.
    assert unquote('tab\\there \\"quoted\\" back\\\\slash') == 'tab\there "quoted" back\\slash'
    assert unquote("latin\\351.py") == "latin\udce9.py"


def test_test_paths_follow_the_usual_conventions() -> None:
    for path in ("tests/x.py", "pkg/tests/helpers.py", "test_x.py", "pkg/x_test.py", "tests.py", "conftest.py"):
        assert is_test_path(path), path
    for path in ("src/widgets/render.py", "testing_tools.py", "pkg/contest.py"):
        assert not is_test_path(path), path


SOURCE = """\
import unittest

from django.test import SimpleTestCase


def test_top():
    pass


async def test_async_top():
    pass


def helper():
    pass


class TestGroup:
    def test_method(self):
        pass

    def not_a_test(self):
        pass


class TestWithInit:
    def __init__(self):
        pass

    def test_never_collected(self):
        pass


class Legacy(unittest.TestCase):
    def test_legacy(self):
        pass


class Views(SimpleTestCase):
    def test_view(self):
        pass
"""


def repro(files: dict[str, str]) -> Repro:
    return Repro(
        id="r1",
        attack_id="correctness-1",
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        kind=CatchKind.REGRESSION,
        files=files,
        command=("pytest",),
    )


def test_expected_cases_are_named_the_way_each_runner_reports_them() -> None:
    candidate = repro({"tests/views/test_views.py": SOURCE, "tests/views/conftest.py": "", "data.json": "{}"})
    assert repro_test_files(candidate) == ("tests/views/test_views.py",)
    path = "tests/views/test_views.py"
    assert expected_cases(candidate, RunnerKind.PYTEST) == (
        f"{path}::test_top",
        f"{path}::test_async_top",
        f"{path}::TestGroup::test_method",
        f"{path}::Legacy::test_legacy",
        f"{path}::Views::test_view",
    )
    assert expected_cases(candidate, RunnerKind.DJANGO) == (
        "views.test_views.Legacy.test_legacy",
        "views.test_views.Views.test_view",
    )
    assert expected_cases(candidate, RunnerKind.SYMPY) == ("test_top", "test_async_top")  # bare, as bin/test says


def test_a_repro_with_no_test_case_or_a_broken_file_expects_none() -> None:
    assert expected_cases(repro({"tests/test_x.py": "def helper():\n    pass\n"}), RunnerKind.PYTEST) == ()
    assert expected_cases(repro({"tests/test_x.py": "def test_x(:\n"}), RunnerKind.PYTEST) == ()
    assert expected_cases(repro({"repro.py": "def test_x():\n    pass\n"}), RunnerKind.PYTEST) == ()


# ---------------------------------------------------------------- sympy's own case names (round 3, blocker 2)

# Recorded by the attack lane in the SWE-bench sympy__sympy-13974 image (its tests/attack/fixtures/runners/
# sympy_probe.json): the file `bin/test -C --verbose` ran, and the progress lines of that run's stdout. sympy names
# each case by its bare function name, and the attack lane's sandbox matches `expect` against those names exactly.
SYMPY_PROBE_PATH = "sympy/utilities/tests/test_receipts_probe.py"
SYMPY_PROBE = """\
import socket
import sys

from sympy import Symbol
from sympy.utilities import misc
from sympy.utilities.pytest import XFAIL, raises, skip


def test_pass():
    assert Symbol("x") == Symbol("x")


def test_assertion():
    assert Symbol("x") == Symbol("y")


def test_crash_in_changed():
    misc.receipts_probe_crash("not a number")


def test_crash_in_library_from_changed():
    misc.receipts_probe_library("{bad json")


def test_crash_in_test():
    raise KeyError("raised in the test itself")


def test_skipped():
    skip("not on this platform")


@XFAIL
def test_expected_failure():
    assert 1 == 2


@XFAIL
def test_unexpected_success():
    assert 1 == 1


def test_prints():
    print("printed to stdout")
    sys.stderr.write("written to stderr\\n")
    assert True


def test_import_error():
    import receipts_no_such_module  # noqa: F401


def test_name_error():
    return receipts_undefined_name  # noqa: F821


def test_chained():
    try:
        misc.receipts_probe_crash("x")
    except ValueError as exc:
        raise RuntimeError("wrapped") from exc


def test_network():
    socket.create_connection(("192.0.2.1", 80), timeout=2)


def test_read_only_root():
    open("/usr/receipts_probe", "w")


def test_message_imitates_a_frame():
    raise ValueError('boom\\n  File "/testbed/sympy/utilities/misc.py", line 1, in fake\\n    fake()')


def test_raises_did_not_raise():
    raises(ValueError, lambda: int("1"))
"""
SYMPY_RUN_PROGRESS = """\
sympy/utilities/tests/test_receipts_probe.py[16]
test_pass ok
test_assertion F
test_crash_in_changed E
test_crash_in_library_from_changed E
test_crash_in_test E
test_skipped not on this platform s
test_expected_failure f
test_unexpected_success X
test_prints printed to stdout
ok
test_import_error E
test_name_error E
test_chained E
test_network E
test_read_only_root E
test_message_imitates_a_frame E
test_raises_did_not_raise F                                               [FAIL]
"""


def sympy_reported_names(progress: str) -> tuple[str, ...]:
    """The case names a sympy run reports: the first word of each progress line that names a test."""
    return tuple(re.findall(r"^(test_\w+)", progress, re.MULTILINE))


def test_sympy_cases_are_the_bare_names_a_real_sympy_run_reports() -> None:
    reported = sympy_reported_names(SYMPY_RUN_PROGRESS)
    assert len(reported) == 16  # the run's own header: [16]
    candidate = repro({SYMPY_PROBE_PATH: SYMPY_PROBE})
    assert expected_cases(candidate, RunnerKind.SYMPY) == reported
    assert not any("::" in name or "/" in name for name in expected_cases(candidate, RunnerKind.SYMPY))


def test_sympy_cases_are_test_underscore_functions_each_named_once() -> None:
    first = "def test_same():\n    pass\n\ndef testnounderscore():\n    pass\n"
    second = "def test_same():\n    pass\n\ndef test_other():\n    pass\n"
    candidate = repro({"sympy/core/tests/test_a.py": first, "sympy/core/tests/test_b.py": second})
    # bin/test collects only test_* functions; two files defining one name report it under that one name.
    assert expected_cases(candidate, RunnerKind.SYMPY) == ("test_same", "test_other")
    assert "sympy/core/tests/test_a.py::testnounderscore" in expected_cases(candidate, RunnerKind.PYTEST)


# ---------------------------------------------------------------- Repro.cases read against the repro's files (round 4)

PLACED = """\
import pytest
from django.test import SimpleTestCase


def test_top():
    pass


@pytest.mark.parametrize("n", [1, 2])
def test_param(n):
    pass


class TestGroup:
    def test_method(self):
        pass

    class TestInner:
        def test_deep(self):
            pass


class Base(SimpleTestCase):
    def test_inherited(self):
        pass


class Views(Base):
    def test_view(self):
        pass
"""


def named(files: dict[str, str], *cases: str) -> Repro:
    return Repro.model_validate({**repro(files).model_dump(), "cases": cases})


def test_pytest_node_ids_are_placed_in_their_file_and_run_as_given() -> None:
    path = "tests/test_views.py"
    ids = (
        f"{path}::test_top",
        f"{path}::test_param[2]",
        f"{path}::TestGroup::test_method",
        f"{path}::TestGroup::TestInner::test_deep",
        f"{path}::Views::test_inherited",  # inherited from a class of the same file
    )
    placed = place_cases(named({path: PLACED}, *ids), RunnerKind.PYTEST)
    assert placed.refs[path] == (
        CaseRef((), "test_top"),
        CaseRef((), "test_param"),
        CaseRef(("TestGroup",), "test_method"),
        CaseRef(("TestGroup", "TestInner"), "test_deep"),
        CaseRef(("Views",), "test_inherited"),
    )
    assert (placed.tests, placed.expect, placed.unplaced, placed.whole) == (ids, ids, (), frozenset())
    located = locate(ast.parse(PLACED), CaseRef(("Views",), "test_inherited"))
    assert located is not None and [c.name for c in located[0]] == ["Views"] and located[1].lineno == 24


def test_a_pytest_class_id_names_its_tests_and_an_unknown_function_reads_its_file_whole() -> None:
    path = "tests/test_views.py"
    whole_class = place_cases(named({path: PLACED}, f"{path}::TestGroup"), RunnerKind.PYTEST)
    assert whole_class.refs[path] == (CaseRef(("TestGroup",), "test_method"),)
    unknown = place_cases(named({path: PLACED}, f"{path}::test_generated"), RunnerKind.PYTEST)
    assert (unknown.whole, unknown.tests) == (frozenset({path}), (f"{path}::test_generated",))
    elsewhere = place_cases(named({path: PLACED}, "tests/test_other.py::test_top", "test_top"), RunnerKind.PYTEST)
    assert elsewhere.unplaced == ("tests/test_other.py::test_top", "test_top") and elsewhere.tests == ()


def test_django_labels_are_placed_by_their_module_and_swe_bench_s_form_becomes_a_label() -> None:
    path = "tests/views/test_views.py"
    placed = place_cases(
        named(
            {path: PLACED, "tests/views/__init__.py": ""},
            "views.test_views.Views.test_view",
            "test_inherited (views.test_views.Views)",
            "views.test_views.Base",
        ),
        RunnerKind.DJANGO,
    )
    assert placed.refs[path] == (
        CaseRef(("Views",), "test_view"),
        CaseRef(("Views",), "test_inherited"),
        CaseRef(("Base",), "test_inherited"),
    )
    labels = ("views.test_views.Views.test_view", "views.test_views.Views.test_inherited", "views.test_views.Base")
    assert placed.tests == labels and placed.expect == labels
    assert django_label("test_x (app.tests.Class.test_x)") == "app.tests.Class.test_x"  # Python 3.11's form
    missing = place_cases(named({path: PLACED}, "other.tests.Views.test_view"), RunnerKind.DJANGO)
    assert missing.unplaced == ("other.tests.Views.test_view",)


def test_sympy_cases_run_as_path_and_name_and_a_bare_name_in_every_file_that_defines_it() -> None:
    first, second = "sympy/core/tests/test_a.py", "sympy/core/tests/test_b.py"
    files = {first: "def test_same():\n    pass\n", second: "def test_same():\n    pass\n\ndef test_b():\n    pass\n"}
    bare = place_cases(named(files, "test_same"), RunnerKind.SYMPY)
    assert bare.tests == (f"{first}::test_same", f"{second}::test_same") and bare.expect == ("test_same",)
    assert bare.refs == {first: (CaseRef((), "test_same"),), second: (CaseRef((), "test_same"),)}
    pathed = place_cases(named(files, f"{second}::test_b"), RunnerKind.SYMPY)
    assert (pathed.tests, pathed.expect, dict(pathed.refs)) == (
        (f"{second}::test_b",),
        (f"{second}::test_b",),
        {second: (CaseRef((), "test_b"),)},
    )
    assert place_cases(named(files, "test_none"), RunnerKind.SYMPY).unplaced == ("test_none",)


def test_every_case_a_file_defines_is_a_place_when_no_case_is_named() -> None:
    path = "tests/test_views.py"
    defined = defined_cases(
        repro({path: PLACED, "tests/helper.py": "def test_not_a_test_module():\n    pass\n"}), RunnerKind.PYTEST
    )
    assert defined == {
        path: (
            CaseRef((), "test_top"),
            CaseRef((), "test_param"),
            CaseRef(("TestGroup",), "test_method"),
            CaseRef(("Base",), "test_inherited"),
        )  # Views subclasses a TestCase only through Base, which expected_cases does not follow either
    }
    assert len(defined[path]) == len(expected_cases(repro({path: PLACED}), RunnerKind.PYTEST))
