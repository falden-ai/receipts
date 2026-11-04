"""AT3's test selection (plain changed-file membership, no score, no ranking) and what a test file defines; and the
diff reader the attackers share."""

from __future__ import annotations

import pytest

from receipts.attack.diffs import changed_files, file_changes, module_name, truncate
from receipts.attack.selection import arguments_for, case_ids, find_case_file, is_test_file, select_tests
from receipts.contracts import CaseResult, FailureType, Outcome, RunnerKind

TREE = {
    "src/pkg/__init__.py": "",
    "src/pkg/parser.py": "def parse(x):\n    return x\n",
    "src/pkg/_util.py": "def helper():\n    return 1\n",
    "src/pkg/tests/test_util.py": "from pkg._util import helper\n",
    "tests/test_parser.py": "from pkg.parser import parse\n",
    "tests/test_imports_parser.py": "import pkg.parser\n",
    "tests/test_from_pkg.py": "from pkg import parser\n",
    "tests/test_mentions_only.py": "# pkg.parser is mentioned in a comment, never imported\n",
    "tests/test_unrelated.py": "import os\n",
    "tests/helpers.py": "import pkg.parser\n",
    "docs/conf.py": "",
}


def read(path: str) -> str:
    return TREE[path]


def test_the_three_rules_in_their_fixed_order() -> None:
    changed = ["src/pkg/parser.py", "tests/test_unrelated.py", "src/pkg/_util.py"]
    sel = select_tests(list(TREE), changed, kind=RunnerKind.PYTEST, read=read)
    assert sel.rules == {
        "tests/test_unrelated.py": 1,  # an existing test the agent edited
        "src/pkg/tests/test_util.py": 2,  # test_<stem>.py in tests/ beside the source, leading underscore dropped
        "tests/test_parser.py": 2,  # and under the top-level tests/
        "tests/test_from_pkg.py": 3,  # imports the changed module by name
        "tests/test_imports_parser.py": 3,
    }
    assert list(sel.tests) == list(sel.rules)  # rule order, then path order within a rule: no ranking
    assert sel.overflow == ()
    # A mention that is not an import, and a helper that is not a test file, are never selected.
    assert "tests/test_mentions_only.py" not in sel.tests and "tests/helpers.py" not in sel.tests


def test_the_cap_lists_the_rest_as_overflow() -> None:
    changed = ["src/pkg/parser.py"]
    sel = select_tests(list(TREE), changed, kind=RunnerKind.PYTEST, read=read, cap=1)
    assert sel.tests == ("tests/test_parser.py",)
    assert sel.overflow == ("tests/test_from_pkg.py", "tests/test_imports_parser.py")


def test_a_new_test_file_is_not_an_existing_test() -> None:
    sel = select_tests(list(TREE), ["tests/test_brand_new.py"], kind=RunnerKind.PYTEST, read=read)
    assert sel.tests == ()


def test_a_file_that_cannot_be_read_is_skipped_not_fatal() -> None:
    def broken(path: str) -> str:
        raise OSError("gone")

    assert select_tests(list(TREE), ["src/pkg/parser.py"], kind=RunnerKind.PYTEST, read=broken).tests == (
        "tests/test_parser.py",
    )


def test_django_and_sympy_test_files() -> None:
    assert is_test_file("tests/model_fields/tests.py", RunnerKind.DJANGO)
    assert is_test_file("tests/model_fields/test_foo.py", RunnerKind.DJANGO)
    assert not is_test_file("tests/test_sqlite.py", RunnerKind.DJANGO)  # the settings module
    assert not is_test_file("tests/runtests.py", RunnerKind.DJANGO)
    assert not is_test_file("django/test/testcases.py", RunnerKind.DJANGO)
    assert is_test_file("sympy/core/tests/test_numbers.py", RunnerKind.SYMPY)
    assert not is_test_file("sympy/core/numbers.py", RunnerKind.SYMPY)
    tree = {
        "django/db/models/query.py": "",
        "tests/queries/tests.py": "from django.db.models.query import QuerySet\n",
        "tests/basic/tests.py": "from django.db import models\n",
    }
    sel = select_tests(list(tree), ["django/db/models/query.py"], kind=RunnerKind.DJANGO, read=tree.__getitem__)
    assert sel.tests == ("tests/queries/tests.py",)
    assert arguments_for(RunnerKind.DJANGO, sel.tests) == ["queries.tests"]
    assert arguments_for(RunnerKind.SYMPY, ["sympy/a/tests/test_b.py"]) == ["sympy/a/tests/test_b.py"]


def test_case_ids_come_from_the_syntax_tree() -> None:
    code = (
        "import unittest\nfrom django.test import TestCase\n\n\ndef test_top():\n    pass\n\n\n"
        "async def test_async():\n    pass\n\n\ndef helper():\n    pass\n\n\n"
        "class TestK:\n    def test_m(self):\n        pass\n\n\n"
        "class TestWithInit:\n    def __init__(self):\n        pass\n\n    def test_never(self):\n        pass\n\n\n"
        "class Django(TestCase):\n    def test_d(self):\n        pass\n\n    def other(self):\n        pass\n"
    )
    assert case_ids(RunnerKind.PYTEST, "tests/test_x.py", code) == [
        "tests/test_x.py::test_top",
        "tests/test_x.py::test_async",
        "tests/test_x.py::TestK::test_m",
        "tests/test_x.py::Django::test_d",
    ]
    assert case_ids(RunnerKind.DJANGO, "tests/app/test_x.py", code) == ["app.test_x.Django.test_d"]
    assert case_ids(RunnerKind.SYMPY, "sympy/a/tests/test_x.py", code) == [
        "sympy/a/tests/test_x.py::test_top",
        "sympy/a/tests/test_x.py::test_async",
    ]
    with pytest.raises(SyntaxError):
        case_ids(RunnerKind.PYTEST, "tests/test_x.py", "def test_broken(:\n")


def test_find_case_file() -> None:
    files = ["sympy/a/tests/test_b.py", "sympy/a/tests/test_c.py"]
    texts = {files[0]: "def test_one():\n    pass\n", files[1]: "def test_two():\n    pass\n"}
    assert (
        find_case_file(
            RunnerKind.SYMPY, CaseResult(name="test_two", outcome=Outcome.PASS, file=files[1]), files, texts.__getitem__
        )
        == files[1]
    )
    assert (
        find_case_file(RunnerKind.SYMPY, CaseResult(name="test_two", outcome=Outcome.PASS), files, texts.__getitem__)
        == files[1]
    )
    assert (
        find_case_file(RunnerKind.SYMPY, CaseResult(name="test_three", outcome=Outcome.PASS), files, texts.__getitem__)
        is None
    )
    node = CaseResult(name="t/test_a.py::test_x", outcome=Outcome.PASS)
    assert find_case_file(RunnerKind.PYTEST, node, ["t/test_a.py"], texts.__getitem__) == "t/test_a.py"
    django = {
        "tests/app/tests.py": "from django.test import TestCase\n\n\n"
        "class T(TestCase):\n    def test_x(self):\n        pass\n"
    }
    label = CaseResult(name="app.tests.T.test_x", outcome=Outcome.PASS)
    assert find_case_file(RunnerKind.DJANGO, label, list(django), django.__getitem__) == "tests/app/tests.py"


def test_a_sympy_case_is_found_in_its_own_file_when_two_files_name_it() -> None:
    # PLANT (review blocker): the first selected file defining the name was taken, so a regression in test_b.py
    # became a repro of test_a.py's test_x, which passes.
    a, b = "sympy/core/tests/test_a.py", "sympy/core/tests/test_b.py"
    texts = {a: "def test_x():\n    pass\n", b: "def test_x():\n    pass\n"}
    case = CaseResult(name="test_x", outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION, file=b)
    assert find_case_file(RunnerKind.SYMPY, case, [a, b], texts.__getitem__) == b
    # With no file, two files that define it cannot be told apart: no file, never the first.
    nameless = CaseResult(name="test_x", outcome=Outcome.PASS)
    assert find_case_file(RunnerKind.SYMPY, nameless, [a, b], texts.__getitem__) is None
    # A file the case names must be selected and must define the case.
    elsewhere = CaseResult(name="test_x", outcome=Outcome.PASS, file="sympy/core/tests/test_c.py")
    assert find_case_file(RunnerKind.SYMPY, elsewhere, [a, b], texts.__getitem__) is None
    missing = CaseResult(name="test_y", outcome=Outcome.PASS, file=a)
    assert find_case_file(RunnerKind.SYMPY, missing, [a, b], texts.__getitem__) is None


# -- diffs -------------------------------------------------------------------------------------------------------------

PATCH = (
    "diff --git a/src/pkg/parser.py b/src/pkg/parser.py\n--- a/src/pkg/parser.py\n+++ b/src/pkg/parser.py\n"
    "@@ -1,2 +1,3 @@\n def parse(x):\n-    return x\n+    y = x\n+    return y\n"
    "diff --git a/new.py b/new.py\nnew file mode 100644\n--- /dev/null\n+++ b/new.py\n@@ -0,0 +1 @@\n+--- a/fake\n"
    "diff --git a/gone.py b/gone.py\ndeleted file mode 100644\n--- a/gone.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-x\n"
    "diff --git a/old_name.py b/new_name.py\nsimilarity index 90%\nrename from old_name.py\nrename to new_name.py\n"
    "diff --git a/../escape.py b/../escape.py\n--- a/../escape.py\n+++ b/../escape.py\n"
)


def test_file_changes_read_only_gits_headers() -> None:
    changes = file_changes(PATCH)
    assert [(c.old, c.new) for c in changes] == [
        ("src/pkg/parser.py", "src/pkg/parser.py"),
        (None, "new.py"),  # a removed-looking line inside the hunk is content, not a header
        ("gone.py", None),
        ("old_name.py", "new_name.py"),
        (None, None),  # a path out of the repository is dropped
    ]
    assert changes[0].lines == [2, 3]
    assert changed_files(PATCH) == ["src/pkg/parser.py", "new.py", "gone.py", "new_name.py", "old_name.py"]


def test_module_names_and_truncation() -> None:
    assert module_name("src/_pytest/python.py") == "_pytest.python"
    assert module_name("django/db/models/fields/__init__.py") == "django.db.models.fields"
    assert module_name("lib/matplotlib/axes/_axes.py") == "matplotlib.axes._axes"
    assert module_name("setup.cfg") is None and module_name("bad-name/x.py") is None
    assert truncate("abcdef", 3) == "abc\n[... 3 characters cut ...]" and truncate("abc", 3) == "abc"
