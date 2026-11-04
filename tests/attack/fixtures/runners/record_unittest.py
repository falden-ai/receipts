"""Record real unittest output, the progress lines and blocks Django's runner prints, under a local Python (no Docker).

Run it with the interpreter to record: ``uv run --no-project --python 3.11 python
tests/attack/fixtures/runners/record_unittest.py OUT_DIR``. It writes the test modules below into a scratch
``tests/`` directory, runs ``python -m unittest -v`` on them there (Django's DiscoverRunner prints its progress through
the same unittest result class), and rewrites the scratch path to /testbed/tests, as in the SWE-bench images, and the
interpreter's own library path to a python image's (/usr/local/lib/python3.X). Only the run time ("in 0.123s") is
normalized besides.

What is recorded, per Python version: ``unittest_subtests_py<v>.txt``, the stderr of a test with a failing subtest, a
passing test and a test whose labeled subtest errors; ``unittest_hooks_py<v>.json``, a run with a setUpClass that
errors, a setUpClass that skips, a tearDownClass that errors and a setUpModule that errors, among tests that pass and
fail; ``unittest_unexpected_success_py<v>.json``, a run whose only odd case is an unexpected success;
``unittest_no_newline_py<v>.json``, tests that write to stderr with no newline before their status, or end their own
output in "ok" before they skip; ``unittest_assert_hooks_py<v>.json``, a setUpClass, a tearDownClass and a
setUpModule that each fail an assert, among tests that pass and fail; ``unittest_failed_imports_py<v>.json``, a
run whose labels name two modules called tests that fail to import (one imports a module that does not exist, one a
name its module does not have), a module that imports, a class that module does not have, and an app that does not
exist; ``unittest_expected_failures_py<v>.json``, expectedFailure tests that fail as expected (a plain one, one with a
docstring, one that writes to stderr with no newline first, and a whole class marked), beside an unexpected
success, a skip and a pass; ``unittest_imitated_status_py<v>.json``, a passing test whose own output is the line
"expected failure", which unittest's summary does not count; and ``unittest_masked_expected_failure_py<v>.json``,
an expectedFailure test whose own output is the line "ok" before it fails as expected, so its progress line reads
"ok" while unittest's summary counts one expected failure more than the progress lines show.
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

SUBTESTS = """import unittest


class T(unittest.TestCase):
    def test_a_sub(self):
        for i in range(2):
            with self.subTest(i=i):
                self.assertEqual(i % 2, 0)

    def test_b_ok(self):
        pass

    def test_c_sub_error(self):
        with self.subTest("labeled"):
            raise ValueError("boom")
"""
HOOKS = """import unittest


class A(unittest.TestCase):
    def test_ok(self):
        pass


class B(unittest.TestCase):
    def test_fails(self):
        self.assertEqual(1, 2)


class C(unittest.TestCase):
    def test_after_the_hooks(self):
        pass


class SetUpClassFails(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise RuntimeError("setUpClass boom")

    def test_never_runs(self):
        pass


class SetUpClassSkips(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        raise unittest.SkipTest("no backend")

    def test_skipped_by_its_class(self):
        pass


class TearDownClassFails(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        raise RuntimeError("tearDownClass boom")

    def test_runs(self):
        pass
"""
HOOK_MODULE = """import unittest


def setUpModule():
    raise RuntimeError("setUpModule boom")


class D(unittest.TestCase):
    def test_never_runs(self):
        pass
"""
UNEXPECTED = """import unittest


class OnlyAnUnexpectedSuccess(unittest.TestCase):
    @unittest.expectedFailure
    def test_passes_unexpectedly(self):
        self.assertEqual(1, 1)
"""
NO_NEWLINE = """import sys
import unittest


class NoNewline(unittest.TestCase):
    def test_a_writes_then_passes(self):
        sys.stderr.write("x")

    def test_b_writes_then_fails(self):
        sys.stderr.write("x")
        self.assertEqual(1, 2)

    def test_c_ends_its_output_in_ok_then_skips(self):
        sys.stderr.write("it is ok\\n")
        self.skipTest("skipped after its own output")

    def test_d_after(self):
        pass
"""
ASSERT_HOOKS = """import unittest


class A(unittest.TestCase):
    def test_ok(self):
        pass


class SetUpClassAsserts(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        assert 1 == 2, "setUpClass assert"

    def test_never_runs(self):
        pass


class TearDownClassAsserts(unittest.TestCase):
    @classmethod
    def tearDownClass(cls):
        assert 1 == 2, "tearDownClass assert"

    def test_runs(self):
        pass


class B(unittest.TestCase):
    def test_fails(self):
        self.assertEqual(1, 2)
"""
ASSERT_HOOK_MODULE = """import unittest


def setUpModule():
    assert 1 == 2, "setUpModule assert"


class D(unittest.TestCase):
    def test_never_runs(self):
        pass
"""
IMPORTS_MISSING_MODULE = """import unittest

import receipts_nowhere  # noqa: F401


class T(unittest.TestCase):
    def test_x(self):
        pass
"""
IMPORTS_MISSING_NAME = """import unittest
from json import receipts_no_such_name  # noqa: F401


class T(unittest.TestCase):
    def test_y(self):
        pass
"""
IMPORTS_FINE = """import unittest


class T(unittest.TestCase):
    def test_ok(self):
        pass
"""
EXPECTED_FAILURES = '''import sys
import unittest


class ExpectedFailures(unittest.TestCase):
    @unittest.expectedFailure
    def test_a_fails_as_expected(self):
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_b_with_a_docstring(self):
        """A docstring's first line shows in the verbose output."""
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_c_writes_then_fails_as_expected(self):
        sys.stderr.write("x")
        self.assertEqual(1, 2)

    @unittest.expectedFailure
    def test_d_passes_unexpectedly(self):
        pass

    @unittest.skip("not here")
    def test_e_skipped(self):
        pass

    def test_f_ok(self):
        pass


@unittest.expectedFailure
class WholeClassExpected(unittest.TestCase):
    def test_g_fails_as_expected(self):
        self.assertEqual(1, 2)
'''
IMITATED_STATUS = """import sys
import unittest


class Imitates(unittest.TestCase):
    def test_prints_a_status_then_passes(self):
        sys.stderr.write("expected failure\\n")

    def test_after(self):
        pass
"""
MASKED_EXPECTED_FAILURE = """import sys
import unittest


class Masked(unittest.TestCase):
    @unittest.expectedFailure
    def test_a_says_ok_then_fails_as_expected(self):
        sys.stderr.write("ok\\n")
        self.assertEqual(1, 2)

    def test_b_after(self):
        pass
"""
MODULES = {
    "subtests": SUBTESTS,
    "hooks": HOOKS,
    "hookmodule": HOOK_MODULE,
    "unexpected": UNEXPECTED,
    "nonewline": NO_NEWLINE,
    "asserthooks": ASSERT_HOOKS,
    "asserthookmodule": ASSERT_HOOK_MODULE,
    "brokena": IMPORTS_MISSING_MODULE,
    "brokenb": IMPORTS_MISSING_NAME,
    "fine": IMPORTS_FINE,
    "expected": EXPECTED_FAILURES,
    "imitates": IMITATED_STATUS,
    "masked": MASKED_EXPECTED_FAILURE,
}
RUNS = {
    "hooks": ["hooks.tests", "hookmodule.tests"],
    "unexpected_success": ["unexpected.tests"],
    "no_newline": ["nonewline.tests"],
    "assert_hooks": ["asserthooks.tests", "asserthookmodule.tests"],
    "failed_imports": [
        "brokena.tests",
        "brokenb.tests.T.test_y",
        "fine.tests",
        "fine.tests.Missing",
        "nosuchapp.tests",
    ],
    "expected_failures": ["expected.tests"],
    "imitated_status": ["imitates.tests"],
    "masked_expected_failure": ["masked.tests"],
}


def run(tests: Path, modules: list[str]) -> tuple[list[str], int, str]:
    argv = [sys.executable, "-m", "unittest", "-v", *modules]
    proc = subprocess.run(argv, cwd=tests, capture_output=True, text=True, timeout=120, check=False)
    text = proc.stderr.replace(str(tests), "/testbed/tests")
    # The interpreter's own library (its unittest loader, its json) becomes that of a python image, so nothing of the
    # machine that recorded it stays in the file.
    stdlib = Path(json.__file__).parent.parent
    image_lib = f"/usr/local/lib/python{sys.version_info.major}.{sys.version_info.minor}"
    for spelling in dict.fromkeys((str(stdlib.resolve()), str(stdlib))):
        text = text.replace(spelling, image_lib)
    text = re.sub(r"(Ran \d+ tests? in )[0-9.]+s", r"\g<1>0.001s", text)
    return ["python", "-m", "unittest", "-v", *modules], proc.returncode, text


def main() -> int:
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    version = f"{sys.version_info.major}{sys.version_info.minor}"
    with tempfile.TemporaryDirectory(prefix="receipts-ut-") as tmp:
        tests = Path(tmp).resolve() / "tests"
        for name, source in MODULES.items():
            (tests / name).mkdir(parents=True)
            (tests / name / "__init__.py").write_text("")
            (tests / name / "tests.py").write_text(source)
        _, _, subtests = run(tests, ["subtests.tests"])
        (out / f"unittest_subtests_py{version}.txt").write_text(subtests)
        for name, modules in RUNS.items():
            argv, code, stderr = run(tests, modules)
            record = {"python": sys.version.split()[0], "argv": argv, "exit_code": code, "stderr": stderr}
            (out / f"unittest_{name}_py{version}.json").write_text(json.dumps(record, indent=1) + "\n")
            print(f"wrote unittest_{name}_py{version}.json: exit {code}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
