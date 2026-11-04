"""SWE-bench's test command for every repository and version on the pool, read from the pinned swebench 4.1.0 wheel,
and the runner each repository needs in the run phase (contracts.RunnerKind).

`Target.test_command` is the repository's own test command; for Tier A it is SWE-bench's spec, the `test_cmd` of
MAP_REPO_VERSION_TO_SPECS for the instance's version (swebench/harness/constants/python.py). That module is plain
data, built with dict comprehensions, a few loops and `.update` calls. It is never imported or run as it ships: `derive`
parses it, refuses any construct outside a short list of data-building ones (no import, no attribute but the three
list and dict methods, no call but those and `int`), and only then evaluates it with no builtins beyond `int`. The
result is committed as eval/test-specs.json with the wheel's sha256, so building targets needs neither swebench nor
the network.

Test files: the harness runs the files its test patch touches (get_test_directives), minus a fixed list of non-test
extensions; for Django it turns each path into a module label. `harness_test_files` gives the paths, untransformed:
the run phase's runners take repository paths and convert them themselves.

Environment: the harness's eval script runs a spec's eval_commands before the tests (test_spec/python.py,
make_eval_script_list_py, lines 431 to 432), and for Django 1.11 to 3.2 those export LANG, LC_ALL, LANGUAGE and, up to
2.2, PYTHONIOENCODING. `run_phase_test_command` carries every such export into Target.test_command as a leading
NAME=value assignment, which the run phase's runners pass to the test as its environment. The other line Django 3.x has
(`sed ... /etc/locale.gen && locale-gen`) needs root and only builds a locale the images already ship: the harness's
base image installs locales-all (dockerfiles/python.py, line 19). The SWE-ABS label runs and the harness gold check run
the harness's own eval script, so they get the exports from it.

Expected failures: swebench 4.1.0 grades a test by its repository's log parser (`LOG_PARSERS`), and grading.py's
test_passed (lines 27 to 28) counts XFAIL as passing. Only the pytest-style parsers read pytest's `XFAIL <id>` line
(`XFAIL_PARSERS`); the Django, seaborn and sympy parsers never write XFAIL, so an expected failure there is absent from
the status map, which test_failed (lines 31 to 35) counts as failed. `expected_failure_passes` says which repositories
count one as passing.
"""

from __future__ import annotations

import ast
import json
import re
import shlex
import zipfile
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from receipts.contracts import Frozen, RunnerKind
from receipts.eval.hashing import SHA256, pretty_json, sha256_bytes

SPECS_FORMAT: Literal["receipts.eval.test-specs/2"] = "receipts.eval.test-specs/2"
CONSTANTS = "swebench/harness/constants/python.py"
SPECS_FILE = "test-specs.json"  # under eval/

# swebench 4.1.0, harness/constants/__init__.py NON_TEST_EXTS, copied as the harness has it ("csv" has no dot).
NON_TEST_EXTS = (".json", ".png", "csv", ".txt", ".md", ".jpg", ".jpeg", ".pkl", ".yml", ".yaml", ".toml")
_DIFF = re.compile(r"diff --git a/.* b/(.*)")

_NODES: tuple[type[ast.AST], ...] = (
    ast.Module,
    ast.Assign,
    ast.AugAssign,
    ast.Expr,
    ast.For,
    ast.If,
    ast.Compare,
    ast.In,
    ast.NotIn,
    ast.Eq,
    ast.NotEq,
    ast.Name,
    ast.Load,
    ast.Store,
    ast.Constant,
    ast.List,
    ast.Tuple,
    ast.Dict,
    ast.Set,
    ast.DictComp,
    ast.ListComp,
    ast.comprehension,
    ast.Subscript,
    ast.BinOp,
    ast.Add,
    ast.BitOr,
    ast.Call,
    ast.Attribute,
    ast.keyword,
    ast.Starred,
)
_METHODS = frozenset({"update", "extend", "append"})
_CALLS = frozenset({"int"})


class SpecsError(ValueError):
    """A constants module that is not plain data, or a spec this lane cannot use."""


def _check_tree(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if not isinstance(node, _NODES):
            raise SpecsError(f"line {getattr(node, 'lineno', '?')}: {type(node).__name__} is not plain data")
        if isinstance(node, ast.Attribute) and node.attr not in _METHODS:
            raise SpecsError(f"line {node.lineno}: attribute .{node.attr} is not one of {sorted(_METHODS)}")
        if isinstance(node, ast.Name) and node.id.startswith("__"):
            raise SpecsError(f"line {node.lineno}: dunder name {node.id}")
        if isinstance(node, ast.Call):
            f = node.func
            if isinstance(f, ast.Attribute):
                continue  # its attribute name was checked above
            if not (isinstance(f, ast.Name) and f.id in _CALLS):
                raise SpecsError(f"line {node.lineno}: call to {ast.unparse(f)} is not allowed")


def evaluate_constants(source: str) -> dict[str, Any]:
    """The module's globals, evaluated only after every node passed the whitelist."""
    tree = ast.parse(source)
    _check_tree(tree)
    scope: dict[str, Any] = {"__builtins__": {"int": int}}
    exec(compile(tree, CONSTANTS, "exec"), scope)  # plain data only: every node passed _check_tree
    scope.pop("__builtins__")
    return scope


class RepoSpec(Frozen):
    repo: str
    version: str
    test_cmd: str = Field(min_length=1)
    runner: RunnerKind
    install: str = ""  # the eval script reinstalls the package with this, as root, before the tests
    eval_commands: tuple[str, ...] = ()  # extra setup the harness runs first (Django's locale lines)


class SpecTable(Frozen):
    """eval/test-specs.json. `instances` gives the version of every pool instance (its dataset row's), so a committed
    target can be checked against its own version's spec without the rows, which are not in git."""

    format: Literal["receipts.eval.test-specs/2"] = SPECS_FORMAT
    swebench_version: str
    wheel_sha256: str = Field(pattern=SHA256)
    constants_sha256: str = Field(pattern=SHA256)
    specs: tuple[RepoSpec, ...]
    instances: dict[str, str] = Field(default_factory=dict)  # pool instance id -> its version

    def get(self, repo: str, version: str) -> RepoSpec:
        for s in self.specs:
            if s.repo == repo and s.version == version:
                return s
        raise SpecsError(f"no SWE-bench spec for {repo} {version}")

    def for_instance(self, instance_id: str, repo: str) -> RepoSpec:
        """The spec of the version this pool instance runs, never another version's."""
        version = self.instances.get(instance_id)
        if version is None:
            raise SpecsError(f"{instance_id} is not in the table's pool")
        return self.get(repo, version)


def runner_kind(repo: str) -> RunnerKind:
    """django for django/django, sympy for sympy/sympy, pytest for everything else (sphinx's tox ends in pytest)."""
    return {"django/django": RunnerKind.DJANGO, "sympy/sympy": RunnerKind.SYMPY}.get(repo, RunnerKind.PYTEST)


# swebench 4.1.0, harness/log_parsers/python.py: MAP_REPO_TO_PARSER_PY (lines 270 to 289), each alias resolved to the
# function it names (lines 253 to 267).
LOG_PARSERS: dict[str, str] = {
    "astropy/astropy": "parse_log_pytest_v2",
    "django/django": "parse_log_django",
    "marshmallow-code/marshmallow": "parse_log_pytest",
    "matplotlib/matplotlib": "parse_log_matplotlib",
    "mwaskom/seaborn": "parse_log_seaborn",
    "pallets/flask": "parse_log_pytest",
    "psf/requests": "parse_log_pytest_options",
    "pvlib/pvlib-python": "parse_log_pytest",
    "pydata/xarray": "parse_log_pytest",
    "pydicom/pydicom": "parse_log_pytest_options",
    "pylint-dev/astroid": "parse_log_pytest",
    "pylint-dev/pylint": "parse_log_pytest_options",
    "pytest-dev/pytest": "parse_log_pytest",
    "pyvista/pyvista": "parse_log_pytest",
    "scikit-learn/scikit-learn": "parse_log_pytest_v2",
    "sqlfluff/sqlfluff": "parse_log_pytest",
    "sphinx-doc/sphinx": "parse_log_pytest_v2",
    "sympy/sympy": "parse_log_sympy",
}
# The parsers that keep every line starting with a TestStatus value, XFAIL included, so pytest's -rA line
# `XFAIL <id>` becomes status XFAIL: parse_log_pytest (line 7), parse_log_pytest_options (29), parse_log_pytest_v2
# (144) and parse_log_matplotlib (229). parse_log_django (64), parse_log_seaborn (173) and parse_log_sympy (199) never
# write XFAIL.
XFAIL_PARSERS = frozenset(
    {"parse_log_pytest", "parse_log_pytest_options", "parse_log_pytest_v2", "parse_log_matplotlib"}
)


def expected_failure_passes(repo: str) -> bool:
    """Whether swebench 4.1.0 grades a test that failed as expected as passing in this repository: only where its log
    parser reads XFAIL. False for a repository it has no parser for, so a guess never keeps a gold."""
    return LOG_PARSERS.get(repo) in XFAIL_PARSERS


_EXPORT = re.compile(r"(?P<name>[A-Za-z_][A-Za-z0-9_]*)=(?P<value>[A-Za-z0-9_.:,/@%+=-]*)")


def env_exports(eval_commands: Iterable[str]) -> tuple[str, ...]:
    """The environment the harness's eval script exports before the tests: every `export NAME=value` line of a spec's
    eval_commands, as NAME=value words, in order (a later export of a name sets its value). A line that is anything
    else, or whose value a shell would expand or split, exports nothing here."""
    out: dict[str, str] = {}
    for line in eval_commands:
        try:
            words = shlex.split(line)
        except ValueError:
            continue
        if len(words) < 2 or words[0] != "export":
            continue
        found = [_EXPORT.fullmatch(w) for w in words[1:]]  # a value with $, a backquote or a space never matches
        if any(m is None for m in found):
            continue
        for m in found:
            assert m is not None
            out[m.group("name")] = m.group("value")
    return tuple(f"{name}={value}" for name, value in out.items())


def run_phase_test_command(spec: RepoSpec) -> str:
    """Target.test_command for an instance of this spec: SWE-bench's test_cmd, after the environment its eval script
    exports first, as leading NAME=value assignments (`env_exports`). A spec that exports nothing keeps its test_cmd."""
    exports = env_exports(spec.eval_commands)
    return " ".join((*exports, spec.test_cmd)) if exports else spec.test_cmd


def _test_cmd(spec: Mapping[str, Any], where: str) -> str:
    cmd = spec.get("test_cmd")
    if isinstance(cmd, list):  # get_logs_eval reads the last one
        cmd = cmd[-1] if cmd else None
    if not isinstance(cmd, str) or not cmd.strip():
        raise SpecsError(f"{where}: no test_cmd")
    return cmd


def derive(
    wheel: Path,
    wheel_sha256: str,
    pairs: Iterable[tuple[str, str]],
    *,
    instances: Mapping[str, tuple[str, str]] | None = None,
) -> SpecTable:
    """The spec for every (repo, version) in `pairs`, from the pinned wheel; refuses a wheel with other bytes.
    `instances` (pool instance id -> its repo and version) is recorded as each instance's version, and each of its
    pairs must be one of `pairs`."""
    wanted = set(pairs)
    versions: dict[str, str] = {}
    for iid, pair in sorted((instances or {}).items()):
        if pair not in wanted:
            raise SpecsError(f"{iid}: {pair[0]} {pair[1]} is not one of the table's repository versions")
        versions[iid] = pair[1]
    data = wheel.read_bytes()
    if sha256_bytes(data) != wheel_sha256:
        raise SpecsError(f"{wheel}: not the pinned swebench wheel")
    with zipfile.ZipFile(wheel) as z:
        source = z.read(CONSTANTS)
        metadata = next((n for n in z.namelist() if n.endswith(".dist-info/METADATA")), None)
        meta = z.read(metadata).decode("utf-8", "replace") if metadata else ""
    version = next((line.split(":", 1)[1].strip() for line in meta.splitlines() if line.startswith("Version:")), "")
    if not version:
        raise SpecsError(f"{wheel}: no Version in its METADATA")
    table = evaluate_constants(source.decode("utf-8")).get("MAP_REPO_VERSION_TO_SPECS_PY")
    if not isinstance(table, dict):
        raise SpecsError("MAP_REPO_VERSION_TO_SPECS_PY is missing")
    specs = []
    for repo, ver in sorted(wanted):
        spec = table.get(repo, {}).get(ver)
        if not isinstance(spec, dict):
            raise SpecsError(f"swebench {version} has no spec for {repo} {ver}")
        install = spec.get("install", "")
        evals = spec.get("eval_commands", [])
        specs.append(
            RepoSpec(
                repo=repo,
                version=ver,
                test_cmd=_test_cmd(spec, f"{repo} {ver}"),
                runner=runner_kind(repo),
                install=install if isinstance(install, str) else "",
                eval_commands=tuple(str(c) for c in evals) if isinstance(evals, list) else (),
            )
        )
    return SpecTable(
        swebench_version=version,
        wheel_sha256=wheel_sha256,
        constants_sha256=sha256_bytes(source),
        specs=tuple(specs),
        instances=versions,
    )


def dump(specs: SpecTable) -> str:
    return pretty_json(specs.model_dump(mode="json"))


def load(text: str) -> SpecTable:
    return SpecTable.model_validate(json.loads(text))


def harness_test_files(test_patch: str) -> list[str]:
    """The files the harness runs for a test patch (get_test_directives), as repository paths, in patch order."""
    out: list[str] = []
    for path in _DIFF.findall(test_patch):
        if not any(path.endswith(ext) for ext in NON_TEST_EXTS) and path not in out:
            out.append(path)
    return out


def changed_files(patch: str) -> list[str]:
    """Every file a unified diff touches, by its new path, in patch order."""
    out: list[str] = []
    for path in _DIFF.findall(patch):
        if path not in out:
            out.append(path)
    return out
