"""Which existing tests are over the changed files (AT3), and what a test file defines (the attackers' repros).

Selection is plain changed-file membership, with no score and no ranking (the spec's AT3 row: nothing weighted, no
IR match, never coverage counts, TF-IDF or BM25). A repository test file is selected when one of three yes-or-no
rules holds, applied in a fixed order:

1. it is itself a changed file that exists before the patch (the agent edited an existing test);
2. it is a conventional test path for a changed source file: ``test_<stem>.py`` beside it, in a ``tests/`` directory
   beside it, or under the repository's top-level ``tests/`` or ``testing/`` directory (``<stem>`` without leading
   underscores);
3. it imports a changed module by name (``import a.b.c``, ``from a.b.c import x`` or ``from a.b import c``).

Within a rule the files go in path order, and past the cap the rest are listed as unchecked, not dropped silently.

A test file's cases come from its syntax tree, never from a model's say-so: the functions and methods a runner of its
kind would collect, named the way that runner names them (``receipts.attack.runners``).
"""

from __future__ import annotations

import ast
import posixpath
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from receipts.attack.diffs import module_name
from receipts.attack.runners import django_label
from receipts.attack.tracebacks import is_test_path
from receipts.contracts import CaseResult, RunnerKind

Reader = Callable[[str], str]


@dataclass(frozen=True)
class Selection:
    tests: tuple[str, ...]  # repo-relative test files, in rule order
    rules: dict[str, int]  # file -> the rule (1, 2 or 3) that selected it
    overflow: tuple[str, ...]  # files the rules selected past the cap


def is_test_file(path: str, kind: RunnerKind) -> bool:
    """A file the runner of ``kind`` would collect as tests."""
    if not path.endswith(".py"):
        return False
    base = posixpath.basename(path)
    if kind is RunnerKind.DJANGO:
        # tests/<app>/tests.py or tests/<app>/test_*.py; the top of tests/ holds the runner and its settings
        return path.startswith("tests/") and path.count("/") >= 2 and (base == "tests.py" or base.startswith("test"))
    if kind is RunnerKind.SYMPY:
        return "/tests/" in f"/{path}" and base.startswith("test_")
    return base.startswith("test_") or base.endswith("_test.py")


def _stem(path: str) -> str:
    return posixpath.splitext(posixpath.basename(path))[0].lstrip("_")


def _conventional(source: str) -> set[str]:
    """The conventional test paths for a source file, as plain paths to look up."""
    stem = _stem(source)
    if not stem or stem == "init":
        return set()
    here = posixpath.dirname(source)
    name = f"test_{stem}.py"
    parts = [p for p in here.split("/") if p and p not in ("src", "lib")]
    paths = {
        posixpath.join(here, name),
        posixpath.join(here, "tests", name),
        posixpath.join("tests", name),
        posixpath.join("testing", name),
    }
    for top in ("tests", "testing"):
        if parts:
            paths.add(posixpath.join(top, *parts[1:], name))
    return {posixpath.normpath(p) for p in paths}


def _imports(text: str) -> set[str]:
    """Every module a file imports by name, and each ``from a.b import c`` as a.b.c too."""
    try:
        tree = ast.parse(text)
    except (SyntaxError, ValueError):
        return set()
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
            names.update(f"{node.module}.{alias.name}" for alias in node.names)
    return names


def select_tests(
    files: Sequence[str],
    changed: Sequence[str],
    *,
    kind: RunnerKind,
    read: Reader,
    cap: int = 8,
    max_bytes: int = 2 * 1024 * 1024,
) -> Selection:
    """The existing test files over the changed files, by the three membership rules, capped at ``cap``."""
    present = set(files)
    tests = sorted(f for f in files if is_test_file(f, kind))
    sources = [c for c in changed if c.endswith(".py") and not is_test_path(c)]
    rule1 = [c for c in sorted(changed) if c in present and is_test_file(c, kind)]
    wanted = set().union(*(_conventional(s) for s in sources)) if sources else set()
    rule2 = [t for t in tests if t in wanted]
    modules = {m for s in sources if (m := module_name(s)) is not None}
    rule3: list[str] = []
    if modules:
        for test in tests:
            try:
                text = read(test)
            except (OSError, ValueError, RuntimeError):
                continue
            if len(text) > max_bytes or not any(m.rsplit(".", 1)[-1] in text for m in modules):
                continue  # a plain substring check first; the import check below decides
            if _imports(text) & modules:
                rule3.append(test)
    ordered: dict[str, int] = {}
    for rule, picked in ((1, rule1), (2, rule2), (3, rule3)):
        for path in picked:
            ordered.setdefault(path, rule)
    chosen = list(ordered)[:cap]
    return Selection(tuple(chosen), {p: ordered[p] for p in chosen}, tuple(list(ordered)[cap:]))


def arguments_for(kind: RunnerKind, files: Sequence[str]) -> list[str]:
    """What a runner of ``kind`` takes to run whole test files: paths, or Django labels made from them."""
    return [django_label(f) for f in files] if kind is RunnerKind.DJANGO else list(files)


def _is_testcase(node: ast.ClassDef) -> bool:
    return any("TestCase" in ast.unparse(base) for base in node.bases)


def case_ids(kind: RunnerKind, path: str, code: str) -> list[str]:
    """The cases a runner of ``kind`` collects from test file ``path``, in that runner's own id form: pytest node ids,
    Django labels, or ``path::name`` for sympy (whose case names are the bare function names). Each once: a name the
    file defines twice is one case, the later definition."""
    tree = ast.parse(code)
    cases: list[str] = []
    module = django_label(path) if kind is RunnerKind.DJANGO else path
    for node in tree.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
            if kind is RunnerKind.PYTEST or kind is RunnerKind.SYMPY:
                cases.append(f"{path}::{node.name}")
        elif isinstance(node, ast.ClassDef):
            methods = [
                item.name
                for item in node.body
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name.startswith("test")
            ]
            if not methods:
                continue
            if kind is RunnerKind.DJANGO and _is_testcase(node):
                cases += [f"{module}.{node.name}.{m}" for m in methods]
            elif kind is RunnerKind.PYTEST and (node.name.startswith("Test") or _is_testcase(node)):
                has_init = any(isinstance(i, ast.FunctionDef) and i.name == "__init__" for i in node.body)
                if not has_init:
                    cases += [f"{path}::{node.name}::{m}" for m in methods]
    return list(dict.fromkeys(cases))


def find_case_file(kind: RunnerKind, case: CaseResult, files: Sequence[str], read: Reader) -> str | None:
    """The selected test file that defines a case the runner reported. A pytest node id names its file; a sympy case
    names its file (``CaseResult.file``), since its name repeats across files, and that file must define it; a
    Django label names its module. A case whose file cannot be told (a sympy case with no file whose name more than
    one selected file defines) has none."""
    if kind is RunnerKind.PYTEST:
        path = case.name.split("::", 1)[0]
        return path if path in files else None
    if kind is RunnerKind.SYMPY and case.file is not None:
        return case.file if case.file in files and _defines(kind, case.file, case.name, read) else None
    found = [path for path in files if _defines(kind, path, case.name, read)]
    return found[0] if len(found) == 1 else None


def _defines(kind: RunnerKind, path: str, name: str, read: Reader) -> bool:
    try:
        ids = case_ids(kind, path, read(path))
    except (SyntaxError, ValueError, OSError, RuntimeError):
        return False
    return (f"{path}::{name}" if kind is RunnerKind.SYMPY else name) in ids
