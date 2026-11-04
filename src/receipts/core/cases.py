"""A repro's own test files, the cases it runs and where each lives in its files, read with no run.

The skeptic hands these to Sandbox.run_test as `tests` and `expect`, so a run passes only if every case the repro
names ran and passed, and a case that never ran is an error, never a pass.

A repro names its cases in Repro.cases (contract v3), in the runner's own ids: pytest node ids, Django labels, sympy
'path::name' or bare names. place_cases reads those ids against the repro's own files: which file and function each
case is, so the fake-test rules can read the named cases and what they use rather than every test in the file, and
what to point the runner at so it runs exactly those cases. A Django case written the way SWE-bench writes it,
'test_x (app.tests.Class)', is the label 'app.tests.Class.test_x'. A case none of the repro's files defines cannot be
checked, and the skeptic refuses the repro.

When Repro.cases is empty, the repro runs every case its test files define, named the way each runner reports them
(expected_cases, the default the integrator may replace):

- pytest: node ids, '<path>::<function>' or '<path>::<Class>::<method>', for test functions, methods of classes named
  Test*, and test methods of unittest TestCase subclasses;
- django: labels relative to the repository's tests/ folder, '<module>.<Class>.<method>', for TestCase subclasses
  (Django's runner collects only unittest classes);
- sympy: the bare function name of each module-level function named test_*, the only tests bin/test collects, once
  each: bin/test reports 'test_issue_1 ok', never a path (contracts.CaseResult.name is the runner's own id).
"""

from __future__ import annotations

import ast
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import PurePosixPath
from typing import Final

from receipts.contracts import Repro, RunnerKind

CaseNamer = Callable[[Repro, RunnerKind], tuple[str, ...]]
Function = ast.FunctionDef | ast.AsyncFunctionDef

_TEST_CASE_BASES: Final = ("TestCase",)  # unittest.TestCase and Django's SimpleTestCase, TestCase, ...
# 'test_x (app.tests.Class)', how SWE-bench and unittest's progress line describe a Django test.
_DJANGO_DESCRIPTION = re.compile(r"(?P<method>[A-Za-z_]\w*) \((?P<qual>[A-Za-z_][\w.]*)\)")


@dataclass(frozen=True)
class CaseRef:
    """Where one case lives in its test file: its classes, outermost first (none for a module-level function), and
    the name of the function it runs."""

    classes: tuple[str, ...]
    function: str


@dataclass(frozen=True)
class Placement:
    """Repro.cases read against the repro's own files.

    `refs` holds, for each file, the named cases it defines; `whole` the files holding a named case the rules cannot
    pin to its function (the rules then read the whole file); `unplaced` the case ids no file of the repro defines;
    `tests` what the runner is pointed at so it runs exactly the named cases; `expect` the cases each run must report,
    in the runner's own ids."""

    refs: Mapping[str, tuple[CaseRef, ...]]
    whole: frozenset[str]
    unplaced: tuple[str, ...]
    tests: tuple[str, ...]
    expect: tuple[str, ...]


def repro_test_files(repro: Repro) -> tuple[str, ...]:
    """The repro's Python test files, the ones a runner is pointed at: test_*.py, *_test.py and tests.py."""
    return tuple(path for path in repro.files if is_test_module(path))


def is_test_module(path: str) -> bool:
    name = PurePosixPath(path).name
    return name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py") or name == "tests.py")


def expected_cases(repro: Repro, runner: RunnerKind) -> tuple[str, ...]:
    """Every case the repro's test files define, named the way `runner` reports it. A file that does not parse
    defines none (the skeptic's static gate refuses it first)."""
    names: list[str] = []
    for path in repro_test_files(repro):
        module = _parse(repro.files[path], path)
        if module is None:
            continue
        functions, methods = _tests_in(module)
        if runner is RunnerKind.PYTEST:
            names += [f"{path}::{name}" for name in functions]
            names += [f"{path}::{cls}::{name}" for cls, name, _ in methods]
        elif runner is RunnerKind.DJANGO:
            label = _django_module(path)
            names += [f"{label}.{cls}.{name}" for cls, name, unittest_style in methods if unittest_style]
        else:  # sympy: bin/test runs module-level test_* functions and reports each by its bare name
            names += [name for name in functions if name.startswith("test_")]
    return tuple(dict.fromkeys(names))


def defined_cases(repro: Repro, runner: RunnerKind) -> dict[str, tuple[CaseRef, ...]]:
    """For each test file of the repro, every case `runner` collects from it, as places in the file: the cases a
    repro with no Repro.cases runs."""
    found: dict[str, tuple[CaseRef, ...]] = {}
    for path in repro_test_files(repro):
        module = _parse(repro.files[path], path)
        if module is None:
            continue
        functions, methods = _tests_in(module)
        refs: list[CaseRef] = []
        if runner is RunnerKind.PYTEST:
            refs += [CaseRef((), name) for name in functions]
            refs += [CaseRef((cls,), name) for cls, name, _ in methods]
        elif runner is RunnerKind.DJANGO:
            refs += [CaseRef((cls,), name) for cls, name, unittest_style in methods if unittest_style]
        else:
            refs += [CaseRef((), name) for name in functions if name.startswith("test_")]
        found[path] = tuple(dict.fromkeys(refs))
    return found


def django_label(case: str) -> str:
    """A Django case id as tests/runtests.py's own label: 'test_x (app.tests.Class)' is 'app.tests.Class.test_x'
    (Python 3.11 and later print 'test_x (app.tests.Class.test_x)'); any other id is kept as it is."""
    m = _DJANGO_DESCRIPTION.fullmatch(case.strip())
    if m is None:
        return case
    method, qual = m["method"], m["qual"]
    return qual if qual.endswith("." + method) else f"{qual}.{method}"


def place_cases(repro: Repro, runner: RunnerKind) -> Placement:
    """Where each of Repro.cases lives in the repro's files, and how to run exactly those cases: a pytest node id and
    a Django label are what the runner takes, and a bare sympy name runs as 'path::name' in each repro file that
    defines it."""
    modules = {path: module for path in repro.files if (module := _parse(repro.files[path], path)) is not None}
    refs: dict[str, list[CaseRef]] = {}
    whole: set[str] = set()
    unplaced: list[str] = []
    tests: list[str] = []
    expect: list[str] = []
    for case in repro.cases:
        if runner is RunnerKind.DJANGO:
            label = django_label(case)
            found = _place_django(label, repro, modules)
            wanted = label
        elif runner is RunnerKind.SYMPY:
            found, wanted = _place_sympy(case, repro, modules), case
        else:
            found, wanted = _place_pytest(case, repro, modules), case
        if not found:
            unplaced.append(case)
            continue
        expect.append(wanted)
        for path, ref, test in found:
            if ref is None:
                whole.add(path)
            elif ref not in refs.setdefault(path, []):
                refs[path].append(ref)
            tests.append(test)
    return Placement(
        refs={path: tuple(found) for path, found in refs.items()},
        whole=frozenset(whole),
        unplaced=tuple(unplaced),
        tests=tuple(dict.fromkeys(tests)),
        expect=tuple(dict.fromkeys(expect)),
    )


def locate(module: ast.Module, ref: CaseRef) -> tuple[tuple[ast.ClassDef, ...], Function] | None:
    """The classes (outermost first) and the function a case runs, or None when the file does not define it. A method
    a class inherits from another class of the same file is found there. A name defined twice is its last
    definition, as Python binds it."""
    body: list[ast.stmt] = module.body
    chain: list[ast.ClassDef] = []
    for name in ref.classes:
        cls = _last_class(body, name)
        if cls is None:
            return None
        chain.append(cls)
        body = cls.body
    if not chain:
        function = _last_function(module.body, ref.function)
        return ((), function) if function is not None else None
    method = _method(module, chain[-1], ref.function, set())
    return (tuple(chain), method) if method is not None else None


# ---------------------------------------------------------------- placing each runner's ids


Found = list[tuple[str, CaseRef | None, str]]  # (file, the case's place or None for the whole file, what to run)


def _place_pytest(case: str, repro: Repro, modules: Mapping[str, ast.Module]) -> Found:
    """A node id names its file: '<path>::<function>', '<path>::<Class>::<method>', perhaps with a
    parametrization ('[1-2]') that runs the same function."""
    path, sep, rest = case.partition("::")
    if not sep or path not in repro.files:
        return []
    parts = rest.split("::")
    module = modules.get(path)
    if module is None or not all(parts):
        return [(path, None, case)]
    ref = CaseRef(tuple(parts[:-1]), parts[-1].split("[", 1)[0])
    if locate(module, ref) is not None:
        return [(path, ref, case)]
    whole_class = _class_tests(module, (*ref.classes, ref.function))  # a node id that names a class runs its tests
    if whole_class:
        return [(path, test, case) for test in whole_class]
    return [(path, None, case)]


def _place_django(label: str, repro: Repro, modules: Mapping[str, ast.Module]) -> Found:
    """A label is '<module>.<Class>.<method>', its module relative to tests/; it lives in the repro file whose module
    is the longest one the label starts with."""
    best: tuple[str, str] | None = None
    for path in repro.files:
        if not path.endswith(".py"):
            continue
        prefix = _django_package(path)
        if (label == prefix or label.startswith(prefix + ".")) and (best is None or len(prefix) > len(best[1])):
            best = (path, prefix)
    if best is None:
        return []
    path, prefix = best
    module = modules.get(path)
    rest = label[len(prefix) + 1 :].split(".") if label != prefix else []
    if module is None:
        return [(path, None, label)]
    if not rest:  # the module itself: every test in it
        _, methods = _tests_in(module)
        everything = [CaseRef((cls,), name) for cls, name, unittest_style in methods if unittest_style]
        return [(path, ref, label) for ref in everything] or [(path, None, label)]
    ref = CaseRef(tuple(rest[:-1]), rest[-1])
    if ref.classes and locate(module, ref) is not None:
        return [(path, ref, label)]
    whole_class = _class_tests(module, tuple(rest))
    if whole_class:
        return [(path, test, label) for test in whole_class]
    return [(path, None, label)]


def _place_sympy(case: str, repro: Repro, modules: Mapping[str, ast.Module]) -> Found:
    """'path::name' names its file; a bare name is every repro test file's module-level function of that name, run
    as 'path::name' (bin/test takes file paths and -k names, never a bare name)."""
    path, sep, name = case.partition("::")
    if sep:
        if path not in repro.files:
            return []
        module = modules.get(path)
        ref = CaseRef((), name)
        if module is not None and locate(module, ref) is not None:
            return [(path, ref, case)]
        return [(path, None, case)]
    ref = CaseRef((), case)
    return [
        (path, ref, f"{path}::{case}")
        for path in repro_test_files(repro)
        if (module := modules.get(path)) is not None and locate(module, ref) is not None
    ]


# ---------------------------------------------------------------- the syntax tree


def _parse(source: str, path: str) -> ast.Module | None:
    if not path.endswith(".py"):
        return None
    try:
        return ast.parse(source, filename=path)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def _tests_in(module: ast.Module) -> tuple[list[str], list[tuple[str, str, bool]]]:
    """Module-level test functions, and (class, method, unittest-style) for test methods of test classes."""
    functions: list[str] = []
    methods: list[tuple[str, str, bool]] = []
    for node in module.body:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name.startswith("test"):
            functions.append(node.name)
        elif isinstance(node, ast.ClassDef):
            unittest_style = any(_base_name(base).endswith(_TEST_CASE_BASES) for base in node.bases)
            pytest_style = node.name.startswith("Test") and not _has_init(node)
            if not (unittest_style or pytest_style):
                continue
            for item in node.body:
                if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name.startswith("test"):
                    methods.append((node.name, item.name, unittest_style))
    return functions, methods


def _class_tests(module: ast.Module, classes: tuple[str, ...]) -> list[CaseRef]:
    """Every test method a class of the file defines in its own body, as cases."""
    body: list[ast.stmt] = module.body
    cls: ast.ClassDef | None = None
    for name in classes:
        cls = _last_class(body, name)
        if cls is None:
            return []
        body = cls.body
    if cls is None:
        return []
    return [
        CaseRef(classes, item.name)
        for item in cls.body
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name.startswith("test")
    ]


def _method(module: ast.Module, cls: ast.ClassDef, name: str, seen: set[int]) -> Function | None:
    if id(cls) in seen:
        return None
    seen.add(id(cls))
    own = _last_function(cls.body, name)
    if own is not None:
        return own
    for base in cls.bases:
        parent = _last_class(module.body, base.id) if isinstance(base, ast.Name) else None
        if parent is not None:
            inherited = _method(module, parent, name, seen)
            if inherited is not None:
                return inherited
    return None


def _last_class(body: list[ast.stmt], name: str) -> ast.ClassDef | None:
    found = [node for node in body if isinstance(node, ast.ClassDef) and node.name == name]
    return found[-1] if found else None


def _last_function(body: list[ast.stmt], name: str) -> Function | None:
    found = [node for node in body if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef) and node.name == name]
    return found[-1] if found else None


def _base_name(base: ast.expr) -> str:
    if isinstance(base, ast.Name):
        return base.id
    if isinstance(base, ast.Attribute):
        return base.attr
    return ""


def _has_init(node: ast.ClassDef) -> bool:
    return any(isinstance(item, ast.FunctionDef) and item.name == "__init__" for item in node.body)


def _django_module(path: str) -> str:
    parts = list(PurePosixPath(path).with_suffix("").parts)
    if parts and parts[0] == "tests":  # tests/runtests.py labels are relative to the tests folder
        parts = parts[1:]
    return ".".join(parts)


def _django_package(path: str) -> str:
    """A file's module as a Django label: a package's __init__.py is the package."""
    module = _django_module(path)
    return module.removesuffix(".__init__") if module.endswith(".__init__") else module
