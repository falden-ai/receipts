"""SK1 (b), the fake-test rules: static checks over a repro's Python files, run before anything executes. A repro that
breaks one is refuted; it never reaches a sandbox. The rules follow the fake-test assessor of Becker et al. 2026
(section 6), each as a plain rule on the syntax tree:

- reflection: the test reads or rewrites code instead of calling it. Such a test fails on the patch and passes on
  base by construction, whatever the code does. It is:
  - source and frame introspection (inspect.getsource, sys._getframe, __code__, __globals__ and the like),
    importlib.reload, or a module swapped in sys.modules;
  - any use of a module's or an object's __file__, __spec__, __loader__ or __cached__, as an attribute
    (render.__file__) or named in a string (getattr(render, "__file__")). The repro's own bare __file__ is allowed,
    so a test can still find data files next to itself;
  - any call into linecache, which reads a module's source lines, and locating a changed module's file through
    importlib.util.find_spec or pkgutil.get_loader (its origin and loader are where its code lives);
  - opening, reading or writing a file the patch changed: its path given to open, io.open, builtins.open,
    codecs.open, tokenize.open and the like, or a pathlib path of it read with read_text, read_bytes or open, or the
    file read as a package resource (pkgutil.get_data, importlib.resources). A path is read from the repro alone:
    string constants and f-strings, '/', os.path.join and os.path.dirname, pathlib.Path with its parent, parents and
    joinpath, importlib.resources.files and a package's __path__ (a package of the patch's own modules is in the
    folder its changed files show), simple assignments, and the repro's own __file__, which is its path in the
    repository. A relative path is read from the repository root, the run's working directory.
  Running code with eval or exec is not reflection: an eval(repr(x)) round trip is an ordinary behavioral test;
- call_count: the test asserts how many times something was called (call_count, assert_called_once,
  assert_not_called, mock_calls and the like), which pins the implementation, not the behavior;
- mocks_changed: the test replaces a function or class the patch changed (a mock.patch, patch.object, patch.multiple
  or patch.dict target under any import name, mocker.patch, monkeypatch.setattr or setitem, setattr, an imported
  module's attribute assigned, or a module's namespace dict written), or writes over a file the patch changed, so it
  tests a stand-in instead of the change. A target named by import path ('pkg.mod.name', or an attribute of an
  imported module or class) counts only if the patch changed that name in that module; a target the rule cannot
  trace to a module counts if the patch changed a function or class of that name, a dunder name excepted;
- infra: the test depends on the environment, not the code: the network (which the run phase blocks), another
  process, or a file outside the repository, so a failure says nothing about the change. The repository's own
  packages are never the network: a repro for psf/requests imports requests;
- unparseable: a Python file that does not parse cannot be a runnable repro.

Imports are resolved, so 'from inspect import getsource as gs' is caught as inspect.getsource, and 'from
unittest.mock import patch as p' as unittest.mock.patch.

What the rules read (Scope). By default, every statement of every file. A repro that names its cases (Repro.cases)
is checked on what runs for those cases, never on the other tests beside them: its module-level code, the named
cases themselves, and what they use: their class's body apart from its other tests (setUp, setUpClass, tearDown,
tearDownClass, setUpTestData and every other helper method, the class's decorators and bases, and the same for its
bases in the file), the fixtures they request, autouse fixtures, module hooks (setup_module, setUpModule and the
like) and pytestmark, and every function, class, import or assignment of the file a name they read is bound to,
followed to what those use in turn. A test file that is the repository's own, the base tree's content at its own
path (Scope.own, which the skeptic decides), is not something the repro wrote: writing it back is no violation even
where the patch changed it, and its module-level code counts only where the named cases use it.
"""

from __future__ import annotations

import ast
import posixpath
from collections import deque
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from receipts.core.cases import CaseRef, is_test_module, locate
from receipts.core.diff import PatchFacts, module_name


class FakeRule(StrEnum):
    REFLECTION = "reflection"
    CALL_COUNT = "call_count"
    MOCKS_CHANGED = "mocks_changed"
    INFRA = "infra"
    UNPARSEABLE = "unparseable"


@dataclass(frozen=True)
class Violation:
    rule: FakeRule
    path: str
    line: int
    detail: str

    def __str__(self) -> str:
        return f"{self.rule.value} at {self.path}:{self.line}: {self.detail}"


@dataclass(frozen=True)
class Scope:
    """What the rules read in one repro file. `cases` None reads every statement; a tuple reads the named cases and
    what they use (the module docstring), with the file's module-level code unless the file is `own`: the
    repository's own test, the base tree's content at its own path, which may also be written back over a file the
    patch changed."""

    cases: tuple[CaseRef, ...] | None = None
    own: bool = False


WHOLE_FILE: Final = Scope()


REFLECTION_CALLS: Final = frozenset(
    {
        "inspect.getsource",
        "inspect.getsourcelines",
        "inspect.getsourcefile",
        "inspect.getfile",
        "inspect.getclosurevars",
        "inspect.currentframe",
        "inspect.stack",
        "sys._getframe",
        "dis.dis",
        "dis.get_instructions",
        "dis.Bytecode",
        "importlib.reload",
    }
)
REFLECTION_ATTRIBUTES: Final = frozenset(
    {
        "__code__",
        "__closure__",
        "__globals__",
        "__defaults__",
        "__kwdefaults__",
        "__wrapped__",
        "co_code",
        "co_consts",
        "co_names",
        "co_varnames",
        "f_back",
        "f_code",
        "f_globals",
        "f_locals",
        "gi_frame",
        "cr_frame",
    }
)
# Where a module's code lives on disk: a test that touches one is looking at the code, not running it.
MODULE_FILE_ATTRIBUTES: Final = frozenset({"__file__", "__spec__", "__loader__", "__cached__"})
# Attribute names that are reflection when a string names them too (getattr(render, "__file__")).
REFLECTION_NAMES: Final = MODULE_FILE_ATTRIBUTES | frozenset(a for a in REFLECTION_ATTRIBUTES if a.startswith("__"))
# Calls that open, read, write or copy the file a path names: given a file the patch changed, they are reflection.
SOURCE_OPENERS: Final = frozenset(
    {
        "open",
        "io.open",
        "io.open_code",
        "io.FileIO",
        "builtins.open",
        "codecs.open",
        "tokenize.open",
        "os.open",
        "fileinput.input",
        "fileinput.FileInput",
        "shutil.copy",
        "shutil.copy2",
        "shutil.copyfile",
        "shutil.move",
    }
)
# pathlib methods that read or write the file their path names.
PATH_IO_METHODS: Final = frozenset({"read_text", "read_bytes", "open", "write_text", "write_bytes"})
# Calls that read a package's resource: (package, resource name) as their first two arguments.
PACKAGE_READERS: Final = frozenset(
    {
        "pkgutil.get_data",
        "importlib.resources.read_text",
        "importlib.resources.read_binary",
        "importlib.resources.open_text",
        "importlib.resources.open_binary",
        "importlib.resources.path",
        "importlib_resources.read_text",
        "importlib_resources.read_binary",
        "importlib_resources.open_text",
        "importlib_resources.open_binary",
        "importlib_resources.path",
    }
)
PACKAGE_FILES: Final = frozenset({"importlib.resources.files", "importlib_resources.files"})
# Calls that locate a module's file by its dotted name.
MODULE_FINDERS: Final = frozenset(
    {
        "importlib.util.find_spec",
        "importlib.machinery.PathFinder.find_spec",
        "importlib.find_loader",
        "pkgutil.get_loader",
        "pkgutil.find_loader",
    }
)
PATH_TYPES: Final = frozenset({"pathlib.Path", "pathlib.PurePath", "pathlib.PosixPath", "pathlib.PurePosixPath"})
# Calls that hand back the path they are given, as far as which file it names goes.
PATH_KEEPERS: Final = frozenset(
    {"os.path.abspath", "os.path.realpath", "os.path.normpath", "os.fspath", "os.fsdecode", "str"}
)
PATH_METHODS_KEEPING: Final = frozenset({"resolve", "absolute"})
CWD_CALLS: Final = frozenset({"os.getcwd", "pathlib.Path.cwd", "pathlib.PosixPath.cwd"})
CALL_COUNT_ATTRIBUTES: Final = frozenset(
    {"call_count", "await_count", "call_args_list", "await_args_list", "mock_calls", "method_calls"}
)
CALL_COUNT_ASSERTS: Final = frozenset(
    {
        "assert_called",
        "assert_called_once",
        "assert_called_once_with",
        "assert_not_called",
        "assert_has_calls",
        "assert_awaited",
        "assert_awaited_once",
        "assert_awaited_once_with",
        "assert_not_awaited",
        "assert_has_awaits",
    }
)
NETWORK_MODULES: Final = (
    "socket",
    "urllib.request",
    "http.client",
    "http.server",
    "requests",
    "httpx",
    "aiohttp",
    "urllib3",
    "ftplib",
    "smtplib",
    "poplib",
    "imaplib",
    "telnetlib",
    "xmlrpc.client",
    "websocket",
    "websockets",
    "paramiko",
)
PROCESS_MODULES: Final = ("subprocess", "pty", "pexpect")
PROCESS_CALLS: Final = frozenset(
    {
        "os.system",
        "os.popen",
        "os.fork",
        "os.forkpty",
        "os.posix_spawn",
        "os.posix_spawnp",
        "asyncio.create_subprocess_exec",
        "asyncio.create_subprocess_shell",
    }
)
PROCESS_PREFIXES: Final = ("os.exec", "os.spawn")
FILE_CALLS: Final = frozenset(
    {
        "open",
        "io.open",
        "builtins.open",
        "os.open",
        "os.listdir",
        "os.scandir",
        "os.stat",
        "os.walk",
        "os.remove",
        "os.unlink",
        "os.path.exists",
        "os.path.isfile",
        "os.path.isdir",
        "os.path.getsize",
        "os.path.expanduser",
        "shutil.copy",
        "shutil.copyfile",
        "shutil.copytree",
        "shutil.move",
        "pathlib.Path",
        "glob.glob",
    }
)
ALLOWED_ABSOLUTE: Final = ("/tmp", "/dev/null")  # the run phase's own writable /tmp, and the null device
# Module-level functions a runner calls around every test of the module (pytest's and nose's, and unittest's).
MODULE_HOOKS: Final = frozenset(
    {
        "setup_module",
        "teardown_module",
        "setup_function",
        "teardown_function",
        "setup",
        "teardown",
        "setUpModule",
        "tearDownModule",
        "load_tests",
    }
)
_MAX_DEPTH: Final = 12  # how deep an expression a path is read from
_MAX_PATH_CHARS: Final = 4_096  # PATH_MAX: anything longer names no file, and '+' chains cannot grow without bound


def check_repro_files(
    files: Mapping[str, str], facts: PatchFacts, scopes: Mapping[str, Scope] | None = None
) -> tuple[Violation, ...]:
    """Every fake-test rule a repro's files break, in file order, each file read as far as its Scope says (every
    statement unless `scopes` names it). Only Python files are read; a repro file at a path the patch changed breaks
    mocks_changed whatever its language, unless it is the repository's own test written back at its own path."""
    found: list[Violation] = []
    for path, source in files.items():
        scope = (scopes or {}).get(path, WHOLE_FILE)
        if path in facts.files and not (scope.own and is_test_module(path)):
            found.append(Violation(FakeRule.MOCKS_CHANGED, path, 1, "writes over a file the patch changed"))
        if not path.endswith(".py"):
            continue
        try:
            module = ast.parse(source, filename=path)
        except (SyntaxError, ValueError, RecursionError, MemoryError) as exc:  # model output: anything can come back
            line = exc.lineno if isinstance(exc, SyntaxError) and exc.lineno else 1
            found.append(Violation(FakeRule.UNPARSEABLE, path, line, f"does not parse: {type(exc).__name__}"))
            continue
        try:
            found.extend(_Checker(path, module, facts, scope).violations())
        except RecursionError:  # a tree that parses but is too deep to walk is no runnable repro either
            found.append(Violation(FakeRule.UNPARSEABLE, path, 1, "nested too deeply to check"))
    return tuple(found)


class _Checker:
    def __init__(self, path: str, module: ast.Module, facts: PatchFacts, scope: Scope = WHOLE_FILE) -> None:
        self.path = path
        self.module = module
        self.facts = facts
        self.names = facts.names
        self.aliases = _aliases(module)
        self.read = _read_nodes(module, scope)  # every node the rules read, each once
        self.assigned = _assignments(self.read)
        self._names: dict[str, str | None] = {}  # the path each name holds, worked out once

    def violations(self) -> Iterator[Violation]:
        for node in self.read:
            if isinstance(node, ast.Import | ast.ImportFrom):
                yield from self._imports(node)
            elif isinstance(node, ast.Call):
                yield from self._call(node)
            elif isinstance(node, ast.Attribute):
                yield from self._attribute(node)
            elif isinstance(node, ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete):
                yield from self._assignment(node)
            elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value in REFLECTION_NAMES:
                yield self._hit(FakeRule.REFLECTION, node, f"names {node.value}")

    def _hit(self, rule: FakeRule, node: ast.AST, detail: str) -> Violation:
        return Violation(rule, self.path, getattr(node, "lineno", 1), detail)

    def _imports(self, node: ast.Import | ast.ImportFrom) -> Iterator[Violation]:
        for name in _imported_names(node):
            if name.split(".", 1)[0] in self.facts.packages:
                continue  # the repository's own code
            if _within(name, NETWORK_MODULES):
                yield self._hit(FakeRule.INFRA, node, f"imports {name}: the run phase has no network")
            elif _within(name, PROCESS_MODULES):
                yield self._hit(FakeRule.INFRA, node, f"imports {name}: the test runs another process")

    def _call(self, node: ast.Call) -> Iterator[Violation]:
        name = self._dotted(node.func)
        if name in REFLECTION_CALLS:
            yield self._hit(FakeRule.REFLECTION, node, f"calls {name}")
        if name is not None and name.split(".", 1)[0] == "linecache":
            yield self._hit(FakeRule.REFLECTION, node, f"calls {name}: linecache reads a module's source lines")
        if name in MODULE_FINDERS and node.args:
            located = _string(node.args[0])
            if located is not None and self._changed_module(located):
                yield self._hit(FakeRule.REFLECTION, node, f"calls {name} on {located}, a module the patch changed")
        if name in SOURCE_OPENERS:
            changed = next(filter(None, (self._changed_file(arg) for arg in _path_arguments(node))), None)
            if changed is not None:
                yield self._hit(FakeRule.REFLECTION, node, f"{name}() of {changed}, a file the patch changed")
        if isinstance(node.func, ast.Attribute) and node.func.attr in PATH_IO_METHODS:
            changed = self._changed_file(node.func.value)
            if changed is not None:
                detail = f"{node.func.attr}() of {changed}, a file the patch changed"
                yield self._hit(FakeRule.REFLECTION, node, detail)
        if name in PACKAGE_READERS and len(node.args) >= 2:
            folder, resource = self._package_dir(node.args[0]), self._path(node.args[1], 1)
            changed = (
                _changed_path(posixpath.join(folder, resource), self.facts.files)
                if folder is not None and resource is not None
                else None
            )
            if changed is not None:
                yield self._hit(FakeRule.REFLECTION, node, f"{name}() of {changed}, a file the patch changed")
        if name in PROCESS_CALLS or (name and name.startswith(PROCESS_PREFIXES)):
            yield self._hit(FakeRule.INFRA, node, f"calls {name}: the test runs another process")
        if name in FILE_CALLS and node.args:
            outside = _outside_path(node.args[0])
            if outside is not None:
                yield self._hit(FakeRule.INFRA, node, f"reads {outside!r}, outside the repository")
        if isinstance(node.func, ast.Attribute) and node.func.attr in CALL_COUNT_ASSERTS:
            yield self._hit(FakeRule.CALL_COUNT, node, f"asserts a call count with {node.func.attr}")
        for where, attribute in self._patched(node, name):
            if self._changed(where, attribute):
                yield self._hit(FakeRule.MOCKS_CHANGED, node, f"replaces {attribute}, which the patch changed")

    def _attribute(self, node: ast.Attribute) -> Iterator[Violation]:
        if node.attr in REFLECTION_ATTRIBUTES:
            yield self._hit(FakeRule.REFLECTION, node, f"reads {node.attr}")
        if node.attr in MODULE_FILE_ATTRIBUTES:
            yield self._hit(FakeRule.REFLECTION, node, f"reads {node.attr}: where a module's code lives on disk")
        if node.attr in CALL_COUNT_ATTRIBUTES:
            yield self._hit(FakeRule.CALL_COUNT, node, f"reads {node.attr}")

    def _assignment(self, node: ast.Assign | ast.AugAssign | ast.AnnAssign | ast.Delete) -> Iterator[Violation]:
        targets = node.targets if isinstance(node, ast.Assign | ast.Delete) else [node.target]
        for target in targets:
            if isinstance(target, ast.Subscript) and self._dotted(target.value) == "sys.modules":
                yield self._hit(FakeRule.REFLECTION, node, "swaps a module in sys.modules")
            elif isinstance(target, ast.Subscript):
                namespace, owner = self._namespace(target.value)
                key = _string(target.slice)
                if namespace and key is not None and self._changed(owner, key):
                    detail = f"assigns over {key} in a module's namespace, which the patch changed"
                    yield self._hit(FakeRule.MOCKS_CHANGED, node, detail)
            elif isinstance(target, ast.Attribute):
                owner = self._imported(target.value)
                if owner is not None and self._changed(owner, target.attr):
                    yield self._hit(
                        FakeRule.MOCKS_CHANGED, node, f"assigns over {target.attr}, which the patch changed"
                    )

    def _patched(self, node: ast.Call, name: str | None) -> Iterator[tuple[str | None, str]]:
        """(where, attribute) for each attribute a patch-like call replaces; `where` is the dotted module or class it
        lives in when the call names one, else None. The call is known by its resolved name, so an import alias
        ('patch as p', 'unittest.mock as m') is the call it stands for."""
        last = name.rsplit(".", 1)[-1] if name is not None else ""
        if last == "patch":  # mock.patch, mocker.patch, under any import name
            target = _string(_argument(node, 0, "target"))
            if target and "." in target:
                module, attribute = target.rsplit(".", 1)
                yield module, attribute
        elif last == "object" and name is not None and name.endswith("patch.object"):
            named = _string(_argument(node, 1, "attribute"))
            if named:
                yield self._owner(_argument(node, 0, "target")), named
        elif last == "multiple" and name is not None and name.endswith("patch.multiple"):
            owner = _argument(node, 0, "target")
            where = _string(owner) or self._owner(owner)
            for keyword in node.keywords:
                if keyword.arg:
                    yield where, keyword.arg
        elif last == "dict" and name is not None and name.endswith("patch.dict"):
            namespace, held_by = self._namespace(_argument(node, 0, "in_dict"))
            if namespace:
                values = _argument(node, 1, "values")
                keys = [_string(key) for key in values.keys] if isinstance(values, ast.Dict) else []
                entries = [k.arg for k in node.keywords if k.arg not in ("in_dict", "values", "clear")]
                for named in (*keys, *entries):
                    if named:
                        yield held_by, named
        elif last in ("setattr", "delattr"):
            func = node.func
            first = _string(_argument(node, 0, "target"))
            if first is not None and "." in first and not isinstance(func, ast.Name):
                module, attribute = first.rsplit(".", 1)  # monkeypatch.setattr("pkg.mod.name", value)
                yield module, attribute
            else:
                named = _string(_argument(node, 1, "name"))
                if named:
                    yield self._owner(_argument(node, 0, "target")), named
        elif last in ("setitem", "delitem"):  # monkeypatch.setitem(module.__dict__, "name", value)
            namespace, held_by = self._namespace(_argument(node, 0, "dic"))
            named = _string(_argument(node, 1, "name"))
            if namespace and named:
                yield held_by, named

    def _changed(self, where: str | None, attribute: str) -> bool:
        """Whether replacing `attribute` of `where` replaces something the patch changed."""
        if where is None:  # an object the rule cannot trace to a module
            dunder = attribute.startswith("__") and attribute.endswith("__")
            return attribute in self.names and not dunder
        return any(
            (where == module or where.startswith(module + ".")) and attribute in names
            for module, names in self.facts.names_by_module.items()
        )

    def _namespace(self, node: ast.expr | None) -> tuple[bool, str | None]:
        """Whether an expression is a module's or class's namespace dict (X.__dict__ or vars(X)), and the dotted name
        of X when it can be traced."""
        if isinstance(node, ast.Attribute) and node.attr == "__dict__":
            return True, self._imported(node.value)
        if isinstance(node, ast.Call) and self._dotted(node.func) == "vars" and len(node.args) == 1:
            return True, self._imported(node.args[0])
        return False, None

    def _owner(self, node: ast.expr | None) -> str | None:
        return self._imported(node) if node is not None else None

    def _imported(self, node: ast.expr) -> str | None:
        """The dotted name an expression refers to when it starts from an import, sys.modules['pkg.mod'] or
        importlib.import_module('pkg.mod'), else None."""
        base = node
        trail: list[str] = []
        while isinstance(base, ast.Attribute):
            trail.append(base.attr)
            base = base.value
        module: str | None = None
        if isinstance(base, ast.Subscript) and self._dotted(base.value) == "sys.modules":
            module = _string(base.slice)
        elif isinstance(base, ast.Call) and self._dotted(base.func) == "importlib.import_module" and base.args:
            module = _string(base.args[0])
        if module is not None:
            return ".".join([module, *reversed(trail)])
        if not isinstance(base, ast.Name) or base.id not in self.aliases:
            return None
        return self._dotted(node)

    def _dotted(self, node: ast.expr) -> str | None:
        """The dotted name an expression refers to, through the module's imports, or None if it is not one."""
        parts: list[str] = []
        while isinstance(node, ast.Attribute):
            parts.append(node.attr)
            node = node.value
        if not isinstance(node, ast.Name):
            return None
        parts.append(self.aliases.get(node.id, node.id))
        return ".".join(reversed(parts))

    # ------------------------------------------------ paths

    def _changed_module(self, dotted: str) -> bool:
        """Whether a dotted name is a module the patch changed, or a package one of them is in."""
        return any(module == dotted or module.startswith(dotted + ".") for module in self.facts.names_by_module)

    def _package_dir(self, node: ast.expr) -> str | None:
        """The folder of the package an expression names, by its dotted name as a string or as an imported module."""
        package = _string(node) or self._imported(node)
        return _package_folder(package, self.facts.files) if package else None

    def _changed_file(self, node: ast.expr | None) -> str | None:
        """The changed file an expression names as a path, or None."""
        text = self._path(node, 0)
        return _changed_path(text, self.facts.files) if text is not None else None

    def _path(self, node: ast.expr | None, depth: int) -> str | None:
        """The path an expression names, as far as the repro alone fixes it, or None."""
        path = self._path_of(node, depth)
        return path if path is None or len(path) <= _MAX_PATH_CHARS else None

    def _path_of(self, node: ast.expr | None, depth: int) -> str | None:
        if node is None or depth > _MAX_DEPTH:
            return None
        if isinstance(node, ast.Constant):
            return node.value if isinstance(node.value, str) and node.value else None
        if isinstance(node, ast.JoinedStr):
            return self._fstring(node, depth)
        if isinstance(node, ast.Name):
            return self._name_path(node.id)
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div | ast.Add):
            left, right = self._path(node.left, depth + 1), self._path(node.right, depth + 1)
            if left is None or right is None:
                return None
            return posixpath.join(left, right) if isinstance(node.op, ast.Div) else left + right
        if isinstance(node, ast.Attribute) and node.attr == "parent":
            inner = self._path(node.value, depth + 1)
            return _parent(inner) if inner is not None else None
        if isinstance(node, ast.Subscript):
            return self._parents(node, depth)
        if isinstance(node, ast.Call):
            return self._call_path(node, depth)
        return None

    def _name_path(self, name: str) -> str | None:
        """The path a plain name holds: the repro's own path for __file__, else what its assignments give. Each name is
        worked out once, with a depth budget of its own, so the answer never depends on where the name was met; a name
        met again while it is being worked out (p = p / 'x') gives None."""
        if name == "__file__":
            return self.path  # the repro's own file, at its path in the repository
        if name in self._names:
            return self._names[name]
        self._names[name] = None
        paths = [p for p in (self._path(value, 0) for value in self.assigned.get(name, ())) if p]
        # A name assigned more than once: a value naming a changed file counts, since any of them may be the one used.
        chosen = next((p for p in paths if _changed_path(p, self.facts.files)), paths[-1] if paths else None)
        self._names[name] = chosen
        return chosen

    def _fstring(self, node: ast.JoinedStr, depth: int) -> str | None:
        parts: list[str] = []
        for value in node.values:
            if isinstance(value, ast.Constant) and isinstance(value.value, str):
                parts.append(value.value)
            elif isinstance(value, ast.FormattedValue) and value.conversion == -1 and value.format_spec is None:
                inner = self._path(value.value, depth + 1)
                if inner is None:
                    return None
                parts.append(inner)
            else:
                return None
        return "".join(parts) or None

    def _parents(self, node: ast.Subscript, depth: int) -> str | None:
        """path.parents[n], the (n + 1)th folder up, or package.__path__[n], the package's folder."""
        holder = node.value
        if isinstance(holder, ast.Attribute) and holder.attr == "__path__":
            return self._package_dir(holder.value)
        if not (isinstance(holder, ast.Attribute) and holder.attr == "parents"):
            return None
        index = node.slice
        if not (isinstance(index, ast.Constant) and type(index.value) is int and 0 <= index.value < 64):
            return None
        path = self._path(holder.value, depth + 1)
        for _ in range(index.value + 1):
            if path is None:
                return None
            path = _parent(path)
        return path

    def _call_path(self, node: ast.Call, depth: int) -> str | None:
        name = self._dotted(node.func)
        args = node.args
        if any(isinstance(arg, ast.Starred) for arg in args):
            return None
        if name in PATH_TYPES or name == "os.path.join":
            parts = [self._path(arg, depth + 1) for arg in args]
            if any(part is None for part in parts):
                return None
            return posixpath.join(*[p for p in parts if p is not None]) if parts else "."
        if name in CWD_CALLS:
            return "."
        if name in PACKAGE_FILES and len(args) == 1:
            return self._package_dir(args[0])
        if name == "os.path.dirname" and len(args) == 1:
            inner = self._path(args[0], depth + 1)
            return _parent(inner) if inner is not None else None
        if name in PATH_KEEPERS and len(args) == 1:
            return self._path(args[0], depth + 1)
        if not isinstance(node.func, ast.Attribute):
            return None
        method, receiver = node.func.attr, node.func.value
        if method in PATH_METHODS_KEEPING and not args:
            return self._path(receiver, depth + 1)
        if method == "cwd" and not args:
            return "."
        if method == "joinpath":
            parts = [self._path(item, depth + 1) for item in (receiver, *args)]
            if any(part is None for part in parts):
                return None
            return posixpath.join(*[p for p in parts if p is not None])
        if method == "with_name" and len(args) == 1:
            base, leaf = self._path(receiver, depth + 1), self._path(args[0], depth + 1)
            return posixpath.join(_parent(base), leaf) if base is not None and leaf is not None else None
        if method == "with_suffix" and len(args) == 1:
            base, suffix = self._path(receiver, depth + 1), _string(args[0])
            return posixpath.splitext(base)[0] + suffix if base is not None and suffix is not None else None
        return None


def _package_folder(package: str, files: frozenset[str]) -> str | None:
    """The folder a package of the patch's own modules lives in, worked out from a changed file in it ('widgets', with
    src/widgets/render.py changed, is src/widgets); None for a package the patch has no Python file in. A plain
    module's resources are read from its own folder."""
    for path in sorted(files):
        module = module_name(path)
        if module is None:
            continue
        folder = _parent(path)
        is_init = posixpath.basename(path) == "__init__.py"
        if module == package and not is_init:
            return folder
        dotted = module if is_init else module.rpartition(".")[0]  # the package that folder is
        if dotted == package or dotted.startswith(package + "."):
            for _ in range(dotted.count(".") - package.count(".")):
                folder = _parent(folder)
            return folder
    return None


def _changed_path(text: str, files: frozenset[str]) -> str | None:
    """The changed file a path names, or None. A relative path is read from the repository root, the run's working
    directory; an absolute one names a changed file when it ends with that file's repository path (the workspace root
    differs between images)."""
    path = posixpath.normpath(text)
    for changed in sorted(files):
        if path == changed or (path.startswith("/") and path.endswith("/" + changed)):
            return changed
    return None


def _parent(path: str) -> str:
    """A path's folder: '.' (the repository root) for a top-level name, and '..' above the root."""
    norm = posixpath.normpath(path)
    if norm == "/":
        return "/"
    if norm.split("/")[-1] in (".", ".."):
        return posixpath.normpath(norm + "/..")
    return posixpath.dirname(norm) or "."


def _path_arguments(node: ast.Call) -> Iterator[ast.expr]:
    """Every argument a path could be in, list and tuple items included (fileinput.input(['a.py']))."""
    for arg in (*node.args, *(keyword.value for keyword in node.keywords)):
        if isinstance(arg, ast.List | ast.Tuple | ast.Set):
            yield from arg.elts
        else:
            yield arg


def _aliases(module: ast.Module) -> dict[str, str]:
    """Each name an import binds, to the dotted name it stands for."""
    aliases: dict[str, str] = {}
    for node in ast.walk(module):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.asname:
                    aliases[alias.asname] = alias.name
                else:
                    top = alias.name.split(".", 1)[0]
                    aliases[top] = top
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            for alias in node.names:
                aliases[alias.asname or alias.name] = f"{node.module}.{alias.name}"
    return aliases


def _assignments(nodes: Iterable[ast.AST]) -> dict[str, list[ast.expr]]:
    """Each value a plain name is bound to in the nodes the rules read (an assignment, a walrus, an annotated
    assignment, or a for loop over a literal list), for following a path through a variable."""
    found: dict[str, list[ast.expr]] = {}

    def bind(target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name):
            found.setdefault(target.id, []).append(value)

    for node in nodes:
        if isinstance(node, ast.Assign):
            for target in node.targets:
                bind(target, node.value)
        elif isinstance(node, ast.AnnAssign | ast.NamedExpr) and node.value is not None:
            bind(node.target, node.value)
        elif isinstance(node, ast.For | ast.AsyncFor) and isinstance(node.iter, ast.List | ast.Tuple | ast.Set):
            for item in node.iter.elts:
                bind(node.target, item)
    return found


# ---------------------------------------------------------------- what the rules read


def _read_nodes(module: ast.Module, scope: Scope) -> list[ast.AST]:
    """Every node the rules read in a module under `scope`, each once, in the order they are met: the whole module,
    or the named cases and what they use. A named case the file does not define reads the whole module."""
    roots = _scope_roots(module, scope)
    if roots is None:
        return list(ast.walk(module))
    seen: set[int] = set()
    nodes: list[ast.AST] = []
    for root in roots:
        for node in ast.walk(root):
            if id(node) not in seen:
                seen.add(id(node))
                nodes.append(node)
    return nodes


def _scope_roots(module: ast.Module, scope: Scope) -> list[ast.AST] | None:
    """The top nodes the rules read, or None for the whole module."""
    if scope.cases is None:
        return None
    if not scope.cases:
        return []  # nothing of the file runs: the repro names no case in it
    index = _ModuleIndex(module)
    roots: list[ast.AST] = list(index.around_every_test)
    for ref in scope.cases:
        found = locate(module, ref)
        if found is None:
            return None
        classes, function = found
        for cls in classes:
            roots.extend(_class_parts(cls))
        roots.append(function)
    if not scope.own:  # what the repro wrote at module level runs whatever it names
        roots.extend(
            stmt for stmt in module.body if not isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef)
        )
    return index.closure(roots)


class _ModuleIndex:
    """A module's top-level bindings, its fixtures, and what runs around every one of its tests."""

    def __init__(self, module: ast.Module) -> None:
        self.bindings: dict[str, list[ast.AST]] = {}  # a name, to the module-level nodes that bind it
        self.fixtures: dict[str, list[ast.AST]] = {}  # a fixture's name, to its definitions
        self.around_every_test: list[ast.AST] = []  # module hooks, autouse fixtures and pytestmark
        for stmt in module.body:
            parts = _class_parts(stmt) if isinstance(stmt, ast.ClassDef) else [stmt]
            for name in _bound_names(stmt):
                self.bindings.setdefault(name, []).extend(parts)
            if isinstance(stmt, ast.FunctionDef | ast.AsyncFunctionDef):
                fixture = self._fixture(stmt)
                if fixture is not None:
                    name, autouse = fixture
                    self.fixtures.setdefault(name, []).append(stmt)
                    if autouse:
                        self.around_every_test.append(stmt)
                if stmt.name in MODULE_HOOKS:
                    self.around_every_test.append(stmt)
            elif "pytestmark" in _bound_names(stmt):
                self.around_every_test.append(stmt)

    def closure(self, roots: Sequence[ast.AST]) -> list[ast.AST]:
        """The roots and everything they use in the module, followed until nothing new comes in."""
        ordered: list[ast.AST] = []
        seen: set[int] = set()
        queue: deque[ast.AST] = deque(roots)
        while queue:
            top = queue.popleft()
            if id(top) in seen:
                continue
            seen.add(id(top))
            ordered.append(top)
            for node in ast.walk(top):
                if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load):
                    queue.extend(self.bindings.get(node.id, ()))
                elif isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef):
                    for parameter in _parameters(node):
                        queue.extend(self.fixtures.get(parameter, ()))
                elif isinstance(node, ast.Call) and _called(node.func) == "usefixtures":
                    for argument in node.args:
                        named = _string(argument)
                        if named is not None:
                            queue.extend(self.fixtures.get(named, ()))
        return ordered

    def _fixture(self, function: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[str, bool] | None:
        """(its fixture name, whether it is autouse) for a function decorated as a pytest fixture, else None."""
        for decorator in function.decorator_list:
            call = decorator if isinstance(decorator, ast.Call) else None
            if _called(call.func if call is not None else decorator) not in ("fixture", "yield_fixture"):
                continue
            name, autouse = function.name, False
            for keyword in call.keywords if call is not None else ():
                if keyword.arg == "name" and _string(keyword.value):
                    name = _string(keyword.value) or name
                elif keyword.arg == "autouse":
                    autouse = isinstance(keyword.value, ast.Constant) and keyword.value.value is True
            return name, autouse
        return None


def _class_parts(cls: ast.ClassDef) -> list[ast.AST]:
    """What a class brings to a test it runs, apart from its other tests: its decorators, bases and keywords, and
    every statement of its body but its test methods, nested classes read the same way."""
    parts: list[ast.AST] = [*cls.decorator_list, *cls.bases, *(keyword.value for keyword in cls.keywords)]
    for item in cls.body:
        if isinstance(item, ast.FunctionDef | ast.AsyncFunctionDef) and item.name.startswith("test"):
            continue
        if isinstance(item, ast.ClassDef):
            parts.extend(_class_parts(item))
            continue
        parts.append(item)
    return parts


def _bound_names(stmt: ast.stmt) -> set[str]:
    """The names a module-level statement binds, compound statements' bodies included (a try around an import)."""
    names: set[str] = set()

    def targets(node: ast.expr) -> None:
        if isinstance(node, ast.Name):
            names.add(node.id)
        elif isinstance(node, ast.Tuple | ast.List):
            for item in node.elts:
                targets(item)
        elif isinstance(node, ast.Starred):
            targets(node.value)

    def visit(node: ast.stmt) -> None:
        if isinstance(node, ast.FunctionDef | ast.AsyncFunctionDef | ast.ClassDef):
            names.add(node.name)
            return  # what its body binds is its own
        for walrus in (n for n in ast.walk(node) if isinstance(n, ast.NamedExpr)):
            targets(walrus.target)
        if isinstance(node, ast.Import):
            names.update(alias.asname or alias.name.split(".", 1)[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom):
            names.update(alias.asname or alias.name for alias in node.names if alias.name != "*")
        elif isinstance(node, ast.Assign):
            for target in node.targets:
                targets(target)
        elif isinstance(node, ast.AnnAssign | ast.AugAssign):
            targets(node.target)
        else:
            if isinstance(node, ast.For | ast.AsyncFor):
                targets(node.target)
            elif isinstance(node, ast.With | ast.AsyncWith):
                for item in node.items:
                    if item.optional_vars is not None:
                        targets(item.optional_vars)
            for field in ("body", "orelse", "finalbody", "handlers"):
                for child in getattr(node, field, ()):
                    if isinstance(child, ast.ExceptHandler):
                        if child.name:
                            names.add(child.name)
                        for inner in child.body:
                            visit(inner)
                    elif isinstance(child, ast.stmt):
                        visit(child)

    visit(stmt)
    return names


def _parameters(function: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    arguments = function.args
    return [a.arg for a in (*arguments.posonlyargs, *arguments.args, *arguments.kwonlyargs)]


def _called(node: ast.expr) -> str:
    """The last part of a called or decorating name: 'fixture' for pytest.fixture, 'usefixtures' for
    pytest.mark.usefixtures."""
    if isinstance(node, ast.Attribute):
        return node.attr
    if isinstance(node, ast.Name):
        return node.id
    return ""


def _imported_names(node: ast.Import | ast.ImportFrom) -> Iterator[str]:
    if isinstance(node, ast.Import):
        yield from (alias.name for alias in node.names)
    elif node.module and not node.level:
        yield node.module
        yield from (f"{node.module}.{alias.name}" for alias in node.names)


def _within(name: str, modules: Sequence[str]) -> bool:
    return any(name == module or name.startswith(module + ".") for module in modules)


def _argument(node: ast.Call, position: int, keyword: str) -> ast.expr | None:
    if len(node.args) > position:
        return node.args[position]
    for item in node.keywords:
        if item.arg == keyword:
            return item.value
    return None


def _string(node: ast.expr | None) -> str | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, str) else None


def _outside_path(node: ast.expr) -> str | None:
    text = _string(node)
    if text is None or not text.startswith(("/", "~")):
        return None
    if any(text == root or text.startswith(root + "/") for root in ALLOWED_ABSOLUTE):
        return None
    return text
