"""The repositories' own test runners as ``contracts.SuiteRunner`` values: pytest, Django and sympy.

Each runner builds the command that runs some tests and reads that run's cases back. All three type failures with one
traceback reader (``receipts.attack.tracebacks``): pytest is forced to native tracebacks, and Django's unittest runner
and sympy's bin/test print native tracebacks of their own. Each runner resolves frames against the workspace root it
is bound to (``with_root``); the sandbox binds it to the workspace's root before every run.

All three treat the target's test command the same way: its leading ``NAME=value`` assignments are the run's own
environment, put in front of what the runner runs through ``env``, and a name that looks like a credential or a
proxy is refused, never passed on (``command_environment``). That is how a Tier A target carries what SWE-bench's
harness exports before it runs the tests: the locale its Django 1.11 to 3.2 specs export in their eval_commands
(LANG, LC_ALL and LANGUAGE, and PYTHONIOENCODING for 1.11 and 2.2), and the PYTHONWARNINGS its sympy specs set.

- ``PytestRunner`` runs the target's own pytest command (Tier A: SWE-bench's, a ``tox ... --`` launcher included),
  then its own options, which come after the repository's and so win: native tracebacks, strict xfail, no cache
  provider, junit in the xunit1 family (it writes each case's file, where the case's function is defined, beside its
  classname, which is the node id's path and classes; ``classify.nodeid`` rebuilds the node id from the two, and from
  the modules the report's other cases establish where the file is not the module), and the report at a path unique
  to the run. Cases come from that report. The rootdir is pytest's own choice, as in the harness's run
  (``PYTEST_OPTIONS`` says why).
- ``DjangoRunner`` runs ``tests/runtests.py --verbosity 2 --parallel 1`` with the settings named in the target's
  command (``test_sqlite`` by default) and test labels made from file paths the way the SWE-bench harness makes them
  (``harness_label``) or given as labels at any granularity (a module, a class, a method). Cases come from unittest's
  verbose progress lines and its FAIL and ERROR blocks on stderr, checked against its own summary. A class- or
  module-level hook that errored or skipped is a case of its own, so one broken hook never hides the rest of the run,
  and a test module that failed to import is a case named by the module's label.
- ``SympyRunner`` runs ``bin/test -C --verbose`` on file or directory paths, with one ``-k`` followed by the names
  of the ``path::name`` entries; a bare name, which names no file, is refused. Cases come from its verbose progress
  lines and its traceback sections on stdout, checked against its own summary. Every case names its file, since
  sympy's test names repeat across files; a test module that failed to import, or whose own code raised, is a case
  named by its file.

An expected failure that failed as expected is a SKIPPED case with ``expected_failure`` set (contract v3.1): pytest's
xfail in its junit report, unittest's expectedFailure ("expected failure") under Django, and sympy's XFAIL ("f"). An
unexpected pass is a case that passed, with a message that says so.

A run whose output does not add up (a count that disagrees with the runner's own summary, a test with no status) is
refused with ValueError, which the sandbox records as infra.
"""

from __future__ import annotations

import re
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Literal

from receipts.attack import tracebacks
from receipts.attack.classify import UNEXPECTED_PASS, case, junit_cases, outcome_for
from receipts.attack.sandbox import refused_env_names
from receipts.contracts import CaseResult, FailureType, Outcome, RunnerKind, SuiteRunner, Target

Stream = Literal["stdout", "stderr", "both", "none"]

_ASSIGNMENT = re.compile(r"[A-Za-z_][A-Za-z0-9_]*=.*")
# A Django settings module: a dotted Python module path.
_SETTINGS = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*")
# A test label runtests.py gets: dotted segments of ASCII letters, digits, underscores and hyphens, never starting with
# a hyphen. A segment may start with a digit, as the harness's own labels do (a migration module, '0001_initial'), and
# every character here is one the harness's shell passes through as it is.
_LABEL = re.compile(r"(?!-)[A-Za-z0-9_-]+(?:\.[A-Za-z0-9_-]+)*")
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
_ENV_REFUSED = re.compile(r"(?i)(?:proxy|key|secret|token|password|passwd|credential|auth)")


def _tokens(command: str) -> tuple[list[str], list[str]]:
    """A test command split into its leading NAME=value assignments and the command itself."""
    words = shlex.split(command)
    env: list[str] = []
    while words and _ASSIGNMENT.fullmatch(words[0]):
        env.append(words.pop(0))
    pairs = dict(assignment.split("=", 1) for assignment in env)
    bad = sorted({*refused_env_names(pairs), *(name for name in pairs if _ENV_REFUSED.search(name))})
    if bad:
        # Names only, never values: the same rule the sandbox applies to a run's own environment.
        raise ValueError(f"the test command sets variables that look like a credential or a proxy: {bad}")
    return env, words


def command_environment(target: Target) -> tuple[list[str], list[str]]:
    """The target's test command as its leading NAME=value assignments and the command itself, the one way every
    runner here reads it. The assignments go in front of whatever the runner runs, through ``env``, in the order the
    command gives them (the last of a repeated name wins, as in a shell); a name that looks like a credential or a
    proxy is refused with ValueError, which names the variable and never its value. An empty command has neither."""
    return _tokens(target.test_command) if target.test_command.strip() else ([], [])


def _with_env(env: Sequence[str], words: Sequence[str]) -> list[str]:
    return ["env", *env, *words] if env else list(words)


def _check_tests(tests: Sequence[str]) -> list[str]:
    """Test arguments are paths or ids, never options: an entry starting with '-' could pass a flag to the runner."""
    if not tests:
        raise ValueError("a run names at least one test")
    for test in tests:
        if not test or test.startswith("-") or "\n" in test or "\x00" in test:
            raise ValueError(f"{test!r} is not a test path or id")
    return list(tests)


class _Bound:
    """A runner bound to a workspace root, the directory its frames resolve against."""

    kind: RunnerKind
    stream: Stream

    def __init__(self, root: str = "/testbed") -> None:
        if not root.startswith("/"):
            raise ValueError("a runner's root is an absolute path in the sandbox")
        self.root = root

    def with_root(self, root: str) -> _Bound:
        bound = type(self)(root)
        return bound

    def __repr__(self) -> str:
        return f"{type(self).__name__}(root={self.root!r})"


# ---------------------------------------------------------------- pytest

# The options every pytest run gets after the repository's own command, so they win over its addopts: native
# tracebacks (the one format the traceback reader reads, with no argument reprs), an unexpected pass of an xfail test
# is a failure, no cache carries state between reruns, and junit writes each case's file (xunit1), which
# classify.nodeid reads beside the case's classname to rebuild its node id. Every one of them is an option pytest
# 3.3.1 takes (it ignores an -o for an ini key it does not know, as junit_family is there). No
# --rootdir: pytest 3.3.1, which SWE-bench's astropy 0.1 to 1.3 environments pin, has no such option and refuses the
# whole run over it (exit 2, no report; fixtures/runners/pytest3_xfail.json), so the rootdir is pytest's own choice,
# as in the harness's run: the repository root, where every SWE-bench repository keeps its setup.py or its config.
PYTEST_OPTIONS: tuple[str, ...] = (
    "-p",
    "no:cacheprovider",
    "--tb=native",
    "-o",
    "xfail_strict=true",
    "-o",
    "junit_family=xunit1",
)


class PytestRunner(_Bound):
    kind = RunnerKind.PYTEST
    stream: Stream = "none"

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        env, words = command_environment(target)
        base = words or ["python", "-m", "pytest"]
        return _with_env(env, [*base, *PYTEST_OPTIONS, f"--junitxml={report_path}", *_check_tests(tests)])

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        if report is None:
            raise ValueError("pytest wrote no report")
        return junit_cases(report, root=self.root, changed_files=changed_files)


# ---------------------------------------------------------------- Django

_DESC = r"(?P<method>[A-Za-z_]\w*) \((?P<qual>[A-Za-z_][\w.]*)\)"
_PROGRESS = re.compile(rf"^{_DESC}(?: \.\.\. (?P<rest>.*))?$")
_STATUS_WORDS = r"ok|FAIL|ERROR|expected failure|unexpected success"
_STATUS = re.compile(rf"^(?P<status>{_STATUS_WORDS}|skipped(?: .*)?)$")
# A status at the end of a line: a test that wrote to a stream without a newline puts its status after that text.
_STATUS_AT_END = re.compile(rf"(?P<status>{_STATUS_WORDS}|skipped (?:'.*'|\".*\"))$")
_BLOCK = re.compile(rf"^(?P<kind>FAIL|ERROR|UNEXPECTED SUCCESS): {_DESC}(?: \[.*\])?(?: \(.*\))?$")
_RAN = re.compile(r"^Ran (?P<n>\d+) tests? in ")
_SEP_EQ = "=" * 70
_SEP_DASH = "-" * 70
_CLASS_HOOKS = frozenset({"setUpClass", "tearDownClass", "setUpModule", "tearDownModule"})
# unittest's stand-in for a label it could not load ("tests (unittest.loader._FailedTest) ... ERROR"): its name is only
# the part of the label that failed, so two modules called tests that fail print the same line. Each one is told
# apart by its place in the run, and named by its module once its traceback says which (``_failed_module``).
_FAILED_TEST = "unittest.loader._FailedTest"
_NO_MODULE = re.compile(r"ModuleNotFoundError: No module named '(?P<module>[A-Za-z_][\w.]*)'")


def harness_label(path: str) -> str:
    """The label runtests.py gets for a file the test patch touches, made the way swebench 4.1.0 makes it
    (harness/test_spec/python.py, get_test_directives, lines 251 to 259, the django/django branch): drop a '.py'
    ending, then a leading 'tests/', then turn every '/' into '.'. Nothing else is checked there, so a segment may
    start with a digit ('tests/migrations/test_migrations_namespace_package/0001_initial.py' gives
    'migrations.test_migrations_namespace_package.0001_initial', django__django-11141), and a file that is not a .py
    keeps its ending as a segment."""
    label = path[: -len(".py")] if path.endswith(".py") else path
    label = label[len("tests/") :] if label.startswith("tests/") else label
    return label.replace("/", ".")


def django_label(test: str) -> str:
    """A Django test label: a file path the way the harness makes one (``harness_label``: 'tests/model_fields/tests.py'
    gives 'model_fields.tests'), a 'path.py::Class::method' id, or a label as given, which the harness's mapping leaves
    as it is. A label is refused when the harness's shell would split or change it, or runtests.py would read it as
    an option."""
    path, sep, rest = test.partition("::")
    if sep and path.endswith(".py"):
        label = ".".join([harness_label(path), *[part for part in rest.split("::") if part]])
    else:
        label = harness_label(test)
    if _LABEL.fullmatch(label) is None:
        raise ValueError(f"{test!r} does not make a Django test label")
    return label


def django_label_of(swebench_id: str) -> str:
    """SWE-bench names a Django test 'test_x (app.tests.Class)'; the runner's own id is 'app.tests.Class.test_x'."""
    m = re.fullmatch(_DESC, swebench_id.strip())
    if m is None:
        raise ValueError(f"{swebench_id!r} is not a Django test description")
    return _label(m.group("method"), m.group("qual"))


def _label(method: str, qual: str) -> str:
    # Python 3.11 and later print "test_x (app.tests.Class.test_x)"; earlier versions print "test_x (app.tests.Class)".
    return qual if qual.endswith("." + method) else f"{qual}.{method}"


def _settings(words: Sequence[str]) -> str:
    """The settings module a Django test command names (after its leading assignments), or runtests.py's default."""
    for i, word in enumerate(words):
        if word.startswith("--settings="):
            return word.split("=", 1)[1]
        if word == "--settings" and i + 1 < len(words):
            return words[i + 1]
    return "test_sqlite"


DJANGO_RUN: tuple[str, ...] = ("python", "tests/runtests.py", "--verbosity", "2", "--parallel", "1")


class DjangoRunner(_Bound):
    kind = RunnerKind.DJANGO
    stream: Stream = "stderr"

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        env, words = command_environment(target)
        settings = _settings(words)
        if _SETTINGS.fullmatch(settings) is None:
            raise ValueError(f"{settings!r} is not a settings module")
        labels = [django_label(test) for test in _check_tests(tests)]
        return _with_env(env, [*DJANGO_RUN, f"--settings={settings}", *labels])

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return django_cases(output, root=self.root, changed_files=changed_files)


def _exact_status(text: str) -> str | None:
    """A status a line is made of: the whole line, or what follows its last ' ... ' (after a docstring line)."""
    whole = text.strip()
    if _STATUS.match(whole):
        return whole
    rest = text.rsplit(" ... ", 1)[1].strip() if " ... " in text else None
    return rest if rest is not None and _STATUS.match(rest) else None


def _status_at_end(text: str) -> str | None:
    """A status a line ends with, after text a test wrote to the stream with no newline ('... xok')."""
    found = _STATUS_AT_END.search(text.rstrip())
    return found.group("status") if found is not None else None


def _is_failed_load(qual: str) -> bool:
    return qual == _FAILED_TEST or qual.startswith(_FAILED_TEST + ".")  # 3.11 and later add the name


def _occurrence(label: str, seen: dict[str, int]) -> str:
    """A failed load's label, made unique by its place among the run's failed loads of the same name ('#0', '#1'):
    unittest prints their progress lines and their ERROR blocks in the same order."""
    n = seen.get(label, 0)
    seen[label] = n + 1
    return f"{label}#{n}"


def _module_label(rel: str) -> str:
    """The Django label of a test module under tests/ ('tests/app/tests.py' gives 'app.tests'; a package's
    __init__.py gives the package)."""
    parts = rel[len("tests/") : -len(".py")].split("/")
    return ".".join(parts[:-1] if parts[-1] == "__init__" else parts)


def _failed_module(text: str, root: str) -> str | None:
    """Which test module a failed load stands for, from its ImportError's traceback: the outermost repository frame of
    a test module's own code (its <module>, under tests/), or, when no repository code ran at all, the module a
    ModuleNotFoundError names. None when neither says, and the case keeps unittest's own name."""
    final = tracebacks.final_block(tracebacks.parse(text))
    if final is None:
        return None
    inside = [(frame, rel) for frame in final.frames if (rel := tracebacks.repo_path(frame.path, root)) is not None]
    for frame, rel in inside:
        if frame.function == "<module>" and rel.startswith("tests/") and rel.endswith(".py"):
            return _module_label(rel) or None  # tests/__init__.py would name no module
    missing = _NO_MODULE.fullmatch(final.exception.split("\n", 1)[0].strip())
    return missing.group("module") if missing is not None and not inside else None


def _django_progress(lines: Sequence[str]) -> tuple[dict[str, str], list[str], dict[str, str]]:
    """Each test's status from unittest's verbose progress lines, read up to the first FAIL or ERROR block; every test
    that started (what 'Ran N tests' counts); and the status of each class- or module-level hook that printed a
    progress line of its own (Python 3.11 and later print 'setUpClass (app.tests.C) ... ERROR', outside that count).

    unittest prints a test's status after the test's own output. A line that is a status (or ends in one after
    ' ... ') settles it, and any status line after that belongs to something else: before Python 3.11 a class-level
    hook's error or skip prints a bare 'ERROR' or 'skipped ...' line with no description. A status at the end of other
    text ('... xok', a test that wrote to the stream with no newline) holds until the next test starts, unless a
    later line gives one, since a test's own output can end in a word like 'ok' before unittest prints the real
    status. A test whose subtest failed prints no status of its own: before Python 3.11 the next test's description
    follows on the same line, and from 3.11 the subtest's result is an indented line; either way its FAIL or ERROR
    block gives the outcome."""
    statuses: dict[str, str] = {}
    started: list[str] = []
    hooks: dict[str, str] = {}
    # Before Python 3.11 a hook's error is a bare 'ERROR' line and a hook block, never a progress line of its own.
    hook_lines = any((m := _PROGRESS.match(line)) and m.group("method") in _CLASS_HOOKS for line in lines)
    bare_hook_errors = not hook_lines and any(
        (m := _BLOCK.match(line)) and m.group("kind") == "ERROR" and m.group("method") in _CLASS_HOOKS for line in lines
    )
    pending: str | None = None
    hook = False
    last: str | None = None
    final = False
    failed_loads: dict[str, int] = {}

    def settle() -> None:
        nonlocal pending, last, final
        if pending is not None and last is not None:
            (hooks if hook else statuses)[pending] = last
        pending, last, final = None, None, False

    def note(text: str) -> None:
        nonlocal last, final
        if final:
            return
        exact = _exact_status(text)
        if exact is not None:
            if exact == "ERROR" and text.strip() == "ERROR" and last is not None and bare_hook_errors:
                return  # a hook's own line, before Python 3.11
            last, final = exact, True
            return
        last = _status_at_end(text) or last

    for i, line in enumerate(lines):
        if (line == _SEP_EQ and i + 1 < len(lines) and _BLOCK.match(lines[i + 1])) or _RAN.match(line):
            break
        if line[:1].isspace():
            continue  # a subtest's own line (Python 3.11 and later)
        segment: str | None = line
        consumed = False
        while segment is not None:
            m = _PROGRESS.match(segment)
            if m is None:
                break
            consumed = True
            settle()
            method, label = m.group("method"), _label(m.group("method"), m.group("qual"))
            if _is_failed_load(m.group("qual")):
                label = _occurrence(label, failed_loads)
            hook = method in _CLASS_HOOKS
            if not hook and label not in started:
                started.append(label)
            pending, segment = label, None
            rest = m.group("rest")
            if rest is None:
                break  # a docstring line comes next, then the status
            if not _STATUS.match(rest) and _PROGRESS.match(rest):
                pending, segment = None, rest  # no status of its own; the next test's line follows
            else:
                note(rest)
        if consumed or pending is None:
            continue
        note(line)
    settle()
    return statuses, started, hooks


def _django_blocks(lines: Sequence[str]) -> list[tuple[str, str, str]]:
    """Every FAIL and ERROR block, and each test of the UNEXPECTED SUCCESS list (Python 3.11 and later):
    (kind, label, traceback text)."""
    blocks: list[tuple[str, str, str]] = []
    failed_loads: dict[str, int] = {}
    i, n = 0, len(lines)
    while i < n:
        m = _BLOCK.match(lines[i + 1]) if lines[i] == _SEP_EQ and i + 1 < n else None
        if m is None:
            i += 1
            continue
        if m.group("kind") == "UNEXPECTED SUCCESS":
            j = i + 1
            while j < n and (u := _BLOCK.match(lines[j])) is not None and u.group("kind") == "UNEXPECTED SUCCESS":
                blocks.append((u.group("kind"), _label(u.group("method"), u.group("qual")), ""))
                j += 1
            i = j
            continue
        label = _label(m.group("method"), m.group("qual"))
        if _is_failed_load(m.group("qual")):
            label = _occurrence(label, failed_loads)
        j = i + 2
        while j < n and lines[j] != _SEP_DASH:
            j += 1  # a docstring line may follow the header
        k = j + 1
        while k < n and lines[k] not in (_SEP_EQ, _SEP_DASH):
            k += 1
        blocks.append((m.group("kind"), label, "\n".join(lines[j + 1 : k])))
        i = k
    return blocks


_SUMMARY_COUNTS = re.compile(r"(?P<name>failures|errors|skipped|expected failures|unexpected successes)=(?P<n>\d+)")


def django_cases(output: str, *, root: str, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
    """Cases from a Django runtests.py run at verbosity 2 (its stderr). An expected failure that failed as expected
    (unittest's expectedFailure, 'expected failure') is a SKIPPED case with ``expected_failure`` set, and unittest's
    own summary must count each one; an unexpected success is a case that passed, with a message that says so; a
    class- or module-level hook that errored is an ERROR case typed FIXTURE, and one that skipped is a SKIPPED case,
    both named '<class or module>.<hook>'. A test module that failed to import is an ERROR case typed IMPORT named by
    the module's label ('app.tests'), once per module, so an expected case under it is known to have never run for
    that reason (``classify.import_failure_of``)."""
    changed = tracebacks.normalize(changed_files)
    lines = _ANSI.sub("", output).replace("\r\n", "\n").split("\n")
    statuses, started, hooks = _django_progress(lines)
    blocks = _django_blocks(lines)
    results: dict[str, CaseResult] = {}
    for label, status in statuses.items():
        if status == "ok":
            results[label] = case(label, Outcome.PASS)
        elif status.startswith("skipped"):
            results[label] = case(label, Outcome.SKIPPED, message=status)
        elif status == "expected failure":
            results[label] = case(label, Outcome.SKIPPED, message="expected failure", expected_failure=True)
        elif status == "unexpected success":
            results[label] = case(label, Outcome.PASS, message=UNEXPECTED_PASS)
    for label, status in hooks.items():
        if status.startswith("skipped"):
            results[label] = case(label, Outcome.SKIPPED, message=status)
    for kind, label, text in blocks:
        parsed = tracebacks.parse(text)
        final = tracebacks.final_block(parsed)
        message = final.exception if final is not None else ""
        if kind == "FAIL":
            results[label] = case(label, Outcome.FAIL, FailureType.ASSERTION, None, message)
            continue
        if kind == "UNEXPECTED SUCCESS":
            results[label] = case(label, Outcome.PASS, message=UNEXPECTED_PASS)
            continue
        if label.rsplit(".", 1)[-1] in _CLASS_HOOKS:
            ftype, raising = FailureType.FIXTURE, None
        else:
            ftype, raising = tracebacks.type_failure(text, root=root, changed=changed)
        name = label
        if label.startswith(_FAILED_TEST + "."):
            module = _failed_module(text, root) if ftype is FailureType.IMPORT else None
            name = module if module is not None else label.rsplit("#", 1)[0]
        results[label] = case(name, outcome_for(ftype), ftype, raising, message)
    blocked = {b[1] for b in blocks if b[0] != "UNEXPECTED SUCCESS"}
    for label, status in (*statuses.items(), *hooks.items()):
        if status in ("FAIL", "ERROR") and label not in blocked:
            raise ValueError(f"{label} reported {status} with no {status} block")
    for label in started:
        if label not in statuses and label not in blocked:
            raise ValueError(f"{label} started but reported no status")
    _check_django_summary(lines, started, blocks, sum(1 for r in results.values() if r.expected_failure))
    out: list[CaseResult] = []
    for label, result in results.items():
        repeated = label.startswith(_FAILED_TEST + ".") and any(r.name == result.name for r in out)
        if not repeated:  # two labels in one module that failed to import: one case for the module
            out.append(result)
    return tuple(out)


def _check_django_summary(
    lines: Sequence[str], started: Sequence[str], blocks: list[tuple[str, str, str]], expected_failures: int
) -> None:
    """The run's counts against unittest's own summary: the tests that started, the FAIL and ERROR blocks, and the
    expected failures, so a test's own output that imitates an 'expected failure' status is refused, never read as
    one (unittest counts only the expected failures it saw)."""
    ran = next((int(m.group("n")) for line in lines if (m := _RAN.match(line))), None)
    if ran is None:
        if started or blocks:
            raise ValueError("the run reported tests but no 'Ran N tests' line")
        return
    if ran != len(started):
        raise ValueError(f"unittest ran {ran} tests but {len(started)} reported a status")
    tail = "\n".join(lines[-50:])
    counts = {m.group("name"): int(m.group("n")) for m in _SUMMARY_COUNTS.finditer(tail)}
    failures = sum(1 for kind, _, _ in blocks if kind == "FAIL")
    errors = sum(1 for kind, _, _ in blocks if kind == "ERROR")
    if counts.get("failures", 0) != failures or counts.get("errors", 0) != errors:
        raise ValueError(
            f"the summary counts {counts.get('failures', 0)} failures and {counts.get('errors', 0)} errors, "
            f"the blocks {failures} and {errors}"
        )
    if counts.get("expected failures", 0) != expected_failures:
        raise ValueError(
            f"the summary counts {counts.get('expected failures', 0)} expected failures, the progress lines "
            f"{expected_failures}"
        )


# ---------------------------------------------------------------- sympy

_SYMPY_NAME = re.compile(r"[A-Za-z_]\w*")


class SympyRunner(_Bound):
    kind = RunnerKind.SYMPY
    stream: Stream = "stdout"

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        env, words = command_environment(target)
        base = ["python", *words] if words[:1] == ["bin/test"] else (words or ["python", "bin/test", "-C", "--verbose"])
        for option in ("-C", "--verbose"):
            if option not in base:
                base.append(option)
        paths: list[str] = []
        names: list[str] = []
        whole = False
        for test in _check_tests(tests):
            path, sep, name = test.partition("::")
            if "/" not in path and not path.endswith(".py"):
                # bin/test reads a bare word as part of a path to match, so a lone name would run the wrong files.
                raise ValueError(f"{test!r} names no file: a sympy test is a path, or path::name")
            if path not in paths:
                paths.append(path)
            if not sep:
                whole = True
                continue
            if _SYMPY_NAME.fullmatch(name) is None:
                raise ValueError(f"{test!r} does not name a sympy test function")
            if name not in names:
                names.append(name)
        # bin/test reads -k with a callback that takes every argument up to the next option and replaces the value on
        # each -k, so every name goes after one -k, and the -k goes last. A whole file runs every test in it.
        keywords = [] if whole else ["-k", *names]
        return _with_env(env, [*base, *paths, *keywords])

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return sympy_cases(output, root=self.root, changed_files=changed_files)


# A test file's header: any status characters of module-level exceptions printed before it (sympy writes an "E" with
# no newline for a test module whose own code raised), then its path and its test count.
_SYMPY_FILE = re.compile(r"^(?P<pre>[ET]*)(?P<path>[a-z_][\w./-]*\.py)\[(?P<n>\d+)\]\s*$")
_SYMPY_IMPORT_FAILED = re.compile(r"^(?P<pre>[ET]*)(?P<path>[a-z_][\w./-]*\.py)\[\?\]\s+Failed to import\b")
_SYMPY_MODULE_ONLY = re.compile(r"^(?P<pre>[ET]+)\s*$")  # module-level exceptions with no file header after them
_SYMPY_PROGRESS = re.compile(r"^(?P<name>test_\w*)(?: (?P<rest>.*?))?\s*$")
_SYMPY_STATUS = re.compile(r"(?:^|\s)(?P<status>ok|F|E|T|K|f|X|s|w)$")
_SYMPY_FILE_RESULT = re.compile(r"\s*\[(?:OK|FAIL)\]$")  # the file's own verdict, printed after its last test
_SYMPY_SECTION = re.compile(r"^_*\s(?P<path>\S+?\.py)(?::(?P<name>\S+?))?\s_*$")
_SYMPY_RULE = re.compile(r"^_{10,}$")
_SYMPY_FINISHED = re.compile(r"tests finished: (?P<counts>.*?),\s+in [0-9.]+ seconds", re.DOTALL)
_SYMPY_COUNT = re.compile(
    r"(?P<n>\d+) (?P<what>passed|failed|skipped|expected to fail but passed|expected to fail|exceptions?)"
)
_TIMEOUT_SKIP = "t"  # a skip whose reason is "Timeout": sympy prints it "Timeout T", and counts it as skipped
_SKIPS = ("s", "w", "K", _TIMEOUT_SKIP)


@dataclass(frozen=True)
class _Exception:
    """One entry of sympy's own exception list, in the order the run reported it: a test's E or T, a test module whose
    code raised (no test named), or a test module that failed to import."""

    kind: Literal["test", "module", "import"]
    file: str | None = None
    name: str | None = None


def _sympy_status(text: str) -> str | None:
    found = _SYMPY_STATUS.search(text)
    if found is None:
        return None
    status = found.group("status")
    return _TIMEOUT_SKIP if status == "T" and text.endswith("Timeout T") else status


def sympy_cases(output: str, *, root: str, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
    """Cases from a sympy ``bin/test --verbose`` run (its stdout). A case's name is its test function's name, as sympy
    and SWE-bench name it, and its file is the test file, since names repeat across files.

    sympy prints each test file's path and test count, then each test on its own line, the name then its status
    (``ok``, ``F`` for an AssertionError, ``E`` for any other exception, ``f`` for an XFAIL test that failed as
    expected, a SKIPPED case with ``expected_failure`` set, ``X`` for one that passed unexpectedly, a case that
    passed, ``s``, ``w`` and ``K`` for skips, ``T`` for its own timeout); a test's own output can push the
    status to a later line. A test module whose own code raised prints an ``E`` with no newline (so it lands in front
    of the next file's header) and no header of its own; one that failed to import prints its path and "Failed to
    import". Each of those is an ERROR case typed IMPORT, named by its file. After every test sympy prints one
    traceback per exception, in the order they happened, and then one per failure: a rule of underscores, then
    ``<path>:<name>`` centered in underscores (for a module-level exception, the name of whatever test ran before it;
    for an import error, the module's absolute path), then the traceback without its first header line, which is put
    back here so the traceback reader reads it. Every count is checked against sympy's own summary."""
    changed = tracebacks.normalize(changed_files)
    lines = _ANSI.sub("", output).replace("\r\n", "\n").split("\n")
    statuses: dict[tuple[str, str], str] = {}
    order: list[tuple[str, str] | int] = []  # each test's key, or the index of a module-level entry in `raised`
    raised: list[_Exception] = []
    failed: list[tuple[str, str]] = []
    pending: tuple[str, str] | None = None
    current = ""
    entered = 0  # file headers: a run that entered a file and printed no summary crashed
    end = len(lines)

    def module_errors(pre: str) -> None:
        for _ in pre:
            order.append(len(raised))
            raised.append(_Exception("module"))

    def settle(key: tuple[str, str], status: str) -> None:
        if key in statuses:
            raise ValueError(f"{key[0]}:{key[1]} reported a status twice")
        statuses[key] = status
        order.append(key)
        if status in ("E", "T"):
            raised.append(_Exception("test", *key))
        elif status == "F":
            failed.append(key)

    for i, raw in enumerate(lines):
        if raw.startswith("____") or "tests finished:" in raw:
            end = i
            break
        line = _SYMPY_FILE_RESULT.sub("", raw)
        header = _SYMPY_FILE.match(line)
        if header is not None:
            module_errors(header.group("pre"))
            current, pending = header.group("path"), None
            entered += 1
            continue
        broken = _SYMPY_IMPORT_FAILED.match(line)
        if broken is not None:
            module_errors(broken.group("pre"))
            order.append(len(raised))
            raised.append(_Exception("import", broken.group("path")))
            pending = None
            continue
        m = _SYMPY_PROGRESS.match(line)
        if m is not None:
            key = (current, m.group("name"))
            status = _sympy_status(m.group("rest") or "")
            if status is not None:
                settle(key, status)
                pending = None
            else:
                pending = key
            continue
        if pending is not None:
            status = _sympy_status(line.strip())
            if status is not None:
                settle(pending, status)
                pending = None
            continue
        only = _SYMPY_MODULE_ONLY.match(line)
        if only is not None:
            module_errors(only.group("pre"))
    _check_sympy_summary(output, list(statuses.values()), len(raised), entered=entered)
    sections = _sympy_sections(lines[end:])
    if len(sections) != len(raised) + len(failed):
        raise ValueError(
            f"sympy printed {len(sections)} tracebacks for {len(raised)} exceptions and {len(failed)} failures"
        )
    texts: dict[tuple[str, str], str] = {}
    modules: dict[int, CaseResult] = {}
    for n, (entry, (path, name, text)) in enumerate(zip(raised, sections, strict=False)):
        if entry.kind == "test":
            if entry.file is None or entry.name is None or (path, name) != (entry.file, entry.name):
                raise ValueError(f"traceback {n + 1} names {path}:{name}, the run reported {entry.file}:{entry.name}")
            texts[(entry.file, entry.name)] = text
            continue
        modules[n] = _module_case(entry, path, text, root)
    for key, (path, name, text) in zip(failed, sections[len(raised) :], strict=True):
        if (path, name) != key:
            raise ValueError(f"a failure's traceback names {path}:{name}, the run reported {key[0]}:{key[1]}")
        texts[key] = text
    results: list[CaseResult] = []
    for item in order:
        if isinstance(item, int):
            results.append(modules[item])
            continue
        path, name = item
        status = statuses[item]
        text = texts.get(item, "")
        final = tracebacks.final_block(tracebacks.parse(text))
        message = final.exception if final is not None else ""
        if status == "ok":
            results.append(case(name, Outcome.PASS, file=path))
        elif status in _SKIPS:
            results.append(case(name, Outcome.SKIPPED, message="skipped", file=path))
        elif status == "f":
            results.append(case(name, Outcome.SKIPPED, message="expected failure", file=path, expected_failure=True))
        elif status == "X":
            results.append(case(name, Outcome.PASS, message=UNEXPECTED_PASS, file=path))
        elif status == "T":
            results.append(case(name, Outcome.TIMEOUT, FailureType.TIMEOUT, message="timed out", file=path))
        elif status == "F":
            results.append(case(name, Outcome.FAIL, FailureType.ASSERTION, None, message, file=path))
        else:  # E
            ftype, raising = tracebacks.type_failure(text, root=root, changed=changed)
            results.append(case(name, outcome_for(ftype), ftype, raising, message, file=path))
    return tuple(results)


def _module_case(entry: _Exception, header: str, text: str, root: str) -> CaseResult:
    """A test module that failed to import, or whose own code raised: an ERROR case typed IMPORT, named by its file.
    For an import error sympy names the module (by its absolute path); for a module whose code raised, the module is
    the traceback's first frame, its ``<module>`` code."""
    blocks = tracebacks.parse(text)
    final = tracebacks.final_block(blocks)
    if entry.kind == "import":
        if tracebacks.repo_path(header, root) != entry.file:
            raise ValueError(f"an import error's traceback names {header}, the run reported {entry.file}")
        file = entry.file
    else:
        first = final.frames[0] if final is not None and final.frames else None
        file = tracebacks.repo_path(first.path, root) if first is not None and first.function == "<module>" else None
        if file is None:
            raise ValueError("a module-level exception's traceback names no test module")
    message = final.exception if final is not None else "the test module did not import"
    assert file is not None
    return case(file, Outcome.ERROR, FailureType.IMPORT, None, message, file=file)


def _sympy_sections(lines: Sequence[str]) -> list[tuple[str, str | None, str]]:
    """Every traceback sympy printed after its tests, in order: (the path in its header, the test name in its header,
    the traceback with the header line sympy leaves out put back)."""
    sections: list[tuple[str, str | None, str]] = []
    i, n = 0, len(lines)
    while i < n:
        m = _SYMPY_SECTION.match(lines[i]) if i > 0 and _SYMPY_RULE.match(lines[i - 1]) else None
        if m is None:
            i += 1
            continue
        j = i + 1
        while j < n and not _SYMPY_RULE.match(lines[j]) and "tests finished:" not in lines[j]:
            j += 1
        body = "\n".join(lines[i + 1 : j]).rstrip()
        if body.startswith("  File "):
            body = f"{tracebacks.HEADER}\n{body}"
        sections.append((m.group("path"), m.group("name"), body))
        i = j
    return sections


def _check_sympy_summary(output: str, statuses: Sequence[str], exceptions: int, *, entered: int = 0) -> None:
    # The last summary is sympy's own: it prints it after every test, so text a test printed comes before it.
    at = output.rfind("tests finished:")
    m = _SYMPY_FINISHED.match(output, at) if at >= 0 else None
    if m is None:
        if statuses or exceptions or entered:
            raise ValueError("the run reported tests but no 'tests finished' line")
        return
    counts: dict[str, int] = {}
    for c in _SYMPY_COUNT.finditer(" ".join(m.group("counts").split())):
        what = "exception" if c.group("what").startswith("exception") else c.group("what")
        counts[what] = counts.get(what, 0) + int(c.group("n"))
    seen = {
        "passed": sum(1 for s in statuses if s == "ok"),
        "failed": sum(1 for s in statuses if s == "F"),
        "exception": exceptions,
        "skipped": sum(1 for s in statuses if s in _SKIPS),
        "expected to fail": sum(1 for s in statuses if s == "f"),
        "expected to fail but passed": sum(1 for s in statuses if s == "X"),
    }
    for what, n in seen.items():
        if counts.get(what, 0) != n:
            raise ValueError(f"sympy's summary counts {counts.get(what, 0)} {what}, the progress lines {n}")


# ---------------------------------------------------------------- choosing a runner


def runner_for(target: Target, *, root: str = "/testbed") -> SuiteRunner:
    """The runner for a target's kind, bound to ``root``."""
    kinds: dict[RunnerKind, type[_Bound]] = {
        RunnerKind.PYTEST: PytestRunner,
        RunnerKind.DJANGO: DjangoRunner,
        RunnerKind.SYMPY: SympyRunner,
    }
    runner = kinds[target.test_runner](root)
    assert isinstance(runner, PytestRunner | DjangoRunner | SympyRunner)
    return runner


def expected_cases(kind: RunnerKind, tests: Sequence[str]) -> list[str]:
    """The cases a run of ``tests`` reports, for ``expect``: the ids themselves for pytest and sympy (a sympy
    ``path::name`` answers for the case of that name in that file only), and labels for Django."""
    if kind is RunnerKind.DJANGO:
        return [django_label(test) for test in tests]
    return list(tests)
