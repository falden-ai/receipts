"""Turn a test run into contract-grade results (SK1 part a, spec hard rules 2 and 3).

A run is read in three steps, and each step fails closed.

1. The cases. A runner reads its own output into ``contracts.CaseResult`` values (``receipts.attack.runners``); for
   pytest that is the junit report this module parses, each case named by its node id in the module it was
   collected from (``nodeid``), never the file its function is defined in. A skipped or expected-to-fail test is
   SKIPPED, never a pass; an expected failure that failed as expected (pytest's xfail, unittest's expectedFailure,
   sympy's XFAIL) also has ``expected_failure`` set (contract v3.1), so a reader can tell it from a plain skip, and
   nothing in this lane counts it as a pass (only R24's gold check does, from the cases). An unexpected pass (sympy's
   X, unittest's "unexpected success") is not a failure and not infra: it is a case that passed, with a message that
   says it passed unexpectedly. Failure types come from the
   report's structure and the traceback's structure (``receipts.attack.tracebacks``), never from a word in a message:
   an ``<error>`` element is a collection, fixture or infra error and never a counted failure, and only an
   AssertionError, pytest's DID NOT RAISE or DID NOT WARN (REG-565), or a crash whose deepest repository frame is a
   changed non-test file counts.
2. The evidence. A run the sandbox killed for time is TIMEOUT. A pytest run with no report, a report or output that
   cannot be read, or an exit code that disagrees with the cases is ERROR (infra), and its cases are not used. The
   exit code is checked on every run, with or without ``expect``: pytest exits 0 only when no case failed and 1 only
   when one did (2 for a collection error, 5 when nothing was collected), and Django's and sympy's runners exit 0 or
   1 the same way, except that a run whose only odd cases are unexpected passes may exit either way (sympy and
   Django 3.0 exit 0 on one, later Django exits 1). Each run writes its report to its own path, so a report an
   earlier run left can never be read as this one's, and a run that died before writing one has none.
3. The outcome. The run takes its most catch-relevant failure; without one it is ERROR (infra) when nothing ran,
   SKIPPED when a case was skipped and PASS only when every case passed. With ``expect`` (a repro's own case names,
   or R24's SWE-bench ids), only those cases count, every one of them must have run, and a run where one never ran
   is ERROR: typed IMPORT when the run reports that case's test module failing to import (sympy's case for a module,
   named by its file; a pytest collection error for its file; Django's failed import of its module), and INFRA
   otherwise, a missing case nothing explains included. A case answers an expected id as ``contracts.answers``
   matches them in the target's repository: by its own name, or by the id swebench 4.1.0's log parser records for
   it there (pylint's 'test_stdin[/mymodule.py]'). Every case that answers one id must pass: a name that repeats
   across files (sympy) matches every case of that name, and ``path::name`` only the case of that name in that file.

Whenever a run could not be read, or an expected case never ran, its result says why in ``RunResult.problem``, so a
reader never rebuilds the reason from the output's tails; a run that was read and answered has none.

junit text is attacker-influenced. The parse is stdlib ElementTree, which resolves no external entity, and a DOCTYPE
or ENTITY declaration (which pytest never writes) is refused before parsing.
"""

from __future__ import annotations

import posixpath
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from xml.etree.ElementTree import Element, fromstring

from receipts.attack import tracebacks
from receipts.contracts import CaseResult, FailureType, Outcome, RunnerKind, RunResult, SuiteRunner, answers

# A report larger than this is refused rather than parsed: pytest's junit for a repro is a few kilobytes.
MAX_REPORT_BYTES = 8 * 1024 * 1024
# A case's message is kept short: it reaches a model only through the attackers, which truncate again.
MAX_MESSAGE_CHARS = 2000

_COLLECTION_ERROR = "collection failure"
_FIXTURE_PREFIXES = ("failed on setup with ", "failed on teardown with ")
_FIXTURE_MESSAGES = ("test setup failure", "test teardown failure")
# The message of a case that passed although it was marked to fail (sympy's X, unittest's unexpected success).
UNEXPECTED_PASS = "passed unexpectedly"

# Highest priority first: when a run has several failing cases, the run takes the most catch-relevant type.
_PRIORITY: tuple[FailureType, ...] = (
    FailureType.ASSERTION,
    FailureType.CRASH_IN_CHANGED,
    FailureType.TIMEOUT,
    FailureType.CRASH_ELSEWHERE,
    FailureType.IMPORT,
    FailureType.NAME,
    FailureType.FIXTURE,
    FailureType.INFRA,
    FailureType.OTHER,
)
_COUNTED = frozenset({FailureType.ASSERTION, FailureType.CRASH_IN_CHANGED})


class JunitParseError(ValueError):
    """The junit report is malformed, too large, or carries a document type the writer never emits."""


def outcome_for(ftype: FailureType) -> Outcome:
    if ftype in _COUNTED:
        return Outcome.FAIL
    if ftype is FailureType.TIMEOUT:
        return Outcome.TIMEOUT
    return Outcome.ERROR


def clip(text: str, limit: int = MAX_MESSAGE_CHARS) -> str:
    """Keep the head of a message, valid as UTF-8."""
    text = text.encode("utf-8", "replace").decode("utf-8", "replace")
    return text if len(text) <= limit else text[: limit - 3] + "..."


def case(
    name: str,
    outcome: Outcome,
    ftype: FailureType | None = None,
    raising: str | None = None,
    message: str = "",
    *,
    file: str | None = None,
    expected_failure: bool = False,
) -> CaseResult:
    return CaseResult(
        name=name,
        outcome=outcome,
        failure_type=ftype,
        raising_file=raising,
        message=clip(message.strip()),
        file=file,
        expected_failure=expected_failure,
    )


def unexpected_pass(case: CaseResult) -> bool:
    """A case that passed although it was marked to fail."""
    return case.outcome is Outcome.PASS and case.message == UNEXPECTED_PASS


def case_id(case: CaseResult) -> str:
    """A case's id across files: ``path::name`` for a case that names its file (sympy), else its name."""
    return f"{case.file}::{case.name}" if case.file is not None and case.file != case.name else case.name


# ---------------------------------------------------------------- pytest's junit


def _parse(xml: str) -> Element:
    if len(xml.encode("utf-8", "replace")) > MAX_REPORT_BYTES:
        raise JunitParseError(f"junit report is larger than {MAX_REPORT_BYTES} bytes; refusing to parse")
    lowered = xml.lower()
    if "<!doctype" in lowered or "<!entity" in lowered:
        raise JunitParseError("junit report carries a DOCTYPE or ENTITY declaration; refusing to parse")
    try:
        return fromstring(xml)
    except Exception as exc:
        raise JunitParseError(f"could not parse junit report: {exc}") from exc


def _testcases(root: Element) -> list[Element]:
    suites = [root] if root.tag == "testsuite" else list(root.iter("testsuite"))
    return [case for suite in suites for case in suite.findall("testcase")]


def _dotted(file: str) -> str:
    """A file path as pytest's junit writes it into a classname (junitxml.py, mangle_test_address): slashes as dots,
    and no .py."""
    module = file[:-3] if file.endswith(".py") else file
    return module.replace("/", ".")


def _starts(classname: str, module: str) -> bool:
    """Whether a dotted module is the classname, or the classname's start up to a dot."""
    return classname == module or classname.startswith(module + ".")


Modules = Mapping[str, frozenset[str]]  # a test module as a classname writes it -> the files a report names for it


def collection_modules(testcases: Sequence[Element]) -> dict[str, frozenset[str]]:
    """The test modules a report establishes: a testcase whose file matches its classname was collected from that
    file, so the file's dotted name is a module, and the file is its path."""
    found: dict[str, set[str]] = {}
    for element in testcases:
        classname, file = element.get("classname", ""), element.get("file")
        if classname and file and _starts(classname, _dotted(file)):
            found.setdefault(_dotted(file), set()).add(file)
    return {module: frozenset(files) for module, files in found.items()}


def _class_name(part: str) -> bool:
    return part[:1].isascii() and part[:1].isupper() and part.isidentifier()


def _collected_from(classname: str, modules: Modules) -> tuple[str, str] | None:
    """The module a classname's testcase was collected from, dotted and as a path, when its file is not that module.
    The longest start of the classname, up to a dot, that the report establishes (``collection_modules``) decides;
    a start established by two files decides nothing. When the report establishes none, the classes are the trailing
    parts that start with a capital letter and the module is the rest: Python's own convention (CapWords classes,
    lowercase modules, PEP 8), which each of the 36,682 pytest ids R24 can name in SWE-bench Verified's 500 rows
    follows (checked 2026-10-28): for every one of them it gives back the id's own path. None when neither
    answers."""
    parts = classname.split(".")
    if not all(parts):
        return None
    for end in range(len(parts), 0, -1):
        module = ".".join(parts[:end])
        files = modules.get(module)
        if files:
            return (module, next(iter(files))) if len(files) == 1 else None
    end = len(parts)
    while end > 0 and _class_name(parts[end - 1]):
        end -= 1
    if end == 0:
        return None
    module = ".".join(parts[:end])
    return module, module.replace(".", "/") + ".py"


def nodeid(classname: str, name: str, file: str | None, modules: Modules | None = None) -> str:
    """Rebuild pytest's node id from a junit (xunit1) testcase: the module it was collected from, the classes after
    that module in its classname, and its name. pytest writes the classname from the node id itself (the module's
    path with dots for slashes and no .py, then any classes), and the file from where the test function is defined,
    which is another file for a test a class inherits from another module (astropy__astropy-13236's TestMetaTable)
    or one whose decorator's wrapper sets no ``__wrapped__`` (matplotlib's ``check_figures_equal``, whose file is
    lib/matplotlib/testing/decorators.py). So when the file matches the classname it is the module, and when it does
    not, the module comes from the classname alone (``_collected_from``, over the ``modules`` the same report
    establishes). A testcase whose module cannot be told keeps ``<classname>::<name>``, dotted, which no SWE-bench id
    is: an expected case of that name never ran, never a false pass."""
    if not classname:
        return file or name  # a collection error or skip names the module itself
    if file and _starts(classname, _dotted(file)):
        module, path = _dotted(file), file
    else:
        collected = _collected_from(classname, modules or {})
        if collected is None:
            return f"{classname}::{name}"
        module, path = collected
    classes = classname[len(module) + 1 :].split(".") if classname != module else []
    return "::".join([path, *classes, name])


def junit_cases(xml: str, *, root: str, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
    """Every testcase in a pytest junit report (xunit1, ``--tb=native``), named by node id (``nodeid``: from the module
    it was collected from, which the report's other testcases establish) and typed against the changed files, with
    frames resolved against the workspace root. A ``<skipped>`` is a SKIPPED case, and one that records an xfail test
    failing as expected (``xfailed``) sets ``expected_failure``."""
    changed = tracebacks.normalize(changed_files)
    results: list[CaseResult] = []
    testcases = _testcases(_parse(xml))
    modules = collection_modules(testcases)
    for element in testcases:
        name = nodeid(element.get("classname", ""), element.get("name", ""), element.get("file"), modules)
        failure, error, skipped = element.find("failure"), element.find("error"), element.find("skipped")
        if failure is not None:
            message = failure.get("message", "")
            text = "".join(failure.itertext())
            ftype, raising = tracebacks.type_failure(text, root=root, changed=changed, exception_text=message)
            results.append(case(name, outcome_for(ftype), ftype, raising, message))
        elif error is not None:
            message = error.get("message", "")
            ftype = _error_type(message)
            if message == _COLLECTION_ERROR:  # what failed: pytest's last 'E   ' line names the import error
                said = [line[1:].strip() for line in "".join(error.itertext()).splitlines() if line.startswith("E ")]
                message = f"{message}: {said[-1]}" if said else message
            results.append(case(name, outcome_for(ftype), ftype, None, message))
        elif skipped is not None:
            xfail = xfailed(skipped)
            message = skipped.get("message", "") or ("expected failure" if xfail else "skipped")
            results.append(case(name, Outcome.SKIPPED, None, None, message, expected_failure=xfail))
        else:
            results.append(case(name, Outcome.PASS))
    return tuple(results)


# How pytest's junit writer records an xfail test that failed as expected (its LogXML append_skipped, read in the
# released wheels): from 4.5.0, <skipped type="pytest.xfail" message="<reason>"/>; before it (3.3.1, which SWE-bench's
# astropy 0.1 to 1.3 environments pin, through 4.4.2), <skipped message="expected test failure"><reason></skipped>,
# with no type. A plain skip carries type="pytest.skip" in every version, whatever its message says, and a skipped
# module ("collection skipped") or pytest's "xfail-marked test passes unexpectedly" has another message.
_XFAIL_TYPE = "pytest.xfail"
_XFAIL_MESSAGE_BEFORE_4_5 = "expected test failure"


def xfailed(skipped: Element) -> bool:
    """Whether a junit ``<skipped>`` element is pytest's record of an xfail test that failed as expected (an xfail
    marker, ``pytest.xfail()``, a ``run=False`` xfail, or unittest's expectedFailure run by pytest): what the -rA
    summary prints as XFAIL. Read from the element's structure, never from a reason a test chose."""
    kind = skipped.get("type")
    if kind is not None:
        return kind == _XFAIL_TYPE
    return skipped.get("message") == _XFAIL_MESSAGE_BEFORE_4_5


def _error_type(message: str) -> FailureType:
    """An <error> element: pytest writes one for a collection error and for a fixture's setup or teardown. Neither is
    a counted failure, and nor is anything else written as an error."""
    if message == _COLLECTION_ERROR:
        return FailureType.IMPORT
    if message.startswith(_FIXTURE_PREFIXES) or message in _FIXTURE_MESSAGES:
        return FailureType.FIXTURE
    return FailureType.INFRA


# ---------------------------------------------------------------- reducing a run


Homes = Mapping[str, Sequence[str]]  # a bare sympy name -> the run's test files that define it


def import_failure_of(wanted: str, cases: Sequence[CaseResult], homes: Homes | None = None) -> CaseResult | None:
    """The case by which a run reports that the test module ``wanted`` lives in failed to import, or None.

    Such a case stands for a whole module and is ERROR typed IMPORT: sympy names it by its file and sets that file;
    pytest's collection error names its file (a package's ``__init__.py`` stands for the package); the Django runner
    names it by the module's label. ``wanted`` is a case id: ``path::name`` (pytest, sympy) whose path is that module
    or lies under that package, a Django label under that module's label, or a bare sympy name, whose files ``homes``
    says (SWE-bench names a sympy test by its function alone)."""
    for case in cases:
        if case.outcome is Outcome.ERROR and case.failure_type is FailureType.IMPORT and _holds(case, wanted, homes):
            return case
    return None


def _holds(module: CaseResult, wanted: str, homes: Homes | None) -> bool:
    path = wanted.partition("::")[0] if "::" in wanted else None
    if module.file is not None:  # sympy's case for a module, named by its file
        defines = module.file in (homes or {}).get(wanted, ())
        return module.name == module.file and (path == module.file or defines)
    if "::" in module.name:
        return False  # a case inside a module (an import a test made when it ran), never a module
    if module.name.endswith(".py"):  # pytest's collection error, named by its file
        package = posixpath.dirname(module.name) if posixpath.basename(module.name) == "__init__.py" else None
        return path is not None and (path == module.name or (package is not None and path.startswith(package + "/")))
    return wanted.startswith(module.name + ".")  # a Django module that failed to import, named by its label


def reduce_cases(
    cases: Sequence[CaseResult],
    *,
    expect: Sequence[str] | None = None,
    homes: Homes | None = None,
    repo: str = "",
) -> tuple[Outcome, FailureType | None]:
    """Fold a run's cases into one outcome and failure type (``reduce_run`` without the reason)."""
    outcome, ftype, _ = reduce_run(cases, expect=expect, homes=homes, repo=repo)
    return outcome, ftype


def reduce_run(
    cases: Sequence[CaseResult],
    *,
    expect: Sequence[str] | None = None,
    homes: Homes | None = None,
    repo: str = "",
) -> tuple[Outcome, FailureType | None, str]:
    """Fold a run's cases into one outcome and failure type, and why when the run cannot be answered from them (the
    problem a RunResult carries; empty when there is none). With ``expect``, only those cases count, and a run where
    one never ran is ERROR: IMPORT when the run reports that case's test module failing to import, INFRA when
    anything else, or nothing, explains a missing case. A case counts for an expected id as ``contracts.answers``
    matches them in ``repo``, the target's repository (no repository: no harness renames), and when several cases
    answer one id, all of them must pass. A case that failed as expected is SKIPPED like any skip, so a run of one
    is SKIPPED, never PASS (contract v3.1: only R24's gold check counts it as passing, from the cases)."""
    chosen = list(cases)
    if expect is not None:
        wanted = list(dict.fromkeys(expect))
        if not wanted:
            raise ValueError("expect names at least one case")
        chosen = [c for c in cases if any(answers(c, w, repo=repo) for w in wanted)]
        missing = [w for w in wanted if not any(answers(c, w, repo=repo) for c in chosen)]
        if missing:
            why = [(w, import_failure_of(w, cases, homes)) for w in missing]
            ftype = FailureType.IMPORT if all(module is not None for _, module in why) else FailureType.INFRA
            return Outcome.ERROR, ftype, _missing(why, cases)
    types = [c.failure_type for c in chosen if c.failure_type is not None and c.outcome is not Outcome.SKIPPED]
    if types:
        best = min(types, key=_PRIORITY.index)
        return outcome_for(best), best, ""
    if not chosen:
        return Outcome.ERROR, FailureType.INFRA, "the run reported no case"
    if any(c.outcome is Outcome.SKIPPED for c in chosen):
        return Outcome.SKIPPED, None, ""
    if all(c.outcome is Outcome.PASS for c in chosen):
        return Outcome.PASS, None, ""
    odd = next(c for c in chosen if c.outcome is not Outcome.PASS)
    return Outcome.ERROR, FailureType.OTHER, f"case {odd.name} ended {odd.outcome.value} with no failure type"


def _listed(names: Sequence[str], limit: int = 5) -> str:
    shown = ", ".join(names[:limit])
    return shown if len(names) <= limit else f"{shown} and {len(names) - limit} more"


_HOOK_NAMES = ("setUpClass", "setUpModule")


def hook_failure_of(wanted: str, cases: Sequence[CaseResult]) -> CaseResult | None:
    """A class's setUpClass or a module's setUpModule that errored over ``wanted`` (Django labels): the run's own
    reason a case under it never ran. It is not an import error, so the case stays infra; the problem names it."""
    for case in cases:
        owner, _, hook = case.name.rpartition(".")
        if case.outcome is Outcome.ERROR and hook in _HOOK_NAMES and owner and wanted.startswith(owner + "."):
            return case
    return None


def _missing(why: Sequence[tuple[str, CaseResult | None]], cases: Sequence[CaseResult]) -> str:
    """Why expected cases never ran: for each, the import error of its test module, or that the run reports none (and
    the class or module hook that errored over it, when the run reports one)."""
    by_module: dict[str, list[str]] = {}
    messages: dict[str, str] = {}
    unexplained: dict[str, list[str]] = {}
    for wanted, module in why:
        if module is not None:
            by_module.setdefault(module.name, []).append(wanted)
            messages[module.name] = module.message
            continue
        hook = hook_failure_of(wanted, cases)
        if hook is not None:
            messages[hook.name] = hook.message
        unexplained.setdefault(hook.name if hook is not None else "", []).append(wanted)
    parts = []
    for name, missing in by_module.items():
        whose = "its" if len(missing) == 1 else "their"
        parts.append(
            f"expected case{'s' * (len(missing) > 1)} {_listed(missing)} never ran: {whose} test module {name} failed "
            f"to import ({clip(messages[name], 300) or 'no message'})"
        )
    for hook_name, missing in unexplained.items():
        whose = "its test module" if len(missing) == 1 else "their test modules"
        part = (
            f"expected case{'s' * (len(missing) > 1)} {_listed(missing)} never ran, and the run reports no import "
            f"error for {whose}"
        )
        if hook_name:
            part += f" ({hook_name} errored: {clip(messages[hook_name], 300) or 'no message'})"
        parts.append(part)
    return clip("; ".join(parts))


def exit_agrees(kind: RunnerKind, exit_code: int | None, cases: Sequence[CaseResult]) -> bool:
    """Whether the runner's exit code agrees with the cases it reported. Every runner here exits 0 only when no case
    failed and 1 only when one did; pytest also exits 2 for a collection error and 5 when it collected nothing. An
    unexpected pass is not a failed case, but runners disagree about it (sympy and Django 3.0 exit 0, later Django
    counts it and exits 1), so a run whose only odd cases are unexpected passes may exit 0 or 1."""
    if exit_code is None:
        return True
    bad = any(c.outcome not in (Outcome.PASS, Outcome.SKIPPED) for c in cases)
    if exit_code == 0:
        return not bad
    if exit_code == 1:
        return bad or any(unexpected_pass(c) for c in cases)
    if kind is RunnerKind.PYTEST and exit_code == 2:
        return any(c.outcome is Outcome.ERROR and c.failure_type is FailureType.IMPORT for c in cases)
    if kind is RunnerKind.PYTEST and exit_code == 5:
        return all(c.outcome is Outcome.SKIPPED for c in cases)
    return False


@dataclass(frozen=True)
class SuiteRun:
    """One run of some tests: the run's result over every case it reported, the cases themselves (empty when the
    run's evidence could not be used), and, when it could not, why."""

    result: RunResult
    cases: tuple[CaseResult, ...]
    problem: str | None = None


def _with(result: RunResult, outcome: Outcome, ftype: FailureType | None, problem: str = "") -> RunResult:
    # Validated again, so the result's own rules (a failure names its type, a pass has none) still hold.
    fields = {**result.model_dump(), "outcome": outcome, "failure_type": ftype, "problem": clip(problem)}
    return RunResult.model_validate(fields)


def _last_words(output: str | None, coarse: RunResult) -> str:
    """The last line the run printed (a crashed runner's exception, as a rule), for a problem that would otherwise
    say only that the exit code and the cases disagree."""
    for text in (output or "", coarse.stderr_tail, coarse.stdout_tail):
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        if lines:
            return f"; the run's last line: {clip(lines[-1], 300)}"
    return ""


def classify_suite(
    coarse: RunResult,
    *,
    runner: SuiteRunner,
    output: str | None,
    report: str | None,
    changed_files: Sequence[str],
    timed_out: bool,
    problem: str | None = None,
) -> SuiteRun:
    """Classify one run from what it printed and the report it wrote. ``output`` is None when the output was too
    large to read whole; ``problem`` is one the sandbox found while collecting the evidence (a report too large).
    A run that could not be read carries why in its result's ``problem``, and so does one whose cases reduce to no
    answer (no case at all)."""
    if timed_out:
        why = f"the run was killed for time after {coarse.duration_s:.0f} s"
        return SuiteRun(_with(coarse, Outcome.TIMEOUT, FailureType.TIMEOUT, why), (), why)
    if problem is None and runner.kind is RunnerKind.PYTEST and report is None:
        problem = "the run wrote no report" + _last_words(None, coarse)
    if problem is None and output is None:
        problem = "the run's output was too large to read"
    if problem is None:
        try:
            cases = tuple(runner.cases(output or "", report, changed_files))
        except ValueError as exc:  # JunitParseError, or a runner that refuses what it read
            problem = f"the run's report could not be read: {exc}"
        else:
            if not exit_agrees(runner.kind, coarse.exit_code, cases):
                problem = (
                    f"exit code {coarse.exit_code} disagrees with the {len(cases)} cases the run reported"
                    + _last_words(output, coarse)
                )
            else:
                outcome, ftype, why = reduce_run(cases)
                return SuiteRun(_with(coarse, outcome, ftype, why), cases)
    return SuiteRun(_with(coarse, Outcome.ERROR, FailureType.INFRA, problem), (), problem)


def with_expect(
    run: SuiteRun, expect: Sequence[str] | None, *, homes: Homes | None = None, repo: str = ""
) -> RunResult:
    """The run's result for ``expect``: the cases named there, every one of which must have run (``reduce_run``;
    ``homes`` says which files define each bare sympy name, ``repo`` is the target's repository). A run that could
    not be read keeps its own result."""
    if expect is None or run.problem is not None:
        return run.result
    outcome, ftype, problem = reduce_run(run.cases, expect=expect, homes=homes, repo=repo)
    return _with(run.result, outcome, ftype, problem)
