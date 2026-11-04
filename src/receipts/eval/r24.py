"""R24 on the run phase's real path (spec section 3): an instance stays in Tier A only if its gold passes FAIL_TO_PASS
and PASS_TO_PASS in the sandbox the attackers and the skeptic use, under gVisor. Every other instance is dropped, and
the log says why.

For each instance: `Sandbox.prepare(target, Side.GOLD, patch=gold patch + test patch)`, a kernel probe in a fresh
container (`dmesg` must show "Starting gVisor..."), then `Sandbox.run_test` with the instance's SuiteRunner on the test
files the harness runs (the test patch's files), with `expect` set to FAIL_TO_PASS plus PASS_TO_PASS, so PASS means
every one of those cases ran and passed. This module codes against contracts.Sandbox and contracts.SuiteRunner only;
the integrator wires the attack lane's DockerSandbox (runsc, the run phase's flags: uid 65534, a read-only root, the
repository in a RAM-backed workspace) and its runners. Nothing here runs on the Mac or in tests but fakes. Each worker
thread gets its own sandbox and runners from the wiring's factories: the contract says nothing about thread safety.

Names. `expect` holds case ids that the sandbox matches to the cases the run phase's runner reports
(contracts.CaseResult.name: a pytest node id, a Django label, a sympy test name) with contracts.answers, the one
matching the attack lane uses too. SWE-bench's pytest and sympy ids go into `expect` as they are: a case answers one
by its own name, or by the id swebench 4.1.0's log parser records for it in the instance's repository, which renames
a few (pylint, requests and pydicom keep only a slash and the last path component of a parameter that starts with
one slash, so pylint-dev__pylint-7080's 'tests/test_self.py::TestRunTC::test_stdin[/mymodule.py]' is its case
'test_stdin[/testbed/tests/mymodule.py-mymodule-/testbed/tests/mymodule.py]'; matplotlib writes MouseButton.LEFT and
MouseButton.RIGHT as 1 and 3). Its Django ids read 'test_x (app.tests.Class)' (or 'test_x (app.tests.Class.test_x)'
under Python 3.11 and later), whose label is 'app.tests.Class.test_x'. A Django test with a docstring SWE-bench names
by the docstring's first line; R24 names it from the source: inside the prepared gold workspace, through
`Sandbox.run`, a small stdlib script (`DOCSTRINGS`) reads the files the run will test with `ast` and prints the label
of every test method, its own or inherited within its file, of every class unittest would load there, whose
docstring starts with that line. A label it gets wrong can only be an expected case that never runs, which run_test
answers ERROR (INFRA): never a false keep.
What still names no case is left out of `expect` and listed on the result: a docstring the script did not find (an
inherited test from another file, say), and a pytest id SWE-bench's log parser cut at a space inside its parameters
(unbalanced brackets). An instance none of whose FAIL_TO_PASS ids names a case is unchecked (INFRA): R24 cannot check
its fix's own tests.

Before any instance runs, `naming_before_run` lists every instance with SWE-bench ids R24 cannot name yet, and
`run_check` writes them to unnamed.json: ids cut at a space, which no run can name, and Django docstring ids, which the
resolver names during the run when it finds them. An instance none of whose FAIL_TO_PASS ids can be named either way
is unchecked whatever the run does (sphinx-doc__sphinx-8265 and 8621 on this pool); a rerun cannot change that, only a
founder `--gold-drop` or a founder rule can, and its result says so (`unnameable`).

Test files. When the test patch touches no test file (django__django-10097 changes only .txt fixtures), the harness
runs the whole suite; R24 runs instead the test modules its named FAIL_TO_PASS and PASS_TO_PASS cases live in (a
Django label's module, a pytest id's file). An instance with neither is INFRA, with that reason, so `--gold-drop` can
land on it. No instance ever stops the run: anything one raises is that instance's INFRA, except a runtime mismatch.

Timeouts. A run of the test patch's files gets `--timeout` (the harness's 1800 s). A run of the modules the named
cases live in stands in for the harness's whole suite, so its timeout scales with its expected cases (`run_timeout`):
DERIVED_SECONDS_PER_CASE seconds per expected case, never less than `--timeout` and never more than
DERIVED_TIMEOUT_CAP_S unless `--timeout` is more. django__django-10097's 122 modules and about 1,870 expected cases
get about an hour instead of 1800 s.

Patches. prepare() applies gold plus the test patch as one diff, through the harness's chain (contracts.PatchMethod,
REG-569); each attempt and the result record the method the workspace says applied it, and the report lists every
instance that needed more than a plain `git apply --verbose`.

Verdicts. PASS keeps the instance. A FAIL (an assertion, or a crash in the gold's own files), an ERROR with a typed
failure (a crash elsewhere, an import, name or fixture error: the gold breaks in the run phase), or a SKIPPED (a case
never ran) drops it, and a TIMEOUT drops it only when both attempts, in fresh containers, ran out of time. An
expected case that never ran because its test module failed to import is ERROR typed IMPORT (contract v3): a drop,
as the harness drops it, never INFRA. A sandbox that raises, a probe that never ran, or a run_test that answers
ERROR with failure type INFRA or none (no report, or an expected case missing for any other reason) is INFRA:
unchecked, never a drop, and `merge` lets a rerun replace it. An instance whose gold passes as root under the harness
(`receipts.eval.runsc_check`) but fails here needs root or a writable root, which goes back to the founder (REG-565);
`diagnose` records that for every drop.

Expected failures. A case the runner reports as failing as expected (contract v3.1, CaseResult.expected_failure) is
graded as swebench 4.1.0 grades it for the instance's repository (`specs.expected_failure_passes`): where its log
parser reads pytest's XFAIL line, grading.py's test_passed counts it as passing, and so does R24, for FAIL_TO_PASS and
PASS_TO_PASS alike; on django, seaborn and sympy the parser writes nothing for it, test_failed counts it as failed,
and R24 keeps it SKIPPED, a drop. R24 does this through the runner it hands run_test (`GoldRunner`), so the sandbox's
own expect rule still decides; a plain skip is never touched and still drops the gold. Each attempt records the
expected cases its run reported as expected failures, and the report lists them per instance.

Environment. The gold target's test command carries the exports of the spec's eval_commands as leading assignments
(`specs.run_phase_test_command`): Django 1.11 to 3.2 run with LANG and LC_ALL set to en_US.UTF-8, as the harness runs
them.

The kernel probe answers three ways: the banner (gVisor); a kernel log that reads (exit 0) without the banner, which
stops the whole check (the run phase is not on gVisor); or anything else (the container died, or nothing could be
read): this attempt's INFRA. `receipts eval r24` runs `runsc_check.preflight` first, which proves the VM's runsc is
gVisor before any instance runs.
"""

from __future__ import annotations

import json
import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import Field, model_validator

from receipts.contracts import (
    CaseResult,
    FailureType,
    Frozen,
    Outcome,
    PatchMethod,
    RunnerKind,
    RunResult,
    Sandbox,
    Side,
    SuiteRunner,
    Target,
    Tier,
    Workspace,
    answers,
)
from receipts.eval.hashing import canonical_json, pretty_json, sha256_text, write_text_atomic
from receipts.eval.rows import Row
from receipts.eval.runsc_check import GVISOR_BANNER, ResultLog, RuntimeMismatchError, shows_gvisor
from receipts.eval.specs import (
    RepoSpec,
    changed_files,
    expected_failure_passes,
    harness_test_files,
    run_phase_test_command,
)

R24_FORMAT: Literal["receipts.eval.r24/2"] = "receipts.eval.r24/2"
GOLD_PREFIX = "gold"  # gold targets are "gold~<instance_id>", never an entry's target id
ATTEMPTS = 2  # each in a fresh container
PROBE_TIMEOUT_S = 60
MAX_DROP_PCT = 25  # `receipts eval r24` exits 3 when more than this share is dropped
TAIL = 2000
# A run of the modules the named cases live in (the harness ran the whole suite): 2 s per expected case, at least
# --timeout and at most 2 hours, 4 times the harness's 1800 s. A Django test takes well under a second even under
# gVisor; the rest of the allowance covers the other cases those modules hold, which the harness ran too but SWE-bench's
# lists do not name. The cap bounds what one stuck instance costs: 2 attempts of 2 hours on one worker.
DERIVED_SECONDS_PER_CASE = 2
DERIVED_TIMEOUT_CAP_S = 7200


class R24Verdict(StrEnum):
    KEPT = "kept"
    GOLD_FAILS = "gold_fails"  # an expected case failed: an assertion, or a crash in the gold's own files
    GOLD_ERRORS = "gold_errors"  # an expected case errored: a crash elsewhere, an import, name or fixture error
    GOLD_SKIPS = "gold_skips"  # an expected case was skipped: it never ran, so it never passes
    TIMEOUT = "timeout"  # out of time in both attempts, in fresh containers
    INFRA = "infra"  # the sandbox could not answer: unchecked until a rerun


DROPS = frozenset({R24Verdict.GOLD_FAILS, R24Verdict.GOLD_ERRORS, R24Verdict.GOLD_SKIPS, R24Verdict.TIMEOUT})
DECISIVE = frozenset({R24Verdict.KEPT, R24Verdict.GOLD_FAILS, R24Verdict.GOLD_ERRORS, R24Verdict.GOLD_SKIPS})


class R24Attempt(Frozen):
    gvisor: bool | None = None  # the probe showed the banner; None when it never ran
    outcome: Outcome | None = None  # run_test's answer; None when the sandbox raised or the run never started
    failure_type: str | None = None
    exit_code: int | None = None
    seconds: float = Field(default=0.0, ge=0)
    error: str = ""  # what the sandbox raised, or why the probe or the run did not count
    stdout_tail: str = ""
    stderr_tail: str = ""
    problem: str = ""  # run_test's own word on why it could not read or classify the run (RunResult.problem)
    patch_method: PatchMethod | None = None  # how prepare() applied gold plus the test patch; None: never prepared
    timeout_s: float | None = Field(default=None, gt=0)  # run_test's timeout (run_timeout); None: it never ran
    # The expected cases the run reported as failing as expected (CaseResult.expected_failure), graded as the result's
    # xfail_passes says: passing where swebench 4.1.0 reads XFAIL, else SKIPPED.
    expected_failures: tuple[str, ...] = ()


class R24Result(Frozen):
    format: Literal["receipts.eval.r24/2"] = R24_FORMAT
    run_id: str = Field(min_length=1)
    instance_id: str
    target_id: str
    image: str
    runner: RunnerKind
    tests: tuple[str, ...]  # what run_test ran: the test patch's files, or the named cases' modules
    tests_from: Literal["test_patch", "ids", "none"]
    expected: int = Field(ge=0)  # the cases expect named: FAIL_TO_PASS plus PASS_TO_PASS, as the runner names them
    unnamed: tuple[str, ...] = ()  # SWE-bench ids that name no case the runner reports, left out of expect
    resolved: int = Field(default=0, ge=0)  # Django docstring ids the source named (they are in expect)
    resolver: str = ""  # why the docstring resolver could not answer, when it could not
    verdict: R24Verdict
    attempts: tuple[R24Attempt, ...] = Field(min_length=1)
    diagnosis: str = ""  # for a drop: what the harness gold check (root) says about the same instance
    patch_method: PatchMethod | None = None  # how the last prepared workspace applied gold plus the test patch
    unnameable: bool = False  # no FAIL_TO_PASS id names a case, before or during the run: no rerun can check it
    # swebench 4.1.0 grades an expected failure in this repository as passing (specs.expected_failure_passes), so R24
    # counted each one its runs reported among the expected cases as passing; False: they stayed SKIPPED, a drop.
    xfail_passes: bool = False

    @property
    def kept(self) -> bool:
        return self.verdict is R24Verdict.KEPT


# ---------------------------------------------------------------- SWE-bench ids as the runner names its cases

_DJANGO_ID = re.compile(r"(?P<method>[A-Za-z_]\w*) \((?P<qual>[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)+)\)")


def django_label(swebench_id: str) -> str | None:
    """'test_x (app.tests.Class)' or 'test_x (app.tests.Class.test_x)' -> 'app.tests.Class.test_x'; None for an id
    that is a docstring's first line, which names no label."""
    m = _DJANGO_ID.fullmatch(swebench_id.strip())
    if m is None:
        return None
    method, qual = m.group("method"), m.group("qual")
    return qual if qual.endswith("." + method) else f"{qual}.{method}"


def _pytest_name(swebench_id: str) -> str | None:
    """A pytest id as `expect` takes it, the id itself (a case answers it by its node id, or by the id swebench 4.1.0
    records for it: contracts.answers); None when SWE-bench's log parser cut it at a space in its parameters."""
    path, sep, rest = swebench_id.partition("::")
    if not sep or not path or not rest or swebench_id.count("[") != swebench_id.count("]") or " " in swebench_id:
        return None
    return swebench_id


def case_names(kind: RunnerKind, ids: Iterable[str]) -> tuple[list[str], list[str]]:
    """(the cases these SWE-bench ids name, in the runner's own names; the ids that name none), in order."""
    named: list[str] = []
    unnamed: list[str] = []
    for i in ids:
        if kind is RunnerKind.DJANGO:
            name = django_label(i)
        elif kind is RunnerKind.PYTEST:
            name = _pytest_name(i)
        else:
            name = i if re.fullmatch(r"\w+", i) else None
        if name is None:
            unnamed.append(i)
        elif name not in named:
            named.append(name)
    return named, unnamed


def suite_modules(kind: RunnerKind, names: Iterable[str]) -> list[str]:
    """What the runner runs to reach these cases when the test patch names no test file: a Django label's module
    ('app.tests' for 'app.tests.Class.test_x'), a pytest id's file. Sympy names carry no file, so none."""
    out: list[str] = []
    for n in names:
        if kind is RunnerKind.DJANGO:
            parts = n.split(".")
            module = ".".join(parts[:-2]) if len(parts) >= 3 else None
        elif kind is RunnerKind.PYTEST:
            module = n.partition("::")[0]
        else:
            module = None
        if module and module not in out:
            out.append(module)
    return out


class InstanceNaming(Frozen):
    """The SWE-bench ids of one instance that R24 cannot name before the run. A cut id (a pytest id SWE-bench's log
    parser cut at a space inside its parameters, or a sympy id that is no test name) names no case any run reports, so
    it stays out of `expect`. A Django id that is no label is a docstring's first line, which the resolver names from
    the source during the run when it finds it."""

    instance_id: str
    runner: RunnerKind
    fail_to_pass_named: int = Field(ge=0)  # FAIL_TO_PASS ids named before the run
    cut_fail_to_pass: tuple[str, ...] = ()
    cut_pass_to_pass: tuple[str, ...] = ()
    docstring_fail_to_pass: tuple[str, ...] = ()
    docstring_pass_to_pass: tuple[str, ...] = ()
    # No FAIL_TO_PASS id can be named, before the run or during it: R24 can never check the fix's own tests, and no
    # rerun changes that.
    unnameable: bool

    @model_validator(mode="after")
    def _consistent(self) -> InstanceNaming:
        if self.unnameable != (self.fail_to_pass_named == 0 and not self.docstring_fail_to_pass):
            raise ValueError(f"{self.instance_id}: unnameable says {self.unnameable}, its ids say otherwise")
        return self

    @property
    def listed(self) -> bool:
        ids = (self.cut_fail_to_pass, self.cut_pass_to_pass, self.docstring_fail_to_pass, self.docstring_pass_to_pass)
        return any(ids) or self.unnameable


def naming(row: Row, kind: RunnerKind) -> InstanceNaming:
    f2p, f2p_unnamed = case_names(kind, row.FAIL_TO_PASS)
    p2p_unnamed = case_names(kind, row.PASS_TO_PASS)[1]
    docstrings = kind is RunnerKind.DJANGO
    return InstanceNaming(
        instance_id=row.instance_id,
        runner=kind,
        fail_to_pass_named=len(f2p),
        cut_fail_to_pass=() if docstrings else tuple(f2p_unnamed),
        cut_pass_to_pass=() if docstrings else tuple(p2p_unnamed),
        docstring_fail_to_pass=tuple(f2p_unnamed) if docstrings else (),
        docstring_pass_to_pass=tuple(p2p_unnamed) if docstrings else (),
        unnameable=not f2p and not (docstrings and f2p_unnamed),
    )


def naming_before_run(
    rows: Mapping[str, Row], specs: Mapping[tuple[str, str], RepoSpec], instance_ids: Iterable[str]
) -> tuple[InstanceNaming, ...]:
    """Every instance with an id R24 cannot name before the run starts, in instance order. An id with no row or no
    spec is left to run_check, which refuses it."""
    out = []
    for iid in sorted(set(instance_ids)):
        row = rows.get(iid)
        spec = None if row is None else specs.get((row.repo, row.version))
        if row is None or spec is None:
            continue
        n = naming(row, spec.runner)
        if n.listed:
            out.append(n)
    return tuple(out)


UNNAMEABLE_HINT = (
    "no rerun can change that: only a founder --gold-drop (manifest --gold-drop, one '<instance><TAB><reason>' line "
    "each) or the founder's rule for ids SWE-bench cut at a space can"
)


def naming_lines(namings: Sequence[InstanceNaming]) -> list[str]:
    """What `receipts eval r24` prints before the run: every instance whose ids R24 cannot name yet, and what that
    means for it."""
    if not namings:
        return ["before the run: R24 names every FAIL_TO_PASS and PASS_TO_PASS id of these instances"]
    cut = [n for n in namings if n.cut_fail_to_pass or n.cut_pass_to_pass]
    docs = [n for n in namings if n.docstring_fail_to_pass or n.docstring_pass_to_pass]
    cut_ids = sum(len(n.cut_fail_to_pass) + len(n.cut_pass_to_pass) for n in cut)
    doc_ids = sum(len(n.docstring_fail_to_pass) + len(n.docstring_pass_to_pass) for n in docs)
    out = [
        f"before the run: {len(namings)} instances have SWE-bench ids R24 cannot name yet: {cut_ids} ids on "
        f"{len(cut)} instances that SWE-bench's log parser cut at a space (no run names them), and {doc_ids} Django "
        f"docstring ids on {len(docs)} instances (the resolver names them from the source during the run); every id "
        "is in unnamed.json under --out"
    ]
    out.extend(
        f"  {n.instance_id}: {len(n.cut_fail_to_pass)} FAIL_TO_PASS and {len(n.cut_pass_to_pass)} PASS_TO_PASS ids "
        "cut at a space stay out of expect"
        for n in cut
    )
    out.extend(
        f"  {n.instance_id}: {len(n.docstring_fail_to_pass)} FAIL_TO_PASS and {len(n.docstring_pass_to_pass)} "
        "PASS_TO_PASS ids are a docstring's first line"
        for n in docs
    )
    out.extend(
        f"  {n.instance_id}: none of its FAIL_TO_PASS ids can be named, so it stays unchecked whatever the run does; "
        + UNNAMEABLE_HINT
        for n in namings
        if n.unnameable
    )
    return out


def run_timeout(tests_from: str, expected: int, base: float) -> float:
    """run_test's timeout: `base` (--timeout, the harness's 1800 s) for a run of the test patch's files; for a run of
    the modules the named cases live in, which stands in for the harness's whole suite, DERIVED_SECONDS_PER_CASE per
    expected case, never under `base` and never over DERIVED_TIMEOUT_CAP_S unless `base` is more."""
    if tests_from != "ids":
        return base
    return max(base, min(float(DERIVED_TIMEOUT_CAP_S), float(DERIVED_SECONDS_PER_CASE * expected)))


# Run in the prepared gold workspace (the repository root) with the image's own Python, 3.6 or later: it only reads
# files. Arguments: the docstring first lines wanted, then the test files the run will test. It prints the labels it
# found as '<index of the wanted line>\t<label>' between a header and a trailer that carry the count, so a cut-off
# output is never read as an answer.
DOCSTRINGS = r"""
import ast, json, os, sys

want = json.loads(sys.argv[1])
paths = json.loads(sys.argv[2])
index = {}
for n, d in enumerate(want):
    index.setdefault(d, []).append(n)


def files(path):
    if os.path.isfile(path):
        return [path]
    base = path[:-3] if path.endswith(".py") else path
    out = []
    for dirpath, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if not d.startswith("."))
        out.extend(os.path.join(dirpath, f) for f in sorted(filenames) if f.endswith(".py"))
    return out


def module_of(path):
    rel = path[len("tests/"):] if path.startswith("tests/") else path
    rel = rel[:-3]
    if rel.endswith("/__init__"):
        rel = rel[:-len("/__init__")]
    return rel.replace("/", ".")


def first_line(node):
    doc = ast.get_docstring(node, clean=False)
    if not doc or not doc.strip():
        return None
    return doc.strip().split("\n")[0].strip()


def base_name(node):
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


found = set()
done = set()
for p in paths:
    for path in files(p):
        if path in done:
            continue
        done.add(path)
        try:
            with open(path, "rb") as fh:
                tree = ast.parse(fh.read(), path)
        except Exception:
            continue
        module = module_of(path)
        classes = dict((c.name, c) for c in tree.body if isinstance(c, ast.ClassDef))

        def methods(cls, chain):
            out = {}
            for base in reversed(cls.bases):
                name = base_name(base)
                if isinstance(base, ast.Name) and name in classes and name not in chain:
                    out.update(methods(classes[name], chain + (name,)))
            for node in cls.body:
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out[node.name] = node
            return out

        def loaded(cls, chain):
            if cls.name.endswith("Mixin"):
                return False
            for base in cls.bases:
                name = base_name(base)
                if name is None or name == "object":
                    continue
                if isinstance(base, ast.Name) and name in classes:
                    if name not in chain and loaded(classes[name], chain + (name,)):
                        return True
                    continue
                return True
            return False

        for cls in classes.values():
            if not loaded(cls, (cls.name,)):
                continue
            for name, fn in methods(cls, (cls.name,)).items():
                line = first_line(fn) if name.startswith("test") else None
                if line is not None and line in index:
                    for n in index[line]:
                        found.add((n, module + "." + cls.name + "." + name))
print("R24-NAMES %d" % len(found))
for n, label in sorted(found):
    print("%d\t%s" % (n, label))
print("R24-END %d" % len(found))
"""
DOCSTRINGS_TIMEOUT_S = 300
_LABEL = re.compile(r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*){2,}")


def django_test_files(tests: Iterable[str]) -> list[str]:
    """The files a Django run tests: the test patch's paths as they are, and a module label as its file under tests/."""
    out: list[str] = []
    for t in tests:
        path = t if t.endswith(".py") else "tests/" + t.replace(".", "/") + ".py"
        if path not in out:
            out.append(path)
    return out


def parse_docstring_labels(output: str, wanted: Sequence[str]) -> dict[str, tuple[str, ...]] | None:
    """The resolver's answer, wanted line -> labels; None when the output is cut, malformed or names a bad label."""
    lines = output.splitlines()
    heads = [n for n, line in enumerate(lines) if re.fullmatch(r"R24-NAMES \d+", line)]
    if not heads:
        return None
    start = heads[-1]
    count = int(lines[start].split()[1])
    if len(lines) < start + count + 2 or lines[start + count + 1] != f"R24-END {count}":
        return None
    out: dict[str, list[str]] = {}
    for line in lines[start + 1 : start + 1 + count]:
        m = re.fullmatch(r"(\d+)\t(\S+)", line)
        if m is None or int(m.group(1)) >= len(wanted) or _LABEL.fullmatch(m.group(2)) is None:
            return None
        labels = out.setdefault(wanted[int(m.group(1))], [])
        if m.group(2) not in labels:
            labels.append(m.group(2))
    return {k: tuple(v) for k, v in out.items()}


def resolve_docstrings(
    sandbox: Sandbox, workspace: Workspace, wanted: Sequence[str], files: Sequence[str]
) -> tuple[dict[str, tuple[str, ...]] | None, str]:
    """Labels for SWE-bench's docstring-named Django ids, read from the source in the workspace; (None, why) when the
    resolver could not answer."""
    try:
        r = sandbox.run(
            workspace,
            ["python", "-c", DOCSTRINGS, json.dumps(list(wanted)), json.dumps(list(files))],
            timeout_s=DOCSTRINGS_TIMEOUT_S,
            fresh=True,
        )
    except Exception as e:  # the sandbox's failure costs the docstring names, never the attempt
        return None, f"the docstring resolver raised {type(e).__name__}: {e}"[-300:]
    if r.outcome is not Outcome.PASS:
        tail = (r.stderr_tail or r.stdout_tail).strip()[-300:]
        return None, f"the docstring resolver did not run ({r.outcome.value}, exit {r.exit_code}): {tail}"
    parsed = parse_docstring_labels(r.stdout_tail, wanted)
    if parsed is None:
        return None, "the docstring resolver's output was cut off or malformed"
    return parsed, ""


# ---------------------------------------------------------------- expected failures, graded as the harness grades them

XFAIL_AS_PASS = "failed as expected; graded as passing, as swebench 4.1.0 grades XFAIL in this repository"


class GoldRunner:
    """The instance's runner as R24's gold check reads its cases (contracts.SuiteRunner). A case the runner reports as
    an expected failure that failed as expected (CaseResult.expected_failure, outcome SKIPPED) becomes a passing case
    when `xfail_passes` (swebench 4.1.0 grades XFAIL as passing in the instance's repository), so the sandbox's own
    expect rule counts it as run and passed; otherwise it stays as reported. A plain skip is never touched. Every
    expected failure the run reported is kept in `seen`, which a copy bound to another root shares.

    Everything else is the wrapped runner's: its kind, its command, and any attribute a sandbox reads from the runner
    it is given (the attack lane's DockerSandbox reads `stream`, and binds a runner to the workspace root with
    `with_root`, keeping the bound runner only when it is of the same type)."""

    def __init__(self, runner: SuiteRunner, *, xfail_passes: bool, seen: list[CaseResult] | None = None) -> None:
        self._runner = runner
        self.kind: RunnerKind = runner.kind
        self.xfail_passes = xfail_passes
        self.seen: list[CaseResult] = [] if seen is None else seen

    def __getattr__(self, name: str) -> object:
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._runner, name)

    def with_root(self, root: str) -> GoldRunner:
        bind = getattr(self._runner, "with_root", None)
        runner = bind(root) if callable(bind) else self._runner
        return GoldRunner(runner, xfail_passes=self.xfail_passes, seen=self.seen)

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return self._runner.command(target, tests, report_path=report_path)

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        out: list[CaseResult] = []
        for case in self._runner.cases(output, report, changed_files):
            if case.expected_failure:
                self.seen.append(case)
                if self.xfail_passes:
                    message = f"{XFAIL_AS_PASS}: {case.message}" if case.message else XFAIL_AS_PASS
                    case = CaseResult(name=case.name, outcome=Outcome.PASS, message=message, file=case.file)
            out.append(case)
        return tuple(out)


def expected_failures(seen: Iterable[CaseResult], expect: Sequence[str], repo: str) -> tuple[str, ...]:
    """The expected cases a run reported as expected failures, by case id (path::name where a case names its file).
    A case is an expected one as the sandbox's own expect rule matches it (contracts.answers, in `repo`)."""
    out: list[str] = []
    for case in seen:
        name = f"{case.file}::{case.name}" if case.file is not None and case.file != case.name else case.name
        if name not in out and any(answers(case, w, repo=repo) for w in expect):
            out.append(name)
    return tuple(out)


# ---------------------------------------------------------------- one instance


def gold_target(row: Row, image: str, spec: RepoSpec) -> Target:
    """The instance as a target for the gold check: its base, image, runner and test command, which carries the
    environment the harness's eval script exports (`specs.run_phase_test_command`). The gold patch rides in `patch`
    only because a Target needs one; prepare() gets gold plus the test patch explicitly."""
    if (spec.repo, spec.version) != (row.repo, row.version):
        raise ValueError(f"{row.instance_id}: spec {spec.repo} {spec.version} is not the row's")
    return Target(
        id=f"{GOLD_PREFIX}~{row.instance_id}",
        tier=Tier.A,
        repo=row.repo,
        base_sha=row.base_commit,
        patch_sha=sha256_text(row.patch),
        patch=row.patch,
        issue=row.problem_statement,
        image=image,
        test_runner=spec.runner,
        test_command=run_phase_test_command(spec),
    )


def gold_and_tests(row: Row) -> str:
    """One unified diff: the gold patch, then the test patch, each ending in a newline."""
    parts = [p if p.endswith("\n") else p + "\n" for p in (row.patch, row.test_patch)]
    return "".join(parts)


def _attempt_from(
    result: RunResult,
    gvisor: bool,
    *,
    patch_method: PatchMethod | None,
    timeout_s: float,
    expected_failures: tuple[str, ...] = (),
) -> R24Attempt:
    return R24Attempt(
        gvisor=gvisor,
        outcome=result.outcome,
        failure_type=None if result.failure_type is None else result.failure_type.value,
        exit_code=result.exit_code,
        seconds=round(result.duration_s, 3),
        stdout_tail=result.stdout_tail[-TAIL:],
        stderr_tail=result.stderr_tail[-TAIL:],
        problem=result.problem[-TAIL:],
        patch_method=patch_method,
        timeout_s=timeout_s,
        expected_failures=expected_failures,
    )


def _probe(sandbox: Sandbox, workspace: Workspace, target_id: str) -> tuple[bool | None, str]:
    """True when gVisor answered; None (with why) when the probe read nothing. A log that reads and is not gVisor's
    stops everything: the run phase is on the wrong kernel."""
    probe = sandbox.run(workspace, ["dmesg"], timeout_s=PROBE_TIMEOUT_S, fresh=True)
    if probe.outcome is Outcome.PASS and shows_gvisor(probe.stdout_tail):
        return True, ""
    if probe.outcome is Outcome.PASS:
        raise RuntimeMismatchError(f"{target_id}: dmesg read a kernel log without {GVISOR_BANNER!r}: not gVisor")
    return None, (
        f"the kernel probe read no kernel log ({probe.outcome.value}, exit {probe.exit_code}): the container died, "
        "or the run phase is not on gVisor"
    )


def _infra_only(attempt: R24Attempt) -> bool:
    return attempt.outcome is Outcome.ERROR and attempt.failure_type in (None, FailureType.INFRA.value)


def verdict_of(attempts: Sequence[R24Attempt]) -> R24Verdict:
    """The instance's verdict from its attempts: the last decides, except that TIMEOUT needs every attempt (both) to
    have run out of time. A run counts only when the probe showed gVisor."""
    last = attempts[-1]
    if last.outcome is not None and last.gvisor is True:
        if last.outcome is Outcome.PASS:
            return R24Verdict.KEPT
        if last.outcome is Outcome.FAIL:
            return R24Verdict.GOLD_FAILS
        if last.outcome is Outcome.SKIPPED:
            return R24Verdict.GOLD_SKIPS
        if last.outcome is Outcome.ERROR and not _infra_only(last):
            return R24Verdict.GOLD_ERRORS
    timed_out = [a.outcome is Outcome.TIMEOUT and a.gvisor is True for a in attempts]
    if len(attempts) >= ATTEMPTS and all(timed_out):
        return R24Verdict.TIMEOUT
    return R24Verdict.INFRA  # raised, probe read nothing, ERROR with no type or INFRA, or a lone timeout


def _unchecked(row: Row, image: str, spec: RepoSpec, *, run_id: str, why: str, **kw: object) -> R24Result:
    """An instance R24 could not check, with the reason: INFRA, so a rerun or a founder drop can settle it."""
    return R24Result.model_validate(
        {
            "run_id": run_id,
            "instance_id": row.instance_id,
            "target_id": f"{GOLD_PREFIX}~{row.instance_id}",
            "image": image,
            "runner": spec.runner,
            "tests": (),
            "tests_from": "none",
            "expected": 0,
            "verdict": R24Verdict.INFRA,
            "attempts": (R24Attempt(error=why[-TAIL:]),),
            "xfail_passes": expected_failure_passes(row.repo),
            **kw,
        }
    )


def check_instance(
    row: Row,
    image: str,
    spec: RepoSpec,
    sandbox: Sandbox,
    runners: Mapping[RunnerKind, SuiteRunner],
    *,
    run_id: str,
    timeout_s: float = 1800,
    diagnose: Callable[[str], str] | None = None,
    clock: Callable[[], float] = time.monotonic,
) -> R24Result:
    target = gold_target(row, image, spec)
    runner = runners.get(target.test_runner)
    if runner is None or runner.kind is not target.test_runner:
        raise ValueError(f"no {target.test_runner} runner is wired")
    kind = target.test_runner
    f2p, f2p_unnamed = case_names(kind, row.FAIL_TO_PASS)
    p2p, p2p_unnamed = case_names(kind, row.PASS_TO_PASS)
    names = naming(row, kind)
    docstrings = list(dict.fromkeys((*names.docstring_fail_to_pass, *names.docstring_pass_to_pass)))
    if names.unnameable:
        return _unchecked(
            row,
            image,
            spec,
            run_id=run_id,
            unnamed=(*f2p_unnamed, *p2p_unnamed),
            unnameable=True,
            why=(
                f"none of its {len(row.FAIL_TO_PASS)} FAIL_TO_PASS ids names a case the {kind.value} runner reports "
                f"(e.g. {row.FAIL_TO_PASS[0]!r}): R24 cannot check the fix's own tests"
            ),
        )
    tests = tuple(harness_test_files(row.test_patch))
    tests_from: Literal["test_patch", "ids"] = "test_patch"
    if not tests:
        tests, tests_from = tuple(suite_modules(kind, (*f2p, *p2p))), "ids"
    if not tests:
        return _unchecked(
            row,
            image,
            spec,
            run_id=run_id,
            unnamed=(*f2p_unnamed, *p2p_unnamed),
            expected=len(set(f2p) | set(p2p)),
            why="the test patch touches no test file, and no FAIL_TO_PASS or PASS_TO_PASS id names a test module",
        )
    changed = changed_files(row.patch)
    xfail_passes = expected_failure_passes(row.repo)  # how swebench 4.1.0 grades an expected failure here
    resolution: dict[str, tuple[str, ...]] | None = None if docstrings else {}
    resolver = ""

    def named(ids_named: Sequence[str], ids_unnamed: Sequence[str]) -> tuple[list[str], list[str]]:
        found = resolution or {}
        labels = [*ids_named, *(label for i in ids_unnamed for label in found.get(i, ()))]
        return list(dict.fromkeys(labels)), [i for i in ids_unnamed if i not in found]

    expect: list[str] = []
    unnamed: tuple[str, ...] = ()
    attempts: list[R24Attempt] = []
    unnameable = False
    for _ in range(ATTEMPTS):
        start = clock()
        try:
            workspace = sandbox.prepare(target, Side.GOLD, patch=gold_and_tests(row))
        except Exception as e:  # the sandbox's own failure: recorded, never a verdict on the instance
            attempts.append(R24Attempt(error=f"prepare: {type(e).__name__}: {e}"[-TAIL:], seconds=clock() - start))
            continue
        method = workspace.patch_method  # how gold plus the test patch applied (contracts.PatchMethod, REG-569)
        try:
            gvisor, why = _probe(sandbox, workspace, target.id)
            if gvisor is None:
                attempts.append(R24Attempt(error=why, seconds=round(clock() - start, 3), patch_method=method))
                continue
            if resolution is None:  # a failed resolver is tried again in the next attempt's workspace
                resolution, resolver = resolve_docstrings(sandbox, workspace, docstrings, django_test_files(tests))
            f2p_expect, f2p_left = named(f2p, f2p_unnamed)
            p2p_expect, p2p_left = named(p2p, p2p_unnamed)
            expect, unnamed = list(dict.fromkeys((*f2p_expect, *p2p_expect))), (*f2p_left, *p2p_left)
            if not f2p_expect:
                attempts.append(
                    R24Attempt(
                        gvisor=True,
                        error=(
                            f"none of its {len(row.FAIL_TO_PASS)} FAIL_TO_PASS ids names a case the django runner "
                            f"reports, even from the source ({resolver or 'the resolver found none'}): R24 cannot "
                            "check the fix's own tests"
                        )[-TAIL:],
                        seconds=round(clock() - start, 3),
                        patch_method=method,
                    )
                )
                if resolution is not None:
                    unnameable = True
                    break  # the source was read and names none: another attempt, or a rerun, finds the same
                continue
            limit = run_timeout(tests_from, len(expect), timeout_s)
            graded = GoldRunner(runner, xfail_passes=xfail_passes)  # a fresh record of expected failures per run
            result = sandbox.run_test(
                workspace,
                tests,
                runner=graded,
                changed_files=changed,
                timeout_s=limit,
                seed=0,
                fresh=True,
                expect=expect,
            )
            attempts.append(
                _attempt_from(
                    result,
                    gvisor,
                    patch_method=method,
                    timeout_s=limit,
                    expected_failures=expected_failures(graded.seen, expect, row.repo),
                )
            )
        except RuntimeMismatchError:
            raise
        except Exception as e:
            attempts.append(
                R24Attempt(error=f"run: {type(e).__name__}: {e}"[-TAIL:], seconds=clock() - start, patch_method=method)
            )
            continue
        finally:
            sandbox.close(workspace)
        if verdict_of(attempts[-1:]) in DECISIVE:
            break  # PASS, FAIL, SKIPPED or a typed ERROR; INFRA and TIMEOUT get one more fresh try
    if not attempts[-1].gvisor:  # no workspace ever got as far as naming the cases
        f2p_expect, f2p_left = named(f2p, f2p_unnamed)
        p2p_expect, p2p_left = named(p2p, p2p_unnamed)
        expect, unnamed = list(dict.fromkeys((*f2p_expect, *p2p_expect))), (*f2p_left, *p2p_left)
    verdict = verdict_of(attempts)
    diagnosis = ""
    if verdict in DROPS and diagnose is not None:
        try:
            diagnosis = diagnose(row.instance_id)
        except Exception as e:
            diagnosis = f"the harness diagnosis failed: {type(e).__name__}: {e}"[-TAIL:]
    return R24Result(
        run_id=run_id,
        instance_id=row.instance_id,
        target_id=target.id,
        image=image,
        runner=kind,
        tests=tests,
        tests_from=tests_from,
        expected=len(expect),
        unnamed=unnamed,
        resolved=len(docstrings) - len([i for i in docstrings if i in unnamed]) if docstrings else 0,
        resolver=resolver,
        verdict=verdict,
        attempts=tuple(attempts),
        diagnosis=diagnosis,
        patch_method=next((a.patch_method for a in reversed(attempts) if a.patch_method is not None), None),
        unnameable=unnameable,
        xfail_passes=xfail_passes,
    )


# ---------------------------------------------------------------- the pool


class R24Report(Frozen):
    run_id: str
    instances: int
    verdicts: dict[str, int]
    dropped: dict[str, str]  # instance -> verdict
    unchecked: tuple[str, ...]
    probe_failed: tuple[str, ...] = ()  # unchecked because every attempt's kernel probe read nothing
    unnamed: dict[str, int] = Field(default_factory=dict)  # instance -> SWE-bench ids left out of expect
    unnameable: tuple[str, ...] = ()  # unchecked whatever a run does: no FAIL_TO_PASS id names a case
    patch_methods: dict[str, int] = Field(default_factory=dict)  # how prepare() applied gold plus the test patch
    patch_fallbacks: dict[str, str] = Field(default_factory=dict)  # instance -> method, where `git apply` alone failed
    # instance -> expected cases its last run reported as failing as expected, which R24 counted as passing (swebench
    # 4.1.0 grades XFAIL as passing in its repository)
    xfail_counted: dict[str, int] = Field(default_factory=dict)
    # instance -> the same, in a repository whose swebench parser writes no XFAIL (django, seaborn, sympy): SKIPPED
    xfail_skipped: dict[str, int] = Field(default_factory=dict)
    max_drop_pct: int

    @property
    def drop_pct_over_limit(self) -> bool:
        return len(self.dropped) * 100 > self.max_drop_pct * self.instances


SandboxFactory = Callable[[], Sandbox]
RunnersFactory = Callable[[], Mapping[RunnerKind, SuiteRunner]]


def run_check(
    rows: Mapping[str, Row],
    images: Mapping[str, str],
    specs: Mapping[tuple[str, str], RepoSpec],
    instance_ids: Sequence[str],
    sandbox: SandboxFactory,
    runners: RunnersFactory,
    out_dir: Path,
    *,
    run_id: str,
    timeout_s: float = 1800,
    workers: int = 16,
    max_drop_pct: int = MAX_DROP_PCT,
    diagnose: Callable[[str], str] | None = None,
) -> R24Report:
    """Check every instance. Before any instance runs, unnamed.json lists the ids R24 cannot name yet; results.jsonl
    gets each result as it lands (a stop keeps what finished), then dropped.tsv and report.json. Each worker thread
    makes its own sandbox and runners. An instance whose check raises anything but a runtime mismatch is that
    instance's INFRA; a runtime mismatch stops everything still queued."""
    ids = sorted(set(instance_ids))
    for iid in ids:
        if iid not in rows or iid not in images:
            raise ValueError(f"{iid}: no row or no pinned image")
        if (rows[iid].repo, rows[iid].version) not in specs:
            raise ValueError(f"{iid}: no SWE-bench spec for {rows[iid].repo} {rows[iid].version}")
    wired = runners()
    for kind in sorted({specs[(rows[i].repo, rows[i].version)].runner for i in ids}):
        if kind not in wired or wired[kind].kind is not kind:
            raise ValueError(f"no {kind.value} runner is wired")
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ResultLog(out_dir / "results.jsonl")
    if log.lines():
        raise ValueError(f"{log.path} already has results; a rerun goes to a new --out and `manifest --gold-rerun`")
    namings = naming_before_run(rows, specs, ids)  # before any instance runs
    write_text_atomic(out_dir / "unnamed.json", pretty_json([n.model_dump(mode="json") for n in namings]))
    local = threading.local()

    def one(iid: str) -> R24Result:
        row = rows[iid]
        spec = specs[(row.repo, row.version)]
        try:
            if not hasattr(local, "sandbox"):
                local.sandbox, local.runners = sandbox(), runners()
            return check_instance(
                row,
                images[iid],
                spec,
                local.sandbox,
                local.runners,
                run_id=run_id,
                timeout_s=timeout_s,
                diagnose=diagnose,
            )
        except RuntimeMismatchError:
            raise
        except Exception as e:  # this instance's infrastructure: recorded, and every other instance still runs
            return _unchecked(row, images[iid], spec, run_id=run_id, why=f"{type(e).__name__}: {e}")

    results: list[R24Result] = []
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(one, iid) for iid in ids]
        for future in as_completed(futures):
            r = future.result()
            log.append(r)
            results.append(r)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    results.sort(key=lambda r: r.instance_id)
    lines = []
    for r in results:
        if r.kept or r.verdict is R24Verdict.INFRA:
            continue
        last = r.attempts[-1]
        why = (last.failure_type or "") + " " + (last.stdout_tail or last.stderr_tail)[-200:].replace("\n", " ")
        lines.append(f"{r.instance_id}\t{r.verdict.value}\t{why.strip()}\t{r.diagnosis}\n")
    write_text_atomic(out_dir / "dropped.tsv", "".join(lines))
    report = R24Report(
        run_id=run_id,
        instances=len(results),
        verdicts=dict(sorted(Counter(r.verdict.value for r in results).items())),
        dropped={r.instance_id: r.verdict.value for r in results if r.verdict in DROPS},
        unchecked=tuple(r.instance_id for r in results if r.verdict is R24Verdict.INFRA),
        probe_failed=tuple(
            r.instance_id
            for r in results
            if r.verdict is R24Verdict.INFRA and all(a.error.startswith("the kernel probe") for a in r.attempts)
        ),
        unnamed={r.instance_id: len(r.unnamed) for r in results if r.unnamed},
        unnameable=tuple(r.instance_id for r in results if r.verdict is R24Verdict.INFRA and r.unnameable),
        patch_methods=dict(
            sorted(Counter("none" if r.patch_method is None else r.patch_method.value for r in results).items())
        ),
        patch_fallbacks={
            r.instance_id: r.patch_method.value
            for r in results
            if r.patch_method is not None and r.patch_method is not PatchMethod.GIT_APPLY
        },
        xfail_counted={
            r.instance_id: len(r.attempts[-1].expected_failures)
            for r in results
            if r.xfail_passes and r.attempts[-1].expected_failures
        },
        xfail_skipped={
            r.instance_id: len(r.attempts[-1].expected_failures)
            for r in results
            if not r.xfail_passes and r.attempts[-1].expected_failures
        },
        max_drop_pct=max_drop_pct,
    )
    write_text_atomic(out_dir / "report.json", json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return report


# ---------------------------------------------------------------- feeding R24 back into the manifest


class R24ResultsError(ValueError):
    """Results that cannot be fed back into the manifest."""


class GoldDrop(Frozen):
    """An instance the founder drops by hand after its gold could not be checked (REG-565: back to the founder)."""

    instance_id: str
    reason: str = Field(min_length=1)


class R24Input(Frozen):
    role: Literal["base", "rerun", "drop"]
    sha256: str
    run_ids: tuple[str, ...]
    lines: int


class R24Merge(Frozen):
    verdicts: dict[str, bool | None]  # instance -> kept (True), dropped (False), unchecked (None)
    reasons: dict[str, str]  # instance -> verdict value, or "founder: <reason>"
    inputs: tuple[R24Input, ...]

    @property
    def sha256(self) -> str:
        return sha256_text(canonical_json({"verdicts": self.verdicts, "reasons": self.reasons}))


def _parse(text: str, pool: set[str], images: Mapping[str, str], what: str) -> dict[str, R24Result]:
    out: dict[str, R24Result] = {}
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = R24Result.model_validate_json(line)
        except ValueError as e:
            raise R24ResultsError(f"{what} line {n}: not a {R24_FORMAT} result: {str(e)[:300]}") from e
        if r.instance_id in out:
            raise R24ResultsError(f"{what} line {n}: {r.instance_id} appears twice")
        if r.instance_id not in pool:
            raise R24ResultsError(f"{what} line {n}: {r.instance_id} is not on the pool")
        if r.image != images.get(r.instance_id):
            raise R24ResultsError(
                f"{what} line {n}: {r.instance_id} ran {r.image!r}, not its pin in eval/images.json "
                f"({images.get(r.instance_id)!r})"
            )
        if r.target_id != f"{GOLD_PREFIX}~{r.instance_id}":
            raise R24ResultsError(f"{what} line {n}: {r.instance_id} names target {r.target_id!r}")
        if r.verdict is not verdict_of(r.attempts):
            raise R24ResultsError(
                f"{what} line {n}: {r.instance_id} says {r.verdict.value}, its runs say {verdict_of(r.attempts).value}"
            )
        out[r.instance_id] = r
    return out


def merge(
    base: str,
    pool_ids: Iterable[str],
    images: Mapping[str, str],
    *,
    reruns: Sequence[str] = (),
    drops: Sequence[GoldDrop] = (),
) -> R24Merge:
    """The base results.jsonl, then each rerun's lines, then the founder's drops. Every result must have run its
    instance's pinned image (`images`, from eval/images.json). A rerun may replace only an INFRA verdict, and a drop
    may land only on an instance still unchecked: a kept or dropped instance never changes. Every pool instance must
    have a verdict, and nothing off the pool may."""
    pool = set(pool_ids)
    merged = _parse(base, pool, images, "base")
    missing = sorted(pool - set(merged))
    if missing:
        raise R24ResultsError(f"{len(missing)} pool instances have no R24 verdict, e.g. {missing[:3]}")
    inputs = [
        R24Input(
            role="base",
            sha256=sha256_text(base),
            run_ids=tuple(sorted({r.run_id for r in merged.values()})),
            lines=len(merged),
        )
    ]
    for k, text in enumerate(reruns, 1):
        rerun = _parse(text, pool, images, f"rerun {k}")
        for iid, r in rerun.items():
            if merged[iid].verdict is not R24Verdict.INFRA:
                raise R24ResultsError(
                    f"rerun {k}: {iid} already has a verdict ({merged[iid].verdict.value}); only INFRA is rerun"
                )
            merged[iid] = r
        inputs.append(
            R24Input(
                role="rerun",
                sha256=sha256_text(text),
                run_ids=tuple(sorted({r.run_id for r in rerun.values()})),
                lines=len(rerun),
            )
        )
    reasons = {iid: r.verdict.value for iid, r in merged.items()}
    verdicts: dict[str, bool | None] = {
        iid: (True if r.kept else (None if r.verdict is R24Verdict.INFRA else False)) for iid, r in merged.items()
    }
    seen: set[str] = set()
    for d in drops:
        if d.instance_id not in pool:
            raise R24ResultsError(f"drop: {d.instance_id} is not on the pool")
        if d.instance_id in seen:
            raise R24ResultsError(f"drop: {d.instance_id} appears twice")
        seen.add(d.instance_id)
        if verdicts[d.instance_id] is not None:
            raise R24ResultsError(f"drop: {d.instance_id} has a verdict ({reasons[d.instance_id]}); only unchecked")
        verdicts[d.instance_id] = False
        reasons[d.instance_id] = f"founder: {d.reason}"
    if drops:
        text = "".join(d.model_dump_json() + "\n" for d in drops)
        inputs.append(R24Input(role="drop", sha256=sha256_text(text), run_ids=(), lines=len(drops)))
    return R24Merge(
        verdicts=dict(sorted(verdicts.items())), reasons=dict(sorted(reasons.items())), inputs=tuple(inputs)
    )


def parse_drops(text: str) -> list[GoldDrop]:
    """The founder's drop file: one `<instance_id><TAB><reason>` per line; blank lines and # comments are skipped."""
    out = []
    for n, line in enumerate(text.splitlines(), 1):
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        iid, sep, reason = line.partition("\t")
        if not sep or not reason.strip():
            raise R24ResultsError(f"drop file line {n}: want '<instance_id><TAB><reason>'")
        out.append(GoldDrop(instance_id=iid.strip(), reason=reason.strip()))
    return out
