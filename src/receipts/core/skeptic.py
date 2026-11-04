"""SK1, the skeptic's code gates: every candidate repro meets them before any judgment, and code beats judgment (spec
hard rule 4). In order:

(b) the fake-test rules (receipts.core.fake_tests), static checks over the repro's files, run first because they cost
    nothing. A repro that breaks one is refuted and never reaches a sandbox. So is one that answers for another claim
    or lens than its attack's, names a case none of its files defines, or defines no test case to run. A repro that
    names its cases (Repro.cases, contract v3) is checked on those cases and what they use, never on the other tests
    beside them (AT3 hands over an existing test file whole, naming the one case that regressed). A test file that is
    the repository's own, the base tree's content at its own path, is no violation by itself: when that is all that
    stands between a repro and the rules (writing back a test file the agent edited, or module-level code the named
    cases do not use), one command in the base workspace compares the file with the base tree's, and only the
    repository's own content is let through. Before the rules, SK2's code check (its quote row): a fix-validity or
    behavior repro quotes the sentence it enforces (Repro.quote, contract v3.2). The quote must be a verbatim part of
    the issue (Target.issue), or of the pull request's title or body only when the target has no issue, never of its
    commit messages or the agent's session summary; it is matched on word boundaries (intake's find_quote, the check
    a claim's quote meets) once runs of whitespace on both sides are made one space. A quote that is missing, under 3
    words or not found so refutes the repro. A regression repro's oracle is base, so it quotes nothing.
(a) the failure-type gate: only a failure in contracts.COUNTED_FAILURES (an assertion, or a crash whose top non-test
    frame is in a changed file) counts. A failure of any other type (import, name, fixture, a test timeout, a crash
    elsewhere) is refuted, whether the run reports it as FAIL or as ERROR with its type; an ERROR typed INFRA or with
    no type is a run that could not be read.
(c) the rerun policy: the candidate runs 5 times on the patch and 5 on base, each in a fresh container with its own
    seed (0 to 4), through Sandbox.run_test with the target's SuiteRunner, pointed at exactly the repro's cases and
    with `expect` set to them, so a case that never ran is an error and never a pass. With no Repro.cases, the cases
    are every case the repro's test files define (the integrator's case namer, receipts.core.cases by default).

(d) the surrogate oracle (SK3), for a fix-validity or behavior candidate past its patch and base runs: with no import
    error on base, it runs 5 times more (fresh, seeds 0 to 4) on each plausible surrogate fix of its claim in turn,
    each in a workspace prepared from base with that surrogate's patch, until one passes it 5 of 5. A surrogate whose
    run has no result (unread, timed out, no workspace) is passed over for the next one. The surrogates are built
    once per claim (contracts.SurrogateBuilder), when the claim's first such candidate gets here, and shared by the
    claim's other candidates and by every arm; what building them spent goes on that first candidate's finding only,
    so a ledger rollup counts it. Surrogates have their own budget per target (contracts.
    SURROGATE_TARGET_CAP_MICRO_USD, `surrogate_budget`), outside every arm's per-target cap, so an arm over its own
    cap still gets them and every arm judges a candidate by the same ones. A build makes only as many as what is
    left of that budget covers at contracts.SURROGATE_CAP_MICRO_USD each, at most k, so the budget is crossed by one
    surrogate's spend at most; with room for none, the candidate is s_sat_inconclusive. A claim built with fewer is
    never built again, and surrogates already built are used at no cost.

The first patch run is the screen: the pipeline escalates to a lens's next attacker only when no candidate failed it on
a counted failure. A candidate the screen stops gets its status from that run alone; the rest get the full reruns.

A repro is judged one test case at a time. split() turns a repro whose files define several cases into one repro per
case, in the order the cases are named (Repro.cases, else the case namer's order), each with that case alone as its
cases and its id the repro's id plus '.c<n>'; the pipeline screens each as a candidate of its own, so one case that
catches is never lost inside a repro whose other cases fail on gold, and gold replays each case alone. A repro of one
case is left as it is. At most MAX_CASES_PER_REPRO cases of one repro are judged; the rest are returned for the
pipeline to list as unchecked (budget).

Statuses:
- blocker: a regression catch that fails on the patch in all 5 runs, on a counted failure, and passes on base in all
  5; or a fix-validity (or adds-behavior) catch that fails on the patch in all 5 runs, on a counted failure, and
  passes on a plausible surrogate fix of its claim in all 5 (its runs on that surrogate go on the finding,
  Side.SURROGATE). Either is built only if contracts.Finding accepts it (a refusal there makes it suspected, with the
  reason). Tier A's gold patch is used only at scoring, never here;
- s_sat_inconclusive: a fix-validity (or adds-behavior) catch that fails on the patch in all 5 runs with no plausible
  surrogate to judge it by: no builder is wired, the target's surrogate budget had no room left for one, none was
  built, or none of those built is plausible (the detail gives each one's reason);
- possibly_toxic: a fix-validity (or adds-behavior) catch that fails on the patch in all 5 runs and fails, in all 5,
  on every plausible surrogate too: the repro may demand more than the issue does;
- flaky: mixed results on a side, a counted failure in some patch runs and not others, or base passing in some runs,
  or a fix-validity catch that passes on a surrogate in some runs and not others while no surrogate passes it 5 of 5;
- refuted: a code gate rejects it, a regression candidate fails on base too, or a fix-validity candidate fails on base
  with an import error (base lacks what the repro imports, so it shows no bug of base's);
- not_reproduced: it passes, or is skipped, on the patch;
- unchecked: it could not be run to a result: no workspace (no_env), a run that could not be read or a sandbox error
  (runtime), or a run killed for time (timeout); on surrogates, only when no surrogate passes it 5 of 5 and one at
  least had a run with no result.

Each finding carries every run, the claim's quote, a replay key when a workspace was built, why it has its status in
Finding.detail (with a run's own problem when it could not be read), and the skeptic's seconds as a code-only stage
cost, and the costs of building its claim's surrogates when it built them. Finding.refutation is left for SK2's
judgment, a different model's refutation (receipts.core.refuter), which the pipeline runs on each blocker the skeptic
returns, before it is recorded, and which can only make it suspected; the pipeline also holds each claim to its top-n
candidates rerun per arm (those that failed their first patch run). The skeptic calls no model itself; a surrogate
builder may. It keeps every workspace it prepared, with how the patch applied (REG-569), for the target's summary.
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Final

from pydantic import ValidationError

from receipts import __version__
from receipts.contracts import (
    BLOCKER_RUNS,
    COUNTED_FAILURES,
    SURROGATE_CAP_MICRO_USD,
    SURROGATE_TARGET_CAP_MICRO_USD,
    SURROGATES_PER_CLAIM,
    Attack,
    CatchKind,
    Claim,
    FailureType,
    Finding,
    Outcome,
    ReplayKey,
    Repro,
    RunResult,
    Sandbox,
    Side,
    StageCost,
    Status,
    SuiteRunner,
    Surrogate,
    SurrogateBuilder,
    Target,
    UncheckedReason,
    Workspace,
)
from receipts.core.cases import (
    CaseNamer,
    defined_cases,
    expected_cases,
    is_test_module,
    place_cases,
    repro_test_files,
)
from receipts.core.diff import changed_files, patch_facts
from receipts.core.fake_tests import WHOLE_FILE, Scope, Violation, check_repro_files
from receipts.core.intake import MIN_QUOTE_WORDS, count_words, find_quote

log = logging.getLogger(__name__)

SKEPTIC_STAGE: Final = "skeptic"
DEFAULT_RUN_TIMEOUT_S: Final = 300.0
BASE_CHECK_TIMEOUT_S: Final = 60.0
MAX_CASES_PER_REPRO: Final = 3  # the cases of one repro that split() hands on to be judged
# Whether a file in the base workspace is the given text, read the way the attack lane reads the base tree (UTF-8,
# bad bytes replaced, universal newlines): exit 0 when its SHA-256 matches, 3 when it differs, 4 when it cannot be
# read. Python in isolated mode, so nothing in the repository (a hashlib.py at its root) is imported.
SAME_AS_BASE: Final = (
    "import hashlib, sys\n"
    "try:\n"
    "    with open(sys.argv[1], encoding='utf-8', errors='replace') as f:\n"
    "        text = f.read()\n"
    "except OSError:\n"
    "    sys.exit(4)\n"
    "sys.exit(0 if hashlib.sha256(text.encode('utf-8')).hexdigest() == sys.argv[2] else 3)\n"
)
_DIFFERS, _UNREADABLE = 3, 4


def replay_patch_sha(target: Target) -> str:
    """The replay identity: a supplied revision key, otherwise the patch bytes."""
    return target.patch_sha or hashlib.sha256(target.patch.encode("utf-8", "surrogatepass")).hexdigest()


def base_check_command(path: str, digest: str) -> list[str]:
    """The command that asks the base workspace whether `path` holds the text whose SHA-256 is `digest`: the
    repository's own Python, isolated (-I), run from the workspace root."""
    return ["python", "-I", "-c", SAME_AS_BASE, "./" + path, digest]


# A run's ERROR carries a failure type when the runner read and typed the failure (an import error, a crash elsewhere);
# without one, or typed infra, the run could not be read.
_UNREAD: Final = frozenset({None, FailureType.INFRA})


@dataclass(frozen=True)
class RerunPolicy:
    """How many runs a side gets, each in a fresh container, and how long each may take."""

    runs: int = BLOCKER_RUNS
    timeout_s: float = DEFAULT_RUN_TIMEOUT_S

    def __post_init__(self) -> None:
        if self.runs < BLOCKER_RUNS:
            raise ValueError(f"a blocker needs at least {BLOCKER_RUNS} runs on each side")
        if not self.timeout_s > 0:
            raise ValueError("a run's timeout is positive")


@dataclass(frozen=True)
class Verdict:
    status: Status
    detail: str
    reason: UncheckedReason | None = None


@dataclass(frozen=True)
class _Plan:
    """What SK1 checks and runs for one repro: the tests the runner is pointed at, the cases each run must report,
    the ids no file of the repro defines, and each file's Scope for the fake-test rules: as written (`strict`), and,
    for a test file, as the repository's own (`own`) should it be the base tree's content at its path."""

    tests: tuple[str, ...]
    cases: tuple[str, ...]
    unplaced: tuple[str, ...]
    strict: dict[str, Scope]
    own: dict[str, Scope]


@dataclass(frozen=True)
class Screened:
    """A candidate after the static gates and its first patch run. `stop` is its verdict when the screen decides it;
    when it is None the first patch run failed on a counted failure, and the candidate goes on to the reruns."""

    repro: Repro
    attack: Attack
    tests: tuple[str, ...]
    cases: tuple[str, ...]
    violations: tuple[Violation, ...]
    runs: tuple[RunResult, ...]
    stop: Verdict | None
    seconds: float

    @property
    def failed_first_patch_run(self) -> bool:
        return self.stop is None


def counted_failure(run: RunResult) -> bool:
    return run.outcome is Outcome.FAIL and run.failure_type in COUNTED_FAILURES


class TargetSkeptic:
    """SK1 for one target. It builds the patch and base workspaces once, when a candidate first needs them, shares
    them across candidates and arms (every run is in a fresh container), and closes them in close(). So too each
    surrogate's workspace, and each claim's surrogates (`surrogates`, up to `k` of them; module docstring, (d))."""

    def __init__(
        self,
        target: Target,
        *,
        sandbox: Sandbox,
        runner: SuiteRunner,
        policy: RerunPolicy | None = None,
        case_names: CaseNamer = expected_cases,
        clock: Callable[[], float] = time.monotonic,
        surrogates: SurrogateBuilder | None = None,
        k: int = SURROGATES_PER_CLAIM,
        surrogate_budget: int = SURROGATE_TARGET_CAP_MICRO_USD,
    ) -> None:
        if k < 1:
            raise ValueError("a claim gets at least 1 surrogate")
        if surrogate_budget < 0:
            raise ValueError("the target's surrogate budget is not negative")
        self.target = target
        self._sandbox = sandbox
        self._runner = runner
        self._policy = policy if policy is not None else RerunPolicy()
        self._case_names = case_names
        self._clock = clock
        self.changed_files = changed_files(target.patch)
        self.facts = patch_facts(target.patch, target.repo)
        self._workspaces: dict[Side, Workspace] = {}
        self._prepared: list[Workspace] = []  # every workspace prepared, kept after close() for the summary
        self._broken: dict[Side, str] = {}
        self._own: dict[tuple[str, str], bool] = {}  # (path, SHA-256 of a text): whether the base tree has that text
        self._builder = surrogates
        self._k = k
        self._surrogates: dict[str, tuple[Surrogate, ...]] = {}  # claim id: its surrogates, built once
        self._surrogate_workspaces: dict[str, Workspace] = {}  # surrogate id: its workspace
        self._surrogate_broken: dict[str, str] = {}  # surrogate id: why its workspace could not be prepared
        self._surrogate_budget = surrogate_budget
        self._surrogate_micro_usd = 0

    @property
    def surrogate_spent(self) -> int:
        """The micro-dollars building every claim's surrogates has spent so far, every arm's builds included, against
        the target's surrogate budget (module docstring, (d))."""
        return self._surrogate_micro_usd

    @property
    def prepared(self) -> tuple[Workspace, ...]:
        """Every workspace this skeptic prepared, in order, each with how its patch applied (REG-569)."""
        return tuple(self._prepared)

    def split(self, repro: Repro) -> tuple[tuple[Repro, ...], tuple[str, ...]]:
        """One repro per test case (module docstring), and the cases past MAX_CASES_PER_REPRO, left unjudged."""
        cases = repro.cases or self._case_names(repro, self._runner.kind)
        if len(cases) <= 1:
            return (repro,), ()
        parts = tuple(
            Repro.model_validate({**repro.model_dump(), "id": f"{repro.id}.c{n}", "cases": (case,)})
            for n, case in enumerate(cases[:MAX_CASES_PER_REPRO], start=1)
        )
        return parts, tuple(cases[MAX_CASES_PER_REPRO:])

    def screen(self, repro: Repro, attack: Attack) -> Screened:
        """The static gates, then the first patch run (fresh, seed 0)."""
        started = self._clock()
        plan = self._plan(repro)
        violations: tuple[Violation, ...] = ()
        runs: tuple[RunResult, ...] = ()
        stop: Verdict | None = None
        unquoted = self._unquoted(repro)
        if repro.claim_id != attack.claim_id or repro.lens is not attack.lens:
            stop = Verdict(Status.REFUTED, "SK1: the repro answers for another claim or lens than its attack's")
        elif unquoted is not None:
            stop = Verdict(Status.REFUTED, unquoted)
        else:
            judged = self._fake_tests(repro, plan)
            if isinstance(judged, Verdict):
                stop = judged
            else:
                violations = judged
                stop = self._static_stop(plan, violations)
        if stop is None:
            first = self._run(Side.PATCH, repro, plan.tests, plan.cases, seed=0)
            if isinstance(first, Verdict):
                stop = first
            else:
                runs = (first,)
                stop = None if counted_failure(first) else _patch_stop(first)
        return Screened(repro, attack, plan.tests, plan.cases, violations, runs, stop, self._clock() - started)

    def _unquoted(self, repro: Repro) -> str | None:
        """Why a fix-validity or behavior repro's quote does not stand (module docstring); None when it does, and for a
        regression repro, which quotes nothing."""
        if repro.kind is CatchKind.REGRESSION:
            return None
        quote = " ".join(repro.quote.split())
        if not quote:
            return "SK2 quote check: the repro quotes no sentence of the issue it enforces"
        if count_words(quote) < MIN_QUOTE_WORDS:
            return f"SK2 quote check: the repro's quote {quote[:200]!r} is under {MIN_QUOTE_WORDS} words"
        target = self.target
        texts: tuple[str, ...]
        if target.issue.strip():
            texts, where = (target.issue,), "the issue"
        else:
            texts, where = (target.title, target.body), "the pull request's title or body"
        if any(isinstance(find_quote(quote, " ".join(text.split())), int) for text in texts):
            return None
        return f"SK2 quote check: the repro's quote is not in {where} verbatim, on word boundaries: {quote[:200]!r}"

    def _plan(self, repro: Repro) -> _Plan:
        """The tests, the cases and the fake-test scopes for a repro (module docstring)."""
        kind = self._runner.kind
        strict: dict[str, Scope] = {}
        own: dict[str, Scope] = {}
        if not repro.cases:  # every case the repro's test files define
            defined = defined_cases(repro, kind)
            for path in repro.files:
                strict[path] = WHOLE_FILE
                if path in defined:
                    own[path] = Scope(cases=defined[path], own=True)
            return _Plan(repro_test_files(repro), self._case_names(repro, kind), (), strict, own)
        placed = place_cases(repro, kind)
        for path in repro.files:
            if path in placed.whole:
                strict[path], scope = WHOLE_FILE, Scope(own=True)
            elif path in placed.refs:
                strict[path], scope = Scope(cases=placed.refs[path]), Scope(cases=placed.refs[path], own=True)
            else:  # the repro names no case in this file: what it wrote here is read whole
                strict[path], scope = WHOLE_FILE, Scope(cases=(), own=True)
            if is_test_module(path):
                own[path] = scope
        return _Plan(placed.tests, placed.expect, placed.unplaced, strict, own)

    def _fake_tests(self, repro: Repro, plan: _Plan) -> tuple[Violation, ...] | Verdict:
        """The fake-test violations that stand. A test file whose violations would all go if it were the repository's
        own test is compared with the base tree's file first, and only when no other violation already refutes."""
        standing: list[Violation] = []
        hanging: dict[str, tuple[Violation, ...]] = {}  # path: its violations, unless the base tree has its text
        for path, source in repro.files.items():
            as_written = check_repro_files({path: source}, self.facts, {path: plan.strict[path]})
            if path not in plan.own or not as_written:
                standing += as_written
                continue
            as_own = check_repro_files({path: source}, self.facts, {path: plan.own[path]})
            standing += as_own
            if as_own != as_written and not as_own:
                hanging[path] = as_written
        if standing or not hanging or plan.unplaced or not plan.tests or not plan.cases:
            return tuple(standing)  # refuted, or let through, on what the repro alone shows
        for path, found in hanging.items():
            own = self._is_base_content(path, repro.files[path])
            if isinstance(own, Verdict):
                return own
            if not own:
                standing += found
        return tuple(standing)

    def _static_stop(self, plan: _Plan, violations: Sequence[Violation]) -> Verdict | None:
        if violations:
            return Verdict(Status.REFUTED, "SK1 fake-test rules: " + "; ".join(str(v) for v in violations))
        if plan.unplaced:
            named = ", ".join(repr(case[:200]) for case in plan.unplaced[:5])
            return Verdict(Status.REFUTED, f"SK1: the repro names cases none of its files defines: {named}")
        if not plan.tests or not plan.cases:
            return Verdict(Status.REFUTED, "SK1: the repro defines no test case for the runner to run")
        return None

    def _is_base_content(self, path: str, text: str) -> bool | Verdict:
        """Whether the base tree holds exactly `text` at `path` (SAME_AS_BASE, run in the base workspace)."""
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        known = self._own.get((path, digest))
        if known is not None:
            return known
        workspace = self._workspace(Side.BASE)
        if isinstance(workspace, Verdict):
            return workspace
        why = f"SK1: could not tell whether {path} is the repository's own test at base"
        try:
            run = self._sandbox.run(workspace, base_check_command(path, digest), timeout_s=BASE_CHECK_TIMEOUT_S)
        except Exception as exc:  # another lane's sandbox: unchecked, never a crash of the run
            log.error("skeptic target=%s base check of %s raised %r", self.target.id, path, exc)
            return Verdict(Status.UNCHECKED, f"{why}: the check raised {type(exc).__name__}", UncheckedReason.RUNTIME)
        if run.side is Side.BASE and run.outcome is Outcome.PASS:
            own = True
        elif run.side is Side.BASE and run.outcome is Outcome.ERROR and run.exit_code in (_DIFFERS, _UNREADABLE):
            own = False
        else:
            said = f" ({run.problem})" if run.problem else ""
            detail = f"{why}: the check ended {run.outcome.value}, exit {run.exit_code}{said}"
            return Verdict(Status.UNCHECKED, detail, UncheckedReason.RUNTIME)
        self._own[(path, digest)] = own
        return own

    def judge(
        self,
        screened: Screened,
        *,
        claim: Claim,
        finding_id: str,
        costs: Sequence[StageCost] = (),
        wall: Callable[[], float],
    ) -> Finding:
        """The candidate's finding: its reruns, its status and its evidence. `costs` are what producing it spent
        elsewhere (an attack's model calls); the skeptic adds its own seconds, and what building its claim's
        surrogates spent when this candidate built them (stage "surrogate"). `wall` gives the seconds since the
        target started, read when the finding is made."""
        started = self._clock()
        runs = list(screened.runs)
        verdict = screened.stop
        built: list[StageCost] = []  # what building the claim's surrogates spent, when this candidate built them
        if verdict is None:
            verdict = self._rerun(screened, runs, claim, built)
        seconds = screened.seconds + self._clock() - started
        spent = (*costs, StageCost(stage=SKEPTIC_STAGE, seconds=max(seconds, 0.0)), *built)
        return self._finding(screened, claim, finding_id, verdict, tuple(runs), spent, wall)

    def close(self) -> None:
        named = [(side.value, w) for side, w in self._workspaces.items()]
        named += [(f"surrogate {sid}", w) for sid, w in self._surrogate_workspaces.items()]
        for name, workspace in named:
            try:
                self._sandbox.close(workspace)
            except Exception as exc:  # another lane's sandbox: log it and keep closing the rest
                log.error("skeptic target=%s could not close the %s workspace: %r", self.target.id, name, exc)
        self._workspaces.clear()
        self._surrogate_workspaces.clear()

    def _rerun(self, screened: Screened, runs: list[RunResult], claim: Claim, built: list[StageCost]) -> Verdict:
        repro, tests, cases = screened.repro, screened.tests, screened.cases
        total = self._policy.runs
        for seed in range(1, total):
            run = self._run(Side.PATCH, repro, tests, cases, seed=seed)
            if isinstance(run, Verdict):
                return run
            runs.append(run)
        failed = sum(counted_failure(r) for r in runs)
        if failed < total:
            return Verdict(
                Status.FLAKY, f"SK1: fails on the patch on a counted failure in {failed} of {total} fresh runs"
            )
        base: list[RunResult] = []
        for seed in range(total):
            run = self._run(Side.BASE, repro, tests, cases, seed=seed)
            if isinstance(run, Verdict):
                return run
            base.append(run)
            runs.append(run)
        passed = sum(r.outcome is Outcome.PASS for r in base)
        imports = sum(r.failure_type is FailureType.IMPORT for r in base)
        if repro.kind is CatchKind.FIX_VALIDITY and imports:
            return Verdict(
                Status.REFUTED,
                f"SK1: fails on base with an import error in {imports} of {total} runs: the repro imports what base "
                "lacks, so it shows no bug of base's",
            )
        if repro.kind is CatchKind.REGRESSION:
            if passed == total:
                return Verdict(Status.BLOCKER, "")
            if passed == 0:
                return Verdict(Status.REFUTED, f"SK1: fails on base too, in all {total} runs: not a regression")
            return Verdict(Status.FLAKY, f"SK1: passes on base in {passed} of {total} fresh runs")
        lead = (
            f"SK1: fails on the patch in all {total} runs and passes on base in {passed} of {total}; a "
            f"{repro.kind.value} catch needs a plausible surrogate fix it passes on (SK3)"
        )
        return self._on_surrogates(screened, runs, claim, built, lead)

    def _on_surrogates(
        self, screened: Screened, runs: list[RunResult], claim: Claim, built: list[StageCost], lead: str
    ) -> Verdict:
        """The surrogate oracle (module docstring, (d)): blocker on the first plausible surrogate that passes the
        candidate 5 of 5. Failing that: unchecked when a surrogate's run could not be read or ran out of time, flaky
        when one passes it only sometimes, possibly toxic when every one fails it every run."""
        if self._builder is None:
            return Verdict(Status.S_SAT_INCONCLUSIVE, f"{lead}, and no surrogate builder is wired")
        k = self._k
        if claim.id not in self._surrogates:
            left = max(self._surrogate_budget - self._surrogate_micro_usd, 0)
            k = min(self._k, left // SURROGATE_CAP_MICRO_USD)  # as many as the target's surrogate budget covers
            if k < 1:
                return Verdict(
                    Status.S_SAT_INCONCLUSIVE,
                    f"{lead}; over the target's surrogate budget, no surrogate built ({left:,} of "
                    f"{self._surrogate_budget:,} micro-dollars left, and one surrogate may spend "
                    f"{SURROGATE_CAP_MICRO_USD:,})",
                )
        surrogates = self._surrogates_of(claim, built, k)
        plausible = [s for s in surrogates if s.plausible]
        if not surrogates:
            return Verdict(Status.S_SAT_INCONCLUSIVE, f"{lead}, and none was built")
        if not plausible:
            reasons = "; ".join(f"{s.id}: {s.why[:300]}" for s in surrogates)
            return Verdict(
                Status.S_SAT_INCONCLUSIVE, f"{lead}, and none of the {len(surrogates)} built is plausible: {reasons}"
            )
        repro, tests, cases = screened.repro, screened.tests, screened.cases
        total = self._policy.runs
        tried: list[RunResult] = []
        mixed: list[str] = []
        unread: list[Verdict] = []  # a surrogate whose run could not be read or timed out: the next one is tried
        for surrogate in plausible:
            own: list[RunResult] = []
            for seed in range(total):
                run = self._run(Side.SURROGATE, repro, tests, cases, seed=seed, surrogate=surrogate)
                if isinstance(run, Verdict):
                    unread.append(run)
                    break
                own.append(run)
            if len(own) < total:  # stopped by a run with no result
                tried += own
                continue
            passed = sum(r.outcome is Outcome.PASS for r in own)
            if passed == total:
                runs.extend(own)  # the blocker's evidence is the surrogate it passes on
                return Verdict(Status.BLOCKER, f"SK3: passes on surrogate {surrogate.id} in all {total} fresh runs")
            if passed:
                mixed.append(f"{surrogate.id} in {passed} of {total}")
            tried += own
        runs.extend(tried)
        if unread:
            first = unread[0]
            note = f"; {len(unread)} of {len(plausible)} plausible surrogates had a run with no result"
            return Verdict(first.status, first.detail + note, first.reason)
        if mixed:
            return Verdict(
                Status.FLAKY, f"SK3: passes on a plausible surrogate only in some fresh runs: {', '.join(mixed)}"
            )
        return Verdict(
            Status.POSSIBLY_TOXIC,
            f"SK3: fails on all {len(plausible)} plausible surrogate{'s' if len(plausible) != 1 else ''} in every "
            f"one of {total} fresh runs; it may ask for more than the issue does",
        )

    def _surrogates_of(self, claim: Claim, built: list[StageCost], k: int) -> tuple[Surrogate, ...]:
        """The claim's surrogates, `k` of them built on first use and kept, however few (a claim built under a
        tight surrogate budget is never built again); their costs go into `built` only that once."""
        known = self._surrogates.get(claim.id)
        if known is not None:
            return known
        assert self._builder is not None
        try:
            made = tuple(self._builder.build(self.target, claim, sandbox=self._sandbox, runner=self._runner, k=k))
        except Exception as exc:  # another lane's builder never raises (the contract); one that does builds nothing
            log.error("skeptic target=%s surrogate build for %s raised %r", self.target.id, claim.id, exc)
            made = ()
        made = tuple(s for s in made if s.claim_id == claim.id)[:k]
        self._surrogates[claim.id] = made
        for surrogate in made:
            built.extend(surrogate.costs)
            self._surrogate_micro_usd += sum(cost.micro_usd for cost in surrogate.costs)
        return made

    def _run(
        self,
        side: Side,
        repro: Repro,
        tests: Sequence[str],
        cases: Sequence[str],
        *,
        seed: int,
        surrogate: Surrogate | None = None,
    ) -> RunResult | Verdict:
        workspace = self._surrogate_workspace(surrogate) if surrogate is not None else self._workspace(side)
        if isinstance(workspace, Verdict):
            return workspace
        try:
            run = self._sandbox.run_test(
                workspace,
                tests,
                runner=self._runner,
                changed_files=self.changed_files,
                timeout_s=self._policy.timeout_s,
                files=repro.files,
                seed=seed,
                fresh=True,
                expect=cases,
            )
        except Exception as exc:  # another lane's sandbox: the candidate is unchecked, never a crash of the run
            log.error("skeptic target=%s %s run %d raised %r", self.target.id, side.value, seed, exc)
            return Verdict(
                Status.UNCHECKED, f"SK1: the {side.value} run raised {type(exc).__name__}", UncheckedReason.RUNTIME
            )
        if run.side is not side or not run.fresh_container:
            return Verdict(
                Status.UNCHECKED,
                f"SK1: the sandbox returned a {run.side.value} run (fresh: {run.fresh_container}) for a fresh "
                f"{side.value} run",
                UncheckedReason.RUNTIME,
            )
        said = f": {run.problem}" if run.problem else ""
        if run.outcome is Outcome.ERROR and run.failure_type in _UNREAD:
            return Verdict(
                Status.UNCHECKED,
                f"SK1: the {side.value} run {seed} could not be read"
                + (said or " (no report, or output the runner cannot read)"),
                UncheckedReason.RUNTIME,
            )
        if run.outcome is Outcome.TIMEOUT:
            return Verdict(
                Status.UNCHECKED,
                f"SK1: the {side.value} run {seed} was killed after {self._policy.timeout_s:g} s{said}",
                UncheckedReason.TIMEOUT,
            )
        return run

    def _workspace(self, side: Side) -> Workspace | Verdict:
        if side in self._broken:
            return Verdict(Status.UNCHECKED, self._broken[side], UncheckedReason.NO_ENV)
        workspace = self._workspaces.get(side)
        if workspace is not None:
            return workspace
        try:
            workspace = self._sandbox.prepare(self.target, side)
        except Exception as exc:  # another lane's sandbox: no workspace means no environment for this side
            log.error("skeptic target=%s could not prepare the %s workspace: %r", self.target.id, side.value, exc)
            self._broken[side] = f"SK1: no {side.value} workspace ({type(exc).__name__})"
            return Verdict(Status.UNCHECKED, self._broken[side], UncheckedReason.NO_ENV)
        self._workspaces[side] = workspace
        self._prepared.append(workspace)
        return workspace

    def _surrogate_workspace(self, surrogate: Surrogate) -> Workspace | Verdict:
        """The surrogate's workspace: base with its patch, prepared once and kept until close()."""
        broken = self._surrogate_broken.get(surrogate.id)
        if broken is not None:
            return Verdict(Status.UNCHECKED, broken, UncheckedReason.NO_ENV)
        workspace = self._surrogate_workspaces.get(surrogate.id)
        if workspace is not None:
            return workspace
        try:
            workspace = self._sandbox.prepare(self.target, Side.SURROGATE, patch=surrogate.patch)
        except Exception as exc:  # another lane's sandbox: no workspace means no environment for this surrogate
            log.error("skeptic target=%s could not prepare surrogate %s: %r", self.target.id, surrogate.id, exc)
            self._surrogate_broken[surrogate.id] = (
                f"SK3: no workspace for surrogate {surrogate.id} ({type(exc).__name__})"
            )
            return Verdict(Status.UNCHECKED, self._surrogate_broken[surrogate.id], UncheckedReason.NO_ENV)
        self._surrogate_workspaces[surrogate.id] = workspace
        self._prepared.append(workspace)
        return workspace

    def _finding(
        self,
        screened: Screened,
        claim: Claim,
        finding_id: str,
        verdict: Verdict,
        runs: tuple[RunResult, ...],
        costs: tuple[StageCost, ...],
        wall: Callable[[], float],
    ) -> Finding:
        repro, attack = screened.repro, screened.attack
        base = {
            "id": finding_id,
            "target_id": self.target.id,
            "claim_id": claim.id,
            "lens": attack.lens,
            "kind": repro.kind,
            "repro": repro,
            "runs": runs,
            "quote": claim.span.text,
            "replay": self._replay(screened),
            "costs": costs,
        }
        if verdict.status is Status.BLOCKER:
            try:
                return Finding.model_validate(
                    {**base, "status": Status.BLOCKER, "detail": verdict.detail or None, "wall_seconds": wall()}
                )
            except ValidationError as exc:  # the evidence passed SK1 but the finding is refused: never a blocker
                why = "; ".join(e["msg"] for e in exc.errors(include_url=False, include_input=False))
                verdict = Verdict(Status.SUSPECTED, f"SK1 passed, but the finding is not a blocker: {why}")
        return Finding.model_validate(
            {
                **base,
                "status": verdict.status,
                "unchecked_reason": verdict.reason if verdict.status is Status.UNCHECKED else None,
                "detail": verdict.detail or None,
                "wall_seconds": wall(),
            }
        )

    def _replay(self, screened: Screened) -> ReplayKey | None:
        workspace = self._workspaces.get(Side.PATCH)
        if workspace is None:
            return None
        target = self.target
        harness = {
            "files": screened.repro.files,
            "command": list(screened.repro.command),
            "tests": list(screened.tests),
            "cases": list(screened.cases),
            "runner": self._runner.kind.value,
            "test_command": target.test_command,
            "runs": self._policy.runs,
            "timeout_s": self._policy.timeout_s,
            "receipts": __version__,
        }
        harness_hash = hashlib.sha256(json.dumps(harness, sort_keys=True).encode()).hexdigest()
        patch_sha = replay_patch_sha(target)
        return ReplayKey(
            base_sha=target.base_sha,
            patch_sha=patch_sha,
            harness_hash=harness_hash,
            seed=0,
            image_digest=workspace.image_digest,
        )


def _patch_stop(run: RunResult) -> Verdict:
    """The verdict of a first patch run that did not fail on a counted failure."""
    if run.outcome is Outcome.PASS:
        return Verdict(Status.NOT_REPRODUCED, "SK1: passes on the patch")
    if run.outcome is Outcome.SKIPPED:
        return Verdict(Status.NOT_REPRODUCED, "SK1: skipped on the patch, so it never ran")
    failure = run.failure_type.value if run.failure_type is not None else "an untyped failure"
    said = f" ({run.problem})" if run.problem else ""
    return Verdict(
        Status.REFUTED, f"SK1 failure-type gate: fails on the patch with {failure}, which never counts{said}"
    )
