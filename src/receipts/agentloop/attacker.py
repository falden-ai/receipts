"""The correctness attacker's stage 2, at its minimum (spec AT2): an explore loop that ends in a handoff
`{hypothesis, target_symbols, inputs, expected, observed_if_bug}`, then one structured call that writes up to 3 repro
tests in the target's runner form, then one run of each on the patch through Sandbox.run_test. The repros returned are
those that failed on a counted failure (spec hard rule 3); the skeptic reruns them for the evidence a blocker needs.
The critic and rewrite rounds come later. The concept follows the dodgy-diff workflow, after Becker et al. 2026.

The pipeline runs it for a claim only when stage 1 left no candidate (`needs_stage_two`). One Budget covers the whole
attacker: the arm's step limit counts every model call, the write call included, the $0.25 cap covers all of them,
and the attacker's 15 minutes cover the runs too. Every model or sandbox failure comes back as an Unchecked item with
its reason; nothing here raises for one. A repro run the sandbox could not read (ERROR with an infra failure type or
none, and its RunResult.problem) or that was killed for time is unchecked too, never a repro that did not fail; a
repro whose own module fails to import is checked, and rejected by hard rule 3.

A written test goes where the model put it, renamed the way stage 1 renames its candidates: a file not already named
test_receipts_<words>.py becomes test_receipts_<attack>_<n>.py in the same directory, a name this attacker owns, and a
Django test below its app's directory goes in the app's directory, where the runner imports it by label. Only a
directory the runner does not collect from is refused (and retried once with the reason). The trace lists every
rename, and keeps the head of each write answer that could not be used.

Each repro is a whole test file the model wrote, so its Repro.cases is empty: every case the file defines (the
skeptic splits a file of several into one candidate per case). The recorded command names REPLAY_REPORT as its report
path; it is a record of how the file runs, and every real run, the attacker's, the skeptic's and replay's, goes
through Sandbox.run_test, which gives each run a report path of its own. Each written test quotes the sentence of the
issue it enforces, and the quote goes on its repro (Repro.quote) for the skeptic's check.

The same attacker takes the regression lens (``LoopAttacker(lens=Lens.REGRESSION)``) for every regression attack but
AT3's (contracts.Attack.existing_tests, which it leaves alone): its prompts ask for tests of behavior the code had
before the change, and each repro is a regression catch whatever the claim.
"""

from __future__ import annotations

import json
import logging
import math
import posixpath
import re
import time
from collections.abc import Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator, model_validator

from receipts.agentloop.diff import changed_files
from receipts.agentloop.loop import ATTACKER_WALL_S, Budget, LoopConfig, LoopResult, run_loop
from receipts.agentloop.prompts import explore_messages, write_messages
from receipts.contracts import (
    COUNTED_FAILURES,
    Arm,
    Attack,
    AttackOutcome,
    CatchKind,
    FailureType,
    Frozen,
    Lens,
    Message,
    ModelCallError,
    ModelClient,
    ModelRequest,
    Outcome,
    PatchMethod,
    Repro,
    RunnerKind,
    Sandbox,
    Side,
    StageCost,
    SuiteRunner,
    Target,
    Unchecked,
    UncheckedReason,
    Workspace,
)

logger = logging.getLogger(__name__)

EXPLORE_STAGE = "attack.correctness.explore"
WRITE_STAGE = "attack.correctness.write"
SANDBOX_STAGE = "attack.correctness.sandbox"  # code only: the seconds the sandbox ran this attacker's commands
MAX_REPROS = 3
WRITE_CALLS = 2  # the write call and its one retry, kept back from the explore loop's share of the step limit
WRITE_SHARE = 0.2  # of the cost cap, kept back from the explore loop for the write call
TEST_TIMEOUT_S = 60.0  # one run of one repro, a command like any other (spec: 60 s per command)
REPLAY_REPORT = "/tmp/receipts-replay-report"  # the report path a recorded repro command names; never read
CONTENT_LIMIT = 20_000  # characters in one repro test file

_KINDS = {"fixed_x": CatchKind.FIX_VALIDITY, "adds_behavior": CatchKind.BEHAVIOR}
_NAME = r"test_receipts_[a-z0-9_]+\.py"
_OWNED_NAME = re.compile(_NAME)
_SLUG = re.compile(r"[^a-z0-9]+")
ANSWER_CHARS = 300  # of a write answer that could not be used, kept in the trace
_PATHS = {
    RunnerKind.PYTEST: re.compile(rf"^(?:[A-Za-z0-9_][A-Za-z0-9_.\-]*/)*{_NAME}$"),
    RunnerKind.DJANGO: re.compile(rf"^tests/[A-Za-z0-9_]+/{_NAME}$"),
    RunnerKind.SYMPY: re.compile(rf"^sympy/(?:[A-Za-z0-9_]+/)*tests/{_NAME}$"),
}
WRITE_SCHEMA: dict[str, object] = {
    "title": "repro_tests",
    "type": "object",
    "properties": {
        "tests": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_REPROS,
            "items": {
                "type": "object",
                "properties": {"path": {"type": "string"}, "content": {"type": "string"}, "quote": {"type": "string"}},
                "required": ["path", "content", "quote"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["tests"],
    "additionalProperties": False,
}


class Handoff(BaseModel):
    """The explore loop's submission (spec AT2's handoff). A hypothesis of "none found" means no failure was seen."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    hypothesis: str = Field(min_length=1, max_length=2_000)
    target_symbols: list[str] = Field(default_factory=list, max_length=20)
    inputs: str | list[object] | dict[str, object] = ""
    expected: str | list[object] | dict[str, object] = ""
    observed_if_bug: str | list[object] | dict[str, object] = ""

    @property
    def found_nothing(self) -> bool:
        return self.hypothesis.strip().lower().rstrip(".") == "none found"


class WrittenTest(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    path: str
    content: str = Field(min_length=1, max_length=CONTENT_LIMIT)
    quote: str = Field(default="", max_length=2_000)  # the issue's sentence the test enforces (prompts.SCOPE_RULE)

    @field_validator("content")
    @classmethod
    def _utf8(cls, v: str) -> str:
        v.encode("utf-8")  # a lone surrogate raises here, as a validation error
        return v


class WrittenTests(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    tests: list[WrittenTest] = Field(min_length=1, max_length=MAX_REPROS)

    @model_validator(mode="after")
    def _distinct(self) -> WrittenTests:
        paths = [t.path for t in self.tests]
        if len(set(paths)) != len(paths):
            raise ValueError("each test has its own path")
        return self


class TraceRun(Frozen):
    path: str
    outcome: Outcome
    failure_type: str | None
    seconds: float
    kept: bool
    problem: str = ""  # why the sandbox could not read the run (RunResult.problem); empty when it was read


class AttackTrace(Frozen):
    """One attacker's whole run, for the trajectory log: every step, the handoff, and each repro's one run."""

    target_id: str
    attack_id: str
    arm: str
    model: str
    # How the agent's patch applied in the workspace this attack explored and ran its repros in (REG-569): a tree that
    # took --reject or GNU patch's fuzz is not the diff as written. None when no workspace was prepared.
    patch_method: PatchMethod | None = None
    explore: LoopResult | None
    handoff: dict[str, object] | None = None
    tests_written: int = 0
    runs: tuple[TraceRun, ...] = ()
    write_steps: int = 0
    renamed: dict[str, str] = Field(default_factory=dict)  # a written test's path -> where it ran (module docstring)
    write_rejected: tuple[str, ...] = ()  # each unusable write answer: why, and the head of the answer
    unchecked: tuple[Unchecked, ...] = ()
    micro_usd: int = 0


def needs_stage_two(stage_one: AttackOutcome) -> bool:
    """Stage 2 runs only when stage 1 left no candidate repro for the attack (spec AT2: "only if needed")."""
    return not stage_one.repros


def catch_kind(claim_id: str) -> CatchKind:
    """The kind a repro of this claim would be: a fixed_x claim's repro needs a plausible fix to pass (fix validity),
    an adds_behavior claim's has its own oracle, and any other claim's is a regression against base."""
    return _KINDS.get(claim_id.partition(":")[0], CatchKind.REGRESSION)


class LoopAttacker:
    """contracts.Attacker for the correctness lens, or the regression lens's model attacks, built on the AT1 loop."""

    lens: Lens

    def __init__(
        self,
        *,
        lens: Lens = Lens.CORRECTNESS,
        config: LoopConfig | None = None,
        write_max_tokens: int | None = None,
        test_timeout_s: float = TEST_TIMEOUT_S,
        wall_s: float = ATTACKER_WALL_S,
        clock: Callable[[], float] = time.monotonic,
        on_trace: Callable[[AttackTrace], None] | None = None,
    ) -> None:
        if lens not in (Lens.CORRECTNESS, Lens.REGRESSION):
            raise ValueError(f"this attacker takes the correctness or the regression lens, not {lens}")
        self.lens = lens
        self._config = config or LoopConfig()
        self._write_max_tokens = self._config.max_tokens if write_max_tokens is None else write_max_tokens
        if self._write_max_tokens <= 0:
            raise ValueError("the write output limit must be positive")
        self._test_timeout_s = test_timeout_s
        self._wall_s = wall_s
        self._clock = clock
        self._on_trace = on_trace

    def attack(
        self,
        target: Target,
        attack: Attack,
        *,
        arm: Arm,
        client: ModelClient,
        sandbox: Sandbox,
        runner: SuiteRunner,
    ) -> AttackOutcome:
        if attack.existing_tests:
            return AttackOutcome(attack_id=attack.id)  # AT3's attack: it reruns the existing tests, and writes none
        run = _Run(self, target, attack, arm, client, sandbox, runner)
        try:
            return run.go()
        finally:
            run.close()
            self._trace(run.trace())

    def _trace(self, trace: AttackTrace) -> None:
        if self._on_trace is None:
            return
        try:
            self._on_trace(trace)
        except Exception:  # a broken log must never fail the attack; it is reported here instead
            logger.exception("the trajectory sink raised for attack %s", trace.attack_id)


class _Run:
    """The state of one attack, from the first prompt to the last repro run."""

    def __init__(
        self,
        owner: LoopAttacker,
        target: Target,
        attack: Attack,
        arm: Arm,
        client: ModelClient,
        sandbox: Sandbox,
        runner: SuiteRunner,
    ) -> None:
        self.owner, self.target, self.attack, self.arm = owner, target, attack, arm
        self.client, self.sandbox, self.runner = client, sandbox, runner
        self.config = owner._config.model_copy(update={"step_limit": arm.step_limit})
        self.budget = Budget(
            step_limit=arm.step_limit,
            cost_limit_micro_usd=self.config.cost_limit_micro_usd,
            wall_s=owner._wall_s,
            clock=owner._clock,
        )
        self.workspace: Workspace | None = None
        self.explore: LoopResult | None = None
        self.handoff: Handoff | None = None
        self.tests: list[WrittenTest] = []
        self.runs: list[TraceRun] = []
        self.repros: list[Repro] = []
        self.unchecked: list[Unchecked] = []
        self.sandbox_seconds: list[float] = []
        self.write_steps = 0
        self.renamed: dict[str, str] = {}
        self.write_rejected: list[str] = []

    # ------------------------------------------------------------ the attack

    def go(self) -> AttackOutcome:
        if self.attack.lens is not self.owner.lens:
            return self.outcome(
                UncheckedReason.RUNTIME, f"this attacker takes the {self.owner.lens.value} lens, not {self.attack.lens}"
            )
        try:
            workspace = self.sandbox.prepare(self.target, Side.PATCH)
        except Exception as err:  # no environment for this target: nothing else can run
            return self.outcome(UncheckedReason.NO_ENV, f"the patch workspace could not be prepared: {_name(err)}")
        self.workspace = workspace
        explore_calls = max(1, self.arm.step_limit - WRITE_CALLS)
        system, task = explore_messages(
            self.target, self.attack, steps=explore_calls, timeout_s=self.config.command_timeout_s
        )
        self.explore = run_loop(
            self.client,
            self.sandbox,
            workspace,
            [Message(role="system", content=system), Message(role="user", content=task)],
            model=self.arm.attacker_model,
            temperature=self.attack.temperature,
            stage=EXPLORE_STAGE,
            budget=self.budget,
            config=self.config,
            check=_check_handoff,
            max_calls=explore_calls,
            max_micro_usd=int(self.budget.cost_limit * (1 - WRITE_SHARE)),
        )
        self.sandbox_seconds.append(self.explore.sandbox_seconds)
        if self.explore.submission is None:
            reason = self.explore.reason or UncheckedReason.RUNTIME
            return self.outcome(reason, f"explore: {self.explore.detail}")
        try:
            # The loop accepted it through _check_handoff, so this holds; if it ever does not, the attack is
            # unchecked rather than raised (the Attacker contract).
            self.handoff = Handoff.model_validate_json(self.explore.submission)
        except ValidationError as err:
            return self.outcome(UncheckedReason.UNPARSED, f"explore: the handoff is not usable: {_short(err)}")
        if self.handoff.found_nothing:
            return self.outcome()  # checked, and nothing found
        written = self.write(self.handoff)
        if written is None:
            return self.outcome()
        self.tests = written.tests
        self.run_tests(workspace)
        return self.outcome()

    def write(self, handoff: Handoff) -> WrittenTests | None:
        """One structured call for the tests, and one more with the validation error if the first does not parse."""
        system, user = write_messages(self.target, self.attack, handoff.model_dump_json(indent=1))
        messages = [Message(role="system", content=system), Message(role="user", content=user)]
        for attempt in (1, 2):
            stop = self.budget.stop()
            if stop is not None:
                self.unchecked.append(self.item(stop[1], f"write: {stop[2]}"))
                return None
            request = ModelRequest(
                stage=WRITE_STAGE,
                model=self.arm.attacker_model,
                messages=tuple(messages),
                temperature=self.attack.temperature,
                max_tokens=self.owner._write_max_tokens,
                reasoning_effort=self.config.reasoning_effort,
                json_schema=WRITE_SCHEMA,
                schema_name="repro_tests",
            )
            self.write_steps += 1
            try:
                response = self.client.complete(request)
            except ModelCallError as err:
                self.budget.record(err.cost)
                self.unchecked.append(self.item(err.reason, f"write: the model call failed: {err}"))
                return None
            except Exception as err:
                self.budget.record(None)
                self.unchecked.append(
                    self.item(UncheckedReason.RUNTIME, f"write: the model client raised {_name(err)}")
                )
                return None
            self.budget.record(response.cost)
            if response.cost is None:
                self.unchecked.append(self.item(UncheckedReason.BUDGET, "write: the model client returned no cost"))
                return None
            problem = "the answer was cut off or empty" if response.truncated else None
            if problem is None:
                try:
                    tests = WrittenTests.model_validate(
                        response.parsed if response.parsed is not None else _json_payload(response.text)
                    )
                    placed = self.place(tests)
                    if isinstance(placed, WrittenTests):
                        return placed
                    problem = placed
                except (ValidationError, ValueError) as err:
                    problem = _short(err)
            self.write_rejected.append(f"{problem} | {response.text[:ANSWER_CHARS]}")
            if attempt == 1:
                messages += [
                    Message(role="assistant", content=response.text),
                    Message(role="user", content=f"That answer was not usable: {problem} Answer again with the JSON."),
                ]
                continue
            self.unchecked.append(self.item(UncheckedReason.UNPARSED, f"write: {problem}"))
        return None

    def place(self, tests: WrittenTests) -> WrittenTests | str:
        """The tests at the paths they run at (module docstring), or what is wrong with a path."""
        slug = _SLUG.sub("_", self.attack.id.lower()).strip("_")[:40] or "attack"
        out: list[WrittenTest] = []
        renamed: dict[str, str] = {}
        for n, test in enumerate(tests.tests):
            path = owned_path(self.target.test_runner, test.path, f"test_receipts_{slug}_{n}.py")
            if path is None:
                return (
                    f"the path {test.path!r} is not in a directory where the {self.target.test_runner.value} runner "
                    "collects tests in the repository"
                )
            if path != test.path:
                renamed[test.path] = path
            out.append(WrittenTest(path=path, content=test.content, quote=test.quote))
        try:
            placed = WrittenTests(tests=out)
        except ValidationError as err:
            return f"after renaming: {_short(err)}"
        self.renamed.update(renamed)
        return placed

    def run_tests(self, workspace: Workspace) -> None:
        """Each test once on the patch, in a fresh container, so nothing the explore loop changed reaches it."""
        changed = changed_files(self.target.patch)
        regression = self.owner.lens is Lens.REGRESSION
        kind = CatchKind.REGRESSION if regression else catch_kind(self.attack.claim_id)
        for i, test in enumerate(self.tests, start=1):
            if self.budget.out_of_time():
                rest = ", ".join(t.path for t in self.tests[i - 1 :])
                self.unchecked.append(self.item(UncheckedReason.TIMEOUT, f"run: no time left for {rest}"))
                return
            # The runner is the sandbox side's code: its failure is an unchecked repro, never this attacker's raise.
            try:
                command = tuple(self.runner.command(self.target, [test.path], report_path=REPLAY_REPORT))
            except Exception as err:
                self.unchecked.append(self.item(UncheckedReason.RUNTIME, f"run: the runner raised {_name(err)}"))
                continue
            try:
                repro = Repro(
                    id=f"{self.attack.id}.r{i}",
                    attack_id=self.attack.id,
                    claim_id=self.attack.claim_id,
                    lens=self.owner.lens,
                    kind=kind,
                    files={test.path: test.content},
                    command=command,
                    quote="" if regression else test.quote,
                )
            except (ValidationError, ValueError) as err:
                self.unchecked.append(
                    self.item(UncheckedReason.RUNTIME, f"run: {test.path} is not a repro: {_short(err)}")
                )
                continue
            try:
                result = self.sandbox.run_test(
                    workspace,
                    [test.path],
                    runner=self.runner,
                    changed_files=changed,
                    timeout_s=self.owner._test_timeout_s,
                    files={test.path: test.content},
                    seed=self.arm.seed,
                    fresh=True,
                )
            except Exception as err:
                self.unchecked.append(self.item(UncheckedReason.RUNTIME, f"run: the sandbox raised {_name(err)}"))
                continue
            self.sandbox_seconds.append(result.duration_s)
            kept = result.outcome is Outcome.FAIL and result.failure_type in COUNTED_FAILURES
            failure = result.failure_type.value if result.failure_type else None
            self.runs.append(
                TraceRun(
                    path=test.path,
                    outcome=result.outcome,
                    failure_type=failure,
                    seconds=result.duration_s,
                    kept=kept,
                    problem=result.problem,
                )
            )
            if kept:
                self.repros.append(repro)
            elif result.outcome is Outcome.TIMEOUT:
                self.unchecked.append(self.item(UncheckedReason.TIMEOUT, f"run: {test.path} was killed for time"))
            elif result.outcome is Outcome.ERROR and result.failure_type in (FailureType.INFRA, None):
                why = " ".join(result.problem.split())[:500] or "the sandbox gave no reason"
                self.unchecked.append(self.item(UncheckedReason.RUNTIME, f"run: {test.path} could not be read: {why}"))

    def close(self) -> None:
        if self.workspace is None:
            return
        try:
            self.sandbox.close(self.workspace)
        except Exception:  # a container left behind is the sandbox's to clean; the attack's result stands
            logger.exception("closing the workspace of attack %s failed", self.attack.id)

    # ------------------------------------------------------------ results

    def item(self, reason: UncheckedReason, detail: str) -> Unchecked:
        return Unchecked(what=self.attack.id, reason=reason, detail=detail)

    def outcome(self, reason: UncheckedReason | None = None, detail: str = "") -> AttackOutcome:
        if reason is not None:
            self.unchecked.append(self.item(reason, detail))
        return AttackOutcome(
            attack_id=self.attack.id,
            repros=tuple(self.repros),
            costs=self.costs(),
            unchecked=tuple(self.unchecked),
        )

    def costs(self) -> tuple[StageCost, ...]:
        """One StageCost per stage and model, adding up every call the client priced, then the sandbox's seconds."""
        totals: dict[tuple[str, str | None], list[StageCost]] = {}
        for cost in self.budget.costs:
            totals.setdefault((cost.stage, cost.model), []).append(cost)
        rows = [
            StageCost(
                stage=stage,
                model=model,
                input_tokens=sum(c.input_tokens for c in group),
                cached_tokens=sum(c.cached_tokens for c in group),
                output_tokens=sum(c.output_tokens for c in group),
                reasoning_tokens=sum(c.reasoning_tokens for c in group),
                micro_usd=sum(c.micro_usd for c in group),
                seconds=math.fsum(c.seconds for c in group),
            )
            for (stage, model), group in totals.items()
        ]
        seconds = math.fsum(self.sandbox_seconds)
        if seconds:
            rows.append(StageCost(stage=SANDBOX_STAGE, model=None, seconds=seconds))
        return tuple(rows)

    def trace(self) -> AttackTrace:
        return AttackTrace(
            target_id=self.target.id,
            attack_id=self.attack.id,
            arm=self.arm.name,
            model=self.arm.attacker_model,
            patch_method=None if self.workspace is None else self.workspace.patch_method,
            explore=self.explore,
            handoff=self.handoff.model_dump() if self.handoff else None,
            tests_written=len(self.tests),
            runs=tuple(self.runs),
            write_steps=self.write_steps,
            renamed=dict(self.renamed),
            write_rejected=tuple(self.write_rejected),
            unchecked=tuple(self.unchecked),
            micro_usd=self.budget.micro_usd,
        )


def owned_path(kind: RunnerKind, proposed: str, owned: str) -> str | None:
    """Where a written test runs: its own path when it is already a test_receipts_<words>.py module the runner
    collects, else ``owned`` in the proposed directory (a Django test in its app's own directory). None when the
    path leaves the repository, is hidden, is not a .py file, or is in a directory the runner does not collect."""
    raw = proposed.strip()
    raw = raw[2:] if raw.startswith("./") else raw
    if raw.startswith("/") or ".." in raw.split("/"):
        return None
    norm = posixpath.normpath(raw)
    parts = norm.split("/")
    if any(p.startswith(".") for p in parts) or not norm.endswith(".py") or len(parts) > 10:
        return None
    pattern = _PATHS[kind]
    if pattern.fullmatch(norm):
        return norm
    directory = posixpath.dirname(norm)
    if kind is RunnerKind.DJANGO and len(parts) >= 3 and parts[0] == "tests":
        directory = f"tests/{parts[1]}"
    path = posixpath.join(directory, owned) if directory else owned
    return path if pattern.fullmatch(path) and _OWNED_NAME.fullmatch(owned) else None


def _json_payload(text: str) -> object:
    """The JSON object in an answer: the whole answer, else its one ```json block, else its outermost braces. A model
    asked for bare JSON often fences it anyway when the provider does not enforce the schema."""
    candidates = [text.strip()]
    fenced = re.findall(r"^```json[ \t]*\n(.*?)\n```[ \t]*$", text, re.M | re.S)
    if len(fenced) == 1:
        candidates.append(fenced[0])
    if "{" in text and "}" in text:
        candidates.append(text[text.index("{") : text.rindex("}") + 1])
    for candidate in candidates:
        try:
            return json.loads(candidate)
        except ValueError:
            continue
    raise ValueError("the answer holds no JSON object")


def _check_handoff(text: str) -> str | None:
    try:
        Handoff.model_validate_json(text)
    except ValidationError as err:
        return f"the handoff is not the JSON object asked for ({_short(err)})."
    return None


def _short(err: Exception) -> str:
    if isinstance(err, ValidationError):
        first = err.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(p) for p in first["loc"])
        return f"{where}: {first['msg']}" if where else str(first["msg"])
    return " ".join(str(err).split())[:300]


def _name(err: Exception) -> str:
    return f"{type(err).__name__}: {' '.join(str(err).split())[:300]}"
