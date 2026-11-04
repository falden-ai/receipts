"""AT1, the attacker loop: one model, one bash workspace in the sandbox, and a linear message history.

Each step asks the model for one answer and reads exactly one fenced block from it (after mini-swe-agent: Yang et al.,
NeurIPS 2024, MIT; its design, none of its code):

- a ```bash block is an action: it runs through Sandbox.run as `bash -lc <block>` in a fresh shell in the repository
  root, killed after 60 s, and its exit code and output come back as the next user message, the output cut to its
  head and tail when it is long;
- a ```json block is the submission, which ends the loop if the caller's check accepts it. The submission comes from
  the model's own answer, not from a command's output, so nothing the repository prints can end the loop or forge it.

Blocks in any other language are prose and are not counted. An opening ```bash or ```json fence that ends a line after
other text counts as if it began the line (own_line_fences: Spark writes "Fixing the regression.```bash"), when its
closing fence stands on a line of its own, as every block's must. Anything else (no block, two blocks, a truncated or
empty answer) is a format error, answered with a reminder of the format; three in a row end the loop. Each Step of an
answer the loop could not use (a format error, a truncated or empty answer, a refused submission) keeps the head of
the answer (ANSWER_CHARS) and why it could not be used (for a format error, the fenced blocks it held, so a ```sh block
shows as one), and each action's Step keeps the head of the output the model was shown (OUTPUT_CHARS), so a trajectory
log shows what the model sent and saw.

Every call is told how many model answers are left, the call itself included, counted against the step limit the loop
runs under: the last user message before the call ends with that count. With 2 or fewer left it asks for the ```json
submission only and no more commands, so a loop that is about to hit its step limit hands off instead of exploring.

A shared Budget stops the loop before a call once the step limit is reached, once the next call would likely cross
the cost cap ($0.25 per attacker by default: what was spent plus the last call's cost, since each call resends a
longer history), or once the trajectory's time is up. A model call that fails, and a sandbox that fails, end the loop
with the reason; nothing here raises for either.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable
from typing import Literal

from pydantic import Field

from receipts.contracts import (
    Frozen,
    Message,
    ModelCallError,
    ModelClient,
    ModelRequest,
    Outcome,
    RunResult,
    Sandbox,
    StageCost,
    UncheckedReason,
    Workspace,
    check_model,
)

SHELL = ("bash", "-lc")
ATTACKER_CAP_MICRO_USD = 250_000  # spec AT1: $0.25 per attacker
COMMAND_TIMEOUT_S = 60.0  # spec parameters: 60 s per command
TRAJECTORY_TIMEOUT_S = 300.0  # 5 min per trajectory
ATTACKER_WALL_S = 900.0  # 15 min wall per attacker
PROBLEM_CHARS = 1_000  # of a run's problem shown to the model
ANSWER_CHARS = 300  # of an answer the loop could not use, kept in its Step
OUTPUT_CHARS = 500  # of the output an action showed the model, kept in its Step
FINAL_STEPS = 2  # with this many answers or fewer left, the loop asks for the submission only

# One fenced block: three or more backticks, an info string, the body, and a closing fence of the same backticks.
_BLOCK = re.compile(r"^(?P<fence>`{3,})(?P<info>[A-Za-z0-9_+-]*)[ \t]*\n(?P<body>.*?)\n(?P=fence)[ \t]*$", re.M | re.S)
_OPENING_FENCE = re.compile(r"(?P<fence>`{3,})[A-Za-z0-9_+-]*[ \t]*")
# An opening ```bash or ```json fence that ends its line, for own_line_fences. It starts at the first backtick of its
# run, so a search is linear in a line of backticks rather than retrying at each one.
_FENCE_AT_LINE_END = re.compile(r"(?<!`)`{3,}(?:bash|json)[ \t]*$", re.IGNORECASE)

Exit = Literal[
    "submitted",
    "step_limit",
    "cost_limit",
    "timeout",
    "format_errors",
    "model_error",
    "unpriced",
    "sandbox_error",
]
StepKind = Literal["action", "submission", "format_error", "truncated", "refused_submission"]


class LoopConfig(Frozen):
    step_limit: int = Field(default=20, ge=1, le=30)  # model calls, from the Arm
    cost_limit_micro_usd: int = Field(default=ATTACKER_CAP_MICRO_USD, gt=0)
    command_timeout_s: float = Field(default=COMMAND_TIMEOUT_S, gt=0)
    trajectory_timeout_s: float = Field(default=TRAJECTORY_TIMEOUT_S, gt=0)
    observation_chars: int = Field(default=10_000, ge=200)  # a longer output keeps its head and tail
    max_tokens: int = Field(default=8_192, gt=0)  # room for a reasoning model to reach its answer
    max_format_errors: int = Field(default=3, ge=1)
    reasoning_effort: Literal["low", "medium", "high"] | None = "medium"  # spec parameters: attackers medium


class Budget:
    """Calls, spend and time, shared by every phase of one attacker. Every call's cost is kept as the client priced
    it, failed calls included."""

    def __init__(
        self,
        *,
        step_limit: int,
        cost_limit_micro_usd: int = ATTACKER_CAP_MICRO_USD,
        wall_s: float = ATTACKER_WALL_S,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.step_limit = step_limit
        self.cost_limit = cost_limit_micro_usd
        self.clock = clock
        self.started = clock()
        self.deadline = self.started + wall_s
        self.calls = 0
        self.micro_usd = 0
        self.last_call_micro_usd = 0
        self.costs: list[StageCost] = []

    def stop(
        self, *, deadline: float | None = None, calls: int | None = None, micro_usd: int | None = None
    ) -> tuple[Exit, UncheckedReason, str] | None:
        """Why the next model call must not be made, or None when it may. `calls` and `micro_usd` are a phase's own
        tighter limits, counted from the attacker's start, so a phase can leave room for the ones after it."""
        step_limit = self.step_limit if calls is None else min(calls, self.step_limit)
        cost_limit = self.cost_limit if micro_usd is None else min(micro_usd, self.cost_limit)
        if self.calls >= step_limit:
            return "step_limit", UncheckedReason.BUDGET, f"the step limit of {step_limit} model calls was reached"
        if self.micro_usd + self.last_call_micro_usd > cost_limit:
            return (
                "cost_limit",
                UncheckedReason.BUDGET,
                f"{self.micro_usd} micro-dollars spent, and the next call would likely cross the limit of {cost_limit}",
            )
        if self.clock() >= min(self.deadline, math.inf if deadline is None else deadline):
            return "timeout", UncheckedReason.TIMEOUT, "the time for this trajectory ran out"
        return None

    def out_of_time(self) -> bool:
        return self.clock() >= self.deadline

    def record(self, cost: StageCost | None) -> None:
        """Count one call. A call with no cost still counts as a step."""
        self.calls += 1
        if cost is not None:
            self.costs.append(cost)
            self.micro_usd += cost.micro_usd
            self.last_call_micro_usd = cost.micro_usd


class Step(Frozen):
    """One model call and what came of it, for the trajectory log (the pilot reads tokens and seconds per step)."""

    index: int
    kind: StepKind
    finish_reason: str
    input_tokens: int
    output_tokens: int
    model_seconds: float
    action: str | None = None
    exit_code: int | None = None
    outcome: Outcome | None = None
    command_seconds: float | None = None
    problem: str = ""  # why the sandbox could not read the command's run (RunResult.problem)
    answer: str = ""  # the head of an answer the loop could not use (format error, truncated, refused submission)
    why: str = ""  # why that answer could not be used
    output: str = ""  # the head of the output an action showed the model (its observation)


class LoopResult(Frozen):
    exit: Exit
    reason: UncheckedReason | None  # None only when submitted
    detail: str
    submission: str | None = None  # the accepted json block's text
    steps: tuple[Step, ...] = ()
    sandbox_seconds: float = 0.0


SubmissionCheck = Callable[[str], str | None]  # the submission's text -> None when accepted, else what is wrong


class _Call:
    """What one model call reported, for its Step."""

    def __init__(self, finish_reason: str, input_tokens: int, output_tokens: int, seconds: float) -> None:
        self.finish_reason = finish_reason
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.seconds = seconds

    def step(
        self,
        index: int,
        kind: StepKind,
        *,
        action: str | None = None,
        exit_code: int | None = None,
        outcome: Outcome | None = None,
        command_seconds: float | None = None,
        problem: str = "",
        answer: str = "",
        why: str = "",
        output: str = "",
    ) -> Step:
        return Step(
            index=index,
            kind=kind,
            finish_reason=self.finish_reason,
            input_tokens=self.input_tokens,
            output_tokens=self.output_tokens,
            model_seconds=self.seconds,
            action=action,
            exit_code=exit_code,
            outcome=outcome,
            command_seconds=command_seconds,
            problem=problem,
            answer=answer[:ANSWER_CHARS],
            why=why[:ANSWER_CHARS],
            output=output[:OUTPUT_CHARS],
        )


def steps_left_note(left: int) -> str:
    """What a call is told about the answers it has left, itself included (module docstring)."""
    if left > FINAL_STEPS:
        return f"<steps_left>{left}</steps_left> You have {left} answers left, this one included."
    return (
        f"<steps_left>{left}</steps_left> You have {left} answer{'s' if left != 1 else ''} left, this one included: "
        "run no more commands. Answer now with your ```json block, the submission, and no bash block."
    )


def run_loop(
    client: ModelClient,
    sandbox: Sandbox,
    workspace: Workspace,
    messages: list[Message],
    *,
    model: str,
    temperature: float,
    stage: str,
    budget: Budget,
    config: LoopConfig,
    check: SubmissionCheck,
    max_calls: int | None = None,
    max_micro_usd: int | None = None,
) -> LoopResult:
    """Run the loop from `messages` (the system and task messages) until the model submits or the loop stops.
    `messages` grows with the conversation. `max_calls` and `max_micro_usd` hold the loop under the budget's own
    limits, to leave room for what runs after it. A model error or a sandbox failure ends the loop; neither raises.
    A model id that must never be called (a -contributor id, or one off the allowlist) raises before any call."""
    check_model(model)
    trajectory_deadline = budget.clock() + config.trajectory_timeout_s
    steps: list[Step] = []
    sandbox_seconds: list[float] = []
    format_errors = 0
    limit = budget.step_limit if max_calls is None else min(max_calls, budget.step_limit)

    def tell(content: str) -> None:
        """Add the next user message, ending with the answers left for the call it goes to."""
        left = limit - budget.calls
        messages.append(Message(role="user", content=f"{content}\n\n{steps_left_note(left)}" if left > 0 else content))

    if messages and messages[-1].role == "user" and limit > budget.calls:
        last = messages.pop()
        tell(last.content)

    def done(exit: Exit, reason: UncheckedReason | None, detail: str, submission: str | None = None) -> LoopResult:
        return LoopResult(
            exit=exit,
            reason=reason,
            detail=detail,
            submission=submission,
            steps=tuple(steps),
            sandbox_seconds=math.fsum(sandbox_seconds),
        )

    while True:
        stop = budget.stop(deadline=trajectory_deadline, calls=max_calls, micro_usd=max_micro_usd)
        if stop is not None:
            return done(*stop)
        request = ModelRequest(
            stage=stage,
            model=model,
            messages=tuple(messages),
            temperature=temperature,
            max_tokens=config.max_tokens,
            reasoning_effort=config.reasoning_effort,
        )
        asked = budget.clock()
        try:
            response = client.complete(request)
        except ModelCallError as err:
            budget.record(err.cost)
            return done("model_error", err.reason, f"the model call failed: {err}")
        except Exception as err:  # any other failure of the client is still a model failure for this attacker
            budget.record(None)
            return done("model_error", UncheckedReason.RUNTIME, f"the model client raised {type(err).__name__}: {err}")
        budget.record(response.cost)
        if response.cost is None:
            return done(
                "unpriced",
                UncheckedReason.BUDGET,
                "the model client returned a call with no cost, so the attacker's cost cap cannot hold",
            )
        messages.append(Message(role="assistant", content=response.text))
        call = _Call(
            response.finish_reason,
            response.usage.input_tokens,
            response.usage.output_tokens,
            max(0.0, budget.clock() - asked),
        )
        if response.truncated:
            why = f"cut off or empty (finish_reason {response.finish_reason}, {len(response.text)} characters)"
            steps.append(call.step(len(steps) + 1, "truncated", answer=response.text, why=why))
            note = (
                "Your answer was cut off or empty, so nothing ran. Answer again, shorter: a line of reasoning and one "
                "fenced block."
            )
        else:
            # Only bash and json blocks count; an example in another language beside them is prose, as in
            # mini-swe-agent, which reads its action blocks only.
            parsed = own_line_fences(response.text)
            blocks = [
                (m.group("info").lower(), m.group("body"))
                for m in _BLOCK.finditer(parsed)
                if m.group("info").lower() in ("bash", "json")
            ]
            if len(blocks) == 1 and blocks[0][0] == "bash" and blocks[0][1].strip():
                format_errors = 0
                action = blocks[0][1]
                try:
                    result = sandbox.run(workspace, [*SHELL, action], timeout_s=config.command_timeout_s)
                except Exception as err:  # the sandbox's failure ends the loop; it never reaches the model as output
                    steps.append(call.step(len(steps) + 1, "action", action=action))
                    return done(
                        "sandbox_error", UncheckedReason.RUNTIME, f"the sandbox raised {type(err).__name__}: {err}"
                    )
                sandbox_seconds.append(result.duration_s)
                seen = observation(result, config)
                steps.append(
                    call.step(
                        len(steps) + 1,
                        "action",
                        action=action,
                        exit_code=result.exit_code,
                        outcome=result.outcome,
                        command_seconds=result.duration_s,
                        problem=result.problem,
                        output=seen,
                    )
                )
                tell(seen)
                continue
            if len(blocks) == 1 and blocks[0][0] == "json":
                problem = check(blocks[0][1])
                if problem is None:
                    steps.append(call.step(len(steps) + 1, "submission"))
                    return done("submitted", None, "the model submitted", submission=blocks[0][1])
                steps.append(call.step(len(steps) + 1, "refused_submission", answer=response.text, why=problem))
                note = f"Your submission was not accepted: {problem} Fix it and submit again, or keep exploring."
            else:
                steps.append(call.step(len(steps) + 1, "format_error", answer=response.text, why=_unusable(parsed)))
                note = (
                    f"Your answer held {len(blocks)} fenced blocks of the kinds this loop reads. Answer with a line "
                    "of reasoning and exactly one block: a ```bash block with one command, or a ```json block with "
                    "your submission."
                )
        format_errors += 1
        if format_errors >= config.max_format_errors:
            return done(
                "format_errors", UncheckedReason.UNPARSED, f"{format_errors} answers in a row could not be used"
            )
        tell(note)


def own_line_fences(text: str) -> str:
    """An answer as the loop parses it: an opening ```bash or ```json fence that ends a line after other text moves to
    a line of its own, and nothing else changes. Spark writes "Fixing the regression.```bash" (FAL-637: 194 of its 243
    format errors in the rehearsal's attacks, and every one of the fixer's), which the fence rule reads as no block at
    all. A fence indented by spaces alone, a ```sh block, a fence named mid-line in prose, a closing fence and every
    other answer parse as before. Block bodies and the model's own text in the history stay unchanged."""
    lines: list[str] = []
    fence: str | None = None
    for line in text.split("\n"):
        if fence is not None:
            lines.append(line)
            if line.rstrip(" \t") == fence:
                fence = None
            continue
        match = _FENCE_AT_LINE_END.search(line)
        if match is not None and line[: match.start()].strip():
            lines.extend((line[: match.start()], line[match.start() :]))
            opening = _OPENING_FENCE.fullmatch(line[match.start() :])
        else:
            lines.append(line)
            opening = _OPENING_FENCE.fullmatch(line)
        if opening is not None:
            fence = opening.group("fence")
    return "\n".join(lines)


def _unusable(text: str) -> str:
    """Why an answer was a format error: how many blocks of the kinds this loop reads it held, and each fenced block's
    language, so a ```sh or ```python block, or no block at all, shows in the trajectory log."""
    infos = [m.group("info").lower() for m in _BLOCK.finditer(text)]
    read = [info for info in infos if info in ("bash", "json")]
    if read == ["bash"]:
        return "an empty bash block"
    return (
        f"{len(read)} bash or json blocks; fenced blocks seen: {', '.join(i or 'unlabelled' for i in infos) or 'none'}"
    )


def observation(result: RunResult, config: LoopConfig) -> str:
    """A command's exit code and output for the model, and, when the sandbox could not read the run, why. Output past
    the limit keeps its head and tail with the count of characters left out between them, so a traceback at the end
    and the command's first lines both stay."""
    if result.outcome is Outcome.TIMEOUT:
        status = f"<timeout>the command ran past {config.command_timeout_s:g} s and was killed</timeout>\n"
    else:
        status = f"<returncode>{'none' if result.exit_code is None else result.exit_code}</returncode>\n"
    if result.problem:
        problem = " ".join(result.problem.split())
        problem = problem if len(problem) <= PROBLEM_CHARS else problem[: PROBLEM_CHARS - 3] + "..."
        status += f"<problem>{problem}</problem>\n"
    parts = [(tag, text) for tag, text in (("stdout", result.stdout_tail), ("stderr", result.stderr_tail)) if text]
    if not parts:
        return status + "<output>(no output)</output>"
    share = config.observation_chars // len(parts)
    return status + "\n".join(
        f"<{tag}>\n{_head_and_tail(text.rstrip(chr(10)), share)}\n</{tag}>" for tag, text in parts
    )


def _head_and_tail(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    half = limit // 2
    left_out = len(text) - 2 * half
    return (
        f"{text[:half]}\n<elided>{left_out} characters left out; run a narrower command (head, tail, grep) to see "
        f"them</elided>\n{text[-half:]}"
    )
