"""The shared contracts every lane codes against: claims, targets, attack plans, repros, runs, findings, ledger entries,
and the model-client, sandbox, surrogate-builder and ledger interfaces.

The integrator owns this file. A lane that needs a change asks for it instead of editing it, so the four lanes stay
compatible. The spec's hard rules that a type can carry are enforced here, at construction: a blocker cannot exist
without a runnable repro and the run evidence its kind requires, and no model id ending in -contributor gets through.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Literal, Protocol, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

# Version 2: target ids carry no slash, claim ids are relative to their target, and a finding id starts with both.
SCHEMA_VERSION: Literal[2] = 2
SHA = r"^[0-9a-f]{7,64}$"

# Spec hard rule 2: a blocker fails on the patch 5 of 5 times.
BLOCKER_RUNS = 5
# SK3: the surrogate fixes built for each fix-validity or behavior claim, one temperature each.
SURROGATES_PER_CLAIM = 3
# SK3: what one surrogate's build may spend, an attacker's $0.25.
SURROGATE_CAP_MICRO_USD = 250_000
# SK3: the surrogates' own budget per target, outside every arm's per-target cap (the founder's call, 2026-10-29),
# so both arms judge a candidate by the same surrogates: 3 fix-validity claims x 3 surrogates x $0.25. The skeptic
# builds only as many as what is left of it covers, SURROGATE_CAP_MICRO_USD each, so it is crossed by one
# surrogate's spend at most.
SURROGATE_TARGET_CAP_MICRO_USD = 2_250_000


class Frozen(BaseModel):
    # Floats are finite: an inf or a NaN would serialize as null and come back as something else.
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)


def _utf8(text: str, where: str) -> str:
    """Refuse text that cannot be written as UTF-8 (lone surrogates), at the producer rather than the ledger."""
    try:
        text.encode("utf-8")
    except UnicodeEncodeError as e:
        raise ValueError(f"{where} is not valid UTF-8 text") from e
    return text


def _segment(text: str, where: str) -> str:
    """One segment of a finding id, <target_id>/<claim_id>/<attack or lens>: no slash, no whitespace or control
    character, so the id splits the one way."""
    if re.search(r"[/\s\x00-\x1f\x7f]", text):
        raise ValueError(f"{where} {text!r} has a slash, whitespace or a control character")
    return text


# ---------------------------------------------------------------- models

ALLOWED_MODELS: frozenset[str] = frozenset(
    {
        "muse-spark-1.1",
        "muse-spark-1.2",
        "muse-spark-1.3",
        "muse-wonder-1.0",
        "sam-3.1",
        "muse-glimmer-30b",  # open weights on Falden's own vLLM (its --served-model-name), never Meta's API
    }
)


class ModelRefusedError(ValueError):
    """A model id that must never be called."""


def check_model(model_id: str) -> str:
    """Refuse every -contributor id and anything off the allowlist (spec hard rule 5)."""
    if "contributor" in model_id.lower():
        raise ModelRefusedError(f"{model_id}: -contributor model ids are never used")
    if model_id not in ALLOWED_MODELS:
        raise ModelRefusedError(f"{model_id}: not on the model allowlist")
    return model_id


# ---------------------------------------------------------------- claims and targets


class ClaimType(StrEnum):
    FIXED_X = "fixed_x"
    NOTHING_ELSE_CHANGED = "nothing_else_changed"
    BACKWARD_COMPATIBLE = "backward_compatible"
    ADDS_BEHAVIOR = "adds_behavior"
    TESTS_PASS = "tests_pass"
    PERF = "perf"
    UI = "ui"


class SpanSource(StrEnum):
    PR_TITLE = "pr_title"
    PR_BODY = "pr_body"
    COMMIT_MESSAGE = "commit_message"
    SESSION_SUMMARY = "session_summary"
    ISSUE = "issue"
    TEMPLATE = "template"  # Tier A's fixed claims, which have no PR body


class Span(Frozen):
    """The quoted source of a claim (IN1)."""

    source: SpanSource
    text: str = Field(min_length=1)
    start: int | None = Field(default=None, ge=0)
    end: int | None = Field(default=None, ge=0)


class Claim(Frozen):
    """A claim's id is relative to its target: the type alone for a template claim ('fixed_x'), the type and a 10-hex
    digest for an extracted one ('tests_pass:0123456789'). The target id goes only in the finding id."""

    id: str = Field(min_length=1)
    type: ClaimType
    text: str = Field(min_length=1)
    span: Span
    high_risk: bool = False

    @model_validator(mode="after")
    def _relative_id(self) -> Self:
        kind, sep, digest = self.id.partition(":")
        if kind != self.type.value or (sep and re.fullmatch(r"[0-9a-f]{10}", digest) is None):
            raise ValueError(f"claim id {self.id!r} is its type, or its type, a colon and 10 hex digits")
        return self


class RunnerKind(StrEnum):
    """How a repository runs its tests. Django and sympy have their own runners (54% of the Tier A pool)."""

    PYTEST = "pytest"  # pytest itself, or a launcher that ends in it (sphinx's tox)
    DJANGO = "django"  # tests/runtests.py, test labels
    SYMPY = "sympy"  # bin/test, file paths


class Tier(StrEnum):
    A = "A"  # SWE-bench Verified, SWE-ABS strengthened
    B = "B"  # time machine
    C = "C"  # live hunting
    LIVE = "live"  # the GitHub App on a real PR


# A digest-pinned image in docker's reference grammar (distribution/reference): an optional registry host and port, a
# lowercase repository path, an optional tag, then the sha256 digest. It starts with a letter or a digit, so docker
# never reads it as a flag.
_HOST_PART = r"(?:[a-zA-Z0-9]|[a-zA-Z0-9][a-zA-Z0-9-]*[a-zA-Z0-9])"
_PATH_PART = r"[a-z0-9]+(?:(?:[._]|__|-+)[a-z0-9]+)*"
IMAGE_REF = (
    rf"^(?:{_HOST_PART}(?:\.{_HOST_PART})*(?::[0-9]+)?/)?{_PATH_PART}(?:/{_PATH_PART})*"
    r"(?::[A-Za-z0-9_][A-Za-z0-9_.-]{0,127})?@sha256:[0-9a-f]{64}$"
)


class Target(Frozen):
    """One agent change to verify. Its id has no slash (Tier A ids are <entry>~<instance_id>)."""

    id: str = Field(min_length=1)
    tier: Tier
    repo: str = Field(min_length=1)
    base_sha: str = Field(pattern=SHA)
    patch_sha: str | None = Field(default=None, pattern=SHA)
    patch: str = Field(min_length=1)  # the agent's change as a unified diff
    title: str = ""
    body: str = ""
    issue: str = ""  # Tier A: the issue text
    commit_messages: str = ""
    session_summary: str = ""
    image: str | None = Field(default=None, pattern=IMAGE_REF)  # digest-pinned only
    test_runner: RunnerKind = RunnerKind.PYTEST
    test_command: str = ""  # the repository's own test command (Tier A: SWE-bench's spec); empty means plain pytest

    @field_validator("id")
    @classmethod
    def _id(cls, v: str) -> str:
        return _segment(v, "target id")

    @field_validator("image")
    @classmethod
    def _image_name(cls, v: str | None) -> str | None:
        if v is not None:
            ref = v.rsplit("@", 1)[0]
            name = ref[: ref.rfind(":")] if ref.rfind(":") > ref.rfind("/") else ref
            if len(name) > 255:
                raise ValueError("an image name is at most 255 characters")
        return v


# ---------------------------------------------------------------- plans


class Lens(StrEnum):
    REGRESSION = "regression"
    CORRECTNESS = "correctness"
    DIFFERENTIAL = "differential"
    CRASH_RESTART = "crash_restart"
    INTENT_RISK = "intent_risk"
    MUTATION = "mutation"


class ContextMask(StrEnum):
    DIFF = "diff"
    DIFF_FUNCTIONS = "diff_functions"
    DIFF_TESTS = "diff_tests"
    CLAIM_ONLY = "claim_only"
    CLAIM_SPEC = "claim_spec"


class UncheckedReason(StrEnum):
    BUDGET = "budget"
    TIMEOUT = "timeout"
    RUNTIME = "runtime"
    UNPARSED = "unparsed"
    NO_ENV = "no_env"
    NO_INPUT = "no_input"  # the target has no agent text to take claims from


class ModelCallError(RuntimeError):
    """A model call that could not complete. Its reason goes on the receipt's unchecked list, and what it still spent
    (a failed call can be metered) goes in `cost`, so it reaches Finding.costs."""

    def __init__(self, message: str, reason: UncheckedReason, *, cost: StageCost | None = None) -> None:
        super().__init__(message)
        self.reason = reason
        self.cost = cost

    def __reduce__(self) -> tuple[object, ...]:
        # So it crosses a process pool: the default pickles only the message and loses the reason.
        return (type(self), (str(self), self.reason), {"cost": self.cost})


class Attack(Frozen):
    id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    lens: Lens
    hypothesis: str = Field(min_length=1)
    mask: ContextMask
    temperature: float = Field(ge=0, le=2)
    # True for the plan's one attack per target that AT3 takes: it reruns the repository's existing tests, so its mask
    # and hypothesis are not used. Every other regression attack goes to a model attacker that writes new tests of
    # behavior the change should not alter (contract v3.2).
    existing_tests: bool = False

    @field_validator("claim_id")
    @classmethod
    def _claim(cls, v: str) -> str:
        return _segment(v, "claim id")

    @model_validator(mode="after")
    def _existing_tests_are_regression(self) -> Self:
        if self.existing_tests and self.lens is not Lens.REGRESSION:
            raise ValueError("only a regression attack reruns the existing tests")
        return self


class Unchecked(Frozen):
    """Something the run did not check, and why; the receipt lists every one (spec hard rule 6)."""

    what: str = Field(min_length=1)  # a claim id, a lens or an attack id
    reason: UncheckedReason
    detail: str = ""


class AttackPlan(Frozen):
    target_id: str = Field(min_length=1)
    attacks: tuple[Attack, ...]
    unchecked: tuple[Unchecked, ...] = ()


# ---------------------------------------------------------------- repros and runs


class CatchKind(StrEnum):
    REGRESSION = "regression"  # fails on the patch, passes on base
    FIX_VALIDITY = "fix_validity"  # fails on the patch, passes on an independent plausible fix
    BEHAVIOR = "behavior"  # adds_behavior; same evidence as FIX_VALIDITY until its oracle is defined


class Side(StrEnum):
    BASE = "base"
    PATCH = "patch"
    SURROGATE = "surrogate"  # an independent plausible fix (SK3)
    GOLD = "gold"  # Tier A scoring only


class Outcome(StrEnum):
    PASS = "pass"
    FAIL = "fail"
    ERROR = "error"
    TIMEOUT = "timeout"
    # Skipped, or an expected failure that failed as expected (CaseResult.expected_failure tells them apart). Neither
    # counts as a pass anywhere, except that R24's gold check counts an expected failure the way swebench 4.1.0's
    # grading counts XFAIL: as passing (harness/grading.py, test_passed).
    SKIPPED = "skipped"


class FailureType(StrEnum):
    ASSERTION = "assertion"
    CRASH_IN_CHANGED = "crash_in_changed"  # top non-test frame in a changed file
    CRASH_ELSEWHERE = "crash_elsewhere"
    IMPORT = "import"
    NAME = "name"
    FIXTURE = "fixture"
    TIMEOUT = "timeout"
    INFRA = "infra"
    OTHER = "other"


# Spec hard rule 3: only these failures count.
COUNTED_FAILURES: frozenset[FailureType] = frozenset({FailureType.ASSERTION, FailureType.CRASH_IN_CHANGED})


class Repro(Frozen):
    id: str = Field(min_length=1)
    attack_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    lens: Lens
    kind: CatchKind
    files: dict[str, str]  # path relative to the repository root -> content
    command: tuple[str, ...] = Field(min_length=1)
    # The runner's own case ids this repro runs, as CaseResult.name gives them: pytest node ids, Django labels, sympy
    # path::name or bare names. Empty means every case the repro's test files define.
    cases: tuple[str, ...] = ()
    # The sentence of the issue (or of the agent's own text) the repro enforces, as the attacker quoted it (SK2). The
    # skeptic refuses a fix-validity or behavior repro whose quote is not in that text verbatim, spacing aside. Empty
    # for a regression repro, whose oracle is base (contract v3.2).
    quote: str = ""

    @field_validator("claim_id")
    @classmethod
    def _claim(cls, v: str) -> str:
        return _segment(v, "claim id")

    @field_validator("files")
    @classmethod
    def _relative_paths(cls, v: dict[str, str]) -> dict[str, str]:
        if not v:
            raise ValueError("a repro writes at least one file")
        for path, content in v.items():
            if path.startswith("/") or ".." in path.split("/"):
                raise ValueError(f"repro path {path!r} must be relative, inside the repository")
            _utf8(path, "a repro path")
            _utf8(content, f"repro file {path!r}")
        return v

    @field_validator("cases")
    @classmethod
    def _cases(cls, v: tuple[str, ...]) -> tuple[str, ...]:
        seen: set[str] = set()
        for case in v:
            if not case.strip():
                raise ValueError("a repro's case id is blank")
            _utf8(case, "a repro's case id")
            if case in seen:
                raise ValueError(f"a repro names case {case!r} twice")
            seen.add(case)
        return v

    @field_validator("quote")
    @classmethod
    def _quote(cls, v: str) -> str:
        return _utf8(v, "a repro's quote")


class RunResult(Frozen):
    side: Side
    outcome: Outcome
    failure_type: FailureType | None = None
    exit_code: int | None = None
    seed: int | None = None
    fresh_container: bool
    duration_s: float = Field(ge=0)
    stdout_tail: str = ""  # truncated before any model sees it (RT2)
    stderr_tail: str = ""
    image_digest: str = ""
    # Why the run could not be read or classified (an expected case that never ran, an unreadable report); empty when
    # it was read.
    problem: str = ""

    @field_validator("stdout_tail", "stderr_tail")
    @classmethod
    def _text(cls, v: str) -> str:
        return _utf8(v, "a run's output")

    @field_validator("problem")
    @classmethod
    def _problem(cls, v: str) -> str:
        return _utf8(v, "a run's problem")

    @model_validator(mode="after")
    def _failure_needs_type(self) -> Self:
        if self.outcome is Outcome.FAIL and self.failure_type is None:
            raise ValueError("a failed run names its failure type")
        if self.outcome is Outcome.PASS and self.failure_type is not None:
            raise ValueError("a passing run has no failure type")
        return self


class CaseResult(Frozen):
    """One test case as the repository's own runner reported it, before a run's cases reduce to one RunResult."""

    name: str = Field(min_length=1)  # the runner's own id: a pytest node id, a Django label, a sympy function name
    outcome: Outcome
    failure_type: FailureType | None = None
    raising_file: str | None = None  # for a crash: the top non-test frame's file, relative to the repository
    message: str = ""
    file: str | None = None  # the test file, relative to the repository, where names repeat across files (sympy)
    # True when the runner reported an expected failure that failed as expected: pytest's xfail, unittest's
    # expectedFailure, sympy's XFAIL. Its outcome is SKIPPED, so it never counts as a pass except in R24's gold check.
    expected_failure: bool = False

    @field_validator("name", "message")
    @classmethod
    def _text(cls, v: str) -> str:
        return _utf8(v, "a test case's text")

    @field_validator("file")
    @classmethod
    def _relative_file(cls, v: str | None) -> str | None:
        if v is None:
            return v
        if not v or v.startswith("/") or ".." in v.split("/"):
            raise ValueError(f"case file {v!r} must be relative, inside the repository")
        return _utf8(v, "a test case's file")

    @model_validator(mode="after")
    def _failure_needs_type(self) -> Self:
        if self.outcome is Outcome.FAIL and self.failure_type is None:
            raise ValueError("a failed case names its failure type")
        if self.outcome is Outcome.PASS and self.failure_type is not None:
            raise ValueError("a passing case has no failure type")
        return self

    @model_validator(mode="after")
    def _expected_failure_is_skipped(self) -> Self:
        if self.expected_failure and self.outcome is not Outcome.SKIPPED:
            raise ValueError("a case that failed as expected is skipped, so it never counts as a pass")
        return self


def answers(case: CaseResult, wanted: str, *, repo: str) -> bool:
    """Whether a run's case answers an expected case id (``Sandbox.run_test``'s ``expect``: a repro's cases, or R24's
    FAIL_TO_PASS and PASS_TO_PASS). It is the one matching every lane uses, so the run phase's never-ran check and
    R24's gold check cannot drift apart. ``repo`` is the repository of the target the run's workspace was prepared
    from (``Target.repo``).

    A case answers ``wanted`` when its name is ``wanted``, in any file (so a bare sympy name, as SWE-bench writes it,
    matches every case of that name); for a case that names its file (sympy), when ``wanted`` is ``path::name`` with
    that file and name; for a pytest node id, when ``wanted`` is the id without its parameters; and when ``wanted`` is
    the id swebench 4.1.0's log parser for ``repo`` records for the case. Two of its parsers rename an id before they
    record it (harness/log_parsers/python.py, in the wheel eval/manifest.json pins). parse_log_pytest_options, the
    parser of psf/requests, pylint-dev/pylint and pydicom/pydicom (lines 29 to 61), splits the id with
    ``(.*?)\\[(.*)\\]`` and, when the part in brackets starts with one slash (not two) and has no '*', keeps only a
    slash and what follows the last slash: pylint's
    ``tests/test_self.py::TestRunTC::test_stdin[/testbed/tests/mymodule.py-mymodule-/testbed/tests/mymodule.py]`` is
    recorded as ``tests/test_self.py::TestRunTC::test_stdin[/mymodule.py]``, and anything after the last ']' is
    dropped. parse_log_matplotlib (lines 229 to 250) writes 'MouseButton.LEFT' as '1' and 'MouseButton.RIGHT' as
    '3'. Every other repository's parser records the id as the run printed it, but for one rule not mirrored here:
    parse_log_pytest_v2 (astropy, scikit-learn, sphinx; lines 144 to 170) first deletes every '[<digits>m' from the
    line, and no FAIL_TO_PASS or PASS_TO_PASS id of the 251-instance pool shows a trace of it (none holds
    '[<digits>m', none has more ']' than '['; read 2026-10-28). A name with whitespace in it is never renamed: the
    harness reads an id as one whitespace-separated word of its line, so it never records that name whole.

    Several cases of one run can answer one id (a sympy name repeated across files, or two cases the harness records
    under one id); a reader of the run takes every one of them, and the id passes only when all of them ran and
    passed. A case keeps its own name, the runner's id, so a rerun by that name still selects it."""
    if case.name == wanted:
        return True
    if "::" not in wanted:
        return False
    if case.file is not None:
        return wanted == f"{case.file}::{case.name}"
    if "[" not in wanted and case.name.startswith(wanted + "["):
        return True
    if any(c.isspace() for c in case.name):
        return False
    recorded = case.name
    if repo in ("psf/requests", "pylint-dev/pylint", "pydicom/pydicom"):
        found = re.search(r"(.*?)\[(.*)\]", recorded)
        if found is not None:
            main, option = found.groups()
            if option.startswith("/") and not option.startswith("//") and "*" not in option:
                option = "/" + option.split("/")[-1]
            recorded = f"{main}[{option}]"
    elif repo == "matplotlib/matplotlib":
        recorded = recorded.replace("MouseButton.LEFT", "1").replace("MouseButton.RIGHT", "3")
    return recorded == wanted


# ---------------------------------------------------------------- findings and the ledger


class Status(StrEnum):
    BLOCKER = "blocker"
    SUSPECTED = "suspected"
    FLAKY = "flaky"
    REFUTED = "refuted"
    POSSIBLY_TOXIC = "possibly_toxic"
    S_SAT_INCONCLUSIVE = "s_sat_inconclusive"
    NOT_REPRODUCED = "not_reproduced"
    UNCHECKED = "unchecked"


class StageCost(Frozen):
    """What one stage spent. Money is integer micro-dollars, never a float."""

    stage: str = Field(min_length=1)
    model: str | None = None  # None for a code-only stage
    input_tokens: int = Field(default=0, ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(default=0, ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)  # part of output_tokens, metered separately for Wonder
    micro_usd: int = Field(default=0, ge=0)
    seconds: float = Field(default=0.0, ge=0)


class ReplayKey(Frozen):
    base_sha: str = Field(pattern=SHA)
    patch_sha: str = Field(pattern=SHA)
    harness_hash: str = Field(min_length=1)
    seed: int
    image_digest: str = Field(min_length=1)


class Finding(Frozen):
    id: str = Field(min_length=1)
    target_id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    lens: Lens
    status: Status
    kind: CatchKind | None = None
    unchecked_reason: UncheckedReason | None = None
    repro: Repro | None = None
    runs: tuple[RunResult, ...] = ()
    quote: str | None = None  # the claim or issue sentence that requires the behavior (SK2)
    refutation: str | None = None  # only a different-model refutation's evidence, if one ran
    # Why the finding has its status, in plain words, when that is not a different-model refutation: an unchecked
    # cause, flaky k of 5, s_sat_inconclusive, plan overflow, the per-target cap and the like.
    detail: str | None = None
    replay: ReplayKey | None = None
    costs: tuple[StageCost, ...] = ()  # incremental: what producing this entry spent, never a restatement
    # A state like status, not an increment like costs: the rounds so far. A rollup takes the most any entry records.
    critic_rounds: int = Field(default=0, ge=0, le=3)
    wall_seconds: float | None = Field(default=None, ge=0)  # from the target's start to this status

    @field_validator("target_id")
    @classmethod
    def _target(cls, v: str) -> str:
        return _segment(v, "target id")

    @field_validator("claim_id")
    @classmethod
    def _claim(cls, v: str) -> str:
        return _segment(v, "claim id")

    @model_validator(mode="after")
    def _hard_rules(self) -> Self:
        # Finding ids are unique across targets: <target_id>/<claim_id>/<attack or lens>.
        prefix = f"{self.target_id}/{self.claim_id}/"
        if not self.id.startswith(prefix) or len(self.id) == len(prefix):
            raise ValueError("a finding id starts with its target id and a slash, then its claim id and a slash")
        if self.status is Status.UNCHECKED and self.unchecked_reason is None:
            raise ValueError("an unchecked finding names its reason")
        if self.status is not Status.BLOCKER:
            return self
        # Hard rule 1: no runnable repro means no blocker.
        if self.repro is None or self.kind is None or self.replay is None or not self.quote:
            raise ValueError("a blocker needs a repro, its kind, a replay key and the quoted claim or issue sentence")
        if self.kind is not self.repro.kind:
            raise ValueError("a blocker's kind is its repro's kind")
        # Hard rules 2 and 3: the patch fails 5 of 5 times, on a counted failure, with no mixed result.
        patch = [r for r in self.runs if r.side is Side.PATCH]
        if len(patch) < BLOCKER_RUNS or not all(
            r.outcome is Outcome.FAIL and r.failure_type in COUNTED_FAILURES for r in patch
        ):
            raise ValueError(
                f"a blocker fails on the patch in every run, at least {BLOCKER_RUNS}, on an assertion or a crash in "
                "changed code"
            )
        if self.kind is CatchKind.REGRESSION:
            base = [r for r in self.runs if r.side is Side.BASE]
            if len(base) < BLOCKER_RUNS or not all(r.outcome is Outcome.PASS for r in base):
                raise ValueError(f"a regression catch passes on base in every run, at least {BLOCKER_RUNS}")
        elif not any(r.side in (Side.SURROGATE, Side.GOLD) and r.outcome is Outcome.PASS for r in self.runs):
            raise ValueError("a fix-validity catch passes on at least 1 plausible surrogate (or gold, at scoring)")
        return self


class LedgerEntry(Frozen):
    """One state of one finding. A later state of the same finding is a new entry with the same finding id, and its
    costs are only what that step spent, so rollups sum every entry."""

    v: Literal[2] = SCHEMA_VERSION
    at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")  # UTC
    receipts_version: str = Field(min_length=1)
    finding: Finding


# ---------------------------------------------------------------- interfaces


class Message(Frozen):
    role: Literal["system", "user", "assistant"]
    content: str


class ModelRequest(Frozen):
    stage: str = Field(min_length=1)
    model: str
    messages: tuple[Message, ...] = Field(min_length=1)
    temperature: float = Field(ge=0, le=2)
    max_tokens: int = Field(gt=0)
    reasoning_effort: Literal["low", "medium", "high"] | None = None
    json_schema: dict[str, object] | None = None  # the raw JSON Schema of the wanted response
    schema_name: str | None = None  # names the response format; defaults to the schema's title, then the stage

    @field_validator("model")
    @classmethod
    def _allowed(cls, v: str) -> str:
        return check_model(v)


class Usage(Frozen):
    input_tokens: int = Field(ge=0)
    cached_tokens: int = Field(default=0, ge=0)
    output_tokens: int = Field(ge=0)
    reasoning_tokens: int = Field(default=0, ge=0)  # part of output_tokens, metered separately for Wonder


class ModelResponse(Frozen):
    model: str
    text: str
    finish_reason: str
    usage: Usage
    request_id: str | None = None
    parsed: dict[str, object] | None = None
    cost: StageCost | None = None  # this call's cost, priced by the client

    @property
    def truncated(self) -> bool:
        """A truncated response invalidates the item (RT1): one cut off at the token limit, or an empty answer, which
        can come back with finish_reason 'stop'."""
        return self.finish_reason == "length" or not self.text.strip()


class ModelClient(Protocol):
    def complete(self, request: ModelRequest) -> ModelResponse: ...


DOCKER_SIZE = r"^[1-9][0-9]*[bkmg]?$"
_MOUNT_PATH = re.compile(r"/[A-Za-z0-9._/-]+")


def docker_bytes(size: str) -> int:
    """A docker size such as '64m' or '4g', in bytes."""
    m = re.fullmatch(r"([1-9][0-9]*)([bkmg]?)", size)
    if m is None:
        raise ValueError(f"not a docker size: {size!r}")
    return int(m.group(1)) * {"": 1, "b": 1, "k": 1024, "m": 1024**2, "g": 1024**3}[m.group(2)]


class SandboxLimits(Frozen):
    """The run phase's isolation (RT2). The sandbox and the eval VM's runsc check both render their docker flags from
    here, so they cannot drift apart. The root is read-only, and the workspace, /tmp and /dev/shm mounts are the only
    writable places; they are RAM, so their sizes are the disk limit and must fit under the memory cap."""

    network: Literal["none"] = "none"
    cap_drop: Literal["ALL"] = "ALL"
    no_new_privileges: Literal[True] = True
    pids_limit: int = Field(default=512, gt=0)
    memory: str = Field(default="4g", pattern=DOCKER_SIZE)  # also --memory-swap, so there is no swap
    cpus: float = Field(default=2.0, gt=0)
    shm_size: str = Field(default="64m", pattern=DOCKER_SIZE)
    ipc: Literal["private"] = "private"
    msgqueue: Literal[0] = 0  # --ulimit msgqueue: no POSIX message queues
    read_only: Literal[True] = True
    user: str = Field(default="65534:65534", pattern=r"^[0-9]+:[0-9]+$")  # nobody:nogroup; uid:gid, never root
    workspace_bytes: int = Field(default=1024**3, gt=0)
    tmp_bytes: int = Field(default=256 * 1024**2, gt=0)

    @model_validator(mode="after")
    def _sound(self) -> Self:
        if any(int(i) == 0 for i in self.user.split(":")):
            raise ValueError("the run phase runs as an unprivileged user, never root")
        if self.workspace_bytes + self.tmp_bytes + docker_bytes(self.shm_size) >= docker_bytes(self.memory):
            raise ValueError("the workspace, /tmp and /dev/shm are RAM, so together they fit under the memory cap")
        return self

    def limit_flags(self) -> list[str]:
        """The limits the build and run phases share."""
        return [
            f"--cap-drop={self.cap_drop}",
            "--security-opt=no-new-privileges",
            f"--pids-limit={self.pids_limit}",
            f"--memory={self.memory}",
            f"--memory-swap={self.memory}",
            f"--cpus={self.cpus:g}",
            f"--shm-size={self.shm_size}",
            f"--ipc={self.ipc}",
            f"--ulimit=msgqueue={self.msgqueue}",
        ]

    def run_flags(self, workspace: str = "/workspace", tmp: str = "/tmp") -> list[str]:
        """Every run-phase flag, in the order `docker create` gets them. `workspace` may be an image's repository path
        (for example /testbed), so its mount covers whatever the image baked there. The workspace mount belongs to the
        run user (a fresh tmpfs is root's, mode 755); /tmp is mode 1777, as usual."""
        for path in (workspace, tmp):
            parts = PurePosixPath(path).parts
            if not _MOUNT_PATH.fullmatch(path) or len(parts) < 2 or parts[1] in ("dev", "proc", "sys") or ".." in parts:
                raise ValueError(f"{path!r} cannot be a writable mount in the run phase")
        if PurePosixPath(workspace) == PurePosixPath(tmp):
            raise ValueError("the workspace and /tmp are different mounts")
        uid, gid = self.user.split(":")
        return [
            f"--network={self.network}",
            *self.limit_flags(),
            f"--user={self.user}",
            "--read-only",
            f"--tmpfs={workspace}:rw,exec,nosuid,nodev,size={self.workspace_bytes},uid={uid},gid={gid},mode=0755",
            f"--tmpfs={tmp}:rw,exec,nosuid,nodev,size={self.tmp_bytes},mode=1777",
        ]


RUN_PHASE_LIMITS = SandboxLimits()


class PatchMethod(StrEnum):
    """How a workspace's patch applied: the SWE-bench harness's chain, in its order (swebench 4.1.0,
    harness/run_evaluation.py GIT_APPLY_CMDS). Each command runs on the tree the previous attempt left, as the harness
    does, and the first exit 0 wins, so Receipts runs the code the Tier A labels were graded on (REG-569)."""

    GIT_APPLY = "git_apply"  # git apply --verbose
    GIT_APPLY_REJECT = "git_apply_reject"  # git apply --verbose --reject
    GNU_PATCH_FUZZ = "gnu_patch_fuzz"  # patch --batch --fuzz=5 -p1 -i


class Workspace(Frozen):
    id: str = Field(min_length=1)
    side: Side
    root: str = Field(min_length=1)  # the repository root inside the sandbox
    image_digest: str = Field(min_length=1)
    patch_method: PatchMethod | None = None  # None when the workspace was prepared with no patch


class SuiteRunner(Protocol):
    """A repository's own test runner: the command that runs some tests, and its output read back as cases. It runs
    nothing itself; the sandbox runs the command and hands back what the run printed and the report it wrote."""

    kind: RunnerKind

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        """`tests` may be paths relative to the repository root or the runner's own ids at any granularity, including
        a Django module label such as 'validators.tests'; `report_path` is unique to the run, so a report left by an
        earlier run, or planted by the code under test, is never read as this one's."""
        ...

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        """Every case the run reported, each failure typed against the changed files (spec hard rule 3)."""
        ...


class Sandbox(Protocol):
    """run() returns a coarse result: PASS on exit 0, TIMEOUT when killed for time, ERROR otherwise, with no failure
    type but TIMEOUT. run_test() runs tests with the target's runner and returns the classified result: FAIL with its
    type, SKIPPED, PASS, ERROR (no report or no output it can read: infra) or TIMEOUT. It reads the run's output and
    report before a fresh container is removed, and it checks the exit code against the cases in every mode."""

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace: ...

    def run(
        self,
        workspace: Workspace,
        command: Sequence[str],
        *,
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
    ) -> RunResult: ...

    def run_test(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: SuiteRunner,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
        expect: Sequence[str] | None = None,
    ) -> RunResult:
        """With `expect` (a repro's own case names, or R24's SWE-bench ids), PASS needs every expected case to have run
        and passed. A case answers an expected id as `answers` matches them, for the repository of the target the
        workspace was prepared from: by its own name, or by the id swebench 4.1.0's log parser records for it there.
        An expected case that never ran is ERROR, typed IMPORT when the run reports its test module's import error and
        INFRA otherwise. Every case that answers one id counts (a name that repeats across files in sympy, cases the
        harness records under one id), and all of them must pass."""
        ...

    def close(self, workspace: Workspace) -> None: ...


# ---------------------------------------------------------------- the pipeline's seams


class Arm(Frozen):
    """One way to run a target's attackers. A pilot runs two arms on the same targets and the same plan, with seeds,
    step limits and the SK1 gates equal, so only the attacker model differs (REG-561)."""

    name: str = Field(pattern=r"^[a-z0-9][a-z0-9_-]{0,31}$")
    attacker_model: str = "muse-spark-1.3"
    step_limit: int = Field(default=20, ge=1, le=30)  # 30 for crash-restart
    seed: int = 0

    @field_validator("attacker_model")
    @classmethod
    def _allowed(cls, v: str) -> str:
        return check_model(v)


class Divergence(Frozen):
    """A difference between base and patch that the differential attacker (AT4) found and could not make a blocker:
    its intent gate called it minor, non-deterministic, intended or uncertain, or quoted no contradiction it could
    verify (decision D3, 2026-10-30). It reaches the ledger as a suspected finding with `reason`, never as a catch."""

    attack_id: str = Field(min_length=1)
    example: str = Field(min_length=1)  # the usage example, reduced, as it ran on both sides
    base_output: str
    patch_output: str
    reason: str = Field(min_length=1)  # why it is suspected and not a blocker

    @field_validator("example", "base_output", "patch_output", "reason")
    @classmethod
    def _text(cls, v: str) -> str:
        return _utf8(v, "a divergence's text")


class AttackOutcome(Frozen):
    """What one attack produced for the skeptic: its candidate repros, what it spent, what it could not check, and the
    suspected divergences that are no candidate (AT4)."""

    attack_id: str = Field(min_length=1)
    repros: tuple[Repro, ...] = ()
    costs: tuple[StageCost, ...] = ()
    unchecked: tuple[Unchecked, ...] = ()
    suspected: tuple[Divergence, ...] = ()

    @model_validator(mode="after")
    def _own_repros(self) -> Self:
        if any(r.attack_id != self.attack_id for r in self.repros):
            raise ValueError("an attack outcome holds only its own attack's repros")
        if any(d.attack_id != self.attack_id for d in self.suspected):
            raise ValueError("an attack outcome holds only its own attack's divergences")
        return self


class Attacker(Protocol):
    """One lens's attacker. It prepares the workspaces it needs through the sandbox, closes them, and never raises
    for a model or sandbox failure: those come back as unchecked items with their reasons (spec hard rule 6)."""

    lens: Lens

    def attack(
        self,
        target: Target,
        attack: Attack,
        *,
        arm: Arm,
        client: ModelClient,
        sandbox: Sandbox,
        runner: SuiteRunner,
    ) -> AttackOutcome: ...


class Surrogate(Frozen):
    """An independent fix of a claim's issue (SK3), written by a fixer that never saw the agent's patch, and whether it
    is plausible (R5): the existing tests over its changed source files that pass on base still pass on it, and the
    tests it wrote pass on it. `why` says why it is or is not. A build that produced no usable diff (the fixer gave
    up, changed only test files, or wrote a diff over the cap) is kept with an empty patch, never plausible, so its
    reason reaches the finding and its costs are counted. `costs` are what building and checking it spent."""

    id: str = Field(min_length=1)
    claim_id: str = Field(min_length=1)
    patch: str  # a unified diff against base, UTF-8; empty only for a failed build
    plausible: bool
    why: str = Field(min_length=1)
    costs: tuple[StageCost, ...] = ()

    @field_validator("claim_id")
    @classmethod
    def _claim(cls, v: str) -> str:
        return _segment(v, "claim id")

    @field_validator("patch", "why")
    @classmethod
    def _text(cls, v: str) -> str:
        return _utf8(v, "a surrogate's text")

    @model_validator(mode="after")
    def _plausible_has_a_patch(self) -> Self:
        if self.plausible and not self.patch.strip():
            raise ValueError("a plausible surrogate has a patch")
        return self


class SurrogateBuilder(Protocol):
    """Builds up to `k` surrogate fixes of one claim's issue on base, through the sandbox and the target's runner.
    It never raises for a model or sandbox failure: a failed build is absent, or present with plausible=False and its
    reason."""

    def build(
        self, target: Target, claim: Claim, *, sandbox: Sandbox, runner: SuiteRunner, k: int
    ) -> tuple[Surrogate, ...]: ...


class Ledger(Protocol):
    def append(self, entry: LedgerEntry) -> None: ...

    def entries(self, target_id: str | None = None) -> Sequence[LedgerEntry]: ...
