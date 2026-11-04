"""RT1, the Meta client: contracts.ModelClient over the Meta Model API's OpenAI-compatible Chat Completions endpoint.

A call meets these checks in order:

1. The model id passes contracts.check_model (no id ending in -contributor, nothing off the allowlist), is not the
   self-hosted Glimmer model, and has a list price in LIST_PRICES. All of it happens before any request is built; an
   unpriced model is refused, never guessed.
2. The spend meter reserves the call's worst case: its token bounds (every request byte an input token, plus the
   whole output budget) at the highest list price on file, whichever model was asked for, since another model may
   answer. It refuses every call until an operator arms the spend file after confirming the Meta credit is applied
   ('receipts spend arm --credit-confirmed'). Once armed, it refuses the call when metered spend has reached the
   cap, or when the worst case would carry spend past it. The cap defaults to 150,000,000 micro-dollars, the $150
   Meta credit, so the default never spends cash. Only an arm marked above the credit ('receipts spend arm
   --above-credit'), on a spend file named with RECEIPTS_SPEND_FILE, lifts it, to at most 425,000,000 micro-dollars,
   the founder's $425 ceiling, and the spend past the credit is cash.
3. The tokens-per-minute governor waits until the model's tokens over the last 60 seconds leave room for the call.
4. A 429, a 5xx other than 504, and a transport timeout or dropped connection are retried at most 5 times, with
   exponential backoff and jitter that never waits less than Retry-After.
5. A 200 response is validated with Pydantic, metered from its usage at the higher of the requested model's and the
   answering model's list prices, in integer micro-dollars, so the meter never undercounts whichever one Meta bills,
   and logged with its finish_reason, tokens and cached_tokens under its stage. An answer from a model with no list
   price, or off the allowlist, is metered at the highest list price on file and fails the call. A truncated response
   (finish_reason length, or an empty answer) is logged as invalid and carries no parsed object. The response carries
   its cost (ModelResponse.cost), which names the model that answered.

A usage within the call's token bounds settles at or under its hold, so spend never passes the cap. A usage past them
(more input tokens than request bytes, or more output than the budget) is metered in full, because the meter never
undercounts; that alone can carry spend past the cap, and then every later call is refused.

Structured output, with its one retry and its unparsed result, is receipts.core.structured.complete_structured.

The key is read only from FALDEN_META_API_KEY or the constructor. It goes only into the Authorization header of
requests to BASE_URL and is kept out of every log line, exception, repr and response; any server text the client keeps
or logs (error messages, the answering model, the request id, the answer itself) is scrubbed of it first. The client
never acts on model output, and its egress is pinned to one host: redirects are not followed, and httpx's trust_env is
off, so no proxy or certificate setting in the environment can reroute or re-trust the connection. Untrusted prompt
text therefore cannot steer the key or the egress path (spec hard rule 7).

The spend file is RECEIPTS_SPEND_FILE, else .receipts/spend.jsonl under the current directory. It is JSONL written
under an exclusive file lock: an arm record per operator go, then a hold record (the call's worst case, written before
the request) and a settle record (what it cost) per attempt. Every client built without a meter shares the one meter
for that file, and every process that uses the file counts every record in it, open holds included, so the cap spans
every client and process, calls in flight too.

A checkout can ship its own .receipts/spend.jsonl, so the meter honors a spend file only when 'receipts spend arm'
could have made it: a regular file of mode 0600 in a directory of mode 0700, both owned by the current user, and
neither of them a symbolic link (the path is made absolute, never resolved, so a link a checkout ships to an armed
file elsewhere is seen and refused). Anything else refuses every call, saying why. An arm record also names the inode
of the file it was written to, and arms only that file: an arm record copied or planted into another file (a clone
made under umask 077 has the right modes) never arms it. Only a spend file the operator names with RECEIPTS_SPEND_FILE
can be armed past the credit: an arm record in the default .receipts/spend.jsonl under the current directory counts at
most at the credit, so a checkout that ships one, and guesses its inode, gains nothing past the credit.

Meter policy: a 200 response is metered from its usage. A 200 without valid usage, a 504, a read timeout, a connection
dropped after the request went out and a response body that cannot be decoded are metered at the worst case (the
hold: the token bounds at the highest list price), because Meta may bill them and the meter must never undercount.
Other errors are rejections and are not metered.

Sources, all read 2026-10-27:
- https://dev.meta.ai/docs/models.md: the model ids, and Standard versus Contributor mode chosen by model id.
- https://dev.meta.ai/docs/pricing-rate-limits: LIST_PRICES (per 1M tokens); limits per team per model, 3,000 RPM and
  4,000,000 TPM for each Muse text model; the x-ratelimit-* headers; the retry advice (start near 500 ms, double,
  add jitter, wait at least Retry-After).
- https://dev.meta.ai/docs/protocols/chat-completions: POST /v1/chat/completions, max_completion_tokens,
  reasoning_effort, prompt_cache_key; the system role is accepted and treated like developer.
- https://dev.meta.ai/docs/api-reference/chat-completions/schemas: the response shape, including the response's model,
  the "ID of the model used to generate the response"; the finish_reason values stop, length, tool_calls,
  content_filter and function_call; and json_schema.name (letters, digits, underscores and dashes, up to 64
  characters).
- https://dev.meta.ai/docs/structured-output: response_format json_schema; decoding is constrained to the schema.
- https://dev.meta.ai/docs/reasoning: reasoning tokens count toward max_completion_tokens and bill as output.
- https://dev.meta.ai/docs/prompt-caching and https://dev.meta.ai/docs/token-counting:
  usage.prompt_tokens_details.cached_tokens is part of prompt_tokens and bills at the cached rate;
  usage.completion_tokens_details.reasoning_tokens is part of completion_tokens.
- https://dev.meta.ai/docs/error-handling: the error envelope; 429 and 503 carry Retry-After; 400, 401, 402, 403, 404
  and 413 are not retried; a 504 gateway_timeout "will usually time out again", so it is not retried by default.
- https://dev.meta.ai/docs/authentication and https://dev.meta.ai/docs/api-reference: Bearer auth on api.meta.ai/v1.
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import random
import re
import stat
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import MappingProxyType, TracebackType
from typing import IO, Annotated, ClassVar, Final, Literal, NoReturn, Self

import httpx
from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

from receipts import __version__
from receipts.contracts import (
    ALLOWED_MODELS,
    ModelRefusedError,
    ModelRequest,
    ModelResponse,
    StageCost,
    UncheckedReason,
    Usage,
    check_model,
)
from receipts.core.chat_wire import (
    NAME_CHARS,
    WireCompletion,
    backoff_seconds,
    clip,
    count_header,
    error_detail,
    json_object,
    json_or_none,
    parse_retry_after,
    response_format,
    usage_or_none,
)
from receipts.core.errors import CoreCallError
from receipts.core.glimmer_client import GLIMMER_MODEL
from receipts.core.structured import describe_validation_error

__all__ = [
    "API_KEY_ENV",
    "ARM_COMMAND",
    "BASE_URL",
    "DEFAULT_SPEND_CAP_MICRO_USD",
    "DEFAULT_TPM",
    "HIGHEST_PRICE",
    "LIST_PRICES",
    "MAX_RETRIES",
    "SPEND_CEILING_MICRO_USD",
    "SPEND_DIR_MODE",
    "SPEND_FILE_ENV",
    "SPEND_FILE_MODE",
    "ArmRecord",
    "CallRecord",
    "HoldRecord",
    "MetaAPIError",
    "MetaClient",
    "MetaClientError",
    "MetaResponseError",
    "MetaRetriesExhaustedError",
    "MissingKeyError",
    "Price",
    "SettleRecord",
    "SpendCapError",
    "SpendFileError",
    "SpendMeter",
    "SpendNotArmedError",
    "TokenGovernor",
    "UnpricedModelError",
    "backoff_seconds",
    "default_spend_path",
    "encode_request",
    "named_spend_file",
    "parse_retry_after",
    "price_for",
    "shared_governor",
    "shared_meter",
    "stage_cost",
    "usd",
]

log = logging.getLogger(__name__)

API_KEY_ENV: Final = "FALDEN_META_API_KEY"
BASE_URL: Final = "https://api.meta.ai/v1"
DOCS_READ_ON: Final = "2026-10-27"
PRICES_SOURCE: Final = "https://dev.meta.ai/docs/pricing-rate-limits"
DEFAULT_SPEND_CAP_MICRO_USD: Final = 150_000_000  # the $150 Meta credit (the founder, 10/27)
SPEND_CEILING_MICRO_USD: Final = 425_000_000  # an above-credit arm's most (the founder, 10/31: "Cap $425", REG-612)
SPEND_FILE_ENV: Final = "RECEIPTS_SPEND_FILE"
DEFAULT_SPEND_FILE: Final = Path(".receipts") / "spend.jsonl"  # under the current directory
SPEND_DIR_MODE: Final = 0o700
SPEND_FILE_MODE: Final = 0o600
ARM_COMMAND: Final = "receipts spend arm --credit-confirmed"
MAX_RETRIES: Final = 5
PER_MILLION: Final = 1_000_000
TPM_WINDOW_S: Final = 60.0

_KEY_CHARS = re.compile(r"[\x21-\x7e]+")  # visible ASCII: a key a header can carry
_KEY_SHAPE = re.compile(r"LLM_\d+_[A-Za-z0-9_.~+/=-]+")  # the documented key format, LLM_{id}_{secret}
_KEY_PARTS = re.compile(r"LLM_\d+_(.+)")
_MIN_SECRET_CHARS = 8
_NOFOLLOW: Final = getattr(os, "O_NOFOLLOW", 0)
_CLOEXEC: Final = getattr(os, "O_CLOEXEC", 0)


# ---------------------------------------------------------------- errors


class MetaClientError(CoreCallError):
    """A Meta model call that produced no response. `reason` is what the caller records as unchecked. `cost` is what
    the call metered before it failed, with no token counts, set on every error raised once the call began. It
    survives a pickle round trip, as every core error does."""

    cost: StageCost | None = None


class MissingKeyError(MetaClientError):
    """No usable key in FALDEN_META_API_KEY or the constructor."""


class SpendCapError(MetaClientError):
    """The kill switch: metered spend has reached the cap, or this call could carry it past."""

    default_reason: ClassVar[UncheckedReason] = UncheckedReason.BUDGET


class SpendNotArmedError(SpendCapError):
    """No operator has armed the spend file, so every call is refused."""


class SpendFileError(SpendCapError):
    """The spend file cannot be trusted, read or written, so spend is unknown and every call is refused."""


class MetaAPIError(MetaClientError):
    """The API refused the call, or a transport failure that is not retried."""

    def __init__(self, message: str, *, status: int | None = None, reason: UncheckedReason | None = None) -> None:
        super().__init__(message, reason=reason)
        self.status = status


class MetaRetriesExhaustedError(MetaAPIError):
    """Retryable failures that outlasted the retries, or a Retry-After longer than the client may wait."""

    def __init__(
        self, message: str, *, attempts: int, status: int | None = None, reason: UncheckedReason | None = None
    ) -> None:
        super().__init__(message, status=status, reason=reason)
        self.attempts = attempts


class MetaResponseError(MetaClientError):
    """A 200 response the client cannot use: malformed, without usage, or from a model with no list price or off the
    allowlist (a -contributor id included)."""


class UnpricedModelError(ModelRefusedError):
    """An allowed model with no list price on file. It is refused rather than metered at a guess."""


# ---------------------------------------------------------------- prices


@dataclass(frozen=True)
class Price:
    """A list price in micro-dollars per million tokens: $1.25 per 1M tokens is 1_250_000."""

    input: int
    cached_input: int
    output: int

    def micro_usd(self, usage: Usage) -> int:
        """Usage times list price, rounded up to a whole micro-dollar so the meter never undercounts.

        cached_tokens is the part of input_tokens served from the prompt cache, billed at the cached rate; the rest
        bills at the input rate. output_tokens includes reasoning tokens and bills at the output rate.
        """
        uncached = usage.input_tokens - usage.cached_tokens
        if uncached < 0:
            raise ValueError("cached_tokens cannot exceed input_tokens")
        total = uncached * self.input + usage.cached_tokens * self.cached_input + usage.output_tokens * self.output
        return -(-total // PER_MILLION)

    def worst_case_micro_usd(self, request_bytes: int, max_tokens: int) -> int:
        """The most one call can cost: every request byte billed as an uncached input token, on the reading that a
        token covers at least one byte, plus the whole output budget, which includes reasoning."""
        return -(-(request_bytes * self.input + max_tokens * self.output) // PER_MILLION)


_SPARK = Price(input=1_250_000, cached_input=150_000, output=4_250_000)

# Standard mode list prices, read on DOCS_READ_ON from PRICES_SOURCE. sam-3.1 is billed per image or video frame and
# served on the Responses API, so it has no token price here and this client refuses it.
LIST_PRICES: Final[Mapping[str, Price]] = MappingProxyType(
    {
        "muse-spark-1.1": _SPARK,
        "muse-spark-1.2": _SPARK,
        "muse-spark-1.3": _SPARK,
        "muse-wonder-1.0": Price(input=4_000_000, cached_input=200_000, output=20_000_000),
    }
)

# What an answer from a model with no list price is metered at: the highest rate on file for each kind of token.
HIGHEST_PRICE: Final = Price(
    input=max(p.input for p in LIST_PRICES.values()),
    cached_input=max(p.cached_input for p in LIST_PRICES.values()),
    output=max(p.output for p in LIST_PRICES.values()),
)

# Tokens per minute for each model, per team, from the same page.
DEFAULT_TPM: Final[Mapping[str, int]] = MappingProxyType(dict.fromkeys(LIST_PRICES, 4_000_000))

for _model in LIST_PRICES:
    check_model(_model)  # a priced model is always an allowed one


def price_for(model: str) -> Price:
    """The list price of an allowed model. Refuses a -contributor id, an id off the allowlist and an unpriced model."""
    check_model(model)
    price = LIST_PRICES.get(model)
    if price is None:
        raise UnpricedModelError(
            f"{model}: no list price on file (prices read {DOCS_READ_ON}); refusing rather than guessing"
        )
    return price


def answering_price(model: object) -> tuple[Price | None, str]:
    """The list price of the model a response names, or None and why: it names no model, a -contributor one, one off
    the allowlist or one with no list price."""
    if not isinstance(model, str) or not model.strip():
        return None, "names no model"
    if "contributor" in model.lower():
        return None, "names a -contributor model, never used"
    price = LIST_PRICES.get(model)
    if price is not None:
        return price, ""
    if model in ALLOWED_MODELS:
        return None, "names a model with no list price on file"
    return None, "names a model off the allowlist"


def stage_cost(stage: str, model: str, usage: Usage, seconds: float = 0.0) -> StageCost:
    """One call's cost for the ledger, at list price."""
    return StageCost(
        stage=stage,
        model=model,
        input_tokens=usage.input_tokens,
        cached_tokens=usage.cached_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
        micro_usd=price_for(model).micro_usd(usage),
        seconds=seconds,
    )


def usd(micro_usd: int) -> str:
    """Micro-dollars as a dollar amount, with no float rounding: 1_234_567 is $1.234567."""
    sign = "-" if micro_usd < 0 else ""
    whole, part = divmod(abs(micro_usd), PER_MILLION)
    return f"{sign}${whole:,}.{part:06d}"


def _utcnow() -> datetime:
    return datetime.now(UTC)


def _current_uid() -> int:
    return os.getuid()


# ---------------------------------------------------------------- the spend file


class _Record(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    v: Literal[2]  # required on every line: an unversioned or older line is refused
    at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")


class ArmRecord(_Record):
    """An operator's go: the Meta credit is applied, so calls may spend up to cap_micro_usd. The latest one written to
    this file counts. `inode` is the file's own: an arm record found in any other file (copied, or planted in a
    checkout) is ignored. `above_credit` marks an arm past the credit ('receipts spend arm --above-credit'); it is
    written only when true, so a record without it, and every record written before it existed, is not above the
    credit."""

    kind: Literal["arm"]
    credit_confirmed: Literal[True]
    cap_micro_usd: int = Field(gt=0)
    inode: int = Field(ge=0)
    above_credit: bool = False


class HoldRecord(_Record):
    """A call about to run, holding its worst case against the cap until its settle record. A hold that is never
    settled (its process died) stays counted at the worst case, because the call may have been billed."""

    kind: Literal["hold"]
    id: str = Field(pattern=r"^[0-9a-f]{32}$")
    stage: str = Field(min_length=1)
    model: str = Field(min_length=1)
    micro_usd: int = Field(ge=0)


class SettleRecord(_Record):
    """What a held call cost (0 for a rejection that is never billed). It closes its hold."""

    kind: Literal["settle"]
    hold: str = Field(pattern=r"^[0-9a-f]{32}$")
    stage: str = Field(min_length=1)
    model: str = Field(min_length=1)
    micro_usd: int = Field(ge=0)
    estimated: bool


SpendFileRecord = ArmRecord | HoldRecord | SettleRecord
_SPEND_LINE: TypeAdapter[SpendFileRecord] = TypeAdapter(Annotated[SpendFileRecord, Field(discriminator="kind")])


def default_spend_path() -> Path:
    """RECEIPTS_SPEND_FILE when it is set, else .receipts/spend.jsonl under the current directory, as an absolute
    path. It is never resolved: a symbolic link on the way (a checkout's .receipts/spend.jsonl linking to an armed file
    elsewhere) stays in the path, where the meter sees it and refuses it."""
    value = os.environ.get(SPEND_FILE_ENV, "").strip()
    return Path(value).absolute() if value else DEFAULT_SPEND_FILE.absolute()


def named_spend_file(path: Path) -> bool:
    """Whether a spend file is one an operator named, anything but .receipts/spend.jsonl under the current directory,
    which a checkout can ship. Only a named file can be armed past the credit. Compare device and inode when both
    files can be checked, otherwise resolved paths, then absolute paths if resolution fails. This comparison never
    changes the path the meter opens and validates."""
    absolute_path = path.absolute()
    absolute_default = DEFAULT_SPEND_FILE.absolute()
    if absolute_path == absolute_default:
        return False
    try:
        return not path.samefile(DEFAULT_SPEND_FILE)
    except (OSError, RuntimeError):
        try:
            return path.resolve() != DEFAULT_SPEND_FILE.resolve()
        except (OSError, RuntimeError):
            return absolute_path != absolute_default


def _not_private(st: os.stat_result, mode: int, *, directory: bool) -> str | None:
    """Why a spend file (or its directory) is not one 'receipts spend arm' could have made, or None."""
    if stat.S_ISLNK(st.st_mode):
        return "is a symbolic link, which is never followed"
    if directory and not stat.S_ISDIR(st.st_mode):
        return "is not a directory"
    if not directory and not stat.S_ISREG(st.st_mode):
        return "is not a regular file"
    if st.st_uid != _current_uid():
        return f"belongs to another user (uid {st.st_uid})"
    actual = stat.S_IMODE(st.st_mode)
    if actual != mode:
        return f"has mode {actual:04o}, not {mode:04o}"
    return None


# ---------------------------------------------------------------- the spend meter and its kill switch


@dataclass(frozen=True)
class Hold:
    """A reservation against the cap, written to the spend file. Settle it exactly once."""

    id: str
    stage: str
    model: str
    micro_usd: int


class SpendMeter:
    """Metered spend against the kill-switch cap, in integer micro-dollars, kept in a spend file, safe across threads
    and processes.

    Every call is refused until the file holds an arm record written to it. The cap in effect is the lower of this
    meter's cap and the latest arm record's, so neither a meter configured high nor an arm record written high can
    raise it alone. One exception: a meter built at the default cap (no cap given, as shared_meter() builds it) on a
    spend file the operator named takes an arm record marked above the credit at its own cap. On the default
    .receipts/spend.jsonl under the current directory an arm counts at most at the credit, and no cap in effect is ever
    past SPEND_CEILING_MICRO_USD.
    A call first writes a hold for its worst case, under an exclusive file lock and only if every
    settled call plus every open hold in the file, from any process, leaves room for it; it then settles the hold with
    what it cost. So every client and process that shares the file shares one cap, calls in flight included, and a
    hold whose process died stays counted. A spend file that is not private to this user (module docstring), was
    replaced, cannot be read or written, or is not a whole sequence of valid records refuses every later call.
    """

    def __init__(
        self,
        path: Path,
        *,
        cap_micro_usd: int | None = None,
        now: Callable[[], datetime] = _utcnow,
    ) -> None:
        if cap_micro_usd is not None and cap_micro_usd <= 0:
            raise ValueError("the spend cap is a positive number of micro-dollars")
        self.path = path
        self.cap_micro_usd = DEFAULT_SPEND_CAP_MICRO_USD if cap_micro_usd is None else cap_micro_usd
        self._cap_given = cap_micro_usd is not None  # a meter given a cap keeps it, even under an above-credit arm
        self._named = named_spend_file(path)  # only a file the operator named is ever armed past the credit
        self._now = now
        self._lock = threading.Lock()
        self._spent = 0
        self._open: dict[str, int] = {}
        self._offset = 0
        self._inode: int | None = None
        self._armed: ArmRecord | None = None
        self._foreign_arms = 0
        self._broken: str | None = None
        with self._lock:
            self._refresh()

    def __repr__(self) -> str:
        # Use cached state only. Budget wrappers may skip SpendMeter.__init__ and have only their own cap.
        details = []
        path = getattr(self, "path", None)
        if path is not None:
            details.append(f"path={path!r}")
        cap = getattr(self, "cap_micro_usd", None)
        if cap is not None:
            details.append(f"cap_micro_usd={cap}")
        try:
            limit = self._limit()
        except AttributeError:
            pass
        else:
            details.append(f"cap_in_effect={limit}")
        return f"{type(self).__name__}({', '.join(details)})"

    @property
    def named(self) -> bool:
        """Whether the operator named this spend file (named_spend_file), so an arm above the credit counts on it."""
        return self._named

    @property
    def spent_micro_usd(self) -> int:
        """What settled calls cost: every settle record in the file, plus any this meter could not write."""
        with self._lock:
            self._refresh()
            return self._spent

    @property
    def held_micro_usd(self) -> int:
        """The worst cases of the calls still held: in flight in any process, or left open by one that died."""
        with self._lock:
            self._refresh()
            return sum(self._open.values())

    @property
    def armed(self) -> ArmRecord | None:
        """The latest arm record written to this file, or None while no operator has armed it."""
        with self._lock:
            self._refresh()
            return self._armed

    @property
    def foreign_arms(self) -> int:
        """Arm records in the file that name another file's inode, and so arm nothing."""
        with self._lock:
            self._refresh()
            return self._foreign_arms

    @property
    def limit_micro_usd(self) -> int:
        """The cap in effect: the lower of this meter's cap and the armed cap, or 0 while unarmed. An arm marked above
        the credit counts at its own cap on a meter built at the default; nothing passes SPEND_CEILING_MICRO_USD."""
        with self._lock:
            self._refresh()
            return self._limit()

    def arm(
        self, *, credit_confirmed: bool, cap_micro_usd: int = DEFAULT_SPEND_CAP_MICRO_USD, above_credit: bool = False
    ) -> ArmRecord:
        """Write an operator's arm record, once the Meta credit is confirmed as applied, creating the directory (mode
        0700) and the file (mode 0600) if they are missing. An existing directory or file is never changed: one that
        is not private to this user is refused. The armed cap is at most the default, the credit. With above_credit,
        on a spend file the operator named (never the current directory's default), it is above the credit and at most
        SPEND_CEILING_MICRO_USD, the founder's call of 10/31, and the spend past the credit is cash; raising the
        ceiling is a code change, the founder's call."""
        if credit_confirmed is not True:
            raise ValueError("arming needs credit_confirmed=True: confirm the Meta credit is applied to the account")
        if above_credit and not self._named:
            raise ValueError(
                f"an arm above the credit needs a spend file named with {SPEND_FILE_ENV}: the default "
                f".receipts/spend.jsonl under the current directory counts at most at the credit"
            )
        if above_credit:
            if not DEFAULT_SPEND_CAP_MICRO_USD < cap_micro_usd <= SPEND_CEILING_MICRO_USD:
                raise ValueError(
                    f"an armed cap above the credit is {DEFAULT_SPEND_CAP_MICRO_USD + 1:,} to "
                    f"{SPEND_CEILING_MICRO_USD:,} micro-dollars (past the {usd(DEFAULT_SPEND_CAP_MICRO_USD)} Meta "
                    f"credit, at most the {usd(SPEND_CEILING_MICRO_USD)} ceiling)"
                )
        elif not 0 < cap_micro_usd <= DEFAULT_SPEND_CAP_MICRO_USD:
            raise ValueError(
                f"an armed cap is 1 to {DEFAULT_SPEND_CAP_MICRO_USD:,} micro-dollars "
                f"({usd(DEFAULT_SPEND_CAP_MICRO_USD)}, the Meta credit)"
            )
        stamp = self._stamp()

        def record(inode: int) -> ArmRecord:
            return ArmRecord(
                v=2,
                kind="arm",
                at=stamp,
                credit_confirmed=True,
                cap_micro_usd=cap_micro_usd,
                inode=inode,
                above_credit=above_credit,
            )

        with self._lock:
            self._refresh()  # a broken spend file is never armed
            written = self._append(record, create=True)
        assert isinstance(written, ArmRecord)
        return written

    def reserve(self, worst_micro_usd: int, *, stage: str, model: str) -> Hold:
        """Hold a call's worst case against the cap in the spend file, or refuse the call."""
        if worst_micro_usd < 0:
            raise ValueError("a worst case is never negative")
        hold_id, stamp = uuid.uuid4().hex, self._stamp()

        def record(inode: int) -> HoldRecord:
            return HoldRecord(
                v=2, kind="hold", at=stamp, id=hold_id, stage=stage, model=model, micro_usd=worst_micro_usd
            )

        with self._lock:
            self._refresh()
            self._check_room(worst_micro_usd)  # refuse without touching the file
            failed: str | None = None
            try:
                self._append(record, before_write=lambda: self._check_room(worst_micro_usd))
            except OSError as exc:
                failed = type(exc).__name__
            if failed is not None:  # raised outside the except block, so nothing chains to it
                self._broken = f"the spend file could not be written ({failed}); every call is refused"
                raise SpendFileError(self._broken)
        return Hold(id=hold_id, stage=stage, model=model, micro_usd=worst_micro_usd)

    def settle(self, hold: Hold, micro_usd: int, *, estimated: bool) -> None:
        """Close a hold with what the call cost. It never raises: a spend file that cannot be written still counts the
        spend here and refuses every later call."""
        cost = max(micro_usd, 0)
        stamp = self._stamp()

        def record(inode: int) -> SettleRecord:
            return SettleRecord(
                v=2,
                kind="settle",
                at=stamp,
                hold=hold.id,
                stage=hold.stage,
                model=hold.model,
                micro_usd=cost,
                estimated=estimated,
            )

        with self._lock:
            try:
                if self._broken is not None:
                    raise SpendFileError(self._broken)
                self._append(record)
            except (OSError, ValidationError, SpendFileError) as exc:
                self._open.pop(hold.id, None)
                self._spent += cost
                if self._broken is None:
                    self._broken = f"the spend file could not be written ({type(exc).__name__}); every call is refused"
                log.error("spend meter: %s", self._broken)

    def _check_room(self, worst_micro_usd: int) -> None:
        if self._armed is None:
            foreign = (
                f" ({self._foreign_arms} arm record(s) in it were written to another file, so they arm nothing)"
                if self._foreign_arms
                else ""
            )
            raise SpendNotArmedError(
                f"the spend meter is not armed{foreign}, so every model call is refused; once the Meta credit is "
                f"confirmed as applied, an operator runs '{ARM_COMMAND}'"
            )
        limit = self._limit()
        if self._spent >= limit:
            raise SpendCapError(
                f"metered spend {usd(self._spent)} has reached the {usd(limit)} cap; every call is refused"
            )
        held = sum(self._open.values())
        if self._spent + held + worst_micro_usd > limit:
            raise SpendCapError(
                f"this call could cost up to {usd(worst_micro_usd)}; with {usd(self._spent)} spent and {usd(held)} "
                f"held in flight it could pass the {usd(limit)} cap"
            )

    def _limit(self) -> int:
        if self._armed is None:
            return 0
        armed = self._armed.cap_micro_usd
        if not self._named:  # a checkout can ship this file: whatever its arm says, it counts at most at the credit
            armed = min(armed, DEFAULT_SPEND_CAP_MICRO_USD)
        lifted = self._armed.above_credit and self._named and not self._cap_given
        own = SPEND_CEILING_MICRO_USD if lifted else self.cap_micro_usd
        return min(own, armed, SPEND_CEILING_MICRO_USD)

    def _stamp(self) -> str:
        return self._now().astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    def _refresh(self) -> None:
        if self._broken is not None:
            raise SpendFileError(self._broken)
        try:
            fd = os.open(self.path, os.O_RDONLY | _NOFOLLOW | _CLOEXEC)
        except FileNotFoundError:
            if self._offset or self._inode is not None:
                self._fail("the spend file disappeared")
            return  # no file yet: nothing is armed, so every call is refused
        except OSError as exc:
            self._fail(_open_problem(exc, "read"))
        with os.fdopen(fd, "rb") as handle:
            self._check_private(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_SH)
            try:
                self._read_new(handle)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)

    def _check_private(self, fd: int) -> None:
        """Refuse a spend file this user's 'receipts spend arm' could not have made, or one replaced since the last
        read."""
        file_stat = os.fstat(fd)
        problem = _not_private(file_stat, SPEND_FILE_MODE, directory=False) or self._directory_problem()
        if problem is not None:
            self._refuse(problem)
        if self._inode is None:
            self._inode = file_stat.st_ino
        elif file_stat.st_ino != self._inode:
            self._fail("the spend file was replaced by another file")

    def _directory_problem(self) -> str | None:
        try:
            problem = _not_private(os.lstat(self.path.parent), SPEND_DIR_MODE, directory=True)
        except OSError as exc:
            problem = f"cannot be checked ({type(exc).__name__})"
        return None if problem is None else f"is in a directory that {problem}"

    def _refuse(self, problem: str) -> NoReturn:
        self._fail(
            f"the spend file {self.path} {problem}. Only a file that '{ARM_COMMAND}' makes is honored, mode 0600 "
            "in a mode 0700 directory, both this user's, since a checkout can ship its own"
        )

    def _read_new(self, handle: IO[bytes]) -> None:
        """Apply the records added since the last read. Any line that is not a whole, valid record, a hold that reuses
        an open id, or a settle with no open hold refuses every later call."""
        handle.seek(0, os.SEEK_END)
        size = handle.tell()
        if size < self._offset:
            self._fail("the spend file shrank")
        handle.seek(self._offset)
        chunk = handle.read()
        if chunk and not chunk.endswith(b"\n"):
            self._fail("the spend file ends in a partial line")
        records: list[SpendFileRecord] = []
        for number, raw in enumerate(chunk.split(b"\n")[:-1], start=1):
            try:
                records.append(_SPEND_LINE.validate_json(raw))
            except ValidationError:
                self._fail(f"the spend file has an invalid line ({number} after byte {self._offset})")
        for record in records:
            self._apply(record)
        self._offset += len(chunk)

    def _apply(self, record: SpendFileRecord) -> None:
        if isinstance(record, ArmRecord):
            if record.inode == self._inode:
                self._armed = record
            else:
                self._foreign_arms += 1
                log.warning(
                    "spend meter: an arm record in %s names another file's inode, so it arms nothing", self.path
                )
        elif isinstance(record, HoldRecord):
            if record.id in self._open:
                self._fail("the spend file holds one call twice")
            self._open[record.id] = record.micro_usd
        else:
            if self._open.pop(record.hold, None) is None:
                self._fail("the spend file settles a call it never held, or one it already settled")
            self._spent += record.micro_usd

    def _append(
        self,
        make: Callable[[int], SpendFileRecord],
        *,
        create: bool = False,
        before_write: Callable[[], None] | None = None,
    ) -> SpendFileRecord:
        """Write one record under an exclusive lock, after checking the file is private, applying what other
        processes wrote since the last read and rechecking against that (before_write may refuse, and then nothing is
        written). `make` builds the record from the file's inode. Only arming creates the file."""
        fd = self._open_for_append(create)
        with os.fdopen(fd, "rb+") as handle:
            self._check_private(handle.fileno())
            fcntl.flock(handle, fcntl.LOCK_EX)
            try:
                self._read_new(handle)
                if before_write is not None:
                    before_write()
                assert self._inode is not None  # set by _check_private
                record = make(self._inode)
                line = record.model_dump_json(exclude_defaults=True).encode() + b"\n"  # above_credit only when true
                handle.write(line)
                handle.flush()
                os.fsync(handle.fileno())
                self._offset += len(line)
                self._apply(record)
            finally:
                fcntl.flock(handle, fcntl.LOCK_UN)
        return record

    def _open_for_append(self, create: bool) -> int:
        flags = os.O_RDWR | os.O_APPEND | _NOFOLLOW | _CLOEXEC
        if not create:
            return os.open(self.path, flags)
        _make_directories(self.path.parent)
        problem = self._directory_problem()
        if problem is not None:
            self._refuse(problem)  # before the file exists: nothing is made in a directory that is not private
        try:
            fd = os.open(self.path, flags | os.O_CREAT | os.O_EXCL, SPEND_FILE_MODE)
        except FileExistsError:
            return os.open(self.path, flags)
        os.fchmod(fd, SPEND_FILE_MODE)  # exactly 0600, whatever the umask
        return fd

    def _fail(self, why: str) -> NoReturn:
        self._broken = f"{why}; every call is refused"
        raise SpendFileError(self._broken)


def _make_directories(directory: Path) -> None:
    """Create the missing levels of a directory, each mode 0700 exactly, whatever the umask (one that strips the
    owner's write bit would leave a level nobody can create the next one in). An existing level is never changed."""
    missing: list[Path] = []
    level = directory
    while not level.exists():
        missing.append(level)
        if level.parent == level:
            break
        level = level.parent
    for level in reversed(missing):
        try:
            os.mkdir(level, SPEND_DIR_MODE)
        except FileExistsError:
            continue  # made by another process meanwhile: checked, never changed
        os.chmod(level, SPEND_DIR_MODE)


def _open_problem(exc: OSError, verb: str) -> str:
    if exc.errno == errno.ELOOP:
        return "the spend file is a symbolic link, which is never followed"
    return f"the spend file cannot be {verb} ({type(exc).__name__})"


# ---------------------------------------------------------------- the tokens-per-minute governor


@dataclass
class _Slot:
    tokens: int
    at: float
    settled: bool = False


class TokenGovernor:
    """A tokens-per-minute governor per model, safe across threads.

    A call holds its worst case (request bytes plus max_tokens) while in flight, then its reported usage for 60 seconds
    after it ends. A call that would carry the model past its budget for the last 60 seconds waits. The budgets
    default to the documented limits, which are per team: every key and process on the team counts against them, so a
    caller that shares the team can pass a smaller share. Every client built without a governor shares one per process
    (shared_governor).
    """

    def __init__(
        self,
        limits: Mapping[str, int] = DEFAULT_TPM,
        *,
        window_s: float = TPM_WINDOW_S,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        poll_s: float = 1.0,
    ) -> None:
        if any(limit <= 0 for limit in limits.values()):
            raise ValueError("a tokens-per-minute limit is positive")
        self._limits = dict(limits)
        self._window_s = window_s
        self._clock = clock
        self._sleep = sleep
        self._poll_s = poll_s
        self._lock = threading.Lock()
        self._slots: dict[str, list[_Slot]] = {model: [] for model in self._limits}

    @property
    def limits(self) -> Mapping[str, int]:
        return MappingProxyType(self._limits)

    def acquire(self, model: str, tokens: int) -> _Slot:
        """Wait until the call fits the model's budget, then hold its tokens. A call bigger than the whole budget
        waits for an otherwise empty window."""
        limit = self._limits.get(model)
        if limit is None:
            raise ModelRefusedError(f"{model}: no tokens-per-minute limit on file; refusing rather than guessing")
        want = min(max(tokens, 0), limit)
        while True:
            with self._lock:
                now = self._clock()
                live = [s for s in self._slots[model] if not s.settled or (s.tokens and s.at + self._window_s > now)]
                self._slots[model] = live
                used = sum(s.tokens for s in live)
                if used + want <= limit:
                    slot = _Slot(tokens=want, at=now)
                    live.append(slot)
                    return slot
                wait = self._wait(live, used + want - limit, now)
            log.info("governor model=%s used=%d want=%d limit=%d wait_s=%.2f", model, used, want, limit, wait)
            self._sleep(wait)

    def settle(self, slot: _Slot, tokens: int) -> None:
        """Replace a call's held estimate with the tokens it used; they count for the next 60 seconds."""
        with self._lock:
            slot.tokens = max(tokens, 0)
            slot.at = self._clock()
            slot.settled = True

    def _wait(self, live: list[_Slot], excess: int, now: float) -> float:
        """Seconds until enough settled tokens leave the window. Tokens still in flight never expire on their own."""
        freed = 0
        for slot in sorted((s for s in live if s.settled), key=lambda s: s.at):
            freed += slot.tokens
            if freed >= excess:
                return max(slot.at + self._window_s - now, 0.001)
        return self._poll_s


# ---------------------------------------------------------------- one meter per spend file, one governor per process

_SHARED_LOCK = threading.Lock()
_SHARED_METERS: dict[Path, SpendMeter] = {}
_SHARED_GOVERNORS: dict[str, TokenGovernor] = {}


def shared_meter(path: Path | None = None) -> SpendMeter:
    """The one meter this process keeps for a spend file (default_spend_path() when none is named), at the default
    cap. Every client built without a meter uses it, so they share the cap and the calls in flight."""
    key = (path if path is not None else default_spend_path()).absolute()  # never resolved: a link stays seen
    with _SHARED_LOCK:
        meter = _SHARED_METERS.get(key)
        if meter is None:
            meter = SpendMeter(key)
            _SHARED_METERS[key] = meter
        return meter


def shared_governor() -> TokenGovernor:
    """The one tokens-per-minute governor this process keeps at the documented limits, shared by every client built
    without a governor."""
    with _SHARED_LOCK:
        governor = _SHARED_GOVERNORS.get("default")
        if governor is None:
            governor = TokenGovernor()
            _SHARED_GOVERNORS["default"] = governor
        return governor


# ---------------------------------------------------------------- retries

# Failures after which the request may have run (and been billed) without a usable response reaching us. A body that
# cannot be decoded means the response arrived, so it is metered too, but it is not retried.
_MAYBE_BILLED = (httpx.ReadTimeout, httpx.ReadError, httpx.RemoteProtocolError, httpx.DecodingError)
_RETRIED = (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError)


@dataclass(frozen=True)
class _Failure:
    """An attempt with no usable response. It holds text only, never the request, so nothing built from it can reach
    the Authorization header."""

    detail: str
    retryable: bool
    timeout: bool = False
    maybe_billed: bool = False
    status: int | None = None
    retry_after: float | None = None


@dataclass(frozen=True)
class _Success:
    response: ModelResponse
    micro_usd: int
    answered: str  # the priced model that answered, which the call's cost names
    remaining_tokens: int | None
    remaining_requests: int | None


@dataclass
class _Tally:
    """What one call metered across its attempts."""

    micro_usd: int = 0


def encode_request(request: ModelRequest) -> bytes:
    """The Chat Completions body for a request, as sent."""
    body: dict[str, object] = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_completion_tokens": request.max_tokens,
        "prompt_cache_key": "receipts-" + NAME_CHARS.sub("_", request.stage)[:48],
    }
    if request.reasoning_effort is not None:
        body["reasoning_effort"] = request.reasoning_effort
    fmt = response_format(request)
    if fmt is not None:
        body["response_format"] = fmt
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


# ---------------------------------------------------------------- the client


class _Secret:
    """Holds the key. Its repr and str never show it, and it cannot be pickled."""

    __slots__ = ("_needles", "_value")

    def __init__(self, value: str) -> None:
        self._value = value
        parts = _KEY_PARTS.fullmatch(value)
        secret = parts.group(1) if parts else ""
        # The whole key, and its secret part on its own when the key has the documented shape.
        self._needles = (value, secret) if len(secret) >= _MIN_SECRET_CHARS else (value,)

    def reveal(self) -> str:
        return self._value

    def scrub(self, text: str) -> str:
        for needle in self._needles:
            text = text.replace(needle, "[redacted]")
        return _KEY_SHAPE.sub("[redacted]", text)

    def __repr__(self) -> str:
        return "_Secret('[redacted]')"

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError("the Meta API key cannot be pickled")


@dataclass(frozen=True)
class CallRecord:
    """One successful call, as logged. micro_usd covers every attempt the call metered; `model` is the model asked
    for and `answered` the one that answered."""

    stage: str
    model: str
    answered: str
    finish_reason: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    micro_usd: int
    seconds: float
    attempts: int
    request_id: str | None


@dataclass
class _Totals:
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    micro_usd: int = 0
    seconds: float = 0.0


class MetaClient:
    """contracts.ModelClient over https://api.meta.ai/v1/chat/completions, with the checks in the module docstring.

    Built without a meter, the client uses shared_meter(), the one meter for the spend file named by
    RECEIPTS_SPEND_FILE (else .receipts/spend.jsonl under the current directory); built without a governor, it uses
    shared_governor(). The clock and sleep it is given drive its retries.
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        meter: SpendMeter | None = None,
        governor: TokenGovernor | None = None,
        max_retries: int = MAX_RETRIES,
        retry_gateway_timeout: bool = False,
        max_wait_s: float = 120.0,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] = _utcnow,
        rng: random.Random | None = None,
    ) -> None:
        if not 0 <= max_retries <= MAX_RETRIES:
            raise ValueError(f"max_retries is 0 to {MAX_RETRIES}")
        key = (api_key if api_key is not None else os.environ.get(API_KEY_ENV, "")).strip()
        if not key:
            raise MissingKeyError(f"no Meta API key: set {API_KEY_ENV} or pass api_key")
        if not _KEY_CHARS.fullmatch(key):
            raise MissingKeyError(
                f"the Meta API key from {API_KEY_ENV} or api_key has characters a header cannot carry"
            )
        self._key = _Secret(key)
        self._meter = meter if meter is not None else shared_meter()
        self._governor = governor if governor is not None else shared_governor()
        self._max_retries = max_retries
        self._retry_gateway_timeout = retry_gateway_timeout
        self._max_wait_s = max_wait_s
        self._clock = clock
        self._sleep = sleep
        self._now = now
        self._rng = rng if rng is not None else random.Random()
        self._http = httpx.Client(
            base_url=BASE_URL,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
            follow_redirects=False,
            trust_env=False,  # no proxy or certificate setting from the environment
            headers={"Accept": "application/json", "User-Agent": f"falden-receipts/{__version__}"},
        )
        self._lock = threading.Lock()
        self._records: list[CallRecord] = []
        self._totals: dict[tuple[str, str], _Totals] = {}

    def __repr__(self) -> str:
        return f"MetaClient(base_url={BASE_URL!r}, meter={self._meter!r})"

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    @property
    def meter(self) -> SpendMeter:
        return self._meter

    @property
    def governor(self) -> TokenGovernor:
        return self._governor

    def records(self) -> tuple[CallRecord, ...]:
        with self._lock:
            return tuple(self._records)

    def stage_costs(self) -> tuple[StageCost, ...]:
        """What each stage spent through this client, per stage and model, including calls metered at the worst
        case and the seconds spent waiting and retrying."""
        with self._lock:
            return tuple(
                StageCost(
                    stage=stage,
                    model=model,
                    input_tokens=t.input_tokens,
                    cached_tokens=t.cached_tokens,
                    output_tokens=t.output_tokens,
                    reasoning_tokens=t.reasoning_tokens,
                    micro_usd=t.micro_usd,
                    seconds=t.seconds,
                )
                for (stage, model), t in sorted(self._totals.items())
            )

    def complete(self, request: ModelRequest) -> ModelResponse:
        """One call. The response's cost covers every attempt the call metered and the seconds it took, waits
        included; its token counts are the answered attempt's. An error raised once the call began carries the same
        cost, with no token counts."""
        model = check_model(request.model)  # again here: a request built with model_construct skips its validator
        if model == GLIMMER_MODEL:
            raise ModelRefusedError(f"{model} runs on Falden's own vLLM (the Glimmer client), never on Meta's API")
        price = price_for(model)
        try:
            body = encode_request(request)  # the body never holds the key, which travels only in the header
        except UnicodeEncodeError:
            raise MetaClientError(
                f"stage {request.stage!r}: the request text is not valid Unicode (a lone surrogate)"
            ) from None
        # The hold: the call's token bounds at the highest list price on file, since any priced model may answer.
        worst = HIGHEST_PRICE.worst_case_micro_usd(len(body), request.max_tokens)
        tokens = len(body) + request.max_tokens  # the governor holds request bytes plus the output budget
        started = self._clock()
        tally = _Tally()
        billed_as = model  # the model the call's seconds are filed under: the one that answered, once one has
        try:
            attempt = 0
            while True:
                attempt += 1
                outcome = self._attempt(request, model, price, body, worst, tokens, tally)
                if isinstance(outcome, _Success):
                    billed_as = outcome.answered
                    cost = self._call_cost(request.stage, outcome.answered, outcome.response.usage, tally, started)
                    response = outcome.response.model_copy(update={"cost": cost})
                    self._log_success(request, response, outcome, attempt)
                    return response
                where = f"stage {request.stage!r}, model {model}"
                reason = UncheckedReason.TIMEOUT if outcome.timeout else UncheckedReason.RUNTIME
                if not outcome.retryable:
                    raise MetaAPIError(f"{outcome.detail} ({where})", status=outcome.status, reason=reason)
                if attempt > self._max_retries:
                    raise MetaRetriesExhaustedError(
                        f"gave up after {attempt} attempts ({self._max_retries} retries); last: {outcome.detail} "
                        f"({where})",
                        attempts=attempt,
                        status=outcome.status,
                        reason=reason,
                    )
                wait = backoff_seconds(attempt, outcome.retry_after, self._rng)
                if wait > self._max_wait_s:
                    raise MetaRetriesExhaustedError(
                        f"asked to wait {wait:.0f} s, past the {self._max_wait_s:.0f} s limit, "
                        f"after attempt {attempt}; last: {outcome.detail} ({where})",
                        attempts=attempt,
                        status=outcome.status,
                        reason=reason,
                    )
                log.warning(
                    "meta retry stage=%s model=%s attempt=%d wait_s=%.2f cause=%s",
                    request.stage,
                    model,
                    attempt,
                    wait,
                    outcome.detail,
                )
                self._sleep(wait)
        except MetaClientError as exc:
            exc.cost = self._call_cost(request.stage, model, None, tally, started)
            raise
        finally:
            self._add(request.stage, billed_as, seconds=self._clock() - started)

    def _attempt(
        self, request: ModelRequest, model: str, price: Price, body: bytes, worst: int, tokens: int, tally: _Tally
    ) -> _Success | _Failure:
        reservation = self._meter.reserve(worst, stage=request.stage, model=model)
        spent, estimated, held = 0, False, 0
        billed_as = model  # the model this attempt's money is filed under
        slot: _Slot | None = None
        try:
            slot = self._governor.acquire(model, tokens)
            # From here the request may run, so an attempt that ends any way not handled below meters the worst case.
            spent, estimated, held = worst, True, tokens
            outcome = self._post(body)
            if isinstance(outcome, _Failure):
                failure = outcome
            elif outcome.status_code != 200:
                failure = self._http_failure(outcome)
            else:
                data = json_or_none(outcome.content)
                wire_usage = usage_or_none(data)
                if wire_usage is None or not isinstance(data, dict):
                    raise MetaResponseError(
                        f"stage {request.stage!r}: a 200 response without valid usage, metered at the worst case"
                    )
                usage = wire_usage.usage()
                held = usage.input_tokens + usage.output_tokens
                # The response's model is the "ID of the model used to generate the response", which need not be
                # the one asked for. Whichever of the two Meta bills, the meter takes the higher list price.
                answered = data.get("model")
                answer_price, problem = answering_price(answered)
                if answer_price is None or not isinstance(answered, str):
                    spent, estimated = HIGHEST_PRICE.micro_usd(usage), True
                    named = f" ({clip(self._redact(answered), 64)!r})" if isinstance(answered, str) and answered else ""
                    raise MetaResponseError(
                        f"stage {request.stage!r}: the response {problem}{named}; metered at the highest list "
                        f"price on file, {usd(spent)}"
                    )
                spent, estimated = max(price.micro_usd(usage), answer_price.micro_usd(usage)), False
                if spent > worst:
                    log.warning(
                        "meta spend stage=%s model=%s answered=%s used tokens past the call's bounds: metered in "
                        "full at micro_usd=%d, over its hold of %d",
                        request.stage,
                        model,
                        answered,
                        spent,
                        worst,
                    )
                response = self._response(request, data, usage)
                billed_as = answered
                return _Success(
                    response=response,
                    micro_usd=spent,
                    answered=answered,
                    remaining_tokens=count_header(outcome.headers.get("x-ratelimit-remaining-tokens")),
                    remaining_requests=count_header(outcome.headers.get("x-ratelimit-remaining-requests")),
                )
            if not failure.maybe_billed:
                spent, held = 0, 0
            return failure
        finally:
            self._meter.settle(reservation, spent, estimated=estimated)
            if slot is not None:
                self._governor.settle(slot, held)
            if spent:
                tally.micro_usd += spent
                self._add(request.stage, billed_as, micro_usd=spent)
                if estimated:
                    log.warning(
                        "meta spend stage=%s model=%s metered an estimate micro_usd=%d", request.stage, model, spent
                    )

    def _post(self, body: bytes) -> httpx.Response | _Failure:
        failure: _Failure
        try:
            return self._http.post(
                "/chat/completions",
                content=body,
                headers={"Authorization": f"Bearer {self._key.reveal()}", "Content-Type": "application/json"},
            )
        except _RETRIED as exc:
            failure = _Failure(
                detail=f"{type(exc).__name__}: {clip(self._redact(str(exc)))}",
                retryable=True,
                timeout=isinstance(exc, httpx.TimeoutException),
                maybe_billed=isinstance(exc, _MAYBE_BILLED),
            )
        except httpx.HTTPError as exc:
            failure = _Failure(
                detail=f"{type(exc).__name__}: {clip(self._redact(str(exc)))}",
                retryable=False,
                maybe_billed=isinstance(exc, _MAYBE_BILLED),
            )
        return failure  # returned outside the except blocks, so nothing raised later chains to the request

    def _http_failure(self, response: httpx.Response) -> _Failure:
        status = response.status_code
        gateway_timeout = status == 504
        retryable = status == 429 or (500 <= status <= 599 and (not gateway_timeout or self._retry_gateway_timeout))
        return _Failure(
            detail=clip(self._redact(error_detail(status, response.content))),
            retryable=retryable,
            timeout=gateway_timeout,
            maybe_billed=gateway_timeout,
            status=status,
            retry_after=parse_retry_after(response.headers.get("retry-after"), self._now()),
        )

    def _response(self, request: ModelRequest, data: object, usage: Usage) -> ModelResponse:
        """The response, its answering model already priced, with every piece of server text scrubbed of the key: the
        answering model, the request id and the answer itself."""
        problem: str | None = None
        try:
            completion = WireCompletion.model_validate(data)
        except ValidationError as exc:
            problem = describe_validation_error(exc)
        if problem is not None:
            raise MetaResponseError(
                f"stage {request.stage!r}: a malformed 200 response ({clip(self._redact(problem))})"
            )
        answered = clip(self._redact(completion.model), 64)
        if answered != request.model:
            log.warning("meta call stage=%s asked for %s, answered by %r", request.stage, request.model, answered)
        choice = completion.choices[0]
        text = self._redact(choice.message.content or "")
        parsed = None
        if request.json_schema is not None and choice.finish_reason == "stop":
            parsed = json_object(text)
        return ModelResponse(
            model=answered,
            text=text,
            finish_reason=choice.finish_reason,
            usage=usage,
            request_id=clip(self._redact(completion.id), 128) if completion.id else None,
            parsed=parsed,
        )

    def _call_cost(self, stage: str, model: str, usage: Usage | None, tally: _Tally, started: float) -> StageCost:
        return StageCost(
            stage=stage,
            model=model,
            input_tokens=usage.input_tokens if usage else 0,
            cached_tokens=usage.cached_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            reasoning_tokens=usage.reasoning_tokens if usage else 0,
            micro_usd=tally.micro_usd,
            seconds=max(self._clock() - started, 0.0),
        )

    def _log_success(self, request: ModelRequest, response: ModelResponse, success: _Success, attempts: int) -> None:
        usage = response.usage
        cost = response.cost
        assert cost is not None
        record = CallRecord(
            stage=request.stage,
            model=request.model,
            answered=success.answered,
            finish_reason=response.finish_reason,
            input_tokens=usage.input_tokens,
            cached_tokens=usage.cached_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            micro_usd=cost.micro_usd,
            seconds=cost.seconds,
            attempts=attempts,
            request_id=response.request_id,
        )
        with self._lock:
            self._records.append(record)
            totals = self._totals.setdefault((request.stage, success.answered), _Totals())
            totals.input_tokens += usage.input_tokens
            totals.cached_tokens += usage.cached_tokens
            totals.output_tokens += usage.output_tokens
            totals.reasoning_tokens += usage.reasoning_tokens
        log.info(
            "meta call stage=%s model=%s answered=%s finish_reason=%s input_tokens=%d cached_tokens=%d "
            "output_tokens=%d reasoning_tokens=%d micro_usd=%d attempts=%d seconds=%.3f request_id=%r "
            "remaining_tokens=%s remaining_requests=%s",
            record.stage,
            record.model,
            record.answered,
            record.finish_reason,
            record.input_tokens,
            record.cached_tokens,
            record.output_tokens,
            record.reasoning_tokens,
            record.micro_usd,
            record.attempts,
            record.seconds,
            clip(record.request_id or "", 80),
            success.remaining_tokens,
            success.remaining_requests,
        )
        if response.truncated:
            log.warning(
                "meta call stage=%s model=%s truncated (finish_reason length, or an empty answer): it invalidates "
                "the item",
                record.stage,
                record.model,
            )

    def _add(self, stage: str, model: str, *, micro_usd: int = 0, seconds: float = 0.0) -> None:
        with self._lock:
            totals = self._totals.setdefault((stage, model), _Totals())
            totals.micro_usd += micro_usd
            totals.seconds += seconds

    def _redact(self, text: str) -> str:
        return self._key.scrub(text)
