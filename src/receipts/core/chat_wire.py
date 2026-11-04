"""The OpenAI-compatible Chat Completions wire format both core clients speak: the Meta Model API (RT1) and Falden's own
vLLM for the Glimmer arm. Responses are validated with Pydantic at this boundary, and the retry timing both clients
share is here too.

Sources, read 2026-10-27: https://dev.meta.ai/docs/api-reference/chat-completions/schemas (the response shape and the
finish_reason values stop, length, tool_calls, content_filter and function_call; json_schema.name allows letters,
digits, underscores and dashes, up to 64 characters) and https://dev.meta.ai/docs/pricing-rate-limits (the retry
advice: start near 500 ms, double, add jitter, never wait less than Retry-After). vLLM's OpenAI-compatible server
returns the same shape.
"""

from __future__ import annotations

import email.utils
import json
import math
import random
import re
from datetime import UTC, datetime
from typing import Final, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from receipts.contracts import ModelRequest, Usage

BACKOFF_BASE_S: Final = 0.5
BACKOFF_CAP_S: Final = 30.0
MAX_SERVER_TEXT: Final = 300

NAME_CHARS: Final = re.compile(r"[^A-Za-z0-9_-]")
_COUNT = re.compile(r"\d{1,18}", re.ASCII)


class Wire(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class PromptDetails(Wire):
    cached_tokens: int | None = Field(default=None, ge=0, strict=True)


class CompletionDetails(Wire):
    reasoning_tokens: int | None = Field(default=None, ge=0, strict=True)


class WireUsage(Wire):
    prompt_tokens: int = Field(ge=0, strict=True)
    completion_tokens: int = Field(ge=0, strict=True)
    prompt_tokens_details: PromptDetails | None = None
    completion_tokens_details: CompletionDetails | None = None

    @property
    def cached_tokens(self) -> int:
        details = self.prompt_tokens_details
        return (details.cached_tokens or 0) if details else 0

    @property
    def reasoning_tokens(self) -> int:
        details = self.completion_tokens_details
        return (details.reasoning_tokens or 0) if details else 0

    @model_validator(mode="after")
    def _parts_fit(self) -> Self:
        if self.cached_tokens > self.prompt_tokens:
            raise ValueError("cached_tokens exceeds prompt_tokens")
        if self.reasoning_tokens > self.completion_tokens:
            raise ValueError("reasoning_tokens exceeds completion_tokens")
        return self

    def usage(self) -> Usage:
        return Usage(
            input_tokens=self.prompt_tokens,
            cached_tokens=self.cached_tokens,
            output_tokens=self.completion_tokens,
            reasoning_tokens=self.reasoning_tokens,
        )


class WireMessage(Wire):
    content: str | None = None
    refusal: str | None = None


class WireChoice(Wire):
    message: WireMessage
    # The documented values are lowercase words; anything else (a key echoed back, a forged line) is malformed.
    finish_reason: str = Field(pattern=r"^[a-z][a-z_]{0,31}$")


class WireCompletion(Wire):
    id: str | None = None
    model: str = Field(min_length=1)
    choices: list[WireChoice] = Field(min_length=1)


class WireErrorBody(Wire):
    message: str = ""
    type: str | None = None
    code: str | None = None


class WireError(Wire):
    error: WireErrorBody


def json_or_none(content: bytes) -> object:
    try:
        return json.loads(content)
    except (ValueError, RecursionError):
        return None


def usage_or_none(data: object) -> WireUsage | None:
    if not isinstance(data, dict) or "usage" not in data:
        return None
    try:
        return WireUsage.model_validate(data["usage"])
    except ValidationError:
        return None


def json_object(text: str) -> dict[str, object] | None:
    try:
        value = json.loads(text)
    except (ValueError, RecursionError):
        return None
    return value if isinstance(value, dict) else None


def count_header(value: str | None) -> int | None:
    """A rate-limit header as a count, or None: server text that is not a plain count never reaches a log line."""
    text = (value or "").strip()
    return int(text) if _COUNT.fullmatch(text) else None


def error_detail(status: int, content: bytes) -> str:
    """'HTTP <status>', plus the error envelope's type, code and message when the body is one."""
    detail = f"HTTP {status}"
    try:
        error = WireError.model_validate_json(content).error
    except ValidationError:
        return detail
    kind = "/".join(part for part in (error.type, error.code) if part)
    return detail + (f" {kind}" if kind else "") + (f": {error.message}" if error.message else "")


def schema_name(request: ModelRequest) -> str:
    """The response format's name: the request's schema_name, else the schema's title, else the stage, cut to the
    characters the API allows."""
    if request.schema_name:
        raw = request.schema_name
    else:
        title = request.json_schema.get("title") if request.json_schema else None
        raw = title if isinstance(title, str) and title else request.stage
    return NAME_CHARS.sub("_", raw)[:64] or "output"


def response_format(request: ModelRequest) -> dict[str, object] | None:
    if request.json_schema is None:
        return None
    return {"type": "json_schema", "json_schema": {"name": schema_name(request), "schema": request.json_schema}}


def clip(text: str, limit: int = MAX_SERVER_TEXT) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


def parse_retry_after(value: str | None, now: datetime) -> float | None:
    """Seconds to wait from a Retry-After header, in delta-seconds or HTTP-date form; None when absent or unreadable."""
    if value is None or not value.strip():
        return None
    text = value.strip()
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(text)
        except (TypeError, ValueError, IndexError):
            return None
        if when.tzinfo is None:
            when = when.replace(tzinfo=UTC)
        seconds = (when - now).total_seconds()
    if not math.isfinite(seconds):
        return None
    return max(0.0, seconds)


def backoff_seconds(attempt: int, retry_after: float | None, rng: random.Random) -> float:
    """Exponential backoff with jitter for the given failed attempt (1 for the first), starting near 500 ms and
    doubling, and never less than Retry-After."""
    step = min(BACKOFF_CAP_S, BACKOFF_BASE_S * (1 << max(attempt - 1, 0)))
    wait = step + rng.uniform(0, step / 2)
    if retry_after is not None:
        wait = max(wait, retry_after + rng.uniform(0, BACKOFF_BASE_S))
    return wait
