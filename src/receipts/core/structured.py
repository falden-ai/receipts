"""Structured output over any contracts.ModelClient (RT1).

Every response is validated with Pydantic. A response that fails validation gets one retry, and the retry carries the
validation error back to the model; a second failure is an Unparsed result (its `invalid` set), which the caller
records as UncheckedReason.UNPARSED. A caller can ask for no retry (retry=False): one call, and a failure is
Unparsed at once. A response that did not finish normally invalidates the item at once, with no retry: above
all a truncated one (finish_reason length, or an empty answer), whose answer was cut off or never came.

This runs over any ModelClient, so lanes can use it with their own fakes. Failures of the call itself (budget,
timeouts, API errors) are raised by the client as contracts.ModelCallError and are not turned into Unparsed here, but
when the retry raises, the error leaves carrying the cost of every attempt the item took (its `cost`), so a caller
that records the error's cost records the first call's too. MetaClient.complete alone checks no schema: its
ModelResponse.parsed is plain JSON, so callers that need a validated answer come through this function.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Final

from pydantic import BaseModel, ValidationError

from receipts.contracts import (
    Message,
    ModelCallError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    StageCost,
    Unchecked,
    UncheckedReason,
)

log = logging.getLogger(__name__)

RETRY_PROMPT: Final = (
    "Your previous answer did not match the required JSON schema. Validation errors: {errors}\n"
    "Answer again with only a JSON object that matches the schema."
)
MAX_ERROR_CHARS: Final = 2_000


@dataclass(frozen=True)
class Parsed[T: BaseModel]:
    """A response that passed validation, with every response the item took."""

    value: T
    responses: tuple[ModelResponse, ...]

    @property
    def costs(self) -> tuple[StageCost, ...]:
        return response_costs(self.responses)


@dataclass(frozen=True)
class Unparsed:
    """An item with no valid answer. The caller records it as unchecked with reason UNPARSED. `invalid` is set when
    every answer finished normally and failed schema validation (twice, with the retry); `truncated` when the last
    one was cut off or empty."""

    detail: str
    responses: tuple[ModelResponse, ...]
    truncated: bool = False
    invalid: bool = False

    @property
    def reason(self) -> UncheckedReason:
        return UncheckedReason.UNPARSED

    @property
    def costs(self) -> tuple[StageCost, ...]:
        return response_costs(self.responses)

    def unchecked(self, what: str) -> Unchecked:
        return Unchecked(what=what, reason=UncheckedReason.UNPARSED, detail=self.detail)


def response_costs(responses: Iterable[ModelResponse]) -> tuple[StageCost, ...]:
    """The costs the responses carry, in order; a response from a client that prices nothing carries none."""
    return tuple(r.cost for r in responses if r.cost is not None)


def combined_cost(stage: str, model: str | None, costs: Iterable[StageCost | None]) -> StageCost | None:
    """One cost for several calls of a stage: tokens, money and seconds added up. None when none had a cost."""
    present = [c for c in costs if c is not None]
    if not present:
        return None
    return StageCost(
        stage=stage,
        model=model,
        input_tokens=sum(c.input_tokens for c in present),
        cached_tokens=sum(c.cached_tokens for c in present),
        output_tokens=sum(c.output_tokens for c in present),
        reasoning_tokens=sum(c.reasoning_tokens for c in present),
        micro_usd=sum(c.micro_usd for c in present),
        seconds=sum(c.seconds for c in present),
    )


def describe_validation_error(exc: ValidationError) -> str:
    """The validation errors as one short line, without the rejected input or documentation links."""
    parts = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    text = "; ".join(parts)
    return text if len(text) <= MAX_ERROR_CHARS else text[: MAX_ERROR_CHARS - 3] + "..."


def complete_structured[T: BaseModel](
    client: ModelClient, request: ModelRequest, output: type[T], *, retry: bool = True
) -> Parsed[T] | Unparsed:
    """Ask for a JSON object matching `output` and validate the answer, with one retry on a validation failure unless
    `retry` is False.

    When the request carries no json_schema, the output model's own JSON Schema is sent. A ModelCallError from the
    retry is raised with its `cost` covering both attempts.
    """
    if request.json_schema is None:
        request = _updated(request, json_schema=output.model_json_schema())
    responses: list[ModelResponse] = []
    current = request
    errors = ""
    attempts = (1, 2) if retry else (1,)
    for attempt in attempts:
        try:
            response = client.complete(current)
        except ModelCallError as exc:
            if responses:  # the earlier attempt's cost leaves with the error, never dropped
                exc.cost = combined_cost(request.stage, request.model, [*response_costs(responses), exc.cost])
            raise
        responses.append(response)
        if response.truncated or response.finish_reason != "stop":
            if response.truncated:
                detail = "truncated (finish_reason length, or an empty answer): it invalidates the item"
            else:
                detail = f"finish_reason {response.finish_reason!r} invalidates the item"
            log.warning("structured stage=%s attempt=%d unparsed: %s", request.stage, attempt, detail)
            return Unparsed(detail=detail, responses=tuple(responses), truncated=response.truncated)
        try:
            value = output.model_validate_json(response.text)
        except ValidationError as exc:
            errors = describe_validation_error(exc)
        else:
            return Parsed(value=value, responses=tuple(responses))
        log.info("structured stage=%s attempt=%d failed validation: %s", request.stage, attempt, errors)
        if attempt < attempts[-1]:
            current = _retry_request(request, response.text, errors)
    times = "twice" if retry else "once, with no retry"
    detail = f"failed schema validation {times}; last errors: {errors}"
    log.warning("structured stage=%s unparsed: %s", request.stage, detail)
    return Unparsed(detail=detail, responses=tuple(responses), invalid=True)


def _retry_request(request: ModelRequest, rejected: str, errors: str) -> ModelRequest:
    messages = list(request.messages)
    if rejected:
        messages.append(Message(role="assistant", content=rejected))
    messages.append(Message(role="user", content=RETRY_PROMPT.format(errors=errors)))
    return _updated(request, messages=tuple(messages))


def _updated(request: ModelRequest, **changes: object) -> ModelRequest:
    """A copy of the request with some fields changed, validated again like any new request."""
    return ModelRequest.model_validate({**request.model_dump(), **changes})
