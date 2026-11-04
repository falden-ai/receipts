"""One structured model call for an attacker, over ``contracts.ModelClient`` only (RT1's rules, kept in this lane).

The answer is validated with Pydantic. A validation failure gets one retry that carries the error back; a second
failure is unparsed. A truncated answer (cut at the token limit, or empty) invalidates the item at once. A failed call
(``contracts.ModelCallError``) is never raised: its reason becomes the attacker's unchecked reason, and whatever it
still spent is kept, so every metered call's cost reaches the attack's outcome exactly once.

The answer is untrusted text: it is parsed as JSON into a fixed schema and nothing else, and the attackers use it only
to propose test code that the sandbox runs, or to take candidates away.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from pydantic import BaseModel, ValidationError

from receipts.contracts import (
    Message,
    ModelCallError,
    ModelClient,
    ModelRequest,
    ModelResponse,
    StageCost,
    UncheckedReason,
)

_FENCE = re.compile(r"^```(?:json)?\s*\n(?P<body>.*)\n```\s*$", re.DOTALL)
_RETRY = (
    "Your previous answer did not match the required JSON schema. Validation errors: {errors}\n"
    "Answer again with only a JSON object that matches the schema."
)
# An answer that is not JSON at all gets the error's place and the escaping rule too: in the 10/28 pilot, 14 Glimmer
# stage-1 answers broke the JSON mid-string (Expecting ',' delimiter at column 633 to 2170, code with a bare quote in
# it), the retry the same way.
_ESCAPES = ' Inside a JSON string, write a double quote as \\", a backslash as \\\\ and a line break as \\n.'
_NEAR = 60  # characters of the answer shown before the place the JSON broke


@dataclass
class Answer[T: BaseModel]:
    """A call's result: the validated value, or why there is none; and every cost the call spent."""

    value: T | None = None
    reason: UncheckedReason | None = None
    detail: str = ""
    costs: list[StageCost] = field(default_factory=list)


def _cost(response: ModelResponse, request: ModelRequest) -> StageCost:
    if response.cost is not None:
        return response.cost
    usage = response.usage
    return StageCost(
        stage=request.stage,
        model=response.model or request.model,
        input_tokens=usage.input_tokens,
        cached_tokens=usage.cached_tokens,
        output_tokens=usage.output_tokens,
        reasoning_tokens=usage.reasoning_tokens,
    )


def _json_of(response: ModelResponse) -> object:
    if response.parsed is not None:
        return response.parsed
    text = response.text.strip()
    fenced = _FENCE.match(text)
    return json.loads(fenced.group("body") if fenced else text)


def _not_json(exc: ValueError) -> str:
    """Why an answer is not JSON, with the answer's text just before the place it broke when the parser says where."""
    detail = f"not JSON: {exc}"[:300]
    if isinstance(exc, json.JSONDecodeError):
        detail += f" (the answer there: {exc.doc[max(0, exc.pos - _NEAR) : exc.pos + 1]!r})"
    return detail


def _errors(exc: ValidationError) -> str:
    parts = []
    for error in exc.errors(include_url=False, include_context=False, include_input=False):
        location = ".".join(str(part) for part in error["loc"]) or "(root)"
        parts.append(f"{location}: {error['msg']}")
    return "; ".join(parts)[:1500]


def structured_call[T: BaseModel](client: ModelClient, request: ModelRequest, output: type[T]) -> Answer[T]:
    """Ask for a JSON object matching ``output``; validate it, with one retry on a validation failure."""
    if request.json_schema is None:
        request = ModelRequest.model_validate({**request.model_dump(), "json_schema": output.model_json_schema()})
    answer: Answer[T] = Answer()
    current = request
    for attempt in (1, 2):
        try:
            response = client.complete(current)
        except ModelCallError as exc:
            if exc.cost is not None:
                answer.costs.append(exc.cost)
            answer.reason, answer.detail = exc.reason, f"model call failed: {exc}"
            return answer
        answer.costs.append(_cost(response, current))
        if response.truncated:
            answer.reason, answer.detail = UncheckedReason.UNPARSED, f"truncated answer ({response.finish_reason})"
            return answer
        try:
            answer.value = output.model_validate(_json_of(response))
            return answer
        except (ValueError, ValidationError) as exc:
            errors = _errors(exc) if isinstance(exc, ValidationError) else _not_json(exc)
            if attempt == 2:
                answer.reason, answer.detail = UncheckedReason.UNPARSED, f"no valid answer after a retry: {errors}"
                return answer
            retry = _RETRY.format(errors=errors) + ("" if isinstance(exc, ValidationError) else _ESCAPES)
            messages = (
                *current.messages,
                Message(role="assistant", content=response.text[:20000]),
                Message(role="user", content=retry),
            )
            current = ModelRequest.model_validate({**current.model_dump(), "messages": messages})
    return answer
