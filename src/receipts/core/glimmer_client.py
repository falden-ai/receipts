"""The Glimmer arm's client: contracts.ModelClient over Falden's own vLLM serving Muse Glimmer 30B (open weights)
through its OpenAI-compatible Chat Completions endpoint.

It is the Meta client's twin without the money. The base URL comes from RECEIPTS_GLIMMER_URL (or the constructor),
there is no spend meter, and each response still records its usage and seconds, with micro_usd 0. A server on
Falden's own GPU VM takes no key. The backup host on Modal (ops/orchestration/receipts/day4/modal-glimmer, the same
vLLM recipe) requires Modal proxy auth: for a base URL whose host ends in MODAL_HOST_SUFFIX, and only then, every
request carries the Modal-Key and Modal-Secret headers, read by exact name from MODAL_KEY_ENV and MODAL_SECRET_ENV in
the environment the caller provides (modal_token). Such a URL must be https, and a missing or malformed value is a
GlimmerConfigError before any request. Neither value is logged or shown: they are attached per request, not kept in
the HTTP client's default headers, and scrubbed from every failure's detail.

Only 'muse-glimmer-30b', the server's --served-model-name, is ever sent; any other id is refused before a request,
and an answer that names another model fails the call. Truncation and structured output are handled the Meta
client's way: a truncated response (finish_reason length, or an empty answer, a Glimmer serving trap) invalidates the
item and carries no parsed object, and receipts.core.structured.complete_structured validates and retries on top of
it.

The egress is pinned to the configured host: redirects are not followed and httpx's trust_env is off. A 429, a 5xx
and a transport timeout or network error are retried with the shared backoff, at most `max_retries` times, and a
timeout at most `max_timeout_retries` times of those: a request the server did not answer in the read timeout is
regenerated from the start, and on a busy server it times out again (the 10/28 pilot: three stage-1 calls timed out
four times each, 2,405 s apiece, and none answered).

vLLM's OpenAI-compatible server takes max_tokens and response_format json_schema; the Meta-only fields
(prompt_cache_key, reasoning_effort) are not sent. Every Glimmer request also carries vLLM's repetition_penalty
(GLIMMER_REPETITION_PENALTY unless the constructor says otherwise): on the GPU VM on 2026-10-29, with the schema
enforced, all 3 stage-1 answers looped inside the JSON "code" string until the 16,000-token limit.

The server runs without a reasoning parser (it stops vLLM applying json_schema), so on a call without json_schema
Glimmer's ATEM channels reach the client: its reasoning to=self, then the answer to=user (chat_template.jinja in the
weights). With special tokens dropped, as vLLM does by default, the channel headers run into the text ('...ls.assistant
to=user```bash'), and on the 10/30 re-pilot all 120 agent-loop answers were format errors. Such a request asks for
skip_special_tokens false, and the answer is the last message to the user (user_message); json_schema calls are left
alone, since the grammar holds their output to the JSON from its first token.

Under json_schema Glimmer never writes the newline escape inside a JSON string: on Modal on 2026-10-30 every test file
it wrote came back as one line (a space where each newline belonged), so it did not parse, with or without the
repetition penalty. Asked for the same file as an array of lines it wrote 23 of 23 that parse. So the schema Glimmer
gets has each string property that holds a whole source file (SOURCE_FIELDS: stage 1's "code", the agent loop write
stage's "content") replaced by "<name>_lines", an array of strings (lines_schema), and the answer has the lines joined
back (joined_code) before any caller sees it: callers, their prompts and their output models are unchanged.
"""

from __future__ import annotations

import json
import logging
import os
import random
import re
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from types import TracebackType
from typing import Final, NoReturn, Self

import httpx
from pydantic import ValidationError

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
    WireCompletion,
    backoff_seconds,
    clip,
    error_detail,
    json_object,
    json_or_none,
    parse_retry_after,
    response_format,
    usage_or_none,
)
from receipts.core.errors import CoreCallError

log = logging.getLogger(__name__)

GLIMMER_MODEL: Final = "muse-glimmer-30b"
GLIMMER_URL_ENV: Final = "RECEIPTS_GLIMMER_URL"
MODAL_HOST_SUFFIX: Final = ".modal.run"  # a Modal web endpoint's host
MODAL_KEY_ENV: Final = "MODAL_PROXY_TOKEN_ID"
MODAL_SECRET_ENV: Final = "MODAL_PROXY_TOKEN_SECRET"
_MODAL_HOST: Final = re.compile(r"[a-z0-9-]+(\.[a-z0-9-]+)*\.modal\.run")
_TOKEN_CHARS: Final = re.compile(r"[\x21-\x7e]+")  # visible ASCII: a value a header can carry
DEFAULT_RETRIES: Final = 3
DEFAULT_TIMEOUT_RETRIES: Final = 1
MAX_RETRIES: Final = 5
# Anti-repetition for Glimmer only. With json_schema enforced (no reasoning parser, receipts.eval.gpuvm) the 10/29 VM
# test's 3 stage-1 answers all degenerated into a loop inside the "code" string (one lint suppression comment repeated,
# 'models as models2, models as models3, ...') until finish_reason length. vLLM's repetition_penalty makes every
# token already seen (prompt or output) less likely by one constant factor, whatever its count. Code repeats
# newlines, indentation, `self` and brackets hundreds of times in a 16,000-token answer, so frequency_penalty, which
# grows with each repeat, would push those out late in a long answer; presence_penalty is flat but additive and
# blunt. A mild 1.05 keeps identifiers copied from the prompt cheap (the model's generation_config.json sets none:
# vLLM's log names only temperature, top_k and top_p). It does not escalate, so it may be too weak for a hard loop:
# the VM retest tries 1.05, 1.1 and 1.15 through the constructor argument.
GLIMMER_REPETITION_PENALTY: Final = 1.05
MAX_REPETITION_PENALTY: Final = 2.0
# Glimmer's ATEM framing, from the chat template: the generation prompt ends '<|start|>assistant', and each message the
# model writes is ' to=<recipient><|message|><body>', ended by <|eom|> or <|eot|> (or the stop the server cut).
# A new message starts only after an end token, so a literal <|start|> inside a body (quoted from a file) stays in it.
# A header with no recipient is the user's: the template writes '<|start|>assistant<|message|>' for a falsy recipient.
ATEM_MESSAGE: Final = "<|message|>"
_ATEM_END: Final = re.compile(r"<\|(?:eom|eot)\|>")
_ATEM_NEXT: Final = re.compile(r"<\|(?:eom|eot)\|>\s*<\|start\|>")
_ATEM_HEADER: Final = re.compile(r"\s*(?:<\|start\|>)?\s*(?:assistant)?\s*(?:to=(?P<recipient>\S+))?\s*")

assert GLIMMER_MODEL in ALLOWED_MODELS


class GlimmerError(CoreCallError):
    """A Glimmer call that produced no response. `cost` holds its seconds (micro_usd 0) once the call began, and the
    tokens the server reported when a 200 it sent could not be used."""


class GlimmerConfigError(GlimmerError):
    """No usable base URL in RECEIPTS_GLIMMER_URL or the constructor."""


class GlimmerAPIError(GlimmerError):
    """The server refused the call, or a transport failure that is not retried, or retries ran out."""

    def __init__(
        self, message: str, *, status: int | None = None, attempts: int = 1, reason: UncheckedReason | None = None
    ) -> None:
        super().__init__(message, reason=reason)
        self.status = status
        self.attempts = attempts


class GlimmerResponseError(GlimmerError):
    """A 200 response the client cannot use: malformed, without usage, or naming a model other than Glimmer."""


def glimmer_base_url(value: str | None = None) -> str:
    """The server's base URL from the argument or RECEIPTS_GLIMMER_URL: http or https, a host, no credentials, query or
    fragment. A trailing slash is dropped; the path is kept (normally /v1)."""
    raw = (value if value is not None else os.environ.get(GLIMMER_URL_ENV, "")).strip()
    if not raw:
        raise GlimmerConfigError(
            f"no Glimmer server: set {GLIMMER_URL_ENV} to its base URL, such as http://host:8000/v1"
        )
    try:
        url = httpx.URL(raw)
    except httpx.InvalidURL:
        raise GlimmerConfigError(f"{GLIMMER_URL_ENV} is not a URL") from None
    if url.scheme not in ("http", "https") or not url.host:
        raise GlimmerConfigError(f"{GLIMMER_URL_ENV} needs an http or https URL with a host")
    if url.userinfo or url.query or url.fragment:
        raise GlimmerConfigError(f"{GLIMMER_URL_ENV} takes no credentials, query or fragment")
    return str(url).rstrip("/")


class ModalToken:
    """Modal's proxy-auth token. Its repr and str never show it; it is attached to each request, never to the client's
    default headers, and scrub() takes both values out of any text before it is kept or shown."""

    __slots__ = ("_key", "_secret")

    def __init__(self, key: str, secret: str) -> None:
        self._key = key
        self._secret = secret

    def headers(self) -> dict[str, str]:
        return {"Modal-Key": self._key, "Modal-Secret": self._secret}

    def scrub(self, text: str) -> str:
        for needle in (self._secret, self._key):
            text = text.replace(needle, "[redacted]")
        return text

    def __repr__(self) -> str:
        return "ModalToken([redacted])"

    __str__ = __repr__

    def __reduce__(self) -> NoReturn:
        raise TypeError("a ModalToken cannot be pickled")


def modal_token(base_url: str, env: Mapping[str, str]) -> ModalToken | None:
    """The proxy-auth token for a base URL on Modal, and None for any other host, so the token never goes to Falden's
    own VM or anywhere else. On Modal the URL must be https with a plain host name (letters, digits, hyphens and dots,
    no trailing dot), and both values visible ASCII, else a GlimmerConfigError that names only the variable. The
    suffix matches any Modal workspace: the URL is the operator's own configuration, beside the token itself."""
    url = httpx.URL(base_url)
    host = url.host
    if host.rstrip(".").endswith(MODAL_HOST_SUFFIX) and host.endswith("."):
        raise GlimmerConfigError(f"{GLIMMER_URL_ENV}: a Modal host takes no trailing dot")
    if not host.endswith(MODAL_HOST_SUFFIX):
        return None
    if url.scheme != "https":
        raise GlimmerConfigError(f"{GLIMMER_URL_ENV}: a Glimmer server on Modal is reached over https only")
    if not _MODAL_HOST.fullmatch(host):
        raise GlimmerConfigError(f"{GLIMMER_URL_ENV}: a Modal host is letters, digits, hyphens and dots")
    values = {name: env.get(name, "").strip() for name in (MODAL_KEY_ENV, MODAL_SECRET_ENV)}
    missing = [name for name, value in values.items() if not value]
    if missing:
        raise GlimmerConfigError(f"a Glimmer server on Modal needs {' and '.join(missing)} in the environment")
    bad = [name for name, value in values.items() if not _TOKEN_CHARS.fullmatch(value)]
    if bad:
        raise GlimmerConfigError(f"{' and '.join(bad)} must be visible ASCII, with no spaces or line breaks")
    return ModalToken(values[MODAL_KEY_ENV], values[MODAL_SECRET_ENV])


# The properties that carry a whole source file in a Glimmer json_schema: stage 1's candidates (attack.correctness) and
# the agent loop's write stage (agentloop.attacker.WRITE_SCHEMA). No other schema Glimmer gets has either name.
SOURCE_FIELDS: Final = ("code", "content")
LINES_SUFFIX: Final = "_lines"


def _lines_property() -> dict[str, object]:
    return {
        "type": "array",
        "items": {"type": "string"},
        "description": "the whole file, one source line per item, indentation kept",
    }


def lines_schema(schema: Mapping[str, object]) -> dict[str, object] | None:
    """The schema Glimmer gets for `schema`: each string property named in SOURCE_FIELDS becomes "<name>_lines", an
    array of strings, at the same place and as required as it was (module docstring). None when nothing changes.
    `schema` is not modified."""
    changed = False

    def walk(node: object) -> object:
        nonlocal changed
        if isinstance(node, list):
            return [walk(item) for item in node]
        if not isinstance(node, Mapping):
            return node
        out = {key: walk(value) for key, value in node.items()}
        props = out.get("properties")
        if isinstance(props, dict):
            names = {
                name: name + LINES_SUFFIX
                for name in SOURCE_FIELDS
                if isinstance(props.get(name), Mapping)
                and props[name].get("type") == "string"
                and name + LINES_SUFFIX not in props
            }
            if names:
                changed = True
                out["properties"] = {
                    names.get(key, key): (_lines_property() if key in names else value) for key, value in props.items()
                }
                required = out.get("required")
                if isinstance(required, list):
                    out["required"] = [names.get(name, name) for name in required]
        return out

    rewritten = walk(schema)
    assert isinstance(rewritten, dict)
    return rewritten if changed else None


def joined_code(value: object) -> object:
    """`value` with each "<name>_lines" array of strings (a name in SOURCE_FIELDS) joined back into "<name>", one line
    per item, at the same place; an object that also has "<name>", or lines that are not all strings, is left for the
    caller's validation to refuse."""
    if isinstance(value, list):
        return [joined_code(item) for item in value]
    if not isinstance(value, dict):
        return value
    out = {key: joined_code(item) for key, item in value.items()}
    for name in SOURCE_FIELDS:
        lines = out.get(name + LINES_SUFFIX)
        if name not in out and isinstance(lines, list) and all(isinstance(line, str) for line in lines):
            text = "\n".join(lines)
            out = {
                (name if key == name + LINES_SUFFIX else key): (
                    (text + "\n" if text else text) if key == name + LINES_SUFFIX else item
                )
                for key, item in out.items()
            }
    return out


def user_message(text: str) -> str:
    """The body of the last ATEM message Glimmer wrote to the user, from an answer with its special tokens kept; text
    with no ATEM framing is returned as is. An answer with framing but no message to the user (only reasoning, or a
    tool channel) is empty, which the caller treats as an empty answer."""
    if ATEM_MESSAGE not in text:
        return text
    answer: str | None = None
    for segment in _ATEM_NEXT.split(text):
        header, sep, body = segment.partition(ATEM_MESSAGE)
        if not sep:
            continue  # a header the server cut before its body
        found = _ATEM_HEADER.fullmatch(header)
        if found is not None and found["recipient"] in (None, "user"):
            answer = _ATEM_END.split(body, maxsplit=1)[0]
    if answer is None:
        # The loop's trace keeps only the answer it got, so show what the model sent (advisory, 10/30 check).
        log.warning("glimmer answer framed with no message to the user: %r", clip(text, 400))
        return ""
    return answer.strip()


def encode_request(request: ModelRequest, *, repetition_penalty: float = GLIMMER_REPETITION_PENALTY) -> bytes:
    """The Chat Completions body vLLM gets for a request. repetition_penalty and skip_special_tokens are vLLM's extra
    sampling parameters; a request without json_schema keeps the special tokens so user_message can read the ATEM
    channels (module docstring)."""
    body: dict[str, object] = {
        "model": request.model,
        "messages": [{"role": m.role, "content": m.content} for m in request.messages],
        "temperature": request.temperature,
        "max_tokens": request.max_tokens,
        "repetition_penalty": repetition_penalty,
    }
    lines = lines_schema(request.json_schema) if request.json_schema is not None else None
    fmt = response_format(request if lines is None else request.model_copy(update={"json_schema": lines}))
    if fmt is not None:
        body["response_format"] = fmt
    else:
        body["skip_special_tokens"] = False
    return json.dumps(body, ensure_ascii=False, separators=(",", ":")).encode()


@dataclass(frozen=True)
class GlimmerCall:
    """One successful call, as logged."""

    stage: str
    finish_reason: str
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    seconds: float
    attempts: int
    request_id: str | None


@dataclass(frozen=True)
class _Failure:
    detail: str
    retryable: bool
    timeout: bool = False
    status: int | None = None
    retry_after: float | None = None


class GlimmerClient:
    """contracts.ModelClient for muse-glimmer-30b on Falden's own vLLM. It meters no money, and holds a secret only
    for a server on Modal, its proxy-auth token (module docstring). `env` stands in for os.environ in tests."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        max_retries: int = DEFAULT_RETRIES,
        max_timeout_retries: int = DEFAULT_TIMEOUT_RETRIES,
        max_wait_s: float = 120.0,
        timeout_s: float = 600.0,
        transport: httpx.BaseTransport | None = None,
        clock: Callable[[], float] = time.monotonic,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], datetime] | None = None,
        rng: random.Random | None = None,
        repetition_penalty: float = GLIMMER_REPETITION_PENALTY,
        env: Mapping[str, str] | None = None,
    ) -> None:
        if not 0 <= max_retries <= MAX_RETRIES:
            raise ValueError(f"max_retries is 0 to {MAX_RETRIES}")
        if not 0 <= max_timeout_retries <= MAX_RETRIES:
            raise ValueError(f"max_timeout_retries is 0 to {MAX_RETRIES}")
        if isinstance(repetition_penalty, bool) or not 1.0 <= repetition_penalty <= MAX_REPETITION_PENALTY:
            raise ValueError(f"repetition_penalty is 1.0 (off) to {MAX_REPETITION_PENALTY}")
        self.base_url = glimmer_base_url(base_url)
        self._max_retries = max_retries
        self._max_timeout_retries = max_timeout_retries
        self._max_wait_s = max_wait_s
        self.repetition_penalty = float(repetition_penalty)
        self._clock = clock
        self._sleep = sleep
        self._now = now if now is not None else (lambda: datetime.now(UTC))
        self._rng = rng if rng is not None else random.Random()
        self._token = modal_token(self.base_url, os.environ if env is None else env)
        self._http = httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(timeout_s, connect=min(10.0, timeout_s)),
            transport=transport,
            follow_redirects=False,
            trust_env=False,
            headers={"Accept": "application/json", "User-Agent": f"falden-receipts/{__version__}"},
        )
        self._lock = threading.Lock()
        self._calls: list[GlimmerCall] = []

    def __repr__(self) -> str:
        return f"GlimmerClient(base_url={self.base_url!r})"

    def _scrub(self, text: str) -> str:
        """Text with the Modal token taken out, before it goes into a failure's detail."""
        return self._token.scrub(text) if self._token is not None else text

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self, exc_type: type[BaseException] | None, exc: BaseException | None, tb: TracebackType | None
    ) -> None:
        self.close()

    def close(self) -> None:
        self._http.close()

    def calls(self) -> tuple[GlimmerCall, ...]:
        with self._lock:
            return tuple(self._calls)

    def complete(self, request: ModelRequest) -> ModelResponse:
        """One call. The response's cost has its tokens and seconds, waits included, and micro_usd 0; an error raised
        once the call began carries the seconds too, and the usage the server reported, if a 200 reported one."""
        model = check_model(request.model)  # again here: a request built with model_construct skips its validator
        if model != GLIMMER_MODEL:
            raise ModelRefusedError(f"{model}: the Glimmer client serves only {GLIMMER_MODEL}")
        try:
            body = encode_request(request, repetition_penalty=self.repetition_penalty)
        except UnicodeEncodeError:
            raise GlimmerError(
                f"stage {request.stage!r}: the request text is not valid Unicode (a lone surrogate)"
            ) from None
        started = self._clock()
        reported: list[Usage] = []  # what a 200 reported, kept when the answer cannot be used
        try:
            attempt = timeouts = 0
            while True:
                attempt += 1
                outcome = self._attempt(request, body, reported)
                if isinstance(outcome, ModelResponse):
                    response = outcome.model_copy(update={"cost": self._cost(request.stage, outcome.usage, started)})
                    self._log_success(request, response, attempt)
                    return response
                where = f"stage {request.stage!r}, model {model}"
                reason = UncheckedReason.TIMEOUT if outcome.timeout else UncheckedReason.RUNTIME
                timeouts += outcome.timeout
                if not outcome.retryable or attempt > self._max_retries or timeouts > self._max_timeout_retries:
                    tried = f"after {attempt} attempts; " if attempt > 1 else ""
                    raise GlimmerAPIError(
                        f"{tried}{outcome.detail} ({where})", status=outcome.status, attempts=attempt, reason=reason
                    )
                wait = backoff_seconds(attempt, outcome.retry_after, self._rng)
                if wait > self._max_wait_s:
                    raise GlimmerAPIError(
                        f"asked to wait {wait:.0f} s, past the {self._max_wait_s:.0f} s limit; last: {outcome.detail} "
                        f"({where})",
                        status=outcome.status,
                        attempts=attempt,
                        reason=reason,
                    )
                log.warning(
                    "glimmer retry stage=%s attempt=%d wait_s=%.2f cause=%s",
                    request.stage,
                    attempt,
                    wait,
                    outcome.detail,
                )
                self._sleep(wait)
        except GlimmerError as exc:
            exc.cost = self._cost(request.stage, reported[-1] if reported else None, started)
            raise

    def _attempt(self, request: ModelRequest, body: bytes, reported: list[Usage]) -> ModelResponse | _Failure:
        failure: _Failure
        try:
            headers = {"Content-Type": "application/json", **(self._token.headers() if self._token else {})}
            reply = self._http.post("/chat/completions", content=body, headers=headers)
        except (httpx.TimeoutException, httpx.NetworkError, httpx.RemoteProtocolError) as exc:
            failure = _Failure(
                detail=f"{type(exc).__name__}: {clip(self._scrub(str(exc)))}",
                retryable=True,
                timeout=isinstance(exc, httpx.TimeoutException),
            )
            return failure
        except httpx.HTTPError as exc:
            failure = _Failure(detail=f"{type(exc).__name__}: {clip(self._scrub(str(exc)))}", retryable=False)
            return failure
        if reply.status_code != 200:
            status = reply.status_code
            return _Failure(
                detail=clip(self._scrub(error_detail(status, reply.content))),
                retryable=status == 429 or 500 <= status <= 599,
                timeout=status == 504,
                status=status,
                retry_after=parse_retry_after(reply.headers.get("retry-after"), self._now()),
            )
        return self._response(request, reply.content, reported)

    def _response(self, request: ModelRequest, content: bytes, reported: list[Usage]) -> ModelResponse:
        data = json_or_none(content)
        wire_usage = usage_or_none(data)
        if wire_usage is None:
            raise GlimmerResponseError(f"stage {request.stage!r}: a 200 response without valid usage")
        reported.append(wire_usage.usage())  # the server's own count, kept even if the answer cannot be used
        problem: str | None = None
        try:
            completion = WireCompletion.model_validate(data)
        except ValidationError as exc:
            problem = "; ".join(
                f"{'.'.join(str(p) for p in e['loc']) or '(root)'}: {e['msg']}"
                for e in exc.errors(include_url=False, include_context=False, include_input=False)
            )
        if problem is not None:
            raise GlimmerResponseError(f"stage {request.stage!r}: a malformed 200 response ({clip(problem)})")
        if completion.model != GLIMMER_MODEL:
            raise GlimmerResponseError(
                f"stage {request.stage!r}: the response names {clip(completion.model, 64)!r}, not {GLIMMER_MODEL}"
            )
        choice = completion.choices[0]
        text = choice.message.content or ""
        parsed = None
        if request.json_schema is None:
            text = user_message(text)
        elif choice.finish_reason == "stop":
            parsed = json_object(text)
            if parsed is not None and lines_schema(request.json_schema) is not None:
                joined = joined_code(parsed)
                assert isinstance(joined, dict)
                parsed = joined
                text = json.dumps(joined)  # ASCII escapes: a lone surrogate the model escaped stays encodable
        return ModelResponse(
            model=GLIMMER_MODEL,
            text=text,
            finish_reason=choice.finish_reason,
            usage=wire_usage.usage(),
            request_id=clip(completion.id, 128) if completion.id else None,
            parsed=parsed,
        )

    def _cost(self, stage: str, usage: Usage | None, started: float) -> StageCost:
        return StageCost(
            stage=stage,
            model=GLIMMER_MODEL,
            input_tokens=usage.input_tokens if usage else 0,
            cached_tokens=usage.cached_tokens if usage else 0,
            output_tokens=usage.output_tokens if usage else 0,
            reasoning_tokens=usage.reasoning_tokens if usage else 0,
            micro_usd=0,  # Falden's own GPU: compute hours, never Meta's meter
            seconds=max(self._clock() - started, 0.0),
        )

    def _log_success(self, request: ModelRequest, response: ModelResponse, attempts: int) -> None:
        usage, cost = response.usage, response.cost
        assert cost is not None
        call = GlimmerCall(
            stage=request.stage,
            finish_reason=response.finish_reason,
            input_tokens=usage.input_tokens,
            cached_tokens=usage.cached_tokens,
            output_tokens=usage.output_tokens,
            reasoning_tokens=usage.reasoning_tokens,
            seconds=cost.seconds,
            attempts=attempts,
            request_id=response.request_id,
        )
        with self._lock:
            self._calls.append(call)
        log.info(
            "glimmer call stage=%s model=%s finish_reason=%s input_tokens=%d cached_tokens=%d output_tokens=%d "
            "reasoning_tokens=%d attempts=%d seconds=%.3f request_id=%r",
            call.stage,
            GLIMMER_MODEL,
            call.finish_reason,
            call.input_tokens,
            call.cached_tokens,
            call.output_tokens,
            call.reasoning_tokens,
            call.attempts,
            call.seconds,
            clip(call.request_id or "", 80),
        )
        if response.truncated:
            log.warning(
                "glimmer call stage=%s truncated (finish_reason length, or an empty answer): it invalidates the item",
                call.stage,
            )
