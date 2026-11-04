"""RT1, the Meta client. Every request goes to httpx.MockTransport: no test calls a model or the network, and the key
is a stand-in made fresh for each test. Each check is fed the input it must refuse."""

from __future__ import annotations

import json
import logging
import os
import pickle
import random
import secrets
import shutil
import subprocess
import sys
import threading
import traceback
from collections.abc import Callable, Iterable, Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict, Field

from receipts.contracts import (
    Message,
    ModelCallError,
    ModelRefusedError,
    ModelRequest,
    ModelResponse,
    StageCost,
    UncheckedReason,
    Usage,
    check_model,
)
from receipts.core import meta_client
from receipts.core.errors import CoreCallError
from receipts.core.glimmer_client import (
    GLIMMER_MODEL,
    GlimmerAPIError,
    GlimmerConfigError,
    GlimmerError,
    GlimmerResponseError,
)
from receipts.core.meta_client import (
    API_KEY_ENV,
    ARM_COMMAND,
    DEFAULT_SPEND_CAP_MICRO_USD,
    DEFAULT_TPM,
    HIGHEST_PRICE,
    LIST_PRICES,
    MAX_RETRIES,
    SPEND_CEILING_MICRO_USD,
    SPEND_DIR_MODE,
    SPEND_FILE_ENV,
    SPEND_FILE_MODE,
    ArmRecord,
    HoldRecord,
    MetaAPIError,
    MetaClient,
    MetaClientError,
    MetaResponseError,
    MetaRetriesExhaustedError,
    MissingKeyError,
    SettleRecord,
    SpendCapError,
    SpendFileError,
    SpendMeter,
    SpendNotArmedError,
    TokenGovernor,
    UnpricedModelError,
    backoff_seconds,
    default_spend_path,
    encode_request,
    named_spend_file,
    parse_retry_after,
    price_for,
    shared_governor,
    shared_meter,
    stage_cost,
    usd,
)
from receipts.core.structured import Parsed, Unparsed, complete_structured, response_costs

SPARK = "muse-spark-1.3"
WALL_START = datetime(2026, 10, 27, 12, 0, tzinfo=UTC)
STAMP = "2026-10-27T12:00:00Z"


class FakeClock:
    """Monotonic time that moves only when the code under test sleeps."""

    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds

    def wall(self) -> datetime:
        return WALL_START + timedelta(seconds=self.now)


Reply = httpx.Response | Callable[[httpx.Request], httpx.Response]


class FakeMeta:
    """A scripted stand-in for the API behind httpx.MockTransport. Each request takes the next reply."""

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.replies: list[Reply] = []
        self.requests: list[httpx.Request] = []
        self.sent_at: list[float] = []

    def queue(self, *replies: Reply) -> None:
        self.replies.extend(replies)

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.sent_at.append(self.clock.now)
        if not self.replies:
            raise AssertionError("the fake API got a request it has no reply for")
        reply = self.replies.pop(0)
        return reply(request) if callable(reply) else reply

    def bodies(self) -> list[dict[str, Any]]:
        return [json.loads(request.content) for request in self.requests]


def completion(
    content: str | None = "ok",
    *,
    finish: str = "stop",
    prompt: int = 100,
    cached: int = 0,
    output: int = 50,
    reasoning: int = 0,
    model: str = SPARK,
    request_id: str = "chatcmpl-1",
    headers: dict[str, str] | None = None,
) -> httpx.Response:
    """A 200 in the documented Chat Completions shape."""
    body = {
        "id": request_id,
        "object": "chat.completion",
        "created": 1_793_000_000,
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": content, "refusal": None},
                "finish_reason": finish,
                "logprobs": None,
            }
        ],
        "usage": {
            "prompt_tokens": prompt,
            "completion_tokens": output,
            "total_tokens": prompt + output,
            "prompt_tokens_details": {"cached_tokens": cached},
            "completion_tokens_details": {"reasoning_tokens": reasoning},
        },
    }
    return httpx.Response(200, json=body, headers=headers)


def api_error(
    status: int, error_type: str, code: str | None, message: str, *, retry_after: str | None = None
) -> httpx.Response:
    """An error in the documented envelope."""
    headers = {"Retry-After": retry_after} if retry_after is not None else None
    envelope = {"error": {"message": message, "type": error_type, "param": None, "code": code}}
    return httpx.Response(status, json=envelope, headers=headers)


def read_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ReadTimeout("timed out waiting for the response", request=request)


def connect_timeout(request: httpx.Request) -> httpx.Response:
    raise httpx.ConnectTimeout("timed out connecting", request=request)


def spark_request(**changes: object) -> ModelRequest:
    fields: dict[str, object] = {
        "stage": "intake",
        "model": SPARK,
        "messages": [
            {"role": "system", "content": "Extract the claims."},
            {"role": "user", "content": "Fixes the crash."},
        ],
        "temperature": 0.0,
        "max_tokens": 256,
        "reasoning_effort": "low",
    }
    fields.update(changes)
    return ModelRequest.model_validate(fields)


def worst_case(request: ModelRequest) -> int:
    """A call's hold, and what an attempt that may have been billed is metered at: its token bounds at the highest list
    price on file, whichever model was asked for."""
    return HIGHEST_PRICE.worst_case_micro_usd(len(encode_request(request)), request.max_tokens)


def armed_meter(path: Path, cap: int = DEFAULT_SPEND_CAP_MICRO_USD, **options: Any) -> SpendMeter:
    """A meter on its own spend file, armed at the credit the way an operator arms one once the credit is confirmed."""
    meter = SpendMeter(path, cap_micro_usd=cap, **options)
    meter.arm(credit_confirmed=True)
    return meter


def hold_line(hold_id: str, micro_usd: int = 10) -> str:
    return HoldRecord(
        v=2, kind="hold", at=STAMP, id=hold_id, stage="intake", model=SPARK, micro_usd=micro_usd
    ).model_dump_json()


def settle_line(hold_id: str, micro_usd: int = 10) -> str:
    return SettleRecord(
        v=2, kind="settle", at=STAMP, hold=hold_id, stage="intake", model=SPARK, micro_usd=micro_usd, estimated=False
    ).model_dump_json()


def arm_line(inode: int, cap: int = DEFAULT_SPEND_CAP_MICRO_USD) -> str:
    return ArmRecord(v=2, kind="arm", at=STAMP, credit_confirmed=True, cap_micro_usd=cap, inode=inode).model_dump_json()


def arm_json(inode: int, cap: int, **extra: object) -> str:
    """An arm line written by hand, in the shape of a record from before above_credit unless `extra` adds it."""
    fields = {"v": 2, "at": STAMP, "kind": "arm", "credit_confirmed": True, "cap_micro_usd": cap, "inode": inode}
    return json.dumps({**fields, **extra}) + "\n"


def private_spend_file(path: Path, build: Callable[[int], str]) -> Path:
    """A spend file written by hand the way 'spend arm' makes one: mode 0600 in a mode 0700 directory. `build` gets
    the file's inode, so an arm record in it can name its own file."""
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    path.touch(mode=0o600)
    path.chmod(0o600)
    path.write_text(build(path.stat().st_ino))
    return path


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")

    bug: bool
    reason: str = Field(min_length=1)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def meta(clock: FakeClock) -> FakeMeta:
    return FakeMeta(clock)


@pytest.fixture
def fake_key() -> str:
    """A stand-in in the documented LLM_{id}_{secret} shape, made fresh for each test. It is never a real key."""
    return f"LLM_{secrets.randbelow(10**15):015d}_{secrets.token_urlsafe(12)}_{secrets.token_urlsafe(12)}"


@pytest.fixture
def spend_path(tmp_path: Path) -> Path:
    return tmp_path / "spend.jsonl"


@pytest.fixture
def make_client(
    meta: FakeMeta, clock: FakeClock, fake_key: str, spend_path: Path
) -> Iterator[Callable[..., MetaClient]]:
    made: list[MetaClient] = []
    armed: list[SpendMeter] = []

    def make(**options: Any) -> MetaClient:
        options.setdefault("api_key", fake_key)
        options.setdefault("governor", TokenGovernor(clock=clock.monotonic, sleep=clock.sleep))
        if "meter" not in options:  # one armed meter per test, shared by the clients it makes
            if not armed:
                armed.append(armed_meter(spend_path))
            options["meter"] = armed[0]
        client = MetaClient(
            transport=httpx.MockTransport(meta.handler),
            clock=clock.monotonic,
            sleep=clock.sleep,
            now=clock.wall,
            rng=random.Random(7),
            **options,
        )
        made.append(client)
        return client

    yield make
    for client in made:
        client.close()


# ---------------------------------------------------------------- a call that works


def test_a_successful_call_is_validated_metered_and_logged(
    make_client: Callable[..., MetaClient], meta: FakeMeta, fake_key: str, caplog: pytest.LogCaptureFixture
) -> None:
    rate_headers = {"x-ratelimit-remaining-tokens": "3999000", "x-ratelimit-remaining-requests": "2999"}
    meta.queue(completion("hello", prompt=1_000, cached=400, output=500, reasoning=120, headers=rate_headers))
    client = make_client()
    with caplog.at_level(logging.INFO, logger="receipts.core.meta_client"):
        response = client.complete(spark_request())

    assert response.text == "hello"
    assert response.finish_reason == "stop"
    assert not response.truncated
    assert response.usage == Usage(input_tokens=1_000, cached_tokens=400, output_tokens=500, reasoning_tokens=120)
    assert response.request_id == "chatcmpl-1"
    assert response.model == SPARK
    assert response.parsed is None

    (sent,) = meta.requests
    # The literal host and path, not the module's own constant: the key goes to api.meta.ai and nowhere else.
    assert str(sent.url) == "https://api.meta.ai/v1/chat/completions"
    assert (sent.url.scheme, sent.url.host, sent.url.port) == ("https", "api.meta.ai", None)
    assert sent.headers["authorization"] == f"Bearer {fake_key}"
    assert json.loads(sent.content) == {
        "model": SPARK,
        "messages": [
            {"role": "system", "content": "Extract the claims."},
            {"role": "user", "content": "Fixes the crash."},
        ],
        "temperature": 0.0,
        "max_completion_tokens": 256,
        "prompt_cache_key": "receipts-intake",
        "reasoning_effort": "low",
    }

    # 600 uncached input tokens at $1.25, 400 cached at $0.15 and 500 output at $4.25 per 1M: 2,935 micro-dollars.
    # The 120 reasoning tokens are part of the 500 output tokens, billed with them and counted on their own (R23).
    cost = StageCost(
        stage="intake",
        model=SPARK,
        input_tokens=1_000,
        cached_tokens=400,
        output_tokens=500,
        reasoning_tokens=120,
        micro_usd=2_935,
        seconds=0.0,
    )
    assert response.cost == cost
    assert client.meter.spent_micro_usd == 2_935
    assert client.meter.held_micro_usd == 0
    assert client.stage_costs() == (cost,)
    (record,) = client.records()
    assert (record.finish_reason, record.reasoning_tokens, record.attempts, record.micro_usd) == ("stop", 120, 1, 2_935)
    (line,) = [r.getMessage() for r in caplog.records if r.getMessage().startswith("meta call ")]
    for part in (
        "stage=intake",
        "finish_reason=stop",
        "input_tokens=1000",
        "cached_tokens=400",
        "output_tokens=500",
        "reasoning_tokens=120",
        "micro_usd=2935",
        "remaining_tokens=3999000",
        "remaining_requests=2999",
    ):
        assert part in line


def test_a_schema_request_sends_response_format_and_parses_the_object(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    meta.queue(completion('{"bug": true, "reason": "off by one"}'), completion('{"bug": false, "reason": "fine"}'))
    client = make_client()
    response = client.complete(spark_request(json_schema=Verdict.model_json_schema()))
    assert response.parsed == {"bug": True, "reason": "off by one"}
    # A schema_name names the format over the schema's title, cut to the characters the API allows.
    client.complete(spark_request(json_schema=Verdict.model_json_schema(), schema_name="verdict v2"))
    first, second = meta.bodies()
    assert first["response_format"] == {
        "type": "json_schema",
        "json_schema": {"name": "Verdict", "schema": Verdict.model_json_schema()},
    }
    assert second["response_format"]["json_schema"]["name"] == "verdict_v2"


def test_the_answering_model_is_reported_as_the_server_names_it(
    make_client: Callable[..., MetaClient], meta: FakeMeta, caplog: pytest.LogCaptureFixture
) -> None:
    meta.queue(completion(model="muse-spark-1.2"))
    with caplog.at_level(logging.WARNING, logger="receipts.core.meta_client"):
        response = make_client().complete(spark_request())
    assert response.model == "muse-spark-1.2"
    assert "asked for muse-spark-1.3, answered by 'muse-spark-1.2'" in caplog.text


# ---------------------------------------------------------------- the egress pin


@pytest.mark.parametrize("status", [301, 302, 303, 307, 308])
def test_a_redirect_is_never_followed(make_client: Callable[..., MetaClient], meta: FakeMeta, status: int) -> None:
    def redirect(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers={"Location": "https://collector.example.net/v1/chat/completions"})

    meta.queue(
        redirect, completion("followed")
    )  # a client that followed the redirect would get this from the other host
    client = make_client()
    with pytest.raises(MetaAPIError) as caught:
        client.complete(spark_request())
    assert caught.value.status == status
    assert [request.url.host for request in meta.requests] == ["api.meta.ai"]
    assert client.meter.spent_micro_usd == 0


def test_the_client_takes_no_proxy_or_certificate_setting_from_the_environment(
    monkeypatch: pytest.MonkeyPatch, fake_key: str, spend_path: Path, tmp_path: Path
) -> None:
    for name in ("HTTPS_PROXY", "https_proxy", "ALL_PROXY", "all_proxy"):
        monkeypatch.setenv(name, "http://proxy.invalid:3128")
    monkeypatch.setenv("SSL_CERT_FILE", str(tmp_path / "planted-ca.pem"))  # absent, so honoring it fails at once
    built: list[dict[str, Any]] = []
    original = httpx.HTTPTransport.__init__

    def recording(self: httpx.HTTPTransport, *args: Any, **kwargs: Any) -> None:
        built.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setattr(httpx.HTTPTransport, "__init__", recording)
    with MetaClient(api_key=fake_key, meter=armed_meter(spend_path)):  # no transport given: the client builds its own
        pass
    (transport,) = built  # one transport, and no proxy transport beside it
    assert transport.get("proxy") is None
    assert transport.get("trust_env") is False


# ---------------------------------------------------------------- structured output


def test_a_schema_failure_gets_one_retry_that_carries_the_validation_error(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    meta.queue(completion('{"bug": "maybe"}'), completion('{"bug": true, "reason": "off by one"}'))
    result = complete_structured(make_client(), spark_request(), Verdict)

    assert isinstance(result, Parsed)
    assert result.value == Verdict(bug=True, reason="off by one")
    assert len(result.responses) == 2
    first, second = meta.bodies()
    assert first["response_format"]["json_schema"]["schema"] == Verdict.model_json_schema()
    assert second["messages"][:2] == first["messages"]
    assert second["messages"][2] == {"role": "assistant", "content": '{"bug": "maybe"}'}
    retry = second["messages"][3]
    assert retry["role"] == "user"
    assert "bug: Input should be a valid boolean" in retry["content"]
    assert "reason: Field required" in retry["content"]


def test_two_schema_failures_give_an_unparsed_result(make_client: Callable[..., MetaClient], meta: FakeMeta) -> None:
    meta.queue(completion("not json at all"), completion('{"bug": true}'))
    result = complete_structured(make_client(), spark_request(), Verdict)

    assert isinstance(result, Unparsed)
    assert result.reason is UncheckedReason.UNPARSED
    assert not result.truncated
    assert len(meta.requests) == 2
    unchecked = result.unchecked("t1:intake")
    assert unchecked.reason is UncheckedReason.UNPARSED
    assert "reason: Field required" in unchecked.detail


@pytest.mark.parametrize("finish", ["length", "content_filter"])
def test_a_response_that_did_not_finish_invalidates_the_item_without_a_retry(
    make_client: Callable[..., MetaClient], meta: FakeMeta, caplog: pytest.LogCaptureFixture, finish: str
) -> None:
    valid = '{"bug": true, "reason": "x"}'
    meta.queue(completion(valid, finish=finish))
    client = make_client()
    result = complete_structured(client, spark_request(), Verdict)

    assert isinstance(result, Unparsed)
    assert result.truncated is (finish == "length")
    assert finish in result.detail
    assert len(meta.requests) == 1

    # The client never hands back a parsed object from a response that did not finish, even when the text parses,
    # and it warns about a truncated one itself (the structured helper's own warning is not counted here).
    meta.queue(completion(valid, finish=finish))
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="receipts.core.meta_client"):
        response = client.complete(spark_request(json_schema=Verdict.model_json_schema()))
    assert response.parsed is None
    assert response.truncated is (finish == "length")
    client_warnings = [
        r.getMessage() for r in caplog.records if r.name == "receipts.core.meta_client" and r.levelno == logging.WARNING
    ]
    assert any("truncated" in message for message in client_warnings) is (finish == "length")


# ---------------------------------------------------------------- retries


def test_a_429_waits_at_least_retry_after(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    meta.queue(
        api_error(429, "rate_limit_error", "rate_limit_exceeded", "Please retry after 7 seconds.", retry_after="7"),
        completion("ok"),
    )
    client = make_client()
    assert client.complete(spark_request()).text == "ok"
    (wait,) = clock.sleeps
    assert 7.0 <= wait < 7.5
    assert meta.sent_at == [0.0, wait]
    assert client.meter.spent_micro_usd == price_for(SPARK).micro_usd(Usage(input_tokens=100, output_tokens=50))


def test_a_503_retry_after_date_is_honored(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    later = "Tue, 27 Oct 2026 12:01:00 GMT"
    meta.queue(api_error(503, "server_error", "service_overloaded", "overloaded", retry_after=later), completion())
    make_client().complete(spark_request())
    (wait,) = clock.sleeps
    assert 60.0 <= wait < 60.5


def test_retry_after_reads_seconds_and_http_dates() -> None:
    assert parse_retry_after("7", WALL_START) == 7.0
    assert parse_retry_after(" 1.5 ", WALL_START) == 1.5
    assert parse_retry_after("Tue, 27 Oct 2026 12:00:30 GMT", WALL_START) == 30.0
    assert parse_retry_after("Tue, 27 Oct 2026 11:59:00 GMT", WALL_START) == 0.0
    assert parse_retry_after("-3", WALL_START) == 0.0
    for unreadable in (None, "", "soon", "nan", "inf"):
        assert parse_retry_after(unreadable, WALL_START) is None


def test_backoff_doubles_from_half_a_second_with_jitter_and_never_undercuts_retry_after() -> None:
    rng = random.Random(1)
    for attempt, step in ((1, 0.5), (2, 1.0), (3, 2.0), (4, 4.0), (5, 8.0)):
        assert step <= backoff_seconds(attempt, None, rng) <= step * 1.5
    assert backoff_seconds(1, 60.0, rng) >= 60.0


def test_server_errors_are_retried_at_most_five_times(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    overloaded = api_error(503, "server_error", "service_overloaded", "temporarily overloaded", retry_after="1")
    meta.queue(*[overloaded] * (MAX_RETRIES + 1))
    client = make_client()
    with pytest.raises(MetaRetriesExhaustedError) as caught:
        client.complete(spark_request())

    assert MAX_RETRIES == 5
    assert len(meta.requests) == 6
    assert len(clock.sleeps) == 5
    assert all(wait >= 1.0 for wait in clock.sleeps)
    assert (caught.value.attempts, caught.value.status, caught.value.reason) == (6, 503, UncheckedReason.RUNTIME)
    assert client.meter.spent_micro_usd == 0
    assert client.meter.held_micro_usd == 0  # every hold was settled, at 0


def test_a_server_error_then_a_success_returns_the_success(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    meta.queue(api_error(500, "server_error", None, "model output parsing failure"), completion("second time"))
    assert make_client().complete(spark_request()).text == "second time"
    assert len(meta.requests) == 2


def test_timeouts_are_retried_then_reported_as_timeouts_and_metered_at_the_worst_case(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    request = spark_request()
    meta.queue(*[read_timeout] * (MAX_RETRIES + 1))
    client = make_client()
    with pytest.raises(MetaRetriesExhaustedError) as caught:
        client.complete(request)

    assert (caught.value.attempts, caught.value.reason) == (6, UncheckedReason.TIMEOUT)
    assert len(meta.requests) == 6
    assert len(clock.sleeps) == 5
    # A read timeout may have run and been billed, so each one is metered at the worst case.
    assert client.meter.spent_micro_usd == 6 * worst_case(request)
    (cost,) = client.stage_costs()
    assert cost.micro_usd == 6 * worst_case(request)
    assert cost.seconds == pytest.approx(sum(clock.sleeps))
    # The error carries what the call metered, with no token counts, for the ledger.
    failed = caught.value.cost
    assert failed is not None
    assert (failed.stage, failed.model, failed.input_tokens, failed.output_tokens) == ("intake", SPARK, 0, 0)
    assert failed.micro_usd == 6 * worst_case(request)
    assert failed.seconds == pytest.approx(sum(clock.sleeps))


@pytest.mark.parametrize("error", [httpx.ReadError, httpx.RemoteProtocolError])
def test_a_dropped_connection_is_retried_and_metered_at_the_worst_case(
    make_client: Callable[..., MetaClient], meta: FakeMeta, error: type[httpx.TransportError]
) -> None:
    request = spark_request()

    def dropped(sent: httpx.Request) -> httpx.Response:
        raise error("the connection dropped after the request went out", request=sent)

    meta.queue(dropped, completion(prompt=1_000, output=0))
    client = make_client()
    response = client.complete(request)
    assert len(meta.requests) == 2
    assert client.meter.spent_micro_usd == worst_case(request) + 1_250
    # The response's cost covers both attempts; its tokens are the answered one's.
    assert response.cost is not None
    assert (response.cost.micro_usd, response.cost.input_tokens) == (worst_case(request) + 1_250, 1_000)


def test_a_body_that_cannot_be_decoded_is_metered_at_the_worst_case_and_not_retried(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    request = spark_request()

    def undecodable(sent: httpx.Request) -> httpx.Response:  # built on arrival, so decoding fails inside the client
        return httpx.Response(200, headers={"Content-Encoding": "gzip"}, content=b"this is not gzip")

    meta.queue(undecodable)
    client = make_client()
    with pytest.raises(MetaAPIError, match="DecodingError") as caught:
        client.complete(request)
    assert not isinstance(caught.value, MetaRetriesExhaustedError)
    assert len(meta.requests) == 1
    assert clock.sleeps == []
    assert client.meter.spent_micro_usd == worst_case(request)


def test_a_connect_timeout_is_retried_and_not_metered(make_client: Callable[..., MetaClient], meta: FakeMeta) -> None:
    meta.queue(connect_timeout, completion(prompt=1_000, output=0))
    client = make_client()
    client.complete(spark_request())
    assert len(meta.requests) == 2
    assert client.meter.spent_micro_usd == 1_250


@pytest.mark.parametrize("status", [400, 401, 402, 403, 404, 413])
def test_client_errors_are_not_retried(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock, status: int
) -> None:
    meta.queue(api_error(status, "invalid_request_error", None, "fix the request"))
    client = make_client()
    with pytest.raises(MetaAPIError) as caught:
        client.complete(spark_request())
    assert not isinstance(caught.value, MetaRetriesExhaustedError)
    assert caught.value.status == status
    assert "fix the request" in str(caught.value)
    assert len(meta.requests) == 1
    assert clock.sleeps == []
    assert client.meter.spent_micro_usd == 0


def test_a_gateway_timeout_is_not_retried_by_default_and_is_metered_at_the_worst_case(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    request = spark_request()
    meta.queue(api_error(504, "server_error", "gateway_timeout", "Request exceeded the non-streaming time limit"))
    client = make_client()
    with pytest.raises(MetaAPIError) as caught:
        client.complete(request)
    assert (caught.value.status, caught.value.reason) == (504, UncheckedReason.TIMEOUT)
    assert len(meta.requests) == 1
    assert clock.sleeps == []
    assert client.meter.spent_micro_usd == worst_case(request)

    meta.queue(api_error(504, "server_error", "gateway_timeout", "again"), completion("ok"))
    assert make_client(retry_gateway_timeout=True).complete(request).text == "ok"
    assert len(meta.requests) == 3


def test_a_retry_after_past_the_wait_limit_gives_up_at_once(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    meta.queue(api_error(429, "rate_limit_error", "rate_limit_exceeded", "slow down", retry_after="3600"))
    with pytest.raises(MetaRetriesExhaustedError, match="past the 120 s limit"):
        make_client().complete(spark_request())
    assert len(meta.requests) == 1
    assert clock.sleeps == []


def test_more_than_five_retries_is_refused(fake_key: str, spend_path: Path) -> None:
    with pytest.raises(ValueError, match="0 to 5"):
        MetaClient(api_key=fake_key, meter=armed_meter(spend_path), max_retries=6)


# ---------------------------------------------------------------- models and prices


@pytest.mark.parametrize(
    "model", ["muse-spark-1.3-contributor", "muse-spark-1.2-contributor", "MUSE-SPARK-1.3-Contributor", "gpt-x"]
)
def test_a_model_off_the_allowlist_is_refused_before_any_request(
    make_client: Callable[..., MetaClient], meta: FakeMeta, model: str
) -> None:
    # model_construct skips the contract's own validator, so this tests the client's check alone.
    request = ModelRequest.model_construct(
        stage="intake", model=model, messages=(Message(role="user", content="hi"),), temperature=0.0, max_tokens=16
    )
    client = make_client()
    with pytest.raises(ModelRefusedError, match=r"never used|not on the model allowlist") as caught:
        client.complete(request)
    assert not isinstance(caught.value, UnpricedModelError)  # refused by the allowlist itself, not by a missing price
    assert meta.requests == []
    assert client.meter.spent_micro_usd == 0


def test_an_allowed_model_with_no_list_price_is_refused_before_any_request(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    with pytest.raises(UnpricedModelError, match="refusing rather than guessing"):
        make_client().complete(spark_request(model="sam-3.1"))
    assert meta.requests == []


def test_request_text_that_is_not_valid_unicode_is_refused_before_any_request(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    broken = spark_request(messages=[{"role": "user", "content": "diff with a lone surrogate \ud800"}])
    with pytest.raises(MetaClientError, match="not valid Unicode") as caught:
        make_client().complete(broken)
    assert caught.value.reason is UncheckedReason.RUNTIME
    assert meta.requests == []


def test_a_response_naming_a_contributor_model_is_refused(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    meta.queue(completion(model="muse-spark-1.3-contributor"))
    with pytest.raises(MetaResponseError, match="contributor"):
        make_client().complete(spark_request())


def test_a_200_the_client_cannot_use_is_an_error_and_is_still_metered(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    request = spark_request()
    client = make_client()

    meta.queue(httpx.Response(200, content=b"<html>gateway page</html>"))
    with pytest.raises(MetaResponseError, match="without valid usage"):
        client.complete(request)
    assert client.meter.spent_micro_usd == worst_case(request)

    no_choices = {"id": "x", "model": SPARK, "choices": [], "usage": {"prompt_tokens": 1_000, "completion_tokens": 0}}
    meta.queue(httpx.Response(200, json=no_choices))
    with pytest.raises(MetaResponseError, match="malformed"):
        client.complete(request)
    assert client.meter.spent_micro_usd == worst_case(request) + 1_250

    meta.queue(completion(prompt=10, cached=20))  # a cached part larger than the prompt is not believed
    with pytest.raises(MetaResponseError, match="without valid usage"):
        client.complete(request)
    assert client.meter.spent_micro_usd == 2 * worst_case(request) + 1_250

    meta.queue(completion(finish="Stop Now", prompt=1_000, output=0))  # not a documented finish_reason word
    with pytest.raises(MetaResponseError, match="malformed"):
        client.complete(request)
    assert client.meter.spent_micro_usd == 2 * worst_case(request) + 2 * 1_250


def test_prices_are_list_prices_in_integer_micro_dollars_rounded_up() -> None:
    assert set(LIST_PRICES) == {"muse-spark-1.1", "muse-spark-1.2", "muse-spark-1.3", "muse-wonder-1.0"}
    assert all(check_model(model) == model for model in LIST_PRICES)
    spark, wonder = LIST_PRICES[SPARK], LIST_PRICES["muse-wonder-1.0"]
    assert (spark.input, spark.cached_input, spark.output) == (1_250_000, 150_000, 4_250_000)
    assert (wonder.input, wonder.cached_input, wonder.output) == (4_000_000, 200_000, 20_000_000)
    assert spark.micro_usd(Usage(input_tokens=1_000_000, output_tokens=0)) == 1_250_000
    assert spark.micro_usd(Usage(input_tokens=1, output_tokens=0)) == 2  # 1.25 rounds up, never down
    assert wonder.micro_usd(Usage(input_tokens=2_000, cached_tokens=1_000, output_tokens=1_000)) == 24_200
    with pytest.raises(ValueError, match="cached_tokens"):
        spark.micro_usd(Usage(input_tokens=1, cached_tokens=2, output_tokens=0))
    assert spark.worst_case_micro_usd(1_000, 1_000) == 5_500
    with pytest.raises(UnpricedModelError):
        price_for("sam-3.1")
    with pytest.raises(ModelRefusedError):
        price_for("muse-spark-1.3-contributor")
    usage = Usage(input_tokens=1_000, cached_tokens=400, output_tokens=500)
    assert stage_cost("intake", SPARK, usage, seconds=1.5) == StageCost(
        stage="intake",
        model=SPARK,
        input_tokens=1_000,
        cached_tokens=400,
        output_tokens=500,
        micro_usd=2_935,
        seconds=1.5,
    )


# ---------------------------------------------------------------- the shared model-call error


def test_every_client_error_is_the_contracts_model_call_error_with_its_reason() -> None:
    cases: list[tuple[ModelCallError, UncheckedReason]] = [
        (MetaClientError("x"), UncheckedReason.RUNTIME),
        (MissingKeyError("x"), UncheckedReason.RUNTIME),
        (SpendCapError("x"), UncheckedReason.BUDGET),
        (SpendNotArmedError("x"), UncheckedReason.BUDGET),
        (SpendFileError("x"), UncheckedReason.BUDGET),
        (MetaAPIError("x", status=500), UncheckedReason.RUNTIME),
        (MetaAPIError("x", status=504, reason=UncheckedReason.TIMEOUT), UncheckedReason.TIMEOUT),
        (MetaRetriesExhaustedError("x", attempts=6), UncheckedReason.RUNTIME),
        (MetaResponseError("x"), UncheckedReason.RUNTIME),
    ]
    for error, reason in cases:
        assert isinstance(error, ModelCallError)
        assert error.reason is reason


# ---------------------------------------------------------------- the kill switch and its spend file


def test_the_default_cap_is_the_150_dollar_meta_credit(fake_key: str, meta: FakeMeta, spend_path: Path) -> None:
    assert DEFAULT_SPEND_CAP_MICRO_USD == 150_000_000
    assert usd(DEFAULT_SPEND_CAP_MICRO_USD) == "$150.000000"
    assert SpendMeter(spend_path).cap_micro_usd == 150_000_000
    with MetaClient(api_key=fake_key, transport=httpx.MockTransport(meta.handler)) as client:
        # The meter a client gets when it is given none: the shared one, at the credit, and closed until armed.
        assert client.meter.cap_micro_usd == 150_000_000
        assert client.meter.limit_micro_usd == 0
        assert client.meter.arm(credit_confirmed=True).cap_micro_usd == 150_000_000
        assert client.meter.limit_micro_usd == 150_000_000


def test_an_unarmed_spend_file_refuses_every_call_and_sends_nothing(
    meta: FakeMeta, clock: FakeClock, fake_key: str, spend_path: Path
) -> None:
    # A default client meters against RECEIPTS_SPEND_FILE, which no operator has armed.
    meta.queue(completion())
    with MetaClient(
        api_key=fake_key, transport=httpx.MockTransport(meta.handler), clock=clock.monotonic, sleep=clock.sleep
    ) as client:
        for _ in range(2):
            with pytest.raises(ModelCallError) as caught:
                client.complete(spark_request())
            assert isinstance(caught.value, SpendNotArmedError)
            assert caught.value.reason is UncheckedReason.BUDGET
            assert ARM_COMMAND in str(caught.value)
    assert meta.requests == []
    assert not default_spend_path().exists()  # a refused call writes nothing

    # Metered calls in a file do not arm it: only an arm record does.
    hold_id = "a" * 32
    private_spend_file(spend_path, lambda inode: hold_line(hold_id) + "\n" + settle_line(hold_id) + "\n")
    meter = SpendMeter(spend_path)
    assert (meter.spent_micro_usd, meter.armed) == (10, None)
    with pytest.raises(SpendNotArmedError):
        meter.reserve(1, stage="intake", model=SPARK)


def test_arming_needs_the_credit_confirmed_and_a_cap_no_higher_than_the_credit(
    spend_path: Path, clock: FakeClock
) -> None:
    meter = SpendMeter(spend_path, now=clock.wall)
    with pytest.raises(ValueError, match="credit_confirmed=True"):
        meter.arm(credit_confirmed=False)
    for cap in (0, -1, DEFAULT_SPEND_CAP_MICRO_USD + 1):
        with pytest.raises(ValueError, match="armed cap"):
            meter.arm(credit_confirmed=True, cap_micro_usd=cap)
    assert not spend_path.exists()
    assert meter.armed is None

    record = meter.arm(credit_confirmed=True, cap_micro_usd=20_000_000)
    assert json.loads(spend_path.read_text()) == {
        "v": 2,
        "at": STAMP,
        "kind": "arm",
        "credit_confirmed": True,
        "cap_micro_usd": 20_000_000,
        "inode": spend_path.stat().st_ino,  # it arms this file and no copy of it
    }
    assert meter.armed == record
    assert meter.limit_micro_usd == 20_000_000


def test_the_cap_in_effect_is_the_lower_of_the_meters_and_the_armed_one(spend_path: Path, tmp_path: Path) -> None:
    meter = SpendMeter(spend_path, cap_micro_usd=2_000)
    meter.arm(credit_confirmed=True)  # armed at the credit
    assert meter.limit_micro_usd == 2_000
    meter.arm(credit_confirmed=True, cap_micro_usd=1_000)  # the latest arm record counts
    assert meter.limit_micro_usd == 1_000
    # An arm record written by hand past the credit cannot lift a meter past its own cap.
    hand = private_spend_file(tmp_path / "hand" / "spend.jsonl", lambda inode: arm_line(inode, 10**15) + "\n")
    assert SpendMeter(hand).limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD


def test_arming_above_the_credit_needs_a_cap_past_the_credit_and_at_most_the_ceiling(
    spend_path: Path, clock: FakeClock
) -> None:
    assert SPEND_CEILING_MICRO_USD == 425_000_000  # the founder's $425 cap for the final, 10/31
    meter = SpendMeter(spend_path, now=clock.wall)
    for cap in (1, DEFAULT_SPEND_CAP_MICRO_USD, SPEND_CEILING_MICRO_USD + 1):
        with pytest.raises(ValueError, match="above the credit"):
            meter.arm(credit_confirmed=True, cap_micro_usd=cap, above_credit=True)
    with pytest.raises(ValueError, match="armed cap is 1 to"):  # without above_credit the credit still bounds it
        meter.arm(credit_confirmed=True, cap_micro_usd=SPEND_CEILING_MICRO_USD)
    assert not spend_path.exists()

    record = meter.arm(credit_confirmed=True, cap_micro_usd=SPEND_CEILING_MICRO_USD, above_credit=True)
    assert json.loads(spend_path.read_text()) == {
        "v": 2,
        "at": STAMP,
        "kind": "arm",
        "credit_confirmed": True,
        "cap_micro_usd": 425_000_000,
        "inode": spend_path.stat().st_ino,
        "above_credit": True,
    }
    assert meter.armed == record
    assert record.above_credit


def test_an_above_credit_arm_lets_a_default_client_spend_past_the_credit_and_stops_it_at_the_ceiling(
    meta: FakeMeta, clock: FakeClock, fake_key: str
) -> None:
    # A client built with defaults meters on the shared meter, at the default cap, which an above-credit arm lifts.
    with MetaClient(
        api_key=fake_key, transport=httpx.MockTransport(meta.handler), clock=clock.monotonic, sleep=clock.sleep
    ) as client:
        meter = client.meter
        assert meter.cap_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD
        meter.arm(credit_confirmed=True, cap_micro_usd=SPEND_CEILING_MICRO_USD, above_credit=True)
        assert meter.limit_micro_usd == SPEND_CEILING_MICRO_USD
        past_credit = meter.reserve(200_000_000, stage="intake", model=SPARK)
        meter.settle(past_credit, 200_000_000, estimated=False)

        meta.queue(completion())
        client.complete(spark_request())  # $200 spent, past the $150 credit, and the call goes through
        assert len(meta.requests) == 1

        edge = SPEND_CEILING_MICRO_USD - meter.spent_micro_usd - worst_case(spark_request()) + 1
        to_edge = meter.reserve(edge, stage="intake", model=SPARK)
        meter.settle(to_edge, edge, estimated=False)
        with pytest.raises(SpendCapError, match=r"could pass the \$425\.000000 cap") as caught:
            client.complete(spark_request())
        assert caught.value.reason is UncheckedReason.BUDGET
    assert len(meta.requests) == 1  # refused before anything was sent


def test_a_default_arm_after_an_above_credit_one_stops_spend_at_the_credit_again(spend_path: Path) -> None:
    meter = SpendMeter(spend_path)  # the default cap, as shared_meter() builds it
    meter.arm(credit_confirmed=True, cap_micro_usd=SPEND_CEILING_MICRO_USD, above_credit=True)
    assert meter.limit_micro_usd == SPEND_CEILING_MICRO_USD
    meter.arm(credit_confirmed=True)  # the latest arm record counts, and a default one is at the credit
    assert meter.limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD
    spent = meter.reserve(DEFAULT_SPEND_CAP_MICRO_USD, stage="intake", model=SPARK)
    meter.settle(spent, DEFAULT_SPEND_CAP_MICRO_USD, estimated=False)
    with pytest.raises(SpendCapError, match=r"reached the \$150\.000000 cap"):
        meter.reserve(1, stage="intake", model=SPARK)
    # A default arm is written exactly as before: above_credit appears only when it is true.
    above, default = (json.loads(line) for line in spend_path.read_text().splitlines()[:2])
    assert above["above_credit"] is True
    assert "above_credit" not in default


def test_an_arm_record_without_above_credit_is_at_the_credit_and_no_cap_passes_the_ceiling(tmp_path: Path) -> None:
    # A record written before above_credit existed has no such field, and reads as an arm at the credit.
    old = private_spend_file(tmp_path / "old" / "spend.jsonl", lambda inode: arm_json(inode, SPEND_CEILING_MICRO_USD))
    meter = SpendMeter(old)
    assert meter.armed is not None
    assert meter.armed.above_credit is False
    assert meter.limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD
    # An above-credit record written by hand past the ceiling stops at the ceiling; a meter given a cap keeps it.
    hand = private_spend_file(
        tmp_path / "hand" / "spend.jsonl", lambda inode: arm_json(inode, 10**15, above_credit=True)
    )
    assert SpendMeter(hand).limit_micro_usd == SPEND_CEILING_MICRO_USD
    assert SpendMeter(hand, cap_micro_usd=2_000).limit_micro_usd == 2_000
    assert SpendMeter(hand, cap_micro_usd=DEFAULT_SPEND_CAP_MICRO_USD).limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD
    assert SpendMeter(hand, cap_micro_usd=10**15).limit_micro_usd == SPEND_CEILING_MICRO_USD  # given high, still capped
    flagless = private_spend_file(tmp_path / "flagless" / "spend.jsonl", lambda inode: arm_json(inode, 10**15))
    assert SpendMeter(flagless, cap_micro_usd=10**15).limit_micro_usd == SPEND_CEILING_MICRO_USD


def test_an_above_credit_arm_planted_in_the_current_directorys_spend_file_counts_at_the_credit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meta: FakeMeta, clock: FakeClock, fake_key: str
) -> None:
    # A checkout ships .receipts/spend.jsonl with an above-credit arm that names the file's own inode (the inode
    # guessed right). Someone runs receipts there with no RECEIPTS_SPEND_FILE: the arm counts at the credit.
    checkout = tmp_path / "checkout"
    planted = private_spend_file(
        checkout / ".receipts" / "spend.jsonl",
        lambda inode: arm_json(inode, SPEND_CEILING_MICRO_USD, above_credit=True),
    )
    monkeypatch.delenv(SPEND_FILE_ENV)
    monkeypatch.chdir(checkout)
    assert default_spend_path() == planted
    assert not named_spend_file(planted)
    meta.queue(completion())
    with MetaClient(
        api_key=fake_key, transport=httpx.MockTransport(meta.handler), clock=clock.monotonic, sleep=clock.sleep
    ) as client:
        meter = client.meter
        assert meter.armed is not None
        assert meter.armed.above_credit  # the planted arm arms this very file
        assert not meter.named
        assert meter.limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD
        assert repr(meter).endswith("cap_in_effect=150000000)")
        credit = meter.reserve(DEFAULT_SPEND_CAP_MICRO_USD, stage="intake", model=SPARK)
        meter.settle(credit, DEFAULT_SPEND_CAP_MICRO_USD, estimated=False)
        with pytest.raises(SpendCapError, match=r"reached the \$150\.000000 cap"):
            client.complete(spark_request())
        with pytest.raises(ValueError, match="needs a spend file named with RECEIPTS_SPEND_FILE"):
            meter.arm(credit_confirmed=True, cap_micro_usd=SPEND_CEILING_MICRO_USD, above_credit=True)
    assert meta.requests == []
    # A meter given its own high cap there still counts the planted arm at the credit at most.
    assert SpendMeter(planted, cap_micro_usd=10**15).limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD


def test_a_meters_repr_shows_the_cap_in_effect_as_last_read(spend_path: Path) -> None:
    meter = SpendMeter(spend_path)
    assert repr(meter) == f"SpendMeter(path={spend_path!r}, cap_micro_usd=150000000, cap_in_effect=0)"
    meter.arm(credit_confirmed=True, cap_micro_usd=SPEND_CEILING_MICRO_USD, above_credit=True)
    assert repr(meter) == f"SpendMeter(path={spend_path!r}, cap_micro_usd=150000000, cap_in_effect=425000000)"


def test_a_clients_repr_shows_the_cap_in_effect_as_last_read(
    make_client: Callable[..., MetaClient], spend_path: Path
) -> None:
    meter = SpendMeter(spend_path)
    client = make_client(meter=meter)
    meter.arm(credit_confirmed=True, cap_micro_usd=SPEND_CEILING_MICRO_USD, above_credit=True)
    assert meter.cap_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD
    assert "cap_in_effect=425000000)" in repr(client)
    meter.arm(credit_confirmed=True)
    assert "cap_in_effect=150000000)" in repr(client)
    last_repr = repr(client)
    spend_path.unlink()
    assert repr(client) == last_repr  # a repr uses the last read and never reads a missing or broken file


@pytest.mark.parametrize("cap", [25_000, None])
def test_a_clients_repr_accepts_a_meter_without_base_initialization(
    make_client: Callable[..., MetaClient], cap: int | None
) -> None:
    class FreshAttackMeter(SpendMeter):
        def __init__(self) -> None:
            if cap is not None:
                self.cap_micro_usd = cap

    meter = FreshAttackMeter()
    client = make_client(meter=meter)
    assert "FreshAttackMeter" in repr(meter)
    assert "FreshAttackMeter" in repr(client)
    if cap is not None:
        assert f"cap_micro_usd={cap}" in repr(client)


def test_the_kill_switch_refuses_every_call_once_spend_reaches_the_cap(
    make_client: Callable[..., MetaClient], meta: FakeMeta, spend_path: Path
) -> None:
    # The cap holds exactly one call; the server reports more input tokens than the request had bytes, so the call
    # is metered in full past its hold (never undercounted) and spend reaches the cap.
    cap = worst_case(spark_request())
    client = make_client(meter=armed_meter(spend_path, cap=cap))
    meta.queue(completion(prompt=cap, output=0))
    client.complete(spark_request())
    assert client.meter.spent_micro_usd == LIST_PRICES[SPARK].micro_usd(Usage(input_tokens=cap, output_tokens=0))
    assert client.meter.spent_micro_usd >= cap
    for _ in range(2):
        with pytest.raises(SpendCapError, match="reached") as caught:
            client.complete(spark_request())
        assert caught.value.reason is UncheckedReason.BUDGET
    assert len(meta.requests) == 1


def test_the_kill_switch_refuses_a_call_whose_worst_case_could_pass_the_cap(
    make_client: Callable[..., MetaClient], meta: FakeMeta, tmp_path: Path
) -> None:
    request = spark_request(max_tokens=10_000)
    with pytest.raises(SpendCapError, match="could pass"):
        make_client(meter=armed_meter(tmp_path / "a.jsonl", cap=worst_case(request) - 1)).complete(request)
    assert meta.requests == []

    meta.queue(completion())
    make_client(meter=armed_meter(tmp_path / "b.jsonl", cap=worst_case(request))).complete(request)
    assert len(meta.requests) == 1


def test_holds_in_flight_in_any_process_count_against_the_cap(spend_path: Path) -> None:
    here = armed_meter(spend_path, cap=1_000)
    there = SpendMeter(spend_path, cap_micro_usd=1_000)  # another process's meter on the same file
    first = here.reserve(600, stage="intake", model=SPARK)
    for meter in (there, here):
        with pytest.raises(SpendCapError, match="held in flight"):
            meter.reserve(600, stage="intake", model=SPARK)
    assert there.held_micro_usd == 600
    here.settle(first, 100, estimated=False)  # it cost less than its worst case
    assert (there.spent_micro_usd, there.held_micro_usd) == (100, 0)
    assert there.reserve(600, stage="intake", model=SPARK).micro_usd == 600  # 100 spent plus 600 held fits


def test_a_hold_rechecks_the_cap_under_the_file_lock(spend_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    here = armed_meter(spend_path, cap=1_000)
    there = SpendMeter(spend_path, cap_micro_usd=1_000)
    read = there._refresh

    def read_then_another_process_holds() -> None:  # another process holds 600 just after this meter read the file
        read()
        monkeypatch.setattr(there, "_refresh", read)
        here.reserve(600, stage="intake", model=SPARK)

    monkeypatch.setattr(there, "_refresh", read_then_another_process_holds)
    with pytest.raises(SpendCapError, match="held in flight"):
        there.reserve(600, stage="intake", model=SPARK)
    assert there.held_micro_usd == 600  # the other process's hold only


def test_a_hold_left_open_by_a_process_that_stopped_stays_counted(spend_path: Path) -> None:
    stopped = armed_meter(spend_path, cap=1_000)
    stopped.reserve(700, stage="intake", model=SPARK)  # its process dies before the call settles
    later = SpendMeter(spend_path, cap_micro_usd=1_000)
    assert (later.spent_micro_usd, later.held_micro_usd) == (0, 700)
    with pytest.raises(SpendCapError, match="held in flight"):
        later.reserve(400, stage="intake", model=SPARK)
    assert later.reserve(300, stage="intake", model=SPARK).micro_usd == 300


def test_concurrent_calls_never_reserve_past_the_cap(spend_path: Path) -> None:
    meter = armed_meter(spend_path, cap=1_000)
    start = threading.Barrier(50)
    admitted: list[int] = []

    def call() -> None:
        start.wait()
        try:
            hold = meter.reserve(30, stage="intake", model=SPARK)
        except SpendCapError:
            return
        admitted.append(hold.micro_usd)
        meter.settle(hold, 30, estimated=False)

    threads = [threading.Thread(target=call) for _ in range(50)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert len(admitted) == 33  # 33 x 30 = 990; a 34th would pass 1,000
    assert (meter.spent_micro_usd, meter.held_micro_usd) == (990, 0)


def test_threads_of_one_process_read_the_spend_file_one_at_a_time(
    spend_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # The file lock lets two readers in at once (a shared lock), and two threads applying the same new records would
    # count them twice and refuse every later call. The meter's own lock keeps a second thread out: the first read
    # waits inside for a second one to join it, at most 0.3 s, and none can.
    meter = armed_meter(spend_path, cap=1_000_000)
    inside = threading.Barrier(2, timeout=0.3)
    together: list[str] = []
    read_new = meter._read_new

    def meeting_read(handle: Any) -> None:
        try:
            inside.wait()
        except threading.BrokenBarrierError:
            pass  # alone in the read, as it should be; the broken barrier lets every later read straight through
        else:
            together.append(threading.current_thread().name)
        read_new(handle)

    monkeypatch.setattr(meter, "_read_new", meeting_read)
    start = threading.Barrier(8)

    def call() -> None:
        start.wait()
        hold = meter.reserve(30, stage="intake", model=SPARK)
        meter.settle(hold, 30, estimated=False)

    threads = [threading.Thread(target=call) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert together == []
    assert (meter.spent_micro_usd, meter.held_micro_usd) == (8 * 30, 0)


CHILD = """
import sys
from pathlib import Path

from receipts.core.meta_client import SpendCapError, SpendMeter

meter = SpendMeter(Path(sys.argv[1]), cap_micro_usd=int(sys.argv[2]))
admitted = 0
while True:
    try:
        hold = meter.reserve(10, stage="child", model="muse-spark-1.3")
    except SpendCapError:
        break
    meter.settle(hold, 10, estimated=False)
    admitted += 1
print(admitted)
"""


def test_processes_sharing_a_spend_file_share_its_cap_exactly(spend_path: Path) -> None:
    armed_meter(spend_path)
    children = [
        subprocess.Popen(
            [sys.executable, "-c", CHILD, str(spend_path), "1000"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(4)
    ]
    results = [child.communicate(timeout=120) for child in children]
    assert [child.returncode for child in children] == [0, 0, 0, 0], [err for _, err in results]
    # Four processes calling 10 at a time against a 1,000 cap: exactly 100 calls get through between them.
    assert sum(int(out) for out, _ in results) == 100
    meter = SpendMeter(spend_path)
    assert (meter.spent_micro_usd, meter.held_micro_usd) == (1_000, 0)
    assert len(spend_path.read_text().splitlines()) == 1 + 2 * 100  # the arm record, then a hold and a settle per call


def test_default_clients_in_one_process_share_one_meter_and_its_cap(
    meta: FakeMeta, clock: FakeClock, fake_key: str
) -> None:
    other = FakeMeta(clock)
    options: dict[str, Any] = {"api_key": fake_key, "clock": clock.monotonic, "sleep": clock.sleep}
    with (
        MetaClient(transport=httpx.MockTransport(meta.handler), **options) as first,
        MetaClient(transport=httpx.MockTransport(other.handler), **options) as second,
    ):
        assert first.meter is second.meter is shared_meter()
        assert first.governor is second.governor is shared_governor()
        cap = worst_case(spark_request()) + 2_000  # one call's hold fits, with room to spare
        first.meter.arm(credit_confirmed=True, cap_micro_usd=cap)

        held = first.meter.reserve(2_001, stage="intake", model=SPARK)  # a call of the first client's, in flight
        other.queue(completion())
        with pytest.raises(SpendCapError, match="held in flight"):
            second.complete(spark_request())
        first.meter.settle(held, 0, estimated=False)

        meta.queue(completion(prompt=cap, output=0))  # metered in full, past the cap
        first.complete(spark_request())
        with pytest.raises(SpendCapError, match="reached"):
            second.complete(spark_request())
    assert other.requests == []


def test_the_spend_file_is_receipts_spend_file_else_under_the_current_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, fake_key: str, meta: FakeMeta
) -> None:
    assert SPEND_FILE_ENV == "RECEIPTS_SPEND_FILE"
    chosen = tmp_path / "runs" / "pilot" / "spend.jsonl"
    monkeypatch.setenv(SPEND_FILE_ENV, str(chosen))
    assert default_spend_path() == chosen.resolve()
    with MetaClient(api_key=fake_key, transport=httpx.MockTransport(meta.handler)) as client:
        assert client.meter.path == chosen.resolve()

    monkeypatch.delenv(SPEND_FILE_ENV)
    monkeypatch.chdir(tmp_path)
    here = (tmp_path / ".receipts" / "spend.jsonl").resolve()
    assert default_spend_path() == here
    with MetaClient(api_key=fake_key, transport=httpx.MockTransport(meta.handler)) as client:
        assert client.meter.path == here
    assert not here.parent.exists()  # reading an unarmed spend file creates nothing
    monkeypatch.setenv(SPEND_FILE_ENV, "  ")  # blank is unset
    assert default_spend_path() == here


@pytest.mark.parametrize(
    ("alias_kind", "default_exists"),
    [
        ("detour", False),
        ("detour", True),
        ("directory_symlink", False),
        ("directory_symlink", True),
        ("hardlink", True),
    ],
)
def test_default_spend_file_aliases_are_not_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, alias_kind: str, default_exists: bool
) -> None:
    checkout = tmp_path / "checkout"
    checkout.mkdir()
    default = checkout / ".receipts" / "spend.jsonl"
    if default_exists:
        default.parent.mkdir()
        default.touch()
    monkeypatch.chdir(checkout)
    if alias_kind == "detour":
        Path("detour").mkdir()
        alias = Path("detour") / ".." / ".receipts" / "spend.jsonl"
    elif alias_kind == "directory_symlink":
        link = tmp_path / "linked-checkout"
        link.symlink_to(checkout, target_is_directory=True)
        alias = link / ".receipts" / "spend.jsonl"
    else:
        alias = checkout / "spend-hardlink.jsonl"
        alias.hardlink_to(default)
    monkeypatch.setenv(SPEND_FILE_ENV, str(alias))
    assert not named_spend_file(default_spend_path())
    assert not named_spend_file(alias)


@pytest.mark.parametrize("default_state", ["missing", "file", "not_directory"])
@pytest.mark.parametrize("named_exists", [False, True])
def test_a_separate_spend_file_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, default_state: str, named_exists: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    if default_state == "file":
        default = tmp_path / ".receipts" / "spend.jsonl"
        default.parent.mkdir()
        default.touch()
    elif default_state == "not_directory":
        (tmp_path / ".receipts").touch()
    chosen = tmp_path / "operator" / "spend.jsonl"
    if named_exists:
        chosen.parent.mkdir()
        chosen.touch()
    monkeypatch.setenv(SPEND_FILE_ENV, str(chosen))
    assert named_spend_file(default_spend_path())


def test_a_separate_spend_file_is_named_when_the_default_is_a_symlink_loop(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    default = tmp_path / ".receipts" / "spend.jsonl"
    default.parent.mkdir()
    default.symlink_to(default.name)
    chosen = private_spend_file(tmp_path / "operator" / "spend.jsonl", lambda inode: "")
    assert named_spend_file(chosen)
    meter = SpendMeter(chosen)
    assert meter.named
    assert meter.arm(credit_confirmed=True).cap_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD


def test_a_separate_spend_file_under_an_unreadable_directory_is_named(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    chosen = private_spend_file(tmp_path / "operator" / "spend.jsonl", lambda inode: "")
    chosen.parent.chmod(0o000)
    try:
        assert named_spend_file(chosen)
    finally:
        chosen.parent.chmod(0o700)


@pytest.mark.parametrize("absolute", [False, True])
def test_the_exact_default_spelling_is_not_named_without_stat(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, absolute: bool
) -> None:
    monkeypatch.chdir(tmp_path)
    path = Path(".receipts/spend.jsonl")
    if absolute:
        path = path.absolute()

    def no_stat(self: Path, *, follow_symlinks: bool = True) -> os.stat_result:
        pytest.fail("the exact default path must be recognized before any stat")

    with monkeypatch.context() as path_checks:
        path_checks.setattr(Path, "stat", no_stat)
        assert not named_spend_file(path)


@pytest.mark.parametrize("identity_error", [OSError, RuntimeError])
@pytest.mark.parametrize("resolve_error", [OSError, RuntimeError])
def test_a_separate_spend_file_is_named_when_identity_and_resolution_fail(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    identity_error: type[Exception],
    resolve_error: type[Exception],
) -> None:
    monkeypatch.chdir(tmp_path)
    chosen = tmp_path / "operator" / "spend.jsonl"

    def cannot_identify(self: Path, other: str | Path) -> bool:
        raise identity_error("cannot compare file identity")

    def cannot_resolve(self: Path, strict: bool = False) -> Path:
        raise resolve_error("cannot resolve the spend path")

    with monkeypatch.context() as path_checks:
        path_checks.setattr(Path, "samefile", cannot_identify)
        path_checks.setattr(Path, "resolve", cannot_resolve)
        assert named_spend_file(chosen)


def test_a_spend_file_carries_the_cap_across_runs(
    tmp_path: Path, make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    path = tmp_path / "runs" / "spend.jsonl"
    request = spark_request()
    meta.queue(completion(prompt=1_000, output=500))
    make_client(meter=armed_meter(path, now=clock.wall)).complete(request)
    arm, hold, settle = (json.loads(line) for line in path.read_text().splitlines())
    assert arm == {
        "v": 2,
        "at": STAMP,
        "kind": "arm",
        "credit_confirmed": True,
        "cap_micro_usd": 150_000_000,
        "inode": path.stat().st_ino,
    }
    assert hold == {
        "v": 2,
        "at": STAMP,
        "kind": "hold",
        "id": hold["id"],
        "stage": "intake",
        "model": SPARK,
        "micro_usd": worst_case(request),
    }
    assert settle == {
        "v": 2,
        "at": STAMP,
        "kind": "settle",
        "hold": hold["id"],
        "stage": "intake",
        "model": SPARK,
        "micro_usd": 3_375,
        "estimated": False,
    }

    later = SpendMeter(path, cap_micro_usd=3_000)  # a later run reading the same file
    assert (later.spent_micro_usd, later.held_micro_usd) == (3_375, 0)
    with pytest.raises(SpendCapError, match="reached"):
        make_client(meter=later).complete(spark_request())
    assert len(meta.requests) == 1


# Each builds a spend file's text from the file's own inode, so only the content is broken, never the file's modes.
BROKEN_SPEND_FILES: dict[str, Callable[[int], str]] = {
    "not-json": lambda inode: "not json\n",
    "partial-line": lambda inode: arm_line(inode),
    "negative": lambda inode: (
        "\n".join(
            [arm_line(inode), hold_line("b" * 32), json.dumps({**json.loads(settle_line("b" * 32)), "micro_usd": -5})]
        )
        + "\n"
    ),
    "unversioned": lambda inode: (
        json.dumps({"at": STAMP, "kind": "arm", "credit_confirmed": True, "cap_micro_usd": 5, "inode": inode}) + "\n"
    ),
    "version-1": lambda inode: (
        json.dumps({"v": 1, "at": STAMP, "kind": "arm", "credit_confirmed": True, "cap_micro_usd": 5, "inode": inode})
        + "\n"
    ),
    "unconfirmed": lambda inode: (
        json.dumps({"v": 2, "at": STAMP, "kind": "arm", "credit_confirmed": False, "cap_micro_usd": 5, "inode": inode})
        + "\n"
    ),
    "unknown-kind": lambda inode: json.dumps({"v": 2, "at": STAMP, "kind": "refund", "micro_usd": 5}) + "\n",
    "settle-never-held": lambda inode: arm_line(inode) + "\n" + settle_line("c" * 32) + "\n",
    "settled-twice": lambda inode: (
        "\n".join([arm_line(inode), hold_line("d" * 32), settle_line("d" * 32), settle_line("d" * 32)]) + "\n"
    ),
    "held-twice": lambda inode: "\n".join([arm_line(inode), hold_line("e" * 32), hold_line("e" * 32)]) + "\n",
}


@pytest.mark.parametrize("build", list(BROKEN_SPEND_FILES.values()), ids=list(BROKEN_SPEND_FILES))
def test_a_spend_file_that_cannot_be_read_refuses_every_call(tmp_path: Path, build: Callable[[int], str]) -> None:
    path = private_spend_file(tmp_path / "spend.jsonl", build)
    with pytest.raises(SpendFileError):
        SpendMeter(path)


def test_a_spend_file_that_shrinks_or_disappears_refuses_every_call(tmp_path: Path) -> None:
    for name, damage, match in (("shrinks.jsonl", "truncate", "shrank"), ("goes.jsonl", "delete", "disappeared")):
        path = tmp_path / name
        meter = armed_meter(path)
        meter.settle(meter.reserve(10, stage="intake", model=SPARK), 10, estimated=False)
        if damage == "truncate":
            path.write_bytes(b"")
        else:
            path.unlink()
        with pytest.raises(SpendFileError, match=match):
            meter.reserve(1, stage="intake", model=SPARK)


def test_a_spend_file_that_cannot_be_written_refuses_every_call(
    tmp_path: Path, make_client: Callable[..., MetaClient], meta: FakeMeta, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "spend.jsonl"
    meter = armed_meter(path)
    hold = meter.reserve(10, stage="intake", model=SPARK)
    path.chmod(0o444)
    meter.settle(hold, 10, estimated=False)  # it never raises
    path.chmod(0o600)  # even once the file is writable again, the meter stays shut: its settle never reached it
    with pytest.raises(SpendFileError, match="could not be written"):
        meter.reserve(1, stage="intake", model=SPARK)

    # A hold that cannot be written (the private file is there, but the write fails) refuses the call as a budget
    # failure before any request.
    fresh = tmp_path / "fresh" / "spend.jsonl"
    armed = armed_meter(fresh)

    def unwritable(create: bool) -> int:
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(armed, "_open_for_append", unwritable)
    with pytest.raises(SpendFileError, match="could not be written") as caught:
        make_client(meter=armed).complete(spark_request())
    assert caught.value.reason is UncheckedReason.BUDGET
    assert meta.requests == []


# ---------------------------------------------------------------- the tokens-per-minute governor


def test_the_governor_spaces_calls_to_the_tokens_per_minute_budget(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    request = spark_request(max_tokens=64)
    hold = len(encode_request(request)) + 64
    # After one call of 1,000 tokens there is no room for a second call's hold until the first leaves the window.
    governor = TokenGovernor({SPARK: 1_000 + hold - 1}, clock=clock.monotonic, sleep=clock.sleep)
    client = make_client(governor=governor)
    meta.queue(completion(prompt=900, output=100), completion(prompt=900, output=100))
    client.complete(request)
    client.complete(request)
    assert clock.sleeps == [60.0]
    assert meta.sent_at == [0.0, 60.0]


def test_the_governor_holds_tokens_in_flight_and_clamps_an_oversized_call(clock: FakeClock) -> None:
    pending: list[Any] = []

    def sleep(seconds: float) -> None:  # the first call finishes while the second waits
        clock.sleep(seconds)
        for slot in pending:
            governor.settle(slot, 0)
        pending.clear()

    governor = TokenGovernor({SPARK: 100}, clock=clock.monotonic, sleep=sleep, poll_s=5.0)
    pending.append(governor.acquire(SPARK, 80))
    governor.acquire(SPARK, 30)
    assert clock.sleeps == [5.0]  # tokens in flight do not expire, so it polled

    full = TokenGovernor({SPARK: 100, "muse-wonder-1.0": 100}, clock=clock.monotonic, sleep=clock.sleep)
    full.settle(full.acquire(SPARK, 100), 100)
    full.acquire("muse-wonder-1.0", 100)  # each model has its own budget, so this does not wait
    assert clock.sleeps == [5.0]
    oversized = full.acquire(SPARK, 10_000)
    assert oversized.tokens == 100
    assert clock.sleeps == [5.0, 60.0]

    with pytest.raises(ModelRefusedError, match="no tokens-per-minute limit"):
        TokenGovernor({SPARK: 100}).acquire("muse-wonder-1.0", 1)


def test_the_default_budgets_are_the_documented_limits_shared_by_every_default_client() -> None:
    documented = dict.fromkeys(["muse-spark-1.1", "muse-spark-1.2", "muse-spark-1.3", "muse-wonder-1.0"], 4_000_000)
    assert dict(DEFAULT_TPM) == documented
    assert dict(TokenGovernor().limits) == documented
    assert shared_governor() is shared_governor()
    assert dict(shared_governor().limits) == documented


# ---------------------------------------------------------------- the key


def test_the_key_comes_from_falden_meta_api_key_only(
    monkeypatch: pytest.MonkeyPatch, meta: FakeMeta, clock: FakeClock, fake_key: str, spend_path: Path
) -> None:
    monkeypatch.setenv("MODEL_API_KEY", fake_key)  # the variable name in Meta's own examples is not read
    with pytest.raises(MissingKeyError, match=API_KEY_ENV) as caught:
        MetaClient(transport=httpx.MockTransport(meta.handler))
    assert fake_key not in str(caught.value)

    monkeypatch.setenv(API_KEY_ENV, fake_key)
    meta.queue(completion())
    with MetaClient(
        meter=armed_meter(spend_path),
        transport=httpx.MockTransport(meta.handler),
        clock=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        client.complete(spark_request())
    assert meta.requests[0].headers["authorization"] == f"Bearer {fake_key}"


def test_a_constructor_key_wins_over_the_environment(
    monkeypatch: pytest.MonkeyPatch, meta: FakeMeta, clock: FakeClock, fake_key: str, spend_path: Path
) -> None:
    other = f"LLM_{secrets.randbelow(10**15):015d}_{secrets.token_urlsafe(24)}"
    monkeypatch.setenv(API_KEY_ENV, other)
    meta.queue(completion())
    with MetaClient(
        api_key=fake_key,
        meter=armed_meter(spend_path),
        transport=httpx.MockTransport(meta.handler),
        clock=clock.monotonic,
        sleep=clock.sleep,
    ) as client:
        client.complete(spark_request())
    assert meta.requests[0].headers["authorization"] == f"Bearer {fake_key}"


def exception_texts(exc: BaseException) -> list[str]:
    """Everything an exception can show: its text, repr, args and attributes, down its cause and context chain, and
    its formatted traceback."""
    texts = ["".join(traceback.format_exception(exc))]
    seen: set[int] = set()
    current: BaseException | None = exc
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        texts += [str(current), repr(current), repr(current.args), repr(vars(current))]
        current = current.__cause__ or current.__context__
    return texts


def leaks(key: str, texts: Iterable[str]) -> list[str]:
    """Every text that shows the key, or its secret part on its own."""
    secret = key.split("_", 2)[-1]
    return [text for text in texts if key in text or secret in text]


def test_the_key_never_reaches_a_log_an_exception_a_repr_or_a_response(
    make_client: Callable[..., MetaClient], meta: FakeMeta, fake_key: str, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.DEBUG)  # every logger, httpx's included
    client = make_client(max_retries=1)
    secret = fake_key.split("_", 2)[-1]
    responses: list[ModelResponse] = []

    meta.queue(completion("fine"))
    responses.append(client.complete(spark_request()))

    # The server echoes the key, and another key-shaped token, in an error message.
    stray = f"LLM_{secrets.randbelow(10**15):015d}_{secrets.token_urlsafe(12)}_{secrets.token_urlsafe(12)}"
    meta.queue(api_error(401, "authentication_error", "invalid_api_key", f"Unauthorized: {fake_key}; {stray} too"))
    with pytest.raises(MetaAPIError) as refused:
        client.complete(spark_request())

    # A transport error carries the Authorization header in its text.
    def echo_in_error(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"proxy refused {request.headers['authorization']}", request=request)

    meta.queue(echo_in_error, echo_in_error)
    with pytest.raises(MetaRetriesExhaustedError) as exhausted:
        client.complete(spark_request())

    # A 200 echoes the Authorization header, and the key's secret part on its own, into every text field it has: the
    # request id, the answer and the rate-limit headers.
    def echo_in_200(request: httpx.Request) -> httpx.Response:
        header = request.headers["authorization"]
        return completion(
            f"{header} and {secret}",
            request_id=f"req-{secret}",
            headers={"x-ratelimit-remaining-tokens": header, "x-ratelimit-remaining-requests": secret},
        )

    meta.queue(echo_in_200)
    responses.append(client.complete(spark_request()))

    # Into the answering model, which names no priced model then, so the call fails with the model in its text.
    meta.queue(lambda request: completion("x", model=f"{SPARK} {request.headers['authorization']} {secret}"))
    with pytest.raises(MetaResponseError) as echoed_model:
        client.complete(spark_request())

    # And into finish_reason, which makes the response malformed.
    meta.queue(lambda request: completion("x", finish=request.headers["authorization"]))
    with pytest.raises(MetaResponseError) as malformed_200:
        client.complete(spark_request())
    with pytest.raises(MissingKeyError) as malformed_key:
        MetaClient(api_key=f"{fake_key}\n{fake_key}")

    # Every surface, gathered after every call.
    surfaces = [repr(client), str(client), repr(vars(client)), repr(client.meter), repr(client.records())]
    surfaces += [repr(client.stage_costs()), client.meter.path.read_text()]
    surfaces += [repr(response) for response in responses] + [response.model_dump_json() for response in responses]
    for caught in (refused, exhausted, echoed_model, malformed_200, malformed_key):
        surfaces += exception_texts(caught.value)
        assert caught.value.__cause__ is None
        assert caught.value.__context__ is None  # no chained exception holds the request and its header
    surfaces += [record.getMessage() for record in caplog.records]
    surfaces.append(caplog.text)

    assert meta.requests[0].headers["authorization"] == f"Bearer {fake_key}"  # it went where it belongs
    assert leaks(fake_key, surfaces) == []
    assert leaks(stray, surfaces) == []
    assert "[redacted]" in str(refused.value)
    assert "[redacted]" in str(exhausted.value)
    echoed = responses[1]
    assert (echoed.model, echoed.request_id) == (SPARK, "req-[redacted]")
    assert echoed.text == "Bearer [redacted] and [redacted]"
    assert f"'{SPARK} Bearer [redacted] [redacted]'" in str(echoed_model.value)
    (line,) = [m for m in (r.getMessage() for r in caplog.records) if "request_id='req-[redacted]'" in m]
    assert "remaining_tokens=None remaining_requests=None" in line


def test_the_leak_check_catches_a_planted_leak(fake_key: str, caplog: pytest.LogCaptureFixture) -> None:
    """The check above only means something if it fails on a leak, so a leak is planted in each kind of surface."""
    with caplog.at_level(logging.INFO):
        logging.getLogger("receipts.core.meta_client").info("header Bearer %s", fake_key)
    assert leaks(fake_key, [caplog.text])
    assert leaks(fake_key, exception_texts(RuntimeError(f"failed with {fake_key}")))
    try:
        try:
            raise ValueError(fake_key)
        except ValueError:
            raise RuntimeError("wrapped") from None
    except RuntimeError as chained:
        assert leaks(fake_key, exception_texts(chained))  # found down the context chain
    secret_only = fake_key.split("_", 2)[-1]
    assert leaks(fake_key, [f"client(secret={secret_only!r})"])
    assert leaks(fake_key, ["nothing here"]) == []


# ---------------------------------------------------------------- the answering model's price (round 2, blocker 1)


def last_settle(path: Path) -> dict[str, Any]:
    settles = [json.loads(line) for line in path.read_text().splitlines() if '"kind":"settle"' in line]
    return settles[-1]


def test_a_200_is_metered_at_the_higher_of_the_requested_and_the_answering_model_s_price(
    make_client: Callable[..., MetaClient], meta: FakeMeta, spend_path: Path
) -> None:
    # A Spark request answered by Wonder: 1,000 input and 1,000 output tokens at Wonder's list price cost 24,000
    # micro-dollars, where Spark's would give 5,500. The cost names the model that answered.
    meta.queue(completion(model="muse-wonder-1.0", prompt=1_000, output=1_000))
    client = make_client()
    response = client.complete(spark_request())
    assert response.model == "muse-wonder-1.0"
    assert response.cost is not None and (response.cost.micro_usd, response.cost.model) == (24_000, "muse-wonder-1.0")
    assert client.meter.spent_micro_usd == 24_000
    assert (last_settle(spend_path)["micro_usd"], last_settle(spend_path)["estimated"]) == (24_000, False)

    # A Wonder request answered by Spark is still metered at Wonder's price: Meta may bill the model asked for, and
    # the meter never undercounts.
    meta.queue(completion(model=SPARK, prompt=1_000, output=1_000))
    cheaper = client.complete(spark_request(model="muse-wonder-1.0"))
    assert cheaper.cost is not None and (cheaper.cost.micro_usd, cheaper.cost.model) == (24_000, SPARK)
    assert client.meter.spent_micro_usd == 48_000
    assert {(c.model, c.micro_usd) for c in client.stage_costs()} == {("muse-wonder-1.0", 24_000), (SPARK, 24_000)}
    assert [(r.model, r.answered) for r in client.records()] == [(SPARK, "muse-wonder-1.0"), ("muse-wonder-1.0", SPARK)]


NO_PRICE_ANSWERS = {
    "allowed-unpriced": "sam-3.1",
    "glimmer": GLIMMER_MODEL,
    "off-the-allowlist": "gpt-x",
    "contributor": "muse-spark-1.3-contributor",
    "contributor-mixed-case": "MUSE-Spark-1.3-Contributor",
    "no-model": None,
    "blank-model": " ",
}


@pytest.mark.parametrize("answered", list(NO_PRICE_ANSWERS.values()), ids=list(NO_PRICE_ANSWERS))
def test_an_answer_from_a_model_with_no_list_price_is_metered_at_the_highest_price_and_fails(
    make_client: Callable[..., MetaClient], meta: FakeMeta, spend_path: Path, answered: str | None
) -> None:
    body = json.loads(completion(prompt=1_000, output=1_000).content)
    if answered is None:
        del body["model"]
    else:
        body["model"] = answered
    meta.queue(httpx.Response(200, json=body))
    client = make_client()
    with pytest.raises(MetaResponseError, match="metered at the highest list price") as caught:
        client.complete(spark_request())

    highest = HIGHEST_PRICE.micro_usd(Usage(input_tokens=1_000, output_tokens=1_000))
    assert highest == 24_000  # Wonder's rates, the highest on file for every kind of token
    assert caught.value.reason is UncheckedReason.RUNTIME
    assert caught.value.cost is not None and caught.value.cost.micro_usd == highest
    assert client.meter.spent_micro_usd == highest
    assert (last_settle(spend_path)["micro_usd"], last_settle(spend_path)["estimated"]) == (highest, True)
    assert len(meta.requests) == 1  # a used answer is never retried
    if answered and "contributor" in answered.lower():
        assert "-contributor model, never used" in str(caught.value)


def test_the_highest_price_is_at_least_every_list_price() -> None:
    for price in LIST_PRICES.values():
        assert HIGHEST_PRICE.input >= price.input
        assert HIGHEST_PRICE.cached_input >= price.cached_input
        assert HIGHEST_PRICE.output >= price.output
    assert LIST_PRICES["muse-wonder-1.0"] == HIGHEST_PRICE


def test_the_glimmer_model_is_never_sent_to_meta(make_client: Callable[..., MetaClient], meta: FakeMeta) -> None:
    assert check_model(GLIMMER_MODEL) == GLIMMER_MODEL  # on the allowlist, for Falden's own vLLM only
    with pytest.raises(ModelRefusedError, match="Falden's own vLLM"):
        make_client().complete(spark_request(model=GLIMMER_MODEL))
    assert meta.requests == []


# ---------------------------------------------------------------- a structured retry that raises (round 2, blocker 2)


def test_a_structured_retry_that_raises_carries_the_first_calls_cost(
    make_client: Callable[..., MetaClient], meta: FakeMeta, clock: FakeClock
) -> None:
    # The first answer fails validation (1,000 input tokens: 1,250 micro-dollars); the retry times out every time,
    # each read timeout metered at the worst case.
    request = spark_request()
    meta.queue(completion('{"bug": "maybe"}', prompt=1_000, output=0), *[read_timeout] * (MAX_RETRIES + 1))
    client = make_client()
    with pytest.raises(MetaRetriesExhaustedError) as caught:
        complete_structured(client, request, Verdict)
    retry_worst = HIGHEST_PRICE.worst_case_micro_usd(len(meta.requests[1].content), request.max_tokens)
    cost = caught.value.cost
    assert cost is not None
    assert cost.micro_usd == 1_250 + (MAX_RETRIES + 1) * retry_worst == client.meter.spent_micro_usd
    assert (cost.stage, cost.model, cost.input_tokens) == ("intake", SPARK, 1_000)
    assert cost.seconds == pytest.approx(sum(clock.sleeps))


class Scripted:
    """A ModelClient that answers, then raises, as scripted; it prices each answer at 7 micro-dollars."""

    def __init__(self, *steps: str | ModelCallError) -> None:
        self.steps = list(steps)

    def complete(self, request: ModelRequest) -> ModelResponse:
        step = self.steps.pop(0)
        if isinstance(step, ModelCallError):
            raise step
        cost = StageCost(stage=request.stage, model=request.model, input_tokens=10, micro_usd=7, seconds=0.5)
        return ModelResponse(
            model=request.model,
            text=step,
            finish_reason="stop",
            usage=Usage(input_tokens=10, output_tokens=0),
            cost=cost,
        )


@pytest.mark.parametrize("own_cost", [None, 5], ids=["error-without-cost", "error-with-cost"])
def test_a_structured_retry_error_from_any_client_keeps_every_attempts_cost(own_cost: int | None) -> None:
    error = ModelCallError(
        "budget",
        UncheckedReason.BUDGET,
        cost=None if own_cost is None else StageCost(stage="intake", model=SPARK, micro_usd=own_cost, seconds=0.25),
    )
    with pytest.raises(ModelCallError) as caught:
        complete_structured(Scripted("not json", error), spark_request(), Verdict)
    assert caught.value is error  # the same error, with the first attempt's cost added to its own
    cost = caught.value.cost
    assert cost is not None
    assert cost.micro_usd == 7 + (own_cost or 0)
    assert cost.input_tokens == 10
    assert cost.seconds == pytest.approx(0.5 + (0.25 if own_cost else 0))

    # The first attempt raising has no earlier cost to add: its own is kept as it is.
    first = ModelCallError("down", UncheckedReason.RUNTIME, cost=StageCost(stage="intake", micro_usd=3))
    with pytest.raises(ModelCallError) as again:
        complete_structured(Scripted(first), spark_request(), Verdict)
    assert again.value.cost == StageCost(stage="intake", micro_usd=3)


def test_an_empty_answer_is_truncated_and_invalidates_the_item_without_a_retry(
    make_client: Callable[..., MetaClient], meta: FakeMeta
) -> None:
    meta.queue(completion(""))  # finish_reason stop, but nothing came back: a Glimmer serving trap (RT1)
    result = complete_structured(make_client(), spark_request(), Verdict)
    assert isinstance(result, Unparsed)
    assert result.truncated
    assert "empty answer" in result.detail
    assert len(meta.requests) == 1
    assert [c.micro_usd for c in response_costs(result.responses)] == [result.costs[0].micro_usd]


# ---------------------------------------------------------------- a spend file only 'spend arm' makes (blocker 3)


def git(*args: str, cwd: Path) -> None:
    """git with no user or system configuration (no signing, no hooks), for a throwaway repository in a test folder."""
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": str(cwd),
        "GIT_CONFIG_GLOBAL": os.devnull,
        "GIT_CONFIG_SYSTEM": os.devnull,
        "GIT_CONFIG_NOSYSTEM": "1",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_AUTHOR_NAME": "planter",
        "GIT_AUTHOR_EMAIL": "planter@example.invalid",
        "GIT_COMMITTER_NAME": "planter",
        "GIT_COMMITTER_EMAIL": "planter@example.invalid",
    }
    options = ["-c", "commit.gpgsign=false", "-c", f"core.hooksPath={os.devnull}", "-c", "init.defaultBranch=main"]
    subprocess.run(["git", *options, *args], cwd=cwd, env=env, check=True, capture_output=True, timeout=60)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize("umask", [0o022, 0o077], ids=["umask-022", "umask-077"])
def test_a_spend_file_shipped_in_a_git_clone_never_arms_the_meter(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, meta: FakeMeta, clock: FakeClock, fake_key: str, umask: int
) -> None:
    # An attacker's repository ships .receipts/spend.jsonl, armed at the full credit by a record that names the
    # attacker's own file. Someone clones it and runs receipts there, with no RECEIPTS_SPEND_FILE.
    origin = tmp_path / "origin"
    origin.mkdir()
    git("init", "-q", cwd=origin)
    planted = private_spend_file(origin / ".receipts" / "spend.jsonl", lambda inode: arm_line(inode) + "\n")
    git("add", "-A", cwd=origin)
    git("commit", "-q", "-m", "ship a spend file", cwd=origin)
    previous = os.umask(umask)
    try:
        git("clone", "-q", str(origin), str(tmp_path / "clone"), cwd=tmp_path)
    finally:
        os.umask(previous)
    cloned = tmp_path / "clone" / ".receipts" / "spend.jsonl"
    assert cloned.read_text() == planted.read_text()  # the arm record came along
    monkeypatch.delenv(SPEND_FILE_ENV)
    monkeypatch.chdir(tmp_path / "clone")
    assert default_spend_path() == cloned.resolve()

    meta.queue(completion())
    options: dict[str, Any] = {"transport": httpx.MockTransport(meta.handler), "clock": clock.monotonic}
    # Refused when the client is built, or at its first call.
    with pytest.raises(SpendCapError) as caught, MetaClient(api_key=fake_key, sleep=clock.sleep, **options) as client:
        client.complete(spark_request())
    assert caught.value.reason is UncheckedReason.BUDGET
    assert meta.requests == []
    if umask == 0o022:  # git wrote the file 0644 in a 0755 directory: not a file 'spend arm' made
        assert isinstance(caught.value, SpendFileError)
        assert "has mode 0644, not 0600" in str(caught.value)
    else:  # under umask 077 the modes look right, but the arm record names another file's inode
        assert isinstance(caught.value, SpendNotArmedError)
        assert "written to another file" in str(caught.value)


@pytest.mark.parametrize("umask", [0o000, 0o022, 0o277], ids=["umask-000", "umask-022", "umask-277"])
def test_arming_makes_the_directory_0700_and_the_file_0600_whatever_the_umask(tmp_path: Path, umask: int) -> None:
    # 0o277 strips the owner's write bit too: only an explicit chmod gives the 0700 and 0600 arming promises.
    path = tmp_path / "runs" / "pilot" / "spend.jsonl"
    previous = os.umask(umask)
    try:
        SpendMeter(path).arm(credit_confirmed=True)
    finally:
        os.umask(previous)
    assert (path.parent.stat().st_mode & 0o777, path.stat().st_mode & 0o777) == (SPEND_DIR_MODE, SPEND_FILE_MODE)
    assert (SPEND_DIR_MODE, SPEND_FILE_MODE) == (0o700, 0o600)
    assert SpendMeter(path).limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD


def test_arming_never_adopts_a_directory_or_file_it_did_not_make_private(tmp_path: Path) -> None:
    shared = tmp_path / "shared"
    shared.mkdir(mode=0o755)
    shared.chmod(0o755)
    with pytest.raises(SpendFileError, match="in a directory that has mode 0755, not 0700"):
        SpendMeter(shared / "spend.jsonl").arm(credit_confirmed=True)
    assert shared.stat().st_mode & 0o777 == 0o755  # left as it was
    assert not (shared / "spend.jsonl").exists()  # and nothing was made in it

    private = tmp_path / "private"
    private.mkdir(mode=0o700)
    loose = private / "spend.jsonl"
    loose.write_text("")
    loose.chmod(0o644)
    with pytest.raises(SpendFileError, match="has mode 0644, not 0600"):
        SpendMeter(loose).arm(credit_confirmed=True)
    assert loose.read_text() == ""


def break_mode(path: Path) -> None:
    path.chmod(0o644)


def break_group_read(path: Path) -> None:
    path.chmod(0o640)


def break_directory(path: Path) -> None:
    path.parent.chmod(0o755)


def break_directory_group(path: Path) -> None:
    path.parent.chmod(0o750)


def break_with_a_link(path: Path) -> None:
    real = path.with_name("real.jsonl")
    path.rename(real)
    path.symlink_to(real)


NOT_PRIVATE = {
    "file-0644": (break_mode, "has mode 0644, not 0600"),
    "file-0640": (break_group_read, "has mode 0640, not 0600"),
    "directory-0755": (break_directory, "in a directory that has mode 0755, not 0700"),
    "directory-0750": (break_directory_group, "in a directory that has mode 0750, not 0700"),
    "symlink": (break_with_a_link, "symbolic link"),
}


@pytest.mark.parametrize("damage", list(NOT_PRIVATE.values()), ids=list(NOT_PRIVATE))
def test_a_spend_file_that_is_not_private_refuses_every_call(
    tmp_path: Path, damage: tuple[Callable[[Path], None], str]
) -> None:
    breaker, message = damage
    path = tmp_path / "spend" / "spend.jsonl"
    armed_meter(path)
    breaker(path)
    with pytest.raises(SpendFileError, match=message) as caught:
        SpendMeter(path)
    assert caught.value.reason is UncheckedReason.BUDGET
    assert "a checkout can ship its own" in str(caught.value) or "symbolic link" in str(caught.value)


def test_a_spend_file_owned_by_another_user_refuses_every_call(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "spend" / "spend.jsonl"
    meter = armed_meter(path)
    monkeypatch.setattr(meta_client, "_current_uid", lambda: os.getuid() + 1)  # the file is now someone else's
    with pytest.raises(SpendFileError, match="belongs to another user"):
        SpendMeter(path)
    with pytest.raises(SpendFileError, match="belongs to another user"):  # a meter already open finds it too
        meter.reserve(1, stage="intake", model=SPARK)


def test_an_arm_record_copied_into_another_file_arms_nothing(tmp_path: Path) -> None:
    original = tmp_path / "a" / "spend.jsonl"
    armed_meter(original)
    copy = private_spend_file(tmp_path / "b" / "spend.jsonl", lambda inode: original.read_text())
    meter = SpendMeter(copy)
    assert (meter.armed, meter.foreign_arms, meter.limit_micro_usd) == (None, 1, 0)
    with pytest.raises(SpendNotArmedError, match="1 arm record"):
        meter.reserve(1, stage="intake", model=SPARK)
    meter.arm(credit_confirmed=True)  # an operator arms the copy itself
    assert meter.limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD


def test_a_spend_file_replaced_by_another_file_refuses_every_call(tmp_path: Path) -> None:
    path = tmp_path / "spend" / "spend.jsonl"
    meter = armed_meter(path)
    stand_in = private_spend_file(tmp_path / "spend" / "other.jsonl", lambda inode: path.read_text())
    os.replace(stand_in, path)  # the same bytes, private, but another file
    with pytest.raises(SpendFileError, match="replaced"):
        meter.reserve(1, stage="intake", model=SPARK)


# ---------------------------------------------------------------- every core error pickles (blocker 4)


def every_subclass(cls: type[BaseException]) -> set[type[BaseException]]:
    found: set[type[BaseException]] = set()
    for sub in cls.__subclasses__():
        found |= {sub, *every_subclass(sub)}
    return found


def test_every_core_error_survives_a_pickle_round_trip() -> None:
    errors: list[BaseException] = [
        MetaClientError("plain", reason=UncheckedReason.TIMEOUT),
        MissingKeyError("no key"),
        SpendCapError("cap"),
        SpendNotArmedError("not armed"),
        SpendFileError("file"),
        MetaAPIError("api", status=503, reason=UncheckedReason.TIMEOUT),
        MetaRetriesExhaustedError("exhausted", attempts=6, status=429),
        MetaResponseError("response"),
        GlimmerError("glimmer"),
        GlimmerConfigError("config"),
        GlimmerAPIError("api", status=500, attempts=3, reason=UncheckedReason.TIMEOUT),
        GlimmerResponseError("response"),
    ]
    assert {type(e) for e in errors} == every_subclass(CoreCallError)  # a new error class needs a case here
    for error in errors:
        assert isinstance(error, ModelCallError)
        error.cost = StageCost(stage="intake", model=SPARK, input_tokens=3, micro_usd=1_234, seconds=1.5)
        back = pickle.loads(pickle.dumps(error))
        assert type(back) is type(error)
        assert (str(back), back.args) == (str(error), error.args)
        assert vars(back) == vars(error)  # reason, cost, and status or attempts where the class has them
        assert back.reason is error.reason
    unpriced = UnpricedModelError("sam-3.1: no list price")
    assert str(pickle.loads(pickle.dumps(unpriced))) == str(unpriced)


# ---------------------------------------------------------------- never undercount, never pass the cap (call a)


def test_a_spark_call_answered_by_wonder_never_settles_past_the_cap(
    make_client: Callable[..., MetaClient], meta: FakeMeta, tmp_path: Path
) -> None:
    request = spark_request(max_tokens=1_000)
    size = len(encode_request(request))
    spark_hold = LIST_PRICES[SPARK].worst_case_micro_usd(size, request.max_tokens)
    hold = worst_case(request)
    # Wonder answers with the call's whole token bounds: more than a Spark-priced hold, within the highest-priced one.
    wonder_cost = LIST_PRICES["muse-wonder-1.0"].micro_usd(Usage(input_tokens=size, output_tokens=request.max_tokens))
    assert spark_hold < wonder_cost <= hold

    # A cap with room for a Spark-priced hold only: the call is held at the highest price and refused before it goes.
    small = armed_meter(tmp_path / "small" / "spend.jsonl", cap=spark_hold)
    meta.queue(completion(model="muse-wonder-1.0", prompt=size, output=request.max_tokens))
    with pytest.raises(SpendCapError, match="could pass"):
        make_client(meter=small).complete(request)
    assert (meta.requests, small.spent_micro_usd) == ([], 0)

    # A cap that holds the highest price: Wonder's answer at the bounds settles within it, and names Wonder.
    path = tmp_path / "exact" / "spend.jsonl"
    exact = armed_meter(path, cap=hold)
    response = make_client(meter=exact).complete(request)
    assert response.cost is not None
    assert (response.cost.micro_usd, response.cost.model) == (wonder_cost, "muse-wonder-1.0")
    assert exact.spent_micro_usd == wonder_cost <= exact.limit_micro_usd == hold
    holds = [json.loads(line) for line in path.read_text().splitlines() if '"kind":"hold"' in line]
    assert [h["micro_usd"] for h in holds] == [hold]


def test_a_meter_whose_spend_equals_its_cap_refuses_even_a_free_call(spend_path: Path) -> None:
    meter = armed_meter(spend_path, cap=1_000)
    meter.settle(meter.reserve(1_000, stage="intake", model=SPARK), 1_000, estimated=False)
    assert meter.spent_micro_usd == meter.limit_micro_usd
    with pytest.raises(SpendCapError, match="reached"):
        meter.reserve(0, stage="intake", model=SPARK)


# ---------------------------------------------------------------- a linked spend file is never followed (call b)


@pytest.mark.skipif(shutil.which("git") is None, reason="needs git")
@pytest.mark.parametrize("umask", [0o022, 0o077], ids=["umask-022", "umask-077"])
@pytest.mark.parametrize("linked", ["file", "directory"])
def test_a_clone_whose_spend_file_links_to_an_armed_file_elsewhere_is_refused(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    meta: FakeMeta,
    clock: FakeClock,
    fake_key: str,
    umask: int,
    linked: str,
) -> None:
    # The operator armed a spend file of their own. An attacker's repository ships .receipts/spend.jsonl as a link to
    # it (or .receipts as a link to its directory); someone clones it and runs receipts there.
    operator = tmp_path / "operator" / "spend.jsonl"
    armed_meter(operator)
    before = operator.read_text()
    origin = tmp_path / "origin"
    origin.mkdir()
    git("init", "-q", cwd=origin)
    if linked == "file":
        (origin / ".receipts").mkdir(mode=0o700)
        (origin / ".receipts" / "spend.jsonl").symlink_to(operator)
    else:
        (origin / ".receipts").symlink_to(operator.parent, target_is_directory=True)
    git("add", "-A", cwd=origin)
    git("commit", "-q", "-m", "ship a link to the operator's spend file", cwd=origin)
    previous = os.umask(umask)
    try:
        git("clone", "-q", str(origin), str(tmp_path / "clone"), cwd=tmp_path)
    finally:
        os.umask(previous)
    cloned = tmp_path / "clone" / ".receipts" / "spend.jsonl"
    assert (cloned.parent if linked == "directory" else cloned).is_symlink()
    monkeypatch.delenv(SPEND_FILE_ENV)
    monkeypatch.chdir(tmp_path / "clone")

    meta.queue(completion())
    options: dict[str, Any] = {"transport": httpx.MockTransport(meta.handler), "clock": clock.monotonic}
    with pytest.raises(SpendFileError) as caught, MetaClient(api_key=fake_key, sleep=clock.sleep, **options) as client:
        client.complete(spark_request())
    assert caught.value.reason is UncheckedReason.BUDGET
    assert meta.requests == []
    assert operator.read_text() == before  # nothing was held or metered on the operator's file
    if linked == "directory" or umask == 0o077:  # the modes look right; the link is what is refused
        assert "symbolic link" in str(caught.value)
    assert default_spend_path() == cloned.absolute()  # the link stays in the path: never resolved


def test_the_spend_path_keeps_a_linked_folder_and_the_meter_refuses_it(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    real = tmp_path / "real"
    armed_meter(real / "spend.jsonl")
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    monkeypatch.setenv(SPEND_FILE_ENV, str(tmp_path / "link" / "spend.jsonl"))
    assert default_spend_path() == tmp_path / "link" / "spend.jsonl"  # never resolved to real/spend.jsonl
    for build in (shared_meter, lambda: SpendMeter(default_spend_path())):
        with pytest.raises(SpendFileError, match="in a directory that is a symbolic link"):
            build()

    # A link further up is the operator's own layout, above any checkout: it stays in the path and is allowed.
    (tmp_path / "up").symlink_to(tmp_path / "upper", target_is_directory=True)
    (tmp_path / "upper").mkdir()
    above = tmp_path / "up" / "runs" / "spend.jsonl"
    armed_meter(above)
    monkeypatch.setenv(SPEND_FILE_ENV, str(above))
    assert shared_meter().path == above != above.resolve()
    assert shared_meter().limit_micro_usd == DEFAULT_SPEND_CAP_MICRO_USD
