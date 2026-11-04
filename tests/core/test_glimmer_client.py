"""The Glimmer arm's client, against httpx.MockTransport: no request reaches a network or a model. It is fed the inputs
it must refuse: other model ids, bad base URLs, answers from another model, and truncated or empty answers."""

from __future__ import annotations

import json
import logging
import pickle
import random
from collections.abc import Callable, Iterator
from typing import Any

import httpx
import pytest
from pydantic import BaseModel, ConfigDict

from receipts.agentloop.attacker import WRITE_SCHEMA
from receipts.attack.correctness import candidates_schema
from receipts.contracts import Message, ModelRefusedError, ModelRequest, ModelResponse, UncheckedReason, Usage
from receipts.core.glimmer_client import (
    GLIMMER_MODEL,
    GLIMMER_REPETITION_PENALTY,
    GLIMMER_URL_ENV,
    MODAL_KEY_ENV,
    MODAL_SECRET_ENV,
    GlimmerAPIError,
    GlimmerClient,
    GlimmerConfigError,
    GlimmerResponseError,
    encode_request,
    glimmer_base_url,
    joined_code,
    lines_schema,
    user_message,
)
from receipts.core.routing import ModelRouter
from receipts.core.structured import Parsed, Unparsed, complete_structured

BASE = "http://10.0.0.7:8000/v1"


class Clock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


class Server:
    """A scripted vLLM behind MockTransport; each request takes the next reply and advances the clock 1.5 s."""

    def __init__(self, clock: Clock) -> None:
        self.clock = clock
        self.replies: list[httpx.Response | Callable[[httpx.Request], httpx.Response]] = []
        self.requests: list[httpx.Request] = []

    def handler(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.clock.now += 1.5
        reply = self.replies.pop(0)
        return reply(request) if callable(reply) else reply


def answer(
    content: str | None = "ok", *, finish: str = "stop", model: str = GLIMMER_MODEL, usage: bool = True
) -> httpx.Response:
    body: dict[str, Any] = {
        "id": "chatcmpl-9",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish}],
    }
    if usage:
        body["usage"] = {"prompt_tokens": 300, "completion_tokens": 40, "total_tokens": 340}
    return httpx.Response(200, json=body)


def request(**changes: Any) -> ModelRequest:
    fields: dict[str, Any] = {
        "stage": "attack",
        "model": GLIMMER_MODEL,
        "messages": [{"role": "user", "content": "write a failing test"}],
        "temperature": 0.8,
        "max_tokens": 512,
        "reasoning_effort": "medium",
    }
    fields.update(changes)
    return ModelRequest.model_validate(fields)


class Verdict(BaseModel):
    model_config = ConfigDict(extra="forbid")
    bug: bool


@pytest.fixture
def clock() -> Clock:
    return Clock()


@pytest.fixture
def server(clock: Clock) -> Server:
    return Server(clock)


@pytest.fixture
def make(server: Server, clock: Clock) -> Iterator[Callable[..., GlimmerClient]]:
    made: list[GlimmerClient] = []

    def build(**options: Any) -> GlimmerClient:
        options.setdefault("base_url", BASE)
        client = GlimmerClient(
            transport=httpx.MockTransport(server.handler),
            clock=clock.monotonic,
            sleep=clock.sleep,
            rng=random.Random(3),
            **options,
        )
        made.append(client)
        return client

    yield build
    for client in made:
        client.close()


def test_a_call_goes_to_the_configured_server_with_no_key_and_records_usage_and_seconds(
    make: Callable[..., GlimmerClient], server: Server, caplog: pytest.LogCaptureFixture
) -> None:
    server.replies.append(answer("here is a test"))
    with caplog.at_level(logging.INFO, logger="receipts.core.glimmer_client"):
        response = make().complete(request())
    (sent,) = server.requests
    assert str(sent.url) == f"{BASE}/chat/completions"
    assert "authorization" not in sent.headers
    assert json.loads(sent.content) == {
        "model": GLIMMER_MODEL,
        "messages": [{"role": "user", "content": "write a failing test"}],
        "temperature": 0.8,
        "max_tokens": 512,
        "repetition_penalty": 1.05,
        "skip_special_tokens": False,
    }  # no Meta-only fields: no prompt_cache_key, no reasoning_effort
    assert (response.text, response.model, response.finish_reason) == ("here is a test", GLIMMER_MODEL, "stop")
    assert response.usage == Usage(input_tokens=300, output_tokens=40)
    cost = response.cost
    assert cost is not None
    assert (cost.stage, cost.model, cost.input_tokens, cost.output_tokens) == ("attack", GLIMMER_MODEL, 300, 40)
    assert (cost.micro_usd, cost.seconds) == (0, 1.5)  # Falden's own GPU: seconds, never money
    assert "glimmer call stage=attack" in caplog.text and "output_tokens=40" in caplog.text


def test_every_request_carries_a_mild_repetition_penalty_and_no_other_penalty() -> None:
    """The 10/29 VM test: with the schema enforced, every stage-1 answer looped inside the JSON "code" string."""
    assert GLIMMER_REPETITION_PENALTY == 1.05
    for schema in (None, Verdict.model_json_schema()):
        body = json.loads(encode_request(request(json_schema=schema)))
        assert body["repetition_penalty"] == 1.05
        assert "frequency_penalty" not in body and "presence_penalty" not in body
        assert ("response_format" in body) is (schema is not None)
    assert json.loads(encode_request(request(), repetition_penalty=1.15))["repetition_penalty"] == 1.15


@pytest.mark.parametrize("penalty", [1.1, 1.15, 1.0])
def test_a_retest_sets_the_penalty_through_the_constructor(
    make: Callable[..., GlimmerClient], server: Server, penalty: float
) -> None:
    server.replies += [answer(), answer('{"bug": true}')]
    client = make(repetition_penalty=penalty)
    client.complete(request())
    complete_structured(client, request(), Verdict)
    assert [json.loads(r.content)["repetition_penalty"] for r in server.requests] == [penalty, penalty]


@pytest.mark.parametrize("penalty", [0.9, 0.0, -1.0, 2.5, float("nan"), True])
def test_a_penalty_outside_one_to_two_is_refused(penalty: float) -> None:
    with pytest.raises(ValueError, match="repetition_penalty"):
        GlimmerClient(base_url=BASE, repetition_penalty=penalty)


def test_the_base_url_comes_from_receipts_glimmer_url_and_bad_ones_are_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert GLIMMER_URL_ENV == "RECEIPTS_GLIMMER_URL"
    monkeypatch.setenv(GLIMMER_URL_ENV, "https://gpu.internal:8443/v1/")
    assert glimmer_base_url() == "https://gpu.internal:8443/v1"
    monkeypatch.delenv(GLIMMER_URL_ENV)
    with pytest.raises(GlimmerConfigError, match=GLIMMER_URL_ENV) as caught:
        glimmer_base_url()
    assert caught.value.reason is UncheckedReason.RUNTIME
    for bad in ("gpu:8000", "ftp://gpu/v1", "http://user:secret@gpu/v1", "http://gpu/v1?x=1", "http:///v1", "  "):
        with pytest.raises(GlimmerConfigError):
            glimmer_base_url(bad)


@pytest.mark.parametrize("model", ["muse-spark-1.3", "muse-spark-1.3-contributor", "gpt-x"])
def test_only_glimmer_is_ever_sent(make: Callable[..., GlimmerClient], server: Server, model: str) -> None:
    built = ModelRequest.model_construct(
        stage="attack", model=model, messages=(Message(role="user", content="x"),), temperature=0.0, max_tokens=8
    )
    with pytest.raises(ModelRefusedError):
        make().complete(built)
    assert server.requests == []


@pytest.mark.parametrize(("content", "finish"), [("cut off mid", "length"), (None, "stop"), ("  ", "stop")])
def test_a_truncated_or_empty_answer_invalidates_the_item(
    make: Callable[..., GlimmerClient],
    server: Server,
    caplog: pytest.LogCaptureFixture,
    content: str | None,
    finish: str,
) -> None:
    # An empty answer with finish_reason stop is the Glimmer serving trap: the reasoning ate the whole budget.
    server.replies.append(answer(content, finish=finish))
    client = make()
    with caplog.at_level(logging.WARNING, logger="receipts.core.glimmer_client"):
        result = complete_structured(client, request(json_schema=Verdict.model_json_schema()), Verdict)
    assert isinstance(result, Unparsed)
    assert result.truncated
    assert len(server.requests) == 1  # no retry
    assert "truncated" in caplog.text
    server.replies.append(answer('{"bug": true}' if finish == "length" else content, finish=finish))
    raw = client.complete(request(json_schema=Verdict.model_json_schema()))
    assert raw.parsed is None and raw.truncated


def test_structured_output_validates_and_retries_once_like_the_meta_client(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    server.replies += [answer('{"bug": "maybe"}'), answer('{"bug": true}')]
    result = complete_structured(make(), request(), Verdict)
    assert isinstance(result, Parsed) and result.value == Verdict(bug=True)
    first, second = (json.loads(r.content) for r in server.requests)
    assert first["response_format"]["type"] == "json_schema"
    assert first["response_format"]["json_schema"]["schema"] == Verdict.model_json_schema()
    assert "bug: Input should be a valid boolean" in second["messages"][-1]["content"]
    assert [c.micro_usd for c in result.costs] == [0, 0]


def test_an_answer_from_another_model_or_without_usage_fails_the_call(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    server.replies += [answer(model="muse-spark-1.3"), answer(usage=False)]
    client = make()
    with pytest.raises(GlimmerResponseError, match="not muse-glimmer-30b") as other:
        client.complete(request())
    assert other.value.reason is UncheckedReason.RUNTIME
    assert other.value.cost is not None and other.value.cost.micro_usd == 0 and other.value.cost.seconds == 1.5
    # The tokens the server reported for the answer it sent are kept: the GPU spent them (round 3, call e).
    assert (other.value.cost.input_tokens, other.value.cost.output_tokens) == (300, 40)
    with pytest.raises(GlimmerResponseError, match="without valid usage") as unknown:
        client.complete(request())
    assert unknown.value.cost is not None and (unknown.value.cost.input_tokens, unknown.value.cost.output_tokens) == (
        0,
        0,
    )


def test_a_malformed_200_keeps_the_usage_the_server_reported(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    body = json.loads(answer().content)
    body["choices"] = []  # malformed: no choice, but the usage is the server's own count
    body["usage"]["completion_tokens_details"] = {"reasoning_tokens": 12}
    server.replies.append(httpx.Response(200, json=body))
    with pytest.raises(GlimmerResponseError, match="malformed") as caught:
        make().complete(request())
    cost = caught.value.cost
    assert cost is not None
    assert (cost.input_tokens, cost.output_tokens, cost.reasoning_tokens, cost.micro_usd) == (300, 40, 12, 0)
    assert cost.seconds == 1.5


def test_overload_is_retried_and_a_refusal_is_not(
    make: Callable[..., GlimmerClient], server: Server, clock: Clock
) -> None:
    busy = httpx.Response(
        503, json={"error": {"message": "busy", "type": "server_error"}}, headers={"Retry-After": "2"}
    )
    server.replies += [busy, answer("second time")]
    assert make().complete(request()).text == "second time"
    assert clock.sleeps and clock.sleeps[0] >= 2.0

    server.replies.append(httpx.Response(400, json={"error": {"message": "bad request", "type": "invalid"}}))
    with pytest.raises(GlimmerAPIError, match="bad request") as refused:
        make().complete(request())
    assert (refused.value.status, refused.value.attempts) == (400, 1)

    def timeout(sent: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("slow", request=sent)

    server.replies += [timeout] * 3
    with pytest.raises(GlimmerAPIError) as exhausted:
        make(max_retries=2, max_timeout_retries=2).complete(request())
    assert (exhausted.value.attempts, exhausted.value.reason) == (3, UncheckedReason.TIMEOUT)


def test_a_read_timeout_is_retried_once_by_default_and_other_failures_still_get_every_retry(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    # The 10/28 pilot: django-13128, sphinx-9591 and matplotlib-22865 stage 1, "after 4 attempts; ReadTimeout",
    # 2,405 s apiece and no answer.
    def timeout(sent: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timed out", request=sent)

    server.replies += [timeout] * 4
    with pytest.raises(GlimmerAPIError, match="after 2 attempts; ReadTimeout") as caught:
        make().complete(request())
    assert (caught.value.attempts, caught.value.reason) == (2, UncheckedReason.TIMEOUT)
    assert len(server.requests) == 2

    busy = httpx.Response(503, json={"error": {"message": "busy", "type": "server_error"}})
    server.replies[:] = [busy, timeout, busy, answer("fourth time")]
    assert make().complete(request()).text == "fourth time"  # one timeout among the 3 retries is allowed
    with pytest.raises(ValueError, match="max_timeout_retries"):
        make(max_timeout_retries=-1)


def test_the_egress_is_pinned_no_redirect_no_proxy(
    make: Callable[..., GlimmerClient], server: Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    server.replies += [httpx.Response(307, headers={"Location": "https://collector.example.net/v1/chat/completions"})]
    with pytest.raises(GlimmerAPIError) as caught:
        make().complete(request())
    assert caught.value.status == 307
    assert [r.url.host for r in server.requests] == ["10.0.0.7"]

    built: list[dict[str, Any]] = []
    original = httpx.HTTPTransport.__init__

    def recording(self: httpx.HTTPTransport, *args: Any, **kwargs: Any) -> None:
        built.append(kwargs)
        original(self, *args, **kwargs)

    monkeypatch.setenv("HTTP_PROXY", "http://proxy.invalid:3128")
    monkeypatch.setattr(httpx.HTTPTransport, "__init__", recording)
    with GlimmerClient(base_url=BASE):
        pass
    (transport,) = built
    assert transport.get("proxy") is None and transport.get("trust_env") is False


def test_the_router_sends_each_model_to_its_own_client() -> None:
    class Echo:
        def __init__(self, name: str) -> None:
            self.name = name

        def complete(self, request: ModelRequest) -> ModelResponse:
            return ModelResponse(
                model=request.model, text=self.name, finish_reason="stop", usage=Usage(input_tokens=1, output_tokens=1)
            )

    router = ModelRouter({"muse-spark-1.3": Echo("meta"), GLIMMER_MODEL: Echo("vllm")})
    assert router.complete(request(model="muse-spark-1.3")).text == "meta"
    assert router.complete(request()).text == "vllm"
    with pytest.raises(ModelRefusedError, match="no client serves it"):
        router.complete(request(model="muse-wonder-1.0"))
    with pytest.raises(ModelRefusedError):
        ModelRouter({"muse-spark-1.3-contributor": Echo("never")})


MODAL_BASE = "https://example--receipts-glimmer-serve.us-east.modal.run/v1"
MODAL_ENV = {MODAL_KEY_ENV: "wk-test-id", MODAL_SECRET_ENV: "ws-test-secret"}


def test_a_server_on_modal_gets_the_proxy_auth_headers_and_never_shows_them(
    make: Callable[..., GlimmerClient], server: Server, caplog: pytest.LogCaptureFixture
) -> None:
    server.replies.append(answer("on modal"))
    with caplog.at_level(logging.DEBUG, logger="receipts.core.glimmer_client"):
        client = make(base_url=MODAL_BASE, env=MODAL_ENV)
        assert client.complete(request()).text == "on modal"
    sent = server.requests[0]
    assert sent.headers["Modal-Key"] == "wk-test-id"
    assert sent.headers["Modal-Secret"] == "ws-test-secret"
    assert "authorization" not in sent.headers
    shown = repr(client) + caplog.text
    assert "wk-test-id" not in shown and "ws-test-secret" not in shown


def test_a_server_anywhere_else_never_gets_the_modal_token(make: Callable[..., GlimmerClient], server: Server) -> None:
    server.replies += [answer("own vm"), answer("look-alike")]
    make(env=MODAL_ENV).complete(request())  # Falden's own VM, BASE
    make(base_url="https://modal.run.example.net/v1", env=MODAL_ENV).complete(request())  # not a .modal.run host
    for sent in server.requests:
        assert "modal-key" not in sent.headers and "modal-secret" not in sent.headers


@pytest.mark.parametrize("missing", [MODAL_KEY_ENV, MODAL_SECRET_ENV])
def test_a_server_on_modal_without_its_token_is_refused_before_any_request(
    make: Callable[..., GlimmerClient], server: Server, missing: str
) -> None:
    env = {k: v for k, v in MODAL_ENV.items() if k != missing}
    with pytest.raises(GlimmerConfigError, match=missing):
        make(base_url=MODAL_BASE, env=env)
    with pytest.raises(GlimmerConfigError, match=missing):
        make(base_url=MODAL_BASE, env={**MODAL_ENV, missing: "  "})
    assert server.requests == []


@pytest.mark.parametrize(
    ("url", "why"),
    [
        ("http://example--receipts-glimmer-serve.us-east.modal.run/v1", "https only"),
        ("https://example--receipts-glimmer-serve.us-east.modal.run./v1", "trailing dot"),
        ("https://exa_mple--serve.us-east.modal.run/v1", "letters, digits, hyphens"),
    ],
)
def test_a_modal_url_that_could_expose_the_token_is_refused(
    make: Callable[..., GlimmerClient], server: Server, url: str, why: str
) -> None:
    with pytest.raises(GlimmerConfigError, match=why):
        make(base_url=url, env=MODAL_ENV)
    assert server.requests == []


@pytest.mark.parametrize("value", ["ws-se\ncret", "ws-se\rcret", "ws se cret", "ws-sécret"])
def test_a_malformed_token_value_is_refused_by_name_only(
    make: Callable[..., GlimmerClient], server: Server, value: str
) -> None:
    with pytest.raises(GlimmerConfigError, match=MODAL_SECRET_ENV) as raised:
        make(base_url=MODAL_BASE, env={**MODAL_ENV, MODAL_SECRET_ENV: value})
    assert value not in str(raised.value)
    assert server.requests == []


def test_a_look_alike_suffix_gets_no_token(make: Callable[..., GlimmerClient], server: Server) -> None:
    server.replies.append(answer("not modal"))
    make(base_url="https://evilmodal.run/v1", env=MODAL_ENV).complete(request())
    assert "modal-secret" not in server.requests[0].headers


def test_the_token_is_kept_out_of_default_headers_and_scrubbed_from_a_failure(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    def leaky(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"broke sending {request.headers['Modal-Secret']} {request.headers['Modal-Key']}")

    server.replies += [leaky] * 10
    client = make(base_url=MODAL_BASE, env=MODAL_ENV, max_retries=0)
    assert "ws-test-secret" not in repr(client._http.headers)
    with pytest.raises(GlimmerAPIError) as raised:
        client.complete(request())
    text = str(raised.value) + repr(raised.value)
    assert "ws-test-secret" not in text and "wk-test-id" not in text
    assert "[redacted]" in text


def test_an_error_body_that_echoes_the_token_is_scrubbed(make: Callable[..., GlimmerClient], server: Server) -> None:
    echo = {"error": {"message": "bad key wk-test-id / ws-test-secret", "type": "auth"}}
    server.replies.append(httpx.Response(401, json=echo))
    with pytest.raises(GlimmerAPIError) as raised:
        make(base_url=MODAL_BASE, env=MODAL_ENV).complete(request())
    text = str(raised.value)
    assert "ws-test-secret" not in text and "wk-test-id" not in text
    assert "[redacted]" in text


def test_the_token_object_never_shows_or_pickles_its_values(make: Callable[..., GlimmerClient]) -> None:
    token = make(base_url=MODAL_BASE, env=MODAL_ENV)._token
    assert token is not None
    for shown in (repr(token), str(token)):
        assert "ws-test-secret" not in shown and "wk-test-id" not in shown
    with pytest.raises(TypeError):
        pickle.dumps(token)


# A real answer, from the Modal endpoint on 2026-10-30 with skip_special_tokens false (reasoning shortened).
LIVE_ANSWER = (
    " to=self<|message|>Repository: django/django. Find where DurationField is defined. Answer now.\n\nWe need exactly "
    "one fenced ```bash block with one command.\n\nFinal response: one line reasoning then code block.\n\n<|eom|>"
    "<|start|>assistant to=user<|message|>The class definition can be located with a recursive grep for the class "
    'name in Python files.\n```bash\ngrep -rn "class DurationField" --include="*.py" .\n```'
)
LIVE_USER = (
    "The class definition can be located with a recursive grep for the class name in Python files.\n"
    '```bash\ngrep -rn "class DurationField" --include="*.py" .\n```'
)


def test_a_free_text_request_keeps_the_special_tokens_and_a_schema_request_does_not() -> None:
    """Without a reasoning parser the ATEM headers run into the text once vLLM drops the special tokens."""
    assert json.loads(encode_request(request()))["skip_special_tokens"] is False
    assert "skip_special_tokens" not in json.loads(encode_request(request(json_schema=Verdict.model_json_schema())))


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (LIVE_ANSWER, LIVE_USER),
        (LIVE_ANSWER + "<|eot|>", LIVE_USER),
        (" to=user<|message|>only the answer<|eot|>", "only the answer"),
        (  # the last message to the user wins; the reasoning between is dropped
            " to=user<|message|>draft<|eom|><|start|>assistant to=self<|message|>rethink<|eom|>"
            "<|start|>assistant to=user<|message|>final",
            "final",
        ),
        ("plain text with no framing", "plain text with no framing"),
        (" to=self<|message|>reasoning that never reached the user", ""),
        (" to=self<|message|>think<|eom|><|start|>assistant to=functions.run<|message|><atem:function_calls>", ""),
        (" to=self<|message|>think<|eom|><|start|>assistant to=us", ""),  # cut inside the header
        (" to=username<|message|>not the user channel", ""),
        (" to=self <|message|> think <|eom|> <|start|> assistant to=user <|message|> spaced answer ", "spaced answer"),
        ("<|start|>assistant<|message|>no recipient is the user", "no recipient is the user"),
        (
            " to=user<|message|>a file says <|start|> here\n```bash\nls\n```",
            "a file says <|start|> here\n```bash\nls\n```",
        ),
    ],
)
def test_the_answer_is_the_last_message_to_the_user(text: str, expected: str) -> None:
    assert user_message(text) == expected


def test_a_free_text_answer_reaches_the_caller_as_its_user_message(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    server.replies.append(answer(LIVE_ANSWER))
    response = make().complete(request())
    assert response.text == LIVE_USER and not response.truncated


def test_reasoning_with_no_answer_is_an_empty_answer(make: Callable[..., GlimmerClient], server: Server) -> None:
    server.replies.append(answer(" to=self<|message|>We need one command.<|eom|>"))
    response = make().complete(request())
    assert response.text == "" and response.truncated


def test_a_schema_answer_is_parsed_as_sent(make: Callable[..., GlimmerClient], server: Server) -> None:
    server.replies.append(answer('{"bug": true}'))
    result = complete_structured(make(), request(json_schema=Verdict.model_json_schema()), Verdict)
    assert isinstance(result, Parsed) and result.value == Verdict(bug=True)


def test_a_framed_answer_with_no_user_message_is_logged_for_the_trace(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="receipts.core.glimmer_client"):
        assert user_message(" to=self<|message|>only reasoning") == ""
    assert "only reasoning" in caplog.text


class Candidate(BaseModel):
    path: str
    code: str
    why: str = ""


class Candidates(BaseModel):
    tests: list[Candidate]


def test_glimmer_gets_code_as_lines_and_the_rest_of_the_schema_as_sent() -> None:
    """On Modal on 10/30 Glimmer wrote every "code" string as one line; as an array of lines, 23 of 23 parsed."""
    schema = candidates_schema()
    before = json.dumps(schema, sort_keys=True)
    lines = lines_schema(schema)
    assert lines is not None and json.dumps(schema, sort_keys=True) == before  # the caller's schema is untouched
    item = lines["$defs"]["_Candidate"]
    assert list(item["properties"]) == ["path", "code_lines", "why", "quote"]
    assert item["properties"]["code_lines"]["type"] == "array"
    assert item["properties"]["code_lines"]["items"] == {"type": "string"}
    assert item["required"] == ["path", "code_lines", "quote"]
    assert {k: v for k, v in item["properties"].items() if k != "code_lines"} == {
        k: v for k, v in schema["$defs"]["_Candidate"]["properties"].items() if k != "code"
    }
    sent = json.loads(encode_request(request(json_schema=schema)))["response_format"]["json_schema"]["schema"]
    assert sent == lines


@pytest.mark.parametrize(
    "schema",
    [
        Verdict.model_json_schema(),
        {"type": "object", "properties": {"code": {"type": "integer"}}},  # not a string: not source code
        {"type": "object", "properties": {"code": {"type": "string"}, "code_lines": {"type": "string"}}},
    ],
)
def test_a_schema_without_a_code_string_is_sent_as_is(schema: dict[str, Any]) -> None:
    assert lines_schema(schema) is None
    assert json.loads(encode_request(request(json_schema=schema)))["response_format"]["json_schema"]["schema"] == schema


def test_the_lines_are_joined_back_before_the_caller_sees_the_answer(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    lines = ["import pytest", "", "def test_add():", "    with pytest.raises(TypeError):", "        add('a', 1)"]
    server.replies.append(answer(json.dumps({"tests": [{"path": "tests/test_a.py", "code_lines": lines, "why": "w"}]})))
    result = complete_structured(make(), request(json_schema=Candidates.model_json_schema()), Candidates)
    assert isinstance(result, Parsed)
    (test,) = result.value.tests
    assert test.code == "\n".join(lines) + "\n" and test.path == "tests/test_a.py" and test.why == "w"
    assert len(server.requests) == 1


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ({"code_lines": []}, {"code": ""}),
        ({"a": [{"code_lines": ["x"]}], "b": 1}, {"a": [{"code": "x\n"}], "b": 1}),
        ({"code_lines": ["x", 2]}, {"code_lines": ["x", 2]}),  # not all strings: validation refuses it
        ({"code": "y", "code_lines": ["x"]}, {"code": "y", "code_lines": ["x"]}),  # both: left for validation
    ],
)
def test_joined_code(value: object, expected: object) -> None:
    assert joined_code(value) == expected


def test_an_answer_that_still_sends_one_code_string_is_validated_as_sent(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    server.replies.append(answer(json.dumps({"tests": [{"path": "p", "code": "import x\n"}]})))
    result = complete_structured(make(), request(json_schema=Candidates.model_json_schema()), Candidates)
    assert isinstance(result, Parsed) and result.value.tests[0].code == "import x\n"


def test_the_agent_loop_write_stage_gets_its_file_content_as_lines_too(
    make: Callable[..., GlimmerClient], server: Server
) -> None:
    lines = lines_schema(WRITE_SCHEMA)
    assert lines is not None
    item = lines["properties"]["tests"]["items"]
    assert list(item["properties"]) == ["path", "content_lines", "quote"]
    assert item["required"] == ["path", "content_lines", "quote"]
    assert WRITE_SCHEMA["properties"]["tests"]["items"]["properties"]["content"] == {"type": "string"}  # untouched
    body = ["def test_x():", "    assert f() == 1"]
    server.replies.append(
        answer(json.dumps({"tests": [{"path": "tests/test_x.py", "content_lines": body, "quote": ""}]}))
    )
    response = make().complete(request(json_schema=WRITE_SCHEMA))
    expected = {"tests": [{"path": "tests/test_x.py", "content": "def test_x():\n    assert f() == 1\n", "quote": ""}]}
    assert response.parsed == expected and json.loads(response.text) == expected
    sent = json.loads(server.requests[0].content)["response_format"]["json_schema"]["schema"]
    assert sent == lines


def test_a_rewritten_property_is_a_fresh_object_each_time() -> None:
    first, second = lines_schema(WRITE_SCHEMA), lines_schema(candidates_schema())
    assert first is not None and second is not None
    a = first["properties"]["tests"]["items"]["properties"]["content_lines"]
    b = second["$defs"]["_Candidate"]["properties"]["code_lines"]
    assert a == b and a is not b


def test_joined_text_keeps_ascii_escapes(make: Callable[..., GlimmerClient], server: Server) -> None:
    server.replies.append(answer('{"tests": [{"path": "p", "code_lines": ["s = \'\\ud800\'"]}]}'))
    response = make().complete(request(json_schema=Candidates.model_json_schema()))
    response.text.encode("utf-8")  # a lone surrogate stays escaped, so the text is still encodable


def test_exactly_the_two_source_file_schemas_are_rewritten() -> None:
    """Pins SOURCE_FIELDS to the schemas that carry a whole file: a new schema with a prose "code" or "content"
    property must be added here on purpose, or the rewrite would turn its text into lines."""
    from receipts.attack.regression import _Verdicts
    from receipts.core.intake import ExtractionOutput
    from receipts.core.planner import PlanOutput
    from receipts.core.refuter import RefuterOutput

    assert lines_schema(candidates_schema()) is not None
    assert lines_schema(WRITE_SCHEMA) is not None
    for model in (_Verdicts, ExtractionOutput, PlanOutput, RefuterOutput, Verdict):
        assert lines_schema(model.model_json_schema()) is None, model.__name__
