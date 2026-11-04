"""The HTTP transports, driven by httpx.MockTransport: nothing reaches the network. Each test checks a request's shape
or a refusal, and that the token never leaves the tracker's own API host."""

from __future__ import annotations

import json
import logging

import httpx
import pytest

from receipts.surface.tickets import RateLimitedError, TrackerError
from receipts.surface.trackers import GITHUB_API, LINEAR_API, GitHubHTTP, LinearHTTP

KEY = "test-key-not-real-0001"  # a placeholder, never a real token
UPLOAD = "https://uploads.linear.app/signed/abc?sig=1"


def mock(*answers: httpx.Response) -> tuple[httpx.Client, list[httpx.Request]]:
    seen: list[httpx.Request] = []
    queue = list(answers)

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return queue.pop(0)

    return httpx.Client(transport=httpx.MockTransport(handler)), seen


def linear(
    *answers: httpx.Response, client_headers: dict[str, str] | None = None
) -> tuple[LinearHTTP, list[httpx.Request]]:
    client, seen = mock(*answers)
    if client_headers:
        client.headers.update(client_headers)
    return LinearHTTP(KEY, client=client), seen


# ---------------------------------------------------------------- Linear


def test_linear_posts_graphql_with_the_key_and_returns_data() -> None:
    transport, seen = linear(httpx.Response(200, json={"data": {"ok": 1}}))
    assert transport.graphql("query Q($u: String!) { x }", {"u": "v"}) == {"ok": 1}
    [request] = seen
    assert (request.method, str(request.url)) == ("POST", LINEAR_API)
    assert request.headers["Authorization"] == KEY
    assert json.loads(request.content) == {"query": "query Q($u: String!) { x }", "variables": {"u": "v"}}


@pytest.mark.parametrize(
    ("answer", "error", "message"),
    [
        (httpx.Response(200, json={"errors": [{"message": "Entity not found"}]}), TrackerError, "Entity not found"),
        (
            httpx.Response(400, json={"errors": [{"message": "bad\x1b[2J input", "extensions": {"code": "X"}}]}),
            TrackerError,
            "bad\\\\u001b\\[2J input",
        ),
        (
            httpx.Response(200, json={"errors": [{"message": "slow down", "extensions": {"code": "RATELIMITED"}}]}),
            RateLimitedError,
            "rate limited",
        ),
        (httpx.Response(429, headers={"Retry-After": "7"}, text=""), RateLimitedError, "rate limited"),
        (httpx.Response(500, text="oops"), TrackerError, "HTTP 500"),
        (httpx.Response(200, text="<html>"), TrackerError, "not JSON"),
        (httpx.Response(200, json={"data": None}), TrackerError, "no data"),
        (httpx.Response(302, headers={"Location": "https://evil.example/graphql"}), TrackerError, "HTTP 302"),
    ],
    ids=["graphql error", "escaped error", "RATELIMITED", "HTTP 429", "HTTP 500", "not json", "no data", "redirect"],
)
def test_linear_errors_are_tracker_errors_without_the_key(
    answer: httpx.Response, error: type[TrackerError], message: str
) -> None:
    transport, seen = linear(answer)
    with pytest.raises(error, match=message) as caught:
        transport.graphql("query Q { x }", {})
    assert KEY not in str(caught.value) and KEY not in repr(caught.value)
    assert len(seen) == 1  # a redirect is never followed
    if answer.status_code == 429:
        assert isinstance(caught.value, RateLimitedError) and caught.value.retry_after == 7.0


def test_a_linear_connection_error_is_a_tracker_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    transport = LinearHTTP(KEY, client=httpx.Client(transport=httpx.MockTransport(handler)))
    with pytest.raises(TrackerError, match="ConnectError") as caught:
        transport.graphql("query Q { x }", {})
    assert KEY not in str(caught.value)


def test_the_upload_puts_the_bytes_and_never_the_key() -> None:
    # Even a client that carries an Authorization header by default sends none to the pre-signed URL.
    transport, seen = linear(httpx.Response(200), client_headers={"Authorization": KEY})
    transport.upload(UPLOAD, {"Content-Type": "text/plain; charset=utf-8", "x-goog-meta": "1"}, b"repro bytes")
    [request] = seen
    assert (request.method, str(request.url), request.content) == ("PUT", UPLOAD, b"repro bytes")
    assert request.headers["x-goog-meta"] == "1" and request.headers["Content-Type"] == "text/plain; charset=utf-8"
    assert "authorization" not in {k.lower() for k in request.headers}
    assert KEY not in json.dumps(dict(request.headers))


@pytest.mark.parametrize(
    "url",
    [
        "http://uploads.linear.app/x",
        "https://evil.example/x",
        "https://linear.app.evil.example/x",
        "https://evillinear.app/x",
        "https://user:pw@uploads.linear.app/x",
        "file:///etc/passwd",
    ],
)
def test_an_upload_anywhere_else_is_refused_before_it_is_sent(url: str) -> None:
    transport, seen = linear(httpx.Response(200))
    with pytest.raises(TrackerError, match="not one this writer uploads to"):
        transport.upload(url, {}, b"x")
    assert seen == []


def test_a_failed_upload_is_a_tracker_error() -> None:
    transport, _ = linear(httpx.Response(403, text="SignatureDoesNotMatch"))
    with pytest.raises(TrackerError, match="HTTP 403"):
        transport.upload(UPLOAD, {}, b"x")


@pytest.mark.parametrize("where", ["https://evil.example/collect", "https://uploads.linear.app/elsewhere"])
def test_the_upload_never_follows_a_redirect(where: str) -> None:
    # A redirect would carry the repro's bytes to wherever the answer points, past the upload host check.
    transport, seen = linear(httpx.Response(307, headers={"Location": where}), httpx.Response(200))
    with pytest.raises(TrackerError, match="HTTP 307"):
        transport.upload(UPLOAD, {"Content-Type": "text/plain; charset=utf-8"}, b"repro bytes")
    assert [str(r.url) for r in seen] == [UPLOAD]


def test_the_key_stays_out_of_reprs_and_logs(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.DEBUG):
        transport, _ = linear(httpx.Response(200, json={"data": {}}), httpx.Response(500))
        transport.graphql("query Q { x }", {})
        with pytest.raises(TrackerError):
            transport.graphql("query Q { x }", {})
        github = GitHubHTTP(KEY, client=mock(httpx.Response(200, json={}))[0])
        github.request("GET", "/x")
    assert KEY not in repr(transport) and KEY not in repr(github)
    assert KEY not in caplog.text
    for bad in ("", "two words", "line\nbreak"):
        with pytest.raises(ValueError, match="no whitespace"):
            LinearHTTP(bad)


# ---------------------------------------------------------------- GitHub


def test_github_sends_a_bearer_token_and_json() -> None:
    client, seen = mock(httpx.Response(201, json={"number": 3}))
    answer = GitHubHTTP(KEY, client=client).request("POST", "/repos/falden/demo/issues", body={"title": "t"})
    assert answer == {"number": 3}
    [request] = seen
    assert (request.method, str(request.url)) == ("POST", f"{GITHUB_API}/repos/falden/demo/issues")
    assert request.headers["Authorization"] == f"Bearer {KEY}"
    assert request.headers["Accept"] == "application/vnd.github+json"
    assert request.headers["X-GitHub-Api-Version"] == "2022-11-28"
    assert json.loads(request.content) == {"title": "t"}
    client, seen = mock(httpx.Response(200, json={"items": []}))
    GitHubHTTP(KEY, client=client).request("GET", "/search/issues", params={"q": "k repo:a/b"})
    assert seen[0].url.params["q"] == "k repo:a/b"


@pytest.mark.parametrize(
    ("answer", "error", "message"),
    [
        (httpx.Response(403, headers={"x-ratelimit-remaining": "0"}, json={}), RateLimitedError, "rate limited"),
        (httpx.Response(429, headers={"Retry-After": "60"}, json={}), RateLimitedError, "rate limited"),
        (
            httpx.Response(404, json={"message": "Not Found\x1b]52;c;x\x07"}),
            TrackerError,
            "HTTP 404: Not Found\\\\u001b",
        ),
        (httpx.Response(301, headers={"Location": "https://evil.example/"}), TrackerError, "HTTP 301"),
        (httpx.Response(200, text="not json"), TrackerError, "not JSON"),
    ],
    ids=["403 rate limit", "429", "404", "redirect", "not json"],
)
def test_github_errors_are_tracker_errors(answer: httpx.Response, error: type[TrackerError], message: str) -> None:
    client, seen = mock(answer)
    with pytest.raises(error, match=message) as caught:
        GitHubHTTP(KEY, client=client).request("GET", "/repos/a/b")
    assert KEY not in str(caught.value) and len(seen) == 1


@pytest.mark.parametrize("path", ["repos/a/b", "//evil.example/x", "/repos/../user", "/repos/./a", "/a?b=c", "/a//b"])
def test_github_refuses_a_path_that_is_not_plain(path: str) -> None:
    client, seen = mock()
    with pytest.raises(ValueError, match="plain segments"):
        GitHubHTTP(KEY, client=client).request("GET", path)
    assert seen == []
    with pytest.raises(ValueError, match="https"):
        GitHubHTTP(KEY, api="http://api.github.com")
