"""The downloader refuses every AWS endpoint but the one approved bucket (redirects included), sends that bucket only
plain anonymous GETs, and refuses any byte that is not the pinned one. No network: every response comes from an
httpx.MockTransport."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest

from receipts.eval.hashing import sha256_bytes
from receipts.eval.sources import (
    APPROVED_AWS_HOSTS,
    SOURCES,
    FetchRefusedError,
    IntegrityError,
    Source,
    check_url,
    fetch,
    make_client,
    verify_file,
)

BODY = b"pinned bytes\n"
SRC = Source(
    name="test:file",
    url="https://example.org/data/file.json",
    path="test/file.json",
    revision="abcdef1",
    sha256=sha256_bytes(BODY),
    size=len(BODY),
)


def test_every_pinned_source_is_https_and_off_aws() -> None:
    assert len({s.name for s in SOURCES}) == len(SOURCES) == len({s.path for s in SOURCES})
    for s in SOURCES:
        check_url(s.url)


@pytest.mark.parametrize(
    "url",
    [
        "http://example.org/x",
        "http://swe-bench-submissions.s3.amazonaws.com/verified/x/logs/a/patch.diff",
        "https://s3.amazonaws.com/swe-bench-submissions/verified/x/logs/a/patch.diff",  # path-style: not approved
        "https://other-bucket.s3.amazonaws.com/key",
        "https://s3.us-east-1.amazonaws.com/bucket/key",
        "https://pricing.us-east-1.amazonaws.com/offers/v1.0/aws/index.json",
        "https://ec2.us-east-1.amazonaws.com/?Action=DescribeInstances",
        "https://AMAZONAWS.COM./x",
        "https://user:pass@swe-bench-submissions.s3.amazonaws.com/k",
    ],
)
def test_plant_aws_endpoints_and_plain_http_are_refused(url: str) -> None:
    with pytest.raises(FetchRefusedError):
        check_url(url)


def test_a_cdn_that_runs_on_aws_is_not_an_aws_endpoint() -> None:
    check_url("https://us.aws.cdn.hf.co/xet-bridge-us/abc")
    check_url("https://b0.p.awsstatic.com/pricing/2.0/x.json")


def test_the_approved_bucket_is_the_one_aws_host_let_through() -> None:
    assert {"swe-bench-submissions.s3.amazonaws.com"} == APPROVED_AWS_HOSTS
    assert check_url("https://swe-bench-submissions.s3.amazonaws.com/verified/e/logs/i/patch.diff")
    assert check_url("https://SWE-BENCH-SUBMISSIONS.s3.amazonaws.com./k")


def _bucket_client(sent: list[httpx.Request]) -> httpx.Client:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(200, content=b"ok")

    return make_client(transport=httpx.MockTransport(handler))


def test_the_bucket_gets_plain_anonymous_gets() -> None:
    sent: list[httpx.Request] = []
    with _bucket_client(sent) as client:
        assert client.get("https://swe-bench-submissions.s3.amazonaws.com/k").content == b"ok"
    assert len(sent) == 1 and "authorization" not in sent[0].headers and sent[0].method == "GET"


@pytest.mark.parametrize(
    ("method", "headers"),
    [
        ("PUT", {}),
        ("POST", {}),
        ("DELETE", {}),
        ("GET", {"Authorization": "AWS4-HMAC-SHA256 Credential=x"}),
        ("GET", {"X-Amz-Security-Token": "t"}),
        ("GET", {"Cookie": "c=1"}),
    ],
)
def test_plant_a_write_or_a_credential_to_the_bucket_is_refused_before_it_is_sent(
    method: str, headers: dict[str, str]
) -> None:
    sent: list[httpx.Request] = []
    with _bucket_client(sent) as client, pytest.raises(FetchRefusedError):
        client.request(method, "https://swe-bench-submissions.s3.amazonaws.com/k", headers=headers)
    assert sent == []


def test_the_client_ignores_proxy_and_netrc_settings_from_the_environment() -> None:
    with make_client() as client:
        assert client.trust_env is False


@pytest.mark.parametrize(
    "target", ["https://evil.s3.amazonaws.com/x", "https://s3.amazonaws.com/swe-bench-submissions/x"]
)
def test_plant_a_redirect_to_another_aws_endpoint_is_refused_before_it_is_followed(target: str) -> None:
    hits: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        hits.append(str(request.url))
        return httpx.Response(302, headers={"location": target})

    with make_client(transport=httpx.MockTransport(handler)) as client, pytest.raises(FetchRefusedError):
        client.get("https://swe-bench-submissions.s3.amazonaws.com/start")
    assert hits == ["https://swe-bench-submissions.s3.amazonaws.com/start"]


def test_fetch_writes_only_the_pinned_bytes(tmp_path: Path) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request.url)
        return httpx.Response(200, content=BODY)

    with make_client(transport=httpx.MockTransport(handler)) as client:
        path = fetch(SRC, tmp_path, client)
        assert path.read_bytes() == BODY
        fetch(SRC, tmp_path, client)  # already there and right: no second download
    assert len(calls) == 1


def test_plant_wrong_bytes_from_upstream_are_refused(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"pinned bytez\n")

    with make_client(transport=httpx.MockTransport(handler)) as client, pytest.raises(IntegrityError, match="sha256"):
        fetch(SRC, tmp_path, client)
    assert not (tmp_path / SRC.path).exists()


def test_plant_a_tampered_file_on_disk_is_caught(tmp_path: Path) -> None:
    target = tmp_path / SRC.path
    target.parent.mkdir(parents=True)
    target.write_bytes(BODY)
    verify_file(SRC, tmp_path)
    target.write_bytes(BODY.replace(b"p", b"q", 1))
    with pytest.raises(IntegrityError):
        verify_file(SRC, tmp_path)
    target.unlink()
    with pytest.raises(IntegrityError, match="missing"):
        verify_file(SRC, tmp_path)
