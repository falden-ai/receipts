"""The ticket entry point, using synthetic ledgers and HTTP fakes. No test reaches a service."""

from __future__ import annotations

import argparse
import json
import os
import unicodedata
from collections.abc import Callable, Iterator, Mapping
from pathlib import Path

import httpx
import pytest

from receipts.contracts import (
    CatchKind,
    FailureType,
    Finding,
    LedgerEntry,
    Lens,
    Outcome,
    ReplayKey,
    Repro,
    RunResult,
    Side,
    StageCost,
    Status,
)
from receipts.surface import ticket_command as command
from receipts.surface.ledger import JsonlLedger
from receipts.surface.text import HIDDEN
from receipts.surface.tickets import RateLimitedError, TrackerError, draft_ticket, receipt_url
from receipts.surface.trackers import LINEAR_API, LinearHTTP

FID = "t1/fixed_empty/a1"
TEAM = "11111111-1111-4111-8111-111111111111"
LABEL = "22222222-2222-4222-8222-222222222222"
OTHER_LABEL = "33333333-3333-4333-8333-333333333333"
PROJECT = "44444444-4444-4444-8444-444444444444"
BASE = "https://receipts.example.test/r"
ORG = "receipts-test"
KEY = "test-ticket-command-sentinel-not-a-real-key"
ASSET = "https://uploads.linear.app/receipts/repro.txt"
IDENTIFIER = "RCP-42"
ISSUE_URL = "https://linear.app/receipts-test/issue/RCP-42"
ORDER = ["ReceiptTicketPreflight", "ReceiptTicket", "ReceiptUpload", "PUT", "ReceiptIssue", "ReceiptAttach"]
HOSTILE = (
    "![x](https://evil.example/a.png?d=secret) [click](http://evil.example) javascript:alert(1) "
    "evil.example/x www.evil.example \N{RIGHT-TO-LEFT OVERRIDE}\N{ZERO WIDTH SPACE} "
    "\x1b]52;c;ZXZpbA==\x07\r\t\n```\n````\n"
)


def entry(*, text: str = "", status: Status = Status.BLOCKER, micro_usd: int = 123_456) -> LedgerEntry:
    repro = Repro(
        id="r1",
        attack_id="a1",
        claim_id="fixed_empty",
        lens=Lens.CORRECTNESS,
        kind=CatchKind.FIX_VALIDITY,
        files={"tests/test_empty.py": f"def test_empty():\n    assert first([]) is None\n{text}"},
        command=("python", "-m", "pytest", "tests/test_empty.py"),
    )
    fail = RunResult(
        side=Side.PATCH,
        outcome=Outcome.FAIL,
        failure_type=FailureType.ASSERTION,
        fresh_container=True,
        exit_code=1,
        duration_s=1.0,
        stdout_tail=f"AssertionError\n{text}",
    )
    passed = RunResult(side=Side.SURROGATE, outcome=Outcome.PASS, fresh_container=True, exit_code=0, duration_s=1.0)
    finding = Finding(
        id=FID,
        target_id="t1",
        claim_id="fixed_empty",
        lens=Lens.CORRECTNESS,
        status=status,
        kind=CatchKind.FIX_VALIDITY,
        repro=repro,
        runs=(fail,) * 5 + (passed,),
        quote=f"Empty input returns None. {text}",
        detail=f"The patch fails five times. {text}",
        replay=ReplayKey(base_sha="a" * 40, patch_sha="b" * 40, harness_hash="h1", seed=7, image_digest="sha256:d"),
        costs=(StageCost(stage="attack", model="muse-spark-1.3", micro_usd=micro_usd),),
    )
    return LedgerEntry(at="2026-10-30T12:00:00Z", receipts_version="0.0.1", finding=finding)


def ledger_with(path: Path, *entries: LedgerEntry) -> Path:
    ledger = JsonlLedger(path)
    for row in entries:
        ledger.append(row)
    return path


def arguments(path: Path, *options: str, finding_id: str = FID) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="receipts")
    command.register(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(
        [
            "ticket",
            finding_id,
            "--ledger",
            str(path),
            "--team-id",
            TEAM,
            "--receipt-base",
            BASE,
            "--expect-org",
            ORG,
            *options,
        ]
    )


def call(path: Path, *options: str, finding_id: str = FID) -> int:
    args = arguments(path, *options, finding_id=finding_id)
    code: int = args.func(args)
    return code


class RecordingEnvironment(Mapping[str, str]):
    def __init__(self, values: Mapping[str, str]) -> None:
        self.values = values
        self.reads: list[str] = []

    def __getitem__(self, name: str) -> str:
        self.reads.append(name)
        return self.values[name]

    def __iter__(self) -> Iterator[str]:
        self.reads.append("<iteration>")
        return iter(self.values)

    def __len__(self) -> int:
        self.reads.append("<length>")
        return len(self.values)


class NeverTransport:
    def __init__(self, key: str) -> None:
        pytest.fail("a dry run or refused command constructed a transport")

    def graphql(self, query: str, variables: Mapping[str, object]) -> Mapping[str, object]:
        pytest.fail("a dry run or refused command made a GraphQL call")

    def upload(self, url: str, headers: Mapping[str, str], content: bytes) -> None:
        pytest.fail("a dry run or refused command uploaded a file")


@pytest.fixture(autouse=True)
def refuse_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(command, "LinearHTTP", NeverTransport)


class FakeLinear:
    def __init__(self) -> None:
        self.calls: list[str] = []
        self.variables: dict[str, dict[str, object]] = {}
        self.requests: list[httpx.Request] = []
        self.overrides: dict[str, httpx.Response] = {}
        self.transport: LinearHTTP | None = None

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        if request.method == "PUT":
            name = "PUT"
            assert "Authorization" not in request.headers
            assert request.headers["Content-Type"] == "text/plain; charset=utf-8"
            assert b"def test_empty" in request.content
            data: dict[str, object] = {}
        else:
            assert str(request.url) == LINEAR_API
            assert request.method == "POST"
            assert request.headers["Authorization"] == KEY
            body = json.loads(request.content)
            name = body["query"].split("(", 1)[0].split()[-1]
            self.variables[name] = body["variables"]
            issue_id = body["variables"].get("input", {}).get("id", "issue-42")
            issue = {"id": issue_id, "identifier": IDENTIFIER, "url": ISSUE_URL}
            answers: dict[str, dict[str, object]] = {
                "ReceiptTicketPreflight": {
                    "viewer": {"organization": {"urlKey": ORG}},
                    "team": {"id": TEAM, "key": "RCP", "name": "Receipts"},
                },
                "ReceiptTicket": {"attachmentsForURL": {"nodes": []}},
                "ReceiptUpload": {
                    "fileUpload": {
                        "success": True,
                        "uploadFile": {
                            "uploadUrl": "https://uploads.linear.app/signed",
                            "assetUrl": ASSET,
                            "headers": [],
                        },
                    }
                },
                "ReceiptIssue": {"issueCreate": {"success": True, "issue": issue}},
                "ReceiptIssueById": {"issue": issue},
                "ReceiptAttach": {"attachmentCreate": {"success": True, "attachment": {"id": "attachment-42"}}},
            }
            data = answers[name]
        self.calls.append(name)
        return self.overrides.get(name, httpx.Response(200, json={"data": data}))

    def input_for(self, operation: str) -> dict[str, object]:
        value = self.variables[operation]["input"]
        assert isinstance(value, dict)
        return value


@pytest.fixture
def linear(monkeypatch: pytest.MonkeyPatch) -> Iterator[FakeLinear]:
    fake = FakeLinear()
    monkeypatch.setenv(command.KEY_NAME, KEY)
    with httpx.Client(transport=httpx.MockTransport(fake.handle)) as client:
        transport = LinearHTTP(KEY, client=client)
        fake.transport = transport

        def build(key: str) -> LinearHTTP:
            assert key == KEY
            return transport

        monkeypatch.setattr(command, "LinearHTTP", build)
        yield fake


@pytest.mark.parametrize("as_json", [False, True])
@pytest.mark.parametrize("values", [{}, {command.KEY_NAME: KEY}])
def test_dry_run_reads_no_environment_or_transport(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    as_json: bool,
    values: dict[str, str],
) -> None:
    row = entry()
    path = ledger_with(tmp_path / "ledger.jsonl", row)
    args = arguments(path, *(["--json"] if as_json else []))
    environment = RecordingEnvironment(values)
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", environment)
        assert args.func(args) == 0
    assert environment.reads == []
    captured = capsys.readouterr()
    assert captured.err == "" and KEY not in captured.out
    draft = draft_ticket(row, receipt_base=BASE, spent_micro_usd=123_456)
    if as_json:
        assert json.loads(captured.out) == draft.model_dump(mode="json")
    else:
        for value in (draft.title, draft.body, draft.receipt_url, "Spend: $0.123456", "attributes:"):
            assert value in captured.out


@pytest.mark.parametrize(
    "values",
    [
        {"LINEAR_API_KEY": KEY, "FALDEN_LINEAR_API_KEY": KEY},
        {command.KEY_NAME: "", "LINEAR_API_KEY": KEY, "FALDEN_LINEAR_API_KEY": KEY},
    ],
)
def test_post_reads_only_the_receipts_key_and_refuses_missing_or_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    values: dict[str, str],
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    args = arguments(path, "--post")
    environment = RecordingEnvironment(values)
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", environment)
        assert args.func(args) == 2
    assert environment.reads == [command.KEY_NAME]
    captured = capsys.readouterr()
    assert command.KEY_NAME in captured.err and captured.out == ""
    assert KEY not in captured.err


@pytest.mark.parametrize("post", [False, True])
@pytest.mark.parametrize("status", [Status.SUSPECTED, Status.FLAKY, Status.REFUTED])
def test_non_blocker_uses_writer_refusal_without_a_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    status: Status,
    post: bool,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry(status=status))
    assert call(path, *(["--post"] if post else [])) == 3
    captured = capsys.readouterr()
    assert "only a blocker becomes a ticket" in captured.err and status.value in captured.err
    assert captured.out == ""


def test_unknown_finding_is_refused_without_a_call(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    assert call(path, "--post", finding_id="t1/other/a2") == 3
    assert "no finding t1/other/a2" in capsys.readouterr().err


@pytest.mark.parametrize(
    "base",
    [
        "http://example.test/r",
        "/r",
        "https:///r",
        "https://u:p@example.test/r",
        "https://example.test/r?q=x",
        "https://example.test/r#x",
    ],
)
def test_bad_receipt_base_is_refused_before_a_call(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], base: str
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    assert call(path, "--post", "--receipt-base", base) == 2
    assert "receipt base URL" in capsys.readouterr().err


@pytest.mark.parametrize(
    "options",
    [
        ["--priority", "-1"],
        ["--priority", "5"],
        ["--team-id", "bad"],
        ["--label-id", "bad"],
        ["--project-id", "bad"],
    ],
)
def test_bad_options_are_parser_errors(tmp_path: Path, options: list[str]) -> None:
    with pytest.raises(SystemExit) as caught:
        call(tmp_path / "unused.jsonl", *options)
    assert caught.value.code == 2


def test_missing_ledger_is_not_created(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "missing" / "ledger.jsonl"
    assert call(path) == 1
    assert not path.parent.exists()
    assert str(path) in capsys.readouterr().err


def test_invalid_ledger_is_reported_without_terminal_controls(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "bad\N{RIGHT-TO-LEFT OVERRIDE}.jsonl"
    path.write_bytes(b"not json\n")
    assert call(path) == 1
    captured = capsys.readouterr()
    assert "\\u202e" in captured.err and "\N{RIGHT-TO-LEFT OVERRIDE}" not in captured.err
    assert path.read_bytes() == b"not json\n"


@pytest.mark.parametrize("post", [False, True])
def test_torn_tail_is_reported_and_left_in_place(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    post: bool,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    original = path.read_bytes() + b'{"v":3,"at":"unfinished'
    path.write_bytes(original)
    assert call(path, *(["--post"] if post else [])) == 0
    captured = capsys.readouterr()
    assert "torn last line" in captured.err and "line stays in the file" in captured.err
    assert path.read_bytes() == original and not path.with_suffix(".jsonl.torn").exists()
    assert linear.calls == (ORDER if post else [])


@pytest.mark.parametrize(
    ("data", "exit_code", "message"),
    [
        (
            {"viewer": {"organization": {"urlKey": "falden"}}, "team": {"id": TEAM, "key": "FAL", "name": "Falden"}},
            3,
            "expected workspace",
        ),
        ({"viewer": {"organization": {"urlKey": ORG}}, "team": None}, 3, "team is missing"),
        ({"viewer": {"organization": {"urlKey": ORG}}}, 3, "team is missing"),
        (
            {"viewer": {"organization": {"urlKey": ORG}}, "team": {"id": PROJECT, "key": "X", "name": "Other"}},
            3,
            "team is missing",
        ),
        ({"viewer": None, "team": None}, 1, "expected shape"),
        ({"viewer": {"organization": {"urlKey": 123}}, "team": None}, 1, "expected shape"),
        ({"viewer": {"organization": {"urlKey": ORG}}, "team": {"id": TEAM}}, 1, "expected shape"),
    ],
)
def test_preflight_fails_closed_after_exactly_one_call(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    data: dict[str, object],
    exit_code: int,
    message: str,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    linear.overrides["ReceiptTicketPreflight"] = httpx.Response(200, json={"data": data})
    assert call(path, "--post") == exit_code
    assert linear.calls == ["ReceiptTicketPreflight"]
    captured = capsys.readouterr()
    assert message in captured.err and captured.out == ""
    assert KEY not in captured.err


@pytest.mark.parametrize("as_json", [False, True])
def test_post_passes_options_and_uses_latest_entry_and_total_spend(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    as_json: bool,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry(text="old detail"), entry(text="latest detail"))
    options = [
        "--post",
        "--label-id",
        LABEL,
        OTHER_LABEL,
        "--label-id",
        LABEL,
        "--project-id",
        PROJECT,
        "--priority",
        "2",
        *(["--json"] if as_json else []),
    ]
    assert call(path, *options) == 0
    assert linear.calls == ORDER
    assert linear.variables["ReceiptTicketPreflight"] == {"team": TEAM}
    query = json.loads(linear.requests[0].content)["query"]
    for expected in ("$team: String!", "viewer { organization { urlKey } }", "team(id: $team) { id key name }"):
        assert expected in query
    issue = linear.input_for("ReceiptIssue")
    assert (issue["teamId"], issue["labelIds"], issue["projectId"], issue["priority"]) == (
        TEAM,
        [LABEL, OTHER_LABEL, LABEL],
        PROJECT,
        2,
    )
    assert "latest detail" in str(issue["description"]) and "old detail" not in str(issue["description"])
    metadata = linear.input_for("ReceiptAttach")["metadata"]
    assert isinstance(metadata, dict)
    assert {"name": "Spend", "value": "$0.246912"} in metadata["attributes"]
    captured = capsys.readouterr()
    assert captured.err == "" and KEY not in captured.out
    if as_json:
        result = json.loads(captured.out)
        assert (result["outcome"], result["issue_identifier"], result["issue_url"], result["calls"]) == (
            "created",
            IDENTIFIER,
            ISSUE_URL,
            5,
        )
    else:
        assert f"created {IDENTIFIER} {ISSUE_URL} calls=5 (plus 1 preflight)" in captured.out


def test_existing_ticket_makes_no_new_issue_or_upload(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    linear.overrides["ReceiptTicket"] = httpx.Response(
        200,
        json={
            "data": {
                "attachmentsForURL": {
                    "nodes": [
                        {"id": "attachment", "issue": {"id": "existing", "identifier": IDENTIFIER, "url": ISSUE_URL}}
                    ]
                }
            }
        },
    )
    assert call(path, "--post") == 0
    assert linear.calls == ORDER[:2]
    captured = capsys.readouterr()
    assert f"existing {IDENTIFIER}" in captured.out and "calls=1" in captured.out and captured.err == ""


def test_recovered_ticket_prints_the_writer_outcome(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    linear.overrides["ReceiptIssue"] = httpx.Response(200, json={"errors": [{"message": "entity already exists"}]})
    assert call(path, "--post") == 0
    assert linear.calls == [*ORDER[:5], "ReceiptIssueById", "ReceiptAttach"]
    assert f"recovered {IDENTIFIER}" in capsys.readouterr().out


def test_rate_limit_during_recovery_is_not_masked_by_the_original_create_error(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    linear.overrides["ReceiptIssue"] = httpx.Response(200, json={"errors": [{"message": "entity already exists"}]})
    linear.overrides["ReceiptIssueById"] = httpx.Response(429, headers={"Retry-After": "7"})
    assert call(path, "--post") == 1
    assert linear.calls == [*ORDER[:5], "ReceiptIssueById"]
    captured = capsys.readouterr()
    assert "retry_after=7.0" in captured.err and "entity already exists" not in captured.err
    assert captured.out == "" and KEY not in captured.err


@pytest.mark.parametrize("as_json", [False, True])
def test_upload_error_reflecting_the_key_is_redacted_in_the_result(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    as_json: bool,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    linear.overrides["ReceiptUpload"] = httpx.Response(200, json={"errors": [{"message": f"upload failed {KEY}"}]})
    assert call(path, "--post", *(["--json"] if as_json else [])) == 0
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err and "[redacted]" in captured.out
    assert linear.calls == [*ORDER[:3], *ORDER[4:]]
    if as_json:
        assert "[redacted]" in json.loads(captured.out)["upload_error"]


@pytest.mark.parametrize("as_json", [False, True])
def test_upload_failure_is_printed_but_the_ticket_is_still_filed(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    as_json: bool,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    linear.overrides["PUT"] = httpx.Response(503, text=KEY)
    assert call(path, "--post", *(["--json"] if as_json else [])) == 0
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err
    assert "the upload answered HTTP 503" in captured.out
    assert linear.calls == ORDER
    assert ASSET not in str(linear.input_for("ReceiptIssue")["description"])
    if as_json:
        assert json.loads(captured.out)["upload_error"] == "the upload answered HTTP 503"
    else:
        assert "upload error:" in captured.out


@pytest.mark.parametrize("operation", ORDER)
@pytest.mark.parametrize("retry_after", ["7", None])
def test_rate_limit_stops_all_further_calls_and_reports_retry_after(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    operation: str,
    retry_after: str | None,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    headers = {"Retry-After": retry_after} if retry_after else {}
    linear.overrides[operation] = httpx.Response(429, headers=headers)
    if operation == "PUT":
        # LinearHTTP reports a raw upload HTTP 429 as TrackerError. Exercise a transport's RateLimitedError here.
        assert linear.transport is not None

        def limited_upload(url: str, headers: Mapping[str, str], content: bytes) -> None:
            linear.calls.append("PUT")
            raise RateLimitedError("Linear rate limited the upload", float(retry_after) if retry_after else None)

        monkeypatch.setattr(linear.transport, "upload", limited_upload)
    assert call(path, "--post") == 1
    assert linear.calls == ORDER[: ORDER.index(operation) + 1]
    captured = capsys.readouterr()
    assert f"retry_after={'7.0' if retry_after else 'None'}" in captured.err
    assert captured.out == "" and KEY not in captured.err


@pytest.mark.parametrize("as_json", [False, True])
def test_r16_canary_is_inert_in_the_posted_body_and_terminal(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    rendered: Callable[[str], str],
    as_json: bool,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry(text=HOSTILE))
    assert call(path, *(["--json"] if as_json else [])) == 0
    dry = capsys.readouterr()
    assert linear.calls == []
    for ch in dry.out + dry.err:
        assert ch == "\n" or unicodedata.category(ch) not in HIDDEN
    assert "\\u202e" in dry.out and "\\u200b" in dry.out
    assert call(path, "--post") == 0
    body = str(linear.input_for("ReceiptIssue")["description"])
    assert "\\u202e" in body and "\\u200b" in body
    assert "\N{RIGHT-TO-LEFT OVERRIDE}" not in body and "\N{ZERO WIDTH SPACE}" not in body
    outside = rendered(body).replace(receipt_url(BASE, FID), "<trusted receipt>").replace(ASSET, "<trusted upload>")
    for forbidden in ("http", "www.", "javascript:", "evil.example", "![x]", "[click]"):
        assert forbidden not in outside.lower()
    assert KEY not in capsys.readouterr().out


@pytest.mark.parametrize("operation", ["ReceiptTicketPreflight", "ReceiptTicket", "ReceiptAttach"])
@pytest.mark.parametrize("graphql_error", [False, True])
def test_failing_http_and_reflected_errors_never_print_the_key(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    linear: FakeLinear,
    operation: str,
    graphql_error: bool,
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    linear.overrides[operation] = (
        httpx.Response(200, json={"errors": [{"message": f"refused {KEY}\x1b[2J"}]})
        if graphql_error
        else httpx.Response(503, text=KEY)
    )
    assert call(path, "--post") == 1
    captured = capsys.readouterr()
    assert KEY not in captured.out + captured.err + repr(linear.transport)
    assert captured.out == "" and "\x1b" not in captured.err
    if graphql_error:
        assert "[redacted]" in captured.err
    else:
        assert linear.transport is not None
        with pytest.raises(TrackerError) as caught:
            linear.transport.graphql(
                command.PREFLIGHT
                if operation == "ReceiptTicketPreflight"
                else {
                    "ReceiptTicket": "query ReceiptTicket($url: String!) { x }",
                    "ReceiptAttach": "mutation ReceiptAttach($input: AttachmentCreateInput!) { x }",
                }[operation],
                {},
            )
        assert KEY not in str(caught.value) + repr(caught.value)


def test_connection_failure_with_key_in_underlying_error_is_handled(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    monkeypatch.setenv(command.KEY_NAME, KEY)

    def fail(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError(f"failed {KEY}", request=request)

    with httpx.Client(transport=httpx.MockTransport(fail)) as client:
        transport = LinearHTTP(KEY, client=client)
        monkeypatch.setattr(command, "LinearHTTP", lambda key: transport)
        assert call(path, "--post") == 1
    captured = capsys.readouterr()
    assert "ConnectError" in captured.err and KEY not in captured.out + captured.err + repr(transport)


def test_invalid_key_is_reported_without_its_value(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", entry())
    monkeypatch.setenv(command.KEY_NAME, KEY + " has whitespace")
    monkeypatch.setattr(command, "LinearHTTP", LinearHTTP)
    assert call(path, "--post") == 2
    captured = capsys.readouterr()
    assert "no whitespace" in captured.err and KEY not in captured.out + captured.err
