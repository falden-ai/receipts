"""TK1, the ticket writer: each blocker becomes one Linear issue, or one GitHub issue, deduped by a stable receipt URL.

What it reads. Only a ledger entry, revalidated the way a ledger reader validates it (`ledger.validated`), so the
writer sees schema-validated fields and nothing else: never a pull request body, a diff or raw tracker data. Only a
blocker becomes a ticket (spec section 0: every blocker is a runnable counterexample that survived the skeptic).

What it posts (R16). Model and repository text, which here is the quoted claim sentence, the repro's files, command
and cases, a run's output and problem, the finding's detail and a refutation, goes out only inside fenced code blocks
and code spans, and everything posted has its links and images stripped and its hidden characters escaped
(`markdown`). Every limit counts characters as they are shown, after escaping, so a blocker of any content fits in one
ticket. Titles and attribute values are built from enums, counts and ids, as plain text. The uploaded repro file is
the repro with the same stripping, served as text/plain; the exact repro stays in the ledger, behind `receipts replay
<finding id>`.

Why it is a blocker (contract v3). The finding's detail, its reason in plain words, is shown as Receipts recorded it;
a refutation is shown only as what it is, a different model's refutation, which did not hold up against the blocker.
A finding with both shows both.

How it dedupes. A finding's receipt URL is a base URL plus its id, so one finding always maps to one URL.

- Linear: `attachmentsForURL(receipt URL)` finds a ticket already filed. A new one takes 5 calls: the lookup,
  `fileUpload`, the upload's PUT, `issueCreate`, and `attachmentCreate`, which attaches the receipt URL with metadata
  attributes that Linear lists in the attachment. The issue's id is a UUID derived from the receipt URL, so a writer
  that crashed between the create and the attach recovers on its next run: the create fails as a duplicate,
  `issue(id)` finds the issue (the 6th call), and the attach completes it. An upload that fails leaves the issue
  without the file link, never without the ticket.
- GitHub: one search for the receipt key in issue bodies, then one create. GitHub Issues have no upload API, so the
  repro is in the body. Only a repository named in `post_repos` is ever posted to; any other gets a draft for a person
  to file (spec hard rule 8: nothing is auto-posted to third-party repositories). Search can lag a new issue by about
  a minute, so filing one finding twice inside that window can duplicate it; the pipeline files each finding once.

Every ticket makes at most 6 calls (`MAX_CALLS`); a 7th is refused before it is sent. Nothing here reads a token: the
transports take theirs from the caller.
"""

from __future__ import annotations

import hashlib
import re
import shlex
import uuid
from collections.abc import Mapping, Sequence
from typing import Literal, Protocol
from urllib.parse import quote, urlsplit

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from receipts.contracts import (
    COUNTED_FAILURES,
    CatchKind,
    Finding,
    Frozen,
    Ledger,
    LedgerEntry,
    Outcome,
    Repro,
    Side,
    Status,
)
from receipts.surface.ledger import usd, validated
from receipts.surface.markdown import code_block, code_span, inline, plain, strip_links
from receipts.surface.text import visible

MAX_CALLS = 6  # spec TK1: at most 6 calls per ticket
BODY_LIMIT = 60_000  # GitHub refuses an issue body over 65,536 characters
# Characters as shown, after escaping. Together they keep the body under BODY_LIMIT whatever a blocker holds.
FILES_LIMIT = 30_000  # repro files shown in one ticket, headings included; files past it are named, not shown
FILE_LIMIT = 8_000  # one repro file shown
FILES_NAMED = 10  # files past FILES_LIMIT named by path; the rest are counted
PATH_LIMIT = 300  # one path, id or case shown in a code span
QUOTE_LIMIT = 2_000  # the quoted sentence, the detail and the refutation, each
COMMAND_LIMIT = 2_000
CASES_LIMIT = 2_000
OUTPUT_LIMIT = 3_000  # each output stream of the first failing patch run
OUTPUT_LINES = 40
RUNS_SHOWN = 20  # runs listed in the runs table; the rest are counted
PROBLEM_LIMIT = 200  # one run's problem in the runs table
REPRO_CONTENT_TYPE = "text/plain; charset=utf-8"  # a browser shows it as text and never renders it

Tracker = Literal["linear", "github"]

KIND_PHRASES = {
    CatchKind.REGRESSION: "regression catch",
    CatchKind.FIX_VALIDITY: "fix-validity catch",
    CatchKind.BEHAVIOR: "behavior catch",
}
_REPO = re.compile(r"^[A-Za-z0-9][A-Za-z0-9-]{0,38}/[A-Za-z0-9._-]{1,100}$")
# An uploaded file's asset URL goes into the description as a link, so it must be Linear's own storage and plain.
_LINEAR_ASSET = re.compile(r"^https://(?:[A-Za-z0-9-]+\.)*linear\.app/[^\s()<>\[\]\\]*$")


# ---------------------------------------------------------------- errors


class TicketError(Exception):
    """A ticket that could not be filed."""


class TicketRefusedError(TicketError):
    """An entry that never becomes a ticket: not a blocker."""


class CallBudgetError(TicketError):
    """A ticket that would need more than MAX_CALLS calls."""


class TrackerError(TicketError):
    """A tracker call that failed: an HTTP error, a GraphQL error or an answer of the wrong shape. Its message never
    holds a token."""


class RateLimitedError(TrackerError):
    """The tracker said to slow down; `retry_after` is its wait in seconds, when it gave one."""

    def __init__(self, message: str, retry_after: float | None = None) -> None:
        super().__init__(message)
        self.retry_after = retry_after


# ---------------------------------------------------------------- what a ticket says


class TicketAttribute(Frozen):
    name: str = Field(min_length=1)
    value: str


class TicketDraft(Frozen):
    """Everything a ticket posts, from one validated ledger entry. `body` is markdown in which every piece of model
    text sits inside a code block or a code span; `repro_text` is the file a tracker with uploads gets."""

    finding_id: str
    receipt_url: str
    receipt_key: str  # a search token unique to the receipt URL, for a tracker that finds tickets by searching
    kind: str  # "regression catch", "fix-validity catch" or "behavior catch"
    patch_runs: str  # "5 of 5 runs": how often the repro failed on the patch, on a counted failure
    title: str
    body: str
    attributes: tuple[TicketAttribute, ...]
    repro_name: str
    repro_text: str


class TicketResult(Frozen):
    tracker: Tracker
    outcome: Literal["created", "existing", "recovered", "draft"]
    finding_id: str
    receipt_url: str
    issue_id: str | None = None
    issue_identifier: str | None = None  # LIN-123, or #45 on GitHub
    issue_url: str | None = None
    calls: int = Field(ge=0, le=MAX_CALLS)
    upload_error: str | None = None
    draft: TicketDraft | None = None  # set for a draft, for which nothing was posted


def latest_entry(ledger: Ledger, finding_id: str) -> LedgerEntry:
    """A finding's latest entry, its current state, read through the ledger (which validates every line it reads):
    what a writer files for a finding."""
    target_id = finding_id.split("/", 1)[0]
    rows = [e for e in ledger.entries(target_id) if e.finding.id == finding_id]
    if not rows:
        raise TicketRefusedError(f"no finding {visible(finding_id)} in the ledger")
    return rows[-1]


def finding_spend(ledger: Ledger, finding_id: str) -> int:
    """What a finding cost, in micro-dollars: every one of its entries' costs, since each entry records only what
    producing it spent. The ticket's Spend attribute is this number."""
    target_id = finding_id.split("/", 1)[0]
    return sum(c.micro_usd for e in ledger.entries(target_id) if e.finding.id == finding_id for c in e.finding.costs)


def receipt_url(base: str, finding_id: str) -> str:
    """The finding's stable receipt URL: `base` (https, with a host and no query, fragment or user) plus the finding
    id, each of its three segments percent-encoded on its own, so an id can never climb out of the base path."""
    parts = urlsplit(base)
    if parts.scheme != "https" or not parts.hostname or parts.query or parts.fragment or "@" in parts.netloc:
        raise ValueError(f"a receipt base URL is https with a host and no query, fragment or user: {base!r}")
    segments = finding_id.split("/", 2)
    if len(segments) != 3 or not all(segments):
        raise ValueError(f"a finding id is <target_id>/<claim_id>/<attack or lens>: {finding_id!r}")
    return base.rstrip("/") + "/" + "/".join(_url_segment(s) for s in segments)


def receipt_key(url: str) -> str:
    """One search token for a receipt URL, letters and digits only, so a tracker's search reads it as one word."""
    return "faldenreceipt" + hashlib.sha256(url.encode("utf-8")).hexdigest()[:20]


def issue_uuid(url: str) -> str:
    """The Linear issue id for a receipt URL: a UUID in version 4 format, derived from the URL rather than random, so
    a create retried after a crash names the same issue."""
    raw = bytearray(hashlib.sha256(b"falden-receipts/linear-issue/" + url.encode("utf-8")).digest()[:16])
    raw[6] = (raw[6] & 0x0F) | 0x40
    raw[8] = (raw[8] & 0x3F) | 0x80
    return str(uuid.UUID(bytes=bytes(raw)))


def draft_ticket(
    entry: LedgerEntry,
    *,
    receipt_base: str,
    repro_link: str | None = None,
    list_attributes: bool = False,
    spent_micro_usd: int | None = None,
) -> TicketDraft:
    """The ticket for one blocker, from its revalidated ledger entry; anything but a blocker is refused. `repro_link`
    is an uploaded repro's URL, from the tracker's own storage; `list_attributes` adds the attributes to the body, for
    a tracker that has no attachment metadata. `spent_micro_usd` is what the finding cost over all of its entries
    (`finding_spend`); without it the ticket names no spend, since one entry holds only its own step's costs."""
    entry = validated(entry)
    finding = entry.finding
    if finding.status is not Status.BLOCKER or finding.repro is None or finding.kind is None:
        raise TicketRefusedError(f"only a blocker becomes a ticket; {visible(finding.id)} is {finding.status.value}")
    if spent_micro_usd is not None and spent_micro_usd < 0:
        raise ValueError("a finding's spend is not negative")
    repro = finding.repro
    url = receipt_url(receipt_base, finding.id)
    key = receipt_key(url)
    kind = KIND_PHRASES[finding.kind]
    patch = [r for r in finding.runs if r.side is Side.PATCH]
    failed = [r for r in patch if r.outcome is Outcome.FAIL and r.failure_type in COUNTED_FAILURES]
    patch_runs = f"{len(failed)} of {len(patch)} runs"
    passes = _passes(finding)
    failures = sorted({r.failure_type.value.replace("_", " ") for r in failed if r.failure_type is not None})
    attributes = [
        TicketAttribute(name="Status", value=finding.status.value),
        TicketAttribute(name="Kind", value=kind),
        TicketAttribute(name="Claim", value=plain(finding.claim_id)),
        TicketAttribute(name="Lens", value=finding.lens.value),
        TicketAttribute(name="Target", value=plain(finding.target_id)),
        TicketAttribute(name="Finding", value=plain(finding.id)),
        TicketAttribute(name="Fails on the patch", value=patch_runs),
        TicketAttribute(name="Passes", value=passes),
        TicketAttribute(name="Replay seed", value=str(finding.replay.seed) if finding.replay else "none"),
    ]
    if spent_micro_usd is not None:
        attributes.append(TicketAttribute(name="Spend", value=usd(spent_micro_usd)))
    attributes.append(TicketAttribute(name="Receipts version", value=plain(entry.receipts_version)))
    lines = [
        f"**Falden Receipts found a {kind}** for the claim {code_span(finding.claim_id, limit=PATH_LIMIT)} of the "
        f"target {code_span(finding.target_id, limit=PATH_LIMIT)}.",
        "",
        f"The repro fails on the agent's change in {patch_runs} ({', '.join(failures)}), and passes {passes}.",
        "",
        f"- Finding: {code_span(finding.id, limit=PATH_LIMIT)}",
        f"- Lens: {finding.lens.value}",
        f"- Receipt: <{url}>",  # an autolink, so a trailing dot or tilde in the id stays part of the link
    ]
    if repro_link is not None:
        lines.append(f"- Repro file: [{_repro_name(key)}]({repro_link})")
    lines.append(f"- Replay: {code_span('receipts replay ' + finding.id, limit=PATH_LIMIT + 20)}")
    if list_attributes:
        listed = [f"- {a.name}: {code_span(a.value, limit=PATH_LIMIT)}" for a in attributes]
        lines += ["", "**Attributes**", "", *listed]
    lines += [
        "",
        "**The sentence it breaks**, quoted from the claim or the issue:",
        "",
        code_block(finding.quote or "", "text", limit=QUOTE_LIMIT, where="the full sentence is in the ledger"),
    ]
    if finding.detail:
        lines += [
            "",
            "**Why it is a blocker**, as Receipts recorded it:",
            "",
            code_block(finding.detail, "text", limit=QUOTE_LIMIT, where="the full text is in the ledger"),
        ]
    if finding.refutation:
        lines += [
            "",
            "**Different-model refutation**: what a different model offered against it, which did not hold up:",
            "",
            code_block(finding.refutation, "text", limit=QUOTE_LIMIT, where="the full text is in the ledger"),
        ]
    lines += [
        "",
        "**Repro command**",
        "",
        code_block(shlex.join(repro.command), "text", limit=COMMAND_LIMIT, where="the full command is in the ledger"),
    ]
    if repro.cases:
        lines += [
            "",
            "**Cases it runs**, as its runner names them:",
            "",
            code_block("\n".join(repro.cases), "text", limit=CASES_LIMIT, where="every case is in the ledger"),
        ]
    lines += _files_lines(repro)
    lines += _runs_lines(finding)
    if failed:
        for stream, text in (("stdout", failed[0].stdout_tail), ("stderr", failed[0].stderr_tail)):
            if not text.strip():
                continue  # Django's runner prints its failures on stderr, pytest's on stdout
            tail = "\n".join(text.rstrip("\n").split("\n")[-OUTPUT_LINES:])
            lines += [
                "",
                f"**Output of the first failing patch run**, its {stream}, the last {OUTPUT_LINES} lines at most:",
                "",
                code_block(tail, "text", limit=OUTPUT_LIMIT, where="the run's output is in the ledger"),
            ]
    if finding.replay is not None:
        r = finding.replay
        lines += [
            "",
            f"**Replay key**: base {code_span(r.base_sha)}, patch {code_span(r.patch_sha)}, seed {r.seed}, harness "
            f"{code_span(r.harness_hash, limit=PATH_LIMIT)}, image {code_span(r.image_digest, limit=PATH_LIMIT)}",
        ]
    lines += [
        "",
        "---",
        "Model-written text is shown as code with its links and images removed; the exact repro is in the ledger "
        f"(`receipts replay`). Receipt key: {key}",
    ]
    body = "\n".join(lines)
    if len(body) > BODY_LIMIT:
        raise TicketError(f"the ticket for {visible(finding.id)} is {len(body):,} characters, over {BODY_LIMIT:,}")
    return TicketDraft(
        finding_id=finding.id,
        receipt_url=url,
        receipt_key=key,
        kind=kind,
        patch_runs=patch_runs,
        title=plain(f"Receipts: {kind} on {finding.claim_id.partition(':')[0]} in {finding.target_id}", limit=200),
        body=body,
        attributes=tuple(attributes),
        repro_name=_repro_name(key),
        repro_text=_repro_text(finding.id, repro),
    )


def _passes(finding: Finding) -> str:
    if finding.kind is CatchKind.REGRESSION:
        base = [r for r in finding.runs if r.side is Side.BASE]
        passed = sum(1 for r in base if r.outcome is Outcome.PASS)
        return f"on the code before the change in {passed} of {len(base)} runs"
    fixes = sum(1 for r in finding.runs if r.side in (Side.SURROGATE, Side.GOLD) and r.outcome is Outcome.PASS)
    return f"on {fixes} plausible independent fix{'' if fixes == 1 else 'es'}"


def _files_lines(repro: Repro) -> list[str]:
    """Each repro file under its path, until FILES_LIMIT characters are shown; the files after that are named."""
    lines: list[str] = []
    shown = 0
    files = sorted(repro.files.items())
    for i, (path, content) in enumerate(files):
        room = min(FILE_LIMIT, FILES_LIMIT - shown)
        if room <= 0:
            rest = files[i:]
            named = ", ".join(code_span(p, limit=PATH_LIMIT) for p, _ in rest[:FILES_NAMED])
            more = f", and {len(rest) - FILES_NAMED:,} more" if len(rest) > FILES_NAMED else ""
            lines += ["", f"**More repro files**, shown in the ledger only: {named}{more}"]
            break
        heading = f"**Repro file** {code_span(path, limit=PATH_LIMIT)}"
        block = code_block(content, _language(path), limit=room, where="the full file is in the ledger")
        shown += len(heading) + len(block)
        lines += ["", heading, "", block]
    return lines


def _runs_lines(finding: Finding) -> list[str]:
    """The runs as a table inside a code block, so a run's problem, which can quote the repository's own output, is
    inert text beside the outcome it explains."""
    header = ("#", "side", "outcome", "failure", "exit", "fresh", "seconds", "problem")
    rows = [
        (
            str(n),
            r.side.value,
            r.outcome.value,
            r.failure_type.value if r.failure_type else "",
            "" if r.exit_code is None else str(r.exit_code),
            "yes" if r.fresh_container else "no",
            f"{r.duration_s:.1f}",
            inline(r.problem, PROBLEM_LIMIT),
        )
        for n, r in enumerate(finding.runs[:RUNS_SHOWN], start=1)
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    table = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in (header, *rows)]
    if len(finding.runs) > RUNS_SHOWN:
        table.append(f"... and {len(finding.runs) - RUNS_SHOWN:,} more runs, in the ledger")
    heading = "**Runs**, with why a run could not be read where one could not:"
    return ["", heading, "", code_block("\n".join(table), "text")]


def _repro_text(finding_id: str, repro: Repro) -> str:
    """The uploaded copy: every line through the same stripping and escaping as the body."""
    safe_id = visible(strip_links(finding_id))
    lines = [
        f"# Falden Receipts repro for finding {safe_id}",
        "# Links are removed from this copy and hidden characters escaped; the exact repro is in the ledger:",
        f"#   receipts replay {safe_id}",
        f"# command: {visible(strip_links(shlex.join(repro.command)))}",
    ]
    lines += [f"# case: {visible(strip_links(case))}" for case in repro.cases]
    for path, content in sorted(repro.files.items()):
        lines += [f"# ---- file {visible(strip_links(path))}", visible(strip_links(content), keep="\t\n")]
    return "\n".join(lines) + "\n"


def _repro_name(key: str) -> str:
    return f"receipts-repro-{key[-12:]}.txt"


def _language(path: str) -> str:
    return "python" if path.endswith(".py") else "text"


def _url_segment(segment: str) -> str:
    encoded = quote(segment, safe="")
    return encoded.replace(".", "%2E") if set(segment) == {"."} else encoded


# ---------------------------------------------------------------- calls


class CallBudget:
    """Counts a ticket's calls and refuses the one past the limit before it is sent."""

    def __init__(self, limit: int = MAX_CALLS) -> None:
        self.limit = limit
        self.used = 0

    def spend(self, what: str) -> None:
        if self.used >= self.limit:
            raise CallBudgetError(f"a ticket makes at most {self.limit} calls; {what} would be call {self.used + 1}")
        self.used += 1


class LinearTransport(Protocol):
    def graphql(self, query: str, variables: Mapping[str, object]) -> Mapping[str, object]:
        """The response's `data`. Raises TrackerError for an HTTP error or any GraphQL error."""
        ...

    def upload(self, url: str, headers: Mapping[str, str], content: bytes) -> None:
        """PUT the bytes to a pre-signed upload URL, without the API key. Raises TrackerError on failure."""
        ...


class GitHubTransport(Protocol):
    def request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> object:
        """The decoded JSON answer. Raises TrackerError for an HTTP error."""
        ...


class _Answer(BaseModel):
    """A tracker's answer, validated at the boundary; fields this writer does not read are ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True)


class _LinearIssue(_Answer):
    id: str = Field(min_length=1)
    identifier: str = Field(min_length=1)
    url: str | None = None


class _LinearAttachmentNode(_Answer):
    id: str
    issue: _LinearIssue | None = None


class _LinearNodes(_Answer):
    nodes: list[_LinearAttachmentNode]


class _AttachmentsForUrl(_Answer):
    attachmentsForURL: _LinearNodes


class _UploadHeader(_Answer):
    key: str
    value: str


class _UploadFile(_Answer):
    uploadUrl: str
    assetUrl: str
    headers: list[_UploadHeader] = []


class _UploadPayload(_Answer):
    success: bool
    uploadFile: _UploadFile | None = None


class _FileUpload(_Answer):
    fileUpload: _UploadPayload


class _IssuePayload(_Answer):
    success: bool
    issue: _LinearIssue | None = None


class _IssueCreate(_Answer):
    issueCreate: _IssuePayload


class _IssueById(_Answer):
    issue: _LinearIssue


class _AttachmentPayload(_Answer):
    success: bool


class _AttachmentCreate(_Answer):
    attachmentCreate: _AttachmentPayload


# The GraphQL documents are constants; every value travels in `variables`, so no ledger text is spliced into a query.
ATTACHMENTS_FOR_URL = """
query ReceiptTicket($url: String!) {
  attachmentsForURL(url: $url, first: 5) { nodes { id issue { id identifier url } } }
}
"""
FILE_UPLOAD = """
mutation ReceiptUpload($contentType: String!, $filename: String!, $size: Int!) {
  fileUpload(contentType: $contentType, filename: $filename, size: $size) {
    success
    uploadFile { uploadUrl assetUrl headers { key value } }
  }
}
"""
ISSUE_CREATE = """
mutation ReceiptIssue($input: IssueCreateInput!) {
  issueCreate(input: $input) { success issue { id identifier url } }
}
"""
ISSUE_BY_ID = """
query ReceiptIssueById($id: String!) {
  issue(id: $id) { id identifier url }
}
"""
ATTACHMENT_CREATE = """
mutation ReceiptAttach($input: AttachmentCreateInput!) {
  attachmentCreate(input: $input) { success attachment { id } }
}
"""


def _parse[A: _Answer](model: type[A], data: object, what: str) -> A:
    try:
        return model.model_validate(data)
    except ValidationError as err:
        raise TrackerError(f"{what}: the answer is not the shape this writer expects") from err


# ---------------------------------------------------------------- Linear


class LinearTicketWriter:
    """Files blockers as Linear issues in one team."""

    def __init__(
        self,
        transport: LinearTransport,
        *,
        team_id: str,
        receipt_base: str,
        label_ids: Sequence[str] = (),
        project_id: str | None = None,
        priority: int | None = None,
    ) -> None:
        if not team_id:
            raise ValueError("a Linear ticket goes to a team")
        if priority is not None and not 0 <= priority <= 4:
            raise ValueError("Linear priorities are 0 (none) to 4 (low)")
        receipt_url(receipt_base, "t/c/a")  # refuse a bad base now, not at the first ticket
        self._transport = transport
        self._team_id = team_id
        self._receipt_base = receipt_base
        self._label_ids = tuple(label_ids)
        self._project_id = project_id
        self._priority = priority

    def file(self, entry: LedgerEntry, *, spent_micro_usd: int | None = None) -> TicketResult:
        """File the blocker `entry` is the latest state of; `spent_micro_usd` is the finding's whole spend
        (`finding_spend`), shown as its Spend attribute."""
        draft = draft_ticket(entry, receipt_base=self._receipt_base, spent_micro_usd=spent_micro_usd)
        calls = CallBudget()
        found = self._find(calls, draft.receipt_url)
        if found is not None:
            return self._result(draft, calls, "existing", found)
        asset_url, upload_error = self._upload(calls, draft)
        if asset_url is not None:
            draft = draft_ticket(
                entry, receipt_base=self._receipt_base, repro_link=asset_url, spent_micro_usd=spent_micro_usd
            )
        issue_id = issue_uuid(draft.receipt_url)
        outcome: Literal["created", "recovered"] = "created"
        try:
            issue = self._create(calls, draft, issue_id)
        except TrackerError as create_error:
            # A crash after an earlier create and before its attach leaves an issue under this id: finish that one.
            try:
                issue = self._issue(calls, issue_id)
            except TrackerError:
                raise create_error from None
            outcome = "recovered"
        self._attach(calls, draft, issue.id)
        return self._result(draft, calls, outcome, issue, upload_error)

    def _find(self, calls: CallBudget, url: str) -> _LinearIssue | None:
        calls.spend("attachmentsForURL")
        answer = _parse(_AttachmentsForUrl, self._transport.graphql(ATTACHMENTS_FOR_URL, {"url": url}), "lookup")
        issues = [node.issue for node in answer.attachmentsForURL.nodes if node.issue is not None]
        return issues[0] if issues else None

    def _upload(self, calls: CallBudget, draft: TicketDraft) -> tuple[str | None, str | None]:
        """The uploaded repro's asset URL, or None and the reason the upload failed."""
        content = draft.repro_text.encode("utf-8")
        try:
            calls.spend("fileUpload")
            variables = {"contentType": REPRO_CONTENT_TYPE, "filename": draft.repro_name, "size": len(content)}
            payload = _parse(_FileUpload, self._transport.graphql(FILE_UPLOAD, variables), "fileUpload").fileUpload
            if not payload.success or payload.uploadFile is None:
                raise TrackerError("fileUpload: Linear could not prepare the upload")
            target = payload.uploadFile
            headers = {"Content-Type": REPRO_CONTENT_TYPE, "Cache-Control": "public, max-age=31536000"}
            headers |= {h.key: h.value for h in target.headers}
            if not _LINEAR_ASSET.fullmatch(target.assetUrl):
                raise TrackerError("fileUpload: the asset URL is not Linear's own storage, so it is not linked")
            calls.spend("the upload's PUT")
            self._transport.upload(target.uploadUrl, headers, content)
        except TrackerError as err:
            return None, visible(str(err))
        return target.assetUrl, None

    def _create(self, calls: CallBudget, draft: TicketDraft, issue_id: str) -> _LinearIssue:
        fields: dict[str, object] = {
            "id": issue_id,
            "teamId": self._team_id,
            "title": draft.title,
            "description": draft.body,
        }
        if self._label_ids:
            fields["labelIds"] = list(self._label_ids)
        if self._project_id:
            fields["projectId"] = self._project_id
        if self._priority is not None:
            fields["priority"] = self._priority
        calls.spend("issueCreate")
        payload = _parse(_IssueCreate, self._transport.graphql(ISSUE_CREATE, {"input": fields}), "issueCreate")
        if not payload.issueCreate.success or payload.issueCreate.issue is None:
            raise TrackerError("issueCreate: Linear did not create the issue")
        return payload.issueCreate.issue

    def _issue(self, calls: CallBudget, issue_id: str) -> _LinearIssue:
        calls.spend("issue")
        return _parse(_IssueById, self._transport.graphql(ISSUE_BY_ID, {"id": issue_id}), "issue").issue

    def _attach(self, calls: CallBudget, draft: TicketDraft, issue_id: str) -> None:
        summary = f"A {draft.kind}: it fails on the patch in {draft.patch_runs}."
        metadata = {
            "title": "Falden Receipts",
            "attributes": [a.model_dump() for a in draft.attributes],
            "messages": [{"subject": "Receipt", "body": summary}],
        }
        variables = {
            "input": {
                "issueId": issue_id,
                "url": draft.receipt_url,
                "title": "Falden Receipts",
                "subtitle": f"{draft.kind}, fails on the patch in {draft.patch_runs}",
                "metadata": metadata,
            }
        }
        calls.spend("attachmentCreate")
        payload = _parse(_AttachmentCreate, self._transport.graphql(ATTACHMENT_CREATE, variables), "attachmentCreate")
        if not payload.attachmentCreate.success:
            raise TrackerError("attachmentCreate: Linear did not attach the receipt")

    @staticmethod
    def _result(
        draft: TicketDraft,
        calls: CallBudget,
        outcome: Literal["created", "existing", "recovered"],
        issue: _LinearIssue,
        upload_error: str | None = None,
    ) -> TicketResult:
        return TicketResult(
            tracker="linear",
            outcome=outcome,
            finding_id=draft.finding_id,
            receipt_url=draft.receipt_url,
            issue_id=issue.id,
            issue_identifier=issue.identifier,
            issue_url=issue.url,
            calls=calls.used,
            upload_error=upload_error,
        )


# ---------------------------------------------------------------- GitHub


class _GitHubIssue(_Answer):
    number: int
    html_url: str
    node_id: str | None = None
    body: str | None = None


class _GitHubSearch(_Answer):
    total_count: int
    items: list[_GitHubIssue]


class GitHubIssueWriter:
    """Files blockers as issues in one repository, and only in a repository named in `post_repos`; for any other the
    ticket comes back as a draft that nothing was posted for. The default posts nowhere."""

    def __init__(
        self,
        transport: GitHubTransport,
        *,
        repo: str,
        receipt_base: str,
        post_repos: frozenset[str] = frozenset(),
        labels: Sequence[str] = (),
    ) -> None:
        for name in (repo, *post_repos):
            if not _REPO.fullmatch(name) or name.split("/")[1] in (".", ".."):
                raise ValueError(f"a repository is owner/name: {name!r}")
        receipt_url(receipt_base, "t/c/a")
        self._transport = transport
        self._repo = repo
        self._receipt_base = receipt_base
        self._post = repo.lower() in {r.lower() for r in post_repos}  # GitHub names ignore case
        self._labels = [plain(label, limit=50) for label in labels]

    def file(self, entry: LedgerEntry, *, spent_micro_usd: int | None = None) -> TicketResult:
        """File the blocker `entry` is the latest state of, or draft it for a repository not in `post_repos`."""
        draft = draft_ticket(
            entry, receipt_base=self._receipt_base, list_attributes=True, spent_micro_usd=spent_micro_usd
        )
        if not self._post:
            return TicketResult(
                tracker="github",
                outcome="draft",
                finding_id=draft.finding_id,
                receipt_url=draft.receipt_url,
                calls=0,
                draft=draft,
            )
        calls = CallBudget()
        found = self._find(calls, draft)
        if found is not None:
            return self._result(draft, calls, "existing", found)
        body: dict[str, object] = {"title": draft.title, "body": draft.body}
        if self._labels:
            body["labels"] = list(self._labels)
        calls.spend("create")
        issue = _parse(
            _GitHubIssue, self._transport.request("POST", f"/repos/{self._repo}/issues", body=body), "create"
        )
        return self._result(draft, calls, "created", issue)

    def _find(self, calls: CallBudget, draft: TicketDraft) -> _GitHubIssue | None:
        calls.spend("search")
        params = {"q": f"{draft.receipt_key} repo:{self._repo} is:issue in:body", "per_page": "20"}
        found = _parse(_GitHubSearch, self._transport.request("GET", "/search/issues", params=params), "search")
        # Search matches loosely; only an issue whose body holds this exact key and receipt URL is this ticket.
        mine = [i for i in found.items if i.body and draft.receipt_key in i.body and draft.receipt_url in i.body]
        return mine[0] if mine else None

    @staticmethod
    def _result(
        draft: TicketDraft, calls: CallBudget, outcome: Literal["created", "existing"], issue: _GitHubIssue
    ) -> TicketResult:
        return TicketResult(
            tracker="github",
            outcome=outcome,
            finding_id=draft.finding_id,
            receipt_url=draft.receipt_url,
            issue_id=issue.node_id,
            issue_identifier=f"#{issue.number}",
            issue_url=issue.html_url,
            calls=calls.used,
        )
