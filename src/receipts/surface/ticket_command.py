"""One blocker as a Linear ticket. Drafts read no key; posting checks the workspace before the writer runs."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections.abc import Mapping
from pathlib import Path
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from receipts.surface.ledger import JsonlLedger, LedgerError
from receipts.surface.text import visible
from receipts.surface.tickets import (
    LinearTicketWriter,
    LinearTransport,
    RateLimitedError,
    TicketDraft,
    TicketRefusedError,
    TicketResult,
    TrackerError,
    draft_ticket,
    finding_spend,
    latest_entry,
)
from receipts.surface.trackers import LinearHTTP

KEY_NAME = "RECEIPTS_LINEAR_API_KEY"
PREFLIGHT = """
query ReceiptTicketPreflight($team: String!) {
  viewer { organization { urlKey } }
  team(id: $team) { id key name }
}
"""


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Register the command without opening a ledger or constructing a transport."""
    parser = subparsers.add_parser("ticket", help="draft one blocker as a Linear ticket; --post files it")
    parser.add_argument("finding_id", help="the finding id in the ledger")
    parser.add_argument("--ledger", type=Path, required=True, help="the ledger file, read without changes")
    parser.add_argument("--team-id", type=_uuid, required=True, help="the Receipts Linear team's UUID")
    parser.add_argument(
        "--receipt-base", required=True, help="the stable HTTPS receipt base URL used for deduplication"
    )
    parser.add_argument("--expect-org", required=True, help="the expected Receipts workspace URL key")
    parser.add_argument("--label-id", type=_uuid, action="extend", nargs="+", default=[], help="Linear label UUIDs")
    parser.add_argument("--project-id", type=_uuid, help="the Linear project's UUID")
    parser.add_argument("--priority", type=int, choices=range(5), help="Linear priority, 0 (none) to 4 (low)")
    parser.add_argument("--post", action="store_true", help=f"file the ticket using {KEY_NAME}")
    parser.add_argument("--json", action="store_true", help="print the draft or result as JSON")
    parser.set_defaults(func=ticket_command)


def _uuid(value: str) -> str:
    try:
        return str(UUID(value))
    except ValueError:
        raise argparse.ArgumentTypeError("expected a UUID") from None


class _Answer(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True, strict=True)


class _Organization(_Answer):
    urlKey: str = Field(min_length=1)


class _Viewer(_Answer):
    organization: _Organization


class _Team(_Answer):
    id: str = Field(min_length=1)
    key: str = Field(min_length=1)
    name: str = Field(min_length=1)


class _Preflight(_Answer):
    viewer: _Viewer
    team: _Team | None = None


class _PostingTransport:
    """Stop sending calls after a rate limit, including writer upload and create recovery paths."""

    def __init__(self, transport: LinearTransport) -> None:
        self.transport = transport
        self.limited: RateLimitedError | None = None

    def graphql(self, query: str, variables: Mapping[str, object]) -> Mapping[str, object]:
        if self.limited is not None:
            raise self.limited
        try:
            return self.transport.graphql(query, variables)
        except RateLimitedError as err:
            self.limited = err
            raise

    def upload(self, url: str, headers: Mapping[str, str], content: bytes) -> None:
        if self.limited is not None:
            raise self.limited
        try:
            self.transport.upload(url, headers, content)
        except RateLimitedError as err:
            self.limited = err
            raise


def _preflight(transport: LinearTransport, team_id: str, expect_org: str) -> None:
    data = transport.graphql(PREFLIGHT, {"team": team_id})
    try:
        answer = _Preflight.model_validate(data)
    except ValidationError:
        raise TrackerError("Linear preflight: the answer is not the expected shape") from None
    actual = answer.viewer.organization.urlKey
    if actual != expect_org:
        raise TicketRefusedError(f"Linear preflight: expected workspace {expect_org!r}, got {actual!r}; nothing filed")
    if answer.team is None or answer.team.id != team_id:
        raise TicketRefusedError("Linear preflight: the requested team is missing; nothing filed")


def _emit(text: str, *, error: bool = False, key: str | None = None) -> None:
    if key:
        text = text.replace(key, "[redacted]")
        text = text.replace(json.dumps(key, ensure_ascii=True)[1:-1], "[redacted]")
    for line in text.split("\n"):
        print(visible(line, keep=""), file=sys.stderr if error else sys.stdout)


def _print_draft(draft: TicketDraft, *, as_json: bool) -> None:
    if as_json:
        _emit(json.dumps(draft.model_dump(mode="json"), indent=2, ensure_ascii=True))
        return
    lines = [f"title: {draft.title}", "attributes:"]
    lines.extend(f"  {attribute.name}: {attribute.value}" for attribute in draft.attributes)
    lines.extend([f"receipt URL: {draft.receipt_url}", "", draft.body])
    _emit("\n".join(lines))


def _print_result(result: TicketResult, *, as_json: bool, key: str) -> None:
    if as_json:
        _emit(json.dumps(result.model_dump(mode="json"), indent=2, ensure_ascii=True), key=key)
        return
    lines = [
        f"{result.outcome} {result.issue_identifier or '(no identifier)'} {result.issue_url or '(no issue URL)'} "
        f"calls={result.calls} (plus 1 preflight)"
    ]
    if result.upload_error is not None:
        lines.append(f"upload error: {result.upload_error}")
    _emit("\n".join(lines), key=key)


def ticket_command(args: argparse.Namespace) -> int:
    """Draft by default; --post checks the workspace and delegates filing to the existing writer."""
    ledger: JsonlLedger | None = None
    try:
        ledger = JsonlLedger(args.ledger, create=False, read_only=True)
        entry = latest_entry(ledger, args.finding_id)
        spent = finding_spend(ledger, args.finding_id)
        draft = draft_ticket(entry, receipt_base=args.receipt_base, spent_micro_usd=spent)
    except TicketRefusedError as err:
        _emit(f"receipts ticket: {err}", error=True)
        return 3
    except (LedgerError, OSError) as err:
        _emit(f"receipts ticket: {err}", error=True)
        return 1
    except ValueError as err:
        _emit(f"receipts ticket: {err}", error=True)
        return 2
    finally:
        if ledger is not None:
            for torn in ledger.torn_tails:
                _emit(
                    f"receipts ticket: {torn.path}: skipped a torn last line (line {torn.line}, "
                    f"{len(torn.dropped)} bytes); the ledger is read-only, so the line stays in the file",
                    error=True,
                )
    if not args.post:
        _print_draft(draft, as_json=args.json)
        return 0

    key = os.environ.get(KEY_NAME)
    if not key:
        _emit(f"receipts ticket: {KEY_NAME} is missing or empty", error=True)
        return 2
    transport: _PostingTransport | None = None
    try:
        transport = _PostingTransport(LinearHTTP(key))
        _preflight(transport, args.team_id, args.expect_org)
        writer = LinearTicketWriter(
            transport,
            team_id=args.team_id,
            receipt_base=args.receipt_base,
            label_ids=args.label_id,
            project_id=args.project_id,
            priority=args.priority,
        )
        result = writer.file(entry, spent_micro_usd=spent)
    except TicketRefusedError as err:
        _emit(f"receipts ticket: {err}", error=True, key=key)
        return 3
    except RateLimitedError as err:
        _emit(f"receipts ticket: {err}; retry_after={err.retry_after}", error=True, key=key)
        return 1
    except TrackerError as err:
        if transport is not None and transport.limited is not None:
            # The writer can restore its original create error after a rate-limited recovery lookup.
            limited = transport.limited
            _emit(f"receipts ticket: {limited}; retry_after={limited.retry_after}", error=True, key=key)
        else:
            _emit(f"receipts ticket: {err}", error=True, key=key)
        return 1
    except ValueError as err:
        _emit(f"receipts ticket: {err}", error=True, key=key)
        return 2
    _print_result(result, as_json=args.json, key=key)
    return 0
