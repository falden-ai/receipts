"""Register `receipts check post`. Credentials are read only after an explicit --post."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from pydantic import BaseModel, ConfigDict, ValidationError

from receipts.contracts import Target
from receipts.surface.checks import (
    AppConfig,
    AppConfigError,
    CheckInputError,
    RepositoryRefusedError,
    check_payload,
    post_check,
    validate_destination,
)
from receipts.surface.ledger import JsonlLedger, LedgerError
from receipts.surface.text import visible
from receipts.surface.tickets import TrackerError


class _RunItem(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    target: Target
    temperatures: str


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    check = subparsers.add_parser("check", help="post a Receipts result as a GitHub check")
    commands = check.add_subparsers(dest="check_command", required=True)
    post = commands.add_parser("post", help="preview a check payload, or post it with --post")
    post.add_argument("--ledger", type=Path, required=True)
    post.add_argument("--targets", type=Path, required=True)
    post.add_argument("--target-id", required=True)
    post.add_argument("--repo", required=True)
    post.add_argument("--head-sha", required=True)
    post.add_argument("--post", action="store_true", help="read App configuration and write the GitHub check")
    post.add_argument("--json", action="store_true", help="print the posted check id and URL as JSON")
    post.set_defaults(func=post_command)


def _target(path: Path, target_id: str) -> Target:
    matches: list[Target] = []
    with path.open(encoding="utf-8") as stream:
        for number, line in enumerate(stream, 1):
            try:
                item = _RunItem.model_validate_json(line)
            except ValidationError:
                raise CheckInputError(f"invalid target run item on line {number}") from None
            if item.target.id == target_id:
                matches.append(item.target)
    if len(matches) != 1:
        raise CheckInputError("--target-id must select exactly one target in --targets")
    return matches[0]


def post_command(args: argparse.Namespace) -> int:
    try:
        target = _target(args.targets, args.target_id)
        validate_destination(target, args.repo, args.head_sha)
        ledger = JsonlLedger(args.ledger, create=False, read_only=True)
        if ledger.torn_tails:
            raise CheckInputError("the ledger has a torn last line; obtain a complete ledger before posting")
        payload = check_payload(ledger.entries(target.id), target, head_sha=args.head_sha)
    except CheckInputError as err:
        _error(str(err))
        return 1
    except (OSError, UnicodeError, LedgerError):
        # Ledger validation can quote private or hostile content. Report no raw exception text.
        _error("could not read a valid ledger or target run file")
        return 1
    if not args.post:
        print(visible(json.dumps(payload, ensure_ascii=True, indent=2), keep="\n"))
        return 0
    try:
        config = AppConfig.from_environ()
        posted = post_check(config, args.repo, payload)
    except AppConfigError as err:
        _error(str(err))
        return 2
    except RepositoryRefusedError:
        _error("the installation token does not include --repo; no check was posted")
        return 3
    except (TrackerError, CheckInputError):
        _error("GitHub check posting failed")
        return 1
    if args.json:
        print(visible(posted.model_dump_json()))
    else:
        print(visible(f"Check run {posted.id}: {posted.html_url}"))
    return 0


def _error(message: str) -> None:
    print(f"check post: {visible(message)}", file=sys.stderr)
