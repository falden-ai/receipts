"""Register the standalone `receipts receipt html` command; cli.py wiring belongs to the integrator."""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path

from pydantic import ValidationError

from receipts.surface.ledger import JsonlLedger, LedgerError
from receipts.surface.receipt_html import UnknownTargetError, build_view, read_summary, render_receipt
from receipts.surface.text import visible


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    receipt = subparsers.add_parser("receipt", help="read a target's receipt")
    formats = receipt.add_subparsers(dest="receipt_format", required=True)
    page = formats.add_parser("html", help="write a self-contained HTML receipt for one target")
    page.add_argument("target_id", help="the target id from the ledger")
    page.add_argument("--ledger", type=Path, required=True, help="one arm's JSONL ledger")
    page.add_argument("--summary", type=Path, help="the target's summary JSON file")
    page.add_argument("--run-ledger", type=Path, help="the whole run's ledger for per-target percentiles")
    page.add_argument("--out", type=Path, required=True, help="the HTML file to write")
    page.set_defaults(func=html_command)


def _warn(message: str) -> None:
    print(f"receipts receipt html: {visible(message)}", file=sys.stderr)


def report_torn_tails(ledger: JsonlLedger, warn: Callable[[str], None] = _warn) -> None:
    """Warn once per torn last line a read-only ledger skipped; `receipts demo` shares it with its own prefix."""
    seen: set[tuple[int, int]] = set()
    for tail in ledger.torn_tails:
        key = (tail.line, tail.offset)
        if key not in seen:
            warn(f"{ledger.path}: skipped torn last line {tail.line}; read-only, so the file is unchanged")
            seen.add(key)


def _protect_inputs(output: Path, inputs: list[Path]) -> None:
    for path in inputs:
        if output.resolve() == path.resolve() or (output.exists() and output.samefile(path)):
            raise ValueError("--out must not overwrite an input file")


def html_command(args: argparse.Namespace) -> int:
    ledgers: list[JsonlLedger] = []
    try:
        ledger_path: Path = args.ledger
        summary_path: Path | None = args.summary
        run_path: Path | None = args.run_ledger
        output: Path = args.out
        inputs = [ledger_path, *([summary_path] if summary_path is not None else [])]
        if run_path is not None:
            inputs.append(run_path)
        _protect_inputs(output, inputs)
        ledger = JsonlLedger(ledger_path, create=False, read_only=True)
        ledgers.append(ledger)
        run_ledger = None
        if run_path is not None:
            run_ledger = JsonlLedger(run_path, create=False, read_only=True)
            ledgers.append(run_ledger)
        summary = read_summary(summary_path) if summary_path is not None else None
        view = build_view(args.target_id, ledger, summary=summary, run_ledger=run_ledger)
        output.write_text(render_receipt(view), encoding="utf-8")
    except UnknownTargetError as err:
        _warn(str(err))
        return 3
    except (LedgerError, OSError, ValueError, OverflowError, ValidationError) as err:
        _warn(str(err))
        return 2
    finally:
        for opened in ledgers:
            report_torn_tails(opened)
    print(f"wrote {visible(str(output))}")
    return 0
