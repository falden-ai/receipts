"""`receipts demo`: the pandas demo's receipts, rebuilt from the records shipped in `demo/pandas-64677/evidence/`.

It writes two HTML pages with the same renderer as `receipts receipt html`: the Oct 31 run on pandas #64677, and the
fresh attack on fix attempt 5, which caught a regression in the fix itself. It replays recorded ledgers only: no model
call, no sandbox, no network, and it reads no credential.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import NamedTuple

from pydantic import ValidationError

from receipts.surface.ledger import JsonlLedger, LedgerError
from receipts.surface.receipt_command import report_torn_tails
from receipts.surface.receipt_html import UnknownTargetError, build_view, render_receipt
from receipts.surface.text import visible

DEFAULT_EVIDENCE = Path("demo") / "pandas-64677" / "evidence"
DEFAULT_OUT_DIR = Path("runs") / "demo"
SHIPPED = Path("demo")  # the shipped demo records, relative to the repository root


class DemoPage(NamedTuple):
    out_name: str
    target_id: str
    ledger: Path  # relative to the evidence folder
    label: str


PAGES = (
    DemoPage(
        "pandas-64677.html",
        "demo~pandas-64677",
        Path("run-2026-10-31T134142Z") / "spark.jsonl",
        "the Oct 31 run on the PR",
    ),
    DemoPage(
        "pandas-64677-fix-5.html",
        "demo~pandas-64677~fix1-a0027791a01e",
        Path("fix-5-20261101T172423Z") / "fix" / "reattack" / "fdd9c76a2d59-a0027791a01e" / "spark.jsonl",
        "the fresh attack on fix attempt 5",
    ),
)


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    demo = subparsers.add_parser(
        "demo",
        help="write the pandas demo's HTML receipts from its shipped records (no key, no model call)",
        description="Write the pandas #64677 demo's HTML receipts from the records in demo/pandas-64677/evidence/. "
        "It replays a recorded run: no model call, no sandbox, no network, no key. Run it from the repository root.",
    )
    demo.add_argument(
        "--evidence",
        type=Path,
        default=DEFAULT_EVIDENCE,
        help=f"the demo's evidence folder (default {DEFAULT_EVIDENCE})",
    )
    demo.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help=f"where the HTML pages go (default {DEFAULT_OUT_DIR})",
    )
    demo.set_defaults(func=demo_command)


def _warn(message: str) -> None:
    print(f"receipts demo: {visible(message)}", file=sys.stderr)


def _inside(path: Path, folder: Path) -> bool:
    return path.resolve().is_relative_to(folder.resolve())


def demo_command(args: argparse.Namespace) -> int:
    evidence: Path = args.evidence
    out_dir: Path = args.out_dir
    if not evidence.is_dir():
        _warn(f"{evidence}: no such folder; run this from the repository root, or name the folder with --evidence")
        return 2
    if _inside(out_dir, evidence) or _inside(out_dir, SHIPPED):
        _warn(f"{out_dir}: --out-dir must be outside demo/ and the evidence folder, which hold the shipped records")
        return 2
    written: list[tuple[Path, str]] = []
    ledgers: list[JsonlLedger] = []
    try:
        out_dir.mkdir(parents=True, exist_ok=True)
        for page in PAGES:
            ledger = JsonlLedger(evidence / page.ledger, create=False, read_only=True)
            ledgers.append(ledger)
            view = build_view(page.target_id, ledger)
            output = out_dir / page.out_name
            output.write_text(render_receipt(view), encoding="utf-8")
            written.append((output, page.label))
    except UnknownTargetError as err:
        _warn(_with_written(str(err), written))
        return 3
    except (LedgerError, OSError, ValueError, OverflowError, ValidationError) as err:
        _warn(_with_written(str(err), written))
        return 2
    finally:
        for opened in ledgers:
            report_torn_tails(opened, _warn)
    for output, label in written:
        print(f"wrote {visible(str(output))}  ({label})")
    print("Replayed from recorded ledgers: no model call, no sandbox, no network. Open the pages in a browser.")
    return 0


def _with_written(message: str, written: list[tuple[Path, str]]) -> str:
    if not written:
        return message
    pages = ", ".join(f"{output} ({label})" for output, label in written)
    return f"{message}; already wrote {pages}"
