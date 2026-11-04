"""The core lane's `receipts` subcommands. None of them calls a model itself; `receipts run` drives the pipeline over
whatever the integrator wires.

The integrator wires `register(sub)` into receipts.cli, and for `receipts run` passes a RunWiring:
`register(sub, wiring=RunWiring(seams=..., ledger=...))`. Every subparser here sets `func` to a handler that takes the
parsed arguments and returns the exit code.

`receipts run --targets FILE --arms FILE --ledger-dir DIR` reads the run file (JSONL, one
{"target": <contracts.Target>, "temperatures": "spec" | "meta_default"} per line) and the arms file (a JSON list of
contracts.Arm), refuses either if it is empty, malformed or names a target or an arm twice, then runs every target
through receipts.core.pipeline: each arm's findings go to DIR/<arm>.jsonl and each target's summary to
DIR/targets/<target id>.json, marked failed for a target that raised. --resume skips the targets already done and runs
again those whose last run failed. It exits 0 when every target ran, 1 when one failed, 2 when it could not start.
Each target's line says when its plan came from the planner's fallback (REG-570) and when the agent's patch applied
only with git apply --reject or GNU patch's fuzz (REG-569), and when no refuter (SK2) ran. --jobs N runs N targets
at once, each on a run phase of its own (Seams.run_phase); seams with one shared sandbox refuse N above 1 before a
ledger is created.

`receipts spend` reads the spend file the Meta client uses: RECEIPTS_SPEND_FILE, else .receipts/spend.jsonl under the
current directory. The command and the client resolve it the same way, so an operator arms exactly the file the
clients meter against.
"""

from __future__ import annotations

import argparse
import functools
import json
import sys
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from receipts.contracts import Arm, Ledger, PatchMethod, Side, Target
from receipts.core.intake import template_claims
from receipts.core.meta_client import (
    ARM_COMMAND,
    DEFAULT_SPEND_CAP_MICRO_USD,
    DOCS_READ_ON,
    LIST_PRICES,
    PER_MILLION,
    PRICES_SOURCE,
    SPEND_CEILING_MICRO_USD,
    SPEND_FILE_ENV,
    ArmRecord,
    SpendFileError,
    SpendMeter,
    default_spend_path,
    named_spend_file,
    usd,
)
from receipts.core.pipeline import RunItem, Seams, TargetSummary, check_jobs, check_run, run_targets, summary_path

_ARMS: TypeAdapter[list[Arm]] = TypeAdapter(list[Arm])


@dataclass(frozen=True)
class RunWiring:
    """What `receipts run` runs on, given by the integrator: the seams, built once the run's arms are read, and a
    ledger opened (or created) at a path, one per arm."""

    seams: Callable[[Sequence[Arm]], Seams]
    ledger: Callable[[Path], Ledger]


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], *, wiring: RunWiring | None = None
) -> None:
    prices = subparsers.add_parser("prices", help="print the list prices the spend meter uses")
    prices.set_defaults(func=run_prices)

    spend = subparsers.add_parser(
        "spend",
        help="print metered spend and whether the spend meter is armed",
        description=f"The spend file is ${SPEND_FILE_ENV}, else .receipts/spend.jsonl under the current directory.",
    )
    spend.set_defaults(func=run_spend)
    spend_commands = spend.add_subparsers(dest="spend_command")
    arm = spend_commands.add_parser(
        "arm",
        help="allow model calls, up to the cap, once the Meta credit is confirmed as applied",
        description="Every model call is refused until the spend file is armed. Arm it only after confirming the "
        f"Meta credit is applied to the account. The cap is at most {usd(DEFAULT_SPEND_CAP_MICRO_USD)}, the credit, "
        f"unless --above-credit is given on a spend file named with ${SPEND_FILE_ENV}: then it is above the credit and "
        f"at most {usd(SPEND_CEILING_MICRO_USD)}, the founder's ceiling, and the spend past the credit is cash.",
    )
    arm.add_argument(
        "--credit-confirmed",
        action="store_true",
        help="confirm that the Meta credit is applied to the account",
    )
    arm.add_argument(
        "--cap-micro-usd",
        type=int,
        default=DEFAULT_SPEND_CAP_MICRO_USD,
        help=f"the kill-switch cap in micro-dollars, at most the default (default {usd(DEFAULT_SPEND_CAP_MICRO_USD)}) "
        f"unless --above-credit is given, then above it and at most {usd(SPEND_CEILING_MICRO_USD)}",
    )
    arm.add_argument(
        "--above-credit",
        action="store_true",
        help=f"arm a cap past the {usd(DEFAULT_SPEND_CAP_MICRO_USD)} Meta credit, at most the "
        f"{usd(SPEND_CEILING_MICRO_USD)} ceiling: the spend past the credit is cash",
    )
    arm.set_defaults(func=run_spend_arm)

    claims = subparsers.add_parser("claims", help="print the template claims of a Tier A target, with no model call")
    claims.add_argument("target", type=Path, help="a Target as JSON")
    claims.set_defaults(func=run_claims)

    run = subparsers.add_parser(
        "run",
        help="run the targets of a run file through intake, the planner, each arm's attackers and the skeptic",
        description="Each arm's findings go to LEDGER_DIR/<arm>.jsonl, each target's summary (claims, plan, and the "
        "intake and plan costs) to LEDGER_DIR/targets/<target id>.json.",
    )
    run.add_argument("--targets", type=Path, required=True, help='JSONL, one {"target", "temperatures"} per line')
    run.add_argument("--arms", type=Path, required=True, help="a JSON list of arms")
    run.add_argument("--ledger-dir", type=Path, required=True, help="where the ledgers and summaries go")
    run.add_argument("--jobs", type=int, default=1, help="targets run at once (default 1)")
    run.add_argument(
        "--resume", action="store_true", help="skip the targets already done, and run again those whose last run failed"
    )
    run.set_defaults(func=functools.partial(run_run, wiring=wiring))


def run_prices(args: argparse.Namespace) -> int:
    print(f"List prices per 1M tokens, Standard mode, read {DOCS_READ_ON} from {PRICES_SOURCE}")
    print(f"{'model':<18}{'input':>10}{'cached input':>14}{'output':>10}")
    for model, price in sorted(LIST_PRICES.items()):
        print(
            f"{model:<18}{_per_million(price.input):>10}{_per_million(price.cached_input):>14}"
            f"{_per_million(price.output):>10}"
        )
    print("Any other model is refused: off the allowlist, a -contributor id, or with no token price (sam-3.1).")
    return 0


def run_spend(args: argparse.Namespace) -> int:
    path = default_spend_path()
    if not path.exists():
        print(
            f"receipts spend: no spend file at {path}, so nothing is armed or metered there and every model call is "
            f"refused. Set {SPEND_FILE_ENV} to the run's spend file, or arm this one with '{ARM_COMMAND}' once the "
            "Meta credit is confirmed as applied.",
            file=sys.stderr,
        )
        return 1
    try:
        meter = SpendMeter(path)
        spent, held, armed, limit = meter.spent_micro_usd, meter.held_micro_usd, meter.armed, meter.limit_micro_usd
    except SpendFileError as exc:
        print(f"receipts spend: {path}: {exc}", file=sys.stderr)
        return 1
    print(f"spend file: {path}")
    if armed is None:
        print(
            "not armed: every model call is refused until an operator confirms the Meta credit and runs "
            f"'{ARM_COMMAND}'"
        )
        print(
            f"spent {usd(spent)} ({spent:,} micro-dollars); once armed, the cap is at most {usd(meter.cap_micro_usd)}, "
            f"or {usd(SPEND_CEILING_MICRO_USD)} armed with --above-credit on a spend file named with {SPEND_FILE_ENV}"
        )
    else:
        past = _past_credit(armed) if meter.named else ""
        print(
            f"armed at {armed.at} for {usd(armed.cap_micro_usd)}{past}, the operator having confirmed the Meta credit"
        )
        if not meter.named and armed.cap_micro_usd > DEFAULT_SPEND_CAP_MICRO_USD:
            print(
                f"this arm counts at the {usd(DEFAULT_SPEND_CAP_MICRO_USD)} credit: {path} is the current directory's "
                f"default spend file, which a checkout can ship, not one named with {SPEND_FILE_ENV}"
            )
        print(f"spent {usd(spent)} ({spent:,} micro-dollars) of the {usd(limit)} cap")
    if held:
        print(f"held {usd(held)} at the worst case by calls in flight, or left open by a process that stopped")
    if armed is not None:
        print(f"left {usd(max(limit - spent - held, 0))}")
        if spent >= limit:
            print("the kill switch is on: every model call is refused")
    return 0


def _past_credit(record: ArmRecord) -> str:
    """How much of an above-credit arm's cap is past the credit, which is cash; nothing for an arm at the credit."""
    if not record.above_credit:
        return ""
    past = record.cap_micro_usd - DEFAULT_SPEND_CAP_MICRO_USD
    return f", {usd(past)} of it past the {usd(DEFAULT_SPEND_CAP_MICRO_USD)} credit"


def run_spend_arm(args: argparse.Namespace) -> int:
    cap: int = args.cap_micro_usd
    above_credit: bool = args.above_credit
    if not args.credit_confirmed:
        command = f"{ARM_COMMAND} --cap-micro-usd {cap} --above-credit" if above_credit else ARM_COMMAND
        print(
            f"receipts spend arm: refused. Confirm the Meta credit is applied to the account, then run '{command}'.",
            file=sys.stderr,
        )
        return 2
    if above_credit and cap <= DEFAULT_SPEND_CAP_MICRO_USD:
        print(
            f"receipts spend arm: refused. --above-credit arms a cap past the {usd(DEFAULT_SPEND_CAP_MICRO_USD)} Meta "
            f"credit: give --cap-micro-usd above {DEFAULT_SPEND_CAP_MICRO_USD:,}, or arm without --above-credit.",
            file=sys.stderr,
        )
        return 2
    if above_credit and cap > SPEND_CEILING_MICRO_USD:
        print(
            f"receipts spend arm: refused. The cap is at most the ceiling, {SPEND_CEILING_MICRO_USD:,} micro-dollars "
            f"({usd(SPEND_CEILING_MICRO_USD)}, the founder's call); raising the ceiling is a code change.",
            file=sys.stderr,
        )
        return 2
    if above_credit and not named_spend_file(default_spend_path()):
        print(
            f"receipts spend arm: refused. An arm above the credit needs a spend file named with {SPEND_FILE_ENV}: "
            f"the default .receipts/spend.jsonl under the current directory, which a checkout can ship, counts at "
            f"most at the {usd(DEFAULT_SPEND_CAP_MICRO_USD)} credit.",
            file=sys.stderr,
        )
        return 2
    if not above_credit and not 0 < cap <= DEFAULT_SPEND_CAP_MICRO_USD:
        print(
            f"receipts spend arm: refused. The cap is 1 to {DEFAULT_SPEND_CAP_MICRO_USD:,} micro-dollars "
            f"({usd(DEFAULT_SPEND_CAP_MICRO_USD)}, the Meta credit); raising it past the credit is a code change.",
            file=sys.stderr,
        )
        return 2
    path = default_spend_path()
    try:
        meter = SpendMeter(path)
        before = meter.limit_micro_usd
        record = meter.arm(credit_confirmed=True, cap_micro_usd=cap, above_credit=above_credit)
        after, spent = meter.limit_micro_usd, meter.spent_micro_usd
    except (SpendFileError, OSError) as exc:
        print(f"receipts spend arm: {path}: {exc}", file=sys.stderr)
        return 1
    print(
        f"armed {path} at {record.at} for {usd(record.cap_micro_usd)}{_past_credit(record)}: model calls may spend up "
        "to the cap"
    )
    # The latest arm counts, so an arm can lower the cap a run is spending against: say so where an operator looks.
    if after < before:
        print(f"receipts spend arm: this lowers the cap in effect from {usd(before)} to {usd(after)}", file=sys.stderr)
    if spent >= after:
        print(
            f"receipts spend arm: {usd(spent)} is already spent, so the kill switch is on: every model call is refused",
            file=sys.stderr,
        )
    return 0


def run_claims(args: argparse.Namespace) -> int:
    path: Path = args.target
    try:
        target = Target.model_validate_json(path.read_bytes())
        claims = template_claims(target)
    except (OSError, ValidationError, ValueError) as exc:
        print(f"receipts claims: {exc}", file=sys.stderr)
        return 1
    print(json.dumps([claim.model_dump(mode="json") for claim in claims], indent=2))
    return 0


def _per_million(micro_usd: int) -> str:
    return f"${Decimal(micro_usd) / PER_MILLION:.2f}"


def read_run_file(path: Path) -> list[RunItem]:
    """The run file's items, each line validated; a bad line is refused by its number."""
    items: list[RunItem] = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            items.append(RunItem.model_validate_json(line))
        except ValidationError as exc:
            first = exc.errors(include_url=False, include_input=False)[0]
            where = ".".join(str(part) for part in first["loc"]) or "(line)"
            raise ValueError(f"run file {path}, line {number}: {where}: {first['msg']}") from None
    return items


def read_arms_file(path: Path) -> list[Arm]:
    try:
        return _ARMS.validate_json(path.read_bytes())
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"]) or "(file)"
        raise ValueError(f"arms file {path}: {where}: {first['msg']}") from None


def run_run(args: argparse.Namespace, *, wiring: RunWiring | None) -> int:
    if wiring is None:
        print(
            "receipts run: not wired: the integrator wires the sandbox, runners, attackers, clients and ledgers into "
            "receipts.cli",
            file=sys.stderr,
        )
        return 2
    try:
        items = read_run_file(args.targets)
        arms = read_arms_file(args.arms)
        check_run(items, arms)
        if args.jobs < 1:
            raise ValueError("--jobs is at least 1")
    except (OSError, ValueError) as exc:
        print(f"receipts run: {exc}", file=sys.stderr)
        return 2
    ledger_dir: Path = args.ledger_dir
    summary_dir = ledger_dir / "targets"
    done = [item.target.id for item in items if summary_path(summary_dir, item.target.id).exists()]
    if done and not args.resume:
        print(
            f"receipts run: {len(done)} target(s) already have a summary in {summary_dir} (first: {done[0]}); pass "
            "--resume to skip the done ones and run the failed ones again, or use a new --ledger-dir",
            file=sys.stderr,
        )
        return 2
    try:
        seams = wiring.seams(arms)
    except Exception as exc:  # the integrator's wiring (a missing key, an endpoint): report it, run nothing
        print(f"receipts run: the run could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        check_jobs(args.jobs, seams)
    except ValueError as exc:  # targets at once on one shared sandbox: refused before a ledger is created
        print(f"receipts run: --jobs {args.jobs} is refused: {exc}", file=sys.stderr)
        return 2
    try:
        ledger_dir.mkdir(parents=True, exist_ok=True)
        ledgers = {arm.name: wiring.ledger(ledger_dir / f"{arm.name}.jsonl") for arm in arms}
    except Exception as exc:  # a ledger the integrator's wiring refuses: report it, run nothing
        print(f"receipts run: the run could not start: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 2
    try:
        report = run_targets(
            items,
            arms=arms,
            seams=seams,
            ledgers=ledgers,
            summary_dir=summary_dir,
            jobs=args.jobs,
            resume=args.resume,
            on_done=_print_done,
        )
    except (FileExistsError, ValueError) as exc:  # a summary already there, or one that cannot be read
        print(f"receipts run: {exc}", file=sys.stderr)
        return 2
    rerun = f", {len(report.rerun)} run again after a failed run" if report.rerun else ""
    print(
        f"{len(report.done)} of {len(items)} targets done, {len(report.skipped)} skipped, {len(report.failed)} failed"
        f"{rerun}"
    )
    for target_id, why in report.failed:
        print(f"receipts run: {target_id} failed: {why}", file=sys.stderr)
    return 1 if report.failed else 0


# How a patch that did not apply cleanly was applied, as the run line says it (REG-569).
_LOOSE_PATCH = {
    PatchMethod.GIT_APPLY_REJECT: "git apply --reject",
    PatchMethod.GNU_PATCH_FUZZ: "GNU patch --fuzz=5",
}


def _print_done(summary: TargetSummary) -> None:
    arms = "; ".join(f"{a.arm} {a.findings} findings, {len(a.blockers)} blockers" for a in summary.arms)
    fallback = f"; planned by {summary.planner_model} (fallback)" if summary.planner_fallback else ""
    loose = [
        f"; {'the agent' if w.side is Side.PATCH else 'the ' + w.side.value}'s patch applied only with "
        f"{_LOOSE_PATCH[w.patch_method]}"
        for w in summary.skeptic_workspaces
        if w.patch_method in _LOOSE_PATCH
    ]
    refuter = "" if summary.refuter_model is not None else "; no refuter ran (SK2), so blockers are SK1's and SK3's"
    print(f"{summary.target_id}: {arms}{fallback}{refuter}{''.join(loose)} ({summary.wall_seconds:.0f} s)", flush=True)
