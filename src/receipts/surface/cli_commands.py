"""The surface lane's subcommands. The integrator wires `register(sub)` into `receipts.cli`; every subcommand sets
`func` to a handler that takes the parsed arguments and returns the exit code.

`replay` is a stub for now: it prints a finding's replay key and repro from the ledger, `.receipts/ledger.jsonl` under
the current directory unless `--ledger` names a run's own file, with why the finding has its status: its unchecked
reason, its detail, a different model's refutation, and its runs, each beside the problem that kept it from being
read, if one did. A finding id starts with its target id, so the id alone names the finding. Rerunning the repro in a
fresh sandbox comes later.

`ledger rollup` is the pilot's cost read: spend and tokens per stage and model, the per-target p50 and p95 of cost and
latency, critic rounds, and how many findings ended in each status. `--ledger` takes one ledger or a directory of
them, one `<arm>.jsonl` per arm. A directory is read arm by arm and without changes: a torn last line is reported and
left in place for the run's own writer to recover. `--arms` names the arms to read, from the run's arms file, so a
missing arm's ledger is an error and no other file is read as one. The spend is what the ledgers' entries record; the
costs no finding carries (intake, the plan, a failed run's spend) are on the run's per-target summaries.

Repro files and commands are model output, and an error can quote ledger text, so every control and format character
is escaped before anything reaches the terminal.
"""

from __future__ import annotations

import argparse
import errno
import json
import shlex
import sys
from collections.abc import Sequence
from pathlib import Path

from pydantic import TypeAdapter, ValidationError

from receipts.contracts import Arm, RunResult, Status
from receipts.surface.ledger import (
    JsonlLedger,
    LatencyBasis,
    LedgerError,
    ReplayRecord,
    Rollup,
    TornTail,
    TornTailSaveError,
    usd,
)
from receipts.surface.text import visible

DEFAULT_LEDGER = Path(".receipts") / "ledger.jsonl"
LEDGER_SUFFIX = ".jsonl"  # a directory of per-arm ledgers holds <arm>.jsonl files; <arm>.jsonl.torn is not one
OUTSIDE_THE_LEDGERS = (
    "spend counts what these ledgers' entries record; costs no finding carries (intake, the plan, a failed run's "
    "spend) are on the run's per-target summaries, outside the ledgers"
)
_ARMS = TypeAdapter(list[Arm])


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    """Add `replay` and `ledger rollup`, each with `func` set to its handler."""
    parser = subparsers.add_parser(
        "replay",
        help="print a finding's replay key and repro from the ledger",
        description="Print a finding's replay key and repro from its latest ledger entry. Rerunning it comes later.",
    )
    parser.add_argument("finding_id", help="the finding id, which starts with its target id")
    parser.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="the ledger file, such as a run's own (default: %(default)s)",
    )
    parser.add_argument("--json", action="store_true", help="print the record as JSON")
    parser.set_defaults(func=replay_command)

    ledger = subparsers.add_parser("ledger", help="read a ledger", description="Read a ledger.")
    ledger_sub = ledger.add_subparsers(dest="ledger_command", required=True)
    rollup = ledger_sub.add_parser(
        "rollup",
        help="spend, tokens, latency and statuses from a ledger",
        description="Spend and tokens per stage and model, per-target p50 and p95 cost and latency, critic rounds and "
        "statuses. A directory is read as one ledger per arm, <arm>.jsonl.",
    )
    rollup.add_argument(
        "--ledger",
        type=Path,
        default=DEFAULT_LEDGER,
        help="a ledger file, or a directory of <arm>.jsonl ledgers (default: %(default)s)",
    )
    rollup.add_argument(
        "--arms",
        type=Path,
        help="the run's arms file (a JSON list of arms): read exactly <ledger>/<arm>.jsonl for each arm in it",
    )
    rollup.add_argument("--target", help="only this target's entries")
    rollup.add_argument("--by-target", action="store_true", help="also print one row per target")
    rollup.add_argument("--json", action="store_true", help="print the rollups as JSON")
    rollup.set_defaults(func=rollup_command)


# ---------------------------------------------------------------- replay


def replay_command(args: argparse.Namespace) -> int:
    """0 when the finding has a replay key and a repro to print; 1 otherwise, with the reason on stderr."""
    path: Path = args.ledger
    try:
        ledger = open_for_reading(path)
        for torn in ledger.torn_tails:
            _warn("replay", _torn_message(ledger, torn))
        record = ledger.replay(args.finding_id)
    except FileNotFoundError:
        _warn("replay", f"no ledger at {path}")
        return 1
    except (LedgerError, OSError) as err:
        _warn("replay", str(err))
        return 1
    if args.json:
        print(_json(record.model_dump(mode="json")))
    else:
        print(_text(record, path))
    missing = [name for name, value in (("replay key", record.replay), ("repro", record.repro)) if value is None]
    if missing:
        _warn("replay", f"finding {record.finding_id} has no {' or '.join(missing)} recorded, so it cannot be replayed")
        return 1
    return 0


def open_for_reading(path: Path) -> JsonlLedger:
    """The ledger opened the way a writer opens it, so a torn last line is saved and cut. A ledger this process may not
    write (a read-only file, directory or mount), or one whose torn line's bytes cannot be saved (the torn-tail path
    is a directory, a dangling link or read-only), is opened read-only instead, which leaves the file as it is. Only
    the ledger itself missing raises FileNotFoundError."""
    try:
        return JsonlLedger(path, create=False)
    except TornTailSaveError:
        pass  # an OSError too, so it is caught before the others
    except PermissionError:
        pass
    except OSError as err:
        if err.errno != errno.EROFS:
            raise
    return JsonlLedger(path, read_only=True)


READ_ONLY = "the ledger is read-only here"
READ_AS_IS = "a directory of ledgers is read without changes"


def _torn_message(ledger: JsonlLedger, torn: TornTail, *, why: str = READ_ONLY) -> str:
    what = f"a torn last line (line {torn.line}, {len(torn.dropped)} bytes) left by a crashed append"
    if torn.cut:
        return f"{torn.path}: cut {what}; its bytes are saved in {ledger.torn_path}"
    return f"{torn.path}: skipped {what}; {why}, so the line stays in the file"


def _text(record: ReplayRecord, path: Path) -> str:
    v = visible
    history = f" (the latest of {record.entries} entries for this finding)" if record.entries > 1 else ""
    lines = [
        f"finding {v(record.finding_id)}  target {v(record.target_id)}  status {record.status.value}  "
        f"kind {record.kind.value if record.kind else 'none'}",
        f"ledger {v(str(path))}, line {record.line}, written {record.at}{history}",
    ]
    lines += _reason_lines(record)
    if record.replay is None:
        lines.append("replay key: none recorded")
    else:
        key = record.replay
        lines.append("replay key")
        lines += [
            f"  {name:<13}{v(str(value))}"
            for name, value in (
                ("base_sha", key.base_sha),
                ("patch_sha", key.patch_sha),
                ("harness_hash", key.harness_hash),
                ("seed", key.seed),
                ("image_digest", key.image_digest),
            )
        ]
    if record.repro is None:
        lines.append("repro: none recorded")
    else:
        repro = record.repro
        lines.append(
            f"repro {v(repro.id)}  attack {v(repro.attack_id)}  claim {v(repro.claim_id)}  "
            f"lens {repro.lens.value}  kind {repro.kind.value}"
        )
        lines.append(f"  command  {v(shlex.join(repro.command))}")
        if repro.cases:
            lines.append("  cases")
            lines += [f"    {v(case)}" for case in repro.cases]
        else:
            lines.append("  cases    every case its files define")
        for name, content in sorted(repro.files.items()):
            lines.append(f"  file {v(name)}")
            lines += [f"    | {v(text)}" for text in content.split("\n")]
    lines += _runs_text(record.runs)
    lines.append(
        "replay itself is not built yet: it will rerun these files with the target's runner through the sandbox, "
        "which gives each run a report path of its own; the command is the record of how they were run"
    )
    return "\n".join(lines)


def _reason_lines(record: ReplayRecord) -> list[str]:
    """Why the finding has its status (contract v3): its unchecked reason, its detail, and a refutation, labeled as
    a different model's only on a line written since v3, when the field held nothing else."""
    lines: list[str] = []
    if record.unchecked_reason is not None:
        lines.append(f"unchecked reason: {record.unchecked_reason.value}")
    if record.detail:
        lines.append(f"detail, why it is {record.status.value}")
        lines += [f"  | {visible(text)}" for text in record.detail.split("\n")]
    elif not record.written_before_v3:
        lines.append("detail: none recorded")
    if record.refutation:
        if record.written_before_v3:
            lines.append(
                "refutation, from a line written before contract v3, when this field also held reasons that were "
                "not a different model's refutation"
            )
        else:
            lines.append("refutation by a different model")
        lines += [f"  | {visible(text)}" for text in record.refutation.split("\n")]
    return lines


def _runs_text(runs: Sequence[RunResult]) -> list[str]:
    """The runs, one row each, with the problem that kept a run from being read beside its outcome."""
    if not runs:
        return ["runs: none recorded"]
    header = ("#", "side", "outcome", "failure", "exit", "fresh", "seed", "seconds", "problem")
    rows = [
        (
            str(n),
            r.side.value,
            r.outcome.value,
            r.failure_type.value if r.failure_type else "",
            "" if r.exit_code is None else str(r.exit_code),
            "yes" if r.fresh_container else "no",
            "" if r.seed is None else str(r.seed),
            f"{r.duration_s:,.1f}",
            visible(" ".join(r.problem.split())),
        )
        for n, r in enumerate(runs, start=1)
    ]
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    table = ["  ".join(cell.ljust(w) for cell, w in zip(row, widths, strict=True)).rstrip() for row in (header, *rows)]
    return ["runs", *(f"  {row}" for row in table)]


# ---------------------------------------------------------------- ledger rollup


def rollup_command(args: argparse.Namespace) -> int:
    """0 when every ledger read; 1 when the path holds no ledger, an arm's ledger is missing, or any ledger could not
    be read."""
    path: Path = args.ledger
    try:
        files = _ledger_files(path, args.arms)
    except LedgerError as err:
        _warn("ledger rollup", str(err))
        return 1
    directory = path.is_dir()
    code = 0
    read: list[tuple[str, Path, Rollup]] = []
    for arm, file in files:
        try:
            # A directory is read without changes. One named ledger is opened as a writer opens it, so a torn last
            # line is saved and cut, as replay does.
            ledger = JsonlLedger(file, read_only=True) if directory else open_for_reading(file)
            for torn in ledger.torn_tails:
                _warn("ledger rollup", _torn_message(ledger, torn, why=READ_AS_IS if directory else READ_ONLY))
            read.append((arm, file, ledger.rollup(args.target)))
        except FileNotFoundError:
            _warn("ledger rollup", f"no ledger at {file}" + (f" for arm {arm}" if args.arms else ""))
            code = 1
        except (LedgerError, OSError) as err:
            _warn("ledger rollup", str(err))
            code = 1
    if args.json:
        print(_json({"ledgers": [{"arm": a, "path": str(f), "rollup": r.model_dump(mode="json")} for a, f, r in read]}))
        return code
    blocks = [_rollup_text(arm, file, r, by_target=args.by_target) for arm, file, r in read]
    if len(read) > 1:
        blocks.append(_arms_text(read))
    if blocks:
        blocks.append(f"note: {OUTSIDE_THE_LEDGERS}")
        print("\n\n".join(blocks))
    return code


def _ledger_files(path: Path, arms_file: Path | None) -> list[tuple[str, Path]]:
    """The (arm, ledger) pairs to read: the arms file's arms in its order, every <arm>.jsonl in a directory, or the
    one ledger named."""
    if arms_file is not None:
        if not path.is_dir():
            raise LedgerError(f"--arms reads a directory of <arm>.jsonl ledgers, and {path} is not one")
        names = [arm.name for arm in _read_arms(arms_file)]
        return [(name, path / f"{name}{LEDGER_SUFFIX}") for name in names]
    if path.is_dir():
        found = sorted(p for p in path.iterdir() if p.name.endswith(LEDGER_SUFFIX) and p.is_file())
        if not found:
            raise LedgerError(f"no ledgers (*{LEDGER_SUFFIX}) in {path}")
        return [(p.name[: -len(LEDGER_SUFFIX)], p) for p in found]
    arm = path.name[: -len(LEDGER_SUFFIX)] if path.name.endswith(LEDGER_SUFFIX) else path.name
    return [(arm, path)]


def _read_arms(path: Path) -> list[Arm]:
    try:
        arms = _ARMS.validate_json(path.read_bytes())
    except OSError as err:
        raise LedgerError(f"the arms file {path} cannot be read: {err.strerror or err}") from err
    except (ValidationError, ValueError) as err:
        raise LedgerError(f"the arms file {path} is not a JSON list of arms: {_first_problem(err)}") from err
    names = [arm.name for arm in arms]
    if not names:
        raise LedgerError(f"the arms file {path} names no arm")
    if len(set(names)) != len(names):
        raise LedgerError(f"the arms file {path} names an arm twice")
    return arms


def _first_problem(err: Exception) -> str:
    if isinstance(err, ValidationError):
        first = err.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        return f"{where}: {first['msg']}" if where else str(first["msg"])
    return " ".join(str(err).split())[:300]


def _seconds(value: float | None) -> str:
    return "none" if value is None else f"{value:,.1f} s"


def _latency_note(r: Rollup) -> str:
    p = r.percentiles
    if p.latency_basis is None:
        return "no targets"
    if p.latency_basis is LatencyBasis.WALL:
        return f"wall time, all {p.targets} targets timed"
    if p.latency_basis is LatencyBasis.SUMMED_STAGE_SECONDS:
        return "summed stage seconds, no target timed: parallel stages count in full, so this is not wall latency"
    return f"mixed: {p.wall_targets} of {p.targets} targets timed, the rest summed stage seconds"


def _rollup_text(arm: str, path: Path, r: Rollup, *, by_target: bool) -> str:
    v = visible
    t, p = r.totals, r.percentiles
    statuses = ", ".join(f"{status.value} {t.statuses.get(status, 0)}" for status in Status)
    lines = [
        f"ledger {v(str(path))} (arm {v(arm)}): {t.entries} entries, {t.findings} findings, {t.targets} targets",
        f"spend {usd(t.micro_usd)}; tokens in {t.input_tokens:,} (cached {t.cached_tokens:,}), out "
        f"{t.output_tokens:,} (reasoning {t.reasoning_tokens:,}); stage seconds {t.seconds:,.1f}",
        f"statuses: {statuses}",
        f"per target: cost p50 {usd(p.p50_micro_usd) if p.p50_micro_usd is not None else 'none'}, p95 "
        f"{usd(p.p95_micro_usd) if p.p95_micro_usd is not None else 'none'}; latency p50 {_seconds(p.p50_seconds)}, "
        f"p95 {_seconds(p.p95_seconds)} ({_latency_note(r)})",
        f"critic rounds: {r.critic.rounds} over {r.critic.findings} findings "
        f"({', '.join(f'{n}: {k}' for n, k in enumerate(r.critic.findings_by_rounds))})",
    ]
    # A record is one StageCost: a client's one call, or an attacker's sum over a stage, so this is not a call count.
    header = ("stage", "model", "records", "input", "cached", "output", "reasoning", "spend", "seconds")
    rows = [
        (
            v(s.stage),
            v(s.model or "(code)"),
            f"{s.records:,}",
            f"{s.input_tokens:,}",
            f"{s.cached_tokens:,}",
            f"{s.output_tokens:,}",
            f"{s.reasoning_tokens:,}",
            usd(s.micro_usd),
            f"{s.seconds:,.1f}",
        )
        for s in r.stages
    ]
    lines += ["", *_table(header, rows)] if rows else ["", "no stage recorded any cost"]
    if by_target:
        target_header = ("target", "entries", "findings", "spend", "latency", "basis", "critic rounds")
        target_rows = [
            (
                v(x.target_id),
                f"{x.entries:,}",
                f"{x.findings:,}",
                usd(x.micro_usd),
                f"{x.latency_seconds:,.1f}",
                x.latency_basis.value,
                str(x.critic_rounds),
            )
            for x in r.targets
        ]
        lines += ["", *_table(target_header, target_rows)]
    return "\n".join(lines)


def _arms_text(read: list[tuple[str, Path, Rollup]]) -> str:
    header = ("arm", "targets", "findings", "blockers", "unchecked", "spend", "cost p50", "cost p95")
    rows = [
        (
            visible(arm),
            str(r.totals.targets),
            str(r.totals.findings),
            str(r.totals.statuses.get(Status.BLOCKER, 0)),
            str(r.totals.statuses.get(Status.UNCHECKED, 0)),
            usd(r.totals.micro_usd),
            usd(r.percentiles.p50_micro_usd) if r.percentiles.p50_micro_usd is not None else "none",
            usd(r.percentiles.p95_micro_usd) if r.percentiles.p95_micro_usd is not None else "none",
        )
        for arm, _, r in read
    ]
    return "\n".join(["arms", *_table(header, rows)])


def _table(header: tuple[str, ...], rows: Sequence[tuple[str, ...]]) -> list[str]:
    widths = [max(len(cell) for cell in column) for column in zip(header, *rows, strict=True)]
    out = []
    for row in (header, *rows):
        cells = [cell.ljust(w) if i < 2 else cell.rjust(w) for i, (cell, w) in enumerate(zip(row, widths, strict=True))]
        out.append("  ".join(cells).rstrip())
    return out


# ---------------------------------------------------------------- output


def _json(data: object) -> str:
    # With ensure_ascii, json.dumps escapes everything outside printable ASCII: the C0 controls, DEL, the C1 controls
    # and every bidirectional override or zero-width mark.
    return json.dumps(data, indent=2, ensure_ascii=True)


def _warn(command: str, message: str) -> None:
    print(f"receipts {command}: {visible(message)}", file=sys.stderr)
