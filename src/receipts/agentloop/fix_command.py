"""The integrator's entry point for receipts fix, without importing the core lane."""

from __future__ import annotations

import argparse
import functools
import sys
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Protocol

from pydantic import TypeAdapter

from receipts.agentloop.fixloop import (
    FixRecord,
    FixStageError,
    FixStatus,
    NonBlockerError,
    ReAttack,
    ReAttackResult,
    confirmed_blocker,
    run_fix,
)
from receipts.contracts import Arm, Frozen, ModelClient, RunnerKind, Sandbox, SuiteRunner, Target
from receipts.surface.ledger import JsonlLedger, LedgerError
from receipts.surface.text import visible


class RunPhase(Protocol):
    @property
    def sandbox(self) -> Sandbox: ...

    @property
    def runners(self) -> Mapping[RunnerKind, SuiteRunner]: ...

    @property
    def release(self) -> Callable[[], None] | None: ...


@dataclass(frozen=True)
class FixWiring:
    client: Callable[[], ModelClient]
    run_phase: Callable[[], RunPhase]
    reattack: Callable[[Path], ReAttack]


class RunItem(Frozen):
    target: Target
    temperatures: Literal["spec", "meta_default"]


def register(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser], *, wiring: FixWiring | None = None
) -> None:
    command = subparsers.add_parser("fix", help="fix one confirmed blocker and try a fresh attack")
    command.add_argument("finding_id")
    command.add_argument("--ledger", type=Path, required=True)
    command.add_argument("--targets", type=Path, required=True)
    command.add_argument("--arms", type=Path, required=True)
    command.add_argument("--out-dir", type=Path, required=True)
    command.add_argument("--max-iterations", type=int, choices=(1, 2, 3), default=3)
    command.set_defaults(func=functools.partial(run_command, wiring=wiring))


def _target(path: Path, target_id: str) -> Target:
    targets: dict[str, Target] = {}
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            item = RunItem.model_validate_json(line)
        except ValueError:
            raise ValueError(f"invalid target row at line {number}") from None
        if item.target.id in targets:
            raise ValueError(f"duplicate target {item.target.id!r}")
        targets[item.target.id] = item.target
    if target_id not in targets:
        raise ValueError(f"target {target_id!r} is absent from the run file")
    return targets[target_id]


def _arm(path: Path, ledger_path: Path) -> Arm:
    arms = TypeAdapter(list[Arm]).validate_json(path.read_bytes())
    if not arms or len({arm.name for arm in arms}) != len(arms):
        raise ValueError("arms must be nonempty and have unique names")
    matches = [arm for arm in arms if arm.name == ledger_path.stem]
    if len(matches) != 1:
        raise ValueError(f"ledger name {ledger_path.stem!r} must match exactly one arm")
    return matches[0]


def print_record(record: FixRecord) -> None:
    for iteration in record.iterations:
        verdict = "accepted" if iteration.guard.accepted else "rejected"
        print(visible(f"iteration {iteration.number}: guard {verdict}: {iteration.guard.reason}"))
        for run in iteration.shown_runs:
            print(visible(f"iteration {iteration.number}: shown seed {run.seed}: {run.outcome.value}"))
    print(visible(f"held_out: {record.held_out or 'none'}"))
    for check in record.independent_checks:
        verdict = "pass" if check.passed else "not verified"
        print(visible(f"independent {check.phase} seed {check.run.seed}: {check.case}: {verdict}"))
        if "not independently verifiable:" in check.run.stderr_tail:
            print(visible(check.run.stderr_tail.strip()))
    print(visible(f"coverage: {record.coverage}"))
    print(visible(f"canary coverage: {record.canary_scope}"))
    if record.reattack_reason:
        print(visible(f"fresh attack: {record.reattack_reason}"))
    if record.reattack_out_dir:
        print(visible(f"fresh attack output: {record.reattack_out_dir}"))
    suffix = f": {record.reason}" if record.reason else ""
    if record.reason and record.detail and record.detail != record.reason:
        suffix += f": {record.detail}"
    print(visible(f"{record.status.value}{suffix}"))


def run_command(args: argparse.Namespace, *, wiring: FixWiring | None) -> int:
    phase: RunPhase | None = None
    exit_code = 2
    try:
        ledger = JsonlLedger(args.ledger, create=False, read_only=True)
        shown = confirmed_blocker(ledger, args.finding_id)
        target = _target(args.targets, shown.target_id)
        arm = _arm(args.arms, args.ledger)
        if not 1 <= args.max_iterations <= 3:
            raise ValueError("--max-iterations must be between 1 and 3")
        if wiring is None:
            raise ValueError("fix is not wired; the integrator supplies FixWiring")
        # Factories are deliberately below the latest-blocker check.
        client = wiring.client()
        phase = wiring.run_phase()
        runner = phase.runners.get(target.test_runner)
        if runner is None:
            raise ValueError(f"no runner for {target.test_runner.value}")

        attack_factory = wiring.reattack

        class CommandReAttack:
            attack: ReAttack

            def validate(self, arm: Arm, *, finding_id: str) -> None:
                try:
                    self.attack = attack_factory(args.out_dir)
                    validate = getattr(self.attack, "validate", None)
                    if validate is not None:
                        validate(arm, finding_id=finding_id)
                except FixStageError:
                    raise
                except Exception as exc:
                    raise FixStageError(f"{type(exc).__name__}: fresh attack preflight failed", costs=()) from exc

            def __call__(self, target: Target, arm: Arm, *, budget_micro_usd: int) -> ReAttackResult:
                return self.attack(target, arm, budget_micro_usd=budget_micro_usd)

        args.out_dir.mkdir(parents=True, exist_ok=True)
        record = run_fix(
            args.finding_id,
            ledger=ledger,
            target=target,
            arm=arm,
            sandbox=phase.sandbox,
            runner=runner,
            client=client,
            reattack=CommandReAttack(),
            records_path=args.out_dir / "fixes.jsonl",
            max_iterations=args.max_iterations,
        )
        print_record(record)
        exit_code = 0 if record.status is FixStatus.SURVIVED_FRESH_ATTACK else 1
    except NonBlockerError as exc:
        print(visible(f"receipts fix: {exc}"), file=sys.stderr)
        exit_code = 3
    except (OSError, ValueError, LedgerError) as exc:
        print(visible(f"receipts fix: {exc}"), file=sys.stderr)
        exit_code = 2
    except Exception as exc:
        print(visible(f"receipts fix: could not start: {type(exc).__name__}"), file=sys.stderr)
        exit_code = 2
    finally:
        if phase is not None and phase.release is not None:
            try:
                phase.release()
            except Exception as exc:
                print(visible(f"receipts fix: sandbox cleanup failed: {type(exc).__name__}"), file=sys.stderr)
                if exit_code == 0:
                    exit_code = 1
    return exit_code
