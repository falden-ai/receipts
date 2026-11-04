"""Command parsing, latest-blocker entry and output safety, without a real run phase."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from receipts.agentloop.fix_command import FixWiring, register
from receipts.agentloop.fixloop import FixRecord, ReAttack, ReAttackResult
from receipts.agentloop.surrogate import DIFF_PATH
from receipts.contracts import Arm, CaseResult, Finding, ModelClient, Outcome, RunnerKind, Status, Target
from receipts.surface.ledger import JsonlLedger, new_entry
from tests.agentloop.test_fixloop import Box, Client, Reattack, Runner, finding, target


@dataclass
class Phase:
    sandbox: Box = field(default_factory=Box)
    runners: dict[RunnerKind, Runner] = field(default_factory=lambda: {RunnerKind.PYTEST: Runner()})
    release: Callable[[], None] | None = None


@dataclass
class Wiring:
    phase: Phase = field(default_factory=Phase)
    client_value: Client = field(default_factory=Client)
    attack: Reattack = field(default_factory=Reattack)
    calls: list[str] = field(default_factory=list)
    reattack_dirs: list[Path] = field(default_factory=list)

    def client(self) -> ModelClient:
        self.calls.append("client")
        return self.client_value

    def run_phase(self) -> Phase:
        self.calls.append("phase")
        self.phase.release = lambda: self.calls.append("release")
        return self.phase

    def reattack(self, out_dir: Path) -> ReAttack:
        self.calls.append("reattack")
        self.reattack_dirs.append(out_dir)
        return self.attack

    def build(self) -> FixWiring:
        return FixWiring(client=self.client, run_phase=self.run_phase, reattack=self.reattack)


def parser(wiring: FixWiring | None = None) -> argparse.ArgumentParser:
    result = argparse.ArgumentParser()
    register(result.add_subparsers(dest="command", required=True), wiring=wiring)
    return result


def inputs(tmp_path: Path, *rows: Finding, extra: Sequence[str] = ()) -> list[str]:
    ledger = tmp_path / "spark.jsonl"
    ledger.write_text("".join(new_entry(row).model_dump_json() + "\n" for row in rows))
    targets = tmp_path / "run.jsonl"
    targets.write_text(json.dumps({"target": target().model_dump(mode="json"), "temperatures": "spec"}) + "\n")
    arms = tmp_path / "arms.json"
    arms.write_text(json.dumps([Arm(name="spark", seed=7).model_dump(mode="json")]))
    return [
        "fix",
        finding().id,
        "--ledger",
        str(ledger),
        "--targets",
        str(targets),
        "--arms",
        str(arms),
        "--out-dir",
        str(tmp_path / "out"),
        *extra,
    ]


def invoke(args: argparse.Namespace) -> int:
    value: int = args.func(args)
    return value


def test_success_prints_each_guard_run_and_exact_status_and_appends_only_fix_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wiring = Wiring()
    argv = inputs(tmp_path, finding(), finding("held"))
    before = (tmp_path / "spark.jsonl").read_bytes()
    assert invoke(parser(wiring.build()).parse_args(argv)) == 0
    lines = capsys.readouterr().out.splitlines()
    assert lines[0].startswith("iteration 1: guard accepted")
    assert [line for line in lines if ": shown seed " in line] == [
        f"iteration 1: shown seed {seed}: pass" for seed in range(5)
    ]
    assert lines[-1] == "survived_fresh_attack"
    assert "coverage: unchecked (not measured)" in lines
    assert (tmp_path / "spark.jsonl").read_bytes() == before
    record = FixRecord.model_validate_json((tmp_path / "out" / "fixes.jsonl").read_text())
    assert record.finding_id == finding().id
    assert wiring.calls == ["client", "phase", "reattack", "release"]
    assert wiring.reattack_dirs == [tmp_path / "out"]


def test_fix_records_do_not_pollute_directory_arm_rollup(tmp_path: Path) -> None:
    from receipts import cli

    wiring = Wiring()
    args = parser(wiring.build()).parse_args(inputs(tmp_path, finding(), finding("held")))
    assert invoke(args) == 0
    assert (args.out_dir / "fixes.jsonl").is_file()
    assert not (tmp_path / "fixes.jsonl").exists()
    # Keep the run input JSONL separate, as production run files are.
    args.targets.rename(tmp_path / "targets.input")
    assert cli.main(["ledger", "rollup", "--ledger", str(tmp_path), "--json"]) == 0


def test_reattack_factory_writes_ledger_in_requested_out_dir(tmp_path: Path) -> None:
    new = finding("new", Status.SUSPECTED)
    wiring = Wiring(attack=Reattack(findings=(new,), micro_usd=579))
    destination = tmp_path / "out" / "nested"
    argv = inputs(tmp_path, finding(), finding("held"))
    argv[argv.index("--out-dir") + 1] = str(destination)

    def factory(out_dir: Path) -> ReAttack:
        assert out_dir == destination and out_dir.is_dir()
        ledger = JsonlLedger(out_dir / "reattack.jsonl")

        def attack(target: Target, arm: Arm, *, budget_micro_usd: int) -> ReAttackResult:
            result = wiring.attack(target, arm, budget_micro_usd=budget_micro_usd)
            for item in result.findings:
                ledger.append(new_entry(item))
            return result

        return attack

    seams = FixWiring(client=wiring.client, run_phase=wiring.run_phase, reattack=factory)
    assert invoke(parser(seams).parse_args(argv)) == 0
    saved = JsonlLedger(destination / "reattack.jsonl", create=False, read_only=True).entries()
    assert [entry.finding for entry in saved] == [
        new.model_copy(
            update={"id": f"{wiring.attack.calls[0][0].id}/claim/new", "target_id": wiring.attack.calls[0][0].id}
        )
    ]
    assert not (tmp_path / "reattack.jsonl").exists()
    assert not (tmp_path / "out" / "reattack.jsonl").exists()
    record = FixRecord.model_validate_json((destination / "fixes.jsonl").read_text())
    assert record.micro_usd == wiring.client_value.cost + wiring.attack.micro_usd


def test_no_held_out_is_printed_honestly_and_exits_one(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wiring = Wiring()
    assert invoke(parser(wiring.build()).parse_args(inputs(tmp_path, finding()))) == 1
    output = capsys.readouterr().out
    assert "held_out: none" in output and output.endswith("passes_shown_repros\n")
    assert "fixed and verified" not in output


def test_empty_reattack_prints_reason_and_cannot_exit_zero(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wiring = Wiring(attack=Reattack(findings=()))
    assert invoke(parser(wiring.build()).parse_args(inputs(tmp_path, finding(), finding("held")))) == 1
    output = capsys.readouterr().out
    assert "fresh attack: fresh attack planned or ran no attacks\n" in output
    assert output.endswith("passes_shown_repros\n")
    record = FixRecord.model_validate_json((tmp_path / "out" / "fixes.jsonl").read_text())
    assert record.reattack_reason == "fresh attack planned or ran no attacks"


def test_forged_pytest_evidence_does_not_override_independent_runner(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    class ForgedRunner(Runner):
        def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
            return tuple(
                case.model_copy(update={"outcome": Outcome.PASS, "failure_type": None})
                for case in super().cases(output, report, changed_files)
            )

    wiring = Wiring()
    wiring.phase.runners[RunnerKind.PYTEST] = ForgedRunner()
    assert invoke(parser(wiring.build()).parse_args(inputs(tmp_path, finding(), finding("held")))) == 0
    assert capsys.readouterr().out.endswith("survived_fresh_attack\n")
    record = FixRecord.model_validate_json((tmp_path / "out" / "fixes.jsonl").read_text())
    assert record.reason is None and all(check.passed for check in record.independent_checks)
    assert any(not check.valid for check in record.canary_checks)
    assert "reattack" in wiring.calls and wiring.calls[-1] == "release"


def test_latest_nonblocker_exits_three_before_wiring_factories(tmp_path: Path) -> None:
    wiring = Wiring()
    args = parser(wiring.build()).parse_args(inputs(tmp_path, finding(), finding(status=Status.REFUTED)))
    assert invoke(args) == 3
    assert not wiring.calls
    assert not (tmp_path / "out" / "fixes.jsonl").exists()


@pytest.mark.parametrize("option", ["0", "4", "bad"])
def test_iteration_cli_range_is_one_through_three(tmp_path: Path, option: str) -> None:
    wiring = Wiring()
    with pytest.raises(SystemExit) as error:
        parser(wiring.build()).parse_args(inputs(tmp_path, finding(), extra=("--max-iterations", option)))
    assert error.value.code == 2 and not wiring.calls


def test_max_iterations_three_stops_at_third_distinct_failure(tmp_path: Path) -> None:
    wiring = Wiring(phase=Phase(sandbox=Box(failures=["a", "b", "c"])))
    args = parser(wiring.build()).parse_args(inputs(tmp_path, finding(), extra=("--max-iterations", "3")))
    # Each fake Spark submission reads a different combined patch from its workspace.
    box = wiring.phase.sandbox
    count = 0
    original_read = box.copy_out_text

    def read(workspace: Any, rel_path: str, *, max_bytes: int = 200 * 1024) -> str:
        nonlocal count
        if rel_path != DIFF_PATH:
            return original_read(workspace, rel_path, max_bytes=max_bytes)
        count += 1
        from tests.agentloop.test_fixloop import diff

        patch = diff(value=f"value = {count}")
        from tests.agentloop.test_fixloop import fixture_images

        box.trees[workspace.id].update(fixture_images(patch))
        return patch

    box.copy_out_text = read  # type: ignore[method-assign]
    assert invoke(args) == 1
    record = FixRecord.model_validate_json((tmp_path / "out" / "fixes.jsonl").read_text())
    assert len(record.iterations) == 3 and count == 3


@pytest.mark.parametrize("broken", ["ledger", "targets", "arms", "missing", "duplicate", "arm-mismatch", "unwired"])
def test_bad_inputs_exit_two_before_factories(tmp_path: Path, broken: str) -> None:
    wiring = Wiring()
    argv = inputs(tmp_path, finding())
    if broken in ("ledger", "targets", "arms"):
        suffix = {"ledger": "spark.jsonl", "targets": "run.jsonl", "arms": "arms.json"}[broken]
        (tmp_path / suffix).write_text("bad input\n")
    elif broken == "missing":
        argv[1] = "example-1/claim/absent"
    elif broken == "duplicate":
        contents = (tmp_path / "run.jsonl").read_text()
        (tmp_path / "run.jsonl").write_text(contents + contents)
    elif broken == "arm-mismatch":
        (tmp_path / "arms.json").write_text('[{"name":"other"}]')
    args = parser(None if broken == "unwired" else wiring.build()).parse_args(argv)
    assert invoke(args) == 2
    assert not wiring.calls


def test_reattack_factory_failure_is_recorded_before_model_spend_and_releases_phase(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wiring = Wiring()
    payload = "model output; repository text; https://endpoint.invalid/private\x1b[2J"

    def unavailable(out_dir: Path) -> ReAttack:
        raise RuntimeError(payload)

    seams = FixWiring(client=wiring.client, run_phase=wiring.run_phase, reattack=unavailable)
    assert invoke(parser(seams).parse_args(inputs(tmp_path, finding(), finding("held")))) == 1
    assert wiring.calls == ["client", "phase", "release"]
    record = FixRecord.model_validate_json((tmp_path / "out" / "fixes.jsonl").read_text())
    assert record.status.value == "unchecked" and record.reason == "runtime"
    assert "RuntimeError" in record.detail and payload not in record.detail
    assert "endpoint.invalid" not in record.model_dump_json()
    output = capsys.readouterr().out
    assert "\x1b" not in output and "RuntimeError" in output
    assert record.micro_usd == 0 and not wiring.client_value.requests


def test_cleanup_failure_is_reported_without_overriding_the_record(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    wiring = Wiring()

    def release() -> None:
        raise RuntimeError("cleanup failure")

    def phase() -> Phase:
        wiring.phase.release = release
        return wiring.phase

    seams = FixWiring(client=wiring.client, run_phase=phase, reattack=wiring.reattack)
    assert invoke(parser(seams).parse_args(inputs(tmp_path, finding(), finding("held")))) == 1
    record = FixRecord.model_validate_json((tmp_path / "out" / "fixes.jsonl").read_text())
    assert record.status.value == "survived_fresh_attack"
    assert "cleanup failed: RuntimeError" in capsys.readouterr().err


def test_untrusted_exception_text_is_visible(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    wiring = Wiring()
    argv = inputs(tmp_path, finding())
    argv[1] = "example-1/claim/escape\x1b[2J\u202e"
    assert invoke(parser(wiring.build()).parse_args(argv)) == 2
    output = capsys.readouterr().err
    assert "\x1b" not in output and "\u202e" not in output


def test_every_printed_evidence_line_uses_visible(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from receipts.agentloop import fix_command

    seen: list[str] = []

    def visible(text: str) -> str:
        seen.append(text)
        return text

    monkeypatch.setattr(fix_command, "visible", visible)
    wiring = Wiring()
    assert invoke(parser(wiring.build()).parse_args(inputs(tmp_path, finding()))) == 1
    assert len(seen) == 15
    assert "canary coverage: lane replays only; internal ReAttack runs are not covered" in seen
    assert seen[-1] == "passes_shown_repros"
