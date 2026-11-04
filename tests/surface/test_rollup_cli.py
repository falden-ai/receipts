"""`receipts ledger rollup`, the pilot's cost read: one ledger or a directory of per-arm ledgers, as text or JSON."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable
from pathlib import Path

import pytest

from receipts import cli
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
    UncheckedReason,
)
from receipts.surface.cli_commands import register
from receipts.surface.ledger import JsonlLedger, usd

AT = "2026-10-27T10:00:00Z"
CLAIM = "fixed_x"
SPARK, WONDER, GLIMMER = "muse-spark-1.3", "muse-wonder-1.0", "muse-glimmer-30b"


def cost(stage: str, model: str | None, micro_usd: int, seconds: float, *, tokens: int = 100) -> StageCost:
    return StageCost(
        stage=stage,
        model=model,
        input_tokens=tokens,
        cached_tokens=tokens // 4,
        output_tokens=tokens // 2,
        reasoning_tokens=tokens // 10,
        micro_usd=micro_usd,
        seconds=seconds,
    )


def blocker(target: str, *, costs: tuple[StageCost, ...] = (), wall: float | None = None) -> Finding:
    fail = RunResult(
        side=Side.PATCH, outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION, fresh_container=True, duration_s=1
    )
    passed = RunResult(side=Side.SURROGATE, outcome=Outcome.PASS, fresh_container=True, duration_s=1)
    return Finding(
        id=f"{target}/{CLAIM}/a1",
        target_id=target,
        claim_id=CLAIM,
        lens=Lens.CORRECTNESS,
        status=Status.BLOCKER,
        kind=CatchKind.FIX_VALIDITY,
        repro=Repro(
            id="r1",
            attack_id="a1",
            claim_id=CLAIM,
            lens=Lens.CORRECTNESS,
            kind=CatchKind.FIX_VALIDITY,
            files={"tests/test_receipts_1.py": "def test_x():\n    assert False\n"},
            command=("python", "-m", "pytest", "tests/test_receipts_1.py"),
        ),
        runs=(fail,) * 5 + (passed,),
        quote="Fixes the crash when the list is empty.",
        replay=ReplayKey(base_sha="a" * 40, patch_sha="b" * 40, harness_hash="h", seed=0, image_digest="sha256:x"),
        costs=costs,
        wall_seconds=wall,
    )


def unchecked(target: str, *, costs: tuple[StageCost, ...] = ()) -> Finding:
    return Finding(
        id=f"{target}/{CLAIM}/correctness",
        target_id=target,
        claim_id=CLAIM,
        lens=Lens.CORRECTNESS,
        status=Status.UNCHECKED,
        unchecked_reason=UncheckedReason.BUDGET,
        costs=costs,
    )


def write(path: Path, *findings: Finding) -> Path:
    ledger = JsonlLedger(path)
    for finding in findings:
        ledger.append(LedgerEntry(at=AT, receipts_version="0.0.1", finding=finding))
    return path


def spark_arm(path: Path) -> Path:
    return write(
        path,
        blocker("t1", costs=(cost("plan", WONDER, 40_000, 2.0), cost("attack", SPARK, 25_000, 30.0)), wall=40.0),
        unchecked("t2", costs=(cost("attack", SPARK, 250_000, 90.0),)),
    )


def call(*argv: str) -> int:
    parser = argparse.ArgumentParser(prog="receipts")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(list(argv))
    code: int = args.func(args)
    return code


def test_usd_is_exact() -> None:
    assert usd(0) == "$0.000000"
    assert usd(1) == "$0.000001"
    assert usd(1_234_567) == "$1.234567"
    assert usd(150_000_000) == "$150.000000"
    assert usd(1_000_000_000_000) == "$1,000,000.000000"
    # Past 2**53 a float cannot hold every micro-dollar; integer division can.
    assert usd(123_456_789_012_345_678) == "$123,456,789,012.345678"


def test_rollup_prints_spend_statuses_and_the_stage_table(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = spark_arm(tmp_path / "spark.jsonl")
    assert call("ledger", "rollup", "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert f"ledger {path} (arm spark): 2 entries, 2 findings, 2 targets\n" in out
    assert "spend $0.315000; tokens in 300 (cached 75), out 150 (reasoning 30); stage seconds 122.0\n" in out
    assert "statuses: blocker 1, suspected 0, flaky 0, refuted 0, possibly_toxic 0, s_sat_inconclusive 0, " in out
    assert "not_reproduced 0, unchecked 1\n" in out
    # Per-target costs 65000 and 250000; latencies 40.0 (wall) and 90.0 (summed), so the basis is mixed.
    assert "per target: cost p50 $0.065000, p95 $0.250000; latency p50 40.0 s, p95 90.0 s (mixed: 1 of 2" in out
    lines = out.splitlines()
    header = next(x for x in lines if x.startswith("stage"))
    assert header.split() == ["stage", "model", "records", "input", "cached", "output", "reasoning", "spend", "seconds"]
    assert next(x for x in lines if x.startswith("attack")).split() == [
        "attack",
        SPARK,
        "2",
        "200",
        "50",
        "100",
        "20",
        "$0.275000",
        "120.0",
    ]
    assert next(x for x in lines if x.startswith("plan")).split()[:3] == ["plan", WONDER, "1"]


def test_rollup_json_is_the_ledgers_rollup(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = spark_arm(tmp_path / "spark.jsonl")
    assert call("ledger", "rollup", "--ledger", str(path), "--json") == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == {
        "ledgers": [{"arm": "spark", "path": str(path), "rollup": JsonlLedger(path).rollup().model_dump(mode="json")}]
    }
    assert printed["ledgers"][0]["rollup"]["totals"]["statuses"]["blocker"] == 1


def test_a_directory_is_read_arm_by_arm(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    arms = tmp_path / "pilot"
    spark_arm(arms / "spark.jsonl")
    write(arms / "glimmer.jsonl", unchecked("t1", costs=(cost("attack", GLIMMER, 0, 300.0),)), unchecked("t2"))
    (arms / "spark.jsonl.torn").write_text("not a ledger\n")  # a torn-tail file is not an arm
    (arms / "notes.txt").write_text("not a ledger\n")
    assert call("ledger", "rollup", "--ledger", str(arms)) == 0
    out = capsys.readouterr().out
    assert out.index("(arm glimmer)") < out.index("(arm spark)")
    assert "(arm spark.jsonl" not in out and "notes" not in out
    table = out[out.index("\narms\n") + 1 :].split("\n\n")[0].splitlines()  # the title, the header, one row per arm
    assert table[1].split()[:5] == ["arm", "targets", "findings", "blockers", "unchecked"]
    assert [row.split() for row in table[2:]] == [
        ["glimmer", "2", "2", "0", "2", "$0.000000", "$0.000000", "$0.000000"],
        ["spark", "2", "2", "1", "1", "$0.315000", "$0.065000", "$0.250000"],
    ]


def test_rollup_of_one_target_and_by_target_rows(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = spark_arm(tmp_path / "spark.jsonl")
    assert call("ledger", "rollup", "--ledger", str(path), "--target", "t2", "--by-target") == 0
    out = capsys.readouterr().out
    assert "1 entries, 1 findings, 1 targets" in out
    rows = [x.split() for x in out.splitlines() if x.startswith(("target", "t1", "t2"))]
    assert rows == [
        ["target", "entries", "findings", "spend", "latency", "basis", "critic", "rounds"],
        ["t2", "1", "1", "$0.250000", "90.0", "summed_stage_seconds", "0"],
    ]
    assert "summed stage seconds, no target timed: parallel stages count in full" in out


def test_rollup_refuses_a_missing_ledger_or_an_empty_directory(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    assert call("ledger", "rollup", "--ledger", str(tmp_path / "nope.jsonl")) == 1
    assert f"receipts ledger rollup: no ledger at {tmp_path / 'nope.jsonl'}" in capsys.readouterr().err
    assert not (tmp_path / "nope.jsonl").exists()
    (tmp_path / "empty").mkdir()
    assert call("ledger", "rollup", "--ledger", str(tmp_path / "empty")) == 1
    assert "no ledgers (*.jsonl) in" in capsys.readouterr().err


def test_a_corrupt_arm_fails_the_command_and_the_others_still_print(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    arms = tmp_path / "pilot"
    spark_arm(arms / "spark.jsonl")
    good = (arms / "spark.jsonl").read_bytes()
    (arms / "glimmer.jsonl").write_bytes(good + b"{not json\n" + good)
    assert call("ledger", "rollup", "--ledger", str(arms)) == 1
    captured = capsys.readouterr()
    assert f"{arms / 'glimmer.jsonl'}:3: " in captured.err
    assert "(arm spark)" in captured.out


def test_rollup_cuts_and_reports_a_torn_tail(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = spark_arm(tmp_path / "spark.jsonl")
    clean = path.read_bytes()
    with path.open("ab") as fh:
        fh.write(b'{"v":2,"at":"2026')
    assert call("ledger", "rollup", "--ledger", str(path)) == 0
    captured = capsys.readouterr()
    assert "receipts ledger rollup: " in captured.err and "cut a torn last line (line 3, 17 bytes)" in captured.err
    assert path.read_bytes() == clean
    assert "2 entries" in captured.out


def test_rollup_escapes_ledger_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    hostile_stage = "attack\x1b]52;c;ZXZpbA==\x07\N{RIGHT-TO-LEFT OVERRIDE}"
    path = write(tmp_path / "x.jsonl", unchecked("t\x9b1", costs=(cost(hostile_stage, SPARK, 5, 1.0),)))
    assert call("ledger", "rollup", "--ledger", str(path), "--by-target") == 0
    out = capsys.readouterr().out
    assert not [ch for ch in ("\x1b", "\x07", "\N{RIGHT-TO-LEFT OVERRIDE}", "\x9b") if ch in out]
    assert "attack\\u001b]52;c;ZXZpbA==\\u0007\\u202e" in out
    assert "t\\u009b1" in out


def test_the_top_level_command_dispatches_ledger_rollup(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    lanes: tuple[Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None], ...] = (register,)
    monkeypatch.setattr(cli, "LANE_COMMANDS", lanes)
    path = spark_arm(tmp_path / ".receipts" / "ledger.jsonl")
    monkeypatch.chdir(tmp_path)
    assert cli.main(["ledger", "rollup"]) == 0  # the default ledger, as replay reads it
    assert "(arm ledger): 2 entries" in capsys.readouterr().out
    assert path.exists()


# ---------------------------------------------------------------- the pilot's directory of arms


def arms_file(path: Path, *names: str) -> Path:
    path.write_text(json.dumps([{"name": name} for name in names]))
    return path


def test_a_directory_is_read_without_changes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A rollup of a run's directory changes nothing in it: a torn tail stays for the run's own writer to recover,
    # and a stray file that happens to end in .jsonl is never cut or moved.
    arms = tmp_path / "pilot"
    spark = spark_arm(arms / "spark.jsonl")
    with spark.open("ab") as fh:
        fh.write(b'{"v":2,"at":"2026')
    (arms / "notes.jsonl").write_bytes(b'{"a note": "with no newline"}')
    before = {p.name: p.read_bytes() for p in arms.iterdir()}
    assert call("ledger", "rollup", "--ledger", str(arms)) == 0
    captured = capsys.readouterr()
    assert {p.name: p.read_bytes() for p in arms.iterdir()} == before  # no byte changed, no .torn file made
    assert "skipped a torn last line (line 3, 17 bytes)" in captured.err
    assert "a directory of ledgers is read without changes, so the line stays in the file" in captured.err
    assert "(arm spark): 2 entries" in captured.out


def test_the_arms_file_names_exactly_the_arms_read(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    run_dir = tmp_path / "pilot"
    spark_arm(run_dir / "spark.jsonl")
    write(run_dir / "glimmer.jsonl", unchecked("t1"))
    (run_dir / "spend.jsonl").write_bytes(b'{"armed": true}\n')  # not a ledger: never read as one
    (run_dir / "targets").mkdir()
    arms = arms_file(tmp_path / "arms.json", "spark", "glimmer")
    assert call("ledger", "rollup", "--ledger", str(run_dir), "--arms", str(arms)) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert captured.out.index("(arm spark)") < captured.out.index("(arm glimmer)")  # the arms file's order
    table = captured.out.split("\narms\n")[1].split("\n\n")[0].splitlines()
    assert [row.split()[0] for row in table[1:]] == ["spark", "glimmer"] and "(arm spend)" not in captured.out
    # An arm whose ledger is missing fails the command; the others still print.
    arms = arms_file(tmp_path / "arms.json", "spark", "glimmer", "llama")
    assert call("ledger", "rollup", "--ledger", str(run_dir), "--arms", str(arms)) == 1
    captured = capsys.readouterr()
    assert f"no ledger at {run_dir / 'llama.jsonl'} for arm llama" in captured.err
    assert "(arm spark)" in captured.out and "(arm glimmer)" in captured.out


@pytest.mark.parametrize(
    ("content", "message"),
    [
        (b"not json", "is not a JSON list of arms"),
        (b'[{"name": "spark", "attacker_model": "muse-spark-1.3-contributor"}]', "is not a JSON list of arms"),
        (b'[{"name": "Spark Arm"}]', "is not a JSON list of arms"),
        (b"[]", "names no arm"),
        (b'[{"name": "spark"}, {"name": "spark", "seed": 1}]', "names an arm twice"),
    ],
    ids=["not json", "a contributor model", "a bad name", "no arms", "an arm twice"],
)
def test_a_bad_arms_file_is_refused_before_any_ledger_is_read(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], content: bytes, message: str
) -> None:
    run_dir = tmp_path / "pilot"
    spark_arm(run_dir / "spark.jsonl")
    (tmp_path / "arms.json").write_bytes(content)
    assert call("ledger", "rollup", "--ledger", str(run_dir), "--arms", str(tmp_path / "arms.json")) == 1
    captured = capsys.readouterr()
    assert message in captured.err and captured.out == ""


def test_an_arms_file_needs_a_directory_of_ledgers(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = spark_arm(tmp_path / "spark.jsonl")
    arms = arms_file(tmp_path / "arms.json", "spark")
    assert call("ledger", "rollup", "--ledger", str(path), "--arms", str(arms)) == 1
    assert "--arms reads a directory of <arm>.jsonl ledgers" in capsys.readouterr().err
    assert call("ledger", "rollup", "--ledger", str(tmp_path), "--arms", str(tmp_path / "missing.json")) == 1
    assert "the arms file" in capsys.readouterr().err


def test_the_rollup_says_what_its_spend_leaves_out(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # The run's shared calls (intake and the plan, Wonder's included) are on its per-target summaries, so a reader of
    # the ledgers' spend is told so, once.
    path = spark_arm(tmp_path / "spark.jsonl")
    assert call("ledger", "rollup", "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert (
        out.count(
            "\nnote: spend counts what these ledgers' entries record; costs no finding carries (intake, "
            "the plan, a failed run's spend) are on the run's per-target summaries, outside the ledgers"
        )
        == 1
    )
