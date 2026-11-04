"""The `replay` stub: it prints a finding's replay key and repro, refuses what it cannot replay, reads a ledger it may
not write, and never lets model or ledger text act on the terminal."""

from __future__ import annotations

import argparse
import errno
import json
import os
import subprocess
import sys
import textwrap
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
    Status,
    UncheckedReason,
)
from receipts.surface.cli_commands import register
from receipts.surface.ledger import JsonlLedger

SHA_A = "a" * 40
SHA_B = "b" * 40
AT = "2026-10-27T10:00:00Z"
TEST_FILE = "tests/test_receipts_r1.py"
CLAIM = "nothing_else_changed"  # contract v2: a template claim's id is its type
FID = f"t1/{CLAIM}/a1"
SID = f"t1/{CLAIM}/s1"
KEY = ReplayKey(base_sha=SHA_A, patch_sha=SHA_B, harness_hash="h1", seed=3, image_digest="sha256:abc")
# Text that acts on a terminal: an OSC 52 clipboard write, a bell, a carriage return, a right-to-left override, a zero
# width space, DEL, a C1 control sequence introducer, and the line and paragraph separators.
HOSTILE = (
    "x\x1b]52;c;ZXZpbA==\x07\r\N{RIGHT-TO-LEFT OVERRIDE}gnp\N{ZERO WIDTH SPACE}\x7f\x9b2J"
    "\N{LINE SEPARATOR}\N{PARAGRAPH SEPARATOR}"
)
RAW = ("\x1b", "\x07", "\r", "\N{RIGHT-TO-LEFT OVERRIDE}", "\N{ZERO WIDTH SPACE}", "\x7f", "\x9b", "\N{LINE SEPARATOR}")
ESCAPED = "x\\u001b]52;c;ZXZpbA==\\u0007\\u000d\\u202egnp\\u200b\\u007f\\u009b2J\\u2028\\u2029"
# What an id may hold under contract v2 (no whitespace, C0 control, DEL or slash), and still acts on a terminal.
ID_HOSTILE = "t\N{RIGHT-TO-LEFT OVERRIDE}\x9b2J\N{ZERO WIDTH SPACE}"
ID_ESCAPED = "t\\u202e\\u009b2J\\u200b"

needs_file_modes = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")


def run(side: Side, outcome: Outcome, failure: FailureType | None = None) -> RunResult:
    return RunResult(side=side, outcome=outcome, failure_type=failure, fresh_container=True, duration_s=1.0)


def make_repro(
    files: dict[str, str] | None = None,
    command: tuple[str, ...] | None = None,
    *,
    repro_id: str = "r1",
    attack_id: str = "a1",
    claim_id: str = CLAIM,
    cases: tuple[str, ...] = (),
) -> Repro:
    return Repro(
        id=repro_id,
        attack_id=attack_id,
        claim_id=claim_id,
        lens=Lens.CORRECTNESS,
        kind=CatchKind.REGRESSION,
        files=files or {TEST_FILE: "def test_empty():\n    assert first([]) is None\n"},
        command=command or ("python", "-m", "pytest", TEST_FILE),
        cases=cases,
    )


def blocker(
    finding_id: str = FID,
    target_id: str = "t1",
    *,
    seed: int = 3,
    files: dict[str, str] | None = None,
    command: tuple[str, ...] | None = None,
    repro: Repro | None = None,
    replay: ReplayKey | None = None,
    detail: str | None = None,
    refutation: str | None = None,
) -> Finding:
    runs = tuple(run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION) for _ in range(5)) + tuple(
        run(Side.BASE, Outcome.PASS) for _ in range(5)
    )
    return Finding(
        id=finding_id,
        target_id=target_id,
        claim_id=CLAIM,
        lens=Lens.CORRECTNESS,
        status=Status.BLOCKER,
        kind=CatchKind.REGRESSION,
        repro=repro or make_repro(files, command),
        runs=runs,
        quote="Fixes the crash when the list is empty.",
        replay=replay
        or ReplayKey(base_sha=SHA_A, patch_sha=SHA_B, harness_hash="h1", seed=seed, image_digest="sha256:abc"),
        detail=detail,
        refutation=refutation,
    )


def suspected(finding_id: str = SID, *, replay: ReplayKey | None = None, repro: Repro | None = None) -> Finding:
    return Finding(
        id=finding_id,
        target_id="t1",
        claim_id=CLAIM,
        lens=Lens.REGRESSION,
        status=Status.SUSPECTED,
        replay=replay,
        repro=repro,
    )


def ledger_with(path: Path, *findings: Finding) -> Path:
    ledger = JsonlLedger(path)
    for finding in findings:
        ledger.append(LedgerEntry(at=AT, receipts_version="0.0.1", finding=finding))
    return path


def call(*argv: str) -> int:
    """Parse the way `receipts.cli` does once `register` is wired in, then run the handler."""
    parser = argparse.ArgumentParser(prog="receipts")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(list(argv))
    code: int = args.func(args)
    return code


def tear(path: Path, fragment: bytes = b'{"v":2,"at":"2026-10-27T1') -> bytes:
    """Leave what a crash part way through an append leaves; return the fragment."""
    with path.open("ab") as fh:
        fh.write(fragment)
    return fragment


# ---------------------------------------------------------------- printing


def test_replay_prints_the_replay_key_and_repro(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    assert call("replay", FID, "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert f"finding {FID}  target t1  status blocker  kind regression" in out
    assert f"ledger {path}, line 1, written {AT}\n" in out
    for name, value in (
        ("base_sha", SHA_A),
        ("patch_sha", SHA_B),
        ("harness_hash", "h1"),
        ("seed", "3"),
        ("image_digest", "sha256:abc"),
    ):
        assert f"  {name:<13}{value}\n" in out
    assert f"  command  python -m pytest {TEST_FILE}\n" in out
    assert f"  file {TEST_FILE}\n    | def test_empty():\n    |     assert first([]) is None\n" in out


def test_replay_reads_the_default_ledger_under_the_current_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    ledger_with(tmp_path / ".receipts" / "ledger.jsonl", blocker())
    monkeypatch.chdir(tmp_path)
    assert call("replay", FID) == 0
    assert f"ledger {Path('.receipts') / 'ledger.jsonl'}, line 1" in capsys.readouterr().out


def test_the_top_level_command_dispatches_replay(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The integrator's dispatch: main() returns args.func(args). The annotation lets mypy check register's type.
    lanes: tuple[Callable[[argparse._SubParsersAction[argparse.ArgumentParser]], None], ...] = (register,)
    monkeypatch.setattr(cli, "LANE_COMMANDS", lanes)
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    assert cli.main(["replay", FID, "--ledger", str(path)]) == 0
    assert "status blocker" in capsys.readouterr().out
    assert cli.main(["replay", f"t1/{CLAIM}/nope", "--ledger", str(path)]) == 1


def test_replay_shows_the_latest_of_several_entries(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker(seed=1), suspected(), blocker(seed=9))
    assert call("replay", FID, "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert f"ledger {path}, line 3, written {AT} (the latest of 2 entries for this finding)\n" in out
    assert f"  {'seed':<13}9\n" in out


def test_replay_json_is_the_record(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    assert call("replay", FID, "--ledger", str(path), "--json") == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed == JsonlLedger(path).replay(FID).model_dump(mode="json")
    assert printed["replay"]["seed"] == 3


# ---------------------------------------------------------------- escaping


def test_replay_escapes_control_characters_from_model_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(
        tmp_path / "ledger.jsonl",
        blocker(files={TEST_FILE: HOSTILE}, command=("python", "-c", "print(1)\x1b[2J")),
    )
    assert call("replay", FID, "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert not [ch for ch in RAW if ch in out]
    assert f"    | {ESCAPED}\n" in out
    assert "  command  python -c 'print(1)\\u001b[2J'\n" in out
    assert call("replay", FID, "--ledger", str(path), "--json") == 0
    out = capsys.readouterr().out
    assert not [ch for ch in RAW if ch in out]
    assert json.loads(out)["repro"]["files"][TEST_FILE] == HOSTILE  # escaped, not lost


def test_replay_escapes_every_field_it_prints(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # Each field the text form prints, one hostile value each: a repro file's name, the repro, attack and claim ids,
    # the replay key's text fields, the target and finding ids, and the ledger's own path.
    target = ID_HOSTILE
    finding_id = f"{target}/{CLAIM}/a1"
    repro = make_repro(
        files={f"tests/{HOSTILE}.py": "assert False\n"},
        repro_id=f"r{HOSTILE}",
        attack_id=f"a{HOSTILE}",
        claim_id=f"c{ID_HOSTILE}",
    )
    key = ReplayKey(base_sha=SHA_A, patch_sha=SHA_B, harness_hash=f"h{HOSTILE}", seed=3, image_digest=f"d{HOSTILE}")
    folder = tmp_path / f"run{HOSTILE.replace(chr(0x2028), '').replace(chr(0x2029), '')}"
    path = ledger_with(folder / "ledger.jsonl", blocker(finding_id, target, repro=repro, replay=key))
    assert call("replay", finding_id, "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert not [ch for ch in RAW if ch in out], [ch for ch in RAW if ch in out]
    assert f"  file tests/{ESCAPED}.py\n" in out
    assert f"repro r{ESCAPED}  attack a{ESCAPED}  claim c{ID_ESCAPED}  lens correctness" in out
    assert f"  harness_hash h{ESCAPED}\n" in out
    assert f"  image_digest d{ESCAPED}\n" in out
    assert f"finding {ID_ESCAPED}/{CLAIM}/a1  target {ID_ESCAPED}  status blocker" in out
    assert f"ledger {tmp_path}/run{ESCAPED.replace(chr(92) + 'u2028', '').replace(chr(92) + 'u2029', '')}/ledger" in out


def test_errors_on_stderr_are_escaped(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    # A malformed line whose unknown field name would act on the terminal: the refusal quotes the name.
    path = tmp_path / "ledger.jsonl"
    data = json.loads(LedgerEntry(at=AT, receipts_version="0.0.1", finding=blocker()).model_dump_json())
    data["\x1b]52;c;ZXZpbA==\x07\N{LINE SEPARATOR}"] = "x"
    path.write_bytes(json.dumps(data).encode() + b"\n")
    raw = ("\x1b", "\x07", "\N{LINE SEPARATOR}")
    assert call("replay", FID, "--ledger", str(path)) == 1
    err = capsys.readouterr().err
    assert not [ch for ch in raw if ch in err]
    assert "\\u001b]52;c;ZXZpbA==\\u0007\\u2028: Extra inputs are not permitted" in err
    # A finding id typed at the shell is echoed back escaped too.
    good = ledger_with(tmp_path / "good.jsonl", blocker())
    assert call("replay", "t1/\x1b[2J", "--ledger", str(good)) == 1
    err = capsys.readouterr().err
    assert "\x1b" not in err
    assert "no finding t1/\\u001b[2J in" in err
    # Bytes that are not UTF-8 reach Python's argv as lone surrogates, which no terminal can be sent as they are.
    assert call("replay", "t1/\udcff\udc9b", "--ledger", str(good)) == 1
    err = capsys.readouterr().err
    assert "no finding t1/\\udcff\\udc9b in" in err
    assert call("replay", FID, "--ledger", str(tmp_path / "missing\udcff" / "ledger.jsonl")) == 1
    assert "no ledger at " in (err := capsys.readouterr().err) and "missing\\udcff" in err


# ---------------------------------------------------------------- what cannot be replayed


def test_replay_refuses_a_finding_with_nothing_to_replay(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", suspected())
    assert call("replay", SID, "--ledger", str(path)) == 1
    captured = capsys.readouterr()
    assert "replay key: none recorded" in captured.out
    assert "repro: none recorded" in captured.out
    assert "has no replay key or repro recorded, so it cannot be replayed" in captured.err


@pytest.mark.parametrize(
    ("finding", "missing"),
    [(suspected(replay=KEY), "repro"), (suspected(repro=make_repro()), "replay key")],
    ids=["a replay key but no repro", "a repro but no replay key"],
)
def test_replay_refuses_a_finding_missing_either_half(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], finding: Finding, missing: str
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", finding)
    assert call("replay", finding.id, "--ledger", str(path)) == 1
    assert f"has no {missing} recorded, so it cannot be replayed" in capsys.readouterr().err


def test_replay_of_an_unknown_finding_fails(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    assert call("replay", f"t1/{CLAIM}/nope", "--ledger", str(path)) == 1
    assert f"no finding t1/{CLAIM}/nope" in capsys.readouterr().err


def test_replay_without_a_ledger_fails_and_creates_nothing(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "missing" / "ledger.jsonl"
    assert call("replay", FID, "--ledger", str(path)) == 1
    assert f"no ledger at {path}" in capsys.readouterr().err
    assert not path.parent.exists()


def test_replay_refuses_a_corrupt_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    good = path.read_bytes()
    path.write_bytes(good + b"{not json\n" + good)
    assert call("replay", FID, "--ledger", str(path)) == 1
    assert f"{path}:2: " in capsys.readouterr().err


def test_replay_refuses_a_line_whose_target_id_nests(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    good = path.read_bytes()
    nested = good.replace(b'"target_id":"t1"', b'"target_id":"t1/x"').replace(
        f'"id":"{FID}"'.encode(), f'"id":"t1/x/{CLAIM}/a1"'.encode()
    )
    assert nested != good
    path.write_bytes(good + nested)
    assert call("replay", FID, "--ledger", str(path)) == 1
    assert f"{path}:2: finding.target_id: Value error, target id 't1/x' has a slash" in capsys.readouterr().err


def test_replay_names_a_day_one_ledger_as_an_older_schema(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(
        b'{"v":1,"at":"2026-10-27T10:00:00Z","receipts_version":"0.0.1","finding":{"id":"s9","target_id":"t1",'
        b'"claim_id":"t1:fixed_x","lens":"regression","status":"suspected"}}\n'
    )
    assert call("replay", FID, "--ledger", str(path)) == 1
    err = capsys.readouterr().err
    assert f"{path}:1: ledger schema version 1 is older than version 2, the one this receipts reads" in err


# ---------------------------------------------------------------- torn tails and read-only ledgers


def test_replay_saves_and_reports_a_torn_tail_it_cut(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    clean = path.read_bytes()
    fragment = tear(path)
    assert call("replay", FID, "--ledger", str(path)) == 0
    err = capsys.readouterr().err
    torn_path = tmp_path / "ledger.jsonl.torn"
    assert f"cut a torn last line (line 2, {len(fragment)} bytes) left by a crashed append" in err
    assert f"its bytes are saved in {torn_path}" in err
    assert path.read_bytes() == clean
    assert [t.dropped for t in JsonlLedger(path, read_only=True).torn_tails] == []
    report = json.loads(torn_path.read_bytes().split(b"\n")[0])
    assert (report["line"], report["cut"]) == (2, True)


@needs_file_modes
def test_replay_reads_a_ledger_it_may_not_write(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    path.chmod(0o400)
    assert call("replay", FID, "--ledger", str(path)) == 0
    captured = capsys.readouterr()
    assert "status blocker" in captured.out
    assert captured.err == ""


@needs_file_modes
def test_replay_leaves_a_torn_tail_in_a_ledger_it_may_not_write(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    fragment = tear(path)
    before = path.read_bytes()
    path.chmod(0o400)
    assert call("replay", FID, "--ledger", str(path)) == 0
    err = capsys.readouterr().err
    assert f"skipped a torn last line (line 2, {len(fragment)} bytes)" in err
    assert "the ledger is read-only here, so the line stays in the file" in err
    assert path.read_bytes() == before
    assert not (tmp_path / "ledger.jsonl.torn").exists()


@needs_file_modes
def test_replay_reads_without_changes_where_the_bytes_cannot_be_saved(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # The ledger itself is writable, but its directory is not, so a torn line's bytes have nowhere to go.
    folder = tmp_path / "run"
    path = ledger_with(folder / "ledger.jsonl", blocker())
    tear(path)
    before = path.read_bytes()
    folder.chmod(0o500)
    try:
        assert call("replay", FID, "--ledger", str(path)) == 0
    finally:
        folder.chmod(0o700)
    assert "skipped a torn last line (line 2" in capsys.readouterr().err
    assert path.read_bytes() == before


@pytest.mark.parametrize("broken", ["a directory", "a dangling link"])
def test_replay_reads_without_changes_where_the_torn_tail_file_is_broken(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], broken: str
) -> None:
    # The ledger is writable, but its torn-tail path cannot take the bytes: that is no reason to call the ledger
    # missing or to refuse to read it.
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    tear(path)
    before = path.read_bytes()
    torn_path = tmp_path / "ledger.jsonl.torn"
    if broken == "a directory":
        torn_path.mkdir()
    else:
        torn_path.symlink_to(tmp_path / "gone" / "torn")
    assert call("replay", FID, "--ledger", str(path)) == 0
    captured = capsys.readouterr()
    assert "status blocker" in captured.out
    assert "skipped a torn last line (line 2" in captured.err
    assert "no ledger" not in captured.err
    assert path.read_bytes() == before


def test_a_dangling_ledger_link_is_a_missing_ledger(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    path = tmp_path / "ledger.jsonl"
    path.symlink_to(tmp_path / "gone.jsonl")
    assert call("replay", FID, "--ledger", str(path)) == 1
    assert f"no ledger at {path}" in capsys.readouterr().err


def test_replay_reads_a_ledger_on_a_read_only_mount(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    real_open = os.open

    def read_only_mount(file: str | os.PathLike[str], flags: int, *args: int) -> int:
        if flags & (os.O_WRONLY | os.O_RDWR):
            raise OSError(errno.EROFS, os.strerror(errno.EROFS), str(file))
        return real_open(file, flags, *args)

    monkeypatch.setattr(os, "open", read_only_mount)
    assert call("replay", FID, "--ledger", str(path)) == 0
    assert "status blocker" in capsys.readouterr().out


REPLAY = textwrap.dedent(
    """
    import argparse, sys
    from receipts.surface.cli_commands import register

    parser = argparse.ArgumentParser(prog="receipts")
    register(parser.add_subparsers(dest="command", required=True))
    args = parser.parse_args(sys.argv[1:])
    sys.exit(args.func(args))
    """
)


def test_a_cut_is_reported_once_outside_pytest(tmp_path: Path) -> None:
    # Outside pytest nothing configures logging, so Python's last-resort handler would print the ledger's warning on
    # top of the command's own line.
    path = ledger_with(tmp_path / "ledger.jsonl", blocker())
    tear(path)
    done = subprocess.run(
        [sys.executable, "-c", REPLAY, "replay", FID, "--ledger", str(path)],
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert done.stderr.count("torn last line") == 1, done.stderr


# ---------------------------------------------------------------- why a finding has its status (contract v3)


def unreadable(problem: str = "the run wrote no report") -> RunResult:
    return RunResult(
        side=Side.PATCH,
        outcome=Outcome.ERROR,
        failure_type=FailureType.INFRA,
        exit_code=2,
        seed=0,
        fresh_container=True,
        duration_s=4.0,
        problem=problem,
    )


def unchecked_finding(
    detail: str | None, *, runs: tuple[RunResult, ...] = (), refutation: str | None = None
) -> Finding:
    return Finding(
        id=f"t1/{CLAIM}/u1",
        target_id="t1",
        claim_id=CLAIM,
        lens=Lens.REGRESSION,
        status=Status.UNCHECKED,
        unchecked_reason=UncheckedReason.RUNTIME,
        runs=runs,
        detail=detail,
        refutation=refutation,
    )


def as_v2(finding: Finding) -> bytes:
    """A ledger line exactly as contract v2 wrote it: no Finding.detail, Repro.cases or RunResult.problem key."""
    data = json.loads(LedgerEntry(at=AT, receipts_version="0.0.1", finding=finding).model_dump_json())
    del data["finding"]["detail"]
    if data["finding"]["repro"] is not None:
        del data["finding"]["repro"]["cases"]
    for result in data["finding"]["runs"]:
        del result["problem"]
    return json.dumps(data).encode() + b"\n"


def test_replay_shows_why_an_unchecked_finding_is_unchecked(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    finding = unchecked_finding("the patch run could not be read: the run wrote no report", runs=(unreadable(),))
    path = ledger_with(tmp_path / "ledger.jsonl", finding)
    assert call("replay", finding.id, "--ledger", str(path)) == 1  # nothing to replay, but the reason is shown
    out = capsys.readouterr().out
    assert "status unchecked  kind none\n" in out
    assert "unchecked reason: runtime\n" in out
    assert "detail, why it is unchecked\n  | the patch run could not be read: the run wrote no report\n" in out
    assert "refutation" not in out
    rows = out.split("\nruns\n", 1)[1].split("\n")
    assert rows[0].split() == ["#", "side", "outcome", "failure", "exit", "fresh", "seed", "seconds", "problem"]
    assert rows[1].split()[:8] == ["1", "patch", "error", "infra", "2", "yes", "0", "4.0"]
    assert rows[1].endswith("  the run wrote no report")  # the problem beside the outcome it explains


def test_replay_shows_a_detail_and_a_different_models_refutation_apart(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    finding = blocker(
        detail="Failed on the patch 5 of 5 times and passed on base 5 of 5.",
        refutation="Wonder says the test is too strict.\nIt did not hold up.",
    )
    path = ledger_with(tmp_path / "ledger.jsonl", finding)
    assert call("replay", FID, "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert "detail, why it is blocker\n  | Failed on the patch 5 of 5 times and passed on base 5 of 5.\n" in out
    assert "refutation by a different model\n  | Wonder says the test is too strict.\n  | It did not hold up.\n" in out
    assert out.index("detail, why") < out.index("refutation by") < out.index("replay key")
    assert "before contract v3" not in out
    rows = out.split("\nruns\n", 1)[1].split("\n")
    assert [r.split()[1:3] for r in rows[1:11]] == [["patch", "fail"]] * 5 + [["base", "pass"]] * 5
    assert call("replay", FID, "--ledger", str(path), "--json") == 0
    printed = json.loads(capsys.readouterr().out)
    assert (printed["detail"], printed["refutation"], printed["written_before_v3"]) == (
        finding.detail,
        finding.refutation,
        False,
    )
    assert len(printed["runs"]) == 10 and printed["runs"][0]["problem"] == ""


def test_replay_names_a_pre_v3_refutation_for_what_it_may_be(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    # Before contract v3 the pipeline put reasons that were not refutations, such as "no candidate repro", in
    # Finding.refutation. A line from then has no detail field, and its refutation is not called a different model's.
    finding = Finding(
        id=f"t1/{CLAIM}/a9",
        target_id="t1",
        claim_id=CLAIM,
        lens=Lens.CORRECTNESS,
        status=Status.FLAKY,
        kind=CatchKind.REGRESSION,
        repro=make_repro(),
        runs=(run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION), run(Side.PATCH, Outcome.PASS)),
        refutation="failed on the patch 1 of 2 times",
    )
    old = tmp_path / "v2.jsonl"
    old.write_bytes(as_v2(finding))
    assert b'"detail"' not in old.read_bytes() and b'"problem"' not in old.read_bytes()
    assert call("replay", finding.id, "--ledger", str(old)) == 1  # no replay key
    out = capsys.readouterr().out
    assert (
        "refutation, from a line written before contract v3, when this field also held reasons that were not a "
        "different model's refutation\n  | failed on the patch 1 of 2 times\n"
    ) in out
    assert "refutation by a different model" not in out and "detail: none recorded" not in out
    assert "  cases    every case its files define\n" in out  # a v2 repro has no cases: every case
    assert call("replay", finding.id, "--ledger", str(old), "--json") == 1  # printed, then refused: no replay key
    printed = json.loads(capsys.readouterr().out)
    assert printed["written_before_v3"] is True and printed["detail"] is None
    # The same finding written since v3 records its (empty) detail, so its refutation is a different model's.
    new = ledger_with(tmp_path / "v3.jsonl", finding)
    assert call("replay", finding.id, "--ledger", str(new)) == 1
    out = capsys.readouterr().out
    assert "refutation by a different model\n  | failed on the patch 1 of 2 times\n" in out
    assert "detail: none recorded\n" in out


def test_replay_escapes_reasons_problems_and_cases(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    finding = unchecked_finding(f"d{HOSTILE}", runs=(unreadable(f"p{HOSTILE}"),), refutation=f"r{HOSTILE}")
    path = ledger_with(tmp_path / "ledger.jsonl", finding)
    assert call("replay", finding.id, "--ledger", str(path)) == 1
    out = capsys.readouterr().out
    assert not [ch for ch in RAW if ch in out], [ch for ch in RAW if ch in out]
    assert f"  | d{ESCAPED}\n" in out and f"  | r{ESCAPED}\n" in out
    assert "  px\\u001b]52;c;ZXZpbA==\\u0007 \\u202egnp\\u200b\\u007f\\u009b2J" in out  # the problem, on one line
    hostile_case = blocker(repro=make_repro(cases=(f"{TEST_FILE}::test_{HOSTILE}",)))
    path = ledger_with(tmp_path / "cases.jsonl", hostile_case)
    assert call("replay", FID, "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert not [ch for ch in RAW if ch in out]
    assert f"  cases\n    {TEST_FILE}::test_{ESCAPED}\n" in out


def test_replay_shows_the_cases_a_repro_runs(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    cases = (f"{TEST_FILE}::test_empty", f"{TEST_FILE}::test_one")
    path = ledger_with(tmp_path / "ledger.jsonl", blocker(repro=make_repro(cases=cases)))
    assert call("replay", FID, "--ledger", str(path)) == 0
    out = capsys.readouterr().out
    assert f"  cases\n    {cases[0]}\n    {cases[1]}\n" in out
    assert out.index("  command  ") < out.index("  cases\n") < out.index("  file ")
