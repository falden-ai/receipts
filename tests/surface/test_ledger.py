"""LE1, the JSONL ledger. Each test feeds the ledger broken input or a crash's leftovers and checks that it holds."""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import os
import stat
import subprocess
import sys
import textwrap
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import pytest
from pydantic import ValidationError

from receipts import __version__
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
from receipts.surface import ledger as ledger_module
from receipts.surface.ledger import (
    CorruptLedgerError,
    CriticRollup,
    InvalidEntryError,
    JsonlLedger,
    LatencyBasis,
    Percentiles,
    ReadOnlyLedgerError,
    SchemaVersionError,
    StageRollup,
    TargetRollup,
    TornTail,
    TornTailSaveError,
    Totals,
    UnknownFindingError,
    nearest_rank,
    new_entry,
    rollup,
    utc_stamp,
    validated,
)

SHA_A = "a" * 40
SHA_B = "b" * 40
AT = "2026-10-27T10:00:00Z"
SPARK, WONDER = "muse-spark-1.3", "muse-wonder-1.0"
CLAIM = "nothing_else_changed"  # contract v2: a template claim's id is its type, relative to its target
TIER_A = "20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11099"  # <entry>~<instance_id>

needs_file_modes = pytest.mark.skipif(os.geteuid() == 0, reason="root ignores file modes")


# ---------------------------------------------------------------- builders


def fid(name: str, target_id: str = "t1", claim_id: str = CLAIM) -> str:
    """A finding id the contract accepts: <target_id>/<claim_id>/<attack id or lens>."""
    return f"{target_id}/{claim_id}/{name}"


def run(side: Side, outcome: Outcome, failure: FailureType | None = None, *, stdout: str = "") -> RunResult:
    return RunResult(
        side=side,
        outcome=outcome,
        failure_type=failure,
        exit_code=0 if outcome is Outcome.PASS else 1,
        fresh_container=True,
        duration_s=1.5,
        stdout_tail=stdout,
    )


def repro() -> Repro:
    return Repro(
        id="r1",
        attack_id="a1",
        claim_id=CLAIM,
        lens=Lens.CORRECTNESS,
        kind=CatchKind.REGRESSION,
        files={"tests/test_receipts_r1.py": "def test_x():\n    assert False\n"},
        command=("python", "-m", "pytest", "tests/test_receipts_r1.py"),
    )


def blocker(
    name: str = "f1",
    target_id: str = "t1",
    *,
    finding_id: str | None = None,
    claim_id: str = CLAIM,
    seed: int = 0,
    costs: tuple[StageCost, ...] = (),
    critic_rounds: int = 0,
    wall_seconds: float | None = None,
) -> Finding:
    runs = tuple(run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION) for _ in range(5)) + tuple(
        run(Side.BASE, Outcome.PASS) for _ in range(5)
    )
    return Finding(
        id=finding_id or fid(name, target_id, claim_id),
        target_id=target_id,
        claim_id=claim_id,
        lens=Lens.CORRECTNESS,
        status=Status.BLOCKER,
        kind=CatchKind.REGRESSION,
        repro=repro(),
        runs=runs,
        quote="Fixes the crash when the list is empty.",
        replay=ReplayKey(base_sha=SHA_A, patch_sha=SHA_B, harness_hash="h1", seed=seed, image_digest="sha256:x"),
        costs=costs,
        critic_rounds=critic_rounds,
        wall_seconds=wall_seconds,
    )


def suspected(
    name: str = "s1",
    target_id: str = "t1",
    *,
    costs: tuple[StageCost, ...] = (),
    stdout: str = "",
    critic_rounds: int = 0,
    wall_seconds: float | None = None,
) -> Finding:
    return Finding(
        id=fid(name, target_id),
        target_id=target_id,
        claim_id=CLAIM,
        lens=Lens.REGRESSION,
        status=Status.SUSPECTED,
        # A skipped run travels as skipped: the contract never counts it as a pass.
        runs=(run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION, stdout=stdout), run(Side.BASE, Outcome.SKIPPED)),
        costs=costs,
        critic_rounds=critic_rounds,
        wall_seconds=wall_seconds,
    )


def entry(finding: Finding, at: str = AT) -> LedgerEntry:
    return LedgerEntry(at=at, receipts_version="0.0.1", finding=finding)


def line(e: LedgerEntry) -> bytes:
    return e.model_dump_json().encode() + b"\n"


def cost(
    stage: str,
    model: str | None,
    tokens: tuple[int, int, int],
    micro_usd: int,
    seconds: float,
    *,
    reasoning: int = 0,
) -> StageCost:
    return StageCost(
        stage=stage,
        model=model,
        input_tokens=tokens[0],
        cached_tokens=tokens[1],
        output_tokens=tokens[2],
        reasoning_tokens=reasoning,
        micro_usd=micro_usd,
        seconds=seconds,
    )


def wait_for(path: Path, timeout_s: float = 30.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not path.exists():
        if time.monotonic() > deadline:
            raise AssertionError(f"{path} never appeared")
        time.sleep(0.005)


def stamped(at: str) -> datetime:
    """A UTC stamp as utc_stamp writes it, parsed back; anything else fails the test."""
    return datetime.strptime(at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)


def saved_reports(ledger: JsonlLedger) -> list[TornTail]:
    """The reports in the ledger's torn-tail file."""
    return [TornTail.model_validate_json(raw) for raw in ledger.torn_path.read_bytes().split(b"\n")[:-1]]


# ---------------------------------------------------------------- append and read back


def test_entries_append_and_read_back(tmp_path: Path) -> None:
    path = tmp_path / "runs" / "r1" / "ledger.jsonl"  # missing parents are made
    ledger = JsonlLedger(path)
    written = [entry(blocker("f1", "t1")), entry(suspected("s1", "t2")), entry(blocker("f2", "t1", seed=7))]
    for e in written:
        ledger.append(e)
    assert ledger.entries() == tuple(written)
    assert ledger.entries("t1") == (written[0], written[2])
    assert ledger.entries("t9") == ()
    assert JsonlLedger(path).entries() == tuple(written)  # a fresh open reads the same
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    assert [LedgerEntry.model_validate_json(x) for x in raw.split(b"\n")[:-1]] == written  # one entry per line
    assert written[1].finding.runs[1].outcome is Outcome.SKIPPED  # and it comes back skipped
    assert ledger.torn_tails == ()
    assert not ledger.torn_path.exists()
    assert stat.S_IMODE(path.stat().st_mode) == 0o600


def test_a_tier_a_finding_with_an_extracted_claim_reads_back(tmp_path: Path) -> None:
    # Contract v2 ids: a Tier A target id is <entry>~<instance_id>, an extracted claim's id is its type and 10 hex
    # digits, and the finding id is <target_id>/<claim_id>/<attack id>.
    claim = "tests_pass:0123456789"
    e = entry(blocker("a1", TIER_A, claim_id=claim))
    path = tmp_path / "ledger.jsonl"
    JsonlLedger(path).append(e)
    assert JsonlLedger(path).entries() == (e,)
    assert JsonlLedger(path).replay(f"{TIER_A}/{claim}/a1").target_id == TIER_A


def test_text_that_looks_like_a_line_break_stays_inside_one_record(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    tricky = "one\ntwo\r\nthree\N{LINE SEPARATOR}four\N{PARAGRAPH SEPARATOR}five\x1b[31m\x00 "
    tricky += "caf\N{LATIN SMALL LETTER E WITH ACUTE} \U0001f600"
    e = entry(suspected(stdout=tricky))
    JsonlLedger(path).append(e)
    assert path.read_bytes().count(b"\n") == 1
    assert JsonlLedger(path).entries() == (e,)


def test_every_append_is_synced_after_its_write(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(path)
    sizes_at_sync: list[int] = []

    def record_sync(fd: int) -> None:
        sizes_at_sync.append(os.fstat(fd).st_size)

    monkeypatch.setattr(ledger_module, "_sync", record_sync)
    lines = [line(entry(suspected(f"s{i}"))) for i in range(3)]
    for raw in lines:
        ledger.append(LedgerEntry.model_validate_json(raw))
    assert sizes_at_sync == [len(b"".join(lines[: i + 1])) for i in range(3)]  # one sync per append, after its write


def test_a_new_ledger_makes_every_new_directory_entry_durable(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[Path] = []
    monkeypatch.setattr(ledger_module, "_sync_directory", synced.append)
    path = tmp_path / "runs" / "r1" / "ledger.jsonl"  # runs/ and runs/r1/ are both new
    JsonlLedger(path)
    # The file's entry is in r1, r1's entry is in runs, and runs' entry is in tmp_path: each one synced once.
    assert sorted(synced) == sorted([tmp_path / "runs" / "r1", tmp_path / "runs", tmp_path])
    synced.clear()
    JsonlLedger(path)  # opening an existing ledger creates nothing
    assert synced == []
    JsonlLedger(tmp_path / "runs" / "r2.jsonl")  # a new file in a directory that already exists
    assert synced == [tmp_path / "runs"]


def test_a_missing_ledger_is_not_created_without_create(tmp_path: Path) -> None:
    path = tmp_path / "nothing.jsonl"
    with pytest.raises(FileNotFoundError):
        JsonlLedger(path, create=False)
    with pytest.raises(FileNotFoundError):
        JsonlLedger(path, read_only=True)  # a read-only ledger is never created
    assert not path.exists()


def test_new_entry_stamps_utc_and_refuses_a_naive_time() -> None:
    eastern = timezone(timedelta(hours=-4))
    e = new_entry(suspected(), at=datetime(2026, 10, 27, 6, 0, tzinfo=eastern))
    assert e.at == "2026-10-27T10:00:00.000000Z"
    assert e.receipts_version == __version__
    assert new_entry(suspected()).at.endswith("Z")
    with pytest.raises(ValueError, match="naive"):
        utc_stamp(datetime(2026, 10, 27, 10, 0))
    assert utc_stamp(datetime(2026, 10, 27, 10, 0, tzinfo=UTC)) == "2026-10-27T10:00:00.000000Z"


# ---------------------------------------------------------------- durability plumbing


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_sync_asks_for_a_full_flush_on_macos_and_fsync_elsewhere(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, platform: str
) -> None:
    calls: list[tuple[str, int, int]] = []
    full_fsync = getattr(fcntl, "F_FULLFSYNC", 51)  # 51 is macOS's value; Linux has no such command

    def fake_fcntl(fd: int, cmd: int, *args: object) -> int:
        calls.append(("fcntl", fd, cmd))
        return 0

    def fake_fsync(fd: int) -> None:
        calls.append(("fsync", fd, 0))

    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", full_fsync, raising=False)
    monkeypatch.setattr(fcntl, "fcntl", fake_fcntl)
    monkeypatch.setattr(os, "fsync", fake_fsync)
    fd = os.open(tmp_path / "file", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        ledger_module._sync(fd)
    finally:
        os.close(fd)
    assert calls == ([("fcntl", fd, full_fsync)] if platform == "darwin" else [("fsync", fd, 0)])


def test_sync_falls_back_to_fsync_where_a_full_flush_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    synced: list[int] = []

    def refuse(fd: int, cmd: int, *args: object) -> int:
        raise OSError(errno.ENOTSUP, "F_FULLFSYNC is not supported here")

    monkeypatch.setattr(sys, "platform", "darwin")
    monkeypatch.setattr(fcntl, "F_FULLFSYNC", getattr(fcntl, "F_FULLFSYNC", 51), raising=False)
    monkeypatch.setattr(fcntl, "fcntl", refuse)
    monkeypatch.setattr(os, "fsync", synced.append)
    fd = os.open(tmp_path / "file", os.O_RDWR | os.O_CREAT, 0o600)
    try:
        ledger_module._sync(fd)
    finally:
        os.close(fd)
    assert synced == [fd]


def test_sync_directory_fsyncs_the_directory_itself(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    synced: list[tuple[int, bool]] = []
    real_fsync = os.fsync

    def record(fd: int) -> None:
        st = os.fstat(fd)
        synced.append((st.st_ino, stat.S_ISDIR(st.st_mode)))
        real_fsync(fd)

    monkeypatch.setattr(os, "fsync", record)
    ledger_module._sync_directory(tmp_path)
    assert synced == [(tmp_path.stat().st_ino, True)]


def test_an_append_survives_short_writes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(path)
    real_write = os.write
    writes: list[int] = []

    def short_write(fd: int, data: bytes | memoryview) -> int:
        written = real_write(fd, bytes(data[:7]))  # a pipe, a signal or a full disk can cut a write short
        writes.append(written)
        return written

    monkeypatch.setattr(os, "write", short_write)
    e = entry(blocker("f1"))
    ledger.append(e)
    monkeypatch.undo()
    assert len(writes) > 1
    assert sum(writes) == len(line(e))
    assert JsonlLedger(path).entries() == (e,)


# ---------------------------------------------------------------- validation on write


def _blocker_without_repro() -> object:
    # model_copy skips validation, the way a buggy caller or model_construct would.
    return entry(blocker()).model_copy(update={"finding": blocker().model_copy(update={"repro": None})})


def _infinite_duration() -> object:
    bad_run = run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION).model_copy(update={"duration_s": float("inf")})
    return entry(suspected()).model_copy(update={"finding": suspected().model_copy(update={"runs": (bad_run,)})})


def _finding_as_a_dict() -> object:
    # The wrong type on purpose: model_construct checks nothing.
    return LedgerEntry.model_construct(v=2, at=AT, receipts_version="0.0.1", finding={"id": "f1"})  # type: ignore[arg-type]


def _wall_time(value: float) -> object:
    # The contract refuses an infinite or NaN wall time where a finding is made; model_copy skips that check, and JSON
    # would carry the value as null, which reads back as no wall time at all.
    return entry(suspected()).model_copy(update={"finding": suspected().model_copy(update={"wall_seconds": value})})


def _lone_surrogate_in_output() -> object:
    # The contract refuses it where a run is made; model_copy skips that check, the way a buggy producer could, and no
    # UTF-8 line can hold it.
    bad_run = run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION).model_copy(update={"stdout_tail": "ok \ud800"})
    return entry(suspected()).model_copy(update={"finding": suspected().model_copy(update={"runs": (bad_run,)})})


@pytest.mark.parametrize(
    ("make", "message"),
    [
        (_blocker_without_repro, "fails validation: finding: Value error, a blocker needs a repro"),
        (_infinite_duration, "fails validation: finding.runs.0.duration_s"),
        (lambda: _wall_time(float("inf")), "would read back as something else: finding.wall_seconds does not survive"),
        (lambda: _wall_time(float("nan")), "would read back as something else: finding.wall_seconds does not survive"),
        (_finding_as_a_dict, "cannot be written as JSON"),
        (_lone_surrogate_in_output, "cannot be written as JSON"),
        (blocker, "not Finding"),
    ],
    ids=[
        "a blocker without a repro",
        "an infinite duration",
        "an infinite wall time",
        "a NaN wall time",
        "a finding that is a dict",
        "a lone surrogate in a run's output",
        "not an entry",
    ],
)
def test_an_invalid_entry_is_refused_on_write(tmp_path: Path, make: Callable[[], object], message: str) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(path)
    ledger.append(entry(suspected("s1")))
    before = path.read_bytes()
    with pytest.raises(InvalidEntryError, match=message):
        ledger.append(make())  # type: ignore[arg-type]
    assert path.read_bytes() == before
    with pytest.raises(InvalidEntryError, match=message):
        validated(make())  # what the ticket writer takes its fields from refuses the same entries


def test_validated_is_the_entry_as_a_reader_gets_it() -> None:
    e = entry(blocker())
    assert validated(e) == e
    assert validated(entry(suspected(wall_seconds=12.5))).finding.wall_seconds == 12.5


# ---------------------------------------------------------------- torn tails

FULL = line(entry(suspected("s9")))


@pytest.mark.parametrize(
    "fragment",
    [FULL[:40], FULL[:-1], b"\x00" * 64, b'{"v":1,"at":"\xe2\x80'],
    ids=["cut mid-record", "only the newline missing", "zero fill", "cut inside a UTF-8 character"],
)
def test_a_torn_last_line_is_saved_then_cut_and_reported(
    tmp_path: Path, caplog: pytest.LogCaptureFixture, fragment: bytes
) -> None:
    path = tmp_path / "ledger.jsonl"
    good = [entry(suspected("s1")), entry(suspected("s2"))]
    ledger = JsonlLedger(path)
    for e in good:
        ledger.append(e)
    clean = path.read_bytes()
    with path.open("ab") as fh:
        fh.write(fragment)  # what a crash part way through an append leaves
    before = datetime.now(UTC)
    with caplog.at_level(logging.WARNING, logger="receipts.surface.ledger"):
        reopened = JsonlLedger(path)
    after = datetime.now(UTC)
    assert path.read_bytes() == clean
    assert reopened.entries() == tuple(good)
    [torn] = reopened.torn_tails
    assert (torn.line, torn.offset, torn.dropped, torn.cut) == (3, len(clean), fragment, True)
    assert before <= stamped(torn.at) <= after  # when the ledger found it, in UTC
    assert "cut a torn last line (line 3" in caplog.text
    assert TornTail.model_validate_json(torn.model_dump_json()) == torn  # the report survives JSON, bytes and all
    assert saved_reports(reopened) == [torn]  # and the bytes outlive the process, beside the ledger
    assert reopened.torn_path == tmp_path / "ledger.jsonl.torn"
    assert stat.S_IMODE(reopened.torn_path.stat().st_mode) == 0o600


def test_the_bytes_are_synced_before_the_cut_and_the_cut_is_synced(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    path = tmp_path / "ledger.jsonl"
    JsonlLedger(path).append(entry(suspected("s1")))
    clean = path.read_bytes()
    with path.open("ab") as fh:
        fh.write(FULL[:40])
    ledger_inode = path.stat().st_ino
    events: list[tuple[str, int]] = []

    def record_sync(fd: int) -> None:
        which = "ledger" if os.fstat(fd).st_ino == ledger_inode else "torn-tail file"
        events.append((which, path.stat().st_size))

    directories: list[Path] = []
    monkeypatch.setattr(ledger_module, "_sync", record_sync)
    monkeypatch.setattr(ledger_module, "_sync_directory", directories.append)
    JsonlLedger(path)
    # The torn-tail file is synced while the ledger still holds the line; then the cut ledger is synced.
    assert events == [("torn-tail file", len(clean) + 40), ("ledger", len(clean))]
    assert directories == [tmp_path]  # the new torn-tail file's directory entry is made durable too


@pytest.mark.parametrize("broken", ["a directory", "a dangling link"])
def test_a_torn_tail_whose_bytes_cannot_be_saved_is_not_cut(tmp_path: Path, broken: str) -> None:
    path = tmp_path / "ledger.jsonl"
    JsonlLedger(path).append(entry(suspected("s1")))
    with path.open("ab") as fh:
        fh.write(FULL[:40])
    before = path.read_bytes()
    torn_path = tmp_path / "ledger.jsonl.torn"
    if broken == "a directory":
        torn_path.mkdir()  # the torn-tail file cannot be written, so the bytes have nowhere to go
    else:
        torn_path.symlink_to(tmp_path / "gone" / "torn")
    with pytest.raises(TornTailSaveError) as caught:
        JsonlLedger(path)
    assert isinstance(caught.value, OSError)  # an OSError with the save's errno, never a missing ledger
    assert caught.value.errno == (errno.EISDIR if broken == "a directory" else errno.ENOENT)
    assert str(torn_path) in str(caught.value) and "was not cut" in str(caught.value)
    assert path.read_bytes() == before  # nothing was cut


def test_a_save_that_crashed_part_way_keeps_its_own_line(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    JsonlLedger(path).append(entry(suspected("s1")))
    piece = b'{"path":"x","li'
    (tmp_path / "ledger.jsonl.torn").write_bytes(piece)  # an earlier save that crashed part way
    with path.open("ab") as fh:
        fh.write(FULL[:40])
    ledger = JsonlLedger(path)
    first, report, end = ledger.torn_path.read_bytes().split(b"\n")
    assert (first, end) == (piece, b"")
    assert TornTail.model_validate_json(report) == ledger.torn_tails[0]


def test_a_torn_tail_left_after_open_is_cut_before_the_next_append(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    first, second = entry(suspected("s1")), entry(suspected("s2"))
    with path.open("ab") as fh:
        fh.write(FULL[:40])  # a crash before this ledger was ever opened
    ledger = JsonlLedger(path)
    ledger.append(first)
    with path.open("ab") as fh:
        fh.write(FULL[:30])  # another writer crashed part way through its append
    ledger.append(second)
    assert JsonlLedger(path).entries() == (first, second)
    assert [(t.line, t.dropped) for t in ledger.torn_tails] == [(1, FULL[:40]), (2, FULL[:30])]
    assert saved_reports(ledger) == list(ledger.torn_tails)  # one saved report per cut


def test_a_torn_tail_left_after_open_is_cut_by_the_next_read(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(path)
    ledger.append(entry(suspected("s1")))
    with path.open("ab") as fh:
        fh.write(FULL[:40])
    assert [e.finding.id for e in ledger.entries()] == [fid("s1")]
    assert [t.line for t in ledger.torn_tails] == [2]
    assert path.read_bytes().endswith(b"\n")


# ---------------------------------------------------------------- malformed lines


def _blocker_line_with_four_patch_runs() -> bytes:
    data = json.loads(entry(blocker()).model_dump_json())
    data["finding"]["runs"] = data["finding"]["runs"][1:]
    return json.dumps(data).encode() + b"\n"


def _edited(**changes: object) -> bytes:
    data = json.loads(entry(suspected("s2")).model_dump_json())
    data.update(changes)
    return json.dumps(data).encode() + b"\n"


def _seed_as_a_string() -> bytes:
    data = json.loads(entry(blocker("f2")).model_dump_json())
    data["finding"]["replay"]["seed"] = "0"
    return json.dumps(data).encode() + b"\n"


def _nested_target() -> bytes:
    data = json.loads(entry(suspected("s2")).model_dump_json())
    data["finding"]["target_id"] = "t1/x"  # a day-1 target id; ids that nest made finding ids ambiguous
    data["finding"]["id"] = fid("s2", "t1/x")
    return json.dumps(data).encode() + b"\n"


def _infinite_wall_line() -> bytes:
    raw = entry(suspected("s2", wall_seconds=3.0)).model_dump_json()
    return raw.replace('"wall_seconds":3.0', '"wall_seconds":1e999').encode() + b"\n"  # JSON reads 1e999 as inf


def _id_outside_its_target() -> bytes:
    data = json.loads(entry(suspected("s2")).model_dump_json())
    data["finding"]["id"] = "s2"  # a day-1 id, from before finding ids started with their target id
    return json.dumps(data).encode() + b"\n"


# Each malformed line and the start of the reason the ledger gives for refusing it.
MALFORMED = {
    "not json": (b"{not json\n", "Invalid JSON"),
    "an empty line": (b"\n", "an empty line"),
    "invalid utf-8": (b"\xff\xfe\n", "Invalid JSON"),
    "an older schema version": (_edited(v=1), "ledger schema version 1 is older than version 2"),
    "a newer schema version": (_edited(v=3), "ledger schema version 3 is newer than version 2"),
    "a schema version that is a boolean": (_edited(v=True), "v: Input should be 2"),
    "an infinite wall time": (_infinite_wall_line(), "finding.wall_seconds: Input should be a finite number"),
    "a target id that nests": (_nested_target(), "finding.target_id: Value error, target id 't1/x' has a slash"),
    "an unknown field": (_edited(extra="x"), "extra: Extra inputs are not permitted"),
    "a string where an integer goes": (_seed_as_a_string(), "finding.replay.seed: Input should be a valid integer"),
    "a blocker the hard rules refuse": (
        _blocker_line_with_four_patch_runs(),
        "finding: Value error, a blocker fails on the patch in every run",
    ),
    "a finding id outside its target": (
        _id_outside_its_target(),
        "finding: Value error, a finding id starts with its target id",
    ),
    "a record glued onto a torn one": (FULL[:40] + FULL, "Invalid JSON"),
}


@pytest.mark.parametrize(("bad", "reason"), MALFORMED.values(), ids=MALFORMED.keys())
def test_a_malformed_line_before_the_last_is_refused_by_number(tmp_path: Path, bad: bytes, reason: str) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(line(entry(suspected("s1"))) + bad + line(entry(suspected("s3"))))
    before = path.read_bytes()
    with pytest.raises(CorruptLedgerError) as caught:
        JsonlLedger(path)
    assert caught.value.line == 2
    assert str(caught.value).startswith(f"{path}:2: {reason}")
    assert path.read_bytes() == before  # nothing repaired, nothing cut


def test_a_malformed_line_is_refused_even_with_a_torn_tail_behind_it(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(line(entry(suspected("s1"))) + b"{not json\n" + line(entry(suspected("s3"))) + FULL[:40])
    before = path.read_bytes()
    with pytest.raises(CorruptLedgerError) as caught:
        JsonlLedger(path)
    assert caught.value.line == 2
    assert path.read_bytes() == before  # the torn tail is not cut from a ledger that is refused
    assert not (tmp_path / "ledger.jsonl.torn").exists()


DAY_ONE = (
    b'{"v":1,"at":"2026-10-27T10:00:00Z","receipts_version":"0.0.1","finding":{"id":"s9","target_id":"t1",'
    b'"claim_id":"t1:fixed_x","lens":"regression","status":"suspected"}}\n'
)  # the day-1 shape: a finding id without its target, a claim id with the target inside


@pytest.mark.parametrize(("version", "word"), [(1, "older"), (3, "newer")])
def test_a_line_from_another_schema_version_is_named_as_that_version(tmp_path: Path, version: int, word: str) -> None:
    path = tmp_path / "ledger.jsonl"
    line_v = DAY_ONE.replace(b'"v":1', b'"v":%d' % version)
    path.write_bytes(line(entry(suspected("s1"))) + line_v)
    before = path.read_bytes()
    with pytest.raises(SchemaVersionError) as caught:
        JsonlLedger(path)
    assert isinstance(caught.value, CorruptLedgerError)  # still a refusal by line number
    assert (caught.value.line, caught.value.version) == (2, version)
    assert str(caught.value).startswith(f"{path}:2: ledger schema version {version} is {word} than version 2")
    assert path.read_bytes() == before


def test_a_malformed_last_line_that_ends_in_a_newline_is_refused_not_cut(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    path.write_bytes(line(entry(suspected("s1"))) + b'{"v": 1}\n')
    with pytest.raises(CorruptLedgerError) as caught:
        JsonlLedger(path)
    assert caught.value.line == 2


# ---------------------------------------------------------------- locks

HOLDER = textwrap.dedent(
    """
    import fcntl, os, sys, time
    from pathlib import Path

    path, mode, held, hold_s, released = sys.argv[1:6]
    fd = os.open(path, os.O_RDONLY)
    fcntl.flock(fd, getattr(fcntl, mode))
    Path(held).touch()
    time.sleep(float(hold_s))
    Path(released).write_text(repr(time.time()))
    os.close(fd)
    """
)


def held_elsewhere(path: Path, mode: str, action: Callable[[], object], hold_s: float = 0.5) -> tuple[float, float]:
    """Run `action` while another process holds the ledger's flock in `mode` (LOCK_EX or LOCK_SH) for `hold_s`
    seconds. Returns when the action finished and when the holder let go."""
    tag = f"{mode}-{time.monotonic_ns()}"
    held, released = path.parent / f"held-{tag}", path.parent / f"released-{tag}"
    holder = subprocess.Popen([sys.executable, "-c", HOLDER, str(path), mode, str(held), str(hold_s), str(released)])
    wait_for(held)
    action()
    finished = time.time()
    assert holder.wait(timeout=30) == 0
    return finished, float(released.read_text())


@pytest.mark.parametrize("holder_mode", ["LOCK_EX", "LOCK_SH"])
@pytest.mark.parametrize("call", ["append", "read", "open", "replay", "history"])
def test_a_writable_ledger_waits_for_any_other_holder_of_the_lock(tmp_path: Path, holder_mode: str, call: str) -> None:
    """Every call on a writable ledger may cut a torn line, so each takes the lock exclusively: it waits for a holder
    of either kind. A shared lock in its place would let an append or a cut run beside another process's call. The
    open's scan and the replay and history lookups each scan the file too, so each is checked on its own."""
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(path)
    first, second = entry(suspected("s1")), entry(suspected("s2"))
    ledger.append(first)

    def act() -> None:
        if call == "append":
            ledger.append(second)
        elif call == "read":
            ledger.entries()
        elif call == "open":
            JsonlLedger(path)
        elif call == "replay":
            ledger.replay(fid("s1"))
        else:
            ledger.history(fid("s1"))

    finished, released = held_elsewhere(path, holder_mode, act)
    assert finished >= released  # the call finished only after the holder let go
    assert ledger.entries() == ((first, second) if call == "append" else (first,))


def test_a_read_only_read_waits_for_a_writer(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    e = entry(suspected("s1"))
    JsonlLedger(path).append(e)
    reader = JsonlLedger(path, read_only=True)
    finished, released = held_elsewhere(path, "LOCK_EX", reader.entries)
    assert finished >= released  # a writer's append is never read half done
    assert reader.entries() == (e,)


# ---------------------------------------------------------------- two processes

WRITER = textwrap.dedent(
    """
    import sys, time
    from pathlib import Path
    from receipts.contracts import LedgerEntry
    from receipts.surface.ledger import JsonlLedger

    path, source, ready, go = (Path(arg) for arg in sys.argv[1:5])
    entries = [LedgerEntry.model_validate_json(raw) for raw in source.read_bytes().split(b"\\n") if raw]
    ledger = JsonlLedger(path)
    ready.touch()
    deadline = time.monotonic() + 60
    while not go.exists():
        if time.monotonic() > deadline:
            sys.exit(3)
        time.sleep(0.001)
    for entry in entries:
        ledger.append(entry)
    """
)


def test_two_processes_append_concurrently_without_losing_or_mixing_lines(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    JsonlLedger(path)
    count = 150
    expected: dict[str, list[str]] = {}
    writers = []
    for tag in ("a", "b"):
        batch = [entry(suspected(f"{tag}{i:04d}", f"t-{tag}", stdout=tag * 20_000)) for i in range(count)]
        expected[tag] = [e.finding.id for e in batch]
        source = tmp_path / f"{tag}.jsonl"
        source.write_bytes(b"".join(line(e) for e in batch))
        args = [str(path), str(source), str(tmp_path / f"ready-{tag}"), str(tmp_path / "go")]
        writers.append(subprocess.Popen([sys.executable, "-c", WRITER, *args]))
    for tag in ("a", "b"):
        wait_for(tmp_path / f"ready-{tag}")
    (tmp_path / "go").touch()
    assert [w.wait(timeout=120) for w in writers] == [0, 0]
    raw = path.read_bytes()
    assert raw.endswith(b"\n")
    lines = raw.split(b"\n")[:-1]
    assert len(lines) == 2 * count
    ids = [LedgerEntry.model_validate_json(x, strict=True).finding.id for x in lines]  # every line is whole
    for tag in ("a", "b"):
        assert [i for i in ids if i.startswith(f"t-{tag}/")] == expected[tag]  # nothing lost, each writer's order kept
    assert JsonlLedger(path).torn_tails == ()


# ---------------------------------------------------------------- read-only ledgers


@needs_file_modes
def test_a_read_only_ledger_reads_a_file_it_may_not_write(tmp_path: Path) -> None:
    path = tmp_path / "ledger.jsonl"
    written = [entry(blocker("f1")), entry(suspected("s1"))]
    ledger = JsonlLedger(path)
    for e in written:
        ledger.append(e)
    path.chmod(0o400)  # a CI artifact or a file on a read-only mount
    before = path.read_bytes()
    with pytest.raises(PermissionError):
        JsonlLedger(path, create=False)  # a writer cannot open it
    reader = JsonlLedger(path, read_only=True)
    assert reader.entries() == tuple(written)
    assert reader.replay(fid("f1")).replay == written[0].finding.replay
    with pytest.raises(ReadOnlyLedgerError, match="read-only"):
        reader.append(entry(suspected("s2")))
    assert path.read_bytes() == before


@needs_file_modes
def test_a_read_only_ledger_skips_a_torn_tail_and_leaves_it(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    path = tmp_path / "ledger.jsonl"
    good = entry(suspected("s1"))
    JsonlLedger(path).append(good)
    with path.open("ab") as fh:
        fh.write(FULL[:40])
    before = path.read_bytes()
    path.chmod(0o400)
    with caplog.at_level(logging.WARNING, logger="receipts.surface.ledger"):
        reader = JsonlLedger(path, read_only=True)
        assert reader.entries() == (good,)
        assert reader.entries() == (good,)  # every read rescans, but the line is reported once
    [torn] = reader.torn_tails
    assert (torn.line, torn.offset, torn.dropped, torn.cut) == (2, len(before) - 40, FULL[:40], False)
    assert caplog.text.count("skipped a torn last line") == 1
    assert path.read_bytes() == before
    assert not reader.torn_path.exists()


# ---------------------------------------------------------------- replay lookup


def test_replay_returns_the_latest_entry_for_a_finding(tmp_path: Path) -> None:
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    first = entry(blocker("f1", seed=1))
    later = entry(blocker("f1", seed=2), at="2026-10-27T11:00:00Z")
    for e in (first, entry(suspected("s1")), later):
        ledger.append(e)
    record = ledger.replay(fid("f1"))
    assert record.replay == later.finding.replay
    assert record.repro == later.finding.repro
    assert (record.target_id, record.status, record.kind) == ("t1", Status.BLOCKER, CatchKind.REGRESSION)
    assert (record.line, record.entries, record.at) == (3, 2, later.at)
    assert ledger.history(fid("f1")) == (first, later)
    unchecked = ledger.replay(fid("s1"))
    assert (unchecked.replay, unchecked.repro, unchecked.entries) == (None, None, 1)
    with pytest.raises(UnknownFindingError, match=f"no finding {fid('nope')}"):
        ledger.replay(fid("nope"))


def test_replay_carries_why_the_finding_has_its_status(tmp_path: Path) -> None:
    # Contract v3: the record holds the finding's unchecked reason, its detail, a refutation and its runs, with each
    # run's problem, and says whether the line was written before v3, when refutation also held other reasons.
    unreadable = RunResult(
        side=Side.BASE,
        outcome=Outcome.ERROR,
        failure_type=FailureType.INFRA,
        fresh_container=True,
        duration_s=2.0,
        problem="expected case tests/test_receipts_r1.py::test_x never ran",
    )
    finding = Finding(
        id=fid("u1"),
        target_id="t1",
        claim_id=CLAIM,
        lens=Lens.REGRESSION,
        status=Status.UNCHECKED,
        unchecked_reason=UncheckedReason.RUNTIME,
        runs=(run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION), unreadable),
        detail="the base run could not be read: expected case tests/test_receipts_r1.py::test_x never ran",
    )
    path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(path)
    ledger.append(entry(finding))
    record = ledger.replay(fid("u1"))
    assert (record.unchecked_reason, record.detail, record.refutation) == (
        UncheckedReason.RUNTIME,
        finding.detail,
        None,
    )
    assert record.runs == finding.runs and record.runs[1].problem == unreadable.problem
    assert record.written_before_v3 is False
    # The same line as v2 wrote it, with its reason in refutation and no detail field.
    data = json.loads(line(entry(finding.model_copy(update={"detail": None, "refutation": "no runner is wired"}))))
    del data["finding"]["detail"]
    for result in data["finding"]["runs"]:
        del result["problem"]
    old = tmp_path / "v2.jsonl"
    old.write_bytes(json.dumps(data).encode() + b"\n")
    record = JsonlLedger(old, create=False).replay(fid("u1"))
    assert (record.written_before_v3, record.detail, record.refutation) == (True, None, "no runner is wired")
    assert [r.problem for r in record.runs] == ["", ""]


def test_target_ids_that_nest_cannot_be_made(tmp_path: Path) -> None:
    # Contract v2 refuses a slash in a target id, so t1 and t1/x can no longer both own a finding id.
    with pytest.raises(ValidationError, match="target id 't1/x' has a slash"):
        blocker(finding_id=fid("a1", "t1/x"), target_id="t1/x", seed=2)


# ---------------------------------------------------------------- rollups


def known_set() -> list[LedgerEntry]:
    unchecked = Finding(
        id=fid("u1", "t4"),
        target_id="t4",
        claim_id=CLAIM,
        lens=Lens.MUTATION,
        status=Status.UNCHECKED,
        unchecked_reason=UncheckedReason.BUDGET,
    )
    return [
        entry(
            blocker(
                "f1",
                "t1",
                costs=(
                    cost("plan", WONDER, (1000, 200, 300), 40_000, 2.5, reasoning=120),
                    cost("attack", SPARK, (5000, 1000, 800), 25_000, 30.0),
                    cost("skeptic", None, (0, 0, 0), 0, 12.25),
                ),
                critic_rounds=1,
                wall_seconds=40.0,
            )
        ),
        entry(
            suspected(
                "s1",
                "t2",
                costs=(
                    cost("plan", WONDER, (800, 0, 200), 30_000, 2.0, reasoning=80),
                    cost("attack", SPARK, (4000, 500, 700), 20_000, 25.0),
                ),
                critic_rounds=3,
                wall_seconds=30.0,
            )
        ),
        entry(suspected("s2", "t3", costs=(cost("plan", SPARK, (600, 0, 100), 5_000, 1.5),))),
        # A later entry for f1. Its costs are what the fix stage spent, on top of the first entry's; its critic rounds
        # and wall time are the finding's state so far, so they replace the first entry's rather than add to them.
        entry(
            blocker(
                "f1",
                "t1",
                costs=(cost("fix", SPARK, (2000, 0, 500), 10_000, 8.0),),
                critic_rounds=2,
                wall_seconds=55.5,
            ),
            at="2026-10-27T11:00:00Z",
        ),
        entry(unchecked),
    ]


def test_rollups_on_a_known_set(tmp_path: Path) -> None:
    result = rollup(known_set())
    assert result.stages == (
        StageRollup(
            stage="attack",
            model=SPARK,
            records=2,
            input_tokens=9000,
            cached_tokens=1500,
            output_tokens=1500,
            reasoning_tokens=0,
            micro_usd=45_000,
            seconds=55.0,
        ),
        StageRollup(
            stage="fix",
            model=SPARK,
            records=1,
            input_tokens=2000,
            cached_tokens=0,
            output_tokens=500,
            reasoning_tokens=0,
            micro_usd=10_000,
            seconds=8.0,
        ),
        StageRollup(
            stage="plan",
            model=SPARK,
            records=1,
            input_tokens=600,
            cached_tokens=0,
            output_tokens=100,
            reasoning_tokens=0,
            micro_usd=5_000,
            seconds=1.5,
        ),
        StageRollup(
            stage="plan",
            model=WONDER,
            records=2,
            input_tokens=1800,
            cached_tokens=200,
            output_tokens=500,
            reasoning_tokens=200,
            micro_usd=70_000,
            seconds=4.5,
        ),
        StageRollup(
            stage="skeptic",
            model=None,
            records=1,
            input_tokens=0,
            cached_tokens=0,
            output_tokens=0,
            reasoning_tokens=0,
            micro_usd=0,
            seconds=12.25,
        ),
    )
    wall, summed = LatencyBasis.WALL, LatencyBasis.SUMMED_STAGE_SECONDS
    assert result.targets == (
        TargetRollup(
            target_id="t1",
            entries=2,
            findings=1,
            micro_usd=75_000,
            stage_seconds=52.75,
            wall_seconds=55.5,
            latency_seconds=55.5,
            latency_basis=wall,
            critic_rounds=2,
        ),
        TargetRollup(
            target_id="t2",
            entries=1,
            findings=1,
            micro_usd=50_000,
            stage_seconds=27.0,
            wall_seconds=30.0,
            latency_seconds=30.0,
            latency_basis=wall,
            critic_rounds=3,
        ),
        TargetRollup(
            target_id="t3",
            entries=1,
            findings=1,
            micro_usd=5_000,
            stage_seconds=1.5,
            wall_seconds=None,
            latency_seconds=1.5,
            latency_basis=summed,
            critic_rounds=0,
        ),
        TargetRollup(
            target_id="t4",
            entries=1,
            findings=1,
            micro_usd=0,
            stage_seconds=0.0,
            wall_seconds=None,
            latency_seconds=0.0,
            latency_basis=summed,
            critic_rounds=0,
        ),
    )
    # Per-target costs sorted: 0, 5000, 50000, 75000; latencies: 0.0, 1.5, 30.0, 55.5. Nearest rank: p50 is the 2nd,
    # p95 the 4th. Two targets have a wall time and two fall back, so the latency percentiles say they are mixed.
    assert result.percentiles == Percentiles(
        targets=4,
        p50_micro_usd=5_000,
        p95_micro_usd=75_000,
        p50_seconds=1.5,
        p95_seconds=55.5,
        latency_basis=LatencyBasis.MIXED,
        wall_targets=2,
    )
    # f1 went through 2 rounds (not 1 + 2), s1 through 3, s2 and u1 through none.
    assert result.critic == CriticRollup(findings=4, rounds=5, findings_by_rounds=(2, 0, 1, 1))
    # Every entry's costs count once; each finding counts once, at its latest status (f1's second entry).
    statuses = dict.fromkeys(Status, 0) | {Status.BLOCKER: 1, Status.SUSPECTED: 2, Status.UNCHECKED: 1}
    assert result.totals == Totals(
        entries=5,
        findings=4,
        targets=4,
        micro_usd=130_000,
        input_tokens=13_400,
        cached_tokens=1_700,
        output_tokens=2_600,
        reasoning_tokens=200,
        seconds=81.25,
        statuses=statuses,
    )
    ledger = JsonlLedger(tmp_path / "ledger.jsonl")
    for e in known_set():
        ledger.append(e)
    assert ledger.rollup() == result
    assert [t.target_id for t in ledger.rollup("t2").targets] == ["t2"]
    assert rollup([]).percentiles == Percentiles(
        targets=0,
        p50_micro_usd=None,
        p95_micro_usd=None,
        p50_seconds=None,
        p95_seconds=None,
        latency_basis=None,
        wall_targets=0,
    )
    assert rollup([]).critic == CriticRollup(findings=0, rounds=0, findings_by_rounds=(0, 0, 0, 0))
    empty = rollup([]).totals
    assert (empty.entries, empty.findings, empty.targets, empty.micro_usd, empty.seconds) == (0, 0, 0, 0, 0.0)
    assert empty.statuses == dict.fromkeys(Status, 0)


def test_a_finding_counts_at_its_latest_status() -> None:
    # A suspected finding later refuted counts once, as refuted, and its two entries' costs both count.
    first = entry(suspected("s1", costs=(cost("attack", SPARK, (10, 0, 5), 7, 1.0),)))
    later = entry(
        suspected("s1", costs=(cost("refute", WONDER, (20, 0, 9), 11, 2.0, reasoning=4),)).model_copy(
            update={"status": Status.REFUTED}
        ),
        at="2026-10-27T11:00:00Z",
    )
    totals = rollup([first, later]).totals
    assert (totals.entries, totals.findings, totals.micro_usd, totals.reasoning_tokens) == (2, 1, 18, 4)
    assert {k: v for k, v in totals.statuses.items() if v} == {Status.REFUTED: 1}


def test_latency_is_wall_time_when_recorded_and_labeled_when_it_falls_back() -> None:
    # Two attackers that ran side by side for 90 s each: 180 summed stage seconds, 95 s on the clock.
    parallel = (cost("attack", SPARK, (1, 0, 1), 10, 90.0), cost("attack", SPARK, (1, 0, 1), 10, 90.0))
    timed = [
        entry(suspected("s1", "t1", costs=parallel, wall_seconds=95.0)),
        entry(suspected("s2", "t2", costs=parallel[:1], wall_seconds=91.0)),
    ]
    untimed = [entry(suspected("s1", "t1", costs=parallel)), entry(suspected("s2", "t2", costs=parallel[:1]))]
    p = rollup(timed).percentiles
    assert (p.latency_basis, p.wall_targets, p.p50_seconds, p.p95_seconds) == (LatencyBasis.WALL, 2, 91.0, 95.0)
    p = rollup(untimed).percentiles
    assert (p.latency_basis, p.wall_targets, p.p50_seconds, p.p95_seconds) == (
        LatencyBasis.SUMMED_STAGE_SECONDS,
        0,
        90.0,
        180.0,
    )
    p = rollup([timed[0], untimed[1]]).percentiles
    assert (p.latency_basis, p.wall_targets, p.p50_seconds, p.p95_seconds) == (LatencyBasis.MIXED, 1, 90.0, 95.0)
    [row] = rollup([timed[0]]).targets
    assert (row.stage_seconds, row.wall_seconds, row.latency_seconds) == (180.0, 95.0, 95.0)


def test_critic_rounds_and_wall_time_are_states_not_increments() -> None:
    # Two findings of one target. A later entry for f1 from a stage that records no critic rounds must not erase the
    # 2 it went through, and the target's wall time is the longest any entry reached, whatever order they came in.
    entries = [
        entry(suspected("f1", "t1", critic_rounds=2, wall_seconds=50.0)),
        entry(suspected("f2", "t1", critic_rounds=1, wall_seconds=70.0)),
        entry(suspected("f1", "t1", wall_seconds=60.0), at="2026-10-27T11:00:00Z"),
    ]
    result = rollup(entries)
    [row] = result.targets
    assert (row.findings, row.critic_rounds, row.wall_seconds, row.latency_seconds) == (2, 3, 70.0, 70.0)
    assert result.critic == CriticRollup(findings=2, rounds=3, findings_by_rounds=(0, 1, 1, 0))


def test_nearest_rank_percentiles() -> None:
    values = list(range(1, 21))
    assert nearest_rank(values, 50) == 10
    assert nearest_rank(values, 95) == 19
    assert nearest_rank(values, 100) == 20
    assert nearest_rank([7], 95) == 7
    assert nearest_rank([3, 1, 2], 50) == 2
    assert nearest_rank([0.25, 0.5], 95) == 0.5
    assert nearest_rank([], 50) is None
    for bad in (0, 101, -5):
        with pytest.raises(ValueError, match="percentile"):
            nearest_rank(values, bad)
