"""LE1, the ledger: an append-only JSONL file of `LedgerEntry` records, one per line.

Every line is validated when it is written and again when it is read. Each call holds a flock(2) on the ledger file
itself, exclusive for a writer, so several processes can append to one ledger, and an append syncs the file to disk
before it returns. Any other tool that writes a ledger takes the same lock.

The newline ends a record. An append that crashed part way leaves a last line with no newline, and the next open,
append or read deals with it. The line's bytes are saved first, to the torn-tail file beside the ledger
(`<ledger>.torn`, one `TornTail` JSON report per line, the bytes in URL-safe base64), and synced; only then is the
line cut from the ledger and the cut reported (`JsonlLedger.torn_tails` and a warning on this module's logger). If
the bytes cannot be saved, nothing is cut and the open raises TornTailSaveError. A ledger opened read-only, for a file
this process may not write, takes a shared lock, never writes, and skips a torn last line instead of cutting it. A
malformed line anywhere else is refused by its line number, and the file is left untouched for a person to look at;
a line written under another schema version is refused as that version, so a day-1 ledger is named as one.

An entry is written only if it reads back as itself. An entry built without validation (model_construct, model_copy)
could hold a value JSON cannot carry, such as an infinite or NaN float, which would be written as null and read back
as something else; the ledger refuses it instead.

The ledger never rewrites a line, so a finding's later state is a new entry with the same finding id: the latest
entry is the finding's current state. An entry's costs are what producing that entry spent, never a restatement of
an earlier entry's costs, so rollups add up the costs of every entry. Critic rounds and wall time are states, not
increments: a finding's critic rounds are the most any of its entries records, and a target's wall time is the
longest any of its entries records.
"""

from __future__ import annotations

import fcntl
import json
import logging
import math
import os
import sys
from collections.abc import Iterable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import TYPE_CHECKING

from pydantic import ConfigDict, ValidationError

from receipts import __version__
from receipts.contracts import (
    SCHEMA_VERSION,
    CatchKind,
    Finding,
    Frozen,
    LedgerEntry,
    ReplayKey,
    Repro,
    RunResult,
    Status,
    UncheckedReason,
)

if TYPE_CHECKING:
    from receipts.contracts import Ledger

logger = logging.getLogger(__name__)
# A library logs and the program decides where it goes. Without this, Python's last-resort handler prints every
# warning to stderr a second time in a program that reports cuts itself, as the CLI does.
logger.addHandler(logging.NullHandler())

_READ_CHUNK = 1 << 20
_FILE_MODE = 0o600
TORN_SUFFIX = ".torn"
CRITIC_ROUNDS_CAP = 3  # the spec's at most 3 critic and rewrite rounds, which Finding.critic_rounds also caps


# ---------------------------------------------------------------- errors


class LedgerError(Exception):
    """A ledger that cannot be read or written as it stands."""


class CorruptLedgerError(LedgerError):
    """A malformed line before the last one. The ledger repairs nothing and refuses to go on."""

    def __init__(self, path: Path, line: int, reason: str) -> None:
        self.path = path
        self.line = line
        self.reason = reason
        super().__init__(f"{path}:{line}: {reason}")


class SchemaVersionError(CorruptLedgerError):
    """A line written under another ledger schema version (`LedgerEntry.v`). It is refused as that version rather
    than read as a malformed record, and nothing is migrated: version 1 is the day-1 shape, whose ids changed."""

    def __init__(self, path: Path, line: int, version: int) -> None:
        self.version = version
        if version < SCHEMA_VERSION:
            reason = (
                f"ledger schema version {version} is older than version {SCHEMA_VERSION}, the one this receipts reads: "
                "the line comes from an earlier receipts and is not migrated"
            )
        else:
            reason = (
                f"ledger schema version {version} is newer than version {SCHEMA_VERSION}, the one this receipts reads: "
                "read it with a newer receipts"
            )
        super().__init__(path, line, reason)


class InvalidEntryError(LedgerError):
    """An entry refused on write: it is not a LedgerEntry, cannot be written as JSON, fails validation, or would read
    back as something else."""


class ReadOnlyLedgerError(LedgerError):
    """An append to a ledger opened read-only."""


class TornTailSaveError(LedgerError, OSError):
    """A torn last line whose bytes could not be saved to the torn-tail file, so nothing was cut. It is an OSError
    with the save's errno; a reader can still open the ledger read-only, which leaves the line in place."""


class UnknownFindingError(LedgerError, LookupError):
    """No entry in the ledger has this finding id."""


# ---------------------------------------------------------------- reports


class TornTail(Frozen):
    """A last line that a crashed append left without its newline. A writable ledger saves it to its torn-tail file
    and then cuts it (`cut` is true); a read-only ledger skips it and leaves it in the file. The bytes need not be
    UTF-8, so JSON carries them as URL-safe base64 (`-` and `_`, not `+` and `/`): decode them with
    `TornTail.model_validate_json` or `base64.urlsafe_b64decode`, not a plain `base64 -d`."""

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_bytes="base64", val_json_bytes="base64")

    path: str  # the ledger
    line: int  # the line number the record would have had
    offset: int  # where the line starts, which is the ledger's length after a cut
    dropped: bytes
    at: str  # when the ledger found it, in UTC
    cut: bool


class ReplayRecord(Frozen):
    """What `receipts replay <id>` needs: the replay key and repro from a finding's latest entry, and why the finding
    has its status: its unchecked reason, its detail, a different model's refutation, and the runs behind it.

    Contract v3 split the reasons: `detail` holds every reason that is not a different-model refutation, and
    `refutation` only that. A line written before v3 has no detail field, and its refutation could hold any reason
    (the day-2 pipeline put unchecked causes and "no candidate repro" there), so `written_before_v3` says which kind of
    line this is. The ledger writes every field, so a line written since v3 always has one."""

    finding_id: str
    target_id: str
    status: Status
    kind: CatchKind | None
    at: str
    line: int  # the latest entry's line in the ledger
    entries: int  # how many entries the ledger holds for this finding
    replay: ReplayKey | None
    repro: Repro | None
    unchecked_reason: UncheckedReason | None = None
    detail: str | None = None
    refutation: str | None = None
    written_before_v3: bool = False
    runs: tuple[RunResult, ...] = ()


class LatencyBasis(StrEnum):
    """What a latency figure measures. Only WALL is wall-clock latency; show the others with their label."""

    WALL = "wall"  # Finding.wall_seconds: from the target's start to its latest status
    SUMMED_STAGE_SECONDS = "summed_stage_seconds"  # the fallback: stages that ran in parallel count in full
    MIXED = "mixed"  # percentiles over targets of both kinds


class StageRollup(Frozen):
    """Everything one stage spent on one model; `model` is None for a code-only stage. Reasoning tokens are part of
    the output tokens, counted again on their own because Wonder meters them separately."""

    stage: str
    model: str | None
    records: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    micro_usd: int
    seconds: float


class TargetRollup(Frozen):
    """One target's spend, time and critic rounds. Its latency is its wall time when any of its entries records one,
    else its summed stage seconds, and `latency_basis` says which."""

    target_id: str
    entries: int
    findings: int
    micro_usd: int
    stage_seconds: float  # every stage's seconds added up, so stages that ran in parallel count in full
    wall_seconds: float | None  # the longest Finding.wall_seconds among its entries; None when none records one
    latency_seconds: float
    latency_basis: LatencyBasis  # WALL or SUMMED_STAGE_SECONDS
    critic_rounds: int  # its findings' critic rounds added up


class Percentiles(Frozen):
    """Nearest-rank p50 and p95 across targets of each target's cost and latency; None when there are no targets.

    `latency_basis` is WALL only when every target has a wall time, SUMMED_STAGE_SECONDS when none has one (the
    fallback, which overstates latency when stages run in parallel), and MIXED otherwise. Only WALL percentiles are
    wall-clock latency."""

    targets: int
    p50_micro_usd: int | None
    p95_micro_usd: int | None
    p50_seconds: float | None
    p95_seconds: float | None
    latency_basis: LatencyBasis | None
    wall_targets: int  # how many targets have a wall time


class CriticRollup(Frozen):
    """Critic and rewrite rounds (AT2, at most 3 per finding) across the findings."""

    findings: int
    rounds: int
    findings_by_rounds: tuple[int, ...]  # index n: how many findings went through n rounds


class Totals(Frozen):
    """The whole ledger (or one target of it): what every entry spent, and how many findings ended in each status."""

    entries: int
    findings: int
    targets: int
    micro_usd: int
    input_tokens: int
    cached_tokens: int
    output_tokens: int
    reasoning_tokens: int
    seconds: float  # every stage's seconds added up
    statuses: dict[Status, int]  # each finding counted once, at its latest entry's status; every status is listed


class Rollup(Frozen):
    stages: tuple[StageRollup, ...]
    targets: tuple[TargetRollup, ...]
    percentiles: Percentiles
    critic: CriticRollup
    totals: Totals


# ---------------------------------------------------------------- the ledger


class JsonlLedger:
    """An append-only JSONL ledger at `path` (the `Ledger` interface).

    Opening it validates every line and deals with a torn last line. Each call opens the file on its own and holds the
    lock only for that call, so one instance can be shared by threads and a forked child never inherits a held lock.
    With `create` the file and its parent directories are made when missing; without it a missing file raises
    FileNotFoundError. `read_only` opens an existing ledger without write access, for a file, directory or mount this
    process may not write: it creates nothing, takes a shared lock, refuses appends and leaves a torn last line in
    place.
    """

    def __init__(self, path: str | os.PathLike[str], *, create: bool = True, read_only: bool = False) -> None:
        self.path = Path(path)
        self.read_only = read_only
        self._torn: list[TornTail] = []
        if create and not read_only:
            self._create()
        with self._locked() as fd:
            self._scan(fd)

    @property
    def torn_path(self) -> Path:
        """Where the bytes of a cut torn line are saved: `<ledger>.torn`, one TornTail JSON report per line."""
        return self.path.with_name(self.path.name + TORN_SUFFIX)

    @property
    def torn_tails(self) -> tuple[TornTail, ...]:
        """Every torn last line this instance cut, or skipped when read-only, oldest first."""
        return tuple(self._torn)

    def append(self, entry: LedgerEntry) -> None:
        if self.read_only:
            raise ReadOnlyLedgerError(f"{self.path} was opened read-only")
        line = _encode(entry)
        with self._locked() as fd:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                # A writer crashed after this ledger was opened; cut its line first, or this record would be glued
                # onto it and the ledger would hold a malformed middle line.
                self._scan(fd)
            _write_all(fd, line)
            _sync(fd)

    def entries(self, target_id: str | None = None) -> tuple[LedgerEntry, ...]:
        with self._locked() as fd:
            rows = self._scan(fd)
        return tuple(entry for _, entry in rows if target_id is None or entry.finding.target_id == target_id)

    def history(self, finding_id: str) -> tuple[LedgerEntry, ...]:
        """Every entry for one finding, oldest first."""
        return tuple(entry for _, entry in self._finding_rows(finding_id))

    def replay(self, finding_id: str) -> ReplayRecord:
        """The replay key and repro of the finding's latest entry."""
        rows = self._finding_rows(finding_id)
        line, latest = rows[-1]
        finding = latest.finding
        return ReplayRecord(
            finding_id=finding.id,
            target_id=finding.target_id,
            status=finding.status,
            kind=finding.kind,
            at=latest.at,
            line=line,
            entries=len(rows),
            replay=finding.replay,
            repro=finding.repro,
            unchecked_reason=finding.unchecked_reason,
            detail=finding.detail,
            refutation=finding.refutation,
            # Every line is read from its JSON, so the fields it was given are exactly the keys the line holds.
            written_before_v3="detail" not in finding.model_fields_set,
            runs=finding.runs,
        )

    def rollup(self, target_id: str | None = None) -> Rollup:
        return rollup(self.entries(target_id))

    # ------------------------------------------------------------ internals

    def _finding_rows(self, finding_id: str) -> list[tuple[int, LedgerEntry]]:
        with self._locked() as fd:
            rows = [(number, entry) for number, entry in self._scan(fd) if entry.finding.id == finding_id]
        if not rows:
            raise UnknownFindingError(f"no finding {finding_id} in {self.path}")
        # Contract v2: a target id holds no slash, so a finding id names one target and its rows are one history.
        return rows

    def _create(self) -> None:
        made = _make_directories(self.path.parent)
        try:
            fd = os.open(self.path, os.O_RDWR | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, _FILE_MODE)
        except FileExistsError:
            fd = None  # another process made it first, and syncs its own entry
        if fd is not None:
            try:
                _sync(fd)
            finally:
                os.close(fd)
            _sync_directory(self.path.parent)  # the new file's entry
        for directory in reversed(made):
            _sync_directory(directory.parent)  # each directory this call made has its entry in its parent

    @contextmanager
    def _locked(self) -> Iterator[int]:
        """The ledger's descriptor under flock: exclusive for a writable ledger, since any of its calls may cut a torn
        line, and shared for a read-only one, which never writes."""
        if self.read_only:
            fd = os.open(self.path, os.O_RDONLY | os.O_CLOEXEC)
            mode = fcntl.LOCK_SH
        else:
            fd = os.open(self.path, os.O_RDWR | os.O_APPEND | os.O_CLOEXEC)
            mode = fcntl.LOCK_EX
        try:
            fcntl.flock(fd, mode)
            try:
                yield fd
            finally:
                fcntl.flock(fd, fcntl.LOCK_UN)
        finally:
            os.close(fd)

    def _scan(self, fd: int) -> list[tuple[int, LedgerEntry]]:
        """Validate every line, then cut a torn last line, or skip it when read-only. Call it with the lock held."""
        data = _read_all(fd)
        lines = data.split(b"\n")  # never splitlines(): a record may hold U+2028 or a carriage return
        tail = lines.pop()  # empty when the file ends with a newline
        rows = [(number, self._parse(raw, number)) for number, raw in enumerate(lines, start=1)]
        if tail:
            torn = TornTail(
                path=_display(self.path),
                line=len(lines) + 1,
                offset=len(data) - len(tail),
                dropped=tail,
                at=utc_stamp(),
                cut=not self.read_only,
            )
            if self.read_only:
                self._skip(torn)
            else:
                self._cut(fd, torn)
        return rows

    def _parse(self, raw: bytes, number: int) -> LedgerEntry:
        if not raw.strip():
            raise CorruptLedgerError(self.path, number, "an empty line")
        try:
            return LedgerEntry.model_validate_json(raw, strict=True)
        except ValidationError as err:
            version = _schema_version(raw)
            if version is not None and version != SCHEMA_VERSION:
                raise SchemaVersionError(self.path, number, version) from err
            raise CorruptLedgerError(self.path, number, _first_error(err)) from err

    def _cut(self, fd: int, torn: TornTail) -> None:
        self._save(torn)  # the bytes are on disk beside the ledger before the ledger lets go of them
        os.ftruncate(fd, torn.offset)
        _sync(fd)
        self._torn.append(torn)
        logger.warning(
            "ledger %s: cut a torn last line (line %d, %d bytes at offset %d) that a crashed append left; "
            "its bytes are saved in %s",
            torn.path,
            torn.line,
            len(torn.dropped),
            torn.offset,
            _display(self.torn_path),
        )

    def _skip(self, torn: TornTail) -> None:
        if any((seen.offset, seen.dropped) == (torn.offset, torn.dropped) for seen in self._torn):
            return  # every read rescans, and a read-only ledger keeps the line, so it is reported once
        self._torn.append(torn)
        logger.warning(
            "ledger %s is read-only: skipped a torn last line (line %d, %d bytes at offset %d) that a crashed append "
            "left; it stays in the file",
            torn.path,
            torn.line,
            len(torn.dropped),
            torn.offset,
        )

    def _save(self, torn: TornTail) -> None:
        """Append the torn line's report, bytes and all, to the torn-tail file and sync it. Any failure raises
        TornTailSaveError before the ledger is cut: a torn-tail path that is a directory, a dangling link or on a
        read-only mount is a save failure, never a missing ledger."""
        try:
            self._write_report(torn)
        except OSError as err:
            raise TornTailSaveError(
                err.errno,
                f"{err.strerror or err}; so the torn last line (line {torn.line}) was not cut",
                _display(self.torn_path),
            ) from err

    def _write_report(self, torn: TornTail) -> None:
        record = torn.model_dump_json().encode("utf-8") + b"\n"
        try:
            fd = os.open(self.torn_path, os.O_RDWR | os.O_APPEND | os.O_CREAT | os.O_EXCL | os.O_CLOEXEC, _FILE_MODE)
            created = True
        except FileExistsError:
            fd = os.open(self.torn_path, os.O_RDWR | os.O_APPEND | os.O_CLOEXEC)
            created = False
        try:
            size = os.fstat(fd).st_size
            if size and os.pread(fd, 1, size - 1) != b"\n":
                _write_all(fd, b"\n")  # an earlier save crashed part way; its piece keeps a line of its own
            _write_all(fd, record)
            _sync(fd)
        finally:
            os.close(fd)
        if created:
            _sync_directory(self.torn_path.parent)


# ---------------------------------------------------------------- entries


def utc_stamp(moment: datetime | None = None) -> str:
    """A UTC time in the form `LedgerEntry.at` takes. A naive datetime is refused: its zone is unknown."""
    if moment is None:
        moment = datetime.now(UTC)
    elif moment.tzinfo is None:
        raise ValueError("a naive datetime has no time zone")
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def new_entry(finding: Finding, *, at: datetime | None = None) -> LedgerEntry:
    return LedgerEntry(at=utc_stamp(at), receipts_version=__version__, finding=finding)


def validated(entry: object) -> LedgerEntry:
    """The entry exactly as a reader of its ledger line would get it: written to JSON, validated back strictly, and
    refused unless it comes back equal. Anything that hands ledger fields onward (the ticket writer) takes them from
    here, so an entry built without validation never gets further than a written one would."""
    return _round_trip(entry)[1]


def _encode(entry: LedgerEntry) -> bytes:
    """The entry as one JSON line, validated the way a reader will validate it: an entry built without validation
    (model_construct, model_copy) cannot slip in a record that a reader would refuse or read as something else."""
    return _round_trip(entry)[0] + b"\n"  # compact JSON escapes every newline inside a string, so this is the only one


def _round_trip(entry: object) -> tuple[bytes, LedgerEntry]:
    if not isinstance(entry, LedgerEntry):
        raise InvalidEntryError(f"a ledger line holds a LedgerEntry, not {type(entry).__name__}")
    try:
        data = entry.model_dump_json(warnings="error").encode("utf-8")
    except ValueError as err:  # pydantic's serialization error: a field of the wrong type, or text that is not Unicode
        raise InvalidEntryError("the entry cannot be written as JSON") from err
    try:
        back = LedgerEntry.model_validate_json(data, strict=True)
    except ValidationError as err:
        raise InvalidEntryError(f"the entry fails validation: {_first_error(err)}") from err
    if back != entry:
        # An infinite or NaN float is written as null; an optional field then reads back as None, silently.
        where = _first_difference(entry.model_dump(), back.model_dump()) or "the entry"
        raise InvalidEntryError(f"the entry would read back as something else: {where} does not survive JSON")
    return data, back


def _first_difference(written: object, read: object, where: str = "") -> str | None:
    """The dotted path of the first value that differs between two model dumps."""
    if isinstance(written, dict) and isinstance(read, dict):
        for key in [*written, *(k for k in read if k not in written)]:
            found = _first_difference(written.get(key), read.get(key), f"{where}.{key}" if where else str(key))
            if found:
                return found
        return None
    if isinstance(written, list | tuple) and isinstance(read, list | tuple) and len(written) == len(read):
        for i, (a, b) in enumerate(zip(written, read, strict=True)):
            found = _first_difference(a, b, f"{where}.{i}" if where else str(i))
            if found:
                return found
        return None
    if written == read:  # NaN never equals itself, so a NaN shows up as a difference here
        return None
    return where or "the entry"


def _schema_version(raw: bytes) -> int | None:
    """The `v` of a line that failed validation, when it is a JSON object with an integer `v`; None otherwise. Only a
    refused line gets here, so a valid ledger is parsed once."""
    try:
        data = json.loads(raw)
    except (ValueError, RecursionError):  # not JSON, not UTF-8, or nested too deep to look at
        return None
    if not isinstance(data, dict):
        return None
    version = data.get("v")
    return version if type(version) is int else None  # not a bool, which is an int subclass


def _first_error(err: ValidationError) -> str:
    """The first validation error as `location: message`, without the input, which can be long or untrusted."""
    errors = err.errors(include_url=False, include_input=False)
    first = errors[0]
    location = ".".join(str(part) for part in first["loc"])
    more = f" (and {len(errors) - 1} more)" if len(errors) > 1 else ""
    return f"{location}: {first['msg']}{more}" if location else f"{first['msg']}{more}"


def _display(path: Path) -> str:
    """A path as text that is always valid UTF-8; a path from the command line can hold bytes that are not."""
    return os.fsencode(path).decode("utf-8", "backslashreplace")


# ---------------------------------------------------------------- rollups


@dataclass
class _StageSum:
    records: int = 0
    input_tokens: int = 0
    cached_tokens: int = 0
    output_tokens: int = 0
    reasoning_tokens: int = 0
    micro_usd: int = 0
    seconds: list[float] = field(default_factory=list)


@dataclass
class _TargetSum:
    entries: int = 0
    micro_usd: int = 0
    seconds: list[float] = field(default_factory=list)
    wall_seconds: float | None = None
    critic: dict[str, int] = field(default_factory=dict)  # finding id: the most rounds any of its entries records


def rollup(entries: Iterable[LedgerEntry]) -> Rollup:
    """Spend per stage and model, totals per target, p50 and p95 of per-target cost and latency, and critic rounds.

    Every entry's costs count, since an entry's costs are what producing it spent. Money stays in integer
    micro-dollars; seconds are added with math.fsum, so the order of entries does not change a total.
    """
    stages: dict[tuple[str, str | None], _StageSum] = {}
    targets: dict[str, _TargetSum] = {}
    latest: dict[str, Status] = {}  # finding id: its latest entry's status, the entries being in ledger order
    count = 0
    for count, entry in enumerate(entries, start=1):  # noqa: B007 (count is read after the loop)
        finding = entry.finding
        latest[finding.id] = finding.status
        target = targets.setdefault(finding.target_id, _TargetSum())
        target.entries += 1
        target.critic[finding.id] = max(target.critic.get(finding.id, 0), finding.critic_rounds)
        wall = finding.wall_seconds
        if wall is not None and (target.wall_seconds is None or wall > target.wall_seconds):
            target.wall_seconds = wall
        for cost in finding.costs:
            stage = stages.setdefault((cost.stage, cost.model), _StageSum())
            stage.records += 1
            stage.input_tokens += cost.input_tokens
            stage.cached_tokens += cost.cached_tokens
            stage.output_tokens += cost.output_tokens
            stage.reasoning_tokens += cost.reasoning_tokens
            stage.micro_usd += cost.micro_usd
            stage.seconds.append(cost.seconds)
            target.micro_usd += cost.micro_usd
            target.seconds.append(cost.seconds)
    stage_rows = tuple(
        StageRollup(
            stage=name,
            model=model,
            records=s.records,
            input_tokens=s.input_tokens,
            cached_tokens=s.cached_tokens,
            output_tokens=s.output_tokens,
            reasoning_tokens=s.reasoning_tokens,
            micro_usd=s.micro_usd,
            seconds=math.fsum(s.seconds),
        )
        for (name, model), s in sorted(stages.items(), key=lambda item: (item[0][0], item[0][1] or ""))
    )
    target_rows = tuple(_target_row(target_id, t) for target_id, t in sorted(targets.items()))
    costs = [t.micro_usd for t in target_rows]
    latencies = [t.latency_seconds for t in target_rows]
    wall_targets = sum(1 for t in target_rows if t.latency_basis is LatencyBasis.WALL)
    percentiles = Percentiles(
        targets=len(target_rows),
        p50_micro_usd=nearest_rank(costs, 50),
        p95_micro_usd=nearest_rank(costs, 95),
        p50_seconds=nearest_rank(latencies, 50),
        p95_seconds=nearest_rank(latencies, 95),
        latency_basis=_latency_basis(wall_targets, len(target_rows)),
        wall_targets=wall_targets,
    )
    rounds = [n for t in targets.values() for n in t.critic.values()]
    by_rounds = [0] * (max([CRITIC_ROUNDS_CAP, *rounds]) + 1)
    for n in rounds:
        by_rounds[n] += 1
    critic = CriticRollup(findings=len(rounds), rounds=sum(rounds), findings_by_rounds=tuple(by_rounds))
    statuses = dict.fromkeys(Status, 0)
    for status in latest.values():
        statuses[status] += 1
    totals = Totals(
        entries=count,
        findings=len(latest),
        targets=len(target_rows),
        micro_usd=sum(s.micro_usd for s in stage_rows),
        input_tokens=sum(s.input_tokens for s in stage_rows),
        cached_tokens=sum(s.cached_tokens for s in stage_rows),
        output_tokens=sum(s.output_tokens for s in stage_rows),
        reasoning_tokens=sum(s.reasoning_tokens for s in stage_rows),
        seconds=math.fsum(t.stage_seconds for t in target_rows),
        statuses=statuses,
    )
    return Rollup(stages=stage_rows, targets=target_rows, percentiles=percentiles, critic=critic, totals=totals)


def _target_row(target_id: str, t: _TargetSum) -> TargetRollup:
    stage_seconds = math.fsum(t.seconds)
    wall = t.wall_seconds
    return TargetRollup(
        target_id=target_id,
        entries=t.entries,
        findings=len(t.critic),
        micro_usd=t.micro_usd,
        stage_seconds=stage_seconds,
        wall_seconds=wall,
        latency_seconds=stage_seconds if wall is None else wall,
        latency_basis=LatencyBasis.SUMMED_STAGE_SECONDS if wall is None else LatencyBasis.WALL,
        critic_rounds=sum(t.critic.values()),
    )


def _latency_basis(wall_targets: int, targets: int) -> LatencyBasis | None:
    if targets == 0:
        return None
    if wall_targets == targets:
        return LatencyBasis.WALL
    return LatencyBasis.SUMMED_STAGE_SECONDS if wall_targets == 0 else LatencyBasis.MIXED


def usd(micro_usd: int) -> str:
    """Integer micro-dollars as dollars, exactly: 1234567 is $1.234567."""
    whole, part = divmod(micro_usd, 1_000_000)
    return f"${whole:,}.{part:06d}"


def nearest_rank[N: (int, float)](values: Sequence[N], pct: int) -> N | None:
    """The nearest-rank percentile: the smallest value with at least `pct` percent of the values at or below it.

    It is always one of the values, so integer money stays integer. None for no values.
    """
    if not 0 < pct <= 100:
        raise ValueError(f"a percentile is in (0, 100], not {pct}")
    if not values:
        return None
    ordered = sorted(values)
    rank = -(-pct * len(ordered) // 100)  # ceil(pct * n / 100) in integers
    return ordered[rank - 1]


# ---------------------------------------------------------------- file plumbing


def _read_all(fd: int) -> bytes:
    size = os.fstat(fd).st_size
    chunks: list[bytes] = []
    offset = 0
    while offset < size:
        chunk = os.pread(fd, min(_READ_CHUNK, size - offset), offset)
        if not chunk:
            break
        chunks.append(chunk)
        offset += len(chunk)
    return b"".join(chunks)


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        written = os.write(fd, view)
        view = view[written:]


def _sync(fd: int) -> None:
    """Flush the file to stable storage. On macOS fsync(2) can leave data in the drive's cache, so this asks for
    F_FULLFSYNC there, and falls back to fsync on a file system that does not support it."""
    if sys.platform == "darwin":
        try:
            fcntl.fcntl(fd, fcntl.F_FULLFSYNC)
        except OSError:
            os.fsync(fd)
        return
    os.fsync(fd)


def _make_directories(directory: Path) -> list[Path]:
    """Make `directory` and any missing parents, and return the ones this call made, outermost first, so each one's
    entry in its parent can be synced. A directory another process makes first is not this call's."""
    missing: list[Path] = []
    while not directory.exists():
        missing.append(directory)
        if directory.parent == directory:
            break
        directory = directory.parent
    made: list[Path] = []
    for path in reversed(missing):
        try:
            path.mkdir()
        except FileExistsError:
            continue
        made.append(path)
    return made


def _sync_directory(directory: Path) -> None:
    """Make a new file's directory entry durable."""
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


if TYPE_CHECKING:

    def _is_a_ledger(ledger: JsonlLedger) -> Ledger:  # mypy checks that JsonlLedger meets the Ledger interface
        return ledger
