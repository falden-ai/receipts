"""SWE-bench Verified rows (base commit, issue text, gold and test patches, FAIL_TO_PASS and PASS_TO_PASS) and the
SWE-ABS rows (each instance's strengthened test patch), from the pinned parquets, converted once into canonical JSONL
so nothing else in the lane needs a parquet reader.

`convert` needs pyarrow, the project's `eval` extra, so it runs as `uv run --frozen --extra eval python -m
receipts.eval rows`. Its output is a pure function of the parquet's bytes: one row per line, sorted by instance id,
keys sorted, FAIL_TO_PASS and PASS_TO_PASS decoded from the JSON strings the parquet stores them as. The manifest
records each JSONL's sha256, so a frozen manifest pins the exact rows every label run and every target was built from.

The SWE-ABS dataset (OpenAgentLab/SWE-Bench_Verified_ABS) keeps SWE-bench's columns: `test_patch` is the SWE-ABS
test patch and `original_test_patch` SWE-bench's. Its FAIL_TO_PASS and PASS_TO_PASS lists are SWE-bench's own, which
the SWE-ABS tests do not run, so SWE-ABS grades a run by every test the SWE-ABS test files report (`labelrun`).
"""

from __future__ import annotations

import importlib
import json
from pathlib import Path
from typing import Any

from pydantic import Field

from receipts.contracts import Frozen
from receipts.eval.hashing import canonical_json, sha256_file, write_text_atomic
from receipts.eval.pool import INSTANCE_ID

ROWS_PATH = "hf/SWE-bench_Verified/rows.jsonl"  # under eval/data/, next to the parquet it comes from
ABS_ROWS_PATH = "hf/SWE-Bench_Verified_ABS/rows.jsonl"
COLUMNS = (
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "base_commit",
    "created_at",
    "difficulty",
    "environment_setup_commit",
    "hints_text",
    "instance_id",
    "patch",
    "problem_statement",
    "repo",
    "test_patch",
    "version",
)


class Row(Frozen):
    """One SWE-bench Verified instance, as the harness's make_test_spec reads it."""

    FAIL_TO_PASS: tuple[str, ...] = Field(min_length=1)
    PASS_TO_PASS: tuple[str, ...]
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    created_at: str
    difficulty: str
    environment_setup_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    hints_text: str
    instance_id: str = Field(pattern=INSTANCE_ID)
    patch: str = Field(min_length=1)  # gold: used only at Tier A scoring and by the runsc check, never by an attacker
    problem_statement: str = Field(min_length=1)
    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    test_patch: str = Field(min_length=1)
    version: str = Field(min_length=1)

    def harness_dict(self) -> dict[str, Any]:
        """The row as swebench's make_test_spec takes it (it accepts lists for the two test fields)."""
        return self.model_dump(mode="json")


class RowsError(ValueError):
    """Rows that are not the pinned dataset's."""


def _decode(raw: dict[str, Any]) -> Row:
    data = {k: raw[k] for k in COLUMNS}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = data[key]
        data[key] = json.loads(value) if isinstance(value, str) else value
    return Row.model_validate(data)


def rows_text(raw_rows: list[dict[str, Any]]) -> str:
    rows = sorted((_decode(r) for r in raw_rows), key=lambda r: r.instance_id)
    ids = [r.instance_id for r in rows]
    if len(ids) != len(set(ids)):
        raise RowsError("duplicate instance ids in the dataset")
    return "".join(canonical_json(r.model_dump(mode="json")) + "\n" for r in rows)


def convert(parquet: Path, out: Path) -> int:
    """Write the canonical JSONL for the pinned parquet; returns the row count. Needs pyarrow (see the module doc)."""
    pq: Any = importlib.import_module("pyarrow.parquet")
    raw_rows: list[dict[str, Any]] = pq.read_table(parquet).to_pylist()
    write_text_atomic(out, rows_text(raw_rows))
    return len(raw_rows)


def load(path: Path, *, expect_sha256: str) -> dict[str, Row]:
    """The rows by instance id, refused unless the file is the one the manifest pinned."""
    if not path.is_file():
        raise RowsError(f"{path} is missing; run `uv run --frozen --extra eval python -m receipts.eval rows`")
    got = sha256_file(path)
    if got != expect_sha256:
        raise RowsError(f"{path}: sha256 {got}, the manifest pinned {expect_sha256}")
    out: dict[str, Row] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = Row.model_validate_json(line)
        if row.instance_id in out:
            raise RowsError(f"{path}:{n}: {row.instance_id} appears twice")
        out[row.instance_id] = row
    return out


def load_unpinned(path: Path) -> tuple[dict[str, Row], str]:
    """The rows and their sha256, for the one command (manifest) that records the pin."""
    digest = sha256_file(path)
    return load(path, expect_sha256=digest), digest


ABS_COLUMNS: tuple[str, ...] = (
    "FAIL_TO_PASS",
    "PASS_TO_PASS",
    "base_commit",
    "instance_id",
    "original_test_patch",
    "patch",
    "repo",
    "test_patch",
    "version",
)


class AbsRow(Frozen):
    """One SWE-ABS instance: the strengthened test patch, and the fields that tie it to its SWE-bench row."""

    FAIL_TO_PASS: tuple[str, ...]
    PASS_TO_PASS: tuple[str, ...]
    base_commit: str = Field(pattern=r"^[0-9a-f]{40}$")
    instance_id: str = Field(pattern=INSTANCE_ID)
    original_test_patch: str = Field(min_length=1)
    patch: str = Field(min_length=1)
    repo: str = Field(pattern=r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
    test_patch: str = Field(min_length=1)  # the SWE-ABS test patch
    version: str = Field(min_length=1)

    def harness_row(self, row: Row) -> dict[str, Any]:
        """The SWE-bench row with its test patch replaced by this one: what the harness builds the SWE-ABS eval
        script from (SWE-ABS's own evaluator does the same, swe-bench/swebench/harness/test_spec/test_spec.py)."""
        check_abs_row(self, row)
        return row.harness_dict() | {"test_patch": self.test_patch}


def check_abs_row(abs_row: AbsRow, row: Row) -> None:
    """The SWE-ABS row is the same instance as the SWE-bench row: same repository, commit, version and gold patch,
    and its original test patch is SWE-bench's."""
    if abs_row.instance_id != row.instance_id:
        raise RowsError(f"SWE-ABS row {abs_row.instance_id} paired with {row.instance_id}")
    mismatched = [
        name
        for name, a, b in (
            ("repo", abs_row.repo, row.repo),
            ("base_commit", abs_row.base_commit, row.base_commit),
            ("version", abs_row.version, row.version),
            ("patch", abs_row.patch, row.patch),
            ("original_test_patch", abs_row.original_test_patch, row.test_patch),
        )
        if a != b
    ]
    if mismatched:
        raise RowsError(f"{row.instance_id}: the SWE-ABS row disagrees with SWE-bench on {mismatched}")


def _decode_abs(raw: dict[str, Any]) -> AbsRow:
    data = {k: raw[k] for k in ABS_COLUMNS}
    for key in ("FAIL_TO_PASS", "PASS_TO_PASS"):
        value = data[key]
        data[key] = json.loads(value) if isinstance(value, str) else value
    return AbsRow.model_validate(data)


def abs_rows_text(raw_rows: list[dict[str, Any]]) -> str:
    rows = sorted((_decode_abs(r) for r in raw_rows), key=lambda r: r.instance_id)
    ids = [r.instance_id for r in rows]
    if len(ids) != len(set(ids)):
        raise RowsError("duplicate instance ids in the SWE-ABS dataset")
    return "".join(canonical_json(r.model_dump(mode="json")) + "\n" for r in rows)


def convert_abs(parquet: Path, out: Path) -> int:
    """Write the canonical SWE-ABS JSONL for the pinned parquet; returns the row count. Needs pyarrow."""
    pq: Any = importlib.import_module("pyarrow.parquet")
    raw_rows: list[dict[str, Any]] = pq.read_table(parquet).to_pylist()
    write_text_atomic(out, abs_rows_text(raw_rows))
    return len(raw_rows)


def load_abs(path: Path, *, expect_sha256: str) -> dict[str, AbsRow]:
    """The SWE-ABS rows by instance id, refused unless the file is the one the manifest pinned."""
    if not path.is_file():
        raise RowsError(f"{path} is missing; run `uv run --frozen --extra eval python -m receipts.eval rows`")
    got = sha256_file(path)
    if got != expect_sha256:
        raise RowsError(f"{path}: sha256 {got}, the manifest pinned {expect_sha256}")
    out: dict[str, AbsRow] = {}
    for n, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = AbsRow.model_validate_json(line)
        if row.instance_id in out:
            raise RowsError(f"{path}:{n}: {row.instance_id} appears twice")
        out[row.instance_id] = row
    return out
