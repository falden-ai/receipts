"""The rows JSONL is a pure function of the dataset rows, decodes the test lists, and is refused unless its bytes are
the ones the manifest pinned."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from receipts.eval.hashing import sha256_file
from receipts.eval.rows import RowsError, load, rows_text


def raw_row(iid: str, **over: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "repo": "django/django",
        "instance_id": iid,
        "base_commit": "a" * 40,
        "patch": "diff --git a/x b/x\n",
        "test_patch": "diff --git a/t b/t\n",
        "problem_statement": "It breaks.",
        "hints_text": "",
        "created_at": "2020-01-01T00:00:00Z",
        "version": "3.0",
        "FAIL_TO_PASS": json.dumps(["t::fix"]),  # the parquet stores both lists as JSON strings
        "PASS_TO_PASS": json.dumps(["t::old"]),
        "environment_setup_commit": "b" * 40,
        "difficulty": "<15 min fix",
    }
    row.update(over)
    return row


def test_rows_are_canonical_whatever_the_input_order() -> None:
    a = rows_text([raw_row("django__django-2"), raw_row("django__django-1")])
    b = rows_text([raw_row("django__django-1"), raw_row("django__django-2")])
    assert a == b
    first = json.loads(a.splitlines()[0])
    assert first["instance_id"] == "django__django-1" and first["FAIL_TO_PASS"] == ["t::fix"]


@pytest.mark.parametrize(
    "bad",
    [
        {"FAIL_TO_PASS": json.dumps([])},  # an instance with no failing test to fix
        {"base_commit": "main"},
        {"instance_id": "../etc"},
        {"problem_statement": ""},
    ],
)
def test_plant_a_malformed_row_is_refused(bad: dict[str, Any]) -> None:
    with pytest.raises(ValidationError):
        rows_text([raw_row("django__django-1", **bad)])


def test_plant_duplicate_rows_are_refused() -> None:
    with pytest.raises(RowsError, match="duplicate"):
        rows_text([raw_row("django__django-1"), raw_row("django__django-1")])


def test_plant_rows_that_are_not_the_pinned_bytes_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text(rows_text([raw_row("django__django-1")]))
    pinned = sha256_file(path)
    assert set(load(path, expect_sha256=pinned)) == {"django__django-1"}
    path.write_text(path.read_text().replace("It breaks.", "It is fine."))
    with pytest.raises(RowsError, match="the manifest pinned"):
        load(path, expect_sha256=pinned)
    with pytest.raises(RowsError, match="missing"):
        load(tmp_path / "absent.jsonl", expect_sha256=pinned)
