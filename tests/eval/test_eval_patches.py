"""The patch pull reads each entry's patch.diff and report.json from the approved bucket, records a missing report as
missing, stops on a missing patch, retries transient errors, and indexes every byte by sha256. MockTransport only."""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

from receipts.eval.entries import SELECTED, PatchStore
from receipts.eval.manifest import load_patch_index, load_reports
from receipts.eval.patches import FILES, PullError, https_url, logs_url, pull_entry
from receipts.eval.sources import FetchRefusedError, make_client

ENTRY = SELECTED[0]
DIFF = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
IDS = ["django__django-1", "sympy__sympy-2"]


def report(iid: str) -> bytes:
    return json.dumps(
        {iid: {"patch_is_None": False, "patch_exists": True, "patch_successfully_applied": True, "resolved": False}}
    ).encode()


def bucket(statuses: dict[str, list[int]] | None = None, seen: list[str] | None = None) -> httpx.MockTransport:
    """Every key answers 200 unless `statuses` lists the codes to answer, in order, for a key ending."""
    queue = {k: list(v) for k, v in (statuses or {}).items()}

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        if seen is not None:
            seen.append(url)
        for suffix, codes in queue.items():
            if url.endswith(suffix) and codes:
                code = codes.pop(0)
                if code != 200:
                    return httpx.Response(code)
        iid = url.rsplit("/", 2)[-2]
        return httpx.Response(200, content=DIFF.encode() if url.endswith("patch.diff") else report(iid))

    return httpx.MockTransport(handler)


def test_every_selected_entry_reads_its_logs_folder_in_the_approved_bucket() -> None:
    for e in SELECTED:
        assert e.patch_store is PatchStore.S3_LOGS
        for f in FILES:
            url = logs_url(e, "django__django-1", f)
            assert url.startswith("https://swe-bench-submissions.s3.amazonaws.com/") and url.endswith(f"/{f}")
    assert https_url("s3://swe-bench-submissions/bash-only/e/logs", "i/patch.diff") == (
        "https://swe-bench-submissions.s3.amazonaws.com/bash-only/e/logs/i/patch.diff"
    )
    with pytest.raises(ValueError):
        https_url("ftp://x/y", "k")
    with pytest.raises(ValueError, match="only"):
        logs_url(ENTRY, "django__django-1", "test_output.txt")


def test_patches_and_reports_land_hashed_and_indexed(tmp_path: Path) -> None:
    with make_client(transport=bucket()) as client:
        index = pull_entry(ENTRY, IDS, tmp_path, client, pause_s=0)
    assert [p.instance_id for p in index.patches] == IDS
    assert all(p.has_report for p in index.patches)
    loaded = load_patch_index(tmp_path, ENTRY.key)
    assert set(loaded) == set(IDS)
    assert (tmp_path / loaded[IDS[0]].path).read_text() == DIFF
    assert set(load_reports(tmp_path, loaded)) == set(IDS)


def test_a_missing_report_is_recorded_as_missing_never_guessed(tmp_path: Path) -> None:
    with make_client(transport=bucket({f"{IDS[1]}/report.json": [404]})) as client:
        index = pull_entry(ENTRY, IDS, tmp_path, client, pause_s=0)
    by_id = {p.instance_id: p for p in index.patches}
    assert by_id[IDS[0]].has_report and not by_id[IDS[1]].has_report
    assert (tmp_path / "patches" / ENTRY.key / IDS[1] / "report.json.missing").is_file()
    seen: list[str] = []
    with make_client(transport=bucket(seen=seen)) as client:  # a rerun does not ask again
        pull_entry(ENTRY, IDS, tmp_path, client, pause_s=0)
    assert seen == []


def test_plant_a_missing_patch_stops_the_pull(tmp_path: Path) -> None:
    with (
        make_client(transport=bucket({f"{IDS[0]}/patch.diff": [404]})) as client,
        pytest.raises(PullError, match=r"no patch\.diff"),
    ):
        pull_entry(ENTRY, IDS, tmp_path, client, pause_s=0)


def test_transient_errors_are_retried_and_a_refusal_is_not(tmp_path: Path) -> None:
    with make_client(transport=bucket({f"{IDS[0]}/patch.diff": [503, 500]})) as client:
        pull_entry(ENTRY, IDS, tmp_path, client, pause_s=0)
    with (
        make_client(transport=bucket({f"{IDS[0]}/report.json": [403]})) as client,
        pytest.raises(PullError, match="HTTP 403"),
    ):
        pull_entry(ENTRY, IDS, tmp_path / "other", client, pause_s=0)
    with (
        make_client(transport=bucket({f"{IDS[0]}/patch.diff": [500, 500, 500, 500]})) as client,
        pytest.raises(PullError, match="after 4 attempts"),
    ):
        pull_entry(ENTRY, IDS, tmp_path / "third", client, pause_s=0)


def test_plant_an_entry_whose_store_is_off_the_approved_bucket_sends_nothing(tmp_path: Path) -> None:
    seen: list[str] = []
    elsewhere = ENTRY.model_copy(update={"patch_location": "s3://some-other-bucket/logs"})
    with make_client(transport=bucket(seen=seen)) as client, pytest.raises(FetchRefusedError):
        pull_entry(elsewhere, IDS, tmp_path, client, pause_s=0)
    assert seen == []


def test_plant_a_report_that_changed_on_disk_is_refused(tmp_path: Path) -> None:
    with make_client(transport=bucket()) as client:
        index = pull_entry(ENTRY, IDS, tmp_path, client, pause_s=0)
    path = tmp_path / str(index.patches[0].report_path)
    path.write_bytes(path.read_bytes().replace(b"false", b"true", 1))
    with pytest.raises(ValueError, match="report bytes do not match"):
        load_patch_index(tmp_path, ENTRY.key)
