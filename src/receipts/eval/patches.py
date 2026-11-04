"""Pull each selected entry's per-instance patch and evaluation report into eval/data/patches/<entry>/ and index them
by sha256.

Every selected entry keeps <logs>/<instance_id>/patch.diff (the patch the SWE-bench harness graded) and report.json
(its verdict, with per-test FAIL_TO_PASS and PASS_TO_PASS results) in the public bucket s3://swe-bench-submissions.
They are read with anonymous HTTPS GETs, which the founder approved on 2026-10-27; `receipts.eval.sources` refuses
every other AWS endpoint and any request to the bucket that is not a plain GET without credentials. A report that
the bucket does not have (HTTP 404) is recorded as missing and its patch stays unlabeled; a missing patch stops the
pull, because a record without its patch cannot be verified.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from receipts.eval.entries import Entry, PatchStore
from receipts.eval.hashing import pretty_json, sha256_bytes, write_bytes_atomic, write_text_atomic
from receipts.eval.manifest import PatchIndex, PatchIndexEntry, patch_index_path
from receipts.eval.sources import check_url

S3_BUCKET = "swe-bench-submissions"
FILES = ("patch.diff", "report.json")


class PullError(RuntimeError):
    """A file the pull needs and could not get."""


def https_url(location: str, key: str) -> str:
    """s3://bucket/prefix + key -> the bucket's public virtual-hosted https address; https locations pass through."""
    if location.startswith("s3://"):
        bucket, _, prefix = location.removeprefix("s3://").partition("/")
        return f"https://{bucket}.s3.amazonaws.com/{prefix.rstrip('/')}/{key}"
    if location.startswith("https://"):
        return f"{location.rstrip('/')}/{key}"
    raise ValueError(f"unsupported patch location {location!r}")


def logs_url(entry: Entry, instance_id: str, file: str) -> str:
    if entry.patch_store is not PatchStore.S3_LOGS:
        raise ValueError(f"{entry.key}: {entry.patch_store} has no per-instance logs folder")
    if file not in FILES:
        raise ValueError(f"{file}: only {FILES} are pulled")
    return https_url(entry.patch_location, f"{instance_id}/{file}")


def _get(client: httpx.Client, url: str, *, retries: int, pause_s: float) -> bytes | None:
    """The body, or None on 404. Retries transport errors and 5xx; anything else stops the pull."""
    last = ""
    for attempt in range(retries):
        try:
            r = client.get(url)
        except httpx.TransportError as exc:
            last = f"{type(exc).__name__}: {exc}"
        else:
            if r.status_code == 200:
                return r.content
            if r.status_code == 404:
                return None
            if r.status_code < 500:
                raise PullError(f"{url}: HTTP {r.status_code}")
            last = f"HTTP {r.status_code}"
        time.sleep(pause_s * 2**attempt)
    raise PullError(f"{url}: failed after {retries} attempts ({last})")


def pull_entry(
    entry: Entry,
    instance_ids: Iterable[str],
    data_dir: Path,
    client: httpx.Client,
    *,
    workers: int = 8,
    retries: int = 4,
    pause_s: float = 1.0,
) -> PatchIndex:
    """Fetch every patch and report not already on disk, then rewrite the entry's index from the bytes on disk."""
    ids = sorted(set(instance_ids))
    for iid in ids:
        for f in FILES:
            check_url(logs_url(entry, iid, f))  # refuse before anything is sent
    out_dir = data_dir / "patches" / entry.key
    missing_marker = ".missing"

    def fetch(iid: str) -> None:
        for f in FILES:
            path = out_dir / iid / f
            if path.is_file() or (path.parent / (f + missing_marker)).is_file():
                continue
            body = _get(client, logs_url(entry, iid, f), retries=retries, pause_s=pause_s)
            if body is None:
                if f == "patch.diff":
                    raise PullError(f"{entry.key} {iid}: the bucket has no patch.diff")
                write_text_atomic(path.parent / (f + missing_marker), "HTTP 404\n")
                continue
            write_bytes_atomic(path, body)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(fetch, ids))

    entries = []
    for iid in ids:
        patch = out_dir / iid / "patch.diff"
        report = out_dir / iid / "report.json"
        data = patch.read_bytes()
        rep = report.read_bytes() if report.is_file() else None
        entries.append(
            PatchIndexEntry(
                instance_id=iid,
                sha256=sha256_bytes(data),
                size=len(data),
                path=str(patch.relative_to(data_dir)),
                report_sha256=None if rep is None else sha256_bytes(rep),
                report_size=None if rep is None else len(rep),
                report_path=None if rep is None else str(report.relative_to(data_dir)),
            )
        )
    index = PatchIndex(entry=entry.key, store=entry.patch_store.value, patches=tuple(entries))
    write_text_atomic(patch_index_path(data_dir, entry.key), pretty_json(index.model_dump(mode="json")))
    return index
