"""Tier A targets (contracts.Target) for sampled items: the agent's patch, the issue text, the base commit, the
digest-pinned Epoch image from eval/images.json (the one R24 ran gold in), and the run phase's test runner and command:
`test_runner` is django for django/django, sympy for sympy/sympy and pytest for everything else, and `test_command` is
SWE-bench's own command for the instance's version (eval/test-specs.json, from the pinned swebench 4.1.0), after the
environment the harness's eval script exports first (Django 1.11 to 3.2 set LANG and LC_ALL; see
`specs.run_phase_test_command`), the same command R24 checked gold with.

A target never carries the gold patch or a test patch: gold is used only at scoring (spec section 3) and the SWE-ABS
tests only for labels. Its id is "<entry>~<instance_id>", with no slash, so a finding id "<target_id>/<claim>/..."
still splits at its first slash.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from pathlib import Path

from receipts.contracts import Target, Tier
from receipts.eval.hashing import sha256_text, write_text_atomic
from receipts.eval.images import ImageSet
from receipts.eval.manifest import Manifest, PatchIndexEntry, load_patch_index, patch_text
from receipts.eval.rows import Row
from receipts.eval.sampler import Item
from receipts.eval.specs import SpecsError, SpecTable, run_phase_test_command

TARGET_SEP = "~"


class TargetError(ValueError):
    """A sampled item that cannot become a target as pinned."""


def target_id(entry: str, instance_id: str) -> str:
    for part in (entry, instance_id):
        if not part or "/" in part or TARGET_SEP in part:
            raise TargetError(f"{part!r} cannot be part of a target id")
    return f"{entry}{TARGET_SEP}{instance_id}"


def build_target(item: Item, row: Row, patch: str, image_ref: str, specs: SpecTable) -> Target:
    if row.instance_id != item.instance_id:
        raise TargetError(f"row {row.instance_id} paired with item {item.instance_id}")
    if not patch.strip():
        raise TargetError(f"{item.entry} {item.instance_id}: an empty patch is never a target")
    try:
        spec = specs.get(row.repo, row.version)
    except SpecsError as e:
        raise TargetError(f"{item.instance_id}: {e}") from e
    return Target(
        id=target_id(item.entry, item.instance_id),
        tier=Tier.A,
        repo=row.repo,
        base_sha=row.base_commit,
        patch_sha=sha256_text(patch),
        patch=patch,
        issue=row.problem_statement,
        image=image_ref,
        test_runner=spec.runner,
        test_command=run_phase_test_command(spec),
    )


def build_targets(
    items: Iterable[Item],
    manifest: Manifest,
    rows: Mapping[str, Row],
    images: ImageSet,
    data_dir: Path,
    specs: SpecTable,
) -> list[Target]:
    """One target per item, each checked against the manifest's pins: the patch bytes hash to the record's
    patch_sha256 and the image is the digest-pinned one for the instance."""
    records = {(r.entry, r.instance_id): r for r in manifest.records}
    pins = images.by_instance
    indexes: dict[str, dict[str, PatchIndexEntry]] = {}
    out = []
    for item in items:
        record = records.get((item.entry, item.instance_id))
        if record is None:
            raise TargetError(f"{item.entry} {item.instance_id} is not in the manifest")
        if record.patch_sha256 is None:
            raise TargetError(f"{item.entry} {item.instance_id}: the manifest has no pulled patch")
        index = indexes.setdefault(item.entry, load_patch_index(data_dir, item.entry))
        entry = index.get(item.instance_id)
        if entry is None or entry.sha256 != record.patch_sha256:
            raise TargetError(f"{item.entry} {item.instance_id}: the patch on disk is not the one the manifest pinned")
        pin = pins.get(item.instance_id)
        if pin is None:
            raise TargetError(f"{item.instance_id}: no digest-pinned image in eval/images.json")
        out.append(build_target(item, rows[item.instance_id], patch_text(data_dir, entry), pin.ref, specs))
    ids = [t.id for t in out]
    if len(ids) != len(set(ids)):
        raise TargetError("two items map to one target id")
    return out


def targets_text(targets: Iterable[Target]) -> str:
    return "".join(t.model_dump_json() + "\n" for t in targets)


def write_targets(path: Path, targets: Iterable[Target]) -> str:
    """Write one target per line; returns the file's sha256."""
    text = targets_text(targets)
    write_text_atomic(path, text)
    return sha256_text(text)
