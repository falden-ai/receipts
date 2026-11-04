"""Pinned Tier B demo inputs and a separate replay of a blocker's repro on the later fix.

Fix metadata never enters the Target or the image. The fix replay supplements the blocker's base/patch evidence.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from receipts.contracts import (
    IMAGE_REF,
    SHA,
    Finding,
    Frozen,
    Outcome,
    RunResult,
    Sandbox,
    Side,
    Status,
    SuiteRunner,
    Target,
    Tier,
)
from receipts.eval.hashing import SHA256, sha256_bytes, sha256_text, write_text_atomic
from receipts.eval.score import repro_tests
from receipts.eval.specs import changed_files

FIX_TIMEOUT_S = 900.0


class Pin(BaseModel):
    """The value and provenance of an input; additional preparation evidence stays in pins.json."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    value: str
    source: str = Field(min_length=1)
    date_read: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")


class CommitPin(Pin):
    value: str = Field(pattern=SHA)


class DiffPin(Pin):
    value: str = Field(pattern=SHA256)


class ImagePin(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    ref: str | None = Field(pattern=IMAGE_REF)


class DemoPins(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    repo: Pin
    base_sha: CommitPin
    merge_sha: CommitPin
    patch_diff_sha256: DiffPin
    fix_diff_sha256: DiffPin
    image: ImagePin


def _diff(pins_dir: Path, name: str, digest: str) -> str:
    data = (pins_dir / name).read_bytes()
    if sha256_bytes(data) != digest:
        raise ValueError(f"{name}: sha256 does not match pins.json")
    text = data.decode("utf-8")
    if not text.strip():
        raise ValueError(f"{name}: an empty diff is refused")
    return text


def load_pins(pins_dir: Path) -> DemoPins:
    """Validate the pins and both diffs, including the fix that is deliberately absent from the Target."""
    pins = DemoPins.model_validate_json((pins_dir / "pins.json").read_bytes())
    _diff(pins_dir, "patch.diff", pins.patch_diff_sha256.value)
    _diff(pins_dir, "fix.diff", pins.fix_diff_sha256.value)
    return pins


def demo_target(pins_dir: Path, image_ref: str) -> Target:
    pins = load_pins(pins_dir)
    template = Target.model_validate_json((pins_dir / "target.template.json").read_bytes())
    patch = _diff(pins_dir, "patch.diff", pins.patch_diff_sha256.value)
    if (
        template.tier is not Tier.B
        or template.repo != pins.repo.value
        or template.base_sha != pins.base_sha.value
        or template.patch_sha != pins.merge_sha.value
        or template.patch != patch
        or template.image is not None
        or template.issue
        or template.commit_messages
        or template.session_summary
    ):
        raise ValueError("target.template.json disagrees with the pinned Tier B claim-only target")
    if pins.image.ref is not None and pins.image.ref != image_ref:
        raise ValueError("--image disagrees with the image pinned on the eval VM")
    return Target.model_validate({**template.model_dump(), "image": image_ref})


def write_run_file(path: Path, target: Target) -> None:
    """One core.pipeline.RunItem line, using the spec temperature schedule."""
    if target.image is None:
        raise ValueError("a demo run needs a digest-pinned image")
    write_text_atomic(path, json.dumps({"target": target.model_dump(mode="json"), "temperatures": "spec"}) + "\n")


class FixCheck(Frozen):
    finding_id: str = Field(min_length=1)
    fix_diff_sha256: str = Field(pattern=SHA256)
    image_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    runs: tuple[RunResult, ...] = Field(min_length=5, max_length=5)
    passes_on_fix: bool

    @model_validator(mode="after")
    def _passes(self) -> Self:
        if any(r.side is not Side.GOLD for r in self.runs):
            raise ValueError("a fix check records only GOLD runs")
        if self.passes_on_fix != all(r.outcome is Outcome.PASS for r in self.runs):
            raise ValueError("passes_on_fix must agree with all five run outcomes")
        return self


def fix_check(
    target: Target,
    finding: Finding,
    fix_diff: str,
    *,
    sandbox: Sandbox,
    runner: SuiteRunner,
    runs: int = 5,
) -> FixCheck:
    """Run five fresh GOLD replays with the run phase's 900 second test timeout; always close the workspace."""
    if finding.status is not Status.BLOCKER or finding.repro is None:
        raise ValueError("fixcheck requires a blocker finding with a runnable repro")
    if finding.target_id != target.id:
        raise ValueError("the finding belongs to a different target")
    if runs != 5:
        raise ValueError("fixcheck requires exactly five runs")
    if not fix_diff.strip():
        raise ValueError("fixcheck requires a nonempty fix diff")
    if target.image is None:
        raise ValueError("fixcheck requires a digest-pinned target image")
    if runner.kind is not target.test_runner:
        raise ValueError("the runner does not match the target")
    replay = finding.replay
    if replay is None or (replay.base_sha, replay.patch_sha, replay.image_digest) != (
        target.base_sha,
        target.patch_sha,
        target.image.split("@", 1)[1],
    ):
        raise ValueError("the target does not match the blocker's replay pins")
    repro = finding.repro
    workspace = sandbox.prepare(target, Side.GOLD, patch=fix_diff)
    try:
        results = tuple(
            sandbox.run_test(
                workspace,
                repro_tests(repro),
                runner=runner,
                changed_files=changed_files(fix_diff),
                timeout_s=FIX_TIMEOUT_S,
                files=repro.files,
                expect=repro.cases or None,
                fresh=True,
                seed=i,
            )
            for i in range(runs)
        )
    finally:
        sandbox.close(workspace)
    return FixCheck(
        finding_id=finding.id,
        fix_diff_sha256=sha256_text(fix_diff),
        image_digest=workspace.image_digest,
        runs=results,
        passes_on_fix=all(r.outcome is Outcome.PASS for r in results),
    )
