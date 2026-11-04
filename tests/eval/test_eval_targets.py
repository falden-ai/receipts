"""Sampled items become contracts.Target objects with the agent's patch, the issue, the base commit, the digest-pinned
image and the run phase's runner and SWE-bench test command; never the gold patch, never a tag, never a patch the
manifest did not pin."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from receipts.contracts import RunnerKind, Target, Tier
from receipts.eval.hashing import pretty_json, sha256_bytes
from receipts.eval.images import ImageRef, ImageSet
from receipts.eval.labels import Label
from receipts.eval.manifest import Manifest, PatchIndex, PatchIndexEntry, Record, patch_index_path
from receipts.eval.rows import Row
from receipts.eval.sampler import Item, Side
from receipts.eval.specs import RepoSpec, SpecTable, runner_kind
from receipts.eval.targets import TargetError, build_target, build_targets, target_id

IID = "django__django-1"
ENTRY = "20260217_mini-v2.0.0_glm-5-high"
PATCH = "diff --git a/x.py b/x.py\n--- a/x.py\n+++ b/x.py\n@@ -1 +1 @@\n-a\n+b\n"
DIGEST = "sha256:" + "e" * 64
ROW = Row(
    FAIL_TO_PASS=("t::fix",),
    PASS_TO_PASS=(),
    base_commit="a" * 40,
    created_at="2020-01-01T00:00:00Z",
    difficulty="<15 min fix",
    environment_setup_commit="b" * 40,
    hints_text="",
    instance_id=IID,
    patch="diff --git a/gold b/gold\n+the gold fix\n",
    problem_statement="Calling f(1) raises.",
    repo="django/django",
    test_patch="diff --git a/t b/t\n+the hidden test\n",
    version="3.0",
)
ITEM = Item(
    entry=ENTRY,
    model="glm-5",
    instance_id=IID,
    repo="django/django",
    label=Label.BAD,
    side=Side.CALIBRATION,
    core=True,
    rank="0" * 64,
)
IMAGES = ImageSet(
    resolved_at="2026-10-27T00:00:00Z",
    images=(
        ImageRef(
            instance_id=IID,
            image=f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{IID}",
            digest=DIGEST,
            media_type="application/vnd.docker.distribution.manifest.v2+json",
        ),
    ),
    unique_compressed_bytes=0,
)


SPECS = SpecTable(
    swebench_version="4.1.0",
    wheel_sha256="1" * 64,
    constants_sha256="2" * 64,
    specs=(
        RepoSpec(
            repo="django/django",
            version="3.0",
            test_cmd="./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1",
            runner=RunnerKind.DJANGO,
        ),
    ),
)


def setup(tmp_path: Path, *, on_disk: str = PATCH, pinned: str = PATCH) -> Manifest:
    rel = f"patches/{ENTRY}/{IID}/patch.diff"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_text(on_disk)
    body = on_disk.encode()
    index = PatchIndex(
        entry=ENTRY,
        store="s3-logs",
        patches=(PatchIndexEntry(instance_id=IID, sha256=sha256_bytes(body), size=len(body), path=rel),),
    )
    patch_index_path(tmp_path, ENTRY).write_text(pretty_json(index.model_dump(mode="json")))
    record = Record(
        entry=ENTRY,
        instance_id=IID,
        repo="django/django",
        pools=("swe_abs_251",),
        resolved=False,
        no_generation=False,
        swe_abs_published=None,
        provisional=Label.BAD,
        assumed=(),
        final=Label.BAD,
        needs=(),
        gold_runsc=True,
        patch_sha256=sha256_bytes(pinned.encode()),
        patch_bytes=len(pinned),
    )
    return Manifest(
        status="frozen",
        created_at="2026-10-28T00:00:00Z",
        pool_choice="swe_abs_251",
        blocked_on=(),
        sources=(),
        pools=(),
        entries=(),
        records=(record,),
    )


def test_a_target_carries_the_patch_issue_base_and_pinned_image_and_never_gold(tmp_path: Path) -> None:
    [t] = build_targets([ITEM], setup(tmp_path), {IID: ROW}, IMAGES, tmp_path, SPECS)
    assert t.id == f"{ENTRY}~{IID}" and "/" not in t.id
    assert (t.tier, t.repo, t.base_sha, t.patch, t.issue) == (
        Tier.A,
        "django/django",
        "a" * 40,
        PATCH,
        ROW.problem_statement,
    )
    assert t.image == f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{IID}@{DIGEST}"
    assert t.patch_sha == sha256_bytes(PATCH.encode())
    assert (t.test_runner, t.test_command) == (RunnerKind.DJANGO, SPECS.specs[0].test_cmd)
    dumped = t.model_dump_json()
    assert "the gold fix" not in dumped and "the hidden test" not in dumped
    assert Target.model_validate_json(dumped) == t


def test_plant_a_patch_on_disk_that_is_not_the_one_the_manifest_pinned_is_refused(tmp_path: Path) -> None:
    manifest = setup(tmp_path, on_disk=PATCH + "+tampered\n", pinned=PATCH)
    with pytest.raises(TargetError, match="not the one the manifest pinned"):
        build_targets([ITEM], manifest, {IID: ROW}, IMAGES, tmp_path, SPECS)


def test_plant_an_instance_without_a_pinned_image_is_refused(tmp_path: Path) -> None:
    empty = IMAGES.model_copy(update={"images": ()})
    with pytest.raises(TargetError, match="no digest-pinned image"):
        build_targets([ITEM], setup(tmp_path), {IID: ROW}, empty, tmp_path, SPECS)


def test_plant_an_instance_whose_version_has_no_swe_bench_spec_is_refused(tmp_path: Path) -> None:
    other = ROW.model_copy(update={"version": "4.2"})
    with pytest.raises(TargetError, match=r"no SWE-bench spec for django/django 4\.2"):
        build_targets([ITEM], setup(tmp_path), {IID: other}, IMAGES, tmp_path, SPECS)


def test_plant_an_empty_patch_or_a_tagged_image_is_never_a_target() -> None:
    with pytest.raises(TargetError, match="empty patch"):
        build_target(ITEM, ROW, "  \n", f"ghcr.io/x@{DIGEST}", SPECS)
    with pytest.raises(ValidationError):
        build_target(ITEM, ROW, PATCH, "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1:latest", SPECS)


@pytest.mark.parametrize(
    ("repo", "kind"),
    [
        ("django/django", RunnerKind.DJANGO),
        ("sympy/sympy", RunnerKind.SYMPY),
        ("sphinx-doc/sphinx", RunnerKind.PYTEST),  # tox ends in pytest
        ("astropy/astropy", RunnerKind.PYTEST),
    ],
)
def test_each_repository_gets_its_runner(repo: str, kind: RunnerKind) -> None:
    assert runner_kind(repo) is kind


def test_target_ids_refuse_a_slash_or_the_separator() -> None:
    assert target_id("e", IID) == f"e~{IID}"
    for bad in (("a/b", IID), ("e", "x~y"), ("", IID)):
        with pytest.raises(TargetError):
            target_id(*bad)


def test_plant_a_django_target_carries_the_harnesss_locale_exports() -> None:
    """Round 5: a pilot target runs its tests the way R24 checked its gold, with the environment the harness's eval
    script exports first (Django 1.11 to 3.2 export LANG and LC_ALL=en_US.UTF-8), as leading NAME=value words."""
    evals = (
        "sed -i '/en_US.UTF-8/s/^# //g' /etc/locale.gen && locale-gen",
        "export LANG=en_US.UTF-8",
        "export LANGUAGE=en_US:en",
        "export LC_ALL=en_US.UTF-8",
    )
    spec = SPECS.specs[0].model_copy(update={"eval_commands": evals})
    t = build_target(ITEM, ROW, PATCH, f"ghcr.io/x@{DIGEST}", SPECS.model_copy(update={"specs": (spec,)}))
    assert t.test_command == f"LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8 {spec.test_cmd}"
    plain = build_target(ITEM, ROW, PATCH, f"ghcr.io/x@{DIGEST}", SPECS)
    assert plain.test_command == SPECS.specs[0].test_cmd  # a spec that exports nothing keeps SWE-bench's command
