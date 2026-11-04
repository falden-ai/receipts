"""The files committed under eval/ agree with each other and with the code that wrote them.

These read only committed files (never eval/data/, which is not in git), so they run anywhere. A stale or hand-edited
file fails here: regenerate it with the `python -m receipts.eval` command named in the failure.
"""

from __future__ import annotations

import gzip
import json
import os
import tomllib
from pathlib import Path
from typing import Any

import pytest

from receipts.eval.cloudinit import SWEBENCH_VERSION, check_gpu
from receipts.eval.cloudinit import check as check_user_data
from receipts.eval.entries import SELECTED
from receipts.eval.hashing import sha256_text
from receipts.eval.images import ImageSet
from receipts.eval.labels import SAMPLED, Label, Need
from receipts.eval.manifest import FOUNDER_POOL_CHOICE, Manifest, content_sha256, load
from receipts.eval.pilot import ARMS, PilotConfig, read_run_file, verify_pilot
from receipts.eval.sampler import (
    FROZEN_PLANS,
    Sample,
    SampleCheckError,
    candidates_from_manifest,
    rank,
    verify_sample,
)
from receipts.eval.sources import SOURCES, source
from receipts.eval.specs import SPECS_FILE, run_phase_test_command, runner_kind
from receipts.eval.specs import load as load_specs
from receipts.eval.vmlaunch import files as vm_files
from receipts.eval.vmplan import DIAGNOSE_RUNS_PER_DROP, LABEL_DAY_RUNS, max_drops, quote

ROOT = Path(__file__).resolve().parents[2]
EVAL = ROOT / "eval"


def committed() -> tuple[Manifest, str, str, Sample]:
    """The manifest, its pool list and its sample: sample.json once the manifest is frozen, else the provisional one."""
    manifest = load((EVAL / "manifest.json").read_text())
    pool_text = (EVAL / "pools" / "swe_abs_251.txt").read_text()
    name = "sample.json" if manifest.status == "frozen" else "sample-provisional.json"
    sample_text = (EVAL / name).read_text()
    return manifest, pool_text, sample_text, Sample.model_validate_json(sample_text)


def test_the_pool_list_matches_the_manifest_and_the_paper() -> None:
    manifest = load((EVAL / "manifest.json").read_text())
    text = (EVAL / "pools" / "swe_abs_251.txt").read_text()
    p251 = next(p for p in manifest.pools if p.name == "swe_abs_251")
    assert sha256_text(text) == p251.list_sha256, "rerun `python -m receipts.eval pool` and `manifest`"
    assert len(text.split()) == p251.size == 251
    assert sum((p251.per_repo or {}).values()) == 251


def test_the_manifest_covers_every_entry_on_every_pool_instance_with_its_patch_and_report() -> None:
    manifest = load((EVAL / "manifest.json").read_text())
    ids = (EVAL / "pools" / "swe_abs_251.txt").read_text().split()
    assert manifest.pool_choice == FOUNDER_POOL_CHOICE == "swe_abs_251"
    assert [e.key for e in manifest.entries] == [e.key for e in SELECTED]
    assert {(r.entry, r.instance_id) for r in manifest.records} == {(e.key, i) for e in SELECTED for i in ids}
    assert [s.name for s in manifest.sources] == [s.name for s in SOURCES]
    assert [s.sha256 for s in manifest.sources] == [s.sha256 for s in SOURCES]
    assert manifest.derived_file("swe-bench-verified:rows").derived_from == "swe-bench-verified:dataset"
    assert all(r.patch_sha256 is not None for r in manifest.records), "rerun `python -m receipts.eval patches`"
    assert sum(1 for r in manifest.records if r.report_sha256 is None) <= 1  # the one the bucket has no report for
    if manifest.status == "draft":
        assert manifest.blocked_on, "a draft names what blocks it"


def test_the_sample_is_exactly_the_seeded_draw_from_this_manifest() -> None:
    manifest, pool_text, _, sample = committed()
    verify_sample(sample, manifest, pool_text)  # rerun `python -m receipts.eval sample --plan core`
    assert sample.seed == sample.manifest_content_sha256 == content_sha256(manifest)
    assert sample.plan in FROZEN_PLANS
    frozen = manifest.status == "frozen"
    labels = {(r.entry, r.instance_id): (r.final if frozen else r.provisional) for r in manifest.records}
    assert all(labels[(i.entry, i.instance_id)] is i.label for i in sample.items)
    assert {i.label for i in sample.items} == SAMPLED == {Label.BAD, Label.GOOD, Label.PLAUSIBLE_WRONG}


def test_plant_the_reviewers_swap_on_the_committed_sample_is_refused() -> None:
    """Swap an item for an unchosen candidate of the same label, side, repo and model, rank recomputed."""
    manifest, pool_text, _, sample = committed()
    chosen = {(i.entry, i.instance_id) for i in sample.items}
    for victim in sample.items:
        spare = next(
            (
                c
                for c in candidates_from_manifest(manifest)
                if c.key not in chosen
                and (c.label, c.repo, c.model) == (victim.label, victim.repo, victim.model)
                and sample.split.get(c.instance_id) is victim.side
            ),
            None,
        )
        if spare is not None:
            break
    assert spare is not None
    forged_item = victim.model_copy(
        update={"entry": spare.entry, "instance_id": spare.instance_id, "rank": rank(sample.seed, *spare.key)}
    )
    forged = sample.model_copy(update={"items": tuple(forged_item if i is victim else i for i in sample.items)})
    with pytest.raises(SampleCheckError, match="not the seeded draw"):
        verify_sample(forged, manifest, pool_text)


def test_the_pilot_follows_from_the_committed_sample_and_pins() -> None:
    manifest, _, sample_text, sample = committed()
    cfg = PilotConfig.model_validate_json((EVAL / "pilot.json").read_text())
    run_text = (ROOT / cfg.run_file).read_text()
    arms_text = (ROOT / cfg.arms_file).read_text()
    targets = [line.target for line in read_run_file(run_text)]
    verify_pilot(cfg, sample, sample_text, manifest, targets, run_text, arms_text)  # rerun `receipts eval pilot`
    assert len(cfg.targets) == 20 and cfg.status == sample.status and cfg.arms == ARMS
    pins = ImageSet.model_validate_json((EVAL / "images.json").read_text()).by_instance
    specs = load_specs((EVAL / SPECS_FILE).read_text())
    for t, p in zip(targets, cfg.targets, strict=True):
        assert t.image == pins[p.instance_id].ref  # the digest R24 checks
        assert t.test_runner is runner_kind(t.repo)
        # SWE-bench's command for the instance's own version, after the exports its eval script runs first (Django
        # 1.11 to 3.2: the locale). The round-5 review's V24: any same-repository command passed, so round 4's bare
        # Django 3.0 commands read as Django 4.0's.
        spec = specs.for_instance(p.instance_id, t.repo)
        assert t.test_command == run_phase_test_command(spec), f"{t.id}: not {spec.repo} {spec.version}'s command"


def test_the_spec_table_names_the_version_of_every_pool_instance() -> None:
    manifest = load((EVAL / "manifest.json").read_text())
    specs = load_specs((EVAL / SPECS_FILE).read_text())
    ids = (EVAL / "pools" / "swe_abs_251.txt").read_text().split()
    assert sorted(specs.instances) == sorted(ids), "rerun `python -m receipts.eval test-specs`"
    repos = {r.instance_id: r.repo for r in manifest.records}
    pairs = {(repos[i], specs.instances[i]) for i in ids}
    assert pairs == {(s.repo, s.version) for s in specs.specs}  # each version has its spec, and no spec is spare
    assert specs.for_instance("django__django-11265", "django/django").version == "3.0"


def test_the_eval_extra_pins_the_harness_the_manifest_pins_and_every_package_it_brings() -> None:
    """Round 5's verify:main: the runbook's `receipts eval r24 ... --diagnose` imports swebench, which neither
    pyproject.toml nor uv.lock carried, so the step exited 1 before any instance ran. The eval extra pins swebench
    4.1.0; uv.lock pins it to the bytes of the wheel eval/manifest.json pins (its URL, sha256 and size, the same
    version the specs table and the VM's boot script read), and every package the extra brings is locked to one
    version from PyPI with a sha256 on every file, so `uv sync --frozen --extra eval` resolves nothing."""
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["optional-dependencies"]["eval"] == ["pyarrow==22.0.0", f"swebench=={SWEBENCH_VERSION}"]
    locked = {p["name"]: p for p in tomllib.loads((ROOT / "uv.lock").read_text())["package"]}
    wheel = source("swebench:wheel")
    swebench = locked["swebench"]
    assert swebench["version"] == SWEBENCH_VERSION == load_specs((EVAL / SPECS_FILE).read_text()).swebench_version
    assert wheel.revision == f"swebench=={SWEBENCH_VERSION}"
    assert [(w["url"], w["hash"], w["size"]) for w in swebench["wheels"]] == [
        (wheel.url, f"sha256:{wheel.sha256}", wheel.size)
    ]

    # Every package the eval extra installs: each dependency, and each extra of one a dependency names (fsspec[http]).
    def needs(deps: list[dict[str, Any]]) -> list[tuple[str, str | None]]:
        return [(d["name"], extra) for d in deps for extra in (None, *d.get("extra", ()))]

    todo = needs(locked["falden-receipts"]["optional-dependencies"]["eval"])
    seen: set[tuple[str, str | None]] = set()
    while todo:
        name, extra = todo.pop()
        if (name, extra) not in seen:
            seen.add((name, extra))
            package = locked[name]
            deps = package.get("dependencies", []) if extra is None else package["optional-dependencies"][extra]
            todo += needs(deps)
    closure = {name for name, _ in seen}
    assert {"swebench", "pyarrow", "datasets", "docker", "modal", "unidiff", "aiohttp"} <= closure
    for name in sorted(closure):
        package = locked[name]
        assert package["source"] == {"registry": "https://pypi.org/simple"}, f"{name} is not from PyPI"
        files = [*package.get("wheels", ()), *([package["sdist"]] if "sdist" in package else [])]
        assert files, f"{name} has no file in uv.lock"
        for f in files:
            assert f["url"].startswith("https://files.pythonhosted.org/") and f["hash"].startswith("sha256:"), name


def test_the_test_specs_come_from_the_pinned_wheel() -> None:
    specs = load_specs((EVAL / SPECS_FILE).read_text())
    assert specs.wheel_sha256 == source("swebench:wheel").sha256 and specs.swebench_version == "4.1.0"
    repos = {r.repo for r in load((EVAL / "manifest.json").read_text()).records}
    assert repos <= {s.repo for s in specs.specs}  # rerun `python -m receipts.eval test-specs`


def test_the_image_pins_cover_the_pool_and_the_user_data_lists_exactly_them() -> None:
    images = ImageSet.model_validate_json((EVAL / "images.json").read_text())
    ids = (EVAL / "pools" / "swe_abs_251.txt").read_text().split()
    assert sorted(i.instance_id for i in images.images) == sorted(ids)
    text = gzip.decompress((EVAL / "vm" / "cloud-init.yaml.gz").read_bytes()).decode()
    check_user_data(text)
    for i in images.images:
        assert f"{i.instance_id} {i.digest}\n" in text, "rerun `python -m receipts.eval cloud-init`"
    check_gpu((EVAL / "vm" / "gpu-cloud-init.yaml").read_text())


def test_the_committed_quote_is_what_the_code_computes() -> None:
    committed_quote = json.loads((EVAL / "vm" / "quote.json").read_text())
    assert committed_quote == quote().model_dump(mode="json"), "rerun `python -m receipts.eval quote`"


def test_the_committed_launch_inputs_are_what_the_code_renders() -> None:
    for name, text in vm_files().items():
        assert (EVAL / "vm" / name).read_text() == text, "rerun `python -m receipts.eval vm-files`"
    for script in ("launch.sh", "gpu.sh"):
        path = EVAL / "vm" / script
        body = path.read_text()
        assert os.access(path, os.X_OK) and "iam-instance-profile" not in body
        assert " ssm " not in body and "start-session" not in body  # REG-562: no Systems Manager


def test_the_label_days_runs_are_the_manifests_counts() -> None:
    manifest = load((EVAL / "manifest.json").read_text())
    if manifest.status == "frozen":
        return  # the runs are done; the counts were checked while the manifest was a draft
    pool = {r.instance_id for r in manifest.records}
    assert {
        "r24": len(pool),
        "r24_diagnose": DIAGNOSE_RUNS_PER_DROP * max_drops(len(pool)),
        "labels_gold": len(pool),
        "labels_resolved": sum(1 for r in manifest.records if Need.SWE_ABS_RUN in r.needs),
        "labels_fail_to_pass": sum(1 for r in manifest.records if Need.FAIL_TO_PASS_DETAIL in r.needs),
    } == LABEL_DAY_RUNS
