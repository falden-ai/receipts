"""The pilot follows from its sample: 20 core calibration targets, two arms that differ only in the attacker model
(REG-561), half of every label at each temperature mode, the move rule written before the run, and a run file and an
arms file in the integrator's format. A forged config, run file or arms file is refused."""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

import pytest

from receipts.contracts import Arm, RunnerKind, Target, Tier
from receipts.eval.labels import Label
from receipts.eval.pilot import (
    ARMS,
    RULE,
    SCHEDULES,
    PilotCheckError,
    PilotConfig,
    TemperatureMode,
    attacker_temperature,
    build_pilot,
    check_arms,
    check_files,
    check_pilot,
    read_arms_file,
    read_run_file,
    rule_sha256,
    verify_pilot,
)
from receipts.eval.sampler import CORE_PLAN, Sample, Side, draw_from_manifest, pilot
from receipts.eval.targets import target_id

DIGEST = "sha256:" + "f" * 64
PATCH_SHA = "a" * 64


def targets_for(sample: Sample) -> list[Target]:
    out = []
    for i in pilot(sample):
        out.append(
            Target(
                id=target_id(i.entry, i.instance_id),
                tier=Tier.A,
                repo=i.repo,
                base_sha="b" * 40,
                patch_sha=PATCH_SHA,
                patch="diff --git a/x b/x\n",
                issue="It breaks.",
                image=f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{i.instance_id}@{DIGEST}",
                test_runner=RunnerKind.DJANGO if i.repo == "django/django" else RunnerKind.PYTEST,
                test_command="pytest -rA",
            )
        )
    return out


@pytest.fixture
def built(synthetic_manifest: Any) -> tuple[PilotConfig, str, str, Sample, str, Any, list[Target]]:
    manifest, text = synthetic_manifest(frozen=True)
    records = tuple(r.model_copy(update={"patch_sha256": PATCH_SHA}) for r in manifest.records)
    manifest = manifest.model_copy(update={"records": records})
    sample = draw_from_manifest(manifest, text, CORE_PLAN)
    sample_text = sample.model_dump_json()
    targets = targets_for(sample)
    cfg, run_text, arms_text = build_pilot(sample, sample_text, manifest, targets)
    return cfg, run_text, arms_text, sample, sample_text, manifest, targets


def test_the_spec_mode_follows_the_spec_and_the_other_runs_everything_at_one() -> None:
    assert [attacker_temperature(TemperatureMode.SPEC, "prompt", n) for n in range(3)] == [0.0, 0.8, 0.8]
    critic = attacker_temperature(TemperatureMode.SPEC, "critic")
    assert (critic, attacker_temperature(TemperatureMode.SPEC, "rewrite")) == (0.0, 0.8)
    for call in ("prompt", "critic", "rewrite"):
        assert attacker_temperature(TemperatureMode.META_DEFAULT, call, 2) == 1.0  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        attacker_temperature(TemperatureMode.SPEC, "prompt", -1)


def test_plant_the_two_schedules_are_the_protocols_numbers_for_every_call() -> None:
    """PI5: only prompt index 2 of meta_default was checked, so its first prompt could drift from 1.0 unseen."""
    for n in range(4):
        assert attacker_temperature(TemperatureMode.META_DEFAULT, "prompt", n) == 1.0
        assert attacker_temperature(TemperatureMode.SPEC, "prompt", n) == (0.0 if n == 0 else 0.8)
    for call, spec in (("critic", 0.0), ("rewrite", 0.8)):
        assert attacker_temperature(TemperatureMode.SPEC, call, 0) == spec  # type: ignore[arg-type]
        assert attacker_temperature(TemperatureMode.META_DEFAULT, call, 0) == 1.0  # type: ignore[arg-type]
    assert {m.value: SCHEDULES[m].model_dump(exclude={"source"}) for m in TemperatureMode} == {
        "spec": {"first_prompt": 0.0, "later_prompts": 0.8, "critic": 0.0, "rewrite": 0.8},
        "meta_default": {"first_prompt": 1.0, "later_prompts": 1.0, "critic": 1.0, "rewrite": 1.0},
    }


def test_half_of_every_label_takes_each_temperature_mode(built: Any) -> None:
    cfg, *_, sample, _, _, _ = built
    modes = Counter((t.label, t.temperatures) for t in cfg.targets)
    assert modes == {
        (Label.BAD, TemperatureMode.SPEC): 4,
        (Label.BAD, TemperatureMode.META_DEFAULT): 4,
        (Label.GOOD, TemperatureMode.SPEC): 4,
        (Label.GOOD, TemperatureMode.META_DEFAULT): 4,
        (Label.PLAUSIBLE_WRONG, TemperatureMode.SPEC): 2,
        (Label.PLAUSIBLE_WRONG, TemperatureMode.META_DEFAULT): 2,
    }
    assert cfg.status == "frozen" and cfg.schedules == SCHEDULES
    by_key = {(i.entry, i.instance_id): i for i in sample.items}
    assert all(by_key[(t.entry, t.instance_id)].side is Side.CALIBRATION for t in cfg.targets)
    assert all(t.target_id == f"{t.entry}~{t.instance_id}" and t.image.endswith(DIGEST) for t in cfg.targets)


def test_the_arms_differ_only_in_the_attacker_model_and_the_rule_is_pinned(built: Any) -> None:
    cfg, _, arms_text, *_ = built
    assert [(a.name, a.attacker_model) for a in cfg.arms] == [
        ("spark", "muse-spark-1.3"),
        ("glimmer", "muse-glimmer-30b"),
    ]
    assert len({(a.step_limit, a.seed) for a in cfg.arms}) == 1
    assert read_arms_file(arms_text) == ARMS and isinstance(json.loads(arms_text), list)
    assert cfg.rule == RULE and cfg.rule_sha256 == rule_sha256()
    assert (RULE.baseline_arm, RULE.candidate_arm, RULE.catch_margin, RULE.max_invalid_repro_pct) == (
        "spark",
        "glimmer",
        1,
        30,
    )


def test_the_run_file_is_one_target_and_its_temperatures_per_line(built: Any) -> None:
    cfg, run_text, *_ = built
    lines = run_text.splitlines()
    assert len(lines) == 20
    first = json.loads(lines[0])
    assert set(first) == {"target", "temperatures"} and first["temperatures"] in ("spec", "meta_default")
    parsed = read_run_file(run_text)
    assert [p.target.id for p in parsed] == [t.target_id for t in cfg.targets]
    assert all(p.temperatures is t.temperatures for p, t in zip(parsed, cfg.targets, strict=True))
    assert all(p.target.test_runner is t.test_runner for p, t in zip(parsed, cfg.targets, strict=True))


def test_verify_accepts_the_pilot_its_sample_gives(built: Any) -> None:
    cfg, run_text, arms_text, sample, sample_text, manifest, targets = built
    verify_pilot(cfg, sample, sample_text, manifest, targets, run_text, arms_text)


@pytest.mark.parametrize(
    "forge",
    [
        lambda c: c.model_copy(
            update={
                "targets": (
                    c.targets[0].model_copy(update={"temperatures": TemperatureMode.META_DEFAULT}),
                    *c.targets[1:],
                )
            }
        ),
        lambda c: c.model_copy(
            update={
                "schedules": {
                    **c.schedules,
                    TemperatureMode.SPEC: c.schedules[TemperatureMode.SPEC].model_copy(update={"later_prompts": 1.0}),
                }
            }
        ),
        lambda c: c.model_copy(update={"targets": c.targets[1:]}),
        lambda c: c.model_copy(
            update={
                "targets": (c.targets[0].model_copy(update={"image": "ghcr.io/x@sha256:" + "0" * 64}), *c.targets[1:])
            }
        ),
        lambda c: c.model_copy(update={"sample_sha256": "0" * 64}),
        lambda c: c.model_copy(update={"rule": c.rule.model_copy(update={"catch_margin": 5})}),
        lambda c: c.model_copy(update={"arms": (c.arms[0], c.arms[1].model_copy(update={"step_limit": 30}))}),
    ],
)
def test_plant_a_forged_pilot_is_refused(forge: object, built: Any) -> None:
    cfg, run_text, arms_text, sample, sample_text, manifest, targets = built
    forged = forge(cfg)  # type: ignore[operator]
    with pytest.raises(PilotCheckError):
        verify_pilot(forged, sample, sample_text, manifest, targets, run_text, arms_text)


def test_plant_a_forged_run_file_or_arms_file_is_refused(built: Any) -> None:
    cfg, run_text, arms_text, *_ = built
    check_files(cfg, run_text, arms_text)
    swapped = run_text.replace('"temperatures":"spec"', '"temperatures":"meta_default"', 1)
    with pytest.raises(PilotCheckError, match="not the run file the pilot pinned"):
        check_files(cfg, swapped, arms_text)
    lines = run_text.splitlines(keepends=True)
    with pytest.raises(PilotCheckError, match="not the run file"):
        check_files(cfg, "".join(reversed(lines)), arms_text)
    more = json.dumps(
        [a.model_dump(mode="json") for a in (*ARMS, Arm(name="wonder", attacker_model="muse-wonder-1.0"))]
    )
    with pytest.raises(PilotCheckError, match="not the arms file"):
        check_files(cfg, run_text, more)


@pytest.mark.parametrize(
    "arms",
    [
        (ARMS[0],),
        (ARMS[0], ARMS[1].model_copy(update={"seed": 7})),
        (ARMS[0], ARMS[1].model_copy(update={"attacker_model": "muse-spark-1.3"})),
        (ARMS[0], ARMS[1].model_copy(update={"name": "other"})),
    ],
)
def test_plant_arms_that_differ_in_more_than_the_model_are_refused(arms: tuple[Arm, ...]) -> None:
    with pytest.raises(PilotCheckError):
        check_arms(arms)


def test_plant_a_pilot_target_from_the_test_side_is_refused(built: Any) -> None:
    cfg, _, _, sample, *_ = built
    test_item = next(i for i in sample.items if i.side is Side.TEST and i.label is cfg.targets[0].label)
    swapped = cfg.targets[0].model_copy(
        update={
            "entry": test_item.entry,
            "instance_id": test_item.instance_id,
            "repo": test_item.repo,
            "model": test_item.model,
            "rank": test_item.rank,
            "target_id": f"{test_item.entry}~{test_item.instance_id}",
        }
    )
    with pytest.raises(PilotCheckError, match="not a core calibration item"):
        check_pilot(cfg.model_copy(update={"targets": (swapped, *cfg.targets[1:])}), sample)


def test_a_pilot_needs_a_target_with_the_pinned_patch_for_every_item(built: Any) -> None:
    _, _, _, sample, sample_text, manifest, targets = built
    with pytest.raises(PilotCheckError, match="no target"):
        build_pilot(sample, sample_text, manifest, targets[1:])
    other = [targets[0].model_copy(update={"patch_sha": "c" * 64}), *targets[1:]]
    with pytest.raises(PilotCheckError, match="not the one the manifest pinned"):
        build_pilot(sample, sample_text, manifest, other)


def test_plant_two_swapped_temperatures_keep_the_half_split_and_are_still_refused(built: Any) -> None:
    cfg, _, _, sample, *_ = built
    bad = [n for n, t in enumerate(cfg.targets) if t.label is Label.BAD]
    a = next(n for n in bad if cfg.targets[n].temperatures is TemperatureMode.SPEC)
    b = next(n for n in bad if cfg.targets[n].temperatures is TemperatureMode.META_DEFAULT)
    targets = list(cfg.targets)
    targets[a] = targets[a].model_copy(update={"temperatures": TemperatureMode.META_DEFAULT})
    targets[b] = targets[b].model_copy(update={"temperatures": TemperatureMode.SPEC})
    with pytest.raises(PilotCheckError, match="not the alternating assignment"):
        check_pilot(cfg.model_copy(update={"targets": tuple(targets)}), sample)
