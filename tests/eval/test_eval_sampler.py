"""The sampler holds every constraint, and `check_sample` catches each one broken on purpose (a plant per rule)."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Iterable
from typing import Any

import pytest

from receipts.eval import sampler as sampler_mod
from receipts.eval.labels import Label
from receipts.eval.manifest import Manifest, content_sha256
from receipts.eval.sampler import (
    CORE_PLAN,
    PILOT_PLAN,
    SPEC_PLAN,
    Candidate,
    InfeasibleSampleError,
    Item,
    Sample,
    SampleCheckError,
    SamplePlan,
    Side,
    candidates_from_manifest,
    cap,
    check_sample,
    check_split,
    draw,
    draw_from_manifest,
    pilot,
    pool_from_manifest,
    rank,
    side_count,
    split_pool,
    verify_sample,
)

REPOS = ["django__django", "sympy__sympy", "astropy__astropy", "pytest-dev__pytest", "psf__requests"]
MODELS = ["m-a", "m-b", "m-c", "m-d"]
DIGEST = "d" * 64
PLAN = SamplePlan(core={Label.BAD: 20, Label.GOOD: 20}, full={Label.BAD: 30, Label.GOOD: 30})


def pool(per_repo: int = 24) -> list[str]:
    return [f"{r}-{1000 + n}" for r in REPOS for n in range(per_repo)]


def repo(iid: str) -> str:
    owner, rest = iid.split("__")
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def candidates(ids: list[str]) -> list[Candidate]:
    """Every model has a patch on every instance; the label alternates so both strata are well supplied."""
    out = []
    for n, iid in enumerate(ids):
        for m, model in enumerate(MODELS):
            label = Label.BAD if (n + m) % 2 == 0 else Label.GOOD
            out.append(Candidate(entry=f"entry-{model}", model=model, instance_id=iid, repo=repo(iid), label=label))
    return out


@pytest.fixture
def sample() -> Sample:
    ids = pool()
    return draw(candidates(ids), ids, PLAN, "seed-1", DIGEST)


def with_items(s: Sample, items: list[Item]) -> Sample:
    return s.model_copy(update={"items": tuple(items)})


def test_a_draw_meets_the_plan_and_every_cap(sample: Sample) -> None:
    check_sample(sample)
    for label in (Label.BAD, Label.GOOD):
        full = [i for i in sample.items if i.label is label]
        core = [i for i in full if i.core]
        assert (len(core), len(full)) == (20, 30)
        assert sum(1 for i in full if i.side is Side.CALIBRATION) == side_count(30, Side.CALIBRATION, 40) == 12
        assert sum(1 for i in core if i.side is Side.CALIBRATION) == 8


def test_the_spec_numbers_round_the_way_the_spec_reads() -> None:
    assert (side_count(100, Side.CALIBRATION, 40), side_count(100, Side.TEST, 40)) == (40, 60)
    assert (side_count(150, Side.CALIBRATION, 40), side_count(150, Side.TEST, 40)) == (60, 90)
    assert (cap(100, 35), cap(150, 35), cap(150, 40)) == (35, 52, 60)


def test_the_same_seed_gives_the_same_sample_whatever_the_input_order() -> None:
    ids = pool()
    cands = candidates(ids)
    shuffled = cands[:]
    random.Random(7).shuffle(shuffled)
    a = draw(cands, ids, PLAN, "seed-1", DIGEST)
    b = draw(shuffled, list(reversed(ids)), PLAN, "seed-1", DIGEST)
    assert a == b
    c = draw(cands, ids, PLAN, "seed-2", DIGEST)
    assert {(i.entry, i.instance_id) for i in a.items} != {(i.entry, i.instance_id) for i in c.items}


def test_the_core_is_nested_in_the_full_sample(sample: Sample) -> None:
    core = {(i.entry, i.instance_id) for i in sample.items if i.core}
    full = {(i.entry, i.instance_id) for i in sample.items}
    assert core < full


def test_the_split_is_by_instance_and_stratified_by_repository() -> None:
    ids = pool(per_repo=7)  # 7 per repo: 40% is 2.8, so each repo gets 2 or 3
    split = split_pool(ids, "seed-1", 40)
    assert set(split) == set(ids)
    for r in REPOS:
        cal = sum(1 for i, s in split.items() if i.startswith(r) and s is Side.CALIBRATION)
        assert cal in (2, 3)
    assert sum(1 for s in split.values() if s is Side.CALIBRATION) == 14  # round(35 * 0.4)
    check_split(split, "seed-1", 40)


def test_a_split_that_does_not_follow_from_the_seed_is_refused() -> None:
    ids = pool(per_repo=7)
    split = split_pool(ids, "seed-1", 40)
    moved = dict(split)
    first = next(i for i, s in split.items() if s is Side.CALIBRATION)
    moved[first] = Side.TEST
    with pytest.raises(SampleCheckError, match="does not follow from the seed"):
        check_split(moved, "seed-1", 40)


# ---------------------------------------------------------------- plants: each rule broken on purpose


def test_plant_a_repository_over_35_percent_is_refused(sample: Sample) -> None:
    """Swap extension items for real, unchosen django candidates of the same model and side, so the repository cap
    is the only rule the planted sample breaks."""
    items = list(sample.items)
    chosen = {(i.entry, i.instance_id) for i in items}
    spare = [
        c
        for c in candidates(pool())
        if c.label is Label.BAD
        and c.repo == "django/django"
        and sample.split[c.instance_id] is Side.TEST
        and c.key not in chosen
    ]
    side_items = [i for i in items if i.label is Label.BAD and i.side is Side.TEST]
    need = cap(len(side_items), 35) + 1 - sum(1 for i in side_items if i.repo == "django/django")
    victims = [i for i in side_items if i.repo != "django/django" and not i.core]
    swapped = 0
    for victim in victims:
        match = next((c for c in spare if c.model == victim.model), None)
        if match is None or swapped == need:
            continue
        spare.remove(match)
        items[items.index(victim)] = victim.model_copy(
            update={
                "entry": match.entry,
                "instance_id": match.instance_id,
                "repo": match.repo,
                "rank": rank(sample.seed, match.entry, match.instance_id),
            }
        )
        swapped += 1
    assert swapped == need > 0
    with pytest.raises(SampleCheckError, match="from django/django, over 35%"):
        check_sample(with_items(sample, items))


def test_plant_a_model_over_40_percent_is_refused(sample: Sample) -> None:
    items = list(sample.items)
    good = [i for i in items if i.label is Label.GOOD and i.side is Side.TEST]
    for victim in good:
        items[items.index(victim)] = victim.model_copy(update={"model": "m-a"})
    with pytest.raises(SampleCheckError, match=r"from m-a, over 40%"):
        check_sample(with_items(sample, items))


def test_plant_fewer_than_three_models_is_refused() -> None:
    ids = pool()
    two_models = [c for c in candidates(ids) if c.model in ("m-a", "m-b")]
    plan = PLAN.model_copy(update={"max_model_pct": 100})
    with pytest.raises(SampleCheckError, match="models, need at least 3"):
        draw(two_models, ids, plan, "seed-1", DIGEST)


def test_plant_an_instance_on_both_sides_is_refused(sample: Sample) -> None:
    items = list(sample.items)
    victim = next(i for i in items if i.side is Side.CALIBRATION)
    items[items.index(victim)] = victim.model_copy(update={"side": Side.TEST})
    with pytest.raises(SampleCheckError, match="is not its instance's side"):
        check_sample(with_items(sample, items))


def test_plant_a_missing_item_is_refused(sample: Sample) -> None:
    with pytest.raises(SampleCheckError, match="planned"):
        check_sample(with_items(sample, list(sample.items)[1:]))


def test_plant_a_duplicate_item_is_refused(sample: Sample) -> None:
    items = list(sample.items)
    with pytest.raises(SampleCheckError, match="appears twice"):
        check_sample(with_items(sample, [*items[:-1], items[0]]))


def test_plant_a_hand_picked_item_is_refused(sample: Sample) -> None:
    items = list(sample.items)
    items[0] = items[0].model_copy(update={"rank": "0" * 64})
    with pytest.raises(SampleCheckError, match="rank does not follow from the seed"):
        check_sample(with_items(sample, items))


def test_plant_a_label_outside_the_plan_is_refused(sample: Sample) -> None:
    items = list(sample.items)
    items[0] = items[0].model_copy(update={"label": Label.PLAUSIBLE_WRONG})
    with pytest.raises(SampleCheckError, match="not in the plan"):
        check_sample(with_items(sample, items))


def test_plant_a_core_item_moved_out_of_the_core_is_refused(sample: Sample) -> None:
    items = list(sample.items)
    victim = next(i for i in items if i.core)
    items[items.index(victim)] = victim.model_copy(update={"core": False})
    with pytest.raises(SampleCheckError, match="core"):
        check_sample(with_items(sample, items))


def test_too_few_candidates_is_infeasible_not_silently_short() -> None:
    ids = pool(per_repo=2)  # 10 instances, 20 BAD candidates, 30 planned
    with pytest.raises(InfeasibleSampleError, match="items under the caps"):
        draw(candidates(ids), ids, PLAN, "seed-1", DIGEST)


def test_caps_bind_the_draw_itself_not_only_the_check() -> None:
    """With one repo holding most candidates, the draw takes at most 35% from it and fills from the rest."""
    ids = [f"django__django-{n}" for n in range(200)] + [f"{r}-{n}" for r in REPOS[1:] for n in range(12)]
    s = draw(candidates(ids), ids, PLAN, "seed-1", DIGEST)
    for label in (Label.BAD, Label.GOOD):
        full = [i for i in s.items if i.label is label]
        assert sum(1 for i in full if i.repo == "django/django") <= cap(30, 35)


def test_candidates_off_the_pool_or_twice_are_refused() -> None:
    ids = pool()
    cands = candidates(ids)
    with pytest.raises(ValueError, match="not on the pool"):
        draw(cands, ids[1:], PLAN, "seed-1", DIGEST)
    with pytest.raises(ValueError, match="duplicate candidate"):
        draw([*cands, cands[0]], ids, PLAN, "seed-1", DIGEST)


def test_a_candidate_with_the_wrong_repository_is_refused() -> None:
    with pytest.raises(ValueError, match="is in django/django"):
        Candidate(entry="e", model="m", instance_id="django__django-1", repo="sympy/sympy", label=Label.BAD)


def test_the_plan_must_nest() -> None:
    with pytest.raises(ValueError, match="core <= full"):
        SamplePlan(core={Label.BAD: 50}, full={Label.BAD: 40})
    with pytest.raises(ValueError, match="never sampled"):
        SamplePlan(core={Label.PENDING: 1}, full={Label.PENDING: 1})


def test_the_pilot_comes_only_from_core_calibration_items(sample: Sample) -> None:
    targets = pilot(sample, per_label={Label.BAD: 4, Label.GOOD: 4})
    assert len(targets) == 8
    assert all(i.core and i.side is Side.CALIBRATION for i in targets)
    assert {i.label for i in targets} == {Label.BAD, Label.GOOD}
    with pytest.raises(InfeasibleSampleError):
        pilot(sample, per_label={Label.BAD: 99})


def test_the_spec_plan_has_the_plausible_wrong_stratum_and_its_core() -> None:
    assert SPEC_PLAN.core == {Label.BAD: 100, Label.GOOD: 100, Label.PLAUSIBLE_WRONG: 50}
    assert SPEC_PLAN.full == {Label.BAD: 150, Label.GOOD: 150, Label.PLAUSIBLE_WRONG: 75}
    assert CORE_PLAN.core == CORE_PLAN.full == SPEC_PLAN.core


def test_rank_is_a_pure_function_of_seed_and_item() -> None:
    assert rank("s", "e", "django__django-1") == rank("s", "e", "django__django-1")
    assert rank("s", "e", "django__django-1") != rank("t", "e", "django__django-1")


# ---------------------------------------------------------------- the sample follows from its manifest (verify_sample)

SMALL = SamplePlan(
    core={Label.BAD: 20, Label.GOOD: 20, Label.PLAUSIBLE_WRONG: 10},
    full={Label.BAD: 30, Label.GOOD: 30, Label.PLAUSIBLE_WRONG: 15},
)


def swap_plant(sample: Sample, manifest: Manifest) -> Sample:
    """The reviewer's refutation: swap one item for an unchosen candidate of the same label, side, repo and model,
    with its rank recomputed, so every rule check_sample knows still holds."""
    chosen = {(i.entry, i.instance_id) for i in sample.items}
    victim = next(i for i in sample.items if not i.core)
    spare = next(
        c
        for c in candidates_from_manifest(manifest)
        if c.key not in chosen
        and c.label is victim.label
        and c.repo == victim.repo
        and c.model == victim.model
        and sample.split.get(c.instance_id) is victim.side
    )
    swapped = victim.model_copy(
        update={
            "entry": spare.entry,
            "instance_id": spare.instance_id,
            "rank": rank(sample.seed, spare.entry, spare.instance_id),
        }
    )
    return sample.model_copy(update={"items": tuple(swapped if i is victim else i for i in sample.items)})


def test_verify_accepts_exactly_the_seeded_draw_for_a_draft_and_a_frozen_manifest(synthetic_manifest: Any) -> None:
    for frozen in (False, True):
        m, text = synthetic_manifest(frozen=frozen)
        plan = CORE_PLAN if frozen else SMALL
        s = draw_from_manifest(m, text, plan)
        assert s.seed == s.manifest_content_sha256 == content_sha256(m)
        assert s.status == ("frozen" if frozen else "provisional")
        verify_sample(s, m, text)


def test_the_core_does_not_depend_on_the_full_extension(synthetic_manifest: Any) -> None:
    m, text = synthetic_manifest(frozen=True)
    core = {(i.entry, i.instance_id) for i in draw_from_manifest(m, text, CORE_PLAN).items}
    assert core == {(i.entry, i.instance_id) for i in draw_from_manifest(m, text, SPEC_PLAN).items if i.core}


def test_plant_a_swapped_item_with_its_rank_recomputed_passes_check_sample_but_not_verify(
    synthetic_manifest: Any,
) -> None:
    m, text = synthetic_manifest(frozen=False)
    s = draw_from_manifest(m, text, SMALL)
    forged = swap_plant(s, m)
    check_sample(forged)  # every rule check_sample knows still holds
    with pytest.raises(SampleCheckError, match="not the seeded draw"):
        verify_sample(forged, m, text)


def test_plant_a_frozen_manifest_drawn_with_another_seed_is_refused(synthetic_manifest: Any) -> None:
    m, text = synthetic_manifest(frozen=True)
    with pytest.raises(SampleCheckError, match="seeded by the manifest's content hash"):
        draw_from_manifest(m, text, CORE_PLAN, seed="lucky-7")
    shopped = draw(
        candidates_from_manifest(m), pool_from_manifest(m, text.split()), CORE_PLAN, "lucky-7", content_sha256(m),
        status="frozen",
    )  # fmt: skip
    check_sample(shopped)
    with pytest.raises(SampleCheckError, match="seed must be its manifest's content hash"):
        verify_sample(shopped, m, text)


def test_plant_a_frozen_sample_on_a_plan_the_spec_does_not_allow_is_refused(synthetic_manifest: Any) -> None:
    m, text = synthetic_manifest(frozen=True)
    with pytest.raises(SampleCheckError, match="spec's plan or its core"):
        draw_from_manifest(m, text, SMALL)


def test_plant_a_sample_from_another_manifest_or_pool_list_is_refused(synthetic_manifest: Any) -> None:
    m, text = synthetic_manifest(frozen=False)
    s = draw_from_manifest(m, text, SMALL)
    other = m.model_copy(update={"records": m.records[:-1]})
    with pytest.raises(SampleCheckError, match="another manifest"):
        verify_sample(s, other, text)
    with pytest.raises(SampleCheckError, match="is not the swe_abs_251 list the manifest pinned"):
        verify_sample(s, m, text + "django__django-9999\n")


# ---------------------------------------------------------------- R24: only gold that passed under runsc is sampled


def test_plant_a_dropped_instance_is_never_a_candidate_or_on_the_split(synthetic_manifest: Any) -> None:
    dropped = "django__django-2000"
    for frozen in (False, True):
        m, text = synthetic_manifest(frozen=frozen, gold={dropped: False})
        assert dropped not in {c.instance_id for c in candidates_from_manifest(m)}
        assert dropped not in pool_from_manifest(m, text.split())
        s = draw_from_manifest(m, text, CORE_PLAN if frozen else SMALL)
        assert dropped not in s.split and all(i.instance_id != dropped for i in s.items)


def test_plant_an_unchecked_instance_in_a_frozen_manifest_is_refused(synthetic_manifest: Any) -> None:
    unchecked = "django__django-2000"
    m, text = synthetic_manifest(frozen=True, gold={unchecked: None})
    with pytest.raises(SampleCheckError, match="runsc never checked"):
        candidates_from_manifest(m)
    with pytest.raises(SampleCheckError, match="runsc never checked"):
        pool_from_manifest(m, text.split())
    draft, _ = synthetic_manifest(frozen=False, gold={unchecked: None})  # a draft may still be waiting on it
    assert unchecked in {c.instance_id for c in candidates_from_manifest(draft)}


def test_plant_records_that_disagree_on_an_instances_runsc_verdict_are_refused(synthetic_manifest: Any) -> None:
    m, text = synthetic_manifest(frozen=False)
    first = m.records[0]
    bent = (first.model_copy(update={"gold_runsc": False}), *m.records[1:])
    with pytest.raises(SampleCheckError, match="disagree"):
        pool_from_manifest(m.model_copy(update={"records": bent}), text.split())


# ---------------------------------------------------------------- the split's per-repository share and the pilot


def test_plant_an_unstratified_split_is_refused_even_if_the_split_function_regressed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_split recomputes the split from the seed, so a broken split_pool would agree with itself; the
    per-repository share check still refuses an unstratified split."""
    ids = pool(per_repo=10)

    def unstratified(instance_ids: Iterable[str], seed: str, calibration_pct: int = 40) -> dict[str, Side]:
        ordered = sorted(instance_ids)
        cut = len(ordered) * calibration_pct // 100
        return {i: Side.CALIBRATION if n < cut else Side.TEST for n, i in enumerate(ordered)}

    monkeypatch.setattr(sampler_mod, "split_pool", unstratified)
    with pytest.raises(SampleCheckError, match="instances in calibration, target 40%"):
        check_split(unstratified(ids, "s", 40), "s", 40)


def test_the_pilot_is_8_bad_8_good_4_plausible_wrong_from_core_calibration_with_no_instance_twice(
    synthetic_manifest: Any,
) -> None:
    m, text = synthetic_manifest(frozen=True)
    s = draw_from_manifest(m, text, CORE_PLAN)
    targets = pilot(s)
    assert Counter(i.label for i in targets) == PILOT_PLAN == {Label.BAD: 8, Label.GOOD: 8, Label.PLAUSIBLE_WRONG: 4}
    assert all(i.core and i.side is Side.CALIBRATION for i in targets)
    assert len({i.instance_id for i in targets}) == 20
    assert max(Counter(i.repo for i in targets).values()) <= cap(20, 35)
    assert max(Counter(i.model for i in targets).values()) <= cap(20, 40)


def test_plant_a_pilot_the_caps_cannot_fill_is_infeasible_not_short(synthetic_manifest: Any) -> None:
    m, text = synthetic_manifest(frozen=True)
    s = draw_from_manifest(m, text, CORE_PLAN)
    only_one_repo = s.model_copy(
        update={"items": tuple(i for i in s.items if i.repo == "django/django" or i.side is Side.TEST)}
    )
    with pytest.raises(InfeasibleSampleError, match="pilot"):
        pilot(only_one_repo)


def test_plant_a_record_the_founder_dropped_is_never_a_candidate(synthetic_manifest: Any) -> None:
    """A label drop (`manifest --label-drop`) makes a record DROPPED; no sample ever draws it, frozen or draft."""
    for frozen in (True, False):
        m, _ = synthetic_manifest(frozen=frozen)
        victim = m.records[0]
        gone = victim.model_copy(update={"final": Label.DROPPED, "provisional": Label.DROPPED, "needs": ()})
        keys = {c.key for c in candidates_from_manifest(m.model_copy(update={"records": (gone, *m.records[1:])}))}
        assert (victim.entry, victim.instance_id) not in keys and len(keys) == len(m.records) - 1
