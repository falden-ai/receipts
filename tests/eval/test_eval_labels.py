"""Label rules (spec section 3) and the evidence conflicts that must never be resolved silently."""

from __future__ import annotations

import pytest

from receipts.eval.labels import (
    ASSUME_PUBLISHED_SWE_ABS,
    ASSUME_UNRESOLVED_FAILS_F2P,
    Evidence,
    EvidenceConflictError,
    Label,
    Need,
    final_label,
    needs,
    provisional_label,
)


@pytest.mark.parametrize(
    ("evidence", "label"),
    [
        (Evidence(resolved=True, swe_abs_passed=True, gold_passes_swe_abs=True), Label.GOOD),
        (Evidence(resolved=True, swe_abs_passed=False, gold_passes_swe_abs=True), Label.PLAUSIBLE_WRONG),
        (Evidence(resolved=True, swe_abs_passed=False, gold_passes_swe_abs=False), Label.NO_ABS_ORACLE),
        (Evidence(resolved=True, swe_abs_passed=True, gold_passes_swe_abs=False), Label.NO_ABS_ORACLE),
        (Evidence(resolved=False, fail_to_pass_passed=False), Label.BAD),
        (Evidence(resolved=False, fail_to_pass_passed=True, pass_to_pass_passed=False), Label.REGRESSION_ONLY),
        (Evidence(resolved=False, empty_patch=True), Label.EMPTY),
        (Evidence(resolved=False), Label.PENDING),
        (Evidence(resolved=True), Label.PENDING),
        (Evidence(), Label.PENDING),
    ],
)
def test_final_labels_follow_the_spec(evidence: Evidence, label: Label) -> None:
    assert final_label(evidence) is label


def test_plant_a_patch_run_without_the_gold_run_is_never_final() -> None:
    """REG-564: GOOD and PLAUSIBLE_WRONG need the SWE-ABS tests on the patch and on gold."""
    assert final_label(Evidence(resolved=True, swe_abs_passed=True)) is Label.PENDING
    assert final_label(Evidence(resolved=True, swe_abs_passed=False)) is Label.PENDING
    assert final_label(Evidence(resolved=True, gold_passes_swe_abs=True)) is Label.PENDING


def test_a_published_verdict_never_makes_a_final_label() -> None:
    assert final_label(Evidence(resolved=True, swe_abs_published=True)) is Label.PENDING


@pytest.mark.parametrize(
    "evidence",
    [
        Evidence(resolved=True, fail_to_pass_passed=False),
        Evidence(resolved=True, pass_to_pass_passed=False),
        Evidence(resolved=False, fail_to_pass_passed=True, pass_to_pass_passed=True),
        Evidence(resolved=True, empty_patch=True),
        Evidence(resolved=False, fail_to_pass_passed=False, swe_abs_passed=True),  # SWE-ABS runs only on resolved
    ],
)
def test_plant_contradicting_evidence_is_refused(evidence: Evidence) -> None:
    with pytest.raises(EvidenceConflictError):
        final_label(evidence)
    with pytest.raises(EvidenceConflictError):
        provisional_label(evidence)


def test_provisional_labels_name_what_they_assumed() -> None:
    bad = provisional_label(Evidence(resolved=False))
    assert (bad.label, bad.assumed) == (Label.BAD, (ASSUME_UNRESOLVED_FAILS_F2P,))
    good = provisional_label(Evidence(resolved=True, swe_abs_published=True))
    assert (good.label, good.assumed) == (Label.GOOD, (ASSUME_PUBLISHED_SWE_ABS,))
    wrong = provisional_label(Evidence(resolved=True, swe_abs_published=False))
    assert wrong.label is Label.PLAUSIBLE_WRONG
    unknown = provisional_label(Evidence(resolved=True))
    assert (unknown.label, unknown.assumed) == (Label.PENDING, ())
    ours = Evidence(resolved=True, swe_abs_passed=False, gold_passes_swe_abs=True, swe_abs_published=True)
    assert provisional_label(ours) == provisional_label(ours).model_copy(
        update={"label": Label.PLAUSIBLE_WRONG, "assumed": ()}
    )  # our runs beat the published one
    no_oracle = provisional_label(Evidence(resolved=True, swe_abs_published=True, gold_passes_swe_abs=False))
    assert no_oracle.label is Label.NO_ABS_ORACLE  # our gold run beats SWE-ABS's validation


def test_needs_lists_what_stands_between_a_patch_and_its_label() -> None:
    assert needs(Evidence(resolved=True)) == (Need.PATCH, Need.SWE_ABS_RUN, Need.GOLD_SWE_ABS, Need.GOLD_RUNSC)
    assert needs(Evidence(resolved=False, patch_pulled=True, gold_passes_runsc=True)) == (Need.FAIL_TO_PASS_DETAIL,)
    assert needs(Evidence(resolved=False, empty_patch=True, patch_pulled=True)) == ()
    assert needs(Evidence(resolved=False, empty_patch=True)) == ()  # an entry that submitted nothing blocks nothing
    done = Evidence(
        resolved=True, swe_abs_passed=True, gold_passes_swe_abs=True, patch_pulled=True, gold_passes_runsc=True
    )
    assert needs(done) == ()
    no_oracle = Evidence(resolved=True, gold_passes_swe_abs=False, patch_pulled=True, gold_passes_runsc=True)
    assert needs(no_oracle) == ()  # a patch run is pointless when gold fails the SWE-ABS tests


def test_plant_an_instance_r24_dropped_needs_no_label_runs() -> None:
    """The re-review's note: freeze must not wait on label runs for instances R24 drops."""
    dropped = Evidence(resolved=True, patch_pulled=True, gold_passes_runsc=False)
    assert needs(dropped) == ()
    assert needs(Evidence(resolved=False, patch_pulled=True, gold_passes_runsc=False)) == ()
    assert needs(dropped.model_copy(update={"gold_passes_runsc": None})) == (
        Need.SWE_ABS_RUN,
        Need.GOLD_SWE_ABS,
        Need.GOLD_RUNSC,
    )


def test_a_patch_whose_resolved_status_is_unknown_needs_a_run_and_is_never_bad() -> None:
    e = Evidence(resolved=None, patch_pulled=True, gold_passes_runsc=True)
    assert needs(e) == (Need.FAIL_TO_PASS_DETAIL,)
    assert provisional_label(e).label is Label.PENDING and final_label(e) is Label.PENDING
