"""Tier A labels (spec section 3) and what each one still needs.

- GOOD: passes FAIL_TO_PASS, PASS_TO_PASS and the SWE-ABS tests.
- BAD: fails FAIL_TO_PASS.
- PLAUSIBLE_WRONG: resolved (passes FAIL_TO_PASS and PASS_TO_PASS) but fails the SWE-ABS tests.

More outcomes keep the three honest. REGRESSION_ONLY is a patch that passes FAIL_TO_PASS and fails PASS_TO_PASS: it
is none of the three, so it stays out of the sample. EMPTY is an entry that submitted no patch: there is nothing to
verify. NO_ABS_ORACLE is a resolved patch on an instance whose own gold fails the SWE-ABS tests in our run: there the
SWE-ABS tests cannot tell GOOD from PLAUSIBLE_WRONG, so the record stays out of the sample too. UNUSABLE is a patch that
did not apply in our run (after SWE-ABS's own removal of the chunks its test patch conflicts with): no attacker could
verify it. DROPPED is the founder's call on a record whose label run kept failing on infrastructure: it is never
sampled, and the manifest records why (`manifest --label-drop`). PENDING means a run that decides the label has not
happened yet.

A final label uses only evidence this project can stand behind: the published resolved status, the published per-test
report of FAIL_TO_PASS and PASS_TO_PASS (checked against the dataset's test lists, `receipts.eval.reports`), and our
own harness runs on the VM (`receipts.eval.labelrun`): the SWE-ABS tests on the patch and on the instance's gold, and a
FAIL_TO_PASS run where no report was published. A GOOD or PLAUSIBLE_WRONG label needs both SWE-ABS runs, the patch's
and gold's (REG-564). A provisional label may also assume, and lists what it assumed:
- `unresolved_fails_f2p`: an unresolved patch fails FAIL_TO_PASS (true unless it only breaks PASS_TO_PASS);
- `published_swe_abs`: SWE-ABS's own published verdict stands in for our run of the SWE-ABS tests.

An instance whose gold fails under the run phase (R24) is dropped: it is never sampled, so its records need no label
runs at all.
"""

from __future__ import annotations

from enum import StrEnum

from receipts.contracts import Frozen


class Label(StrEnum):
    GOOD = "GOOD"
    BAD = "BAD"
    PLAUSIBLE_WRONG = "PLAUSIBLE_WRONG"
    REGRESSION_ONLY = "REGRESSION_ONLY"
    EMPTY = "EMPTY"
    NO_ABS_ORACLE = "NO_ABS_ORACLE"
    UNUSABLE = "UNUSABLE"
    DROPPED = "DROPPED"  # the founder's: its label run was INFRA on every run
    PENDING = "PENDING"


SAMPLED: frozenset[Label] = frozenset({Label.GOOD, Label.BAD, Label.PLAUSIBLE_WRONG})

ASSUME_UNRESOLVED_FAILS_F2P = "unresolved_fails_f2p"
ASSUME_PUBLISHED_SWE_ABS = "published_swe_abs"


class Need(StrEnum):
    PATCH = "patch"  # the patch itself is not pulled yet
    FAIL_TO_PASS_DETAIL = "fail_to_pass_detail"  # no per-test report: does it fail FAIL_TO_PASS, or only PASS_TO_PASS?
    SWE_ABS_RUN = "swe_abs_run"  # resolved: our run of the SWE-ABS tests on the patch
    GOLD_SWE_ABS = "gold_swe_abs"  # resolved: our run of the SWE-ABS tests on the instance's gold
    GOLD_RUNSC = "gold_runsc"  # the instance's gold has not passed in the run phase under runsc yet (R24)


class Evidence(Frozen):
    """What is known about one agent patch on one instance."""

    resolved: bool | None = None  # published: passed FAIL_TO_PASS and PASS_TO_PASS in the submitter's run
    fail_to_pass_passed: bool | None = None  # from a per-test report or our run
    pass_to_pass_passed: bool | None = None
    swe_abs_passed: bool | None = None  # our run of the SWE-ABS tests on the patch
    gold_passes_swe_abs: bool | None = None  # our run of the SWE-ABS tests on the instance's gold
    swe_abs_published: bool | None = None  # SWE-ABS's own verdict, provisional only
    empty_patch: bool | None = None
    patch_applies: bool | None = None  # False: our harness run could not apply it
    patch_pulled: bool = False
    gold_passes_runsc: bool | None = None
    label_dropped: bool = False  # the founder dropped it: a label run it waited on was INFRA on every run


class EvidenceConflictError(ValueError):
    """Evidence that contradicts itself; never silently resolved."""


def _check(e: Evidence) -> None:
    if e.resolved is True and e.fail_to_pass_passed is False:
        raise EvidenceConflictError("resolved, yet a report says FAIL_TO_PASS failed")
    if e.resolved is True and e.pass_to_pass_passed is False:
        raise EvidenceConflictError("resolved, yet a report says PASS_TO_PASS failed")
    if e.resolved is False and e.fail_to_pass_passed is True and e.pass_to_pass_passed is True:
        raise EvidenceConflictError("unresolved, yet a report says both FAIL_TO_PASS and PASS_TO_PASS passed")
    if e.empty_patch is True and e.resolved is True:
        raise EvidenceConflictError("an empty patch cannot resolve an instance")
    if e.swe_abs_passed is not None and e.resolved is not True:
        raise EvidenceConflictError("the SWE-ABS tests run only on resolved patches")


def final_label(e: Evidence) -> Label:
    _check(e)
    if e.empty_patch is True:
        return Label.EMPTY
    if e.patch_applies is False:
        return Label.UNUSABLE
    if e.label_dropped:
        return Label.DROPPED
    if e.resolved is None:
        return Label.PENDING
    if e.resolved is False:
        if e.fail_to_pass_passed is False:
            return Label.BAD
        if e.fail_to_pass_passed is True:
            return Label.REGRESSION_ONLY
        return Label.PENDING
    if e.gold_passes_swe_abs is False:
        return Label.NO_ABS_ORACLE
    if e.gold_passes_swe_abs is None or e.swe_abs_passed is None:
        return Label.PENDING
    return Label.GOOD if e.swe_abs_passed else Label.PLAUSIBLE_WRONG


class Provisional(Frozen):
    label: Label
    assumed: tuple[str, ...]


def provisional_label(e: Evidence) -> Provisional:
    """The best label from what is public today, and every assumption it took to get there."""
    _check(e)
    assumed: list[str] = []
    f2p = e.fail_to_pass_passed
    if e.resolved is False and f2p is None and e.empty_patch is not True:
        f2p = False
        assumed.append(ASSUME_UNRESOLVED_FAILS_F2P)
    swe_abs, gold_abs = e.swe_abs_passed, e.gold_passes_swe_abs
    if e.resolved is True:
        if swe_abs is None and e.swe_abs_published is not None:
            swe_abs = e.swe_abs_published
            assumed.append(ASSUME_PUBLISHED_SWE_ABS)
        if gold_abs is None and swe_abs is not None:
            gold_abs = True  # SWE-ABS validated its tests on gold before publishing them
            if ASSUME_PUBLISHED_SWE_ABS not in assumed:
                assumed.append(ASSUME_PUBLISHED_SWE_ABS)
    label = final_label(
        e.model_copy(update={"fail_to_pass_passed": f2p, "swe_abs_passed": swe_abs, "gold_passes_swe_abs": gold_abs})
    )
    return Provisional(label=label, assumed=tuple(assumed))


def needs(e: Evidence) -> tuple[Need, ...]:
    """The runs and pulls that still stand between this patch and a final label. An empty patch needs nothing, nor
    does one that did not apply, one the founder dropped, or a patch on an instance R24 dropped: none of them is ever
    sampled."""
    if e.empty_patch is True or e.patch_applies is False or e.gold_passes_runsc is False or e.label_dropped:
        return ()
    out: list[Need] = []
    if not e.patch_pulled:
        out.append(Need.PATCH)
    if e.resolved is None or (e.resolved is False and e.fail_to_pass_passed is None):
        out.append(Need.FAIL_TO_PASS_DETAIL)
    if e.resolved is True and e.gold_passes_swe_abs is not False:
        if e.swe_abs_passed is None:
            out.append(Need.SWE_ABS_RUN)
        if e.gold_passes_swe_abs is None:
            out.append(Need.GOLD_SWE_ABS)
    if e.gold_passes_runsc is None:
        out.append(Need.GOLD_RUNSC)
    return tuple(out)
