"""Manifest records, the freeze gate and the patch index, on synthetic inputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from receipts.eval.entries import SELECTED, ResolvedList
from receipts.eval.hashing import pretty_json, sha256_bytes
from receipts.eval.labelrun import LabelEvidence, LandedLabelDrop, RecordLabelEvidence
from receipts.eval.labels import ASSUME_UNRESOLVED_FAILS_F2P, SAMPLED, Label, Need
from receipts.eval.manifest import (
    GoldRunscRecord,
    LabelRunRecord,
    Manifest,
    NotFreezableError,
    PatchIndex,
    PatchIndexEntry,
    R24InputRecord,
    Record,
    _agreement,
    blockers,
    build_records,
    check_label_keys,
    content_sha256,
    dump,
    freeze,
    load,
    load_patch_index,
    patch_index_path,
    pool_records,
)
from receipts.eval.pool import Pool, Verdict
from receipts.eval.reports import Report, ReportError, parse_report
from receipts.eval.rows import Row

ENTRY = SELECTED[0]
POOL_IDS = ("django__django-1", "django__django-2", "sympy__sympy-3", "sympy__sympy-4")
POOL = Pool(
    name="swe_abs_251",
    instance_ids=POOL_IDS,
    per_repo={"django/django": 2, "sympy/sympy": 2},
    rule="synthetic",
    checks={},
)
F2P, P2P = ("t::fix",), ("t::old",)
ROWS = {
    iid: Row(
        FAIL_TO_PASS=F2P,
        PASS_TO_PASS=P2P,
        base_commit="a" * 40,
        created_at="2020-01-01T00:00:00Z",
        difficulty="<15 min fix",
        environment_setup_commit="b" * 40,
        hints_text="",
        instance_id=iid,
        patch="diff --git a/gold b/gold\n",
        problem_statement="It breaks.",
        repo="django/django" if iid.startswith("django") else "sympy/sympy",
        test_patch="diff --git a/t b/t\n",
        version="3.0",
    )
    for iid in POOL_IDS
}


def verdict(iid: str, status: str, original: bool | None) -> Verdict:
    return Verdict(
        agent_id=ENTRY.swe_abs_agent_id,
        agent_name=ENTRY.key,
        instance_id=iid,
        original_tests_passed=original,
        swe_abs_tests_passed={"passed": True, "failed": False}.get(status),
        re_evaluation_status=status,
        source_category={"passed": "pass", "failed": "fail", "not_run": "init_fail"}[status],
    )


RESOLVED = ResolvedList(
    entry=ENTRY.key,
    resolved=frozenset({"django__django-1", "sympy__sympy-3"}),
    no_generation=frozenset(),
    reported=frozenset(POOL_IDS),
)
VERDICTS = {
    (ENTRY.swe_abs_agent_id, "django__django-1"): verdict("django__django-1", "passed", True),
    (ENTRY.swe_abs_agent_id, "django__django-2"): verdict("django__django-2", "not_run", None),
    (ENTRY.swe_abs_agent_id, "sympy__sympy-3"): verdict("sympy__sympy-3", "failed", True),
    (ENTRY.swe_abs_agent_id, "sympy__sympy-4"): verdict("sympy__sympy-4", "not_run", None),
}


def patch_entry(iid: str, body: bytes = b"diff --git a/x b/x\n") -> PatchIndexEntry:
    return PatchIndexEntry(
        instance_id=iid,
        sha256=sha256_bytes(body),
        size=len(body),
        path=f"p/{iid}",
        report_sha256="c" * 64,
        report_size=1,
        report_path=f"r/{iid}",
    )


PATCHES = {
    "django__django-1": patch_entry("django__django-1"),
    "django__django-2": patch_entry("django__django-2"),
    "sympy__sympy-3": patch_entry("sympy__sympy-3"),
    "sympy__sympy-4": patch_entry("sympy__sympy-4", b""),  # the agent gave up: an empty patch.diff
}


def report(iid: str, *, f2p: bool, p2p: bool = True) -> Report:
    body: dict[str, Any] = {
        "patch_is_None": False,
        "patch_exists": True,
        "patch_successfully_applied": True,
        "resolved": f2p and p2p,
        "tests_status": {
            "FAIL_TO_PASS": {"success": list(F2P) if f2p else [], "failure": [] if f2p else list(F2P)},
            "PASS_TO_PASS": {"success": list(P2P) if p2p else [], "failure": [] if p2p else list(P2P)},
        },
    }
    return parse_report(json.dumps({iid: body}), iid)


REPORTS = {
    "django__django-1": report("django__django-1", f2p=True),
    "django__django-2": report("django__django-2", f2p=False),
    "sympy__sympy-3": report("sympy__sympy-3", f2p=True),
    "sympy__sympy-4": report("sympy__sympy-4", f2p=False),
}


def records(**over: Any) -> dict[str, Record]:
    args: dict[str, Any] = {
        "resolved": RESOLVED,
        "verdicts": VERDICTS,
        "patches": PATCHES,
        "reports": REPORTS,
        "gold_runsc": None,
        "labels": None,
    } | over
    built = build_records(
        ENTRY,
        args["resolved"],
        POOL_IDS,
        args["verdicts"],
        args["patches"],
        args["reports"],
        ROWS,
        args["gold_runsc"],
        args["labels"],
    )
    return {r.instance_id: r for r in built}


def manifest(recs: list[Record] | None = None, pool_choice: str | None = None, **kw: Any) -> Manifest:
    rows = list(records().values()) if recs is None else recs
    return Manifest.model_validate(
        {
            "status": "draft",
            "created_at": "2026-10-27T00:00:00Z",
            "pool_choice": pool_choice,
            "blocked_on": [],
            "sources": [],
            "pools": [p.model_dump(mode="json") for p in pool_records(POOL)],
            "entries": [],
            "records": [r.model_dump(mode="json") for r in rows],
            **kw,
        }
    )


def test_published_reports_give_final_bad_and_empty_labels_with_no_assumption() -> None:
    recs = records()
    bad = recs["django__django-2"]
    assert (bad.provisional, bad.final, bad.assumed) == (Label.BAD, Label.BAD, ())
    assert (bad.fail_to_pass_passed, bad.pass_to_pass_passed) == (False, True)
    assert recs["sympy__sympy-4"].final is Label.EMPTY and recs["sympy__sympy-4"].needs == ()
    good = recs["django__django-1"]
    assert (good.provisional, good.final) == (Label.GOOD, Label.PENDING)
    assert good.needs == (Need.SWE_ABS_RUN, Need.GOLD_SWE_ABS, Need.GOLD_RUNSC)
    assert recs["sympy__sympy-3"].provisional is Label.PLAUSIBLE_WRONG


def test_a_patch_that_fails_only_pass_to_pass_is_regression_only_never_bad() -> None:
    recs = records(reports={**REPORTS, "django__django-2": report("django__django-2", f2p=True, p2p=False)})
    assert recs["django__django-2"].final is Label.REGRESSION_ONLY


def test_without_a_report_an_unresolved_patch_stays_pending_and_says_what_it_needs() -> None:
    reports = {k: v for k, v in REPORTS.items() if k != "django__django-2"}
    rec = records(reports=reports)["django__django-2"]
    assert (rec.provisional, rec.assumed, rec.final) == (Label.BAD, (ASSUME_UNRESOLVED_FAILS_F2P,), Label.PENDING)
    assert Need.FAIL_TO_PASS_DETAIL in rec.needs


def test_an_instance_the_results_have_no_logs_for_is_neither_resolved_nor_unresolved() -> None:
    no_logs = RESOLVED.model_copy(update={"no_logs": frozenset({"django__django-2"})})
    reports = {k: v for k, v in REPORTS.items() if k != "django__django-2"}
    rec = records(resolved=no_logs, reports=reports)["django__django-2"]
    assert rec.resolved is None and rec.provisional is Label.PENDING and rec.final is Label.PENDING
    assert Need.FAIL_TO_PASS_DETAIL in rec.needs


def test_plant_a_published_result_that_contradicts_swe_abs_is_refused() -> None:
    planted = dict(VERDICTS)
    planted[(ENTRY.swe_abs_agent_id, "django__django-2")] = verdict("django__django-2", "passed", True)
    with pytest.raises(ValueError, match="disagrees with the published result"):
        records(verdicts=planted)


def test_plant_a_report_that_contradicts_the_published_result_is_refused() -> None:
    with pytest.raises(ReportError, match="published results say False"):
        records(reports={**REPORTS, "django__django-2": report("django__django-2", f2p=True)})


def test_plant_a_report_without_its_patch_is_refused() -> None:
    with pytest.raises(ValueError, match="a report without its patch"):
        records(patches={k: v for k, v in PATCHES.items() if k != "django__django-2"})


def test_runsc_verdicts_reach_every_record_and_its_needs() -> None:
    """R24: a kept instance needs no runsc check, a dropped one is marked dropped, an infra one stays unchecked."""
    gold = {"django__django-1": True, "django__django-2": False, "sympy__sympy-3": None}
    recs = records(gold_runsc=gold)
    assert [recs[i].gold_runsc for i in POOL_IDS] == [True, False, None, None]
    assert Need.GOLD_RUNSC not in recs["django__django-1"].needs
    assert Need.GOLD_RUNSC not in recs["django__django-2"].needs
    assert Need.GOLD_RUNSC in recs["sympy__sympy-3"].needs


def test_blockers_name_the_pool_choice_patches_runsc_and_label_runs() -> None:
    recs = list(
        records(
            patches={k: v for k, v in PATCHES.items() if k != "sympy__sympy-3"},
            reports={k: v for k, v in REPORTS.items() if k != "sympy__sympy-3"},
        ).values()
    )
    found = blockers(recs, None)
    assert found[0].startswith("pool choice")
    assert any(b.startswith("patches: 1 not pulled") for b in found)
    assert any(b.startswith("gold runsc: 3 instances not yet checked") for b in found)  # the empty patch needs none
    assert any(b.startswith("labels: 2 resolved patches need our run of the SWE-ABS tests") for b in found)


def unresolved_only() -> list[Record]:
    """A pool where every label is already final (BAD, REGRESSION_ONLY, EMPTY), so only runsc stands in the way."""
    resolved = RESOLVED.model_copy(update={"resolved": frozenset()})
    verdicts = {k: verdict(k[1], "not_run", None) for k in VERDICTS}
    reports = {
        "django__django-1": report("django__django-1", f2p=True, p2p=False),
        "django__django-2": report("django__django-2", f2p=False),
        "sympy__sympy-3": report("sympy__sympy-3", f2p=False),
        "sympy__sympy-4": report("sympy__sympy-4", f2p=False),
    }
    gold = {"django__django-1": True, "django__django-2": True, "sympy__sympy-3": False}
    return list(records(resolved=resolved, verdicts=verdicts, reports=reports, gold_runsc=gold).values())


GOLD = GoldRunscRecord(
    sha256="d" * 64,
    inputs=(R24InputRecord(role="base", sha256="e" * 64, run_ids=("d1",), lines=4),),
    instances=4,
    kept=2,
    dropped=1,
    unchecked=0,
)
LABELS_RECORD = LabelRunRecord(
    sha256="f" * 64,
    run_ids=("d1",),
    final_results=6,
    gold_oracles=4,
    no_oracle=0,
    agree_with_published=0,
    disagree_with_published=0,
    rule_disagreements=0,
)


def test_freeze_refuses_until_the_pool_runsc_and_every_label_are_settled() -> None:
    with pytest.raises(NotFreezableError, match="no pool choice"):
        freeze(manifest())
    with pytest.raises(NotFreezableError, match="not available"):
        freeze(manifest(pool_choice="swe_abs_206"))
    with pytest.raises(NotFreezableError, match="gold runsc: 3 instances"):
        freeze(manifest(pool_choice="swe_abs_251"))
    checked = list(records(gold_runsc=dict.fromkeys(POOL_IDS, True)).values())
    with pytest.raises(NotFreezableError, match="labels: 2 resolved patches"):
        freeze(manifest(checked, pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump()))
    settled = unresolved_only()
    with pytest.raises(NotFreezableError, match="no R24 results are recorded"):
        freeze(manifest(settled, pool_choice="swe_abs_251"))
    frozen = freeze(manifest(settled, pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump()))
    assert frozen.status == "frozen" and frozen.blocked_on == ()


def test_plant_an_infra_verdict_keeps_the_manifest_from_freezing() -> None:
    settled = unresolved_only()
    infra = [
        r.model_copy(update={"gold_runsc": None, "needs": (Need.GOLD_RUNSC,)})
        if r.instance_id == "django__django-1"
        else r
        for r in settled
    ]
    with pytest.raises(NotFreezableError, match="gold runsc: 1 instances"):
        freeze(manifest(infra, pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump()))


def test_dump_round_trips_and_the_content_hash_ignores_only_the_timestamp() -> None:
    m = manifest()
    assert load(dump(m)) == m
    later = m.model_copy(update={"created_at": "2026-10-28T00:00:00Z"})
    assert content_sha256(later) == content_sha256(m)
    changed = m.model_copy(update={"records": m.records[1:]})
    assert content_sha256(changed) != content_sha256(m)
    block = dump(m).split('"records": [\n', 1)[1]
    assert block.count("\n") - 2 == len(m.records)  # one line per record, then the closing brackets


def test_the_206_pool_is_recorded_as_unavailable_with_its_reason() -> None:
    p251, p206 = pool_records(POOL)
    assert (
        p251.available
        and p251.size == 4
        and p251.list_sha256 == sha256_bytes("".join(f"{i}\n" for i in POOL_IDS).encode())
    )
    assert not p206.available and "53 overfitting instances are not named" in p206.why_unavailable


def test_plant_a_patch_that_changed_on_disk_is_refused(tmp_path: Path) -> None:
    body = b"diff --git a/x b/x\n"
    rel = f"patches/{ENTRY.key}/django__django-1/patch.diff"
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(body)
    index = PatchIndex(
        entry=ENTRY.key,
        store="s3-logs",
        patches=(PatchIndexEntry(instance_id="django__django-1", sha256=sha256_bytes(body), size=len(body), path=rel),),
    )
    patch_index_path(tmp_path, ENTRY.key).write_text(pretty_json(index.model_dump(mode="json")))
    assert set(load_patch_index(tmp_path, ENTRY.key)) == {"django__django-1"}
    (tmp_path / rel).write_bytes(body + b"+evil\n")
    with pytest.raises(ValueError, match="do not match"):
        load_patch_index(tmp_path, ENTRY.key)


def test_plant_a_swe_abs_error_verdict_never_stands_in_for_a_label() -> None:
    """SWE-ABS's own run erred on this patch: its verdict says nothing, so the resolved record stays unlabeled."""
    errored = dict(VERDICTS)
    errored[(ENTRY.swe_abs_agent_id, "django__django-1")] = Verdict(
        agent_id=ENTRY.swe_abs_agent_id,
        agent_name=ENTRY.key,
        instance_id="django__django-1",
        original_tests_passed=True,
        swe_abs_tests_passed=False,
        re_evaluation_status="error",
        source_category="error",
    )
    rec = records(verdicts=errored)["django__django-1"]
    assert (rec.provisional, rec.final, rec.assumed) == (Label.PENDING, Label.PENDING, ())
    assert Need.SWE_ABS_RUN in rec.needs


# ---------------------------------------------------------------- label runs (REG-564) and R24 drops


def labels(**records: RecordLabelEvidence) -> LabelEvidence:
    gold = dict.fromkeys(POOL_IDS, True)
    return LabelEvidence(gold=gold, records={f"{ENTRY.key}~{k}": v for k, v in records.items()})


def test_our_swe_abs_runs_make_good_and_plausible_wrong_final() -> None:
    ours = labels(
        **{
            "django__django-1": RecordLabelEvidence(swe_abs_passed=True, swe_abs_rule=True, patch_applies=True),
            "sympy__sympy-3": RecordLabelEvidence(swe_abs_passed=False, swe_abs_rule=False, patch_applies=True),
        }
    )
    recs = records(labels=ours, gold_runsc=dict.fromkeys(POOL_IDS, True))
    assert (recs["django__django-1"].final, recs["django__django-1"].needs) == (Label.GOOD, ())
    assert (recs["sympy__sympy-3"].final, recs["sympy__sympy-3"].needs) == (Label.PLAUSIBLE_WRONG, ())
    assert recs["django__django-1"].gold_swe_abs is True and recs["django__django-1"].swe_abs_rule is True


def test_plant_a_gold_that_fails_the_swe_abs_tests_labels_nothing() -> None:
    ours = labels(**{"django__django-1": RecordLabelEvidence(patch_applies=True)})
    ours = ours.model_copy(update={"gold": {**ours.gold, "django__django-1": False}})
    rec = records(labels=ours, gold_runsc=dict.fromkeys(POOL_IDS, True))["django__django-1"]
    assert (rec.final, rec.needs) == (Label.NO_ABS_ORACLE, ())


def test_plant_a_patch_that_did_not_apply_in_our_run_is_unusable() -> None:
    ours = labels(**{"django__django-1": RecordLabelEvidence(patch_applies=False)})
    rec = records(labels=ours, gold_runsc=dict.fromkeys(POOL_IDS, True))["django__django-1"]
    assert (rec.final, rec.needs) == (Label.UNUSABLE, ())


def test_our_fail_to_pass_run_labels_a_record_without_a_report() -> None:
    reports = {k: v for k, v in REPORTS.items() if k != "django__django-2"}
    ours = labels(
        **{
            "django__django-2": RecordLabelEvidence(
                fail_to_pass_passed=False, pass_to_pass_passed=True, patch_applies=True
            )
        }
    )
    rec = records(reports=reports, labels=ours, gold_runsc=dict.fromkeys(POOL_IDS, True))["django__django-2"]
    assert (rec.final, rec.f2p_source, rec.needs) == (Label.BAD, "run", ())


def test_plant_our_fail_to_pass_run_contradicting_the_published_report_is_refused() -> None:
    ours = labels(**{"django__django-2": RecordLabelEvidence(fail_to_pass_passed=True, pass_to_pass_passed=False)})
    with pytest.raises(ValueError, match="disagrees with the published report"):
        records(labels=ours)


def test_plant_records_on_an_instance_r24_dropped_never_block_the_freeze() -> None:
    """The re-review's note: freeze waited on label runs for instances R24 drops."""
    gold = {**dict.fromkeys(POOL_IDS, True), "django__django-1": False, "sympy__sympy-3": False}
    ours = labels()
    recs = list(records(gold_runsc=gold, labels=ours).values())
    dropped = [r for r in recs if r.gold_runsc is False]
    assert dropped and all(r.needs == () for r in dropped)
    assert not any(r.final is Label.PENDING and r.gold_runsc is not False for r in recs)
    assert blockers(recs, "swe_abs_251") == ()
    frozen = freeze(
        manifest(recs, pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump(), labels=LABELS_RECORD.model_dump())
    )
    assert frozen.status == "frozen"


def test_plant_a_manifest_with_resolved_records_and_no_label_record_does_not_freeze() -> None:
    ours = labels(
        **{
            "django__django-1": RecordLabelEvidence(swe_abs_passed=True, swe_abs_rule=True, patch_applies=True),
            "sympy__sympy-3": RecordLabelEvidence(swe_abs_passed=True, swe_abs_rule=True, patch_applies=True),
        }
    )
    recs = list(records(labels=ours, gold_runsc=dict.fromkeys(POOL_IDS, True)).values())
    with pytest.raises(NotFreezableError, match="no label-run results"):
        freeze(manifest(recs, pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump()))
    frozen = freeze(
        manifest(recs, pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump(), labels=LABELS_RECORD.model_dump())
    )
    assert frozen.status == "frozen"


def test_the_label_record_counts_agreement_with_swe_abs_published_verdicts() -> None:
    ours = labels(
        **{
            "django__django-1": RecordLabelEvidence(swe_abs_passed=False, swe_abs_rule=True, patch_applies=True),
            "sympy__sympy-3": RecordLabelEvidence(swe_abs_passed=False, swe_abs_rule=False, patch_applies=True),
        }
    )
    recs = list(records(labels=ours).values())
    counted = _agreement(LABELS_RECORD, recs)
    # django-1: SWE-ABS published passed, ours failed (and SWE-ABS's own rule on our run passed): a disagreement
    assert (counted.agree_with_published, counted.disagree_with_published, counted.rule_disagreements) == (1, 1, 1)


# ---------------------------------------------------------------- round 4: the founder's label drop


def test_plant_a_founder_label_drop_unblocks_the_freeze_and_drops_only_the_waiting_records() -> None:
    """Persistent INFRA's exit path (round-3 review): a label job INFRA on every run kept the manifest from freezing,
    forever. The founder's drop makes exactly the records whose label waited on it DROPPED, needing nothing; every
    other record keeps its label; and the manifest keeps the reason."""
    gold = dict.fromkeys(POOL_IDS, True)
    # django__django-1's resolved record waits on gold's SWE-ABS run, which was INFRA on every run
    ours = labels(
        **{"sympy__sympy-3": RecordLabelEvidence(swe_abs_passed=False, swe_abs_rule=False, patch_applies=True)}
    )
    ours = ours.model_copy(update={"gold": {k: v for k, v in ours.gold.items() if k != "django__django-1"}})
    waiting = records(labels=ours, gold_runsc=gold)
    assert waiting["django__django-1"].final is Label.PENDING and Need.GOLD_SWE_ABS in waiting["django__django-1"].needs
    with pytest.raises(NotFreezableError, match="labels: 1 instances need our run of the SWE-ABS tests on gold"):
        freeze(manifest(list(waiting.values()), pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump(),
                        labels=LABELS_RECORD.model_dump()))  # fmt: skip
    reason = "gVisor kills the sandbox at the 4g cap on every run (founder, 10/28)"
    key = f"{ENTRY.key}~django__django-1"
    recs = records(labels=ours.model_copy(update={"dropped": {key: reason}}), gold_runsc=gold)
    rec = recs["django__django-1"]
    assert (rec.final, rec.provisional, rec.needs) == (Label.DROPPED, Label.DROPPED, ())
    assert recs["sympy__sympy-3"].final is Label.PLAUSIBLE_WRONG and recs["django__django-2"].final is Label.BAD
    drop = LandedLabelDrop(instance_id="django__django-1", reason=reason, jobs=("swe_abs_gold~gold~django__django-1",),
                           records=(key,))  # fmt: skip
    record = LABELS_RECORD.model_copy(update={"drops": (drop,)})
    frozen = freeze(manifest(list(recs.values()), pool_choice="swe_abs_251", gold_runsc=GOLD.model_dump(),
                             labels=record.model_dump()))  # fmt: skip
    assert frozen.status == "frozen" and frozen.labels is not None and frozen.labels.drops == (drop,)
    assert load(dump(frozen)) == frozen  # the drop, its reason, jobs and records survive the manifest file
    assert Label.DROPPED not in SAMPLED


def test_plant_label_results_or_drops_for_records_the_manifest_lacks_are_refused() -> None:
    known = {f"{ENTRY.key}~django__django-1"}
    check_label_keys(LabelEvidence(gold={}, records={}, dropped=dict.fromkeys(known, "x")), known)
    with pytest.raises(ValueError, match="label drops for records the manifest does not have"):
        check_label_keys(LabelEvidence(gold={}, records={}, dropped={"stranger~django__django-1": "x"}), known)
    with pytest.raises(ValueError, match="label results for records the manifest does not have"):
        check_label_keys(LabelEvidence(gold={}, records={"stranger~django__django-1": RecordLabelEvidence()}), known)
