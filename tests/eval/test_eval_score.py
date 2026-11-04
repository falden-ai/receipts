"""Pilot scoring: a catch is a blocker whose repro passes on gold at scoring, counted per target for REG-561 with the
per-finding count beside it; everything else a scored replay reports is an invalid repro, per finding; a replay the
sandbox could not answer is unscored, on neither side, and the rule is not judged while one is; costs sum every entry;
compute hours read each finding's last state only; with no blockers there is no rate to judge, so the rule is not
met."""

from __future__ import annotations

from collections.abc import Mapping

import pytest

from receipts.contracts import (
    Arm,
    CatchKind,
    FailureType,
    Finding,
    LedgerEntry,
    Lens,
    Outcome,
    ReplayKey,
    Repro,
    RunnerKind,
    Side,
    StageCost,
    Status,
    Target,
    Tier,
)
from receipts.eval.hashing import canonical_json, sha256_text
from receipts.eval.labels import Label
from receipts.eval.pilot import RULE
from receipts.eval.score import (
    ArmScore,
    GoldReplay,
    ScoreError,
    apply_rule,
    fix_validity_candidate,
    last_states,
    percentiles,
    read_ledger,
    replay_key,
    repro_tests,
    score_arm,
)

from .conftest import run_result

SPARK = Arm(name="spark", attacker_model="muse-spark-1.3")
TARGETS = {
    f"e~django__django-{n}": Target(
        id=f"e~django__django-{n}",
        tier=Tier.A,
        repo="django/django",
        base_sha="a" * 40,
        patch="diff --git a/x b/x\n",
        test_runner=RunnerKind.DJANGO,
    )
    for n in range(4)
}
LABELS = {t: (Label.BAD, Label.BAD, Label.GOOD, Label.PLAUSIBLE_WRONG)[n] for n, t in enumerate(TARGETS)}


def repro(text: str = "assert f(2) == 4") -> Repro:
    return Repro(
        id="r1",
        attack_id="a1",
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        kind=CatchKind.FIX_VALIDITY,
        files={"tests/test_repro.py": f"def test_x():\n    {text}\n", "tests/conftest.py": ""},
        command=("python", "-m", "pytest", "tests/test_repro.py::test_x"),
    )


def entry(
    target: str,
    status: Status,
    *,
    cost: int = 0,
    seconds: float = 0.0,
    model: str | None = "m",
    attack: str = "a1",
    runs: tuple[object, ...] | None = None,
) -> LedgerEntry:
    kw: dict[str, object] = {}
    if status is Status.BLOCKER:
        kw = {
            "kind": CatchKind.FIX_VALIDITY,
            "repro": repro(f"{target} {attack}"),
            "runs": (
                *(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5)),
                run_result(Outcome.PASS, Side.SURROGATE),
            ),
            "quote": "It breaks.",
            "replay": ReplayKey(base_sha="a" * 40, patch_sha="b" * 40, harness_hash="h", seed=0, image_digest="d"),
        }
    elif status is Status.UNCHECKED:
        kw = {"unchecked_reason": "budget"}
    if runs is not None:
        kw["runs"] = runs
    finding = Finding(
        id=f"{target}/fixed_x/{attack}",
        target_id=target,
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        status=status,
        costs=(StageCost(stage="s", model=model, micro_usd=cost, seconds=seconds),),
        **kw,  # type: ignore[arg-type]
    )
    return LedgerEntry(at="2026-10-28T12:00:00Z", receipts_version="0.0.1", finding=finding)


def gold(outcome: Outcome | None, ftype: FailureType | None = None) -> GoldReplay:
    if ftype is None and outcome is Outcome.FAIL:
        ftype = FailureType.ASSERTION
    return GoldReplay(outcome=outcome, failure_type=ftype)


def replay_by(outcomes: Mapping[str, GoldReplay | Outcome]) -> object:
    def replay(target: Target, r: Repro) -> GoldReplay:
        o = outcomes.get(target.id, Outcome.PASS)
        return o if isinstance(o, GoldReplay) else gold(o)

    return replay


def test_a_catch_needs_gold_to_pass_and_a_good_target_catch_is_flagged_disputed() -> None:
    ids = list(TARGETS)
    entries = [entry(t, Status.BLOCKER, cost=100) for t in ids]
    s = score_arm(SPARK, entries, TARGETS, LABELS, replay_by({ids[1]: Outcome.FAIL}))  # type: ignore[arg-type]
    assert (s.blockers, s.catches, s.invalid_repros, s.invalid_repro_pct) == (4, 3, 1, 25.0)
    assert s.catches_by_label == {"BAD": 1, "GOOD": 1, "PLAUSIBLE_WRONG": 1} and s.good_label_disputed == 1
    assert s.invalid_on_gold == {"fail": 1}
    assert s.cost_micro_usd == 400 and s.cost_per_target_micro_usd.p50 == 100


@pytest.mark.parametrize(
    ("on_gold", "kind"),
    [
        (gold(Outcome.FAIL), "fail"),
        (gold(Outcome.ERROR, FailureType.IMPORT), "error:import"),
        (gold(Outcome.ERROR, FailureType.CRASH_ELSEWHERE), "error:crash_elsewhere"),
        (gold(Outcome.SKIPPED), "skipped"),
        (gold(Outcome.TIMEOUT, FailureType.TIMEOUT), "timeout"),
    ],
)
def test_plant_a_repro_that_does_anything_but_pass_on_gold_is_invalid(on_gold: GoldReplay, kind: str) -> None:
    t = next(iter(TARGETS))
    s = score_arm(SPARK, [entry(t, Status.BLOCKER)], TARGETS, LABELS, replay_by({t: on_gold}))  # type: ignore[arg-type]
    assert (s.catches, s.invalid_repros, s.invalid_repro_pct, s.unscored) == (0, 1, 100.0, ())
    assert s.invalid_on_gold == {kind: 1}


@pytest.mark.parametrize("on_gold", [gold(None), gold(Outcome.ERROR), gold(Outcome.ERROR, FailureType.INFRA)])
def test_plant_a_replay_the_sandbox_could_not_answer_is_unscored_never_invalid(on_gold: GoldReplay) -> None:
    """Blocker 4: ERROR with failure type INFRA or none says the sandbox could not answer, nothing about the repro on
    gold. It is on neither side, and the rate is over the scored replays only."""
    a, b = list(TARGETS)[:2]
    entries = [entry(a, Status.BLOCKER), entry(b, Status.BLOCKER)]
    s = score_arm(SPARK, entries, TARGETS, LABELS, replay_by({a: on_gold, b: Outcome.PASS}))  # type: ignore[arg-type]
    assert (s.catches, s.invalid_repros, s.invalid_repro_pct) == (1, 0, 0.0)
    assert s.unscored == (f"{a}/fixed_x/a1",) and s.invalid_on_gold == {}


def test_only_a_findings_last_state_counts_and_costs_sum_every_entry() -> None:
    t = next(iter(TARGETS))
    earlier_runs = tuple(run_result(Outcome.FAIL, Side.PATCH, duration_s=900.0) for _ in range(4))
    first = entry(t, Status.FLAKY, cost=50, seconds=7200, model=None, runs=earlier_runs)
    then = entry(t, Status.BLOCKER, cost=70, seconds=10)
    assert last_states([first, then])[then.finding.id].status is Status.BLOCKER
    s = score_arm(SPARK, [first, then], TARGETS, LABELS, replay_by({}))  # type: ignore[arg-type]
    assert (s.findings, s.blockers, s.cost_micro_usd) == (1, 1, 120)
    assert s.code_stage_hours == 2.0 and s.model_hours == round(10 / 3600, 3)
    # SC5: a finding's runs are a state, restated on each entry; the hours read the last state's six runs (2 s each),
    # never the earlier entry's four 900 s runs as well (that would be 3612 s, an hour more)
    assert s.sandbox_hours == round(6 * 2.0 / 3600, 3)
    assert s.compute_hours == round((12 + 7200) / 3600, 3)


def test_plant_a_finding_for_a_target_off_the_pilot_is_refused() -> None:
    with pytest.raises(ScoreError, match="not a pilot target"):
        score_arm(SPARK, [entry("e~sympy__sympy-1", Status.BLOCKER)], TARGETS, LABELS, replay_by({}))  # type: ignore[arg-type]


def arm(name: str, blockers: int, catches: int, unscored: tuple[str, ...] = ()) -> ArmScore:
    return ArmScore(
        arm=name,
        attacker_model="m",
        targets=20,
        targets_with_findings=0,
        findings=blockers,
        statuses={},
        blockers=blockers,
        catches=catches,
        catches_by_label={},
        catch_findings=catches,
        catch_findings_by_label={},
        good_label_disputed=0,
        unscored=unscored,
        invalid_repros=blockers - catches,
        invalid_repro_pct=None if not blockers else round(100 * (blockers - catches) / blockers, 1),
        invalid_on_gold={},
        cost_micro_usd=0,
        cost_per_target_micro_usd=percentiles([]),
        sandbox_hours=0,
        code_stage_hours=0,
        model_hours=0,
        compute_hours=0,
        per_target=(),
    )


@pytest.mark.parametrize(
    ("spark", "glimmer", "met", "why"),
    [
        ((10, 8), (10, 8), True, "caught 8 targets against spark's 8 (margin 1) with 20.0% invalid repros"),
        ((10, 8), (9, 7), True, "caught 7 targets against spark's 8 (margin 1) with 22.2% invalid repros"),
        ((10, 8), (10, 6), False, "caught 6 targets, under spark's 8 minus 1"),
        ((4, 1), (10, 6), False, "invalid-repro rate is 40.0%, not under 30%"),
        ((4, 1), (0, 0), False, "reported no blockers"),
    ],
)
def test_reg_561s_rule_as_written(spark: tuple[int, int], glimmer: tuple[int, int], met: bool, why: str) -> None:
    out = apply_rule(RULE, {"spark": arm("spark", *spark), "glimmer": arm("glimmer", *glimmer)})
    assert out.met is met and why in out.why, out.why


def test_the_rule_is_not_met_at_exactly_30_percent() -> None:
    out = apply_rule(RULE, {"spark": arm("spark", 10, 7), "glimmer": arm("glimmer", 10, 7)})
    assert out.candidate_invalid_repro_pct == 30.0 and out.met is False


def test_plant_the_rule_counts_caught_targets_not_caught_findings() -> None:
    """The integrator's call on REG-561: a target counts once when any of its blockers fails on the patch and passes
    on gold. Glimmer's three caught findings on one target are one catch against Spark's three targets, so the rule is
    not met; counted per finding (3 against 3, margin 1) it would be."""
    ids = list(TARGETS)
    spark = [entry(t, Status.BLOCKER) for t in ids[:3]]
    glimmer = [entry(ids[0], Status.BLOCKER, attack=a) for a in ("a1", "a2", "a3")]
    s = score_arm(SPARK, spark, TARGETS, LABELS, replay_by({}))  # type: ignore[arg-type]
    g = score_arm(Arm(name="glimmer", attacker_model="muse-glimmer-30b"), glimmer, TARGETS, LABELS, replay_by({}))  # type: ignore[arg-type]
    assert (s.catches, s.catch_findings) == (3, 3) and (g.catches, g.catch_findings) == (1, 3)
    assert g.catches_by_label == {"BAD": 1} and g.catch_findings_by_label == {"BAD": 3}
    out = apply_rule(RULE, {"spark": s, "glimmer": g})
    assert out.met is False and "caught 1 targets, under spark's 3 minus 1" in out.why
    assert (out.candidate_catch_findings, out.baseline_catch_findings) == (3, 3)
    assert g.invalid_repro_pct == 0.0  # the rate stays per replayed finding


def test_plant_a_rule_with_unscored_replays_is_not_judged() -> None:
    out = apply_rule(RULE, {"spark": arm("spark", 10, 8), "glimmer": arm("glimmer", 10, 9, unscored=("x/y/z",))})
    assert (out.judged, out.met) == (False, False) and "not judged: 1 blocker replays" in out.why


def test_a_ledger_is_read_strictly() -> None:
    t = next(iter(TARGETS))
    good = entry(t, Status.SUSPECTED).model_dump_json() + "\n"
    assert len(read_ledger(good * 2, "x")) == 2
    with pytest.raises(ScoreError, match="torn tail"):
        read_ledger(good + good.rstrip("\n"), "x")
    with pytest.raises(ScoreError, match="line 2"):
        read_ledger(good + '{"v": 1}\n', "x")


def test_the_repro_runs_its_own_test_file_and_replays_once_per_target() -> None:
    assert repro_tests(repro()) == ["tests/test_repro.py"]  # named by its command; conftest.py is a helper
    no_command = repro().model_copy(update={"command": ("make", "check")})
    assert repro_tests(no_command) == ["tests/test_repro.py"]
    assert replay_key("t1", repro()) == replay_key("t1", repro()) != replay_key("t2", repro())


def test_percentiles_are_nearest_rank() -> None:
    p = percentiles([5, 1, 3, 2, 4])
    assert (p.mean, p.p50, p.p95, p.max) == (3, 3, 5, 5)


# ---------------------------------------------------------------- round 4: the founder's replay drop


def test_plant_a_founder_replay_drop_takes_only_an_unscored_replay_off_the_books() -> None:
    """Persistent INFRA's exit path for a gold replay (round-3 review): a replay infra on every run left REG-561 'not
    judged' forever. A founder drop by finding id counts that finding on neither side, with its reason; a drop on a
    replay that scored changes nothing here (score() refuses it); and without a drop the replay stays unscored."""
    a, b = list(TARGETS)[:2]
    entries = [entry(a, Status.BLOCKER), entry(b, Status.BLOCKER)]
    stuck, fine = f"{a}/fixed_x/a1", f"{b}/fixed_x/a1"
    replay = replay_by({a: gold(Outcome.ERROR, FailureType.INFRA), b: Outcome.PASS})
    reason = "gVisor kills the replay at the 4g cap (founder, 10/28)"
    s = score_arm(SPARK, entries, TARGETS, LABELS, replay, {stuck: reason, fine: "no"})  # type: ignore[arg-type]
    assert (s.unscored, s.dropped, s.catches, s.invalid_repros, s.invalid_repro_pct) == ((), {stuck: reason}, 1, 0, 0.0)
    kept = score_arm(SPARK, entries, TARGETS, LABELS, replay)  # type: ignore[arg-type]
    assert (kept.unscored, kept.dropped) == ((stuck,), {})
    out = apply_rule(RULE, {"spark": arm("spark", 10, 8), "glimmer": s})
    assert out.judged and out.founder_dropped == 1 and "the founder dropped (score --replay-drop)" in out.why
    assert not apply_rule(RULE, {"spark": arm("spark", 10, 8), "glimmer": kept}).judged


def test_a_repros_own_cases_are_in_its_replay_key() -> None:
    """Contract v3: Repro.cases are what the repro runs (the replay's expect), so a repro with other cases replays on
    its own; one with none keys exactly as round 3 keyed it."""
    plain = repro()
    old = sha256_text(canonical_json({"target": "t1", "files": plain.files, "command": list(plain.command)}))
    assert replay_key("t1", plain) == old
    one = plain.model_copy(update={"cases": ("tests/test_repro.py::test_x",)})
    other = plain.model_copy(update={"cases": ("tests/test_repro.py::test_y",)})
    assert len({replay_key("t1", plain), replay_key("t1", one), replay_key("t1", other)}) == 3


# ---------------------------------------------------------------- round 5: a dropped replay is out of the rate

GLIMMER = Arm(name="glimmer", attacker_model="muse-glimmer-30b")


def test_plant_a_dropped_replay_is_left_out_of_the_invalid_repro_rate() -> None:
    """The round-4 review's blocker (score.py:253): a replay the founder dropped stayed in the rate's denominator, so
    every drop lowered the rate and could flip REG-561 to met. Glimmer's four blockers: two pass on gold, one fails
    there, one is dropped. The pre-registered rate is 1 invalid over the 3 scored, 33.3%, not under 30%: not met. The
    old count gave 1 over 4, 25.0%, and met."""
    ids = list(TARGETS)
    entries = [entry(t, Status.BLOCKER) for t in ids]
    stuck = f"{ids[3]}/fixed_x/a1"
    infra = gold(Outcome.ERROR, FailureType.INFRA)
    replay = replay_by({ids[0]: Outcome.PASS, ids[1]: Outcome.PASS, ids[2]: Outcome.FAIL, ids[3]: infra})
    reason = "gVisor kills the replay at the 4g cap (founder, 10/28)"
    g = score_arm(GLIMMER, entries, TARGETS, LABELS, replay, {stuck: reason})  # type: ignore[arg-type]
    assert (g.blockers, g.catches, g.invalid_repros, g.unscored, g.dropped) == (4, 2, 1, (), {stuck: reason})
    assert g.invalid_repro_pct == 33.3
    out = apply_rule(RULE, {"spark": arm("spark", 3, 2), "glimmer": g})
    assert out.judged and out.met is False and out.candidate_invalid_repro_pct == 33.3
    assert "glimmer's invalid-repro rate is 33.3%, not under 30%" in out.why and out.founder_dropped == 1


def test_plant_an_arm_whose_every_blocker_replay_was_dropped_has_no_rate_and_is_not_met() -> None:
    """With every blocker replay dropped, none was scored: there is no rate, so the rule is not met, and its reason says
    why. The old count gave 0.0% and met (0 caught targets against Spark's 1 minus 1)."""
    t = next(iter(TARGETS))
    stuck = f"{t}/fixed_x/a1"
    g = score_arm(GLIMMER, [entry(t, Status.BLOCKER)], TARGETS, LABELS, replay_by({t: gold(None)}), {stuck: "x"})  # type: ignore[arg-type]
    assert (g.blockers, g.catches, g.invalid_repros, g.unscored, g.dropped) == (1, 0, 0, (), {stuck: "x"})
    assert g.invalid_repro_pct is None
    out = apply_rule(RULE, {"spark": arm("spark", 1, 1), "glimmer": g})
    assert out.judged and out.met is False and out.candidate_invalid_repro_pct is None
    assert "glimmer reported 1 blockers, but the founder dropped the replay of every one" in out.why
    assert "no blocker replay was scored" in out.why and "score --replay-drop" in out.why
    none = apply_rule(RULE, {"spark": arm("spark", 1, 1), "glimmer": arm("glimmer", 0, 0)})
    assert none.met is False and "reported no blockers" in none.why  # an arm with no blockers keeps its own reason


# ---------------------------------------------------------------- round 6: the round-5 review's surviving plants


def test_plant_an_arm_with_an_unscored_replay_is_not_judged_even_beside_a_founder_drop() -> None:
    """The round-5 review's V3: a founder drop is the way out for one replay, never a pass for the others. Glimmer's
    three blockers: one the founder dropped, one still unscored infra, one caught. The rule is not judged and not met
    while the unscored one waits; judged here, Glimmer's 1 catch against Spark's 2 minus 1 at 0.0% would read met."""
    a, b, c = list(TARGETS)[:3]
    entries = [entry(a, Status.BLOCKER), entry(b, Status.BLOCKER), entry(c, Status.BLOCKER)]
    infra = gold(Outcome.ERROR, FailureType.INFRA)
    reason = "gVisor kills the replay at the 4g cap (founder, 10/28)"
    replay = replay_by({a: infra, b: infra, c: Outcome.PASS})
    g = score_arm(GLIMMER, entries, TARGETS, LABELS, replay, {f"{a}/fixed_x/a1": reason})  # type: ignore[arg-type]
    assert (g.dropped, g.unscored, g.catches) == ({f"{a}/fixed_x/a1": reason}, (f"{b}/fixed_x/a1",), 1)
    out = apply_rule(RULE, {"spark": arm("spark", 3, 2), "glimmer": g})
    assert (out.judged, out.met, out.founder_dropped) == (False, False, 1)
    assert out.why.startswith("not judged: 1 blocker replays on gold hit infra twice; rerun score"), out.why


def test_plant_an_unscored_replay_is_left_out_of_the_rate_while_the_rule_waits() -> None:
    """The round-5 review's V4: the rate is invalid repros over the replays that scored, so an unscored one is in
    neither count. One invalid of the 2 scored is 50.0%; with the unscored one in the denominator it would read 33.3%,
    a rate score.json would report while the rule is not judged."""
    a, b, c = list(TARGETS)[:3]
    entries = [entry(t, Status.BLOCKER) for t in (a, b, c)]
    replay = replay_by({a: gold(Outcome.ERROR, FailureType.INFRA), b: Outcome.FAIL, c: Outcome.PASS})
    g = score_arm(GLIMMER, entries, TARGETS, LABELS, replay)  # type: ignore[arg-type]
    assert (g.blockers, g.catches, g.invalid_repros, g.unscored) == (3, 1, 1, (f"{a}/fixed_x/a1",))
    assert g.invalid_repro_pct == 50.0
    out = apply_rule(RULE, {"spark": arm("spark", 3, 2), "glimmer": g})
    assert out.judged is False and out.candidate_invalid_repro_pct == 50.0


# ------------------------------------------------------------ REG-577: fix-validity catches on gold, a second number


def fix_validity(target: str, attack: str = "a1", *, base_passes: int = 0) -> LedgerEntry:
    """What the skeptic writes for a fix-validity repro: s_sat_inconclusive after 5 counted failures on the patch and
    5 base runs, `base_passes` of them passing."""
    runs = (
        *(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5)),
        *(run_result(Outcome.PASS if n < base_passes else Outcome.FAIL, Side.BASE) for n in range(5)),
    )
    finding = Finding(
        id=f"{target}/fixed_x/{attack}",
        target_id=target,
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        status=Status.S_SAT_INCONCLUSIVE,
        kind=CatchKind.FIX_VALIDITY,
        repro=repro(f"{target} {attack} fv"),
        runs=runs,
        detail="SK1: fails on the patch in all 5 runs and passes on base in 0 of 5",
    )
    return LedgerEntry(at="2026-10-29T12:00:00Z", receipts_version="0.0.1", finding=finding)


def test_plant_fix_validity_candidates_are_replayed_on_gold_and_scored_beside_the_blockers() -> None:
    """REG-577: a fix-validity candidate that passes on gold is a valid catch, counted per target by label; one that
    fails there is an invalid repro; one infra twice is unscored. Base passing in any run makes it no candidate. The
    blocker numbers and REG-561's rule read blockers only and do not move."""
    ids = list(TARGETS)
    entries = [
        entry(ids[0], Status.BLOCKER),  # REG-561's: passes on gold
        fix_validity(ids[0], "f1"),  # passes on gold: a catch on a BAD target
        fix_validity(ids[0], "f2"),  # the same target again: one target, two findings
        fix_validity(ids[1], "f1"),  # fails on gold: invalid
        fix_validity(ids[2], "f1"),  # infra twice: unscored
        fix_validity(ids[3], "f1", base_passes=2),  # passes on base sometimes: not a candidate
    ]
    seen: list[str] = []

    def replay(target: Target, r: Repro) -> GoldReplay:
        seen.append(target.id)
        if target.id == ids[1]:
            return gold(Outcome.FAIL)
        if target.id == ids[2]:
            return gold(Outcome.ERROR, FailureType.INFRA)
        return gold(Outcome.PASS)

    s = score_arm(SPARK, entries, TARGETS, LABELS, replay)
    fv = s.fix_validity
    assert (fv.candidates, fv.catch_findings, fv.catches, fv.invalid_repros) == (4, 2, 1, 1)
    assert fv.catches_by_label == {"BAD": 1} and fv.catch_findings_by_label == {"BAD": 2}
    assert fv.invalid_on_gold == {"fail": 1} and fv.unscored == (f"{ids[2]}/fixed_x/f1",)
    assert ids[3] not in seen and len(seen) == 5  # one blocker replay and four candidate replays
    # REG-561's numbers are the blocker's alone: an unscored candidate never holds the rule back
    assert (s.blockers, s.catches, s.invalid_repros, s.invalid_repro_pct, s.unscored) == (1, 1, 0, 0.0, ())
    out = apply_rule(RULE, {"spark": s, "glimmer": s})
    assert out.judged and out.met


def test_without_candidates_the_fix_validity_number_is_zero_and_the_old_score_reads_as_before() -> None:
    t = next(iter(TARGETS))
    s = score_arm(SPARK, [entry(t, Status.S_SAT_INCONCLUSIVE)], TARGETS, LABELS, replay_by({}))  # type: ignore[arg-type]
    assert s.fix_validity.candidates == 0 and s.statuses == {"s_sat_inconclusive": 1}
    assert ArmScore.model_validate(arm("x", 0, 0).model_dump()).fix_validity.candidates == 0


# ---------------------------------------------------------------- round 3: the round-2 review's notes on REG-577

PATCH_FAILS = tuple(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5))
BASE_FAILS = tuple(run_result(Outcome.FAIL, Side.BASE) for _ in range(5))
IMPORT_FAIL = run_result(Outcome.FAIL, Side.PATCH, failure_type=FailureType.IMPORT)  # not a counted failure
BASE_IMPORT_FAIL = run_result(Outcome.FAIL, Side.BASE, failure_type=FailureType.IMPORT)
BASE_IMPORT_ERROR = run_result(Outcome.ERROR, Side.BASE, failure_type=FailureType.IMPORT)


def of_kind(kind: CatchKind) -> dict[str, object]:
    return {"kind": kind, "repro": repro("x").model_copy(update={"kind": kind})}


@pytest.mark.parametrize(
    "update",
    [
        pytest.param({"runs": (*PATCH_FAILS[:4], run_result(Outcome.PASS, Side.PATCH), *BASE_FAILS)}, id="patch-pass"),
        pytest.param({"runs": (*PATCH_FAILS[:4], IMPORT_FAIL, *BASE_FAILS)}, id="patch-fail-not-counted"),
        pytest.param({"runs": (*PATCH_FAILS[:4], *BASE_FAILS)}, id="4-patch-runs"),
        pytest.param({"runs": (*PATCH_FAILS, *BASE_FAILS[:4])}, id="4-base-runs"),
        pytest.param({"runs": (*PATCH_FAILS, *BASE_FAILS[:4], run_result(Outcome.PASS, Side.BASE))}, id="base-pass"),
        # Fix round 5: pilot, pylint-8898 Glimmer 1.1 imported a function only the agent's patch adds, so it failed on
        # base by an import error, which is no bug of base's.
        pytest.param({"runs": (*PATCH_FAILS, *BASE_FAILS[:4], BASE_IMPORT_ERROR)}, id="base-import-error"),
        pytest.param({"runs": (*PATCH_FAILS, *BASE_FAILS[:4], BASE_IMPORT_FAIL)}, id="base-import-fail"),
        pytest.param(of_kind(CatchKind.BEHAVIOR), id="kind-behavior"),
        pytest.param(of_kind(CatchKind.REGRESSION), id="kind-regression"),
        pytest.param({"status": Status.FLAKY}, id="status-flaky"),
        pytest.param({"repro": None}, id="no-repro"),
    ],
)
def test_plant_anything_but_an_unsettled_fix_validity_catch_is_no_candidate(update: dict[str, object]) -> None:
    """The round-2 review's note 4: fix_validity_candidate's patch-side and kind gates had no test, so all( to any(
    on the patch failures, and a kind check that let BEHAVIOR through, both survived. Each plant changes one thing
    about a candidate (fails on the patch on a counted failure in all of at least 5 runs, passes on base in none of at
    least 5 and fails there with no import error, kind fix_validity, status s_sat_inconclusive, a repro), and what is
    left is no candidate: never replayed on gold and never counted."""
    t = next(iter(TARGETS))
    candidate = fix_validity(t).finding
    assert fix_validity_candidate(candidate)
    planted = Finding.model_validate({**candidate.model_dump(), **update})
    assert not fix_validity_candidate(planted)
    seen: list[str] = []

    def replay(target: Target, r: Repro) -> GoldReplay:
        seen.append(target.id)
        return gold(Outcome.PASS)

    e = LedgerEntry(at="2026-10-29T12:00:00Z", receipts_version="0.0.1", finding=planted)
    fv = score_arm(SPARK, [e], TARGETS, LABELS, replay).fix_validity
    assert (fv.candidates, fv.catches, fv.catch_findings, fv.invalid_repros, seen) == (0, 0, 0, 0, [])


def test_plant_a_fix_validity_catch_on_a_good_target_is_flagged_good_label_disputed() -> None:
    """The round-2 review's note 3: a fix-validity catch on a GOOD target showed only in its catches_by_label. It is
    flagged as a blocker's catch is (spec section 3, adjudication): the GOOD targets with a catch, each counted once.
    A GOOD target whose candidates fail on gold is no catch and is not flagged, and the blockers' flag does not move."""
    ids = list(TARGETS)  # BAD, BAD, GOOD, PLAUSIBLE_WRONG
    entries = [
        fix_validity(ids[2], "f1"),
        fix_validity(ids[2], "f2"),  # two catches on one GOOD target: flagged once
        fix_validity(ids[0], "f1"),  # a catch on a BAD target: not flagged
        entry(ids[1], Status.BLOCKER),  # REG-561's, on a BAD target
    ]
    s = score_arm(SPARK, entries, TARGETS, LABELS, replay_by({}))  # type: ignore[arg-type]
    fv = s.fix_validity
    assert (fv.catches, fv.catch_findings, fv.good_label_disputed) == (2, 3, 1)
    assert fv.catches_by_label == {"BAD": 1, "GOOD": 1} and fv.catch_findings_by_label == {"BAD": 1, "GOOD": 2}
    assert (s.catches, s.good_label_disputed) == (1, 0)  # the blockers' own flag reads blockers only
    failed = score_arm(SPARK, entries, TARGETS, LABELS, replay_by({ids[2]: Outcome.FAIL})).fix_validity  # type: ignore[arg-type]
    assert (failed.catches, failed.good_label_disputed, failed.invalid_repros) == (1, 0, 2)


def test_plant_an_unscored_fix_validity_replay_waits_until_the_founder_drops_it() -> None:
    """An unscored candidate replay is handled as a blocker's is: it is listed until it scores, and the founder alone
    may drop it by finding id, which counts it on neither side of the fix-validity number. A drop on a candidate
    whose replay scored changes nothing here (score() refuses it). Neither moves REG-561's rule: an unscored
    candidate never holds it back, and its founder_dropped counts the blockers' drops only."""
    a, b, c = list(TARGETS)[:3]
    entries = [fix_validity(a, "f1"), fix_validity(b, "f1"), entry(c, Status.BLOCKER)]
    stuck, fine = f"{a}/fixed_x/f1", f"{b}/fixed_x/f1"
    replay = replay_by({a: gold(Outcome.ERROR, FailureType.INFRA)})
    reason = "gVisor kills the replay at the 4g cap (founder, 10/29)"
    s = score_arm(SPARK, entries, TARGETS, LABELS, replay, {stuck: reason, fine: "no"})  # type: ignore[arg-type]
    fv = s.fix_validity
    assert (fv.unscored, fv.dropped, fv.catches, fv.invalid_repros) == ((), {stuck: reason}, 1, 0)
    assert (s.unscored, s.dropped, s.catches) == ((), {}, 1)
    kept = score_arm(SPARK, entries, TARGETS, LABELS, replay)  # type: ignore[arg-type]
    waiting = kept.fix_validity
    assert (waiting.unscored, waiting.dropped, waiting.invalid_repros) == ((stuck,), {}, 0)
    for arms in ({"spark": s, "glimmer": s}, {"spark": kept, "glimmer": kept}):
        out = apply_rule(RULE, arms)
        assert (out.judged, out.met, out.founder_dropped) == (True, True, 0), out.why


@pytest.mark.parametrize(
    "base_run",
    [
        pytest.param(
            run_result(Outcome.ERROR, Side.BASE, failure_type=FailureType.CRASH_ELSEWHERE), id="crash-elsewhere"
        ),
        pytest.param(run_result(Outcome.ERROR, Side.BASE, failure_type=FailureType.NAME), id="name"),
        pytest.param(run_result(Outcome.SKIPPED, Side.BASE), id="skipped"),
    ],
)
def test_only_an_import_error_on_base_keeps_a_fix_validity_catch_from_being_a_candidate(base_run: object) -> None:
    candidate = fix_validity(next(iter(TARGETS))).finding
    other = Finding.model_validate({**candidate.model_dump(), "runs": (*PATCH_FAILS, *BASE_FAILS[:4], base_run)})
    assert fix_validity_candidate(other)
