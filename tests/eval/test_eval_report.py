"""EV3, the eval report: Wilson intervals and McNemar's exact test match their closed forms and the hand values, PPV
reads sensitivity and specificity the right way round, the bootstrap is seeded and resamples targets, never findings,
and a hand-built 2-arm, 8-target pilot gives the known WDR, false-alarm rate, precision, MCC, cost per catch and
McNemar counts, with a disputed GOOD target and a pending one; score /3's per-target rows sum to its totals, both
files round-trip, and report.md shows every arm and every rate with its n."""

from __future__ import annotations

import json
import math
import random
from collections.abc import Callable, Mapping
from itertools import product
from pathlib import Path
from typing import Any, Final

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
from receipts.eval import cli
from receipts.eval.hashing import pretty_json
from receipts.eval.labels import Label
from receipts.eval.pilot import RULE
from receipts.eval.report import (
    PREVALENCES,
    ArmRates,
    ArmReport,
    Bounds,
    Rate,
    Report,
    ReportError,
    _usd,
    arm_rates,
    arm_report,
    bootstrap,
    check_rows,
    compare,
    efficiency,
    mcnemar_exact,
    ppv,
    read_score,
    render_markdown,
    report,
    wilson,
)
from receipts.eval.score import ArmScore, GoldReplay, Score, TargetOutcome, apply_rule, score_arm

from .conftest import run_result

Z = 1.959963984540054
TARGETS = {
    f"e~django__django-{n}": Target(
        id=f"e~django__django-{n}",
        tier=Tier.A,
        repo="django/django",
        base_sha="a" * 40,
        patch="diff --git a/x b/x\n",
        test_runner=RunnerKind.DJANGO,
    )
    for n in range(8)
}
T = list(TARGETS)
B, PW, G = Label.BAD, Label.PLAUSIBLE_WRONG, Label.GOOD
LABELS = dict(zip(T, (B, B, B, PW, G, G, G, B), strict=True))
ATTACK = "attack.correctness.write"
SPEND = StageCost(stage=ATTACK, model="muse-spark-1.3", micro_usd=120_000, seconds=10.0)


# ---------------------------------------------------------------- statistics against their closed forms


def _bisect(f: Callable[[float], float], inside: float, outside: float) -> float:
    """The root of f between `inside` (f <= 0) and `outside` (f > 0), by bisection: no closed form used."""
    for _ in range(200):
        mid = (inside + outside) / 2
        if f(mid) > 0:
            outside = mid
        else:
            inside = mid
    return (inside + outside) / 2


def wilson_by_root(k: int, n: int) -> tuple[float, float]:
    """The Wilson interval from its definition: the p where |k/n - p| = z * sqrt(p(1 - p)/n), found numerically."""
    phat = k / n

    def f(p: float) -> float:
        return (phat - p) ** 2 - Z * Z * p * (1 - p) / n

    # f is at most 0 at phat and positive at 0 (when k > 0) and at 1 (when k < n): one root on each side
    low = 0.0 if k == 0 else _bisect(f, phat, 0.0)
    high = 1.0 if k == n else _bisect(f, phat, 1.0)
    return low, high


@pytest.mark.parametrize(
    ("k", "n", "low", "high"),
    [(0, 10, 0.0, 0.2775), (5, 10, 0.2366, 0.7634), (10, 10, 0.7225, 1.0), (1, 3, 0.0615, 0.7923)],
)
def test_plant_wilson_matches_its_definition_and_the_hand_values(k: int, n: int, low: float, high: float) -> None:
    got = wilson(k, n)
    assert got is not None
    assert (round(got[0], 4), round(got[1], 4)) == (low, high)
    ref = wilson_by_root(k, n)
    assert got[0] == pytest.approx(ref[0], abs=1e-9) and got[1] == pytest.approx(ref[1], abs=1e-9)


def test_wilson_with_no_n_is_none_and_refuses_an_impossible_count() -> None:
    assert wilson(0, 0) is None
    with pytest.raises(ValueError):
        wilson(4, 3)


def binomial_two_sided(b: int, c: int) -> float:
    """The exact test from the binomial's mass: twice the smaller tail of Binomial(b + c, 1/2), capped at 1."""
    n = b + c
    if n == 0:
        return 1.0
    mass = [math.comb(n, i) / 2**n for i in range(n + 1)]
    return min(1.0, 2 * min(sum(mass[: min(b, c) + 1]), sum(mass[max(b, c) :])))


@pytest.mark.parametrize(("b", "c", "p"), [(0, 5, 0.0625), (5, 0, 0.0625), (3, 3, 1.0), (0, 0, 1.0), (1, 9, 0.0215)])
def test_plant_mcnemar_exact_matches_the_hand_cases(b: int, c: int, p: float) -> None:
    assert round(mcnemar_exact(b, c), 4) == p
    assert mcnemar_exact(b, c) == pytest.approx(binomial_two_sided(b, c), abs=1e-12)


def test_plant_ppv_reads_sensitivity_and_specificity_the_right_way_round() -> None:
    """sens 0.8, spec 0.95: at 50% 0.4 / (0.4 + 0.025); at 20% 0.16 / (0.16 + 0.04); at 2.66% 0.02128 / (0.02128 +
    0.04867). Swapped, 50% would read 0.475 / (0.475 + 0.1) = 0.826."""
    assert PREVALENCES == (0.5, 0.2, 0.0266)
    got = [ppv(0.8, 0.95, pi) for pi in PREVALENCES]
    assert [round(x, 4) for x in got if x is not None] == [0.9412, 0.8, 0.3042]
    assert ppv(None, 0.95, 0.5) is None and ppv(0.0, 1.0, 0.5) is None


# ---------------------------------------------------------------- the 8-target, 2-arm pilot


def repro(target: str, attack: str, kind: CatchKind = CatchKind.FIX_VALIDITY) -> Repro:
    return Repro(
        id=attack,
        attack_id=attack,
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        kind=kind,
        files={"tests/test_repro.py": f"def test_x():\n    assert '{target} {attack}'\n"},
        command=("python", "-m", "pytest", "tests/test_repro.py"),
    )


def finding(target: str, attack: str, status: Status, costs: tuple[StageCost, ...], **kw: object) -> LedgerEntry:
    f = Finding.model_validate(
        {
            "id": f"{target}/fixed_x/{attack}",
            "target_id": target,
            "claim_id": "fixed_x",
            "lens": Lens.CORRECTNESS,
            "status": status,
            "costs": costs,
            **kw,
        }
    )
    return LedgerEntry(at="2026-10-30T12:00:00Z", receipts_version="0.0.1", finding=f)


def blocker(target: str, attack: str = "a1", *, wall: float | None = None) -> LedgerEntry:
    return finding(
        target,
        attack,
        Status.BLOCKER,
        (SPEND,),
        kind=CatchKind.FIX_VALIDITY,
        repro=repro(target, attack),
        runs=(*(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5)), run_result(Outcome.PASS, Side.SURROGATE)),
        quote="It breaks.",
        replay=ReplayKey(base_sha="a" * 40, patch_sha="b" * 40, harness_hash="h", seed=0, image_digest="d"),
        wall_seconds=wall,
    )


def fv_candidate(target: str, attack: str = "f1") -> LedgerEntry:
    runs = (
        *(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5)),
        *(run_result(Outcome.FAIL, Side.BASE) for _ in range(5)),
    )
    return finding(
        target,
        attack,
        Status.S_SAT_INCONCLUSIVE,
        (SPEND,),
        kind=CatchKind.FIX_VALIDITY,
        repro=repro(target, attack),
        runs=runs,
    )


def refuted(target: str, refutation: str) -> LedgerEntry:
    """A blocker SK2 refuted: suspected, its surrogate's and refuter's spend on it as their own stages."""
    costs = (
        SPEND,
        StageCost(stage="surrogate", model=None, micro_usd=30_000, seconds=60.0),
        StageCost(stage="refuter", model="muse-wonder-1", micro_usd=15_000, seconds=5.0),
    )
    return finding(target, "s1", Status.SUSPECTED, costs, refutation=refutation)


INFRA = GoldReplay(outcome=Outcome.ERROR, failure_type=FailureType.INFRA)
FAIL = GoldReplay(outcome=Outcome.FAIL, failure_type=FailureType.ASSERTION)
PASS = GoldReplay(outcome=Outcome.PASS)


def replay_by(gold: Mapping[tuple[str, str], GoldReplay]) -> object:
    def replay(target: Target, r: Repro) -> GoldReplay:
        return gold.get((target.id, r.id), PASS)

    return replay


def spark() -> ArmScore:
    """t0 BAD: two blockers caught on gold (one target, two findings). t1 BAD: a blocker that fails on gold. t2 BAD: a
    fix-validity catch only. t3 PLAUSIBLE_WRONG: caught. t4 GOOD: caught, so its label is disputed. t5 GOOD: a false
    alarm. t6 GOOD: a blocker SK2 refuted. t7 BAD: its only replay is unscored infra, so it is pending."""
    entries = [
        blocker(T[0], "a1", wall=300.0),
        blocker(T[0], "a2", wall=200.0),
        blocker(T[1]),
        fv_candidate(T[2]),
        blocker(T[3], wall=500.0),
        blocker(T[4], wall=100.0),
        blocker(T[5]),
        refuted(T[6], "test_bug: the repro asserts the old behavior"),
        blocker(T[7], wall=50.0),
    ]
    gold = {(T[1], "a1"): FAIL, (T[5], "a1"): FAIL, (T[7], "a1"): INFRA}
    return score_arm(Arm(name="spark"), entries, TARGETS, LABELS, replay_by(gold))  # type: ignore[arg-type]


def glimmer() -> ArmScore:
    """Caught t0, t1 and t7 (BAD); a false alarm on t6 (GOOD); t3 refuted, by a category the refuter names."""
    entries = [blocker(T[0]), blocker(T[1]), blocker(T[6]), blocker(T[7]), refuted(T[3], "intended_change: asked for")]
    gold = {(T[6], "a1"): FAIL}
    return score_arm(Arm(name="glimmer", attacker_model="muse-glimmer-30b"), entries, TARGETS, LABELS, replay_by(gold))  # type: ignore[arg-type]


def the_score() -> Score:
    s, g = spark(), glimmer()
    return Score(
        pilot_status="provisional",
        arms=(s, g),
        rule=apply_rule(RULE, {"spark": s, "glimmer": g}),
        replays=7,
        unscored=1,
    )


def test_score_3s_per_target_rows_sum_to_the_arm_totals_and_keep_surrogate_and_refuter_apart() -> None:
    s = spark()
    rows = {r.target_id: r for r in s.per_target}
    assert [r.target_id for r in s.per_target] == T  # pilot order, a row for every target
    assert (s.catches, s.catch_findings, s.invalid_repros, len(s.unscored)) == (3, 4, 2, 1)
    assert sum(r.caught for r in rows.values()) == s.catches
    assert sum(r.catch_findings for r in rows.values()) == s.catch_findings
    assert sum(r.invalid_repros for r in rows.values()) == s.invalid_repros
    assert sum(r.unscored for r in rows.values()) == len(s.unscored)
    assert sum(r.cost_micro_usd for r in rows.values()) == s.cost_micro_usd == 9 * 120_000 + 45_000
    assert sum(r.fv_caught for r in rows.values()) == s.fix_validity.catches == 1
    assert sum(sum(r.statuses.values()) for r in rows.values()) == s.findings
    assert rows[T[6]].cost_by_stage_micro_usd == {ATTACK: 120_000, "refuter": 15_000, "surrogate": 30_000}
    assert (rows[T[0]].first_alarm_wall_s, rows[T[0]].first_catch_wall_s, rows[T[0]].catch_findings) == (
        200.0,
        200.0,
        2,
    )
    assert (rows[T[7]].alarm, rows[T[7]].caught, rows[T[7]].unscored, rows[T[7]].first_catch_wall_s) == (
        True,
        False,
        1,
        None,
    )
    assert (rows[T[2]].alarm, rows[T[2]].fv_caught, rows[T[6]].alarm) == (False, True, False)
    assert rows[T[6]].statuses == {"suspected": 1} and rows[T[1]].label == "BAD" and rows[T[1]].repo == "django/django"
    assert s.refutations_by_category == {"test_bug": 1}


def test_score_3_round_trips_and_report_refuses_a_score_2() -> None:
    s = the_score()
    text = s.model_dump_json()
    assert s.format == "receipts.eval.score/3" and Score.model_validate_json(text) == s
    assert read_score(text) == s
    old = json.loads(text)
    old["format"] = "receipts.eval.score/2"
    with pytest.raises(ReportError, match="no per-target rows"):
        read_score(json.dumps(old))
    with pytest.raises(ReportError, match="not JSON"):
        read_score("{")


def rates(r: Rate) -> tuple[int, int]:
    return r.k, r.n


def test_plant_the_hand_built_pilot_gives_its_known_rates() -> None:
    """REG-561's numbers for spark: positives t0 t1 t2 t3 (t7 pending), caught t0 t3; GOOD t4 t5 t6, alarms on t4 t5;
    judged alarms t0 t1 t3 t4 t5 (t7's replay unscored), caught t0 t3 t4 (t4 a disputed GOOD, still a catch); MCC
    over all 8, alarm against positive: tp 4 (t0 t1 t3 t7), fp 2, fn 1 (t2), tn 1. With fix-validity, t2 is caught
    and alarms too."""
    a = arm_report(spark(), seed=0, resamples=200)
    r = a.rates
    assert (a.positives, a.negatives, a.excluded) == (5, 3, 0)
    assert rates(r.wdr) == (2, 4) and r.wdr.value == 0.5
    assert rates(r.false_alarm) == (2, 3) and rates(r.car) == (1, 3)
    assert rates(r.precision) == (3, 5) and r.precision.value == 0.6  # 0.4 if a disputed GOOD were a false alarm
    assert r.mcc == pytest.approx((4 * 1 - 2 * 1) / math.sqrt(6 * 5 * 3 * 2))  # (tp+fp)(tp+fn)(tn+fp)(tn+fn)
    assert rates(r.invalid_repro) == (2, 6)
    assert r.good_label_disputed == (T[4],) and r.pending == (T[7],)
    assert all(x.small_n for x in (r.wdr, r.false_alarm, r.car, r.precision, r.invalid_repro))
    f = a.with_fv
    assert rates(f.wdr) == (3, 4) and rates(f.precision) == (4, 6) and rates(f.invalid_repro) == (2, 7)
    assert f.mcc == pytest.approx(5 / math.sqrt(7 * 5 * 3 * 1))
    assert [p.ppv for p in r.ppv] == [ppv(0.5, 1 / 3, pi) for pi in PREVALENCES]
    g = arm_report(glimmer(), seed=0, resamples=200).rates
    assert rates(g.wdr) == (3, 5) and rates(g.false_alarm) == (1, 3) and rates(g.precision) == (3, 4)
    assert g.mcc == pytest.approx(4 / math.sqrt(4 * 5 * 3 * 4)) and g.pending == ()


def test_plant_cost_per_catch_divides_by_caught_targets_not_findings() -> None:
    """spark spent 9 * 120,000 + 45,000 micro-dollars and caught 3 targets with 4 findings: 375,000 per catch, not
    281,250. The stage split shows surrogate and refuter apart; time to catch is t4 100, t0 200, t3 500 s."""
    e = efficiency(spark())
    assert (e.cost_micro_usd, e.catches, e.cost_per_catch_micro_usd) == (1_125_000, 3, 375_000)
    assert (e.catches_with_fv, e.cost_per_catch_with_fv_micro_usd) == (4, 281_250)
    assert e.cost_by_stage_micro_usd == {ATTACK: 1_080_000, "refuter": 15_000, "surrogate": 30_000}
    assert (e.cost_per_target_p50_micro_usd, e.cost_per_target_p95_micro_usd) == (120_000, 240_000)
    assert (e.timed_catches, e.time_to_catch_p50_s, e.time_to_catch_p95_s) == (3, 200.0, 500.0)
    assert efficiency(glimmer()).cost_per_catch_micro_usd == (5 * 120_000 + 45_000) // 3
    empty = spark().model_copy(update={"per_target": ()})
    assert efficiency(empty).cost_per_catch_micro_usd is None


def test_plant_mcnemar_pairs_the_shared_targets_and_counts_only_discordant_ones() -> None:
    """Positives pending in neither arm (t7 is pending in spark): t0 both, t1 glimmer only, t2 neither, t3 spark only.
    GOOD on alarm: t4 and t5 spark only, t6 glimmer only."""
    c = compare(spark(), glimmer())
    p, n = c.positives_caught, c.negatives_alarm
    assert c.shared_targets == 8 and (c.a, c.b) == ("spark", "glimmer")
    assert (p.pairs, p.a_only, p.b_only, p.p) == (4, 1, 1, 1.0)
    assert (n.pairs, n.a_only, n.b_only, n.p) == (3, 2, 1, 1.0)
    relabeled = glimmer().model_copy(
        update={"per_target": tuple(r.model_copy(update={"label": "GOOD"}) for r in glimmer().per_target)}
    )
    with pytest.raises(ReportError, match="labeled differently"):
        compare(spark(), relabeled)


# ---------------------------------------------------------------- the bootstrap


def row(
    n: int, label: Label, *, alarm: bool = False, caught: bool = False, invalid: int = 0, findings: int = 1
) -> TargetOutcome:
    return TargetOutcome(
        target_id=f"t{n}",
        repo="o/r",
        label=label.value,
        alarm=alarm,
        caught=caught,
        fv_caught=False,
        catch_findings=int(caught),
        invalid_repros=invalid,
        unscored=0,
        fv_unscored=0,
        statuses={"blocker" if alarm else "suspected": findings},
        cost_micro_usd=0,
        cost_by_stage_micro_usd={},
        first_alarm_wall_s=None,
        first_catch_wall_s=None,
    )


BALANCED = (
    *(row(n, B, alarm=True, caught=True) for n in range(6)),
    *(row(n, B, alarm=True, invalid=1) for n in range(6, 8)),
    *(row(n, B) for n in range(8, 10)),
    *(row(n, G, alarm=True, invalid=1) for n in range(10, 12)),
    *(row(n, G) for n in range(12, 20)),
)


def test_plant_the_bootstrap_is_seeded_resamples_targets_and_covers_its_point_estimate() -> None:
    """A target with 40 findings weighs what a target with one does: resampling findings would let it dominate."""
    one = bootstrap(BALANCED, with_fv=False, seed=0, resamples=2_000)
    assert one == bootstrap(BALANCED, with_fv=False, seed=0, resamples=2_000)
    assert one != bootstrap(BALANCED, with_fv=False, seed=1, resamples=2_000)
    heavy = (BALANCED[10].model_copy(update={"statuses": {"blocker": 40}}), *BALANCED[:10], *BALANCED[11:])
    reordered = (BALANCED[10], *BALANCED[:10], *BALANCED[11:])
    assert bootstrap(heavy, with_fv=False, seed=0, resamples=2_000) == bootstrap(
        reordered, with_fv=False, seed=0, resamples=2_000
    )
    precision, mcc_ci = one
    assert precision.defined == 2_000 and mcc_ci.defined == 2_000
    assert (
        precision.low is not None and precision.high is not None and mcc_ci.low is not None and mcc_ci.high is not None
    )
    point = 6 / 10  # 6 caught of 10 judged alarms
    point_mcc = (8 * 8 - 2 * 2) / math.sqrt(10 * 10 * 10 * 10)
    assert precision.low < point < precision.high and mcc_ci.low < point_mcc < mcc_ci.high
    assert bootstrap((), with_fv=False, seed=0, resamples=10)[0].defined == 0


# ---------------------------------------------------------------- report.json, report.md and the CLI


def test_report_json_round_trips_and_report_md_shows_every_arm_and_rate_with_its_n() -> None:
    from receipts.eval.report import _rate

    r = report(the_score(), seed=3, resamples=300)
    assert r.format == "receipts.eval.report/1" and r.seed == 3 and r.final is False
    assert Report.model_validate_json(r.model_dump_json()) == r
    md = render_markdown(r)
    assert "Not final" in md and "GOOD label disputed" in md and len(r.comparisons) == 1
    for a in r.arms:
        assert f"| {a.arm} |" in md
        for x in (a.rates, a.with_fv):
            for one in (x.wdr, x.false_alarm, x.car, x.precision, x.invalid_repro):
                assert _rate(one) in md and f"({one.k}/{one.n})" in _rate(one) and "small n" in _rate(one)
    assert f"| arm | {ATTACK} | refuter | surrogate | total |" in md and "test_bug" in md
    assert "p 1.0000 (1 vs 1 discordant of 4)" in md


def test_the_report_command_writes_both_files_and_refuses_a_score_2(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    s = the_score()
    path = tmp_path / "score.json"
    path.write_text(pretty_json(s.model_dump(mode="json")))
    out = tmp_path / "report"
    assert cli.main(["report", "--score", str(path), "--seed", "7", "--out", str(out)]) == 0
    written = Report.model_validate_json((out / "report.json").read_text())
    assert written == report(s, seed=7)
    assert (out / "report.md").read_text() == render_markdown(written)
    assert "spark: WDR 2/4" in capsys.readouterr().out
    old = json.loads(path.read_text())
    old["format"] = "receipts.eval.score/2"
    path.write_text(json.dumps(old))
    assert cli.main(["report", "--score", str(path), "--out", str(tmp_path / "again")]) == 2
    assert "no per-target rows" in capsys.readouterr().err and not (tmp_path / "again").exists()


# ---------------------------------------------------------------- round 2: the independent check's notes


def test_plant_the_bootstrap_counts_a_caught_good_target_as_a_catch_as_the_point_value_does() -> None:
    """B1: the point value and every draw read one precision. A set of caught GOOD targets alone is precision 1, and
    so is every draw: [1, 1]. Counted as false alarms, it would be [0, 0]."""
    disputed = tuple(row(n, G, alarm=True, caught=True) for n in range(4))
    precision, _ = bootstrap(disputed, with_fv=False, seed=0, resamples=500)
    assert (precision.low, precision.high, precision.defined) == (1.0, 1.0, 500)
    mixed = (*disputed[:1], *BALANCED[1:])  # BALANCED with its first caught BAD target swapped for a caught GOOD one
    arm = spark().model_copy(update={"per_target": mixed, "targets": len(mixed)})
    r = arm_rates(arm, with_fv=False, seed=0, resamples=2_000)
    assert (r.precision.k, r.precision.n) == (6, 10)
    assert r.precision_bootstrap.low is not None and r.precision_bootstrap.high is not None
    assert r.precision_bootstrap.low < 0.6 < r.precision_bootstrap.high


def test_plant_the_bootstrap_draws_as_many_targets_as_the_arm_has() -> None:
    """One judged, caught target: every draw of one target is precision 1. A draw one short would hold none, and no
    draw would define precision."""
    precision, _ = bootstrap((row(0, B, alarm=True, caught=True),), with_fv=False, seed=0, resamples=300)
    assert (precision.defined, precision.low, precision.high) == (300, 1.0, 1.0)


def test_plant_every_bootstrap_draw_on_a_many_target_arm_draws_all_of_its_targets(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Round 3: a one-target arm cannot tell a draw of n from max(1, n - 1). On BALANCED's 20 targets every draw asks
    the seeded generator for exactly 20, from those 20 targets, once per resample."""
    from receipts.eval import report as report_mod

    asked: list[tuple[int, int]] = []

    class Spy(random.Random):
        def choices(self, *args: Any, **kwargs: Any) -> Any:
            asked.append((len(args[0]), kwargs["k"]))
            return super().choices(*args, **kwargs)

    monkeypatch.setattr(report_mod.random, "Random", Spy)
    precision, _ = bootstrap(BALANCED, with_fv=False, seed=0, resamples=250)
    assert asked == [(20, 20)] * 250 and precision.defined == 250


def _changed(field: str, value: object, at: int = 1) -> tuple[TargetOutcome, ...]:
    rows = list(spark().per_target)
    rows[at] = rows[at].model_copy(update={field: value})
    return tuple(rows)


@pytest.mark.parametrize(
    ("rows", "message"),
    [
        pytest.param(spark().per_target[:-1], "7 per-target rows for 8 targets", id="short"),
        pytest.param((*spark().per_target[:-1], spark().per_target[0]), "a target twice", id="duplicate-id"),
        pytest.param(_changed("cost_micro_usd", 120_001), "cost_micro_usd", id="cost-off-by-1"),
        pytest.param(_changed("caught", True, at=2), "'catches'", id="catches"),
        pytest.param(_changed("catch_findings", 1), "catch_findings", id="catch-findings"),
        pytest.param(_changed("invalid_repros", 2), "invalid_repros", id="invalid-repros"),
        pytest.param(_changed("unscored", 1), "'unscored'", id="unscored"),
        pytest.param(_changed("fv_caught", True), "fix_validity.catches", id="fv-catches"),
        pytest.param(_changed("fv_unscored", 1), "fix_validity.unscored", id="fv-unscored"),
    ],
)
def test_plant_rows_that_are_not_the_arms_are_refused(rows: tuple[TargetOutcome, ...], message: str) -> None:
    """B2: each check check_rows makes, broken by one row."""
    check_rows(spark())  # the rows as score wrote them pass
    with pytest.raises(ReportError, match=message):
        check_rows(spark().model_copy(update={"per_target": rows}))


def test_plant_a_target_whose_only_open_replay_is_a_fix_validity_candidate_is_pending_with_fv_only() -> None:
    """B3: t1's blocker scored (an invalid repro) and its fix-validity candidate's replay is unscored infra. REG-561
    has nothing open on t1, a scored miss; with_fv does not know yet, so t1 is pending there and out of its WDR."""
    entries = [blocker(T[1]), fv_candidate(T[1]), blocker(T[0])]
    gold = {(T[1], "a1"): FAIL, (T[1], "f1"): INFRA}
    s = score_arm(Arm(name="spark"), entries, TARGETS, LABELS, replay_by(gold))  # type: ignore[arg-type]
    t1 = next(r for r in s.per_target if r.target_id == T[1])
    assert (t1.unscored, t1.fv_unscored, t1.caught, t1.fv_caught) == (0, 1, False, False)
    a = arm_report(s, seed=0, resamples=50)
    assert a.rates.pending == () and a.with_fv.pending == (T[1],)
    assert (a.rates.wdr.k, a.rates.wdr.n, a.with_fv.wdr.k, a.with_fv.wdr.n) == (1, 5, 1, 4)
    assert a.rates.wdr_bounds is None and a.with_fv.wdr_bounds is not None
    assert (a.with_fv.precision.k, a.with_fv.precision.n) == (1, 1)  # t1's alarm waits on its candidate


def test_pending_targets_bound_wdr_and_precision_in_both_files() -> None:
    """t7 is pending in spark: WDR is 2/4 scored, between 2/5 (t7 missed) and 3/5 (t7 caught); precision 3/5,
    between 3/6 and 4/6. Glimmer has nothing pending, so no bounds."""
    r = report(the_score(), seed=0, resamples=100)
    sp, gl = r.arms
    w, p = sp.rates.wdr_bounds, sp.rates.precision_bounds
    assert w is not None and (w.pending, w.low_k, w.low_n, w.high_k, w.high_n, w.low, w.high) == (
        1,
        2,
        5,
        3,
        5,
        0.4,
        0.6,
    )
    assert p is not None and (p.pending, p.low_k, p.low_n, p.high_k, p.high_n) == (1, 3, 6, 4, 6)
    assert (p.low, p.high) == (0.5, 4 / 6)
    assert gl.rates.wdr_bounds is None and gl.rates.precision_bounds is None
    md = render_markdown(r)
    assert "pending 1: 2/5 to 3/5 (40.0% to 60.0%)" in md and "pending 1: 3/6 to 4/6" in md
    assert '"wdr_bounds"' in r.model_dump_json() and Report.model_validate_json(r.model_dump_json()) == r


def test_report_md_shows_money_under_a_cent_and_ppvs_inputs_with_their_n() -> None:
    assert (_usd(300), _usd(0), _usd(120_000), _usd(1_234_567), _usd(None)) == (
        "$0.0003",
        "$0.00",
        "$0.12",
        "$1.234567",
        "n/a",
    )
    r = report(the_score(), seed=0, resamples=100)
    md = render_markdown(r)
    for a in r.arms:
        for name, x in (("REG-561", a.rates), ("with fv", a.with_fv)):
            mark = "" if x.ppv_final else " not final"
            ppvs = " | ".join(("n/a" if q.ppv is None else f"{100 * q.ppv:.1f}%") + mark for q in x.ppv)
            (row,) = [line for line in md.splitlines() if line.startswith(f"| {a.arm} | {name} | ")]
            assert row.startswith(f"| {a.arm} | {name} | {_rate_text(x.wdr)}") and row.endswith(f"| {ppvs} |")
            assert f"| {_rate_text(x.car)}" in row
            if x.wdr_bounds is not None:  # spark: t7 pending, so the sensitivity shows its bounds and is not final
                b = x.wdr_bounds
                assert f"{b.low_k}/{b.low_n} to {b.high_k}/{b.high_n}" in row and "not final |" in row


def _rate_text(r: Rate) -> str:
    from receipts.eval.report import _rate

    return _rate(r)


def test_plant_precision_bounds_count_a_pending_target_with_no_alarm_yet() -> None:
    """Round 3's probe: t0 BAD caught, a GOOD target with a false alarm, t1 BAD whose only open replay is an unscored
    fix-validity candidate. with_fv precision is 1/2 now; t1 has no alarm, so it is out of the low end (1/2) and, as a
    catch, in the high end: 2/3. REG-561 has nothing pending, so its precision is exact."""
    entries = [blocker(T[0]), blocker(T[4]), fv_candidate(T[1])]
    gold = {(T[4], "a1"): FAIL, (T[1], "f1"): INFRA}
    s = score_arm(Arm(name="spark"), entries, TARGETS, LABELS, replay_by(gold))  # type: ignore[arg-type]
    a = arm_report(s, seed=0, resamples=50)
    f = a.with_fv
    assert (f.precision.k, f.precision.n) == (1, 2) and f.pending == (T[1],)
    b = f.precision_bounds
    assert b is not None and (b.pending, b.low_k, b.low_n, b.high_k, b.high_n) == (1, 1, 2, 2, 3)
    assert (b.low, b.high) == (0.5, 2 / 3)
    assert a.rates.precision_bounds is None and a.rates.wdr_bounds is None and a.rates.pending == ()
    assert "pending 1: 1/2 to 2/3 (50.0% to 66.7%)" in render_markdown(
        Report(
            score_format="receipts.eval.score/3",
            pilot_status="provisional",
            final=False,
            seed=0,
            resamples=50,
            arms=(a,),
            comparisons=(),
        )
    )


def test_plant_with_fv_bounds_its_false_alarm_rate_while_a_good_target_waits_on_a_candidate() -> None:
    """Round 4's probe (the check's probe_far.py): t0 BAD caught; t2 GOOD and t3 BAD each have only an unscored
    fix-validity candidate; every other target GOOD and clean. In with_fv a catch on gold is an alarm, so t2 can still
    become a false alarm: FAR 0/6, bounded 0/6 to 1/6, CAR 6/6 bounded 5/6 to 6/6, and MCC and PPV not final. REG-561's
    alarms are blockers only, which need no gold: exact."""
    labels = dict.fromkeys(T, Label.GOOD) | {T[0]: Label.BAD, T[3]: Label.BAD}
    entries = [blocker(T[0]), fv_candidate(T[2]), fv_candidate(T[3])]
    gold = {(T[2], "f1"): INFRA, (T[3], "f1"): INFRA}
    s = score_arm(Arm(name="spark"), entries, TARGETS, labels, replay_by(gold))  # type: ignore[arg-type]
    a = arm_report(s, seed=0, resamples=50)
    f = a.with_fv
    assert (f.false_alarm.k, f.false_alarm.n) == (0, 6) and f.pending == (T[2], T[3])
    fb, cb = f.false_alarm_bounds, f.car_bounds
    assert fb is not None and (fb.low_k, fb.low_n, fb.high_k, fb.high_n, fb.low, fb.high) == (0, 6, 1, 6, 0.0, 1 / 6)
    assert cb is not None and (cb.low_k, cb.low_n, cb.high_k, cb.high_n) == (5, 6, 6, 6)
    assert f.alarms_final is False
    r = a.rates
    assert (r.false_alarm.k, r.false_alarm.n) == (0, 6) and r.pending == ()
    assert r.false_alarm_bounds is None and r.car_bounds is None and r.alarms_final is True
    md = render_markdown(
        Report(
            score_format="receipts.eval.score/3",
            pilot_status="provisional",
            final=False,
            seed=0,
            resamples=50,
            arms=(a,),
            comparisons=(),
        )
    )
    assert "0.0% [0.0%, 39.0%] (0/6) small n; pending 1: 0/6 to 1/6 (0.0% to 16.7%)" in md
    with_fv_table = md.split("## Rates with fix-validity catches")[1].split("##")[0]
    reg_561_table = md.split("## Rates, REG-561")[1].split("##")[0]
    assert "not final" in with_fv_table and "not final" not in reg_561_table
    ppv_rows = [line for line in md.splitlines() if line.startswith("| spark | with fv |")]
    assert len(ppv_rows) == 1 and ppv_rows[0].count("not final") == len(PREVALENCES) + 1  # each PPV and the WDR


# ---------------------------------------------------------------- round 5: every resolution of every open replay

Fixture = tuple[dict[str, Label], list[LedgerEntry], dict[tuple[str, str], GoldReplay], frozenset[str]]
BOUNDED: Final = ("wdr", "precision", "false_alarm", "car", "invalid_repro")


def labeled(*bad: int, pw: tuple[int, ...] = ()) -> dict[str, Label]:
    """Every target GOOD but the BAD and PLAUSIBLE_WRONG ones named by index."""
    return dict.fromkeys(T, G) | {T[i]: B for i in bad} | {T[i]: PW for i in pw}


def _resolved(fx: Fixture, gold: Mapping[tuple[str, str], GoldReplay]) -> ArmReport:
    labels, entries, _, drops = fx
    founder = {fid: "the founder's drop" for fid in drops} or None
    s = score_arm(Arm(name="spark"), entries, TARGETS, labels, replay_by(gold), founder)  # type: ignore[arg-type]
    return arm_report(s, seed=0, resamples=1)


def resolutions(fx: Fixture) -> list[ArmReport]:
    """The report for every way the open replays can resolve on gold: each unscored replay (a blocker's or a
    fix-validity candidate's) passes or fails. A replay the founder dropped stays dropped."""
    gold = fx[2]
    open_keys = [k for k, g in gold.items() if g is INFRA and f"{k[0]}/fixed_x/{k[1]}" not in fx[3]]
    assert open_keys, "a fixture with nothing pending proves nothing here"
    return [
        _resolved(fx, gold | dict(zip(open_keys, way, strict=True)))
        for way in product((PASS, FAIL), repeat=len(open_keys))
    ]


def assert_bounds_hold_under_every_resolution(fx: Fixture) -> None:
    """Each bounded rate: every resolution's value lies inside its bounds and both ends happen; with no bounds, every
    resolution gives the present value. MCC when alarms are final, PPV when it is final, and efficiency's numbers when
    marked final are the same under every resolution."""
    now = _resolved(fx, fx[2])
    outs = resolutions(fx)
    for variant in ("rates", "with_fv"):
        v: ArmRates = getattr(now, variant)
        rs: list[ArmRates] = [getattr(o, variant) for o in outs]
        for name in BOUNDED:
            b: Bounds | None = getattr(v, f"{name}_bounds")
            vals = [getattr(x, name).value for x in rs]
            where = (variant, name, vals, b)
            if b is None:
                assert all(val == getattr(v, name).value for val in vals), where
                continue
            assert len(set(vals)) > 1, where  # bounds only when some resolution moves the rate
            assert b.low in vals and b.high in vals, where  # both ends happen
            assert all(val is None or (b.low or 0.0) <= val <= b.high for val in vals), where
            assert None not in vals or b.low is None, where
        if v.alarms_final:
            assert all(x.mcc == v.mcc for x in rs), (variant, "mcc", [x.mcc for x in rs])
        if v.ppv_final:
            assert all(x.ppv == v.ppv for x in rs), (variant, "ppv")
    e, es = now.efficiency, [o.efficiency for o in outs]
    if e.catches_final:
        assert all((x.catches, x.cost_per_catch_micro_usd) == (e.catches, e.cost_per_catch_micro_usd) for x in es)
    if e.catches_with_fv_final:
        assert all(
            (x.catches_with_fv, x.cost_per_catch_with_fv_micro_usd)
            == (e.catches_with_fv, e.cost_per_catch_with_fv_micro_usd)
            for x in es
        )
    if e.time_to_catch_final:
        timed = (e.timed_catches, e.time_to_catch_p50_s, e.time_to_catch_p95_s)
        assert all((x.timed_catches, x.time_to_catch_p50_s, x.time_to_catch_p95_s) == timed for x in es)


FIXTURES: Final[dict[str, Fixture]] = {
    # the check's probe_fv.py: t2 BAD has only an unscored candidate, beside a catch and a false alarm
    "probe_fv": (
        labeled(0, 2),
        [blocker(T[0]), blocker(T[1]), fv_candidate(T[2])],
        {(T[0], "a1"): PASS, (T[1], "a1"): FAIL, (T[2], "f1"): INFRA},
        frozenset(),
    ),
    # the check's probe_far.py and probe_far4.py (the same score): a GOOD and a BAD target wait on candidates
    "probe_far": (
        labeled(0, 3),
        [blocker(T[0]), fv_candidate(T[2]), fv_candidate(T[3])],
        {(T[0], "a1"): PASS, (T[2], "f1"): INFRA, (T[3], "f1"): INFRA},
        frozenset(),
    ),
    # the only pending target with no alarm yet is a positive
    "pending-positive-no-alarm": (
        labeled(0, 1),
        [blocker(T[0]), fv_candidate(T[1]), blocker(T[4])],
        {(T[0], "a1"): PASS, (T[1], "f1"): INFRA, (T[4], "a1"): FAIL},
        frozenset(),
    ),
    # the only pending GOOD targets already have alarms: t4 a false alarm waiting on its candidate, t5 a blocker
    "pending-good-with-alarm": (
        labeled(0),
        [blocker(T[0]), blocker(T[4]), fv_candidate(T[4]), blocker(T[5])],
        {(T[0], "a1"): PASS, (T[4], "a1"): FAIL, (T[4], "f1"): INFRA, (T[5], "a1"): INFRA},
        frozenset(),
    ),
    # a mix: a blocker and a candidate open on one positive, a candidate on a BAD and on a GOOD target, a blocker on a
    # GOOD target, beside a miss, a catch and a disputed GOOD catch
    "mix": (
        labeled(0, 1, 2, pw=(3,)),
        [
            blocker(T[0]),
            fv_candidate(T[0]),
            blocker(T[1]),
            fv_candidate(T[2]),
            blocker(T[3]),
            blocker(T[4]),
            fv_candidate(T[5]),
            blocker(T[6]),
        ],
        {
            (T[0], "a1"): INFRA,
            (T[0], "f1"): INFRA,
            (T[1], "a1"): FAIL,
            (T[2], "f1"): INFRA,
            (T[3], "a1"): PASS,
            (T[4], "a1"): PASS,
            (T[5], "f1"): INFRA,
            (T[6], "a1"): INFRA,
        },
        frozenset(),
    ),
    # an alarm whose blocker replay the founder dropped, with its candidate open: a miss would leave it unjudged
    "dropped-blocker-open-candidate": (
        labeled(0, 1),
        [blocker(T[0]), fv_candidate(T[0]), blocker(T[1]), blocker(T[4])],
        {(T[0], "a1"): INFRA, (T[0], "f1"): INFRA, (T[1], "a1"): PASS, (T[4], "a1"): FAIL},
        frozenset({f"{T[0]}/fixed_x/a1"}),
    ),
    # every judged alarm caught and the one pending target has no alarm: precision is 1 however it resolves
    "precision-stays-1": (
        labeled(0, 1),
        [blocker(T[0]), fv_candidate(T[1])],
        {(T[0], "a1"): PASS, (T[1], "f1"): INFRA},
        frozenset(),
    ),
}


@pytest.mark.parametrize("name", sorted(FIXTURES))
def test_plant_every_bound_and_every_final_number_holds_under_every_resolution(name: str) -> None:
    """Round 5: close the class, not the case. For each small score with replays pending, every pass or fail of every
    open replay is scored and reported, and the pending report must bound or fix every number it gives."""
    assert_bounds_hold_under_every_resolution(FIXTURES[name])


@pytest.mark.parametrize(
    "claim",
    [
        pytest.param({"invalid_repro_bounds": None}, id="invalid-repro-bounds-stripped"),
        pytest.param({"alarms_final": True}, id="alarms-final-forced-true"),
    ],
)
def test_the_exhaustive_check_fails_on_a_report_that_claims_too_much(
    monkeypatch: pytest.MonkeyPatch, claim: dict[str, object]
) -> None:
    """The check itself, seen failing: on probe_far (two candidates open), a report that strips invalid_repro's
    bounds, or that marks alarms final, claims a number every resolution keeps, and the exhaustive check refuses it."""
    from receipts.eval import report as report_mod

    real = report_mod.arm_rates

    def claims_too_much(arm: ArmScore, *, with_fv: bool, seed: int, resamples: int = 1) -> ArmRates:
        return real(arm, with_fv=with_fv, seed=seed, resamples=resamples).model_copy(update=claim)

    assert_bounds_hold_under_every_resolution(FIXTURES["probe_far"])  # the honest report passes
    monkeypatch.setattr(report_mod, "arm_rates", claims_too_much)
    with pytest.raises(AssertionError):
        assert_bounds_hold_under_every_resolution(FIXTURES["probe_far"])
