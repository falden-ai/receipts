"""EV3, the eval report (`receipts eval report`): what a score's per-target rows say, with their uncertainty, per arm
and between arms, for the rehearsal and the final.

Every number here is a pure function of a Score of format /3 (a /2 score has no per-target rows and is refused, with
the reason), and nothing here reads a ledger, calls a model or uses the network. Only the CLI wrapper (eval/cli.py)
reads score.json and writes report.json and report.md.

Targets. Positives are BAD and PLAUSIBLE_WRONG targets, negatives GOOD; a target of any other label is in no rate and
is counted as excluded. A target is pending while it is not caught and one of its replays on gold is still unscored
infra: whether it is caught is not known yet, so it is left out of the catch rate and of precision, and the report
says it is not final. In REG-561's numbers an alarm is any final blocker, which needs no gold, so the false-alarm rate
and MCC read every target and are exact. In with_fv a fix-validity catch on gold is an alarm too, so alarms there do
depend on gold: a pending target with no alarm yet can still become one. While one is pending, with_fv's false-alarm
rate and CAR carry bounds (k/n to (k+p)/n, p the pending GOOD targets with no alarm yet; CAR mirrored), and
ArmRates.alarms_final is False, which marks MCC and PPV "not final" in report.md.

Per arm the numbers come twice: REG-561's (blockers only), and `with_fv`, where a fix-validity catch on gold
(REG-577) also counts, as a catch and as an alarm.

- WDR, the catch rate: caught positives over positives that are not pending.
- False-alarm rate: GOOD targets with an alarm over GOOD targets. CAR is 1 minus it.
- Precision: caught targets over targets with a judged alarm, one with at least one scored replay (it is caught, or
  has an invalid repro) and not pending. A catch on a GOOD target is a catch here, as it is in score, and the target
  is listed as "GOOD label disputed".
- MCC over the targets, an alarm as the prediction and a positive label as the truth; None when a margin is empty.
- Invalid-repro rate: score's invalid repros over the replays that scored (for with_fv, the blockers' and the
  fix-validity candidates' together).

Pending replays. Each unscored replay will resolve one way or the other on gold, and a pending target resolves into a
catch (any of its open replays passes) or a miss (all of them fail). Every rate carries Bounds while some resolution
can change it, each end with its own count and n, and both ends are outcomes that can happen; the bounds are None
exactly when every resolution gives the rate its present value (a test enumerates every resolution of small scores
to hold this). WDR: c/(n+p) with every pending positive missed, (c+p)/(n+p) with every one caught. Precision: every
pending target enters the denominator as a catch, and as a miss only if it is judged then (it has an alarm with a
blocker replay that will have scored), so the low end is c/(n+d) and the high end (c+p)/(n+p), d those judged as a
miss. False-alarm rate (with_fv only): k/n to (k+p)/n, p the pending GOOD targets with no alarm yet; CAR mirrored.
Invalid-repro rate: k/(n+u) to (k+u)/(n+u), u the unscored replays. alarms_final is False while any pending target
has no alarm yet (it could become one), and then MCC is not final; ppv_final is False while WDR or CAR has bounds or
alarms are not final. Efficiency says which of its numbers are final the same way. The bounds cover the founder's
drops too: a dropped replay stays dropped under every resolution, and the exhaustive test includes one. An alarm whose
every replay the founder dropped (score --replay-drop) is not pending, since nothing will score it: on a positive it
is a miss in WDR, it is an alarm in MCC and in the false-alarm rate, and it is absent from precision, which has no
scored replay to judge it by.

Uncertainty. Every rate carries its count and n, a Wilson 95% interval, and "small n" when n is under SMALL_N.
Precision and MCC also get a bootstrap clustered by target: RESAMPLES draws of the arm's targets with replacement
(random.Random(seed)), the statistic on each draw where it is defined, and the 2.5 and 97.5 percentiles of those
(linear between order statistics). The target is the unit because the findings on one target are not independent.

Comparisons. McNemar's exact test between any two arms over the targets they share: on the positives that are pending
in neither arm, on `caught`; on the negatives, on `alarm`. The run's arms are compared pairwise in score order;
baselines plug in later as further arms of the same function.

Efficiency and refutations: cost per caught target, the cost split by stage, p50 and p95 cost per target, time to
catch, and the suspected findings SK2 refuted by category, all from the score's rows.
"""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from fractions import Fraction
from itertools import combinations
from typing import Final, Literal

from receipts.contracts import Frozen
from receipts.eval.labels import Label
from receipts.eval.score import SCORE_FORMAT, ArmScore, Score, TargetOutcome, percentiles

REPORT_FORMAT: Literal["receipts.eval.report/1"] = "receipts.eval.report/1"
POSITIVES: Final = frozenset({Label.BAD.value, Label.PLAUSIBLE_WRONG.value})
NEGATIVES: Final = frozenset({Label.GOOD.value})
Z_95: Final = 1.959963984540054  # the standard normal's 97.5th percentile: a two-sided 95% interval
RESAMPLES: Final = 10_000
SMALL_N: Final = 10  # a rate over fewer is marked "small n"
PREVALENCES: Final = (0.5, 0.2, 0.0266)  # for PPV: balanced, one in five, and 2.66%


class ReportError(ValueError):
    """A score that cannot be reported as given."""


# ---------------------------------------------------------------- statistics


def wilson(k: int, n: int, z: float = Z_95) -> tuple[float, float] | None:
    """The Wilson score interval for k successes of n, None with n 0. With p = k/n:

        (p + z^2/(2n) +- z * sqrt(p(1 - p)/n + z^2/(4n^2))) / (1 + z^2/n)

    Its ends are exactly 0 at k = 0 and exactly 1 at k = n, where the closed form meets them up to rounding."""
    if n == 0:
        return None
    if not 0 <= k <= n:
        raise ValueError(f"wilson: {k} of {n}")
    p = k / n
    z2 = z * z
    denom = 1 + z2 / n
    center = (p + z2 / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z2 / (4 * n * n)) / denom
    low = 0.0 if k == 0 else max(0.0, center - half)
    high = 1.0 if k == n else min(1.0, center + half)
    return low, high


def mcnemar_exact(b: int, c: int) -> float:
    """McNemar's exact test, two-sided: under the null each of the b + c discordant pairs falls either way with
    probability 1/2, so p = min(1, 2 * sum_{i=0}^{min(b, c)} C(b + c, i) / 2^(b + c)), and 1.0 with none."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, float(Fraction(2 * tail, 2**n)))


def ppv(sensitivity: float | None, specificity: float | None, prevalence: float) -> float | None:
    """The positive predictive value at a prevalence pi, by Bayes:

        PPV = sens * pi / (sens * pi + (1 - spec) * (1 - pi))

    with sensitivity the arm's WDR and specificity its CAR. None when either is unknown or nothing would alarm."""
    if sensitivity is None or specificity is None:
        return None
    true_alarms = sensitivity * prevalence
    false_alarms = (1 - specificity) * (1 - prevalence)
    if true_alarms + false_alarms == 0:
        return None
    return true_alarms / (true_alarms + false_alarms)


def mcc(tp: int, fp: int, fn: int, tn: int) -> float | None:
    """Matthews correlation: (tp * tn - fp * fn) / sqrt((tp + fp)(tp + fn)(tn + fp)(tn + fn)); None when a margin is
    empty."""
    denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
    if denom == 0:
        return None
    return (tp * tn - fp * fn) / math.sqrt(denom)


def quantile(values: Sequence[float], q: float) -> float:
    """Linear between order statistics (numpy's default): position q * (len - 1) in the sorted values."""
    s = sorted(values)
    h = q * (len(s) - 1)
    lo, hi = math.floor(h), math.ceil(h)
    return s[lo] + (h - lo) * (s[hi] - s[lo])


def nearest_rank(values: Sequence[float], p: float) -> float | None:
    """Nearest rank, as score.percentiles takes it: a value that occurs, never interpolated; None with none."""
    if not values:
        return None
    s = sorted(values)
    return s[max(0, math.ceil(p * len(s)) - 1)]


# ---------------------------------------------------------------- the report's shape


class Rate(Frozen):
    k: int
    n: int
    value: float | None  # k / n; None with n 0
    low: float | None  # Wilson 95%
    high: float | None
    small_n: bool  # n under SMALL_N


def rate(k: int, n: int) -> Rate:
    ci = wilson(k, n)
    return Rate(
        k=k,
        n=n,
        value=None if n == 0 else k / n,
        low=None if ci is None else ci[0],
        high=None if ci is None else ci[1],
        small_n=n < SMALL_N,
    )


class Interval(Frozen):
    low: float | None  # the 2.5th percentile of the resampled statistic
    high: float | None  # the 97.5th
    defined: int  # resamples where the statistic is defined


class Ppv(Frozen):
    prevalence: float
    ppv: float | None


class Bounds(Frozen):
    """A rate while targets are pending: every pending one resolved against it (low) or for it (high)."""

    pending: int  # the pending targets that can change the rate
    low_k: int
    low_n: int
    high_k: int
    high_n: int
    low: float | None  # low_k / low_n; None with low_n 0
    high: float  # high_k / high_n


def bounds(k: int, n: int, *, counted: int, new: int = 0) -> Bounds | None:
    """The bounds of k/n with `counted` pending targets that enter n either way and `new` ones that enter it only
    when they resolve into a success: low = k / (n + counted), high = (k + counted + new) / (n + counted + new).
    None when no resolution can change the rate: it is then exact. That is nothing pending, or the one case where the
    ends meet: k / n = 1 with every pending target `new` (a success keeps it at 1, a failure leaves n alone). From
    k * new = (counted + new) * (n + counted) with k <= n, equality needs counted 0 and k = n > 0."""
    if not counted and not new:
        return None
    if not counted and k == n > 0:
        return None
    low_n, high_k, high_n = n + counted, k + counted + new, n + counted + new
    return Bounds(
        pending=counted + new,
        low_k=k,
        low_n=low_n,
        high_k=high_k,
        high_n=high_n,
        low=None if not low_n else k / low_n,
        high=high_k / high_n,
    )


class ArmRates(Frozen):
    wdr: Rate  # caught positives over positives not pending
    wdr_bounds: Bounds | None  # while positives are pending
    precision_bounds: Bounds | None  # while alarms are pending
    false_alarm: Rate  # GOOD targets with an alarm over GOOD targets
    false_alarm_bounds: Bounds | None  # with_fv, while GOOD targets with no alarm yet are pending
    car: Rate  # 1 minus the false-alarm rate
    car_bounds: Bounds | None  # the false-alarm bounds, mirrored
    alarms_final: bool  # False while a pending target has no alarm yet: FAR, CAR and MCC can still move
    precision: Rate  # caught targets over targets with a judged alarm
    invalid_repro: Rate  # invalid repros over the replays that scored
    invalid_repro_bounds: Bounds | None  # while replays are unscored
    mcc: float | None  # final only when alarms_final
    precision_bootstrap: Interval
    mcc_bootstrap: Interval
    ppv: tuple[Ppv, ...]
    ppv_final: bool  # False while WDR or CAR has bounds or alarms are not final
    good_label_disputed: tuple[str, ...]  # GOOD targets that are caught
    pending: tuple[str, ...]  # targets left out of WDR and precision until their replays score


class Efficiency(Frozen):
    cost_micro_usd: int
    catches: int  # caught targets
    cost_per_catch_micro_usd: int | None  # arm cost over caught targets; None with none
    catches_with_fv: int
    cost_per_catch_with_fv_micro_usd: int | None
    cost_by_stage_micro_usd: dict[str, int]
    cost_per_target_p50_micro_usd: int
    cost_per_target_p95_micro_usd: int
    timed_catches: int  # caught targets whose first catch has wall seconds
    time_to_catch_p50_s: float | None
    time_to_catch_p95_s: float | None
    catches_final: bool  # no uncaught target has an unscored blocker replay: catches and cost per catch are final
    catches_with_fv_final: bool  # the same with the fix-validity candidates' replays
    time_to_catch_final: bool  # no blocker replay is unscored: an earlier catch cannot still appear


class ArmReport(Frozen):
    arm: str
    attacker_model: str
    targets: int
    positives: int
    negatives: int
    excluded: int  # targets of any other label
    rates: ArmRates  # REG-561's, blockers only
    with_fv: ArmRates  # fix-validity catches on gold included
    efficiency: Efficiency
    refutations_by_category: dict[str, int]


class McNemar(Frozen):
    pairs: int  # shared targets in this class
    a_only: int  # discordant: yes in the first arm, no in the second
    b_only: int
    p: float


class Comparison(Frozen):
    a: str
    b: str
    shared_targets: int
    positives_caught: McNemar  # positives pending in neither arm, on caught
    negatives_alarm: McNemar  # GOOD targets, on alarm


class Report(Frozen):
    format: Literal["receipts.eval.report/1"] = REPORT_FORMAT
    score_format: str
    pilot_status: Literal["provisional", "frozen"]
    final: bool  # no replay on gold is still unscored
    seed: int
    resamples: int
    small_n: int = SMALL_N
    arms: tuple[ArmReport, ...]
    comparisons: tuple[Comparison, ...]


# ---------------------------------------------------------------- per arm


@dataclass(frozen=True)
class _Unit:
    """One target as one variant (REG-561's or with_fv) reads it."""

    target_id: str
    positive: bool
    negative: bool
    alarm: bool
    caught: bool
    judged: bool  # an alarm with a scored replay, not pending: in precision's denominator
    pending: bool
    judged_if_missed: bool  # pending, and judged should every open replay fail: an alarm with a blocker replay


def _unit(r: TargetOutcome, with_fv: bool) -> _Unit:
    caught = r.caught or (with_fv and r.fv_caught)
    alarm = r.alarm or (with_fv and r.fv_caught)
    pending = not caught and (r.unscored + (r.fv_unscored if with_fv else 0)) > 0
    return _Unit(
        target_id=r.target_id,
        positive=r.label in POSITIVES,
        negative=r.label in NEGATIVES,
        alarm=alarm,
        caught=caught,
        judged=alarm and (caught or r.invalid_repros > 0) and not pending,
        pending=pending,
        # a failed blocker replay is an invalid repro, so an alarm with one scored or open is judged as a miss; an
        # alarm whose blocker replays were all dropped is not, whatever its candidates do
        judged_if_missed=pending and alarm and (r.invalid_repros > 0 or r.unscored > 0),
    )


def _units(rows: Sequence[TargetOutcome], with_fv: bool) -> list[_Unit]:
    """The arm's positive and negative targets, one unit each."""
    return [_unit(r, with_fv) for r in rows if r.label in POSITIVES or r.label in NEGATIVES]


def _precision_counts(units: Sequence[_Unit]) -> tuple[int, int]:
    """Precision's count and n: caught targets (a disputed GOOD one included) over judged alarms. The point value
    and every bootstrap draw read this one function."""
    return sum(u.caught for u in units), sum(u.judged for u in units)


def _precision(units: Sequence[_Unit]) -> float | None:
    caught, judged = _precision_counts(units)
    return None if not judged else caught / judged


def _mcc(units: Sequence[_Unit]) -> float | None:
    tp = sum(u.positive and u.alarm for u in units)
    fp = sum(u.negative and u.alarm for u in units)
    fn = sum(u.positive and not u.alarm for u in units)
    tn = sum(u.negative and not u.alarm for u in units)
    return mcc(tp, fp, fn, tn)


def _interval(values: Sequence[float]) -> Interval:
    if not values:
        return Interval(low=None, high=None, defined=0)
    return Interval(low=quantile(values, 0.025), high=quantile(values, 0.975), defined=len(values))


def bootstrap(
    rows: Sequence[TargetOutcome], *, with_fv: bool, seed: int, resamples: int = RESAMPLES
) -> tuple[Interval, Interval]:
    """Precision's and MCC's 95% intervals, clustered by target: each resample draws the arm's targets (never its
    findings) with replacement, as many as it has, and the statistic is taken where that draw defines it."""
    units = _units(rows, with_fv)
    rng = random.Random(seed)
    precisions: list[float] = []
    mccs: list[float] = []
    if units:
        for _ in range(resamples):
            draw = rng.choices(units, k=len(units))
            p, m = _precision(draw), _mcc(draw)
            if p is not None:
                precisions.append(p)
            if m is not None:
                mccs.append(m)
    return _interval(precisions), _interval(mccs)


def arm_rates(arm: ArmScore, *, with_fv: bool, seed: int, resamples: int = RESAMPLES) -> ArmRates:
    units = _units(arm.per_target, with_fv)
    positives = [u for u in units if u.positive and not u.pending]
    negatives = [u for u in units if u.negative]
    false_alarms = sum(u.alarm for u in negatives)
    invalid = arm.invalid_repros
    scored = arm.invalid_repros + arm.catch_findings
    unscored = len(arm.unscored)  # replays that will score one way or the other
    if with_fv:
        invalid += arm.fix_validity.invalid_repros
        scored += arm.fix_validity.invalid_repros + arm.fix_validity.catch_findings
        unscored += len(arm.fix_validity.unscored)
    wdr = rate(sum(u.caught for u in positives), len(positives))
    car = rate(len(negatives) - false_alarms, len(negatives))
    precision = rate(*_precision_counts(units))  # the count every bootstrap draw reads too
    precision_ci, mcc_ci = bootstrap(arm.per_target, with_fv=with_fv, seed=seed, resamples=resamples)
    pending_pos = sum(u.positive and u.pending for u in units)
    # precision: a pending target enters n as a catch; as a miss only when it is judged then
    judged_miss = sum(u.judged_if_missed for u in units)
    pending_rest = sum(u.pending and not u.judged_if_missed for u in units)
    # any pending target, positive or GOOD, with no alarm yet: a catch would make one, so alarms are not final
    no_alarm_yet = sum(u.pending and not u.alarm for u in units)
    # GOOD targets that can still become false alarms (with_fv only: in REG-561 a pending target has its alarm). They
    # are in n at both ends already, so the bounds are k/n to (k + p)/n.
    neg_new = sum(u.pending and not u.alarm for u in negatives)
    false_alarm = rate(false_alarms, len(negatives))
    wdr_bounds = bounds(wdr.k, wdr.n, counted=pending_pos)
    car_bounds = bounds(car.k - neg_new, car.n - neg_new, counted=neg_new)
    alarms_final = no_alarm_yet == 0
    return ArmRates(
        wdr=wdr,
        wdr_bounds=wdr_bounds,
        precision_bounds=bounds(precision.k, precision.n, counted=judged_miss, new=pending_rest),
        false_alarm=false_alarm,
        false_alarm_bounds=bounds(false_alarm.k, false_alarm.n - neg_new, counted=neg_new),
        car=car,
        car_bounds=car_bounds,
        alarms_final=alarms_final,
        precision=precision,
        invalid_repro=rate(invalid, scored),
        invalid_repro_bounds=bounds(invalid, scored, counted=unscored),
        mcc=_mcc(units),
        precision_bootstrap=precision_ci,
        mcc_bootstrap=mcc_ci,
        ppv=tuple(Ppv(prevalence=pi, ppv=ppv(wdr.value, car.value, pi)) for pi in PREVALENCES),
        ppv_final=alarms_final and wdr_bounds is None and car_bounds is None,
        good_label_disputed=tuple(u.target_id for u in negatives if u.caught),
        pending=tuple(u.target_id for u in units if u.pending),
    )


def _per_catch(cost: int, catches: int) -> int | None:
    return None if not catches else round(cost / catches)


def efficiency(arm: ArmScore) -> Efficiency:
    """Cost per caught target (arm cost over caught targets, never over findings), the cost by stage, p50 and p95 cost
    per target (nearest rank), and time to catch: p50 and p95 of each caught target's first catch, in wall seconds."""
    rows = arm.per_target
    stages: Counter[str] = Counter()
    for r in rows:
        stages.update(r.cost_by_stage_micro_usd)
    catches = sum(r.caught for r in rows)
    with_fv = sum(r.caught or r.fv_caught for r in rows)
    per_target = percentiles([r.cost_micro_usd for r in rows])
    walls = [r.first_catch_wall_s for r in rows if r.caught and r.first_catch_wall_s is not None]
    return Efficiency(
        cost_micro_usd=arm.cost_micro_usd,
        catches=catches,
        cost_per_catch_micro_usd=_per_catch(arm.cost_micro_usd, catches),
        catches_with_fv=with_fv,
        cost_per_catch_with_fv_micro_usd=_per_catch(arm.cost_micro_usd, with_fv),
        cost_by_stage_micro_usd=dict(sorted(stages.items())),
        cost_per_target_p50_micro_usd=per_target.p50,
        cost_per_target_p95_micro_usd=per_target.p95,
        timed_catches=len(walls),
        time_to_catch_p50_s=nearest_rank(walls, 0.5),
        time_to_catch_p95_s=nearest_rank(walls, 0.95),
        catches_final=not any(not r.caught and r.unscored for r in rows),
        catches_with_fv_final=not any(not (r.caught or r.fv_caught) and r.unscored + r.fv_unscored for r in rows),
        time_to_catch_final=not any(r.unscored for r in rows),
    )


def check_rows(arm: ArmScore) -> None:
    """The rows are the arm's: one per pilot target, once each, and every total they share is their sum."""
    rows = arm.per_target
    ids = [r.target_id for r in rows]
    if len(rows) != arm.targets:
        raise ReportError(f"{arm.arm}: {len(rows)} per-target rows for {arm.targets} targets")
    twice = sorted(t for t, n in Counter(ids).items() if n > 1)
    if twice:
        raise ReportError(f"{arm.arm}: per-target rows name a target twice: {twice}")
    sums = {
        "catches": (arm.catches, sum(r.caught for r in rows)),
        "catch_findings": (arm.catch_findings, sum(r.catch_findings for r in rows)),
        "invalid_repros": (arm.invalid_repros, sum(r.invalid_repros for r in rows)),
        "unscored": (len(arm.unscored), sum(r.unscored for r in rows)),
        "cost_micro_usd": (arm.cost_micro_usd, sum(r.cost_micro_usd for r in rows)),
        "fix_validity.catches": (arm.fix_validity.catches, sum(r.fv_caught for r in rows)),
        "fix_validity.unscored": (len(arm.fix_validity.unscored), sum(r.fv_unscored for r in rows)),
    }
    off = {k: v for k, v in sums.items() if v[0] != v[1]}
    if off:
        raise ReportError(f"{arm.arm}: totals that are not the sums of the per-target rows (total, sum): {off}")


def arm_report(arm: ArmScore, *, seed: int, resamples: int = RESAMPLES) -> ArmReport:
    check_rows(arm)
    labels = [r.label for r in arm.per_target]
    positives = sum(label in POSITIVES for label in labels)
    negatives = sum(label in NEGATIVES for label in labels)
    return ArmReport(
        arm=arm.arm,
        attacker_model=arm.attacker_model,
        targets=arm.targets,
        positives=positives,
        negatives=negatives,
        excluded=len(labels) - positives - negatives,
        rates=arm_rates(arm, with_fv=False, seed=seed, resamples=resamples),
        with_fv=arm_rates(arm, with_fv=True, seed=seed, resamples=resamples),
        efficiency=efficiency(arm),
        refutations_by_category=dict(arm.refutations_by_category),
    )


# ---------------------------------------------------------------- between arms


def _mcnemar(pairs: Sequence[tuple[bool, bool]]) -> McNemar:
    a_only = sum(x and not y for x, y in pairs)
    b_only = sum(y and not x for x, y in pairs)
    return McNemar(pairs=len(pairs), a_only=a_only, b_only=b_only, p=mcnemar_exact(a_only, b_only))


def compare(a: ArmScore, b: ArmScore) -> Comparison:
    """McNemar's exact test between any two arms (a baseline is one more arm) over the targets both have rows for:
    caught on the positives pending in neither arm, alarm on the negatives. REG-561's numbers, blockers only."""
    ua = {u.target_id: u for u in _units(a.per_target, with_fv=False)}
    ub = {u.target_id: u for u in _units(b.per_target, with_fv=False)}
    shared = [t for t in ua if t in ub]
    for t in shared:
        if (ua[t].positive, ua[t].negative) != (ub[t].positive, ub[t].negative):
            raise ReportError(f"{t}: labeled differently in {a.arm} and {b.arm}")
    pos = [(ua[t].caught, ub[t].caught) for t in shared if ua[t].positive and not ua[t].pending and not ub[t].pending]
    neg = [(ua[t].alarm, ub[t].alarm) for t in shared if ua[t].negative]
    return Comparison(
        a=a.arm, b=b.arm, shared_targets=len(shared), positives_caught=_mcnemar(pos), negatives_alarm=_mcnemar(neg)
    )


# ---------------------------------------------------------------- the whole report


def read_score(text: str) -> Score:
    """score.json, format /3 only: a /2 score has no per-target rows to resample or pair."""
    try:
        raw = json.loads(text)
    except ValueError as e:
        raise ReportError(f"score.json is not JSON: {e}") from e
    fmt = raw.get("format") if isinstance(raw, dict) else None
    if fmt != SCORE_FORMAT:
        raise ReportError(
            f"score.json is format {fmt!r}, not {SCORE_FORMAT!r}: an older score has no per-target rows; run "
            "`receipts eval score` again with the same --out (settled replays are cached)"
        )
    try:
        return Score.model_validate(raw)
    except ValueError as e:
        raise ReportError(f"score.json does not validate: {str(e)[:500]}") from e


def report(s: Score, *, seed: int = 0, resamples: int = RESAMPLES) -> Report:
    return Report(
        score_format=s.format,
        pilot_status=s.pilot_status,
        final=s.unscored == 0 and s.fix_validity_unscored == 0,
        seed=seed,
        resamples=resamples,
        arms=tuple(arm_report(a, seed=seed, resamples=resamples) for a in s.arms),
        comparisons=tuple(compare(a, b) for a, b in combinations(s.arms, 2)),
    )


# ---------------------------------------------------------------- report.md


def _pct(x: float | None) -> str:
    return "n/a" if x is None else f"{100 * x:.1f}%"


def _rate(r: Rate) -> str:
    """A rate with its Wilson interval and its n, and "small n" under SMALL_N."""
    ci = "" if r.value is None else f" [{_pct(r.low)}, {_pct(r.high)}]"
    text = f"{_pct(r.value)}{ci} ({r.k}/{r.n})"
    return f"{text} small n" if r.small_n else text


def _num(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.3f}"


def _usd(micro: int | None) -> str:
    """Dollars to the cent, or to the micro-dollar when the amount is not whole cents ($0.0003, never $0.00)."""
    if micro is None:
        return "n/a"
    if micro % 10_000 == 0:
        return f"${micro / 1e6:.2f}"
    whole, frac = divmod(micro, 1_000_000)
    return f"${whole}.{frac:06d}".rstrip("0")


def _bounds(b: Bounds | None) -> str:
    """The bounds while targets are pending, shown next to the rate."""
    if b is None:
        return ""
    return f"; pending {b.pending}: {b.low_k}/{b.low_n} to {b.high_k}/{b.high_n} ({_pct(b.low)} to {_pct(b.high)})"


def _secs(x: float | None) -> str:
    return "n/a" if x is None else f"{x:.0f} s"


def _not_final(r: ArmRates) -> str:
    """Marks a number that reads every target's alarm while some alarm still waits on gold (with_fv only)."""
    return _mark(r.alarms_final)


def _mark(final: bool) -> str:
    return "" if final else " not final"


def _rates_table(arms: Sequence[ArmReport], with_fv: bool) -> list[str]:
    out = [
        "| arm | WDR | false alarms | CAR | precision | precision bootstrap | MCC | MCC bootstrap | invalid repros |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for a in arms:
        r = a.with_fv if with_fv else a.rates
        pb, mb = r.precision_bootstrap, r.mcc_bootstrap
        out.append(
            f"| {a.arm} | {_rate(r.wdr)}{_bounds(r.wdr_bounds)} "
            f"| {_rate(r.false_alarm)}{_bounds(r.false_alarm_bounds)} | {_rate(r.car)}{_bounds(r.car_bounds)} "
            f"| {_rate(r.precision)}{_bounds(r.precision_bounds)} "
            f"| [{_num(pb.low)}, {_num(pb.high)}] ({pb.defined} draws) | {_num(r.mcc)}{_not_final(r)} "
            f"| [{_num(mb.low)}, {_num(mb.high)}] ({mb.defined} draws) "
            f"| {_rate(r.invalid_repro)}{_bounds(r.invalid_repro_bounds)} |"
        )
    return out


def render_markdown(r: Report) -> str:
    lines = [
        "# Eval report",
        "",
        f"Score {r.score_format}, pilot {r.pilot_status}. Bootstrap: {r.resamples} resamples of targets, seed {r.seed}."
        f" Intervals are 95% (Wilson for rates). A rate over fewer than {r.small_n} is marked small n.",
        "",
    ]
    if not r.final:
        lines += ["**Not final:** some replays on gold are still unscored; their targets are pending.", ""]
    lines += [
        "## Arms",
        "",
        "| arm | attacker | targets | positives | negatives | excluded |",
        "|---|---|---|---|---|---|",
    ]
    lines += [
        f"| {a.arm} | {a.attacker_model} | {a.targets} | {a.positives} | {a.negatives} | {a.excluded} |" for a in r.arms
    ]
    lines += ["", "## Rates, REG-561 (blockers only)", "", *_rates_table(r.arms, with_fv=False)]
    lines += ["", "## Rates with fix-validity catches (REG-577)", "", *_rates_table(r.arms, with_fv=True)]
    for title, fv in (("REG-561", False), ("with fix-validity", True)):
        flagged = [(a.arm, a.with_fv if fv else a.rates) for a in r.arms]
        if any(x.good_label_disputed or x.pending for _, x in flagged):
            lines += ["", f"### Targets to read, {title}", ""]
            for arm, x in flagged:
                if x.good_label_disputed:
                    lines.append(f"- {arm}: GOOD label disputed: {', '.join(x.good_label_disputed)}")
                if x.pending:
                    lines.append(f"- {arm}: pending (a replay still unscored): {', '.join(x.pending)}")
    header = " | ".join(f"PPV at {100 * p:g}%" for p in PREVALENCES)
    lines += [
        "",
        "## PPV from WDR and CAR",
        "",
        f"| arm | variant | sensitivity (WDR) | specificity (CAR) | {header} |",
        "|---|---|---|---|" + "---|" * len(PREVALENCES),
    ]
    for a in r.arms:
        for name, x in (("REG-561", a.rates), ("with fv", a.with_fv)):
            mark = _mark(x.ppv_final)
            ppvs = " | ".join(f"{_pct(p.ppv)}{mark}" for p in x.ppv)
            sens = f"{_rate(x.wdr)}{_bounds(x.wdr_bounds)}{mark}"
            spec = f"{_rate(x.car)}{_bounds(x.car_bounds)}"
            lines.append(f"| {a.arm} | {name} | {sens} | {spec} | {ppvs} |")
    stages = sorted({s for a in r.arms for s in a.efficiency.cost_by_stage_micro_usd})
    lines += [
        "",
        "## Cost by stage",
        "",
        f"| arm | {' | '.join(stages)} | total |",
        "|---|" + "---|" * (len(stages) + 1),
    ]
    for a in r.arms:
        e = a.efficiency
        cells = " | ".join(_usd(e.cost_by_stage_micro_usd.get(s, 0)) for s in stages)
        lines.append(f"| {a.arm} | {cells} | {_usd(e.cost_micro_usd)} |")
    lines += [
        "",
        "## Efficiency",
        "",
        "| arm | cost per catch | cost per catch with fv | cost per target p50 | p95 | time to catch p50 | p95 |",
        "|---|---|---|---|---|---|---|",
    ]
    for a in r.arms:
        e = a.efficiency
        lines.append(
            f"| {a.arm} | {_usd(e.cost_per_catch_micro_usd)} ({e.catches} caught){_mark(e.catches_final)} "
            f"| {_usd(e.cost_per_catch_with_fv_micro_usd)} ({e.catches_with_fv} caught)"
            f"{_mark(e.catches_with_fv_final)} "
            f"| {_usd(e.cost_per_target_p50_micro_usd)} | {_usd(e.cost_per_target_p95_micro_usd)} "
            f"| {_secs(e.time_to_catch_p50_s)} | {_secs(e.time_to_catch_p95_s)} ({e.timed_catches} timed)"
            f"{_mark(e.time_to_catch_final)} |"
        )
    lines += ["", "## McNemar exact, between arms", ""]
    if r.comparisons:
        lines += ["| arms | positives, caught | negatives, alarm |", "|---|---|---|"]
        for c in r.comparisons:
            p, n = c.positives_caught, c.negatives_alarm
            lines.append(
                f"| {c.a} vs {c.b} | p {p.p:.4f} ({p.a_only} vs {p.b_only} discordant of {p.pairs}) "
                f"| p {n.p:.4f} ({n.a_only} vs {n.b_only} discordant of {n.pairs}) |"
            )
    else:
        lines.append("One arm: nothing to compare.")
    categories = sorted({c for a in r.arms for c in a.refutations_by_category})
    lines += ["", "## Refutations (SK2), suspected findings by category", ""]
    if categories:
        lines += [f"| arm | {' | '.join(categories)} |", "|---|" + "---|" * len(categories)]
        for a in r.arms:
            cells = " | ".join(str(a.refutations_by_category.get(c, 0)) for c in categories)
            lines.append(f"| {a.arm} | {cells} |")
    else:
        lines.append("None.")
    return "\n".join(lines) + "\n"
