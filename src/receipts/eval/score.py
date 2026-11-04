"""Pilot scoring (`receipts eval score`): per arm, what the attackers caught, what their repros were worth, what the arm
cost and how much compute it used, and REG-561's move rule applied to exactly those numbers.

A catch counts only if the repro fails on the agent patch and passes on gold (spec section 3), and gold is used only
here, at scoring. The patch side comes from the ledger: a finding whose final status is blocker has failed on the patch
in every run (contracts.Finding enforces it). The gold side is judged now: each blocker's repro is replayed in the run
phase's sandbox on gold (`Sandbox.prepare(target, Side.GOLD, patch=gold)`, then `run_test` with the target's runner
and the repro's files), and only PASS counts; a repro that fails, errors with a typed failure, is skipped or times
out on gold is an invalid repro. A replay that answers ERROR with failure type INFRA or none means the sandbox could
not answer: it is retried once in a fresh container, never cached, and if it persists the blocker is unscored infra,
counted on neither side, and `receipts eval score` exits 2. A repro that names its cases (contracts.Repro.cases)
replays with them as `expect`, so PASS means each of them ran and passed on gold.

A replay that is infra on every score has one way out, the founder's: `score --replay-drop <finding>=<reason>`
drops that finding's replay in every arm where it is unscored, so it counts on neither side (never a catch, never an
invalid repro, and never in the invalid-repro rate's denominator) and the rule can be judged; score.json lists it with
the reason. An arm whose every blocker replay was dropped has no rate, so the rule is not met. A drop that names
neither a blocker nor a fix-validity candidate (below), or whose replay scored, is refused, and nothing is ever
dropped on its own.

REG-561's rule counts gated catches per target: a target counts once when any of its blocker findings fails on the
agent patch and passes on gold; the per-finding count is reported beside it. The invalid-repro rate stays per replayed
finding: invalid repros over the blocker findings whose replay was scored. A catch on a GOOD target is still a catch
by that oracle, and its target is also reported as "GOOD label disputed" (spec section 3, adjudication).

Fix-validity catches on gold, a second number beside REG-561's (the founder's decision, REG-577): the skeptic marks a
fix-validity repro s_sat_inconclusive when it fails on the patch in all runs on a counted failure and passes on base
0 of N, since its oracle is a surrogate fix that is not built yet (SK3). Gold is that oracle at scoring, so each such
candidate is replayed on gold exactly as a blocker is (same replay, same cache, same retry), and a PASS there is a valid
fix-validity catch: the patch fails what the reference fix passes. Per arm, ArmScore.fix_validity reports the
candidates, how many pass on gold, the invalid repros, the caught targets by label, the GOOD targets among them (as
"GOOD label disputed", the way a blocker's catch is), the unscored replays and the founder's drops. This number never
enters REG-561's rule: the blocker numbers, the rate, `judged` and Score.replays read blockers only, and the
candidates' replays are counted apart (Score.fix_validity_replays; a repro both need is replayed once and counted in
each). An unscored candidate replay does not hold the rule back, but it is handled as a blocker's is: it is never
cached, `receipts eval score` exits 2 while one is unscored, and if it stays infra the founder may drop it by finding
id (score --replay-drop), which counts it on neither side of this number.

Ledgers: one JSONL file per arm, <arm>.jsonl, of contracts.LedgerEntry (schema 2). Entries are states: a finding's
last entry is its status, and costs are increments, so an arm's cost is the sum over every entry. Compute hours are
the skeptic's sandbox runs (the runs on each finding's last entry) plus every code-only stage (a StageCost with no
model); model-call seconds are reported apart. The planner's cost is shared by both arms and recorded outside the
ledgers, so it is not in either arm's cost here.

Per target (format /3, for EV3's report: its bootstrap and McNemar resample and pair targets, which totals cannot
give): ArmScore.per_target holds one TargetOutcome per pilot target, in pilot order, and every total it shares with
the arm is the sum of its rows. Each row says whether the target raised an alarm (any final blocker), whether REG-561
caught it and whether REG-577's fix-validity number did, its invalid and unscored replays, its final statuses, its
cost split by StageCost.stage, and the wall seconds of its first alarm and first catch. ArmScore.refutations_by_category
counts the suspected findings SK2 refuted by the category their refutation starts with ("<category>: <evidence>").
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Literal

from pydantic import Field

from receipts.contracts import (
    BLOCKER_RUNS,
    COUNTED_FAILURES,
    Arm,
    CatchKind,
    FailureType,
    Finding,
    Frozen,
    LedgerEntry,
    Outcome,
    Repro,
    Side,
    Status,
    Target,
)
from receipts.eval.hashing import canonical_json, sha256_text
from receipts.eval.labels import Label
from receipts.eval.pilot import MoveRule, PilotConfig, rule_sha256

SCORE_FORMAT: Literal["receipts.eval.score/3"] = "receipts.eval.score/3"  # /3: per-target rows (EV3)


class ScoreError(ValueError):
    """Ledgers, labels or arms that cannot be scored as given."""


def read_ledger(text: str, where: str) -> list[LedgerEntry]:
    """Every entry of one ledger, strictly: a line that does not validate is refused by its number. A torn last line
    (a crash mid-write) is refused too: open the ledger with the ledger's own reader first, which recovers it."""
    out = []
    lines = text.split("\n")
    if lines and lines[-1] == "":
        lines.pop()
    elif lines:
        raise ScoreError(f"{where}: the last line has no newline (a torn tail); recover the ledger first")
    for n, line in enumerate(lines, 1):
        if not line.strip():
            continue
        try:
            out.append(LedgerEntry.model_validate_json(line))
        except ValueError as e:
            raise ScoreError(f"{where} line {n}: not a schema-2 ledger entry: {str(e)[:200]}") from e
    return out


def last_states(entries: Iterable[LedgerEntry]) -> dict[str, Finding]:
    """Each finding's last state, by finding id, in ledger order."""
    out: dict[str, Finding] = {}
    for e in entries:
        out[e.finding.id] = e.finding
    return out


def replay_key(target_id: str, repro: Repro) -> str:
    """The same repro on the same target replays once, whichever arm found it. Its cases are part of what it runs
    (the replay's `expect`), so they are in the key; a repro with none keys as it always did."""
    key: dict[str, object] = {"target": target_id, "files": repro.files, "command": list(repro.command)}
    if repro.cases:
        key["cases"] = list(repro.cases)
    return sha256_text(canonical_json(key))


def repro_tests(repro: Repro) -> list[str]:
    """The repro's own test files: those its command names, else the files that look like tests, else every .py
    file but conftest.py and __init__.py."""
    named = []
    for arg in repro.command:
        path = arg.split("::", 1)[0].removeprefix("./")
        if path in repro.files and path not in named:
            named.append(path)
    if named:
        return named
    files = sorted(repro.files)
    helpers = {"conftest.py", "__init__.py"}
    tests = [
        f
        for f in files
        if f.endswith(".py")
        and PurePosixPath(f).name not in helpers
        and (
            PurePosixPath(f).name.startswith("test")
            or f.endswith("_test.py")
            or f.endswith("tests.py")
            or "tests" in PurePosixPath(f).parts
        )
    ]
    return tests or [f for f in files if f.endswith(".py") and PurePosixPath(f).name not in helpers] or files


class Percentiles(Frozen):
    mean: int
    p50: int
    p95: int
    max: int


def percentiles(values: Sequence[int]) -> Percentiles:
    """Nearest rank: p50 and p95 are values that occur, never interpolated."""
    if not values:
        return Percentiles(mean=0, p50=0, p95=0, max=0)
    s = sorted(values)

    def at(p: float) -> int:
        return s[max(0, math.ceil(p * len(s)) - 1)]

    return Percentiles(mean=sum(s) // len(s), p50=at(0.5), p95=at(0.95), max=s[-1])


GOLD_OUTPUT_CHARS = 2_000  # of each stream of a gold replay's run, its last characters, kept in the replay cache


class GoldReplay(Frozen):
    """One repro's replay on gold. `outcome` None: the sandbox could not answer, twice (unscored infra). The run's own
    words, the last GOLD_OUTPUT_CHARS of each stream and its problem, go in the replay cache with it, so why a repro
    failed on gold is read from the run, not inferred; a cache line written before they were kept has none."""

    outcome: Outcome | None
    failure_type: FailureType | None = None
    detail: str = ""
    stdout_tail: str = Field(default="", max_length=GOLD_OUTPUT_CHARS)
    stderr_tail: str = Field(default="", max_length=GOLD_OUTPUT_CHARS)
    problem: str = Field(default="", max_length=GOLD_OUTPUT_CHARS)

    @property
    def scored(self) -> bool:
        """An answer about the repro on gold: never None, and never ERROR with failure type INFRA or none."""
        return self.outcome is not None and not is_infra(self.outcome, self.failure_type)

    @property
    def passed(self) -> bool:
        return self.outcome is Outcome.PASS


def is_infra(outcome: Outcome, failure_type: FailureType | None) -> bool:
    """ERROR with failure type INFRA or none: by contract the sandbox could not answer, which says nothing of gold."""
    return outcome is Outcome.ERROR and failure_type in (None, FailureType.INFRA)


def on_gold(replay: GoldReplay) -> str:
    """What an invalid repro did on gold: its outcome, and for an error its failure type."""
    assert replay.outcome is not None
    if replay.outcome is Outcome.ERROR and replay.failure_type is not None:
        return f"error:{replay.failure_type.value}"
    return replay.outcome.value


def fix_validity_candidate(f: Finding) -> bool:
    """A fix-validity catch the skeptic could not settle: s_sat_inconclusive, kind fix_validity, a repro, and runs that
    fail on the patch in every one of at least BLOCKER_RUNS on a counted failure and pass on base in none of at least
    BLOCKER_RUNS. Base failing too is what a fix-validity catch looks like (the patch did not fix what the issue asks);
    gold passing it is what makes it valid. A base run that fails with an import error shows the repro imports what
    base lacks, not a bug base has, so it is no candidate."""
    if f.status is not Status.S_SAT_INCONCLUSIVE or f.kind is not CatchKind.FIX_VALIDITY or f.repro is None:
        return False
    patch = [r for r in f.runs if r.side is Side.PATCH]
    base = [r for r in f.runs if r.side is Side.BASE]
    if len(patch) < BLOCKER_RUNS or len(base) < BLOCKER_RUNS:
        return False
    failed = all(r.outcome is Outcome.FAIL and r.failure_type in COUNTED_FAILURES for r in patch)
    imports = any(r.failure_type is FailureType.IMPORT for r in base)
    return failed and not imports and not any(r.outcome is Outcome.PASS for r in base)


class FixValidityScore(Frozen):
    """REG-577's second number, per arm: fix-validity candidates (fix_validity_candidate) replayed on gold."""

    candidates: int = 0  # findings
    catch_findings: int = 0  # candidates whose repro passes on gold: valid fix-validity catches
    catches: int = 0  # targets with at least one, each counted once
    catches_by_label: dict[str, int] = Field(default_factory=dict)  # those targets, by label
    catch_findings_by_label: dict[str, int] = Field(default_factory=dict)
    good_label_disputed: int = 0  # GOOD targets with a fix-validity catch
    invalid_repros: int = 0  # scored candidates whose repro does not pass on gold
    invalid_on_gold: dict[str, int] = Field(default_factory=dict)
    unscored: tuple[str, ...] = ()  # candidate ids whose replay hit infra twice: on neither side
    dropped: dict[str, str] = Field(default_factory=dict)  # unscored ones the founder dropped -> the reason


class TargetOutcome(Frozen):
    """One pilot target in one arm. A target with no findings has a row too: all zeros, no alarm."""

    target_id: str
    repo: str
    label: str  # the pilot's label, a labels.Label value
    alarm: bool  # any finding whose final status is blocker
    caught: bool  # REG-561: a blocker whose repro passes on gold
    fv_caught: bool  # REG-577: a fix-validity candidate whose repro passes on gold
    catch_findings: int  # blocker findings whose repro passes on gold
    invalid_repros: int  # scored blocker findings whose repro does not pass on gold
    unscored: int  # blocker findings whose replay hit infra twice and was not dropped
    fv_unscored: int  # the same, for fix-validity candidates
    statuses: dict[str, int]  # final statuses
    cost_micro_usd: int
    cost_by_stage_micro_usd: dict[str, int]  # by StageCost.stage, so surrogate and refuter show apart
    first_alarm_wall_s: float | None  # the earliest blocker's wall_seconds; None with none, or none recorded
    first_catch_wall_s: float | None  # the earliest caught blocker's wall_seconds


def refutation_category(refutation: str) -> str:
    """The SK2 category a refutation starts with ("<category>: <evidence>"), or "unparsed" when it has none."""
    category, sep, _ = refutation.partition(":")
    return category.strip() if sep and category.strip() else "unparsed"


class ArmScore(Frozen):
    arm: str
    attacker_model: str
    targets: int  # pilot targets
    targets_with_findings: int
    findings: int
    statuses: dict[str, int]  # final statuses
    blockers: int
    catches: int  # REG-561: targets with at least one blocker whose repro passes on gold, each counted once
    catches_by_label: dict[str, int]  # those targets, by label
    catch_findings: int  # blocker findings whose repro passes on gold
    catch_findings_by_label: dict[str, int]
    good_label_disputed: int  # GOOD targets with a catch
    invalid_repros: int  # scored blocker findings whose repro does not pass on gold
    invalid_repro_pct: float | None  # invalid over scored blocker findings; None with none scored
    invalid_on_gold: dict[str, int]  # what the invalid repros did on gold
    unscored: tuple[str, ...] = ()  # blocker finding ids whose replay hit infra twice: on neither side
    dropped: dict[str, str] = Field(default_factory=dict)  # unscored ones the founder dropped -> the reason
    cost_micro_usd: int
    cost_per_target_micro_usd: Percentiles
    sandbox_hours: float  # the skeptic's runs, on each finding's last state
    code_stage_hours: float  # StageCost with no model
    model_hours: float  # StageCost with a model (latency, not compute)
    compute_hours: float  # sandbox plus code stages
    fix_validity: FixValidityScore = Field(default_factory=FixValidityScore)  # REG-577, outside REG-561's rule
    per_target: tuple[TargetOutcome, ...]  # one per pilot target, in pilot order; totals are their sums
    refutations_by_category: dict[str, int] = Field(default_factory=dict)  # suspected findings SK2 refuted


class RuleOutcome(Frozen):
    rule: MoveRule
    rule_sha256: str
    baseline_catches: int  # targets
    candidate_catches: int  # targets
    baseline_catch_findings: int
    candidate_catch_findings: int
    candidate_blockers: int
    candidate_invalid_repro_pct: float | None
    judged: bool  # False while any blocker's replay is unscored infra: the numbers are not final
    founder_dropped: int = 0  # blocker replays the founder dropped (score --replay-drop): on neither side
    met: bool
    why: str


class Score(Frozen):
    format: Literal["receipts.eval.score/3"] = SCORE_FORMAT
    pilot_status: Literal["provisional", "frozen"]
    arms: tuple[ArmScore, ...]
    rule: RuleOutcome
    replays: int = Field(ge=0)  # distinct blocker repros replayed on gold, over every arm
    unscored: int = Field(default=0, ge=0)  # blocker findings, over every arm, whose replay is unscored infra
    replay_drops: dict[str, str] = Field(default_factory=dict)  # the founder's drops that landed -> the reason
    fix_validity_replays: int = Field(default=0, ge=0)  # distinct fix-validity candidates' repros replayed on gold
    fix_validity_unscored: int = Field(default=0, ge=0)  # candidates, over every arm, whose replay is unscored infra


Replay = Callable[[Target, Repro], GoldReplay]


def score_arm(
    arm: Arm,
    entries: Sequence[LedgerEntry],
    targets: Mapping[str, Target],
    labels: Mapping[str, Label],
    replay: Replay,
    drops: Mapping[str, str] | None = None,
) -> ArmScore:
    """One arm's numbers. `drops` (the founder's, by finding id) take an unscored replay off the books, a blocker's or
    a fix-validity candidate's; a drop on a replay that scored is ignored here and refused by score()."""
    for e in entries:
        if e.finding.target_id not in targets:
            raise ScoreError(f"{arm.name}: finding {e.finding.id} is for {e.finding.target_id}, not a pilot target")
    states = last_states(entries)
    blockers = [f for f in states.values() if f.status is Status.BLOCKER]
    caught: set[str] = set()  # targets
    finding_catches: Counter[str] = Counter()
    invalid_on_gold: Counter[str] = Counter()
    unscored: list[str] = []
    dropped: dict[str, str] = {}
    # per target: caught findings, invalid repros, unscored replays, and the wall seconds of each alarm and catch
    t_catches: Counter[str] = Counter()
    t_invalid: Counter[str] = Counter()
    t_unscored: Counter[str] = Counter()
    alarm_walls: dict[str, list[float]] = {t: [] for t in targets}
    catch_walls: dict[str, list[float]] = {t: [] for t in targets}
    for f in blockers:
        assert f.repro is not None  # contracts.Finding: a blocker has a repro
        if f.wall_seconds is not None:
            alarm_walls[f.target_id].append(f.wall_seconds)
        r = replay(targets[f.target_id], f.repro)
        if not r.scored:
            if drops is not None and f.id in drops:
                dropped[f.id] = drops[f.id]
            else:
                unscored.append(f.id)
                t_unscored[f.target_id] += 1
        elif r.passed:
            caught.add(f.target_id)
            finding_catches[labels[f.target_id].value] += 1
            t_catches[f.target_id] += 1
            if f.wall_seconds is not None:
                catch_walls[f.target_id].append(f.wall_seconds)
        else:
            invalid_on_gold[on_gold(r)] += 1
            t_invalid[f.target_id] += 1
    catches_by_label = Counter(labels[t].value for t in caught)
    # The rate's denominator is the blocker findings whose replay was scored: an unscored replay is on neither side,
    # and so is one the founder dropped (the pre-registered rule, pilot.MoveRule.definitions).
    scored = len(blockers) - len(unscored) - len(dropped)
    per_target: Counter[str] = Counter({t: 0 for t in targets})
    by_stage: dict[str, Counter[str]] = {t: Counter() for t in targets}
    code_s = model_s = 0.0
    for e in entries:
        for c in e.finding.costs:
            per_target[e.finding.target_id] += c.micro_usd
            by_stage[e.finding.target_id][c.stage] += c.micro_usd
            if c.model is None:
                code_s += c.seconds
            else:
                model_s += c.seconds
    sandbox_s = sum(r.duration_s for f in states.values() for r in f.runs)
    invalid = sum(invalid_on_gold.values())
    fix_validity, fv_caught = _fix_validity(
        [f for f in states.values() if fix_validity_candidate(f)], targets, labels, replay, drops
    )
    fv_unscored = Counter(states[fid].target_id for fid in fix_validity.unscored)
    statuses: dict[str, Counter[str]] = {t: Counter() for t in targets}
    for f in states.values():
        statuses[f.target_id][f.status.value] += 1
    rows = tuple(
        TargetOutcome(
            target_id=t,
            repo=target.repo,
            label=labels[t].value,
            alarm=statuses[t][Status.BLOCKER.value] > 0,
            caught=t in caught,
            fv_caught=t in fv_caught,
            catch_findings=t_catches[t],
            invalid_repros=t_invalid[t],
            unscored=t_unscored[t],
            fv_unscored=fv_unscored[t],
            statuses=dict(sorted(statuses[t].items())),
            cost_micro_usd=per_target[t],
            cost_by_stage_micro_usd=dict(sorted(by_stage[t].items())),
            first_alarm_wall_s=min(alarm_walls[t], default=None),
            first_catch_wall_s=min(catch_walls[t], default=None),
        )
        for t, target in targets.items()
    )
    refuted = Counter(
        refutation_category(f.refutation)
        for f in states.values()
        if f.status is Status.SUSPECTED and f.refutation is not None
    )
    return ArmScore(
        arm=arm.name,
        attacker_model=arm.attacker_model,
        targets=len(targets),
        targets_with_findings=len({f.target_id for f in states.values()}),
        findings=len(states),
        statuses=dict(sorted(Counter(f.status.value for f in states.values()).items())),
        blockers=len(blockers),
        catches=len(caught),
        catches_by_label=dict(sorted(catches_by_label.items())),
        catch_findings=sum(finding_catches.values()),
        catch_findings_by_label=dict(sorted(finding_catches.items())),
        good_label_disputed=catches_by_label.get(Label.GOOD.value, 0),
        invalid_repros=invalid,
        invalid_repro_pct=None if not scored else round(100 * invalid / scored, 1),
        invalid_on_gold=dict(sorted(invalid_on_gold.items())),
        unscored=tuple(unscored),
        dropped=dict(sorted(dropped.items())),
        cost_micro_usd=sum(per_target.values()),
        cost_per_target_micro_usd=percentiles(list(per_target.values())),
        sandbox_hours=round(sandbox_s / 3600, 3),
        code_stage_hours=round(code_s / 3600, 3),
        model_hours=round(model_s / 3600, 3),
        compute_hours=round((sandbox_s + code_s) / 3600, 3),
        fix_validity=fix_validity,
        per_target=rows,
        refutations_by_category=dict(sorted(refuted.items())),
    )


def score_fix_validity(
    candidates: Sequence[Finding],
    targets: Mapping[str, Target],
    labels: Mapping[str, Label],
    replay: Replay,
    drops: Mapping[str, str] | None = None,
) -> FixValidityScore:
    """Each candidate's repro replayed on gold, as a blocker's is: PASS is a valid catch, any other scored answer an
    invalid repro, and infra twice is unscored, unless the founder dropped that finding id (`drops`), which puts it
    on neither side."""
    return _fix_validity(candidates, targets, labels, replay, drops)[0]


def _fix_validity(
    candidates: Sequence[Finding],
    targets: Mapping[str, Target],
    labels: Mapping[str, Label],
    replay: Replay,
    drops: Mapping[str, str] | None,
) -> tuple[FixValidityScore, frozenset[str]]:
    """score_fix_validity, and the targets it caught, for the per-target rows."""
    caught: set[str] = set()
    found: Counter[str] = Counter()
    invalid: Counter[str] = Counter()
    unscored: list[str] = []
    dropped: dict[str, str] = {}
    for f in candidates:
        assert f.repro is not None  # fix_validity_candidate
        r = replay(targets[f.target_id], f.repro)
        if not r.scored:
            if drops is not None and f.id in drops:
                dropped[f.id] = drops[f.id]
            else:
                unscored.append(f.id)
        elif r.passed:
            caught.add(f.target_id)
            found[labels[f.target_id].value] += 1
        else:
            invalid[on_gold(r)] += 1
    by_label = Counter(labels[t].value for t in caught)
    fv = FixValidityScore(
        candidates=len(candidates),
        catch_findings=sum(found.values()),
        catches=len(caught),
        catches_by_label=dict(sorted(by_label.items())),
        catch_findings_by_label=dict(sorted(found.items())),
        good_label_disputed=by_label.get(Label.GOOD.value, 0),
        invalid_repros=sum(invalid.values()),
        invalid_on_gold=dict(sorted(invalid.items())),
        unscored=tuple(unscored),
        dropped=dict(sorted(dropped.items())),
    )
    return fv, frozenset(caught)


def apply_rule(rule: MoveRule, arms: Mapping[str, ArmScore]) -> RuleOutcome:
    base, cand = arms[rule.baseline_arm], arms[rule.candidate_arm]
    rate = cand.invalid_repro_pct
    enough = cand.catches >= base.catches - rule.catch_margin
    unscored = len(base.unscored) + len(cand.unscored)
    founder = len(base.dropped) + len(cand.dropped)
    if unscored:
        met, why = False, f"not judged: {unscored} blocker replays on gold hit infra twice; rerun score"
    elif rate is None and cand.blockers:
        met, why = (
            False,
            (
                f"{rule.candidate_arm} reported {cand.blockers} blockers, but the founder dropped the replay of every "
                "one (score --replay-drop), so no blocker replay was scored and there is no invalid-repro rate to judge"
            ),
        )
    elif rate is None:
        met, why = False, f"{rule.candidate_arm} reported no blockers, so it has no invalid-repro rate to judge"
    elif not enough:
        met, why = (
            False,
            (
                f"{rule.candidate_arm} caught {cand.catches} targets, under {rule.baseline_arm}'s {base.catches} "
                f"minus {rule.catch_margin}"
            ),
        )
    elif rate >= rule.max_invalid_repro_pct:
        met, why = (
            False,
            f"{rule.candidate_arm}'s invalid-repro rate is {rate}%, not under {rule.max_invalid_repro_pct}%",
        )
    else:
        met, why = (
            True,
            (
                f"{rule.candidate_arm} caught {cand.catches} targets against {rule.baseline_arm}'s {base.catches} "
                f"(margin {rule.catch_margin}) with {rate}% invalid repros; move {rule.moves} only once confirmed "
                f"{rule.confirm}"
            ),
        )
    if founder:
        why += f"; {founder} blocker replays the founder dropped (score --replay-drop) count on neither side"
    return RuleOutcome(
        rule=rule,
        rule_sha256=rule_sha256(rule),
        baseline_catches=base.catches,
        candidate_catches=cand.catches,
        baseline_catch_findings=base.catch_findings,
        candidate_catch_findings=cand.catch_findings,
        candidate_blockers=cand.blockers,
        candidate_invalid_repro_pct=rate,
        judged=not unscored,
        founder_dropped=founder,
        met=met,
        why=why,
    )


def score(
    cfg: PilotConfig,
    run_targets: Sequence[Target],
    ledgers: Mapping[str, Sequence[LedgerEntry]],
    replay: Replay,
    *,
    drops: Mapping[str, str] | None = None,
) -> Score:
    """Score every arm of the pilot from its ledger; `replay` judges a repro on gold, a blocker's or a fix-validity
    candidate's. `drops` are the founder's (finding id -> reason): each must name a blocker finding or a fix-validity
    candidate whose replay is unscored in at least one arm, and it lands in every arm where that replay is unscored;
    any other drop is refused."""
    if cfg.rule_sha256 != rule_sha256(cfg.rule):
        raise ScoreError("the pilot's move rule does not match its pinned hash")
    targets = {t.id: t for t in run_targets}
    if sorted(targets) != sorted(t.target_id for t in cfg.targets):
        raise ScoreError("the run file's targets are not the pilot's")
    labels = {t.target_id: t.label for t in cfg.targets}
    missing = [a.name for a in cfg.arms if a.name not in ledgers]
    if missing:
        raise ScoreError(f"no ledger for arms {missing}")
    stray = sorted(set(ledgers) - {a.name for a in cfg.arms})
    if stray:
        raise ScoreError(f"ledgers for arms the pilot does not have: {stray}")
    replays: dict[str, GoldReplay] = {}

    def cached(target: Target, repro: Repro) -> GoldReplay:
        key = replay_key(target.id, repro)
        if key not in replays:
            replays[key] = replay(target, repro)
        return replays[key]

    founder = dict(drops or {})
    arms = {a.name: score_arm(a, ledgers[a.name], targets, labels, cached, founder) for a in cfg.arms}
    states = [f for a in cfg.arms for f in last_states(ledgers[a.name]).values()]
    blockers = [f for f in states if f.status is Status.BLOCKER]
    candidates = [f for f in states if fix_validity_candidate(f)]
    named = {f.id for f in (*blockers, *candidates)}
    for fid in sorted(founder):
        if fid not in named:
            raise ScoreError(
                f"--replay-drop {fid}: no arm has a blocker finding with that id, and none has a fix-validity "
                "candidate with it"
            )
        if not any(fid in a.dropped or fid in a.fix_validity.dropped for a in arms.values()):
            raise ScoreError(
                f"--replay-drop {fid}: its replay on gold scored, so there is nothing to drop; run score without it"
            )
    return Score(
        pilot_status=cfg.status,
        arms=tuple(arms[a.name] for a in cfg.arms),
        rule=apply_rule(cfg.rule, arms),
        replays=len({replay_key(f.target_id, f.repro) for f in blockers if f.repro is not None}),
        unscored=sum(len(a.unscored) for a in arms.values()),
        replay_drops={fid: founder[fid] for fid in sorted(founder)},
        fix_validity_replays=len({replay_key(f.target_id, f.repro) for f in candidates if f.repro is not None}),
        fix_validity_unscored=sum(len(a.fix_validity.unscored) for a in arms.values()),
    )


def ledger_paths(directory: Path, arms: Iterable[Arm]) -> dict[str, Path]:
    return {a.name: directory / f"{a.name}.jsonl" for a in arms}
