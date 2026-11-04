"""The 10/28 pilot (spec run gates): 20 targets from the sample's core calibration side, two attacker arms on all of
them, and two temperature schedules split across them.

Arms (REG-561): `spark` (muse-spark-1.3 on Meta's API) and `glimmer` (muse-glimmer-30b on Falden's own vLLM), as
contracts.Arm with the same step limit and seed, so only the attacker model differs: the same Wonder plan, seeds, step
limits and SK1 gates run in both. The move rule is written here, before the run, and pilot.json pins it by hash:
move the attacker loops and MU1's rewrites to Glimmer if its gated catches are at least Spark's minus 1 and its
invalid-repro rate is under 30%, confirmed on 40 paired targets in the dress rehearsal. Gated catches are counted per
target (a target counts once when any of its blocker findings fails on the agent patch and passes on gold), and the
invalid-repro rate per replayed finding.

Temperatures (the founder's protocol of 2026-10-27, REG-565):
- `spec`: the spec's schedule (parameters table and PL1): per claim, the first attack prompt at temperature 0 and every
  later prompt at 0.8; AT2's critic at 0 and its rewrite at 0.8;
- `meta_default`: every attacker call at 1.0, which the founder's decision calls Meta's recommended default (this lane
  did not check that against Meta's documentation).
Half the targets of every label take each schedule (BAD 4 + 4, GOOD 4 + 4, PLAUSIBLE_WRONG 2 + 2), alternating in rank
order, and a target keeps its schedule in both arms. Only attacker calls change: intake, the planner and the skeptic
keep the spec's settings.

Files: eval/pilot.json (this config), the run file (JSONL, one {"target": <contracts.Target>, "temperatures": "spec" |
"meta_default"} per line) and the arms file (a JSON list of contracts.Arm). Per-arm ledgers go in one directory, one
JSONL ledger per arm named <arm>.jsonl. A provisional pilot (from a provisional sample) is never run.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import Field

from receipts.contracts import IMAGE_REF, Arm, Frozen, RunnerKind, Target
from receipts.eval.hashing import SHA256, canonical_json, pretty_json, sha256_text
from receipts.eval.labels import Label
from receipts.eval.manifest import Manifest
from receipts.eval.sampler import PILOT_PLAN, Item, Sample, Side, pilot
from receipts.eval.targets import target_id

PILOT_FORMAT: Literal["receipts.eval.pilot/2"] = "receipts.eval.pilot/2"
RUN_FILE = "eval/pilot-run.jsonl"
ARMS_FILE = "eval/pilot-arms.json"


class TemperatureMode(StrEnum):
    SPEC = "spec"
    META_DEFAULT = "meta_default"


class Schedule(Frozen):
    first_prompt: float = Field(ge=0, le=2)  # a claim's first attack prompt
    later_prompts: float = Field(ge=0, le=2)  # every other attack prompt for the claim
    critic: float = Field(ge=0, le=2)  # AT2 stage 2 critic
    rewrite: float = Field(ge=0, le=2)  # AT2 stage 2 rewrite
    source: str


SCHEDULES: dict[TemperatureMode, Schedule] = {
    TemperatureMode.SPEC: Schedule(
        first_prompt=0.0,
        later_prompts=0.8,
        critic=0.0,
        rewrite=0.8,
        source="spec v3.1 parameters table and PL1 (temp 0 first, then 0.8; S2, S3) and AT2 (critic 0, rewrite 0.8)",
    ),
    TemperatureMode.META_DEFAULT: Schedule(
        first_prompt=1.0,
        later_prompts=1.0,
        critic=1.0,
        rewrite=1.0,
        source="1.0, Meta's recommended default per the founder's pilot protocol, 2026-10-27",
    ),
}

CallKind = Literal["prompt", "critic", "rewrite"]


def attacker_temperature(mode: TemperatureMode, call: CallKind, prompt_index: int = 0) -> float:
    """The temperature of one attacker call. prompt_index counts a claim's attack prompts from 0."""
    if prompt_index < 0:
        raise ValueError("prompt_index counts from 0")
    s = SCHEDULES[mode]
    if call == "prompt":
        return s.first_prompt if prompt_index == 0 else s.later_prompts
    return s.critic if call == "critic" else s.rewrite


# REG-561: the two arms differ only in the attacker model.
ARMS: tuple[Arm, ...] = (
    Arm(name="spark", attacker_model="muse-spark-1.3", step_limit=20, seed=0),
    Arm(name="glimmer", attacker_model="muse-glimmer-30b", step_limit=20, seed=0),
)


class MoveRule(Frozen):
    """REG-561's decision rule, fixed before the pilot runs. `receipts eval score` applies exactly this."""

    id: Literal["REG-561"] = "REG-561"
    baseline_arm: str = "spark"
    candidate_arm: str = "glimmer"
    catch_margin: int = Field(default=1, ge=0)  # the candidate's caught targets >= the baseline's minus this
    max_invalid_repro_pct: int = Field(default=30, gt=0, le=100)  # the candidate's rate must be strictly under this
    moves: str = "the attacker loops (AT1, AT2 stage 2) and MU1's rewrites"
    confirm: str = "on 40 paired targets in the dress rehearsal (10/29) before anything moves"
    definitions: str = (
        "gated catches are counted per target: a target counts once when any of its findings whose final status is "
        "blocker (it passed SK1's gates and the skeptic) has a repro that passes on gold when replayed at scoring, "
        "and the per-finding count is reported beside it; an invalid repro is a blocker finding whose repro does not "
        "pass on gold; the invalid-repro rate is per replayed finding, invalid repros over the blocker findings whose "
        "replay was scored, and with none scored there is no rate, so the rule is not met; a replay the sandbox "
        "could not answer twice is unscored, and while any is the rule is not judged; the founder alone may drop an "
        "unscored replay by its finding id, with a reason (score --replay-drop), which counts that finding on neither "
        "side and lists it with the reason in score.json"
    )


RULE = MoveRule()


def rule_sha256(rule: MoveRule = RULE) -> str:
    return sha256_text(canonical_json(rule.model_dump(mode="json")))


class PilotTarget(Frozen):
    target_id: str
    entry: str
    model: str
    instance_id: str
    repo: str
    label: Label
    rank: str = Field(pattern=SHA256)
    temperatures: TemperatureMode
    image: str = Field(pattern=IMAGE_REF)  # contracts.Target.image
    patch_sha256: str = Field(pattern=SHA256)
    test_runner: RunnerKind


class PilotConfig(Frozen):
    format: Literal["receipts.eval.pilot/2"] = PILOT_FORMAT
    status: Literal["provisional", "frozen"]  # the sample's status; a provisional pilot is never run
    sample_sha256: str = Field(pattern=SHA256)  # of the sample file's text
    manifest_content_sha256: str = Field(pattern=SHA256)
    plan: dict[Label, int]
    schedules: dict[TemperatureMode, Schedule]
    assignment: str
    applies_to: str
    arms: tuple[Arm, ...]
    rule: MoveRule
    rule_sha256: str = Field(pattern=SHA256)
    targets: tuple[PilotTarget, ...]
    run_file: str
    run_file_sha256: str = Field(pattern=SHA256)
    arms_file: str
    arms_file_sha256: str = Field(pattern=SHA256)


ASSIGNMENT = "per target, the same in both arms; within each label, alternating spec, meta_default, ... in rank order"
APPLIES_TO = "attacker calls only (AT1 to AT3 prompts, AT2 critic and rewrite); intake, planner and skeptic unchanged"


def assign_temperatures(items: Sequence[Item]) -> dict[tuple[str, str], TemperatureMode]:
    out: dict[tuple[str, str], TemperatureMode] = {}
    for label in sorted({i.label for i in items}):
        ordered = sorted((i for i in items if i.label is label), key=lambda i: i.rank)
        for n, i in enumerate(ordered):
            out[(i.entry, i.instance_id)] = TemperatureMode.SPEC if n % 2 == 0 else TemperatureMode.META_DEFAULT
    return out


class RunLine(Frozen):
    """One line of the pilot's run file."""

    target: Target
    temperatures: TemperatureMode


def run_file_text(targets: Sequence[Target], modes: Mapping[str, TemperatureMode]) -> str:
    """One JSON line per target, in the order given: {"target": ..., "temperatures": ...}."""
    return "".join(
        json.dumps(
            {"target": t.model_dump(mode="json"), "temperatures": modes[t.id].value},
            ensure_ascii=True,  # upstream issue text is any Unicode; escaped, the committed file stays ASCII
            separators=(",", ":"),
        )
        + "\n"
        for t in targets
    )


def read_run_file(text: str) -> list[RunLine]:
    lines = [RunLine.model_validate_json(line) for line in text.splitlines() if line.strip()]
    ids = [line.target.id for line in lines]
    if len(ids) != len(set(ids)):
        raise PilotCheckError("a target appears twice in the run file")
    return lines


def arms_file_text(arms: Sequence[Arm] = ARMS) -> str:
    return pretty_json([a.model_dump(mode="json") for a in arms])


def read_arms_file(text: str) -> tuple[Arm, ...]:
    data = json.loads(text)
    if not isinstance(data, list):
        raise PilotCheckError("the arms file is a JSON list of arms")
    return tuple(Arm.model_validate(a) for a in data)


class PilotCheckError(ValueError):
    """A pilot config that does not follow from its sample, or breaks the protocol."""


def check_arms(arms: Sequence[Arm], rule: MoveRule = RULE) -> None:
    """Two arms, the rule's two, equal in everything but the attacker model (REG-561)."""
    names = [a.name for a in arms]
    if sorted(names) != sorted({rule.baseline_arm, rule.candidate_arm}) or len(names) != 2:
        raise PilotCheckError(f"the arms are {names}, the rule compares {rule.baseline_arm} and {rule.candidate_arm}")
    a, b = arms
    if (a.step_limit, a.seed) != (b.step_limit, b.seed):
        raise PilotCheckError("the arms differ in step limit or seed; only the attacker model may differ")
    if a.attacker_model == b.attacker_model:
        raise PilotCheckError("the two arms run the same attacker model")


def build_pilot(
    sample: Sample,
    sample_text: str,
    manifest: Manifest,
    targets: Sequence[Target],
    *,
    per_label: Mapping[Label, int] = PILOT_PLAN,
    arms: Sequence[Arm] = ARMS,
) -> tuple[PilotConfig, str, str]:
    """The config, the run file's text and the arms file's text. `targets` are the pilot items' contracts.Target,
    built by `receipts.eval.targets.build_targets` from the same sample."""
    if Sample.model_validate_json(sample_text) != sample:
        raise PilotCheckError("the sample text is not this sample")
    check_arms(arms)
    records = {(r.entry, r.instance_id): r for r in manifest.records}
    items = pilot(sample, per_label)
    by_id = {t.id: t for t in targets}
    modes = assign_temperatures(items)
    rows = []
    ordered: list[Target] = []
    for i in items:
        tid = target_id(i.entry, i.instance_id)
        r = records.get((i.entry, i.instance_id))
        t = by_id.get(tid)
        if r is None or r.patch_sha256 is None:
            raise PilotCheckError(f"{i.entry} {i.instance_id}: no pulled patch in the manifest")
        if t is None or t.image is None:
            raise PilotCheckError(f"{tid}: no target with a digest-pinned image")
        if t.patch_sha != r.patch_sha256:
            raise PilotCheckError(f"{tid}: the target's patch is not the one the manifest pinned")
        ordered.append(t)
        rows.append(
            PilotTarget(
                target_id=tid,
                entry=i.entry,
                model=i.model,
                instance_id=i.instance_id,
                repo=i.repo,
                label=i.label,
                rank=i.rank,
                temperatures=modes[(i.entry, i.instance_id)],
                image=t.image,
                patch_sha256=r.patch_sha256,
                test_runner=t.test_runner,
            )
        )
    if len(by_id) != len(ordered):
        raise PilotCheckError(f"{len(by_id)} targets given for {len(ordered)} pilot items")
    run_text = run_file_text(ordered, {t.target_id: t.temperatures for t in rows})
    arms_text = arms_file_text(arms)
    cfg = PilotConfig(
        status=sample.status,
        sample_sha256=sha256_text(sample_text),
        manifest_content_sha256=sample.manifest_content_sha256,
        plan=dict(per_label),
        schedules=dict(SCHEDULES),
        assignment=ASSIGNMENT,
        applies_to=APPLIES_TO,
        arms=tuple(arms),
        rule=RULE,
        rule_sha256=rule_sha256(),
        targets=tuple(rows),
        run_file=RUN_FILE,
        run_file_sha256=sha256_text(run_text),
        arms_file=ARMS_FILE,
        arms_file_sha256=sha256_text(arms_text),
    )
    check_pilot(cfg, sample)
    return cfg, run_text, arms_text


def check_pilot(cfg: PilotConfig, sample: Sample) -> None:
    """The protocol, rechecked from the config alone plus its sample."""
    if cfg.schedules != SCHEDULES:
        raise PilotCheckError("the temperature schedules are not the protocol's")
    if cfg.rule != RULE or cfg.rule_sha256 != rule_sha256(cfg.rule):
        raise PilotCheckError("the move rule is not the one written before the run")
    check_arms(cfg.arms, cfg.rule)
    if cfg.status != sample.status:
        raise PilotCheckError(f"a {cfg.status} pilot from a {sample.status} sample")
    items = {(i.entry, i.instance_id): i for i in sample.items}
    by_label = Counter(t.label for t in cfg.targets)
    if dict(by_label) != {k: v for k, v in cfg.plan.items() if v}:
        raise PilotCheckError(f"targets per label {dict(by_label)} are not the plan {cfg.plan}")
    if len({t.instance_id for t in cfg.targets}) != len(cfg.targets):
        raise PilotCheckError("an instance appears twice in the pilot")
    for t in cfg.targets:
        i = items.get((t.entry, t.instance_id))
        if i is None or not i.core or i.side is not Side.CALIBRATION:
            raise PilotCheckError(f"{t.target_id}: not a core calibration item of the sample")
        if (i.label, i.rank, i.model, i.repo) != (t.label, t.rank, t.model, t.repo):
            raise PilotCheckError(f"{t.target_id}: does not match its sample item")
    for label in by_label:
        modes = Counter(t.temperatures for t in cfg.targets if t.label is label)
        if abs(modes[TemperatureMode.SPEC] - modes[TemperatureMode.META_DEFAULT]) > 1:
            raise PilotCheckError(f"{label}: temperatures {dict(modes)} are not split in half")
    expected = assign_temperatures([items[(t.entry, t.instance_id)] for t in cfg.targets])
    for t in cfg.targets:
        if expected[(t.entry, t.instance_id)] is not t.temperatures:
            raise PilotCheckError(f"{t.target_id}: temperatures {t.temperatures} are not the alternating assignment")


def check_files(cfg: PilotConfig, run_text: str, arms_text: str) -> None:
    """The run file and the arms file are the ones the config pins, and they say what the config says."""
    if sha256_text(run_text) != cfg.run_file_sha256:
        raise PilotCheckError(f"{cfg.run_file} is not the run file the pilot pinned")
    if sha256_text(arms_text) != cfg.arms_file_sha256:
        raise PilotCheckError(f"{cfg.arms_file} is not the arms file the pilot pinned")
    if read_arms_file(arms_text) != cfg.arms:
        raise PilotCheckError("the arms file disagrees with the pilot's arms")
    lines = read_run_file(run_text)
    want = {t.target_id: t for t in cfg.targets}
    if [line.target.id for line in lines] != [t.target_id for t in cfg.targets]:
        raise PilotCheckError("the run file's targets are not the pilot's, in its order")
    for line in lines:
        t = want[line.target.id]
        got = (line.temperatures, line.target.image, line.target.patch_sha, line.target.test_runner)
        if got != (t.temperatures, t.image, t.patch_sha256, t.test_runner):
            raise PilotCheckError(f"{t.target_id}: the run file disagrees with the pilot config")


def verify_pilot(
    cfg: PilotConfig,
    sample: Sample,
    sample_text: str,
    manifest: Manifest,
    targets: Sequence[Target],
    run_text: str,
    arms_text: str,
) -> None:
    """The pilot is exactly the one this sample, manifest and targets give: recompute it and refuse any difference."""
    if sha256_text(sample_text) != cfg.sample_sha256:
        raise PilotCheckError("the pilot was built from another sample file")
    expected, run_expected, arms_expected = build_pilot(sample, sample_text, manifest, targets, per_label=cfg.plan)
    if expected != cfg:
        raise PilotCheckError("the pilot is not the one its sample gives")
    if (run_text, arms_text) != (run_expected, arms_expected):
        raise PilotCheckError("the run file or the arms file is not the one the pilot gives")
    check_files(cfg, run_text, arms_text)
