"""The Tier A sampler (spec section 3, R2): seeded, frozen, capped, and split 40/60 by instance.

Per label (BAD, GOOD and PLAUSIBLE_WRONG), the sample holds exactly the planned count, with at most 35% of it from any
one repository and at least 3 models, none over 40%. The plan has two nested sizes: a core (100 BAD + 100 GOOD + 50
PLAUSIBLE_WRONG) and a full sample (150 + 150 + 75, used only if the pilot gate passes); the plausible-wrong stratum
is the founder's, 2026-10-27. Every core item is in the full sample, and the core is drawn before the full extension
and does not depend on it, so the same manifest gives the same core whichever plan is used.

The split is drawn first, over the pool's instances: every instance gets one side, calibration or test, with each
repository's calibration share as close to 40% as whole instances allow (largest remainder). Each label is then drawn
40% from calibration instances and 60% from test instances, with the repository and model caps applied on each side,
so the caps also hold for the whole label and for every item an instance contributes. Thresholds are tuned only on the
calibration side; the test side is reported.

Order is a pure function of the seed: a candidate's rank is sha256(seed, entry, instance), so the draw is the same on
every machine and Python version and does not depend on input order. A frozen manifest's sample has exactly one seed,
the manifest's content hash, so there is no seed to shop for; `verify_sample` re-draws the sample from the manifest
and refuses any difference, so neither a hand-picked item nor another seed gets through.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from enum import StrEnum
from typing import Literal

from pydantic import Field, model_validator

from receipts.contracts import Frozen
from receipts.eval.hashing import SHA256, sha256_text
from receipts.eval.labels import SAMPLED, Label
from receipts.eval.manifest import Manifest, content_sha256
from receipts.eval.pool import INSTANCE_ID, repo_of

SAMPLE_FORMAT: Literal["receipts.eval.sample/1"] = "receipts.eval.sample/1"
_SEP = "\x1f"


class Side(StrEnum):
    CALIBRATION = "calibration"
    TEST = "test"


class Candidate(Frozen):
    entry: str = Field(min_length=1)
    model: str = Field(min_length=1)
    instance_id: str = Field(pattern=INSTANCE_ID)
    repo: str = Field(min_length=1)
    label: Label

    @model_validator(mode="after")
    def _repo_matches(self) -> Candidate:
        if repo_of(self.instance_id) != self.repo:
            raise ValueError(f"{self.instance_id} is in {repo_of(self.instance_id)}, not {self.repo}")
        return self

    @property
    def key(self) -> tuple[str, str]:
        return (self.entry, self.instance_id)


class SamplePlan(Frozen):
    core: dict[Label, int]
    full: dict[Label, int]
    max_repo_pct: int = Field(default=35, gt=0, le=100)
    max_model_pct: int = Field(default=40, gt=0, le=100)
    min_models: int = Field(default=3, ge=1)
    calibration_pct: int = Field(default=40, gt=0, lt=100)

    @model_validator(mode="after")
    def _nested(self) -> SamplePlan:
        if set(self.core) != set(self.full):
            raise ValueError("core and full plan the same labels")
        for label, n in self.full.items():
            if label not in SAMPLED:
                raise ValueError(f"{label} is never sampled")
            if not 0 < self.core[label] <= n:
                raise ValueError(f"{label}: need 0 < core <= full, got {self.core[label]} and {n}")
        return self


# Spec section 3, with the founder's plausible-wrong stratum (2026-10-27).
SPEC_PLAN = SamplePlan(
    core={Label.BAD: 100, Label.GOOD: 100, Label.PLAUSIBLE_WRONG: 50},
    full={Label.BAD: 150, Label.GOOD: 150, Label.PLAUSIBLE_WRONG: 75},
)
# The minimum: the core alone, for when the pilot gate fails or the pool cannot supply the full sample.
CORE_PLAN = SamplePlan(core=dict(SPEC_PLAN.core), full=dict(SPEC_PLAN.core))
FROZEN_PLANS: tuple[SamplePlan, ...] = (SPEC_PLAN, CORE_PLAN)


def side_count(n: int, side: Side, calibration_pct: int) -> int:
    """Items of a label on one side: calibration gets round-half-up of the percentage, test the rest."""
    cal = (n * calibration_pct * 2 + 100) // 200
    return cal if side is Side.CALIBRATION else n - cal


def cap(n: int, pct: int) -> int:
    """The most items one repository or model may contribute to n: count <= pct% of n."""
    return n * pct // 100


class Item(Frozen):
    entry: str
    model: str
    instance_id: str
    repo: str
    label: Label
    side: Side
    core: bool
    rank: str = Field(pattern=SHA256)


class Sample(Frozen):
    format: Literal["receipts.eval.sample/1"] = SAMPLE_FORMAT
    status: Literal["provisional", "frozen"]
    seed: str = Field(min_length=1)
    manifest_content_sha256: str = Field(pattern=SHA256)
    plan: SamplePlan
    split: dict[str, Side]  # every pool instance -> its side
    items: tuple[Item, ...]

    def scope(self, *, core_only: bool) -> list[Item]:
        return [i for i in self.items if i.core or not core_only]


class InfeasibleSampleError(ValueError):
    """The candidates cannot fill the plan under the caps."""


class SampleCheckError(ValueError):
    """A sample that breaks a constraint."""


def rank(seed: str, entry: str, instance_id: str) -> str:
    return sha256_text(_SEP.join(("rank", seed, entry, instance_id)))


def split_pool(instance_ids: Iterable[str], seed: str, calibration_pct: int = 40) -> dict[str, Side]:
    """Calibration or test for every instance, stratified by repository with largest-remainder rounding."""
    ids = sorted(set(instance_ids))
    by_repo: dict[str, list[str]] = {}
    for i in ids:
        by_repo.setdefault(repo_of(i), []).append(i)
    total = (len(ids) * calibration_pct * 2 + 100) // 200
    quota = {r: len(v) * calibration_pct // 100 for r, v in by_repo.items()}
    leftover = total - sum(quota.values())
    remainders = sorted(
        by_repo,
        key=lambda r: (-(len(by_repo[r]) * calibration_pct % 100), sha256_text(_SEP.join(("split-repo", seed, r)))),
    )
    for r in remainders[:leftover]:
        quota[r] += 1
    out: dict[str, Side] = {}
    for r, members in by_repo.items():
        ordered = sorted(members, key=lambda i: sha256_text(_SEP.join(("split", seed, i))))
        for n, i in enumerate(ordered):
            out[i] = Side.CALIBRATION if n < quota[r] else Side.TEST
    return dict(sorted(out.items()))


def _greedy(
    ranked: Sequence[Candidate], n: int, plan: SamplePlan, already: Sequence[Candidate], what: str
) -> list[Candidate]:
    chosen = list(already)
    keys = {c.key for c in chosen}
    repos = Counter(c.repo for c in chosen)
    models = Counter(c.model for c in chosen)
    repo_cap, model_cap = cap(n, plan.max_repo_pct), cap(n, plan.max_model_pct)
    for c in ranked:
        if len(chosen) >= n:
            break
        if c.key in keys or repos[c.repo] >= repo_cap or models[c.model] >= model_cap:
            continue
        chosen.append(c)
        keys.add(c.key)
        repos[c.repo] += 1
        models[c.model] += 1
    if len(chosen) < n:
        raise InfeasibleSampleError(
            f"{what}: {len(chosen)} of {n} items under the caps (repo <= {repo_cap}, model <= {model_cap}) from "
            f"{len(ranked)} candidates; repos {dict(repos)}, models {dict(models)}"
        )
    return chosen


def draw(
    candidates: Iterable[Candidate],
    pool: Iterable[str],
    plan: SamplePlan,
    seed: str,
    manifest_content_sha256: str,
    *,
    status: Literal["provisional", "frozen"] = "provisional",
) -> Sample:
    pool_ids = sorted(set(pool))
    split = split_pool(pool_ids, seed, plan.calibration_pct)
    cands = list(candidates)
    seen: set[tuple[str, str]] = set()
    for c in cands:
        if c.key in seen:
            raise ValueError(f"duplicate candidate {c.key}")
        seen.add(c.key)
        if c.instance_id not in split:
            raise ValueError(f"candidate {c.key} is not on the pool")
    items: list[Item] = []
    for label in sorted(plan.full):
        for side in (Side.CALIBRATION, Side.TEST):
            ranked = sorted(
                (c for c in cands if c.label is label and split[c.instance_id] is side),
                key=lambda c: rank(seed, c.entry, c.instance_id),
            )
            core_n = side_count(plan.core[label], side, plan.calibration_pct)
            full_n = side_count(plan.full[label], side, plan.calibration_pct)
            core = _greedy(ranked, core_n, plan, [], f"{label} {side} core")
            full = _greedy(ranked, full_n, plan, core, f"{label} {side} full")
            core_keys = {c.key for c in core}
            for c in full:
                items.append(
                    Item(
                        entry=c.entry,
                        model=c.model,
                        instance_id=c.instance_id,
                        repo=c.repo,
                        label=c.label,
                        side=side,
                        core=c.key in core_keys,
                        rank=rank(seed, c.entry, c.instance_id),
                    )
                )
    sample = Sample(
        status=status,
        seed=seed,
        manifest_content_sha256=manifest_content_sha256,
        plan=plan,
        split=split,
        items=tuple(sorted(items, key=lambda i: (i.label, i.side, i.rank))),
    )
    check_sample(sample)
    return sample


def _check_scope(items: Sequence[Item], n: int, plan: SamplePlan, what: str) -> None:
    if len(items) != n:
        raise SampleCheckError(f"{what}: {len(items)} items, planned {n}")
    repo_cap, model_cap = cap(n, plan.max_repo_pct), cap(n, plan.max_model_pct)
    for repo, k in Counter(i.repo for i in items).items():
        if k > repo_cap:
            raise SampleCheckError(f"{what}: {k} items from {repo}, over {plan.max_repo_pct}% of {n} ({repo_cap})")
    models = Counter(i.model for i in items)
    for model, k in models.items():
        if k > model_cap:
            raise SampleCheckError(f"{what}: {k} items from {model}, over {plan.max_model_pct}% of {n} ({model_cap})")
    if len(models) < plan.min_models:
        raise SampleCheckError(f"{what}: {len(models)} models, need at least {plan.min_models}")


def check_split(split: Mapping[str, Side], seed: str, calibration_pct: int) -> None:
    """The split is the seeded, stratified one: recomputing it from the seed gives the same sides."""
    expected = split_pool(split, seed, calibration_pct)
    if dict(split) != expected:
        moved = sorted(i for i in split if split[i] != expected.get(i))
        raise SampleCheckError(f"split does not follow from the seed; {len(moved)} instances differ, e.g. {moved[:3]}")
    by_repo: dict[str, list[Side]] = {}
    for i, s in split.items():
        by_repo.setdefault(repo_of(i), []).append(s)
    for repo, sides in by_repo.items():
        cal = sum(1 for s in sides if s is Side.CALIBRATION)
        if abs(cal * 100 - len(sides) * calibration_pct) >= 100:  # within one instance of the exact share
            raise SampleCheckError(
                f"{repo}: {cal} of {len(sides)} instances in calibration, target {calibration_pct}% of {len(sides)}"
            )


def check_sample(sample: Sample) -> None:
    """Every constraint, rechecked from the items alone. A sample that fails here is never used."""
    plan = sample.plan
    check_split(sample.split, sample.seed, plan.calibration_pct)
    keys = [(i.entry, i.instance_id) for i in sample.items]
    if len(keys) != len(set(keys)):
        raise SampleCheckError("an (entry, instance) item appears twice")
    for i in sample.items:
        if i.label not in plan.full:
            raise SampleCheckError(f"{i.entry} {i.instance_id}: label {i.label} is not in the plan")
        if sample.split.get(i.instance_id) is not i.side:
            raise SampleCheckError(f"{i.entry} {i.instance_id}: side {i.side} is not its instance's side")
        if i.rank != rank(sample.seed, i.entry, i.instance_id):
            raise SampleCheckError(f"{i.entry} {i.instance_id}: rank does not follow from the seed")
        if repo_of(i.instance_id) != i.repo:
            raise SampleCheckError(f"{i.instance_id}: repo {i.repo} is wrong")
    for label in plan.full:
        for core_only, sizes in ((True, plan.core), (False, plan.full)):
            scope = "core" if core_only else "full"
            items = [i for i in sample.scope(core_only=core_only) if i.label is label]
            _check_scope(items, sizes[label], plan, f"{label} {scope}")
            for side in (Side.CALIBRATION, Side.TEST):
                part = [i for i in items if i.side is side]
                _check_scope(
                    part, side_count(sizes[label], side, plan.calibration_pct), plan, f"{label} {scope} {side}"
                )


# The 10/28 pilot: 20 targets in the core's 2 : 2 : 1 proportion.
PILOT_PLAN: dict[Label, int] = {Label.BAD: 8, Label.GOOD: 8, Label.PLAUSIBLE_WRONG: 4}


def pilot(sample: Sample, per_label: Mapping[Label, int] = PILOT_PLAN) -> list[Item]:
    """The pilot targets: core calibration items only, never a test item, taken in rank order per label, with no
    instance twice and the sample's repository and model caps applied over the whole pilot."""
    plan = sample.plan
    total = sum(per_label.values())
    repo_cap, model_cap = max(1, cap(total, plan.max_repo_pct)), max(1, cap(total, plan.max_model_pct))
    used: set[str] = set()
    repos: Counter[str] = Counter()
    models: Counter[str] = Counter()
    out: list[Item] = []
    for label in sorted(per_label):
        want = per_label[label]
        pool = sorted(
            (i for i in sample.items if i.core and i.side is Side.CALIBRATION and i.label is label),
            key=lambda i: i.rank,
        )
        got = 0
        for i in pool:
            if got == want:
                break
            if i.instance_id in used or repos[i.repo] >= repo_cap or models[i.model] >= model_cap:
                continue
            out.append(i)
            used.add(i.instance_id)
            repos[i.repo] += 1
            models[i.model] += 1
            got += 1
        if got < want:
            raise InfeasibleSampleError(
                f"pilot: {got} of {want} {label} items from {len(pool)} core calibration items under the caps "
                f"(repo <= {repo_cap}, model <= {model_cap}, one item per instance)"
            )
    return out


def candidates_from_manifest(manifest: Manifest) -> list[Candidate]:
    """One candidate per record with a sampled label, minus every instance whose gold failed under runsc (R24).

    A draft draws on provisional labels; a frozen manifest on final labels only, and every one of its candidates must
    have passed the runsc check: an unchecked instance in a frozen manifest is refused, never sampled."""
    frozen = manifest.status == "frozen"
    models = {e.key: e.model_id for e in manifest.entries}
    out = []
    for r in manifest.records:
        label = r.final if frozen else r.provisional
        if label not in SAMPLED or r.gold_runsc is False:
            continue
        if frozen and r.gold_runsc is not True:
            raise SampleCheckError(f"{r.entry} {r.instance_id}: a frozen manifest with an instance runsc never checked")
        out.append(Candidate(entry=r.entry, model=models[r.entry], instance_id=r.instance_id, repo=r.repo, label=label))
    return out


def pool_from_manifest(manifest: Manifest, pool_ids: Iterable[str]) -> list[str]:
    """The pool minus instances whose gold failed under runsc (spec R24: dropped and logged, never sampled)."""
    ids = set(pool_ids)
    status: dict[str, bool | None] = {}
    for r in manifest.records:
        status.setdefault(r.instance_id, r.gold_runsc)
        if status[r.instance_id] != r.gold_runsc:
            raise SampleCheckError(f"{r.instance_id}: records disagree on its runsc verdict")
    if manifest.status == "frozen":
        unchecked = sorted(i for i in ids if status.get(i) is None)
        if unchecked:
            raise SampleCheckError(f"a frozen manifest with {len(unchecked)} pool instances runsc never checked")
    dropped = {i for i, ok in status.items() if ok is False}
    return sorted(ids - dropped)


def pool_ids_for(manifest: Manifest, pool_text: str) -> list[str]:
    """The pool list, refused unless it is the one the manifest pinned."""
    name = manifest.pool_choice or "swe_abs_251"
    record = next(p for p in manifest.pools if p.name == name)
    if record.list_sha256 is None or sha256_text(pool_text) != record.list_sha256:
        raise SampleCheckError(f"the pool list is not the {name} list the manifest pinned")
    return pool_text.split()


def draw_from_manifest(manifest: Manifest, pool_text: str, plan: SamplePlan, *, seed: str | None = None) -> Sample:
    """The seeded draw from a manifest. A frozen manifest takes no seed but its own content hash and only a plan the
    spec allows (the full sample or its core)."""
    digest = content_sha256(manifest)
    frozen = manifest.status == "frozen"
    if frozen:
        if seed not in (None, digest):
            raise SampleCheckError("a frozen manifest's sample is seeded by the manifest's content hash, nothing else")
        if plan not in FROZEN_PLANS:
            raise SampleCheckError("a frozen sample uses the spec's plan or its core, nothing else")
    pool_ids = pool_from_manifest(manifest, pool_ids_for(manifest, pool_text))
    return draw(
        candidates_from_manifest(manifest),
        pool_ids,
        plan,
        seed or digest,
        digest,
        status="frozen" if frozen else "provisional",
    )


def verify_sample(sample: Sample, manifest: Manifest, pool_text: str) -> None:
    """The sample is exactly the seeded draw from this manifest and pool: recompute it and refuse any difference."""
    check_sample(sample)
    digest = content_sha256(manifest)
    if sample.manifest_content_sha256 != digest:
        raise SampleCheckError("the sample was drawn from another manifest")
    if (sample.status == "frozen") != (manifest.status == "frozen"):
        raise SampleCheckError(f"a {sample.status} sample from a {manifest.status} manifest")
    if sample.status == "frozen" and sample.seed != digest:
        raise SampleCheckError("a frozen sample's seed must be its manifest's content hash")
    expected = draw_from_manifest(
        manifest, pool_text, sample.plan, seed=None if sample.status == "frozen" else sample.seed
    )
    if expected != sample:
        mine = {(i.entry, i.instance_id, i.core) for i in sample.items}
        theirs = {(i.entry, i.instance_id, i.core) for i in expected.items}
        raise SampleCheckError(
            f"the sample is not the seeded draw from this manifest: {len(mine - theirs)} items differ, "
            f"e.g. {sorted(mine - theirs)[:2]}"
        )
