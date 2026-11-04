"""The Tier A pool: the SWE-bench Verified instances that SWE-ABS strengthened (spec section 3, R1).

SWE-ABS (Yu et al., arXiv 2603.00520v1) calls an instance strengthened when at least one patch from its top-30
leaderboard agents passes the original tests and fails the SWE-ABS tests. Every number here is arXiv v1's, which the
published per-patch verdict file reproduces (251 strengthened of 500); a later version of the paper reports other
figures, so any SWE-ABS figure this project cites names v1 (spec section 4 bars an unqualified one). The two
candidate pools are:

- ``swe_abs_251``: the 251 strengthened instances, with the 53 overfitting test patches corrected by the authors
  (arXiv v1's headline configuration, Table 2 row "SWE-ABS" and Table 6; the founder's choice, 2026-10-27);
- ``swe_abs_206``: SWE-ABS*, the strengthened instances once the 53 overfitting instances are excluded (Table 2 row
  "SWE-ABS*"). 251 - 206 = 45 of the 53 are strengthened.

The 251 list is derived from the published per-patch verdicts and checked against every number the paper prints for
it. The 206 list needs the identities of the 53 overfitting instances, which the paper, the repository and the
dataset do not publish; `derive_206` takes that list as an input and checks its size.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from collections.abc import Iterable, Mapping
from typing import Literal

from pydantic import Field

from receipts.contracts import Frozen

INSTANCE_ID = r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+$"
_INSTANCE_RE = re.compile(INSTANCE_ID)

Status = Literal["passed", "failed", "not_run", "error"]


class Verdict(Frozen):
    """One record of SWE-ABS's data/swe_abs_per_instance_verdicts.json."""

    agent_id: str = Field(min_length=1)
    agent_name: str
    instance_id: str = Field(pattern=INSTANCE_ID)
    original_tests_passed: bool | None
    swe_abs_tests_passed: bool | None
    re_evaluation_status: Status
    source_category: Literal["pass", "fail", "init_fail", "error"]


class VerdictFile(Frozen):
    metadata: dict[str, object]
    records: tuple[Verdict, ...]


def load_verdicts(raw: bytes | str) -> VerdictFile:
    return VerdictFile.model_validate(json.loads(raw))


def load_agent_keys(raw: bytes | str) -> frozenset[str]:
    """data/key_map_top30.json maps agent keys to display names; the keys are the agent set."""
    data = json.loads(raw)
    if not isinstance(data, dict) or not all(isinstance(k, str) for k in data):
        raise ValueError("key map must be a JSON object keyed by agent id")
    return frozenset(data)


def repo_of(instance_id: str) -> str:
    """`django__django-11099` -> `django/django`."""
    if not _INSTANCE_RE.match(instance_id):
        raise ValueError(f"{instance_id!r} is not a SWE-bench instance id")
    owner, rest = instance_id.split("__", 1)
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


class PaperNumbers(Frozen):
    """What the paper prints for a pool; a derivation that disagrees with any of them is refused."""

    strengthened: int = Field(gt=0)
    passing_patches: int = Field(gt=0)
    killed_patches: int = Field(ge=0)  # the paper counts re-evaluation errors as kills: 2,177 failed + 7 errors
    per_repo: dict[str, int]


# SWE-ABS v1 (arXiv 2603.00520v1, read 2026-10-27): Table 2 row "SWE-ABS" and Table 6.
SWE_ABS_V1 = PaperNumbers(
    strengthened=251,
    passing_patches=11041,
    killed_patches=2184,
    per_repo={
        "django/django": 102,
        "sympy/sympy": 34,
        "sphinx-doc/sphinx": 35,
        "matplotlib/matplotlib": 22,
        "scikit-learn/scikit-learn": 9,
        "astropy/astropy": 18,
        "pydata/xarray": 9,
        "pytest-dev/pytest": 9,
        "pylint-dev/pylint": 5,
        "psf/requests": 5,
        "mwaskom/seaborn": 2,
        "pallets/flask": 1,
    },
)
SWE_ABS_STAR_V1 = 206  # Table 2 row "SWE-ABS*"
OVERFIT_V1 = 53  # section 4.4.1


class PoolMismatchError(ValueError):
    """A derived pool that does not reproduce the paper's numbers."""


class Pool(Frozen):
    name: Literal["swe_abs_251", "swe_abs_206"]
    instance_ids: tuple[str, ...]
    per_repo: dict[str, int]
    rule: str
    checks: dict[str, str]  # what was checked against the paper, as "derived == printed"

    @property
    def size(self) -> int:
        return len(self.instance_ids)


def strengthened(verdicts: Iterable[Verdict], agents: frozenset[str]) -> frozenset[str]:
    """Instances where an agent in `agents` passes the original tests and fails the SWE-ABS tests."""
    return frozenset(
        v.instance_id
        for v in verdicts
        if v.agent_id in agents and v.re_evaluation_status == "failed" and v.original_tests_passed is True
    )


def per_repo(instance_ids: Iterable[str]) -> dict[str, int]:
    return dict(sorted(Counter(repo_of(i) for i in instance_ids).items()))


def derive_251(file: VerdictFile, agents: frozenset[str], expect: PaperNumbers = SWE_ABS_V1) -> Pool:
    missing = agents - {v.agent_id for v in file.records}
    if missing:
        raise PoolMismatchError(f"agents with no verdicts: {sorted(missing)}")
    mine = [v for v in file.records if v.agent_id in agents]
    ids = strengthened(mine, agents)
    passing = sum(1 for v in mine if v.original_tests_passed is True)
    killed = sum(1 for v in mine if v.original_tests_passed is True and v.re_evaluation_status in ("failed", "error"))
    repos = per_repo(ids)
    problems = []
    if len(ids) != expect.strengthened:
        problems.append(f"strengthened {len(ids)} != {expect.strengthened}")
    if passing != expect.passing_patches:
        problems.append(f"passing patches {passing} != {expect.passing_patches}")
    if killed != expect.killed_patches:
        problems.append(f"killed patches {killed} != {expect.killed_patches}")
    if repos != dict(sorted(expect.per_repo.items())):
        problems.append(f"per-repo {repos} != {dict(sorted(expect.per_repo.items()))}")
    if problems:
        raise PoolMismatchError("; ".join(problems))
    return Pool(
        name="swe_abs_251",
        instance_ids=tuple(sorted(ids)),
        per_repo=repos,
        rule=(
            "instance_id where some agent in data/key_map_top30.json has re_evaluation_status == 'failed' and "
            "original_tests_passed == true in data/swe_abs_per_instance_verdicts.json"
        ),
        checks={
            "strengthened": f"{len(ids)} == {expect.strengthened}",
            "passing_patches": f"{passing} == {expect.passing_patches}",
            "killed_patches_failed_plus_error": f"{killed} == {expect.killed_patches}",
            "per_repo": "matches Table 6 for all 12 repositories",
        },
    )


def derive_206(
    pool_251: Pool, overfit_ids: Iterable[str], *, overfit: int = OVERFIT_V1, expect: int = SWE_ABS_STAR_V1
) -> Pool:
    """SWE-ABS*: drop the overfitting instances. Needs the authors' list of them; refuses any list of the wrong size."""
    if pool_251.name != "swe_abs_251":
        raise PoolMismatchError("SWE-ABS* is derived from the 251 pool")
    overfit_set = frozenset(overfit_ids)
    for i in overfit_set:
        repo_of(i)
    if len(overfit_set) != overfit:
        raise PoolMismatchError(f"{len(overfit_set)} overfitting instances given, the paper reports {overfit}")
    ids = frozenset(pool_251.instance_ids) - overfit_set
    if len(ids) != expect:
        raise PoolMismatchError(f"excluding them leaves {len(ids)} strengthened instances, the paper reports {expect}")
    return Pool(
        name="swe_abs_206",
        instance_ids=tuple(sorted(ids)),
        per_repo=per_repo(ids),
        rule="swe_abs_251 minus the 53 instances the authors found to have overfitting tests (section 4.4.1)",
        checks={"overfit": f"{len(overfit_set)} == {overfit}", "remaining": f"{len(ids)} == {expect}"},
    )


def universe(file: VerdictFile) -> tuple[str, ...]:
    """Every instance the verdict file covers (all 500 of SWE-bench Verified)."""
    return tuple(sorted({v.instance_id for v in file.records}))


def verdict_index(file: VerdictFile) -> Mapping[tuple[str, str], Verdict]:
    """(agent_id, instance_id) -> verdict; refuses duplicates."""
    out: dict[tuple[str, str], Verdict] = {}
    for v in file.records:
        key = (v.agent_id, v.instance_id)
        if key in out:
            raise ValueError(f"duplicate verdict for {key}")
        out[key] = v
    return out
