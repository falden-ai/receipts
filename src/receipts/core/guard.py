"""IN3, the intake guard: whether the GitHub App runs Receipts on a pull request, decided by pure functions over what
the App already knows (R17). The App (SU1) wires it on 10/30.

- Config comes from the base branch only. A ConfigFile names the commit it was read at, and one read anywhere but the
  pull request's base commit is ignored, so a pull request can never loosen its own check by editing receipts.yml.
- Drafts are skipped, and so are pull requests from forks, including one whose fork was deleted.
- Monthly caps per repository and per installation count the runs the App started in the current UTC calendar month.
  A repository's config may lower its own cap, never raise it past the operator's.
- A config whose guard section cannot be read is a skip, not a run: a maintainer who wrote 'enabled: maybe' did not mean
  on. ('no', 'off' and 'false' read as off.)

Every skip carries its reason, a detail and the one comment the App posts.
"""

from __future__ import annotations

from collections.abc import Iterable
from datetime import UTC, datetime
from enum import StrEnum
from typing import Final

from pydantic import AwareDatetime, Field, ValidationError

from receipts.contracts import SHA, Frozen

DEFAULT_REPO_MONTHLY_RUNS: Final = 200
DEFAULT_INSTALLATION_MONTHLY_RUNS: Final = 1_000
CONFIG_SECTION: Final = "guard"  # the section of receipts.yml the guard reads

COMMENTS: Final = {
    "disabled": "Falden Receipts is turned off for this repository by receipts.yml on the base branch.",
    "invalid_config": "Falden Receipts did not run: the guard section of receipts.yml on the base branch is invalid "
    "({detail}).",
    "draft": "Falden Receipts skips draft pull requests. It runs once this one is marked ready for review.",
    "fork": "Falden Receipts does not run on pull requests from forks.",
    "repo_monthly_cap": "Falden Receipts did not run: this repository has used its {cap} runs for this month.",
    "installation_monthly_cap": "Falden Receipts did not run: this installation has used its {cap} runs for this "
    "month.",
}


class SkipReason(StrEnum):
    DISABLED = "disabled"
    INVALID_CONFIG = "invalid_config"
    DRAFT = "draft"
    FORK = "fork"
    REPO_CAP = "repo_monthly_cap"
    INSTALLATION_CAP = "installation_monthly_cap"


class GuardConfig(Frozen):
    """The guard section of receipts.yml, as the base branch has it."""

    enabled: bool = True
    monthly_runs: int | None = Field(default=None, ge=0)  # this repository's own cap; None keeps the operator's


class Limits(Frozen):
    """The operator's caps. A repository's config can only lower its own."""

    repo_monthly_runs: int = Field(default=DEFAULT_REPO_MONTHLY_RUNS, ge=0)
    installation_monthly_runs: int = Field(default=DEFAULT_INSTALLATION_MONTHLY_RUNS, ge=0)


class ConfigFile(Frozen):
    """receipts.yml as the App read it: the commit it was read at and its parsed content (None when there is none)."""

    sha: str = Field(pattern=SHA)
    data: dict[str, object] | None = None


class PullRequest(Frozen):
    installation_id: int = Field(gt=0)
    repo: str = Field(min_length=3)  # owner/name of the base repository
    number: int = Field(gt=0)
    draft: bool
    base_sha: str = Field(pattern=SHA)
    head_sha: str = Field(pattern=SHA)
    head_repo: str | None  # owner/name of the head repository; None when it was deleted


class Run(Frozen):
    """One run the App started."""

    at: AwareDatetime
    installation_id: int = Field(gt=0)
    repo: str = Field(min_length=3)


class GuardDecision(Frozen):
    run: bool
    reason: SkipReason | None = None
    detail: str = ""
    comment: str = ""  # what the App posts, once, on a skip
    notes: tuple[str, ...] = ()


def base_config(pr: PullRequest, file: ConfigFile | None) -> tuple[GuardConfig | None, tuple[str, ...], str]:
    """The guard config the base branch sets, the notes on what was ignored, and, when the section is invalid, None
    with why."""
    if file is None or file.data is None:
        return GuardConfig(), (), ""
    if file.sha != pr.base_sha:
        note = f"receipts.yml read at {file.sha[:12]}, not the base commit {pr.base_sha[:12]}: ignored"
        return GuardConfig(), (note,), ""
    section = file.data.get(CONFIG_SECTION)
    if section is None:
        return GuardConfig(), (), ""
    try:
        return GuardConfig.model_validate(section), (), ""
    except ValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        where = ".".join(str(part) for part in first["loc"])
        path = f"{CONFIG_SECTION}.{where}" if where else CONFIG_SECTION
        return None, (), f"{path}: {first['msg']}"[:200]


def runs_this_month(
    runs: Iterable[Run], now: datetime, *, repo: str | None = None, installation_id: int | None = None
) -> int:
    """Runs started in now's UTC calendar month, for one repository or installation when named."""
    if now.tzinfo is None:
        raise ValueError("now must carry its time zone")
    month = now.astimezone(UTC).strftime("%Y-%m")
    return sum(
        1
        for run in runs
        if run.at.astimezone(UTC).strftime("%Y-%m") == month
        and (repo is None or run.repo.lower() == repo.lower())
        and (installation_id is None or run.installation_id == installation_id)
    )


def decide(
    pr: PullRequest,
    *,
    config: ConfigFile | None,
    runs: Iterable[Run],
    now: datetime,
    limits: Limits | None = None,
) -> GuardDecision:
    """Run or skip, and why, in this order: the base branch's config, a draft, a fork, the repository's monthly cap,
    the installation's."""
    limits = limits if limits is not None else Limits()
    guard, notes, invalid = base_config(pr, config)
    if guard is None:
        return _skip(SkipReason.INVALID_CONFIG, invalid, notes, detail=invalid)
    if not guard.enabled:
        return _skip(SkipReason.DISABLED, "receipts.yml on the base branch turns Receipts off", notes)
    if pr.draft:
        return _skip(SkipReason.DRAFT, "a draft pull request", notes)
    if pr.head_repo is None or pr.head_repo.lower() != pr.repo.lower():
        head = pr.head_repo or "a deleted repository"
        return _skip(SkipReason.FORK, f"the head branch is in {head}, not {pr.repo}", notes)
    history = list(runs)
    repo_cap = limits.repo_monthly_runs
    if guard.monthly_runs is not None:
        repo_cap = min(repo_cap, guard.monthly_runs)
    used = runs_this_month(history, now, repo=pr.repo)
    if used >= repo_cap:
        return _skip(SkipReason.REPO_CAP, f"{used} of {repo_cap} runs used this month", notes, cap=repo_cap)
    installed = runs_this_month(history, now, installation_id=pr.installation_id)
    if installed >= limits.installation_monthly_runs:
        cap = limits.installation_monthly_runs
        return _skip(SkipReason.INSTALLATION_CAP, f"{installed} of {cap} runs used this month", notes, cap=cap)
    return GuardDecision(run=True, notes=notes)


def _skip(reason: SkipReason, why: str, notes: tuple[str, ...], *, detail: str = "", cap: int = 0) -> GuardDecision:
    comment = COMMENTS[reason.value].format(detail=detail, cap=f"{cap:,}")
    return GuardDecision(run=False, reason=reason, detail=why, comment=comment, notes=notes)
