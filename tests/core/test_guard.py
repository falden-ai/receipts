"""IN3, the intake guard, as pure functions. Each rule is fed the pull request it must skip, and the head branch's own
receipts.yml is planted to try to loosen the check."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from receipts.core.guard import (
    ConfigFile,
    Limits,
    PullRequest,
    Run,
    SkipReason,
    decide,
    runs_this_month,
)

BASE, HEAD = "b" * 40, "c" * 40
NOW = datetime(2026, 10, 28, 15, 0, tzinfo=UTC)


def pr(**changes: object) -> PullRequest:
    fields: dict[str, object] = {
        "installation_id": 7,
        "repo": "octo/widgets",
        "number": 42,
        "draft": False,
        "base_sha": BASE,
        "head_sha": HEAD,
        "head_repo": "octo/widgets",
    }
    fields.update(changes)
    return PullRequest.model_validate(fields)


def runs(count: int, *, repo: str = "octo/widgets", installation: int = 7, at: datetime = NOW) -> list[Run]:
    return [Run(at=at - timedelta(minutes=n), installation_id=installation, repo=repo) for n in range(count)]


def test_a_ready_pull_request_from_the_same_repository_runs() -> None:
    decision = decide(pr(), config=None, runs=[], now=NOW)
    assert (decision.run, decision.reason, decision.comment) == (True, None, "")


def test_config_comes_from_the_base_commit_only() -> None:
    # The head branch's receipts.yml lifts the cap and turns Receipts on; neither counts.
    head = ConfigFile(sha=HEAD, data={"guard": {"enabled": True, "monthly_runs": 1_000_000}})
    capped = decide(pr(), config=head, runs=runs(200), now=NOW)
    assert (capped.run, capped.reason) == (False, SkipReason.REPO_CAP)
    assert capped.notes == (f"receipts.yml read at {HEAD[:12]}, not the base commit {BASE[:12]}: ignored",)

    for value in (False, "no", "off"):
        off = ConfigFile(sha=BASE, data={"guard": {"enabled": value}})
        decision = decide(pr(), config=off, runs=[], now=NOW)
        assert (decision.run, decision.reason) == (False, SkipReason.DISABLED)
        assert "base branch" in decision.comment

    lowered = ConfigFile(sha=BASE, data={"guard": {"monthly_runs": 3}})
    assert decide(pr(), config=lowered, runs=runs(3), now=NOW).reason is SkipReason.REPO_CAP
    assert decide(pr(), config=lowered, runs=runs(2), now=NOW).run
    raised = ConfigFile(sha=BASE, data={"guard": {"monthly_runs": 10_000}})  # never past the operator's 200
    assert decide(pr(), config=raised, runs=runs(200), now=NOW).reason is SkipReason.REPO_CAP


@pytest.mark.parametrize("section", [{"enabled": "maybe"}, {"enabld": False}, True, {"monthly_runs": -1}])
def test_an_invalid_guard_section_is_a_skip_never_a_run(section: object) -> None:
    decision = decide(pr(), config=ConfigFile(sha=BASE, data={"guard": section}), runs=[], now=NOW)
    assert (decision.run, decision.reason) == (False, SkipReason.INVALID_CONFIG)
    assert decision.detail.startswith("guard")
    assert "invalid" in decision.comment


def test_drafts_and_forks_are_skipped() -> None:
    draft = decide(pr(draft=True), config=None, runs=[], now=NOW)
    assert (draft.run, draft.reason) == (False, SkipReason.DRAFT)
    assert "ready for review" in draft.comment
    for head in ("someone/widgets", None):
        fork = decide(pr(head_repo=head), config=None, runs=[], now=NOW)
        assert (fork.run, fork.reason) == (False, SkipReason.FORK)
    assert decide(pr(head_repo="Octo/Widgets"), config=None, runs=[], now=NOW).run  # names compare without case


def test_the_monthly_caps_count_this_utc_month_per_repository_and_per_installation() -> None:
    limits = Limits(repo_monthly_runs=5, installation_monthly_runs=8)
    last_month = runs(50, at=datetime(2026, 9, 30, 23, 59, tzinfo=UTC))
    assert decide(pr(), config=None, runs=last_month, now=NOW, limits=limits).run  # last month never counts

    full = decide(pr(), config=None, runs=runs(5), now=NOW, limits=limits)
    assert (full.reason, full.detail) == (SkipReason.REPO_CAP, "5 of 5 runs used this month")
    assert "5 runs" in full.comment

    others = runs(4) + runs(4, repo="octo/gadgets")  # 4 here, 8 across the installation
    busy = decide(pr(), config=None, runs=others, now=NOW, limits=limits)
    assert (busy.reason, busy.detail) == (SkipReason.INSTALLATION_CAP, "8 of 8 runs used this month")
    elsewhere = runs(4) + runs(4, repo="octo/gadgets", installation=9)  # another installation's runs do not count
    assert decide(pr(), config=None, runs=elsewhere, now=NOW, limits=limits).run


def test_month_boundaries_are_utc_and_times_carry_their_zone() -> None:
    late = datetime(2026, 10, 31, 23, 30, tzinfo=UTC)
    early_next = Run(at=datetime(2026, 11, 1, 0, 30, tzinfo=UTC), installation_id=7, repo="octo/widgets")
    assert runs_this_month([early_next], late) == 0
    assert runs_this_month([early_next], late + timedelta(hours=1)) == 1
    with pytest.raises(ValueError, match="time zone"):
        runs_this_month([], datetime(2026, 10, 28))
    with pytest.raises(ValidationError):
        Run(at=datetime(2026, 10, 28), installation_id=7, repo="octo/widgets")


def test_runs_count_against_their_repository_whatever_the_case_of_its_name() -> None:
    # GitHub names compare without case: runs the App logged as Octo/Widgets count against octo/widgets's cap.
    limits = Limits(repo_monthly_runs=3, installation_monthly_runs=100)
    logged = runs(3, repo="Octo/Widgets")
    full = decide(pr(), config=None, runs=logged, now=NOW, limits=limits)
    assert (full.run, full.reason, full.detail) == (False, SkipReason.REPO_CAP, "3 of 3 runs used this month")
    assert runs_this_month(logged, NOW, repo="OCTO/widgets") == 3
    assert runs_this_month(logged, NOW, repo="octo/gadgets") == 0
