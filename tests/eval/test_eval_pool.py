"""The pool derivation reproduces the paper's numbers or refuses; synthetic verdicts stand in for the real file."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from receipts.eval.pool import (
    SWE_ABS_V1,
    PaperNumbers,
    PoolMismatchError,
    Verdict,
    VerdictFile,
    derive_206,
    derive_251,
    load_agent_keys,
    load_verdicts,
    repo_of,
    strengthened,
    verdict_index,
)


def v(agent: str, iid: str, status: str, original: bool | None = True) -> Verdict:
    category = {"passed": "pass", "failed": "fail", "not_run": "init_fail", "error": "error"}[status]
    return Verdict(
        agent_id=agent,
        agent_name=agent,
        instance_id=iid,
        original_tests_passed=original,
        swe_abs_tests_passed={"passed": True, "failed": False}.get(status),
        re_evaluation_status=status,
        source_category=category,
    )


def synthetic() -> tuple[VerdictFile, frozenset[str], PaperNumbers]:
    records = (
        v("a1", "django__django-1", "failed"),
        v("a2", "django__django-1", "passed"),
        v("a1", "django__django-2", "passed"),
        v("a2", "django__django-2", "error"),
        v("a1", "sympy__sympy-3", "passed"),
        v("a2", "sympy__sympy-3", "failed"),
        v("a1", "psf__requests-4", "not_run", original=False),
        v("a2", "psf__requests-4", "passed"),
        v("outsider", "psf__requests-4", "failed"),  # not one of the paper's agents: never counts
    )
    expect = PaperNumbers(
        strengthened=2, passing_patches=7, killed_patches=3, per_repo={"django/django": 1, "sympy/sympy": 1}
    )
    return VerdictFile(metadata={}, records=records), frozenset({"a1", "a2"}), expect


def test_the_251_rule_counts_only_the_papers_agents() -> None:
    file, agents, expect = synthetic()
    assert strengthened(file.records, agents) == {"django__django-1", "sympy__sympy-3"}
    pool = derive_251(file, agents, expect)
    assert pool.instance_ids == ("django__django-1", "sympy__sympy-3")
    assert pool.per_repo == {"django/django": 1, "sympy/sympy": 1}


def test_plant_one_more_failing_patch_breaks_the_paper_check() -> None:
    file, agents, expect = synthetic()
    planted = VerdictFile(metadata={}, records=(*file.records, v("a1", "psf__requests-5", "failed")))
    with pytest.raises(PoolMismatchError, match="strengthened 3 != 2"):
        derive_251(planted, agents, expect)


def test_plant_a_per_repo_count_off_by_one_is_refused() -> None:
    file, agents, expect = synthetic()
    wrong = expect.model_copy(update={"per_repo": {"django/django": 2}})
    with pytest.raises(PoolMismatchError, match="per-repo"):
        derive_251(file, agents, wrong)


def test_plant_a_missing_agent_is_refused() -> None:
    file, agents, expect = synthetic()
    with pytest.raises(PoolMismatchError, match="agents with no verdicts"):
        derive_251(file, agents | {"ghost"}, expect)


def test_the_published_numbers_are_the_papers() -> None:
    assert SWE_ABS_V1.strengthened == 251 == sum(SWE_ABS_V1.per_repo.values())
    assert (SWE_ABS_V1.passing_patches, SWE_ABS_V1.killed_patches) == (11041, 2184)


def test_the_206_needs_exactly_the_authors_list() -> None:
    file, agents, expect = synthetic()
    pool = derive_251(file, agents, expect)
    out = derive_206(pool, ["sympy__sympy-3"], overfit=1, expect=1)
    assert out.instance_ids == ("django__django-1",)
    with pytest.raises(PoolMismatchError, match="overfitting instances given"):
        derive_206(pool, [], overfit=1, expect=1)
    with pytest.raises(PoolMismatchError, match="leaves 2"):
        derive_206(pool, ["psf__requests-9"], overfit=1, expect=1)


def test_repo_of_reads_swe_bench_ids_and_refuses_anything_else() -> None:
    assert repo_of("django__django-11099") == "django/django"
    assert repo_of("scikit-learn__scikit-learn-10297") == "scikit-learn/scikit-learn"
    assert repo_of("pytest-dev__pytest-5262") == "pytest-dev/pytest"
    for bad in ("django-11099", "../etc__passwd-1", "django__django"):
        with pytest.raises(ValueError):
            repo_of(bad)


def test_plant_an_unknown_status_in_the_verdict_file_is_refused() -> None:
    raw = json.dumps(
        {
            "metadata": {},
            "records": [
                {
                    "agent_id": "a",
                    "agent_name": "a",
                    "instance_id": "django__django-1",
                    "original_tests_passed": True,
                    "swe_abs_tests_passed": True,
                    "re_evaluation_status": "maybe",
                    "source_category": "pass",
                }
            ],
        }
    )
    with pytest.raises(ValidationError):
        load_verdicts(raw)


def test_plant_a_duplicate_verdict_is_refused() -> None:
    file, _, _ = synthetic()
    with pytest.raises(ValueError, match="duplicate verdict"):
        verdict_index(VerdictFile(metadata={}, records=(*file.records, file.records[0])))


def test_agent_keys_come_from_the_key_map() -> None:
    assert load_agent_keys('{"a": "A", "b": "B"}') == {"a", "b"}
    with pytest.raises(ValueError):
        load_agent_keys('["a", "b"]')
