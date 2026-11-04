"""The SWE-bench harness's report.json for one agent patch, as the entries publish it in s3://swe-bench-submissions,
and what it says about FAIL_TO_PASS and PASS_TO_PASS.

swebench 4.1.0 (harness/grading.py) puts a test in a list's `success` when its status is PASSED or XFAIL, in
`failure` when it is FAILED, ERROR or missing from the log, and in neither when it has any other status, such as
SKIPPED. So a report whose two lists do not add up to the dataset's FAIL_TO_PASS (or PASS_TO_PASS) list has a test
that did not run, and `report_evidence` refuses it: a skipped test never reads as a pass. An XFAIL cannot be told
from a pass in report.json; the gold check reads per-test statuses itself (runsc_check).
"""

from __future__ import annotations

import json
from collections.abc import Sequence

from receipts.contracts import Frozen


class ReportError(ValueError):
    """A report that is malformed, belongs to another instance, or contradicts the published results or dataset."""


class ResultLists(Frozen):
    success: tuple[str, ...]
    failure: tuple[str, ...]


class StatusLists(Frozen):
    FAIL_TO_PASS: ResultLists
    PASS_TO_PASS: ResultLists
    FAIL_TO_FAIL: ResultLists | None = None
    PASS_TO_FAIL: ResultLists | None = None


class Report(Frozen):
    """One instance's entry in report.json (swebench.harness.grading.get_eval_report)."""

    patch_is_None: bool
    patch_exists: bool
    patch_successfully_applied: bool  # the harness found its test-output markers and no apply, reset or test error
    resolved: bool
    tests_status: StatusLists | None = None


def parse_report(raw: bytes | str, instance_id: str) -> Report:
    """Read report.json in either published shape: {instance_id: report} (the harness's) or the bare report."""
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ReportError(f"{instance_id}: report.json is not JSON: {e}") from e
    if not isinstance(data, dict):
        raise ReportError(f"{instance_id}: report.json is not an object")
    if "resolved" not in data:
        if list(data) != [instance_id]:
            raise ReportError(f"{instance_id}: report.json is keyed by {sorted(data)[:3]}, not this instance")
        data = data[instance_id]
    try:
        return Report.model_validate(data)
    except ValueError as e:
        raise ReportError(f"{instance_id}: report.json does not have the harness's fields: {e}") from e


class ReportEvidence(Frozen):
    """What one report establishes. None means the report cannot say."""

    harness_found_tests: bool
    fail_to_pass_passed: bool | None
    pass_to_pass_passed: bool | None


def _lists_match(lists: ResultLists, expected: Sequence[str], what: str, instance_id: str) -> None:
    got_success, got_failure, want = set(lists.success), set(lists.failure), set(expected)
    if got_success & got_failure:
        raise ReportError(f"{instance_id}: {what} has tests in both success and failure")
    if len(lists.success) + len(lists.failure) != len(got_success | got_failure):
        raise ReportError(f"{instance_id}: {what} lists a test twice")
    if got_success | got_failure != want:
        missing = sorted(want - got_success - got_failure)
        extra = sorted((got_success | got_failure) - want)
        raise ReportError(
            f"{instance_id}: {what} in report.json does not match the dataset: {len(missing)} tests in neither list "
            f"(skipped or not run), e.g. {missing[:2]}; {len(extra)} not in the dataset, e.g. {extra[:2]}"
        )


def report_evidence(
    report: Report,
    instance_id: str,
    fail_to_pass: Sequence[str],
    pass_to_pass: Sequence[str],
    *,
    published_resolved: bool | None,
) -> ReportEvidence:
    """Check a report against the entry's published resolved flag and the dataset's test lists, then read it."""
    if published_resolved is not None and report.resolved != published_resolved:
        raise ReportError(
            f"{instance_id}: report.json says resolved={report.resolved}, "
            f"the published results say {published_resolved}"
        )
    ts = report.tests_status
    if report.patch_is_None or not report.patch_exists or not report.patch_successfully_applied or ts is None:
        if report.resolved:
            raise ReportError(f"{instance_id}: report.json says resolved, with no test results")
        return ReportEvidence(harness_found_tests=False, fail_to_pass_passed=None, pass_to_pass_passed=None)
    if not fail_to_pass:
        raise ReportError(f"{instance_id}: the dataset row has no FAIL_TO_PASS tests")
    _lists_match(ts.FAIL_TO_PASS, fail_to_pass, "FAIL_TO_PASS", instance_id)
    _lists_match(ts.PASS_TO_PASS, pass_to_pass, "PASS_TO_PASS", instance_id)
    f2p = not ts.FAIL_TO_PASS.failure
    p2p = not ts.PASS_TO_PASS.failure
    if report.resolved != (f2p and p2p):
        raise ReportError(
            f"{instance_id}: report.json says resolved={report.resolved}, but its lists give FAIL_TO_PASS "
            f"{'passed' if f2p else 'failed'} and PASS_TO_PASS {'passed' if p2p else 'failed'}"
        )
    return ReportEvidence(harness_found_tests=True, fail_to_pass_passed=f2p, pass_to_pass_passed=p2p)
