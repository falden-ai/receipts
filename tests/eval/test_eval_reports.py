"""report.json is read in both published shapes, checked against the published resolved flag and the dataset's test
lists, and refused when a test sits in neither list (skipped or not run) or the report contradicts itself."""

from __future__ import annotations

import json
from typing import Any

import pytest

from receipts.eval.reports import ReportError, parse_report, report_evidence

IID = "django__django-1"
F2P = ("t::fix_a", "t::fix_b")
P2P = ("t::old_a", "t::old_b")


def raw(
    *,
    f2p_ok: tuple[str, ...] = F2P,
    f2p_bad: tuple[str, ...] = (),
    p2p_ok: tuple[str, ...] = P2P,
    p2p_bad: tuple[str, ...] = (),
    resolved: bool | None = None,
    applied: bool = True,
    wrap: bool = True,
    with_tests: bool = True,
) -> bytes:
    body: dict[str, Any] = {
        "patch_is_None": False,
        "patch_exists": True,
        "patch_successfully_applied": applied,
        "resolved": (not f2p_bad and not p2p_bad) if resolved is None else resolved,
    }
    if with_tests:
        body["tests_status"] = {
            "FAIL_TO_PASS": {"success": list(f2p_ok), "failure": list(f2p_bad)},
            "PASS_TO_PASS": {"success": list(p2p_ok), "failure": list(p2p_bad)},
            "FAIL_TO_FAIL": {"success": [], "failure": []},
            "PASS_TO_FAIL": {"success": [], "failure": []},
        }
    return json.dumps({IID: body} if wrap else body).encode()


def read(data: bytes, published: bool | None = None) -> tuple[bool | None, bool | None]:
    report = parse_report(data, IID)
    ev = report_evidence(report, IID, F2P, P2P, published_resolved=report.resolved if published is None else published)
    return ev.fail_to_pass_passed, ev.pass_to_pass_passed


def test_both_published_shapes_parse_the_same() -> None:
    assert parse_report(raw(wrap=True), IID) == parse_report(raw(wrap=False), IID)


def test_the_three_outcomes_read_as_the_harness_means_them() -> None:
    assert read(raw()) == (True, True)  # resolved
    assert read(raw(f2p_ok=F2P[:1], f2p_bad=F2P[1:])) == (False, True)  # BAD: fails FAIL_TO_PASS
    assert read(raw(p2p_ok=P2P[:1], p2p_bad=P2P[1:])) == (True, False)  # passes FAIL_TO_PASS, breaks PASS_TO_PASS


def test_a_harness_that_found_no_test_output_says_nothing() -> None:
    report = parse_report(raw(applied=False, resolved=False, with_tests=False), IID)
    ev = report_evidence(report, IID, F2P, P2P, published_resolved=False)
    assert (ev.harness_found_tests, ev.fail_to_pass_passed, ev.pass_to_pass_passed) == (False, None, None)


@pytest.mark.parametrize(
    ("data", "match"),
    [
        (raw(f2p_ok=F2P[:1]), "1 tests in neither list"),  # a skipped FAIL_TO_PASS test
        (raw(p2p_ok=P2P[:1]), "PASS_TO_PASS in report.json does not match"),  # a skipped PASS_TO_PASS test
        (raw(f2p_ok=(*F2P, "t::extra")), "1 not in the dataset"),
        (raw(f2p_ok=F2P, f2p_bad=F2P[:1], resolved=False), "both success and failure"),
        (raw(f2p_ok=(F2P[0], F2P[0], F2P[1])), "twice"),
        (raw(f2p_ok=F2P[:1], f2p_bad=F2P[1:], resolved=True), "lists give FAIL_TO_PASS failed"),
        (raw(resolved=True, with_tests=False), "resolved, with no test results"),
    ],
)
def test_plant_a_report_that_disagrees_with_the_dataset_or_itself_is_refused(data: bytes, match: str) -> None:
    with pytest.raises(ReportError, match=match):
        read(data)


def test_plant_a_report_that_contradicts_the_published_results_is_refused() -> None:
    with pytest.raises(ReportError, match="the published results say False"):
        read(raw(), published=False)


@pytest.mark.parametrize(
    "data",
    [
        b"not json",
        b"[]",
        json.dumps({"sympy__sympy-2": {"resolved": True}}).encode(),  # another instance's report
        json.dumps({"resolved": True}).encode(),  # missing the harness's fields
        json.dumps({IID: {**json.loads(raw())[IID], "surprise": 1}}).encode(),  # a field the harness never writes
    ],
)
def test_plant_a_malformed_or_foreign_report_is_refused(data: bytes) -> None:
    with pytest.raises(ReportError):
        parse_report(data, IID)
