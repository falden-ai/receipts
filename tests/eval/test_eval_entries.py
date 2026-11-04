"""The entry selection is the spec's 3 + 1 plus the fifth mini entry (REG-565), and the results parsers refuse
malformed or partial files."""

from __future__ import annotations

import json

import pytest

from receipts.eval.entries import (
    SELECTED,
    Agent,
    SelectionError,
    check_selection,
    parse_resolved,
)
from receipts.eval.sources import SOURCES

UNIVERSE = ("astropy__astropy-1", "django__django-2", "sympy__sympy-3")
MINI = SELECTED[0]
OPENHANDS = SELECTED[4]
FIFTH = "20260217_mini-v2.0.0_gemini-3-flash-high"


def test_the_selected_entries_are_the_specs_three_plus_one_and_the_fifth() -> None:
    check_selection(SELECTED)
    assert [e.agent for e in SELECTED].count(Agent.MINI) == 4
    assert [e.agent for e in SELECTED].count(Agent.OPENHANDS) == 1
    assert FIFTH in [e.key for e in SELECTED] and OPENHANDS.agent is Agent.OPENHANDS
    names = {s.name for s in SOURCES}
    for e in SELECTED:
        assert e.resolved_source in names and e.metadata_source in names


def test_plant_a_v1_mini_entry_is_refused() -> None:
    old = MINI.model_copy(update={"agent_version": "1.17.2", "key": "20251211_mini-v1.17.2_x", "model_id": "x"})
    with pytest.raises(SelectionError, match=r"not v2\.x"):
        check_selection((old, *SELECTED[1:]))


def test_plant_a_tool_calling_mini_entry_is_refused() -> None:
    tools = MINI.model_copy(update={"bash_only": False})
    with pytest.raises(SelectionError, match="bash-only"):
        check_selection((tools, *SELECTED[1:]))


def test_plant_two_openhands_entries_are_refused() -> None:
    second = OPENHANDS.model_copy(update={"key": "other", "model_id": "other", "model_org": "other"})
    with pytest.raises(SelectionError, match="4 mini-swe-agent entries and 1 OpenHands"):
        check_selection((*SELECTED[:3], second, OPENHANDS))


def test_plant_the_spec_four_without_the_fifth_entry_is_refused() -> None:
    four = tuple(e for e in SELECTED if e.key != FIFTH)
    with pytest.raises(SelectionError, match="4 mini-swe-agent entries"):
        check_selection(four)


def test_plant_a_repeated_model_is_refused() -> None:
    twin = SELECTED[1].model_copy(update={"model_id": MINI.model_id})
    with pytest.raises(SelectionError, match="different model"):
        check_selection((MINI, twin, *SELECTED[2:]))


def test_plant_a_repeated_model_maker_is_refused() -> None:
    twin = SELECTED[1].model_copy(update={"model_org": MINI.model_org})
    with pytest.raises(SelectionError, match="different model maker"):
        check_selection((MINI, twin, *SELECTED[2:]))


def details(**resolved: bool) -> str:
    return json.dumps({iid: {"cost": 0.1, "api_calls": 3, "resolved": r} for iid, r in resolved.items()})


def test_per_instance_details_parse() -> None:
    raw = details(**{"astropy__astropy-1": True, "django__django-2": False, "sympy__sympy-3": True})
    out = parse_resolved(MINI, raw, UNIVERSE)
    assert out.resolved == {"astropy__astropy-1", "sympy__sympy-3"}
    assert out.unresolved == {"django__django-2"}


def test_plant_a_partial_results_file_is_refused() -> None:
    raw = details(**{"astropy__astropy-1": True, "django__django-2": False})
    with pytest.raises(ValueError, match="covers 2 of 3"):
        parse_resolved(MINI, raw, UNIVERSE)


def test_plant_a_result_without_a_boolean_is_refused() -> None:
    raw = json.dumps({"astropy__astropy-1": {"resolved": "yes"}})
    with pytest.raises(ValueError, match="no boolean"):
        parse_resolved(MINI, raw, UNIVERSE)


def test_plant_an_instance_outside_verified_is_refused() -> None:
    raw = details(**{"astropy__astropy-1": True, "django__django-2": False, "sympy__sympy-3": True, "x__y-9": True})
    with pytest.raises(ValueError, match="outside SWE-bench Verified"):
        parse_resolved(MINI, raw, UNIVERSE)


def test_results_json_parse_and_plant_a_resolved_empty_patch() -> None:
    raw = json.dumps({"resolved": ["astropy__astropy-1"], "no_generation": ["sympy__sympy-3"], "no_logs": []})
    out = parse_resolved(OPENHANDS, raw, UNIVERSE)
    assert out.resolved == {"astropy__astropy-1"} and out.no_generation == {"sympy__sympy-3"}
    planted = json.dumps({"resolved": ["astropy__astropy-1"], "no_generation": ["astropy__astropy-1"]})
    with pytest.raises(ValueError, match="no patch cannot be resolved"):
        parse_resolved(OPENHANDS, planted, UNIVERSE)


def test_plant_a_results_file_with_no_logs_never_reads_as_unresolved() -> None:
    raw = json.dumps({"resolved": ["astropy__astropy-1"], "no_generation": [], "no_logs": ["django__django-2"]})
    out = parse_resolved(OPENHANDS, raw, UNIVERSE)
    assert out.resolved_status("django__django-2") is None
    assert out.unresolved == {"sympy__sympy-3"} and out.resolved_status("sympy__sympy-3") is False
    planted = json.dumps({"resolved": ["astropy__astropy-1"], "no_logs": ["astropy__astropy-1"]})
    with pytest.raises(ValueError, match="no logs cannot be resolved"):
        parse_resolved(OPENHANDS, planted, UNIVERSE)


def test_every_entry_reads_its_logs_folder_and_says_where_an_unlisted_one_came_from() -> None:
    for e in SELECTED:
        assert e.patch_store.value == "s3-logs" and e.patch_location.endswith(f"{e.key}/logs")
    unlisted = [e.key for e in SELECTED if e.logs_note]
    assert unlisted == ["20260217_mini-v2.0.0_glm-5-high", "20260217_mini-v2.0.0_minimax-2-5-high", FIFTH]
