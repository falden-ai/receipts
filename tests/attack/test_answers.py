"""contracts.answers, the one matching of a run's cases to expected ids that the attack lane's never-ran check and
R24's gold check both use, and the ids swebench 4.1.0's log parsers rename before they record them
(harness/log_parsers/python.py, read in the wheel eval/manifest.json pins: parse_log_pytest_options, lines 29 to 61,
and parse_log_matplotlib, lines 229 to 250). The expected ids below are what those parsers' own code makes of each
name, read from the code, never by running it."""

from __future__ import annotations

import pytest

from receipts.attack.classify import reduce_cases
from receipts.contracts import CaseResult, FailureType, Outcome, answers

PYLINT = "pylint-dev/pylint"


@pytest.mark.parametrize(
    ("repo", "name", "recorded"),
    [
        # parse_log_pytest_options: a part in brackets that starts with one slash and has no '*' keeps a slash and
        # what follows its last slash.
        (PYLINT, "t.py::test_x[/a/b/c.py]", "t.py::test_x[/c.py]"),
        ("psf/requests", "t.py::test_x[/a/b]", "t.py::test_x[/b]"),
        ("pydicom/pydicom", "t.py::T::test_x[/a/b-c]", "t.py::T::test_x[/b-c]"),
        # Nested brackets, as its (.*?)\[(.*)\] reads them: up to the first '[', then all of it up to the last ']'.
        (PYLINT, "t.py::test_x[/a/[b]/c]", "t.py::test_x[/c]"),
        (PYLINT, "t.py::test_x[/a/b][/c/d]", "t.py::test_x[/d]"),
        (PYLINT, "t.py::test_x[/a/b[1]]", "t.py::test_x[/b[1]]"),
        # Whatever follows the last ']' is dropped, renamed or not.
        (PYLINT, "t.py::test_x[/a/b]tail", "t.py::test_x[/b]"),
        (PYLINT, "t.py::test_x[a]tail", "t.py::test_x[a]"),
        # parse_log_matplotlib writes MouseButton.LEFT as 1 and MouseButton.RIGHT as 3, wherever they are.
        ("matplotlib/matplotlib", "t.py::test_click[MouseButton.LEFT]", "t.py::test_click[1]"),
        ("matplotlib/matplotlib", "t.py::test_drag[MouseButton.RIGHT-MouseButton.LEFT]", "t.py::test_drag[3-1]"),
    ],
)
def test_a_case_answers_the_id_swebenchs_parser_records_for_it(repo: str, name: str, recorded: str) -> None:
    case = CaseResult(name=name, outcome=Outcome.PASS)
    assert answers(case, recorded, repo=repo) and answers(case, name, repo=repo)
    assert reduce_cases([case], expect=[recorded], repo=repo) == (Outcome.PASS, None)


@pytest.mark.parametrize(
    ("repo", "name", "other"),
    [
        (PYLINT, "t.py::test_x[//a/b]", "t.py::test_x[/b]"),  # two slashes: kept
        (PYLINT, "t.py::test_x[/a/*/b]", "t.py::test_x[/b]"),  # a '*': kept
        (PYLINT, "t.py::test_x[a/b/c]", "t.py::test_x[/c]"),  # no leading slash: kept
        (PYLINT, "t.py::test_x[a/b/c]", "t.py::test_x[c]"),
        (PYLINT, "t.py::test_x[a[/b/c]]", "t.py::test_x[a[/c]]"),  # the part in brackets starts with 'a'
        (PYLINT, "t.py::test_x[/a b/c]", "t.py::test_x[/c]"),  # whitespace: the harness never records it whole
        (PYLINT, "t.py::test_click[MouseButton.LEFT]", "t.py::test_click[1]"),  # matplotlib's rule is not pylint's
        ("matplotlib/matplotlib", "t.py::test_x[/a/b]", "t.py::test_x[/b]"),  # nor the other way round
        ("astropy/astropy", "t.py::test_x[/a/b]", "t.py::test_x[/b]"),  # a repository without rules
        ("astropy/astropy", "t.py::test_click[MouseButton.LEFT]", "t.py::test_click[1]"),
        ("sympy/sympy", "t.py::test_x[/a/b]", "t.py::test_x[/b]"),
        ("Pylint-Dev/Pylint", "t.py::test_x[/a/b]", "t.py::test_x[/b]"),  # the harness's map takes the name exactly
        ("", "t.py::test_x[/a/b]", "t.py::test_x[/b]"),
    ],
)
def test_a_case_answers_no_id_the_harness_does_not_record_for_it(repo: str, name: str, other: str) -> None:
    case = CaseResult(name=name, outcome=Outcome.PASS)
    assert answers(case, name, repo=repo)
    assert not answers(case, other, repo=repo)
    assert reduce_cases([case], expect=[other], repo=repo) == (Outcome.ERROR, FailureType.INFRA)
