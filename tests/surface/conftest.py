"""Shared test helpers for the surface lane, as fixtures (the suite imports test modules by path, not by name)."""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

_OPEN = re.compile(r"^ {0,3}(`{3,})([^`]*)$")


def outside_code(markdown: str) -> str:
    """What a CommonMark renderer would treat as markdown: the text with every backtick-fenced code block and every
    code span taken out. Model text found here would render; model text only inside code would not.

    Fences follow CommonMark: an opening line of 3 or more backticks (up to 3 spaces in), closed by a line of at least
    as many backticks and nothing else, or by the end of the text. A code span opens with a run of n backticks and
    closes at the next run of exactly n; an unmatched run is literal text."""
    kept: list[str] = []
    fence = 0
    for line in markdown.split("\n"):
        if fence:
            stripped = line.strip(" ")
            if len(line) - len(line.lstrip(" ")) <= 3 and set(stripped) == {"`"} and len(stripped) >= fence:
                fence = 0
            continue
        opened = _OPEN.match(line)
        if opened:
            fence = len(opened.group(1))
            continue
        kept.append(_without_spans(line))
    return "\n".join(kept)


def _without_spans(line: str) -> str:
    out: list[str] = []
    i = 0
    while i < len(line):
        if line[i] != "`":
            out.append(line[i])
            i += 1
            continue
        j = i
        while j < len(line) and line[j] == "`":
            j += 1
        n = j - i
        close = j
        while True:
            k = line.find("`" * n, close)
            if k == -1:
                out.append(line[i:j])  # unmatched: literal backticks
                i = j
                break
            end = k
            while end < len(line) and line[end] == "`":
                end += 1
            if end - k == n:
                out.append("<code>")
                i = end
                break
            close = end
    return "".join(out)


@pytest.fixture
def rendered() -> Callable[[str], str]:
    return outside_code
