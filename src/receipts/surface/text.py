"""Untrusted text made safe to show. Repro files, commands and ids are model output, and ledger errors can quote any
of them, so nothing reaches a terminal or a ticket with a character that acts instead of showing."""

from __future__ import annotations

import unicodedata

# Control, format, surrogate and line or paragraph separator characters: escape sequences, bidirectional overrides,
# zero-width marks and tag characters all live here.
HIDDEN = frozenset({"Cc", "Cf", "Cs", "Zl", "Zp"})


def visible(text: str, *, keep: str = "\t") -> str:
    """Every character in a HIDDEN category shown as an escape instead of acting on the screen. `keep` lists the
    characters that stay as they are (a tab by default; a ticket's code block also keeps the newline)."""
    return "".join(ch if ch in keep or unicodedata.category(ch) not in HIDDEN else escape(ch) for ch in text)


def escape(ch: str) -> str:
    code = ord(ch)
    return f"\\u{code:04x}" if code <= 0xFFFF else f"\\U{code:08x}"
