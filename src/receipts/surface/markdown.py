"""Markdown for posted tickets that model text cannot turn into an exfiltration channel (R16).

A ticket that renders a model-written image or link can leak what the model saw: an image URL is fetched as soon as
the ticket is shown, and Linear also fetches any image URL in a description on its own servers, to re-host it, before
anyone opens the ticket. So every piece of model text goes out one of two ways: in a fenced code block or a code span
whose fence no backtick run inside can close, or as plain text for a title or an attribute. Either way its links and
images are stripped first (a scheme URL wherever it starts, with its colon and slashes plain, backslash-escaped or
written as HTML entities; a www. host; a protocol-relative //host; and the javascript:, vbscript:, data: and mailto:
forms) and its hidden characters (controls, bidirectional overrides, zero-width marks) are shown as escapes. Inside
code nothing is rendered, so markdown image and link syntax that is left over is inert text. Plain text is stricter
still, since nothing fences it: every character but letters, digits and a few marks of punctuation is shown as an
escape, and so is a dot before a letter, so no renderer can read an entity, an email address or a bare domain in it.

Every limit counts characters as they are shown, after escaping, so text full of hidden characters cannot make a
ticket larger than its limits say.
"""

from __future__ import annotations

import re
import string
import unicodedata
from collections.abc import Callable

from receipts.surface.text import HIDDEN, escape

LINK_REMOVED = "[link removed]"

# Characters that end a URL for these patterns: whitespace, quotes, backticks, angle brackets and the closing bracket
# and parenthesis of markdown syntax. What a cut leaves behind is plain text, never a link.
_END = r"[^\s<>\"'`)\]]"
# A scheme's colon and slashes as a renderer may read them: as they are, backslash-escaped (CommonMark drops the
# backslash), or as HTML entities, which a renderer decodes before it reads a link's destination.
_COLON = r"(?::|\\:|&#0*58;|&#x0*3a;|&colon;)"
_SLASH = r"(?:/|\\|&#0*47;|&#x0*2f;|&sol;|&#0*92;|&#x0*5c;|&bsol;)"
_LINKS = re.compile(
    "|".join(
        (
            # Any scheme with an authority, wherever it starts: GitHub links one after an underscore, and other
            # renderers link one inside a word.
            rf"[a-z][a-z0-9+.\-]{{0,31}}{_COLON}{_SLASH}{{2}}{_END}*",
            rf"(?<![a-z0-9])(?:javascript|vbscript){_COLON}{_END}*",
            rf"(?<![a-z0-9])data{_COLON}[a-z]+/[a-z0-9.+\-]+[;,]{_END}*",
            rf"(?<![a-z0-9])mailto{_COLON}{_END}+",
            rf"(?<![a-z0-9])www\d{{0,3}}\.{_END}+",
            rf"(?<![\w:/\\])//[a-z0-9\-]+(?:\.[a-z0-9\-]+)+{_END}*",  # //host.tld/path, never a // b
        )
    ),
    re.IGNORECASE,
)
_INFO = re.compile(r"[a-z0-9_+\-]*")
# A backtick run this long inside code is shown as escapes, so a fence never needs to be longer than this.
_LONG_TICKS = 16
_TICKS = re.compile(r"`+")
# What plain text keeps as it is. Everything else, markdown and HTML punctuation (& < > ( ) * ` | \ @) included, is
# shown as an escape.
_PLAIN_KEEP = frozenset(string.ascii_letters + string.digits + " _-.~:+$,/=%#[]!?'\"^;{}")
_ASCII_SPACE = re.compile(r"[ \t\n\r\f\v]+")


def strip_links(text: str) -> str:
    """Every link and image target in the text replaced by LINK_REMOVED."""
    return _LINKS.sub(LINK_REMOVED, text)


def clip(text: str, limit: int, *, where: str) -> str:
    """At most `limit` characters of the text, with a note saying how much was cut and where the rest is."""
    if limit <= 0:
        raise ValueError("a clip keeps at least one character")
    if len(text) <= limit:
        return text
    return f"{text[:limit]}\n... ({len(text) - limit:,} more characters cut; {where})"


def code_block(text: str, info: str = "", *, limit: int | None = None, where: str = "the rest is in the ledger") -> str:
    """The text as a fenced code block: links stripped, hidden characters escaped (tabs and newlines kept), at most
    `limit` characters as shown with a note on what was cut, and a fence longer than any backtick run in it, so no
    line of the text can close the block early."""
    if _INFO.fullmatch(info) is None:
        raise ValueError(f"a code block's info string is a plain language name, not {info!r}")
    pieces = _pieces(text.replace("\r\n", "\n"), keep="\t\n")
    body = _joined(pieces, limit, lambda rest: f"\n... ({rest:,} more characters cut; {where})")
    fence = "`" * max(3, _longest_run(body) + 1)
    return f"{fence}{info}\n{body}\n{fence}"


def code_span(text: str, *, limit: int | None = None) -> str:
    """The text as an inline code span on one line: links stripped, hidden characters escaped, newlines as spaces, at
    most `limit` characters as shown, and a delimiter no backtick run inside matches."""
    body = inline(text, limit)
    if not body:
        return "(empty)"
    runs = {len(run) for run in _TICKS.findall(body)}
    n = 1
    while n in runs:
        n += 1
    pad = " " if body[0] in "` " or body[-1] in "` " else ""  # one space each side is dropped when rendered
    return f"{'`' * n}{pad}{body}{pad}{'`' * n}"


def inline(text: str, limit: int | None = None) -> str:
    """The text on one line, as code shows it: links stripped, hidden characters escaped, newlines as spaces, and at
    most `limit` characters as shown, ending in "..." when cut. For text that goes inside code."""
    pieces = _pieces(" ".join(text.replace("\r\n", "\n").split("\n")), keep="")
    return _joined(pieces, limit, _ellipsis, reserve=3)


def plain(text: str, limit: int = 200) -> str:
    """The text for a field that is not markdown, such as a title or an attribute value: one line, links stripped,
    and every character but letters, digits and a few marks of punctuation shown as an escape, as is a dot before a
    letter; at most `limit` characters as shown."""
    one_line = _ASCII_SPACE.sub(" ", strip_links(text)).strip(" ")
    following = [*one_line[1:], ""]
    pieces = [
        ch if ch in _PLAIN_KEEP and not (ch == "." and nxt.isascii() and nxt.isalpha()) else escape(ch)
        for ch, nxt in zip(one_line, following, strict=True)
    ]
    return _joined(pieces, limit, _ellipsis, reserve=3)


def _pieces(text: str, *, keep: str) -> list[str]:
    """The text with its links stripped, one shown piece per character: the character, or its escape when it is
    hidden or one of a backtick run too long for a fence."""
    stripped = strip_links(text)
    long_ticks = {
        i for run in _TICKS.finditer(stripped) if len(run.group()) >= _LONG_TICKS for i in range(run.start(), run.end())
    }
    return [
        escape(ch) if i in long_ticks or (ch not in keep and unicodedata.category(ch) in HIDDEN) else ch
        for i, ch in enumerate(stripped)
    ]


def _joined(pieces: list[str], limit: int | None, note: Callable[[int], str], *, reserve: int = 0) -> str:
    """The pieces joined when they fit in `limit` characters; otherwise those that fit in `limit` less `reserve`, then
    `note`, given how many pieces were left out."""
    if limit is None or sum(len(piece) for piece in pieces) <= limit:
        return "".join(pieces)
    room = max(0, limit - reserve)
    out: list[str] = []
    size = 0
    for i, piece in enumerate(pieces):
        if size + len(piece) > room:
            return "".join(out) + note(len(pieces) - i)
        out.append(piece)
        size += len(piece)
    return "".join(out)


def _ellipsis(left_out: int) -> str:
    return "..."


def _longest_run(text: str) -> int:
    return max((len(run) for run in _TICKS.findall(text)), default=0)
