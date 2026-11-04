"""R16's markdown: model text cannot render as a link or an image, cannot close the code around it, and cannot act on
a screen. Each test feeds hostile text and checks what a renderer would see."""

from __future__ import annotations

import re
from collections.abc import Callable

import pytest

from receipts.surface.markdown import LINK_REMOVED, clip, code_block, code_span, inline, plain, strip_links

Rendered = Callable[[str], str]

LINKS = [
    "https://evil.example/x?d=secret",
    "HTTPS://EVIL.EXAMPLE/X",
    "http://evil.example",
    "ftp://evil.example/f",
    "file:///etc/passwd-evil",
    "ws://evil.example/s",
    "s3://evil-bucket/key",
    "xhttps://evil.example",
    "https:\\\\evil.example\\x",
    "javascript:alert('evil')",
    "JavaScript:evil()",
    "vbscript:msgbox(evil)",
    "data:image/png;base64,ZXZpbA==",
    "data:text/html,<b>evil</b>",
    "mailto:evil@evil.example",
    "www.evil.example/x",
    "WWW2.evil.example",
    "//evil.example/pixel.png",
    # Forms a renderer still reads as links: a scheme after an underscore (GitHub links it) or inside a word, its
    # colon backslash-escaped (CommonMark drops the backslash), or its colon and slashes as HTML entities.
    "_https://evil.example/u02",
    "1https://evil.example/u03",
    "https\\://evil.example/u04",
    "https&#58;&#47;&#47;evil.example/u05",
    "HTTPS&#X3A;&#x2F;&#x2f;evil.example/u05b",
    "https&colon;&sol;&sol;evil.example/u05c",
    "_www.evil.example/u14",
    "_javascript:alert('evil')",
]


# What no stripped text may still hold: a host, a scheme, or a www. or // start. What is left after a cut, such as
# the "'evil')" of javascript:alert('evil'), is plain text that no renderer links.
LEFTOVERS = ("evil.example", "evil-bucket", "://", ":\\\\", "javascript:", "vbscript:", "data:", "mailto:", "www")


@pytest.mark.parametrize("link", LINKS)
def test_every_link_form_is_stripped(link: str) -> None:
    for text in (link, f"see {link} now", f"![x]({link})", f"[click]({link})", f"<{link}>", f'<img src="{link}">'):
        out = strip_links(text)
        assert not [left for left in LEFTOVERS if left in out.lower()], (text, out)
        assert LINK_REMOVED in out


@pytest.mark.parametrize(
    "text",
    [
        "a // b",
        "x//y",
        "os.path.join(a, b)",
        "if data:\n    return",
        "{'data': 1}",
        "lambda data: data",
        "ratio: 3:1",
        "C:\\Users\\x",
        "assert parse('a.b') == ('a', 'b')",
    ],
)
def test_code_that_is_not_a_link_is_kept(text: str) -> None:
    assert strip_links(text) == text


def test_a_code_block_cannot_be_closed_from_inside(rendered: Rendered) -> None:
    hostile = "ok\n```\n![x](https://evil.example/p.png) @everyone\n````\n   `````\n~~~\nend"
    block = code_block(hostile, "python")
    fence = block.split("\n", 1)[0]
    assert fence == "``````python"  # one longer than the longest backtick run inside
    assert block.endswith("\n``````")
    assert rendered(f"before\n{block}\nafter") == "before\nafter"  # nothing inside renders
    assert "evil" not in block and LINK_REMOVED in block


def test_a_code_block_escapes_hidden_characters_and_keeps_layout() -> None:
    block = code_block("a\tb\r\nc\rd\x1b[2J\N{RIGHT-TO-LEFT OVERRIDE}e\N{LINE SEPARATOR}f", "text")
    assert block == "```text\na\tb\nc\\u000dd\\u001b[2J\\u202ee\\u2028f\n```"


def test_a_code_block_refuses_an_info_string_that_is_not_a_language() -> None:
    for bad in ("py`", "py thon", "py\n", "x}"):
        with pytest.raises(ValueError, match="info string"):
            code_block("x", bad)


def test_a_code_span_stays_one_span(rendered: Rendered) -> None:
    for text, expected in (
        ("plain", "`plain`"),
        ("a`b", "``a`b``"),
        ("a``b`c", "```a``b`c```"),
        ("`edge`", "`` `edge` ``"),
        ("line\nbreak", "`line break`"),
        ("", "(empty)"),
    ):
        span = code_span(text)
        assert span == expected
        assert rendered(f"x {span} y") == ("x <code> y" if text else "x (empty) y")
    assert code_span("t\x1b[2J/https://evil.example") == "`t\\u001b[2J/[link removed]`"


def test_plain_text_is_one_safe_line() -> None:
    assert plain("a\n b\t\tc") == "a b c"
    assert plain("x https://evil.example y") == f"x {LINK_REMOVED} y"
    assert plain("t\x1b[2J\N{RIGHT-TO-LEFT OVERRIDE}") == "t\\u001b[2J\\u202e"
    assert plain("a" * 300, limit=10) == "aaaaaaa..."


def test_clip_says_what_it_cut() -> None:
    assert clip("short", 10, where="x") == "short"
    assert clip("0123456789abc", 10, where="the rest is in the ledger") == (
        "0123456789\n... (3 more characters cut; the rest is in the ledger)"
    )
    with pytest.raises(ValueError):
        clip("x", 0, where="x")


# The round-2 review's 16 link forms (u01 to u16): six of them survived plain() as links under a linkifying renderer.
FORMS = [
    "https://evil.com/u01?d=secret",
    "_https://evil.com/u02",
    "1https://evil.com/u03",
    "https\\://evil.com/u04",
    "[x](https&#58;&#47;&#47;evil.com/u05)",
    "![x](https\\://evil.com/u06.png)",
    "![x](https&#58;&#47;&#47;evil.com/u07.png)",
    "evil.com/u08",
    "user@evil.com",
    "x//evil.com/u10",
    "<https://evil.com/u11>",
    "[x](//evil.com/u12)",
    "<img src=https://evil.com/u13.png>",
    "www.evil.com/u14",
    "[x]: https://evil.com/u15",
    "javascript:alert(1)//u16",
]


def linkable(text: str) -> list[str]:
    """What a renderer or a linkifier could still read as a link, an image, an entity or a mention in plain text."""
    found = []
    if re.search(r"[a-z0-9]\s*:\s*/", text, re.I):
        found.append("a scheme and a slash")
    if re.search(r"[a-z0-9-]\.[a-z]", text, re.I):
        found.append("a domain")
    found += [mark for mark in ("@", "&", "(", ")", "<", ">", "`", "*", "|") if mark in text]
    return found


@pytest.mark.parametrize("form", FORMS)
def test_plain_text_holds_nothing_a_renderer_could_link(form: str) -> None:
    for text in (form, f"t1/fixed_x/{form}", f"Receipts: regression catch on fixed_x in {form}"):
        out = plain(text, limit=1_000)
        assert not linkable(out), (text, out, linkable(out))


def test_plain_text_keeps_ids_and_counts_as_they_are() -> None:
    title = "Receipts: fix-validity catch on fixed_x in 20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11099"
    for text in (title, "5 of 5 runs", "$0.123456", "on the code before the change in 5 of 5 runs", "0.0.1"):
        assert plain(text) == text
    assert plain("a@b.example (x)") == "a\\u0040b\\u002eexample \\u0028x\\u0029"
    assert plain("x" * 9 + "\N{ZERO WIDTH SPACE}" * 50, limit=30) == "x" * 9 + "\\u200b" * 3 + "..."


def test_every_limit_counts_characters_as_shown() -> None:
    zwsp = "\N{ZERO WIDTH SPACE}"
    block = code_block(zwsp * 1_000, "text", limit=120, where="the rest is in the ledger")
    body = block.split("\n")[1]
    assert body == "\\u200b" * 20  # 120 characters as shown, and never half an escape
    assert block.split("\n")[2] == "... (980 more characters cut; the rest is in the ledger)"
    assert code_block("short", limit=120) == "```\nshort\n```"
    span = code_span(zwsp * 100, limit=40)
    assert span == "`" + "\\u200b" * 6 + "...`" and len(span) <= 42
    assert inline("a\nb" + zwsp * 100, limit=12) == "a b\\u200b..."
    assert inline("fits in ten", limit=11) == "fits in ten"


def test_a_long_backtick_run_shows_as_escapes_so_a_fence_stays_short(rendered: Rendered) -> None:
    block = code_block("x = 1\n" + "`" * 5_000 + "\nend", "python")
    fence = block.split("\n", 1)[0]
    assert fence == "```python" and block.endswith("\n```")
    assert "\\u0060" * 5_000 in block and rendered(block) == ""
    short = code_block("a ``````````````` b")  # 15 in a row: still under the cap, so shown as they are
    assert short.startswith("`" * 16 + "\n") and "`" * 15 in short
    span = code_span("`" * 40)
    assert span == "`" + "\\u0060" * 40 + "`"
