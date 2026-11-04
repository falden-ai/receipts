"""The files a unified diff changes, as the run classifier needs them (spec hard rule 3: a crash counts only when its
top non-test frame is in a changed file)."""

from __future__ import annotations

import re

_QUOTED = r"\"(?:[^\"\\]|\\.)+\""
_GIT_HEADER = re.compile(rf"^diff --git (?:a/\S+|{_QUOTED}) (?:b/(?P<b>\S+)|\"b/(?P<qb>(?:[^\"\\]|\\.)+)\")$")
_NEW_SIDE = re.compile(
    r"^\+\+\+ (?:b/(?P<plain>[^\t]+?)|\"b/(?P<quoted>(?:[^\"\\]|\\.)+)\"|(?P<bare>[^\t\"]+?))(?:\t.*)?$"
)
_HUNK = re.compile(r"^@@ -\d+(?:,(?P<old>\d+))? \+\d+(?:,(?P<new>\d+))? @@")
_ESCAPES = {"n": "\n", "t": "\t", '"': '"', "\\": "\\", "a": "\a", "b": "\b", "f": "\f", "r": "\r", "v": "\v"}


def changed_files(patch: str) -> tuple[str, ...]:
    """Every path the patch leaves in the patched tree, in the order the diff names them: modified, added and renamed
    files by their new path. A deleted file (+++ /dev/null) is not there to raise anything, so it is left out. Hunk
    lines are counted, so an added line that happens to start with "++ " is never read as a header."""
    found: list[str] = []
    pending: str | None = None  # a git header's new path, used when no +++ line follows (a binary or mode change)
    old = new = 0  # lines left in the current hunk
    for line in patch.split("\n"):
        if old > 0 or new > 0:
            if line.startswith("+"):
                new -= 1
                continue
            if line.startswith("-"):
                old -= 1
                continue
            if line.startswith(" ") or line == "":
                old, new = old - 1, new - 1
                continue
            if line.startswith("\\"):  # "\ No newline at end of file"
                continue
            old = new = 0  # a hunk cut short: read the line as a header
        hunk = _HUNK.match(line)
        if hunk:
            old = int(hunk.group("old") or 1)
            new = int(hunk.group("new") or 1)
            continue
        header = _GIT_HEADER.match(line)
        if header:
            if pending is not None:
                found.append(pending)
            pending = header.group("b") or _unquote(header.group("qb") or "")
            continue
        if line.startswith("+++ /dev/null"):
            pending = None  # deleted
            continue
        side = _NEW_SIDE.match(line)
        if side:
            path = side.group("plain") or side.group("bare") or _unquote(side.group("quoted") or "")
            if path:
                found.append(path)
            pending = None
    if pending is not None:
        found.append(pending)
    return tuple(dict.fromkeys(p for p in found if p))


def _unquote(quoted: str) -> str:
    """Git's C-style quoting of a path with special characters, undone; octal escapes are UTF-8 bytes."""
    out = bytearray()
    i = 0
    while i < len(quoted):
        ch = quoted[i]
        if ch != "\\" or i + 1 == len(quoted):
            out += ch.encode("utf-8")
            i += 1
            continue
        if re.fullmatch(r"[0-7]{3}", quoted[i + 1 : i + 4]):
            out.append(int(quoted[i + 1 : i + 4], 8))
            i += 4
        else:
            out += _ESCAPES.get(quoted[i + 1], quoted[i + 1]).encode("utf-8")
            i += 2
    return out.decode("utf-8", "replace")
