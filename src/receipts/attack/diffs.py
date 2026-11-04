"""Read a unified diff (the agent's patch): which files it changes, and which lines of the new files it touches.

The changed-file list feeds the classifier (a crash counts only in a changed file), the regression attacker's test
selection and the correctness attacker's prompt. The parse reads git's own headers (``diff --git``, ``---``, ``+++``,
``@@``) and nothing else; a path is kept only if it is relative and stays inside the repository.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Sequence
from dataclasses import dataclass, field

_GIT_HEADER = re.compile(r"^diff --git a/(?P<old>\S+) b/(?P<new>\S+)$")
_HUNK = re.compile(r"^@@ -\d+(?:,\d+)? \+(?P<start>\d+)(?:,(?P<count>\d+))? @@")


@dataclass
class FileChange:
    """One file's part of a diff. ``old`` or ``new`` is None for a file the diff creates or deletes."""

    old: str | None
    new: str | None
    text: str = ""
    lines: list[int] = field(default_factory=list)  # new-file line numbers the diff adds or changes


def _clean(path: str | None) -> str | None:
    if path is None:
        return None
    norm = posixpath.normpath(path)
    if norm.startswith(("/", "../")) or norm in (".", "..") or ".git" in norm.split("/"):
        return None
    return norm


def file_changes(patch: str) -> list[FileChange]:
    """Every file section of a unified diff, in order. The ``---`` and ``+++`` lines count as headers only between a
    ``diff --git`` line and its first hunk, so a removed line that starts with dashes is never read as one."""
    changes: list[FileChange] = []
    current: FileChange | None = None
    body: list[str] = []
    in_header = False
    new_line = 0
    for line in patch.replace("\r\n", "\n").split("\n"):
        header = _GIT_HEADER.match(line)
        if header is not None:
            if current is not None:
                current.text = "\n".join(body)
            current = FileChange(_clean(header.group("old")), _clean(header.group("new")))
            changes.append(current)
            body, in_header = [line], True
            continue
        if current is None:
            continue
        body.append(line)
        hunk = _HUNK.match(line)
        if hunk is not None:
            new_line, in_header = int(hunk.group("start")), False
            continue
        if in_header:
            if line == "--- /dev/null":
                current.old = None
            elif line == "+++ /dev/null":
                current.new = None
            continue
        if line.startswith("+"):
            current.lines.append(new_line)
            new_line += 1
        elif line.startswith(" "):
            new_line += 1
    if current is not None:
        current.text = "\n".join(body)
    return changes


def changed_files(patch: str) -> list[str]:
    """Repo-relative paths the patch touches: each file's new path, and its old path too for a rename or delete."""
    out: list[str] = []
    for change in file_changes(patch):
        for path in (change.new, change.old):
            if path is not None and path not in out:
                out.append(path)
    return out


def truncate(text: str, limit: int) -> str:
    """The head of ``text``, at most ``limit`` characters, marked when cut."""
    return text if len(text) <= limit else text[:limit] + f"\n[... {len(text) - limit} characters cut ...]"


def module_name(path: str, roots: Sequence[str] = ("src/", "lib/")) -> str | None:
    """The dotted module a repository file defines ('django/db/models/query.py' gives 'django.db.models.query'), with
    a leading src/ or lib/ dropped; None for a file that is not Python."""
    if not path.endswith(".py"):
        return None
    for root in roots:
        if path.startswith(root):
            path = path[len(root) :]
            break
    module = path[: -len(".py")]
    if module.endswith("/__init__"):
        module = module[: -len("/__init__")]
    parts = module.split("/")
    if not all(part.isidentifier() for part in parts):
        return None
    return ".".join(parts)
