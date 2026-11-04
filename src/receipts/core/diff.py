"""What a target's patch (a unified diff, as git writes it) changes: the files, the lines, and the names of the
functions and classes it touches. Intake reads it for risk, the planner for context and the skeptic for its gates.

Hunks are read by their line counts, so a removed line that starts with '--' is never taken for a file header. Names
come from three places in each hunk: the function context git prints after the second '@@', any 'def' or 'class'
line the hunk adds or removes, and the nearest 'def' or 'class' above a changed line among the hunk's own lines.
Git's default function context for a Python file is the last unindented line, so a changed method is found by name
only when its 'def' is in the hunk; its class is found either way.

A path git had to quote (core.quotePath, the default: any byte past ASCII, or a control character, quote or
backslash) comes in double quotes with C escapes, a byte past ASCII as three octal digits; it is read back as those
bytes, decoded as UTF-8, so '"a/caf\\303\\251.py"' is 'caf\\u00e9.py' (an e with an acute accent). Bytes that are not
UTF-8 stay as surrogate escapes, the way Python names such a file.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Final

_DIFF_GIT = re.compile(r'^diff --git ("?)a/(.+?)\1 ("?)b/(.+?)\3$')
_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+\d+(?:,(\d+))? @@ ?(.*)$")
_DEF = re.compile(r"^\s*(?:async\s+)?def\s+([A-Za-z_]\w*)|^\s*class\s+([A-Za-z_]\w*)")

TEST_DIRS: Final = frozenset({"test", "tests", "testing"})
SOURCE_ROOTS: Final = frozenset({"src", "lib"})  # layouts that keep the package one folder down


@dataclass
class FileChange:
    """One file in the patch. A path is None on the side where the file does not exist (an added or deleted file)."""

    old_path: str | None
    new_path: str | None
    added: int = 0
    removed: int = 0
    names: set[str] = field(default_factory=set)
    removed_lines: list[str] = field(default_factory=list)

    @property
    def path(self) -> str:
        """The file's path in the patched tree, or its old path for a deletion."""
        path = self.new_path or self.old_path
        assert path is not None
        return path


_ESCAPES: Final = {"a": 7, "b": 8, "t": 9, "n": 10, "v": 11, "f": 12, "r": 13, '"': 34, "\\": 92}
_OCTAL: Final = frozenset("01234567")


def unquote(text: str) -> str:
    """The path inside git's double quotes: its C escapes read back as bytes (an octal escape is one byte), decoded as
    UTF-8."""
    out = bytearray()
    i = 0
    while i < len(text):
        char = text[i]
        if char != "\\" or i + 1 == len(text):
            out += char.encode("utf-8", "surrogatepass")
            i += 1
            continue
        after = text[i + 1]
        if after in _OCTAL:
            j = i + 1
            while j < len(text) and j < i + 4 and text[j] in _OCTAL:
                j += 1
            out.append(int(text[i + 1 : j], 8) & 0xFF)
            i = j
        elif after in _ESCAPES:
            out.append(_ESCAPES[after])
            i += 2
        else:  # not an escape git writes: kept as it is
            out += ("\\" + after).encode("utf-8", "surrogatepass")
            i += 2
    return out.decode("utf-8", "surrogateescape")


def _path(raw: str, prefix: str) -> str | None:
    text = raw.split("\t", 1)[0].strip()
    if text.startswith('"') and text.endswith('"') and len(text) >= 2:
        text = unquote(text[1:-1])
    if text == "/dev/null":
        return None
    return text[len(prefix) :] if text.startswith(prefix) else text


def _git_path(quote: str, path: str) -> str:
    return unquote(path) if quote else path


def _defined(text: str) -> str | None:
    match = _DEF.match(text)
    return (match.group(1) or match.group(2)) if match else None


def parse_patch(patch: str) -> tuple[FileChange, ...]:
    """The files a unified diff changes, in order, with their added and removed line counts and touched names."""
    files: list[FileChange] = []
    current: FileChange | None = None
    old_left = new_left = 0  # lines still to read in the current hunk
    enclosing: str | None = None
    for line in patch.splitlines():
        if old_left > 0 or new_left > 0:
            assert current is not None
            if line.startswith("\\"):  # "\ No newline at end of file"
                continue
            kind, text = line[:1], line[1:]
            if kind == "+":
                new_left -= 1
                current.added += 1
            elif kind == "-":
                old_left -= 1
                current.removed += 1
                current.removed_lines.append(text)
            else:  # context, including a blank line some tools write without its leading space
                old_left -= 1
                new_left -= 1
                name = _defined(text)
                if name is not None:
                    enclosing = name
                continue
            name = _defined(text)
            if name is not None:
                current.names.add(name)
            elif enclosing is not None:
                current.names.add(enclosing)
            continue
        if line.startswith("diff --git "):
            match = _DIFF_GIT.match(line)
            current = (
                FileChange(old_path=_git_path(match[1], match[2]), new_path=_git_path(match[3], match[4]))
                if match
                else None
            )
            if current is not None:
                files.append(current)
            continue
        if line.startswith("--- "):
            old = _path(line[4:], "a/")
            if current is None or current.added or current.removed:  # a plain diff's next file has no 'diff' line
                current = FileChange(old_path=old, new_path=None)
                files.append(current)
            else:
                current.old_path = old
            continue
        if line.startswith("+++ ") and current is not None:
            current.new_path = _path(line[4:], "b/")
            continue
        hunk = _HUNK.match(line)
        if hunk and current is not None:
            old_left = int(hunk.group(1)) if hunk.group(1) is not None else 1
            new_left = int(hunk.group(2)) if hunk.group(2) is not None else 1
            enclosing = None
            name = _defined(hunk.group(3))
            if name is not None:
                current.names.add(name)
    return tuple(f for f in files if f.old_path is not None or f.new_path is not None)


def changed_files(patch: str) -> tuple[str, ...]:
    """The repository-relative paths the patch changes, in order, each once."""
    seen: dict[str, None] = {}
    for change in parse_patch(patch):
        seen.setdefault(change.path, None)
    return tuple(seen)


def changed_names(patch: str) -> frozenset[str]:
    """The names of the functions and classes the patch touches, in files that are not tests."""
    names: set[str] = set()
    for change in parse_patch(patch):
        if not is_test_path(change.path):
            names |= change.names
    return frozenset(names)


def is_test_path(path: str) -> bool:
    """A test file by the usual conventions: test_*.py, *_test.py, tests.py, conftest.py, or under a tests folder."""
    pure = PurePosixPath(path)
    name = pure.name
    if name in ("tests.py", "conftest.py") or name.startswith("test_") or name.endswith("_test.py"):
        return True
    return any(part in TEST_DIRS for part in pure.parts[:-1])


def module_name(path: str) -> str | None:
    """A Python file's dotted module name, without a src/ or lib/ root: 'lib/matplotlib/axes/_base.py' is
    'matplotlib.axes._base'. None for a file that is not Python."""
    pure = PurePosixPath(path)
    if pure.suffix != ".py":
        return None
    parts = list(pure.with_suffix("").parts)
    if parts and parts[-1] == "__init__":
        parts.pop()
    if len(parts) > 1 and parts[0] in SOURCE_ROOTS:
        parts.pop(0)
    return ".".join(parts) or None


@dataclass(frozen=True)
class PatchFacts:
    """What the skeptic's static gates need from a patch: the files it changes, the names it touches in each module
    that is not a test, and the repository's own top-level packages."""

    files: frozenset[str]
    names_by_module: Mapping[str, frozenset[str]]
    packages: frozenset[str]

    @property
    def names(self) -> frozenset[str]:
        found: set[str] = set()
        for names in self.names_by_module.values():
            found |= names
        return frozenset(found)


def patch_facts(patch: str, repo: str = "") -> PatchFacts:
    """The facts of a target's patch. The repository's own packages are the top folders its changed Python files live
    in, and its name ('psf/requests' gives requests)."""
    names: dict[str, set[str]] = {}
    packages: set[str] = set()
    for change in parse_patch(patch):
        module = module_name(change.path)
        if module is None:
            continue
        packages.add(module.split(".", 1)[0])
        if not is_test_path(change.path):
            names.setdefault(module, set()).update(change.names)
    if repo:
        packages.add(repo.rsplit("/", 1)[-1].replace("-", "_").lower())
    return PatchFacts(
        files=frozenset(changed_files(patch)),
        names_by_module={module: frozenset(found) for module, found in names.items()},
        packages=frozenset(packages),
    )
