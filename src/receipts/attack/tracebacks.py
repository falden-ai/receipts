"""Read Python's own traceback text by its structure, and type a failure from it (SK1 part a, spec hard rule 3).

Every runner the attack lane drives prints the interpreter's native traceback: pytest under ``--tb=native`` (which the
pytest runner forces), Django's unittest runner and sympy's bin/test. The layout is fixed by CPython's traceback
module. A block starts with the line ``Traceback (most recent call last):`` at column 0; each frame is a line
``  File "<path>", line <n>, in <name>`` with two spaces of indent, followed by its source and caret lines, indented
by four or more; then comes the exception line at column 0, then any further lines of the message. A chained
exception is a blank line, one of the two separator sentences, a blank line and the next block, and the last block
is the exception that was raised.

Frames are read only in that frames section, between a block's header and its exception line. A line shaped like a
frame anywhere else (in an exception message, a printed value, a docstring) is never a frame. There are no argument
reprs in this format, so the multi-line repr that faked a frame under pytest's long style cannot appear at all. A
message can still imitate a whole chained block after the real exception line; the pytest runner closes that by
taking the block whose exception text equals junit's own message (see ``final_block``), and for the unittest runners
a forged block can only move a real crash between "elsewhere" and "in changed code" on a run that did crash.

A frame counts only if it resolves inside the workspace root. A frame outside the root (the standard library, site
packages, the runner itself, ``<frozen ...>`` or ``<string>``) never matches a changed file, whatever its name; inside
the root only the identical repo-relative path matches. Frames outside the root are transparent: a crash is
attributed to the deepest frame inside the repository, so a ValueError raised inside the standard library by changed
code counts the same as one raised by a builtin, which leaves no frame of its own. If that deepest repository frame
is test code, the crash is the test's own and does not count, and a crash whose outermost test frame is a setUp-style
hook is a fixture error. Errors the sandbox itself causes (no network, the read-only root, a full tmpfs, a missing
HOME, a dropped capability, the pids and memory caps, a syscall gVisor lacks) are INFRA, never a counted crash; that
check reads only exception lines, and it can only take a count away.
"""

from __future__ import annotations

import posixpath
import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from receipts.contracts import FailureType

HEADER = "Traceback (most recent call last):"
_SEPARATORS = (
    "The above exception was the direct cause of the following exception:",
    "During handling of the above exception, another exception occurred:",
)
_FRAME = re.compile(r'^  File "(?P<path>[^"\n]*)", line (?P<line>\d+)(?:, in (?P<func>.+))?$')
_REPEATED = re.compile(r"^  \[Previous line repeated \d+ more times?\]$")
_EXC_LINE = re.compile(r"^(?P<type>[A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)(?::(?: (?P<msg>.*))?)?$")

# Path parts inside the workspace that are an installed environment, not the repository's own code: a frame under one
# of them is treated like a frame outside the root. Pinned by a test.
_FRAMEWORK_MARKERS: frozenset[str] = frozenset({"site-packages", "dist-packages", ".tox", ".nox", ".venv", ".eggs"})
# A test frame whose function is one of these is a fixture phase, not the test's call.
_FIXTURE_HOOKS: frozenset[str] = frozenset(
    {
        "setUp",
        "tearDown",
        "setUpClass",
        "tearDownClass",
        "setUpTestData",
        "setUpModule",
        "tearDownModule",
        "asyncSetUp",
        "asyncTearDown",
        "setup_method",
        "teardown_method",
        "setup_function",
        "teardown_function",
        "setup_class",
        "teardown_class",
        "setup_module",
        "teardown_module",
    }
)
_IMPORT_ERRORS = frozenset({"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError", "TabError"})
_NOT_COUNTED = frozenset({"ExceptionGroup", "BaseExceptionGroup", "KeyboardInterrupt"})
# pytest.raises and pytest.warns report a missing exception or warning as Failed("DID NOT RAISE ...") and
# Failed("DID NOT WARN ..."); they count as assertion failures (REG-565).
_DID_NOT = ("DID NOT RAISE", "DID NOT WARN")
# pytest-timeout's failure: "Timeout (>1.0s) from pytest-timeout." in 2.x, "Timeout >1.0s" in 1.x.
_HARNESS_TIMEOUT = re.compile(r"Timeout (?:\(>[0-9.]+s\) from pytest-timeout\.|>[0-9.]+s)")

# errno values the sandbox itself causes: EPERM (dropped capabilities), EAGAIN (a fork or a thread past the pids cap),
# ENOMEM (the memory cap), ENOSPC and EDQUOT (a full tmpfs), EROFS (the read-only root), ENOSYS and EOPNOTSUPP
# (syscalls gVisor does not implement), EADDRNOTAVAIL, ENETDOWN, ENETUNREACH and EHOSTUNREACH (no network), and
# getaddrinfo's EAI_* codes (no resolver). A false blocker costs more than a missed one, so the cap side wins: an EAGAIN
# or an ENOSYS raised by changed code is never counted, even where the code could have raised it on its own.
_SANDBOX_ERRNOS = frozenset({1, 11, 12, 28, 30, 38, 95, 99, 100, 101, 113, 122, -2, -3, -4, -5, -9})
_SANDBOX_TEXT = (
    "Network is unreachable",
    "No route to host",
    "Cannot assign requested address",
    "Temporary failure in name resolution",
    "Name or service not known",
    "nodename nor servname provided",
    "Read-only file system",
    "No space left on device",
    "Disk quota exceeded",
    "Operation not permitted",
    "can't start new thread",  # RuntimeError from threading past the pids cap
    "Resource temporarily unavailable",
    "Cannot allocate memory",
    "Function not implemented",
    "Operation not supported",
)
# MemoryError is the memory cap's own exception; socket.gaierror is the missing resolver.
_SANDBOX_TYPES = frozenset({"socket.gaierror", "gaierror", "MemoryError"})
_ERRNO = re.compile(r"\[Errno (?P<errno>-?\d+)\](?P<rest>[^\n]*)")
_QUOTED_PATH = re.compile(r"""['"](?P<path>/[^'"\n]*)['"]""")
_RUN_HOME_DEFAULT = "/nonexistent"  # nobody's home in Debian and Ubuntu images: what HOME is unless the run sets one


@dataclass(frozen=True)
class Frame:
    path: str
    line: int
    function: str


@dataclass(frozen=True)
class Block:
    """One exception's traceback: its frames, outermost first, and its exception text (the exception line and every
    message line after it)."""

    frames: tuple[Frame, ...]
    exception: str
    remainder: str = ""  # the whole text from this block's exception line to the end of the traceback

    @property
    def exc_type(self) -> str | None:
        return exception_type(self.exception)


def exception_type(text: str) -> str | None:
    """The exception's type from its exception line ("ValueError: boom" gives "ValueError"), or None."""
    first = text.split("\n", 1)[0].rstrip()
    m = _EXC_LINE.fullmatch(first)
    return m.group("type") if m else None


def parse(text: str) -> tuple[Block, ...]:
    """Every traceback block in ``text``, in the order printed (the raised exception last)."""
    lines = text.replace("\r\n", "\n").split("\n")
    blocks: list[Block] = []
    i, n = 0, len(lines)
    while i < n:
        if lines[i].rstrip() != HEADER:
            i += 1
            continue
        frames, i = _read_frames(lines, i + 1)
        if i >= n or not lines[i] or lines[i][0].isspace():
            blocks.append(Block(tuple(frames), ""))  # a header and frames with no exception line: no type
            continue
        start = i
        i += 1
        while i < n and not _chain_starts(lines, i):
            i += 1
        exception = "\n".join(lines[start:i]).rstrip()
        blocks.append(Block(tuple(frames), exception, "\n".join(lines[start:]).rstrip()))
        if i < n:
            i += 3  # the blank line, the separator and the blank line; the next header follows
    return tuple(blocks)


def _read_frames(lines: Sequence[str], i: int) -> tuple[list[Frame], int]:
    frames: list[Frame] = []
    n = len(lines)
    while i < n:
        m = _FRAME.match(lines[i])
        if m is not None:
            frames.append(Frame(m.group("path"), int(m.group("line")), m.group("func") or ""))
            i += 1
            while i < n and lines[i].startswith("    "):
                i += 1  # the frame's source and caret lines
            continue
        if _REPEATED.match(lines[i]):
            i += 1
            continue
        break
    return frames, i


def _chain_starts(lines: Sequence[str], i: int) -> bool:
    return (
        i + 3 < len(lines)
        and lines[i] == ""
        and lines[i + 1] in _SEPARATORS
        and lines[i + 2] == ""
        and lines[i + 3].rstrip() == HEADER
    )


def final_block(blocks: Sequence[Block], exception_text: str | None = None) -> Block | None:
    """The raised exception's block. With ``exception_text`` (the runner's own record of the raised exception, such
    as junit's message), it is the block whose text from its exception line to the end is exactly that: the raised
    exception's message is printed last, so a block forged inside it is followed by less than the whole message and
    can never match. pytest's junit drops a leading "AssertionError: " from a rewritten assertion's message, so that
    spelling matches too. Without ``exception_text``, the last block."""
    if not blocks:
        return None
    if exception_text is None:
        return blocks[-1]
    wanted = exception_text.strip()
    spellings = {wanted, f"AssertionError: {wanted}"}
    for block in blocks:
        if block.remainder.strip() in spellings:
            return block
    return None


def repo_path(frame_path: str, root: str) -> str | None:
    """The frame's file relative to the workspace root, or None when it lies outside the root (or under an installed
    environment inside it). Relative frame paths resolve against the root, the runners' working directory."""
    if not frame_path or frame_path.startswith("<"):
        return None
    base = posixpath.normpath(root)
    full = posixpath.normpath(frame_path if frame_path.startswith("/") else posixpath.join(base, frame_path))
    if not full.startswith(base.rstrip("/") + "/"):
        return None
    rel = full[len(base.rstrip("/")) + 1 :]
    if any(part in _FRAMEWORK_MARKERS for part in rel.split("/")):
        return None
    return rel


def normalize(paths: Iterable[str]) -> frozenset[str]:
    """Repo-relative paths in one spelling: no leading ./, no doubled or trailing slashes."""
    out: set[str] = set()
    for path in paths:
        clean = posixpath.normpath(path.strip())
        if clean and clean != "." and not clean.startswith("/") and not clean.startswith("../"):
            out.add(clean)
    return frozenset(out)


def is_test_path(rel: str) -> bool:
    """A repository file that is test code: a test module by name, a conftest, or anything under a tests/ directory."""
    parts = rel.split("/")
    base = parts[-1]
    if base in ("conftest.py", "tests.py") or base.startswith("test_") or base.endswith(("_test.py", "_tests.py")):
        return True
    return "tests" in parts[:-1]


def attribute(frames: Sequence[Frame], *, root: str, changed: frozenset[str]) -> tuple[FailureType, str | None]:
    """Type a crash by where it happened: CRASH_IN_CHANGED only when the deepest frame inside the repository is a
    non-test file the patch changed. Returns the type and that frame's repo-relative file."""
    inside = [(frame, rel) for frame in frames if (rel := repo_path(frame.path, root)) is not None]
    if not inside:
        return FailureType.CRASH_ELSEWHERE, None
    tests = [frame for frame, rel in inside if is_test_path(rel)]
    if tests and tests[0].function in _FIXTURE_HOOKS:
        return FailureType.FIXTURE, None
    _, deepest = inside[-1]
    if is_test_path(deepest):
        return FailureType.CRASH_ELSEWHERE, None
    if deepest in changed:
        return FailureType.CRASH_IN_CHANGED, deepest
    return FailureType.CRASH_ELSEWHERE, deepest


def sandbox_caused(text: str, *, root: str, writable: Sequence[str] = ("/tmp", "/dev/shm")) -> bool:
    """True when an exception in the traceback ``text`` is one the sandbox causes: an errno or message the isolation
    or its caps produce, a MemoryError, a resolver failure, a path under the run user's default HOME, or a permission
    error on a path outside the writable mounts. Only lines at column 0 are read (exception lines and their messages, a
    cause printed without a traceback included), never the indented frame and source lines."""
    lines = [line for line in text.replace("\r\n", "\n").split("\n") if line and not line[0].isspace()]
    for line in lines:
        if exception_type(line) in _SANDBOX_TYPES or any(marker in line for marker in _SANDBOX_TEXT):
            return True
        for m in _ERRNO.finditer(line):
            errno = int(m.group("errno"))
            if errno in _SANDBOX_ERRNOS:
                return True
            path = _QUOTED_PATH.search(m.group("rest"))
            if path is None or errno not in (2, 13):
                continue
            where = posixpath.normpath(path.group("path"))
            if where == _RUN_HOME_DEFAULT or where.startswith(_RUN_HOME_DEFAULT + "/"):
                return True
            if errno == 13 and not any(where == w or where.startswith(w.rstrip("/") + "/") for w in (root, *writable)):
                return True
    return False


def type_failure(
    text: str,
    *,
    root: str,
    changed: frozenset[str],
    exception_text: str | None = None,
) -> tuple[FailureType, str | None]:
    """The FailureType of a test that raised, from its traceback ``text``, and the crash's repo-relative file.
    ``exception_text`` is the runner's own record of the raised exception when it keeps one (junit's message); it picks
    the raised exception's block, and with no matching block the crash is never counted."""
    final = final_block(parse(text), exception_text)
    if final is not None:
        exc_type, raised = final.exc_type, final.exception
    else:
        exc_type = exception_type(exception_text) if exception_text else None
        raised = exception_text or ""
    first = raised.split("\n", 1)[0]
    message = raised.split(": ", 1)[1] if ": " in first else ""
    if exc_type == "AssertionError":
        return FailureType.ASSERTION, None
    if exc_type == "Failed":
        if message.startswith(_DID_NOT):
            return FailureType.ASSERTION, None
        if _HARNESS_TIMEOUT.fullmatch(message.strip()):
            return FailureType.TIMEOUT, None
        return FailureType.OTHER, None
    if exc_type is None or exc_type in _NOT_COUNTED:
        return FailureType.OTHER, None
    if exc_type.rsplit(".", 1)[-1] in _IMPORT_ERRORS:
        return FailureType.IMPORT, None
    if exc_type == "NameError":
        return FailureType.NAME, None
    if sandbox_caused(text, root=root) or sandbox_caused(raised, root=root):
        return FailureType.INFRA, None
    if final is None:
        return FailureType.CRASH_ELSEWHERE, None
    return attribute(final.frames, root=root, changed=changed)
