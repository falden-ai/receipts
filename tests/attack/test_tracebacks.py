"""The traceback reader: frames come only from a traceback's structure, resolve only inside the workspace root, and a
crash counts only when its deepest repository frame is a changed non-test file. Each plant is the input that fooled
an earlier version (the review's spoofs) or a case the rule must refuse.
"""

from __future__ import annotations

import pytest

from receipts.attack import tracebacks
from receipts.attack.tracebacks import Frame, attribute, final_block, parse, repo_path, type_failure
from receipts.contracts import FailureType

ROOT = "/testbed"


def tb(*frames: tuple[str, int, str], exc: str = "ValueError: boom") -> str:
    lines = [tracebacks.HEADER]
    for path, line, func in frames:
        lines += [f'  File "{path}", line {line}, in {func}', "    source_line()"]
    return "\n".join([*lines, exc])


TEST = ("/testbed/tests/test_x.py", 5, "test_x")
CHANGED = ("/testbed/src/pkg/mod.py", 2, "divide")
STDLIB = ("/opt/miniconda3/envs/testbed/lib/python3.9/json/decoder.py", 355, "raw_decode")


def typed(text: str, changed: tuple[str, ...] = ("src/pkg/mod.py",), **kw: str) -> tuple[FailureType, str | None]:
    return type_failure(text, root=ROOT, changed=frozenset(changed), **kw)


# -- structure ---------------------------------------------------------------------------------------------------------


def test_frames_come_only_from_the_frames_section() -> None:
    text = tb(TEST, exc='ValueError: boom\n  File "/testbed/src/pkg/mod.py", line 2, in divide\n    return a / b')
    (block,) = parse(text)
    assert [f.path for f in block.frames] == ["/testbed/tests/test_x.py"]
    assert block.exc_type == "ValueError"
    # PLANT (the old column-0 regex): a frame-shaped line inside the message made this a crash in changed code.
    assert typed(text) == (FailureType.CRASH_ELSEWHERE, None)


def test_source_and_caret_lines_and_repeats_stay_inside_the_frames_section() -> None:
    text = "\n".join(
        [
            tracebacks.HEADER,
            '  File "/testbed/tests/test_x.py", line 5, in test_x',
            "    divide(",
            "    ~~~~~~^",
            "        1, 0)",
            "  [Previous line repeated 996 more times]",
            '  File "/testbed/src/pkg/mod.py", line 2, in divide',
            "    return a / b",
            "           ~~^~~",
            "ZeroDivisionError: division by zero",
        ]
    )
    (block,) = parse(text)
    assert [f.line for f in block.frames] == [5, 2]
    assert typed(text) == (FailureType.CRASH_IN_CHANGED, "src/pkg/mod.py")


def test_a_chain_is_read_in_order_and_the_raised_exception_is_last() -> None:
    first = tb(TEST, CHANGED, exc="ZeroDivisionError: division by zero")
    second = tb(TEST, exc="RuntimeError: wrapped")
    sep = "\n\nThe above exception was the direct cause of the following exception:\n\n"
    blocks = parse(first + sep + second)
    assert [b.exc_type for b in blocks] == ["ZeroDivisionError", "RuntimeError"]
    assert final_block(blocks) is blocks[-1]
    # The raised exception is the test's own RuntimeError, so the cause's frames in changed code do not count.
    assert typed(first + sep + second) == (FailureType.CRASH_ELSEWHERE, None)


def test_a_forged_chain_in_the_message_never_matches_the_runners_record() -> None:
    forged = "\n\nThe above exception was the direct cause of the following exception:\n\n" + tb(
        CHANGED, exc="ZeroDivisionError: forged"
    )
    text = tb(TEST, exc="ValueError: boom" + forged)
    blocks = parse(text)
    assert len(blocks) == 2  # the parse alone cannot tell the forged block from a real one
    real = final_block(blocks, "ValueError: boom" + forged)
    assert real is blocks[0]
    assert typed(text, exception_text="ValueError: boom" + forged) == (FailureType.CRASH_ELSEWHERE, None)
    # PLANT: trusting the last block (no runner record) is exactly what the forged block exploits.
    assert typed(text) == (FailureType.CRASH_IN_CHANGED, "src/pkg/mod.py")
    # A record that matches no block is never a counted crash.
    assert typed(text, exception_text="ValueError: something else") == (FailureType.CRASH_ELSEWHERE, None)


def test_pytest_drops_the_assertion_prefix_and_the_block_still_matches() -> None:
    text = tb(TEST, exc="AssertionError: assert (1 + 1) == 3")
    assert final_block(parse(text), "assert (1 + 1) == 3") is not None
    assert typed(text, exception_text="assert (1 + 1) == 3") == (FailureType.ASSERTION, None)


def test_text_with_no_native_traceback_counts_nothing() -> None:
    long_style = "tests/test_x.py:5: in test_x\n    divide(1, 0)\nsrc/pkg/mod.py:2: in divide\nE   ZeroDivisionError"
    assert parse(long_style) == ()
    assert typed(long_style, exception_text="ZeroDivisionError: division by zero") == (
        FailureType.CRASH_ELSEWHERE,
        None,
    )
    assert typed(long_style, exception_text="assert (1 + 1) == 3") == (FailureType.OTHER, None)


# -- resolution against the workspace root -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "rel"),
    [
        ("/testbed/src/pkg/mod.py", "src/pkg/mod.py"),
        ("src/pkg/mod.py", "src/pkg/mod.py"),  # relative: resolved against the root, the runner's cwd
        ("/testbed/./src//pkg/mod.py", "src/pkg/mod.py"),
        ("/testbed/../etc/utils.py", None),
        ("../outside.py", None),
        ("/usr/local/lib/python3.13/email/utils.py", None),
        ("/testbedx/utils.py", None),  # a sibling directory that only shares the prefix
        ("<frozen importlib._bootstrap>", None),
        ("<string>", None),
        ("/testbed/.venv/lib/python3.9/site-packages/dep/__init__.py", None),
        ("/testbed/deps/site-packages/dep/__init__.py", None),
        ("/testbed/.tox/py39/lib/x.py", None),
        ("/testbed/src/_pytest/python.py", "src/_pytest/python.py"),  # pytest's own repository is the repository
    ],
)
def test_frames_resolve_only_inside_the_root(path: str, rel: str | None) -> None:
    assert repo_path(path, ROOT) == rel


def test_framework_markers_are_pinned() -> None:
    # Path parts inside the workspace that are an installed environment, not repository code. Changing the set
    # changes which crashes count, so it is pinned here.
    assert {"site-packages", "dist-packages", ".tox", ".nox", ".venv", ".eggs"} == tracebacks._FRAMEWORK_MARKERS


@pytest.mark.parametrize("marker", sorted(tracebacks._FRAMEWORK_MARKERS))
def test_each_marker_makes_a_frame_transparent(marker: str) -> None:
    changed = (f"{marker}/dep/mod.py",)
    text = tb(TEST, (f"/testbed/{marker}/dep/mod.py", 1, "f"))
    # The marked frame is environment: the deepest repository frame is the test, so the crash does not count, even
    # though the marked file is in the changed list.
    assert typed(text, changed) == (FailureType.CRASH_ELSEWHERE, None)


def test_a_one_component_changed_path_never_matches_the_standard_library() -> None:
    # PLANT (review blocker): changed 'utils.py' matched /usr/.../email/utils.py by its trailing component.
    text = tb(("/testbed/tests/test_repro.py", 5, "test_repro"), ("/usr/lib/python3.13/email/utils.py", 318, "p"))
    assert typed(text, ("utils.py",)) == (FailureType.CRASH_ELSEWHERE, None)
    in_repo = tb(("/testbed/tests/test_repro.py", 5, "test_repro"), ("/testbed/utils.py", 2, "helper"))
    assert typed(in_repo, ("utils.py",)) == (FailureType.CRASH_IN_CHANGED, "utils.py")
    assert typed(in_repo, ("pkg/utils.py",)) == (FailureType.CRASH_ELSEWHERE, "utils.py")


def test_inside_the_root_only_the_identical_repo_relative_path_matches() -> None:
    # PLANT (review test gap): a suffix match inside the root (changed 'utils.py', a crash in pkg/utils.py) survived
    # the whole suite. The crash is in a file the patch did not change, so it never counts.
    nested = tb(("/testbed/tests/test_repro.py", 5, "test_repro"), ("/testbed/pkg/utils.py", 2, "helper"))
    assert typed(nested, ("utils.py",)) == (FailureType.CRASH_ELSEWHERE, "pkg/utils.py")
    assert typed(nested, ("kg/utils.py",)) == (FailureType.CRASH_ELSEWHERE, "pkg/utils.py")
    assert typed(nested, ("pkg/utils.py",)) == (FailureType.CRASH_IN_CHANGED, "pkg/utils.py")


# -- attribution -------------------------------------------------------------------------------------------------------


def test_a_library_frame_raised_by_changed_code_is_a_crash_in_changed_code() -> None:
    # The decision pinned here: frames outside the repository are transparent, so a crash the standard library raises
    # for changed code counts the same as one a builtin raises, which leaves no frame.
    assert typed(tb(TEST, CHANGED, STDLIB)) == (FailureType.CRASH_IN_CHANGED, "src/pkg/mod.py")
    assert typed(tb(TEST, CHANGED)) == (FailureType.CRASH_IN_CHANGED, "src/pkg/mod.py")
    assert typed(tb(TEST, STDLIB)) == (FailureType.CRASH_ELSEWHERE, None)  # the test called the library itself


def test_a_test_callback_raising_through_changed_code_is_the_tests_own_crash() -> None:
    callback = ("/testbed/tests/test_x.py", 9, "boom")
    assert typed(tb(TEST, CHANGED, callback)) == (FailureType.CRASH_ELSEWHERE, None)


def test_an_unchanged_repository_frame_is_elsewhere() -> None:
    other = ("/testbed/src/pkg/other.py", 3, "helper")
    assert typed(tb(TEST, CHANGED, other)) == (FailureType.CRASH_ELSEWHERE, "src/pkg/other.py")


@pytest.mark.parametrize(
    "path",
    ["tests/helpers.py", "tests/app/models.py", "pkg/tests/util.py", "conftest.py", "test_x.py", "pkg/x_test.py"],
)
def test_test_code_is_never_changed_code(path: str) -> None:
    assert tracebacks.is_test_path(path)
    assert typed(tb(TEST, (f"/testbed/{path}", 1, "f")), (path,)) == (FailureType.CRASH_ELSEWHERE, None)


@pytest.mark.parametrize("path", ["django/test/testcases.py", "lib/matplotlib/testing/decorators.py", "sympy/utils.py"])
def test_library_code_named_for_testing_is_not_test_code(path: str) -> None:
    assert not tracebacks.is_test_path(path)


@pytest.mark.parametrize("hook", ["setUp", "setUpClass", "setUpTestData", "setup_method", "tearDown"])
def test_a_crash_in_a_setup_hook_is_a_fixture_error(hook: str) -> None:
    text = tb(("/testbed/tests/test_x.py", 3, hook), CHANGED)
    assert typed(text) == (FailureType.FIXTURE, None)


def test_attribute_needs_a_repository_frame() -> None:
    frames = [Frame("/usr/lib/python3.13/json/decoder.py", 1, "f")]
    assert attribute(frames, root=ROOT, changed=frozenset({"src/pkg/mod.py"})) == (FailureType.CRASH_ELSEWHERE, None)


# -- types that never count, and the two that do -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("exc", "ftype"),
    [
        ("AssertionError: 1 != 2", FailureType.ASSERTION),
        ("Failed: DID NOT RAISE ValueError", FailureType.ASSERTION),
        ("Failed: DID NOT RAISE <class 'ValueError'>", FailureType.ASSERTION),
        ("Failed: DID NOT WARN. No warnings of type (<class 'UserWarning'>,) were emitted.", FailureType.ASSERTION),
        ("Failed: Timeout (>1.0s) from pytest-timeout.", FailureType.TIMEOUT),
        ("Failed: Timeout >1.0s", FailureType.TIMEOUT),
        ("Failed: some other verdict", FailureType.OTHER),
        ("ModuleNotFoundError: No module named 'x'", FailureType.IMPORT),
        ("ImportError: cannot import name 'y'", FailureType.IMPORT),
        ("SyntaxError: invalid syntax", FailureType.IMPORT),
        ("NameError: name 'z' is not defined", FailureType.NAME),
        ("ExceptionGroup: eg (2 sub-exceptions)", FailureType.OTHER),
        ("mypkg.AssertionError: a look-alike is a crash", FailureType.CRASH_IN_CHANGED),
    ],
)
def test_exception_types(exc: str, ftype: FailureType) -> None:
    assert typed(tb(TEST, CHANGED, exc=exc))[0] is ftype


# -- errors the sandbox causes -----------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "exc",
    [
        "OSError: [Errno 101] Network is unreachable",
        "ConnectionError: [Errno 113] No route to host",
        "OSError: [Errno 99] Cannot assign requested address",
        "socket.gaierror: [Errno -3] Temporary failure in name resolution",
        "socket.gaierror: [Errno -2] Name or service not known",
        "urllib.error.URLError: <urlopen error [Errno -3] Temporary failure in name resolution>",
        "OSError: [Errno 30] Read-only file system: '/usr/lib/x'",
        "OSError: [Errno 28] No space left on device",
        "OSError: [Errno 122] Disk quota exceeded",
        "PermissionError: [Errno 1] Operation not permitted",
        "FileNotFoundError: [Errno 2] No such file or directory: '/nonexistent/.config/app.toml'",
        "PermissionError: [Errno 13] Permission denied: '/nonexistent'",
        "PermissionError: [Errno 13] Permission denied: '/root/.cache'",
        # The run phase's own caps (integrator call, round 3): the pids cap and the memory cap.
        "RuntimeError: can't start new thread",
        "BlockingIOError: [Errno 11] Resource temporarily unavailable",
        "MemoryError",
        "MemoryError: cannot allocate 4096 bytes",
        "OSError: [Errno 12] Cannot allocate memory",
        # Syscalls gVisor does not implement.
        "OSError: [Errno 38] Function not implemented",
        "OSError: [Errno 95] Operation not supported",
        # The same errnos with a library's own words: the errno alone decides.
        "OSError: [Errno 38] fallocate failed",
        "OSError: [Errno 95] O_TMPFILE refused",
        "BlockingIOError: [Errno 11] fork() failed",
        "OSError: [Errno 12] mmap of 1048576 bytes failed",
    ],
)
def test_errors_the_sandbox_causes_are_infra_never_a_counted_crash(exc: str) -> None:
    # PLANT (review note): each of these, raised in a changed file, was fail/crash_in_changed.
    assert typed(tb(TEST, CHANGED, exc=exc)) == (FailureType.INFRA, None)


def test_a_cap_error_raised_inside_the_standard_library_for_changed_code_is_infra() -> None:
    # subprocess forking past the pids cap: the deepest repository frame is changed code, the raise is the library's.
    fork = ("/opt/miniconda3/envs/testbed/lib/python3.9/subprocess.py", 1821, "_execute_child")
    text = tb(TEST, CHANGED, fork, exc="BlockingIOError: [Errno 11] Resource temporarily unavailable")
    assert typed(text) == (FailureType.INFRA, None)
    thread = ("/opt/miniconda3/envs/testbed/lib/python3.9/threading.py", 892, "start")
    assert typed(tb(TEST, CHANGED, thread, exc="RuntimeError: can't start new thread")) == (FailureType.INFRA, None)


def test_a_sandbox_error_as_the_cause_of_the_raised_exception_is_infra() -> None:
    # A cause created without a traceback prints with no header of its own.
    text = "OSError: [Errno 101] Network is unreachable\n\nThe above exception was the direct cause of the following "
    text += "exception:\n\n" + tb(TEST, CHANGED, exc="RuntimeError: fetch failed")
    assert typed(text) == (FailureType.INFRA, None)


@pytest.mark.parametrize(
    "exc",
    [
        "FileNotFoundError: [Errno 2] No such file or directory: '/testbed/missing.cfg'",
        "PermissionError: [Errno 13] Permission denied: '/testbed/locked.txt'",
        "PermissionError: [Errno 13] Permission denied: '/tmp/x'",
        "ConnectionRefusedError: [Errno 111] Connection refused",
        "ValueError: the word Network appears, unreachable too",
    ],
)
def test_ordinary_errors_in_changed_code_still_count(exc: str) -> None:
    assert typed(tb(TEST, CHANGED, exc=exc)) == (FailureType.CRASH_IN_CHANGED, "src/pkg/mod.py")


def test_the_sandbox_check_never_reads_source_lines() -> None:
    text = "\n".join(
        [
            tracebacks.HEADER,
            '  File "/testbed/src/pkg/mod.py", line 2, in divide',
            '    msg = "[Errno 101] Network is unreachable"',
            "ZeroDivisionError: division by zero",
        ]
    )
    assert typed(text) == (FailureType.CRASH_IN_CHANGED, "src/pkg/mod.py")


def test_an_assertion_mentioning_a_sandbox_errno_is_still_an_assertion() -> None:
    assert typed(tb(TEST, exc="AssertionError: [Errno 30] Read-only file system")) == (FailureType.ASSERTION, None)
