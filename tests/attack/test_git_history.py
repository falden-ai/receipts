"""The history strip must leave exactly the commit we intend, and the assertion must catch whatever a strip leaves
behind: a ref, a reflog entry, a stash, a leftover HEAD file, a dangling commit or a nested repository. No git command
the lane runs may leave a gc or maintenance running after it returns, and a gc that did not run is an error."""

from __future__ import annotations

import hashlib
import json
import os
import socket
import subprocess
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from receipts.attack import git_history
from receipts.attack.git_history import (
    HistoryStripError,
    assert_history_stripped,
    find_nested_git,
    init_single_commit,
    reachable_commit_count,
    stored_commit_count,
    strip_git_history,
    strip_refs_in_place,
    strip_workspace,
)
from receipts.attack.sandbox import apply_patch

_ENV = {
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "t",
    "GIT_AUTHOR_EMAIL": "t@localhost",
    "GIT_COMMITTER_NAME": "t",
    "GIT_COMMITTER_EMAIL": "t@localhost",
}
# The test's own commits start no background maintenance either, so none runs under the code a test checks.
_QUIET = ("-c", "gc.auto=0", "-c", "maintenance.auto=false")


def _git(repo: Path, *args: str) -> str:
    proc = subprocess.run(
        ["git", "-C", str(repo), "-c", "commit.gpgsign=false", *_QUIET, *args],
        capture_output=True,
        text=True,
        env=_ENV,
        check=True,
    )
    return proc.stdout.strip()


def _make_repo(root: Path, *, commits: int, name: str = "repo") -> Path:
    repo = root / name
    repo.mkdir(parents=True)
    _git(repo, "init", "-q", "-b", "main")
    for i in range(commits):
        (repo / "file.txt").write_text(f"version {i}\n")
        _git(repo, "add", "-A")
        _git(repo, "commit", "-q", "-m", f"commit {i}")
    return repo


def _object_exists(repo: Path, sha: str) -> bool:
    proc = subprocess.run(["git", "-C", str(repo), "cat-file", "-e", sha], capture_output=True, env=_ENV, check=False)
    return proc.returncode == 0


def test_strip_rebuilds_to_a_single_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=3)
    assert reachable_commit_count(repo) >= 3
    (repo / "patched.txt").write_text("applied\n")  # a patch applied to the tree, not yet committed
    strip_git_history(repo)
    assert reachable_commit_count(repo) == 1
    assert stored_commit_count(repo) == 1
    assert_history_stripped(repo)  # does not raise
    assert (repo / "file.txt").read_text() == "version 2\n"
    assert (repo / "patched.txt").exists()  # the working tree, including the patch, survived


def test_strip_in_place_keeps_head_but_removes_other_refs(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=2)
    _git(repo, "tag", "v1")
    _git(repo, "branch", "feature")
    strip_refs_in_place(repo)
    assert _git(repo, "for-each-ref", "--format=%(refname)") == ""


def test_strip_in_place_expires_reflogs_and_prunes_a_future_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=2)
    base = _git(repo, "rev-parse", "HEAD")
    # A future commit, reachable from a branch and from HEAD's reflog (HEAD moved to it and back).
    future = _git(repo, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "the future fix")
    _git(repo, "update-ref", "refs/heads/future", future)
    _git(repo, "update-ref", "-m", "to the future", "HEAD", future)
    _git(repo, "update-ref", "-m", "back to base", "HEAD", base)
    assert reachable_commit_count(repo) == 3
    strip_refs_in_place(repo)
    # Without the reflog expire, HEAD's reflog keeps the future commit reachable (3); without gc it stays stored.
    assert reachable_commit_count(repo) == 2
    assert not _object_exists(repo, future)
    assert_history_stripped(repo, expected_commits=2)


def test_assert_rejects_an_unstripped_repo(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=2)
    with pytest.raises(HistoryStripError, match="2 commits reachable"):
        assert_history_stripped(repo, expected_commits=1)


def test_assert_catches_a_ref_left_behind(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=3)
    strip_git_history(repo)
    assert_history_stripped(repo)  # clean
    # PLANT: a strip that forgot a ref. Fabricate a parentless commit and point a branch at it.
    tree = _git(repo, "rev-parse", "HEAD^{tree}")
    orphan = _git(repo, "commit-tree", tree, "-m", "leaked future commit")
    _git(repo, "update-ref", "refs/heads/leak", orphan)
    with pytest.raises(HistoryStripError, match="2 commits reachable"):
        assert_history_stripped(repo, expected_commits=1)


def test_assert_catches_a_commit_only_a_reflog_reaches(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=1)
    strip_git_history(repo)
    base = _git(repo, "rev-parse", "HEAD")
    # PLANT: main moved to a future commit and back, so only the reflog still names the future commit.
    future = _git(repo, "commit-tree", f"{base}^{{tree}}", "-p", base, "-m", "future")
    _git(repo, "update-ref", "-m", "forward", "refs/heads/main", future)
    _git(repo, "update-ref", "-m", "back", "refs/heads/main", base)
    assert _git(repo, "rev-list", "--count", "--all") == "1"  # refs alone do not see it
    with pytest.raises(HistoryStripError, match="2 commits reachable"):
        assert_history_stripped(repo, expected_commits=1)


def test_assert_catches_a_dangling_commit(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=1)
    strip_git_history(repo)
    # PLANT: a commit object no ref or reflog names, still readable by its id (git show <sha>).
    dangling = _git(repo, "commit-tree", "HEAD^{tree}", "-m", "the gold patch")
    assert reachable_commit_count(repo) == 1
    assert _object_exists(repo, dangling)
    with pytest.raises(HistoryStripError, match="2 commit objects stored"):
        assert_history_stripped(repo, expected_commits=1)


def test_assert_rejects_a_repo_with_a_stash(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=1)
    strip_git_history(repo)
    # PLANT: a stash left behind.
    (repo / "file.txt").write_text("dirty\n")
    _git(repo, "stash", "push", "-u", "-m", "leftover")
    with pytest.raises(HistoryStripError, match="stash entr"):
        assert_history_stripped(repo, expected_commits=1)


@pytest.mark.parametrize("leftover", ["ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD"])
def test_assert_rejects_a_leftover_head_file(tmp_path: Path, leftover: str) -> None:
    repo = _make_repo(tmp_path, commits=1)
    strip_git_history(repo)
    # PLANT: the file a fetch, merge or reset leaves; it names a commit by id, and git log <name> follows it.
    (repo / ".git" / leftover).write_text(_git(repo, "rev-parse", "HEAD") + "\n")
    with pytest.raises(HistoryStripError, match=f"leftover {leftover}"):
        assert_history_stripped(repo, expected_commits=1)


def test_a_nested_repository_is_removed_and_its_files_kept(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=2)
    nested = _make_repo(repo / "vendor", commits=2, name="lib")
    assert find_nested_git(repo) == ["vendor/lib/.git"]
    strip_git_history(repo)
    assert find_nested_git(repo) == []
    assert (nested / "file.txt").read_text() == "version 1\n"
    assert "vendor/lib/file.txt" in _git(repo, "ls-files").splitlines()  # a plain file now, not a gitlink


def test_assert_rejects_a_nested_repository(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=1)
    strip_git_history(repo)
    # PLANT: a nested repository with its own history appears after the strip.
    _make_repo(repo / "vendor", commits=2, name="lib")
    with pytest.raises(HistoryStripError, match=r"nested repositories remain at vendor/lib/\.git"):
        assert_history_stripped(repo, expected_commits=1)


def test_strip_workspace_handles_a_checkout_without_git(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    root.mkdir()
    (root / "a.py").write_text("x = 1\n")
    _make_repo(root / "third_party", commits=2, name="dep")
    assert strip_workspace(root) is None
    assert find_nested_git(root) == []
    assert (root / "third_party" / "dep" / "file.txt").exists()


def test_strip_workspace_rebuilds_a_repository(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=3)
    sha = strip_workspace(repo)
    assert sha == _git(repo, "rev-parse", "HEAD")
    assert reachable_commit_count(repo) == 1


def test_init_single_commit_makes_a_tree_without_history_one_commit(tmp_path: Path) -> None:
    root = tmp_path / "tree"
    (root / "pkg").mkdir(parents=True)
    (root / "pkg" / "mod.py").write_text("x = 1\n")
    (root / ".gitignore").write_text("*.so\n")
    (root / "pkg" / "ext.so").write_bytes(b"\x7fELF")  # an ignored build artifact is still part of the tree
    _make_repo(root / "vendor", commits=2, name="dep")  # a nested repository with its own history
    sha = init_single_commit(root)
    assert sha == _git(root, "rev-parse", "HEAD")
    assert_history_stripped(root)
    tracked = _git(root, "ls-files").splitlines()
    assert {"pkg/mod.py", "pkg/ext.so", "vendor/dep/file.txt"} <= set(tracked)
    assert find_nested_git(root) == []


def test_init_single_commit_refuses_a_tree_that_has_a_git_directory(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=2)
    with pytest.raises(HistoryStripError, match=r"already has a \.git"):
        init_single_commit(repo)


# -- no git work left running (round 7, R3) ---------------------------------------------------------------------------


def _blobs_in_17(count: int) -> dict[str, bytes]:
    """Files whose blob ids start with 17. `git gc --auto` counts the loose objects in .git/objects/17 alone and acts
    once there are more than gc.auto / 256 of them, rounded up (27 for the default 6700: on git 2.54, 27 stay loose
    and 28 are packed), so 28 put a tree over the line R24's 9,000-file trees crossed, at a test's cost."""
    out: dict[str, bytes] = {}
    i = 0
    while len(out) < count:
        data = b"%d\n" % i
        i += 1
        if hashlib.sha1(b"blob %d\0" % len(data) + data, usedforsecurity=False).hexdigest().startswith("17"):
            out[f"f{len(out)}.txt"] = data
    return out


def _tree_over_the_line(root: Path) -> Path:
    root.mkdir(parents=True)
    for name, data in _blobs_in_17(28).items():
        (root / name).write_bytes(data)
    return root


def _events(trace: Path) -> list[dict[str, Any]]:
    """git's trace2 events (GIT_TRACE2_EVENT): one JSON object a line, from every git process, a detached one too. A
    line a process left in the middle of writing it is skipped."""
    if not trace.exists():
        return []
    out: list[dict[str, Any]] = []
    for line in trace.read_text().splitlines():
        try:
            out.append(json.loads(line))
        except ValueError:
            continue
    return out


def _our_lock() -> str:
    """A gc.pid as git writes one ("<pid> <host>"), naming this live process on this host: an explicit `git gc` that
    finds it refuses to run (exit 128, "gc is already running")."""
    return f"{os.getpid()} {socket.gethostname()}"


def _gc_running(repo: Path) -> bool:
    """Whether a gc holds `repo`: a gc.pid other than the one a test planted."""
    try:
        return (repo / ".git" / "gc.pid").read_text(errors="replace") != _our_lock()
    except OSError:
        return False


def _settle(repo: Path, trace: Path, timeout_s: float = 30.0) -> None:
    """Wait until no git process is still at work in `repo` (no gc holds it, the trace unchanged for half a second), so
    a detached gc the code under test left behind ends before the test does."""
    deadline = time.monotonic() + timeout_s
    size = -1
    while time.monotonic() < deadline:
        now = trace.stat().st_size if trace.exists() else 0
        if now == size and not _gc_running(repo):
            return
        size = now
        time.sleep(0.5)


def _strip_a_repository(tree: Path) -> object:
    _git(tree, "init", "-q", "-b", "main")
    _git(tree, "add", "-A")
    _git(tree, "commit", "-q", "-m", "the history the strip throws away")
    return strip_workspace(tree)


# Configuration that asks for every kind of background work, for a test to put where git reads it.
_ASKS_FOR_BACKGROUND = "[gc]\n\tauto = 1\n\tautoDetach = true\n[maintenance]\n\tauto = true\n\tautoDetach = true\n"


def _configure(where: str, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Git's defaults (GIT_ENV seals the global and system files), a global config file that asks for background
    work, or the repository's own config asking for it (a template `git init` copies into .git/config)."""
    if where == "global":
        config = tmp_path / "gitconfig"
        config.write_text(_ASKS_FOR_BACKGROUND)
        monkeypatch.setitem(git_history.GIT_ENV, "GIT_CONFIG_GLOBAL", str(config))
    elif where == "repository":
        template = tmp_path / "config-template"
        template.mkdir()
        (template / "config").write_text(_ASKS_FOR_BACKGROUND)
        monkeypatch.setitem(git_history.GIT_ENV, "GIT_TEMPLATE_DIR", str(template))


@pytest.mark.parametrize("where", ["defaults", "global", "repository"])
@pytest.mark.parametrize("strip", [init_single_commit, _strip_a_repository], ids=["init_single_commit", "rebuild"])
def test_plant_no_strip_leaves_a_gc_or_maintenance_running(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, strip: Callable[[Path], object], where: str
) -> None:
    """R24's bug D (matplotlib__matplotlib-20676's first attempt, git 2.43.0): right after init_single_commit's commit
    a detached `git gc` was running, the explicit `git gc --prune=now` exited 128 against its lock (swallowed), and the
    sandbox's copytree of the cached tree raised ENOENT while the detached gc packed the loose objects away. git 2.54
    here starts `git maintenance run --auto --quiet --detach` after every commit. trace2 records every git process
    that starts, a detached one included: when the strip returns, none may have been an auto gc or maintenance, none
    may have detached, no gc.pid may be left, and its own gc must have run and exited 0, whatever config asks for."""
    tree = _tree_over_the_line(tmp_path / "tree")
    _configure(where, monkeypatch, tmp_path)
    trace = tmp_path / "trace2.json"
    monkeypatch.setitem(git_history.GIT_ENV, "GIT_TRACE2_EVENT", str(trace))
    try:
        strip(tree)
        returned = _events(trace)
        locked = (tree / ".git" / "gc.pid").exists()
    finally:
        _settle(tree, trace)
    auto = [e["argv"] for e in returned if e["event"] == "start" and "--auto" in e["argv"]]
    detached = [e["category"] for e in returned if e["event"] == "region_enter" and e.get("label") == "detach"]
    assert (auto, detached, locked) == ([], [], False)
    gcs = {e["sid"] for e in returned if e["event"] == "start" and e["argv"][-3:] == ["gc", "--prune=now", "--quiet"]}
    assert len(gcs) == 1
    assert [e["code"] for e in returned if e["event"] == "exit" and e["sid"] in gcs] == [0]
    assert_history_stripped(tree)


def _lock_template(tmp_path: Path) -> Path:
    """A git template holding a live gc lock (``_our_lock``), which `git init` copies into the new .git."""
    template = tmp_path / "template"
    template.mkdir()
    (template / "gc.pid").write_text(_our_lock())
    return template


@pytest.mark.parametrize("strip", [init_single_commit, _strip_a_repository], ids=["init_single_commit", "rebuild"])
def test_plant_a_strip_whose_gc_did_not_run_raises(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, strip: Callable[[Path], object]
) -> None:
    """The explicit gc ran with check=False, so one git refused because another gc held the repository (bug D's exit
    128) passed silently, and the tree went on while the other gc still changed it. It is an error now."""
    tree = _tree_over_the_line(tmp_path / "tree")
    monkeypatch.setitem(git_history.GIT_ENV, "GIT_TEMPLATE_DIR", str(_lock_template(tmp_path)))
    try:
        with pytest.raises(HistoryStripError, match="gc is already running"):
            strip(tree)
    finally:
        _settle(tree, tmp_path / "no-trace")


def test_plant_strip_refs_in_place_raises_when_its_gc_did_not_run(tmp_path: Path) -> None:
    repo = _make_repo(tmp_path, commits=2)
    (repo / ".git" / "gc.pid").write_text(_our_lock())
    try:
        with pytest.raises(HistoryStripError, match="gc is already running"):
            strip_refs_in_place(repo)
    finally:
        _settle(repo, tmp_path / "no-trace")


_NO_BACKGROUND = {"gc.auto=0", "gc.autoDetach=false", "maintenance.auto=false", "maintenance.autoDetach=false"}


def _subcommand(argv: list[str]) -> tuple[str, set[str]]:
    """A git argument list's subcommand and the -c settings in front of it."""
    settings: set[str] = set()
    i = 1
    while argv[i] in ("-C", "-c"):
        if argv[i] == "-c":
            settings.add(argv[i + 1])
        i += 2
    return argv[i], settings


def test_every_git_command_the_lane_runs_carries_the_settings_that_turn_background_work_off(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Every git invocation in the attack lane goes through git_history._git or sandbox._git; each of their callers
    runs here, and every git command they start carries the four settings in front of its subcommand."""
    repo = _make_repo(tmp_path, commits=2, name="history")
    base = _git(repo, "rev-parse", "HEAD")
    (repo / "file.txt").write_text("changed\n")
    _git(repo, "commit", "-q", "-am", "next")
    tree = tmp_path / "tree"
    tree.mkdir()
    (tree / "a.txt").write_text("x\n")
    plain = tmp_path / "plain"
    plain.mkdir()
    seen: list[list[str]] = []
    real = subprocess.run

    def recording(argv: Any, *args: Any, **kwargs: Any) -> Any:
        if isinstance(argv, list) and argv[:1] == ["git"]:
            seen.append([str(a) for a in argv])
        return real(argv, *args, **kwargs)

    monkeypatch.setattr(subprocess, "run", recording)
    assert git_history.git_changed_files(repo, base, "HEAD") == ["file.txt"]
    strip_refs_in_place(repo)
    init_single_commit(tree)
    patch = "diff --git a/a.txt b/a.txt\n--- a/a.txt\n+++ b/a.txt\n@@ -1 +1 @@\n-x\n+y\n"
    apply_patch(tree, patch)
    strip_workspace(tree)
    strip_workspace(plain)
    assert git_history.is_git_repo(tree) and not git_history.is_git_repo(plain)
    monkeypatch.setattr(subprocess, "run", real)
    found = [_subcommand(argv) for argv in seen]
    assert {name for name, _ in found} == {
        "add", "apply", "cat-file", "checkout", "commit", "diff", "for-each-ref", "gc", "init", "reflog", "remote",
        "rev-list", "rev-parse", "stash", "update-ref",
    }  # fmt: skip
    assert [name for name, settings in found if not settings >= _NO_BACKGROUND] == []
