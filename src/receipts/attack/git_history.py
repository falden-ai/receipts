"""Strip a workspace's git history so no future commit, gold patch or other ref stays reachable (RT2).

The eval must not let code running in the sandbox read the fix it is being tested against. The surest way is to
throw the history away and rebuild the checkout as a single commit of the base tree: then ``git log --all`` can only
ever show that one commit. For callers that must keep the real base sha, ``strip_refs_in_place`` does the thorough
in-place removal the spec lists (refs, packed-refs, ORIG_HEAD, FETCH_HEAD, stash, reflogs, then ``gc --prune=now``).
A repository nested inside the checkout (a vendored repo or a submodule's own ``.git``) keeps its own history, so its
``.git`` is removed first and its files become plain files of the single commit.

Either way the guarantee is checked, not assumed: ``assert_history_stripped`` counts the commits reachable from every
ref and every reflog, counts the commit objects stored at all (a dangling commit can still be read by id), and looks
for a stash, leftover HEAD files and nested ``.git`` entries. It raises loudly on any of them.

No git command here leaves work running after it returns (``NO_BACKGROUND``): ``git commit`` would otherwise start
git's auto maintenance, whose gc detaches by default and goes on deleting loose objects after the commit returned,
so a copy of the tree made next could lose files under it (R24 on git 2.43.0: ``shutil.copytree`` of a cached image
tree raised ENOENT on ``.git/objects/a1``). The explicit ``git gc --prune=now`` runs in the foreground, and one that
fails (a gc another process holds the repository for exits 128) raises, never passes silently.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from collections.abc import Sequence
from pathlib import Path

# A fixed identity and date, so the rebuilt commit's sha depends only on the tree (useful for replay keys). PATH is kept
# so git itself can be found; nothing else from the host environment reaches git.
GIT_ENV = {
    "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
    "GIT_CONFIG_GLOBAL": "/dev/null",
    "GIT_CONFIG_SYSTEM": "/dev/null",
    "GIT_TERMINAL_PROMPT": "0",
    "GIT_AUTHOR_NAME": "receipts",
    "GIT_AUTHOR_EMAIL": "receipts@localhost",
    "GIT_COMMITTER_NAME": "receipts",
    "GIT_COMMITTER_EMAIL": "receipts@localhost",
    "GIT_AUTHOR_DATE": "2000-01-01T00:00:00+0000",
    "GIT_COMMITTER_DATE": "2000-01-01T00:00:00+0000",
}
# Every git command this lane runs on a tree carries these, so none starts a background gc or maintenance. As
# git-config(1) has them: maintenance.auto=false stops the commands that write a repository from running `git
# maintenance run --auto` after their own work; gc.auto=0 makes `git gc --auto` find nothing to do, whatever starts
# it; gc.autoDetach=false keeps a `git gc --auto` in the foreground; and maintenance.autoDetach=false does the same
# for the maintenance run (unset, it falls back to gc.autoDetach, and detaches when both are unset). On git 2.54 a
# commit with none of them starts `git maintenance run --auto --quiet --detach`, which repacks in the background (its
# trace2 events, 2026-10-28); on the eval VM's git 2.43.0 it left a detached `git gc` running (R24, bug D).
# Configuration given with -c wins over every config file, the repository's own included, and reaches every git
# process the command starts. An explicit `git gc`, with no --auto, runs in the foreground.
NO_BACKGROUND: tuple[str, ...] = (
    "-c",
    "gc.auto=0",
    "-c",
    "gc.autoDetach=false",
    "-c",
    "maintenance.auto=false",
    "-c",
    "maintenance.autoDetach=false",
)
_CONFIG_FLAGS = ("-c", "commit.gpgsign=false", "-c", "core.hooksPath=/dev/null", *NO_BACKGROUND)
_LEFTOVER_HEADS = ("ORIG_HEAD", "FETCH_HEAD", "MERGE_HEAD")


class HistoryStripError(RuntimeError):
    """The workspace still reaches more commits than intended after a strip."""


def _git(repo: Path, *args: str, check: bool = True) -> subprocess.CompletedProcess[str]:
    """Run one git command in ``repo`` with an argument list (never a shell string) and a sealed config."""
    proc = subprocess.run(
        ["git", "-C", str(repo), *_CONFIG_FLAGS, *args],
        capture_output=True,
        text=True,
        env=GIT_ENV,
        check=False,
    )
    if check and proc.returncode != 0:
        raise HistoryStripError(f"git {' '.join(args)} failed in {repo}: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc


def is_git_repo(repo: Path) -> bool:
    return _git(repo, "rev-parse", "--is-inside-work-tree", check=False).returncode == 0


def reachable_commit_count(repo: Path) -> int:
    """Commits reachable from every ref, reflog, remote and tag. A future commit left anywhere pushes this up."""
    out = _git(repo, "rev-list", "--count", "--all", "--reflog", "--remotes", "--tags", "HEAD").stdout.strip()
    return int(out or "0")


def stored_commit_count(repo: Path) -> int:
    """Commit objects in the object store, reachable or not. A dangling commit is still readable by its id."""
    out = _git(repo, "cat-file", "--batch-all-objects", "--batch-check=%(objecttype)").stdout
    return sum(1 for line in out.splitlines() if line.strip() == "commit")


def list_refs(repo: Path) -> list[str]:
    out = _git(repo, "for-each-ref", "--format=%(refname)", check=False).stdout
    return [line for line in out.splitlines() if line.strip()]


def _stash_count(repo: Path) -> int:
    out = _git(repo, "stash", "list", check=False).stdout
    return len([line for line in out.splitlines() if line.strip()])


def find_nested_git(root: Path) -> list[str]:
    """Every ``.git`` entry (directory, gitdir file or link) below ``root``, other than ``root/.git``, relative."""
    found: list[str] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        here = Path(dirpath)
        # The top-level .git is the strip's own business; any other one is a nested repository.
        if here != root and ".git" in (*dirnames, *filenames):
            found.append(str((here / ".git").relative_to(root)))
        dirnames[:] = [d for d in dirnames if d != ".git"]
    return sorted(found)


def remove_nested_git(root: Path) -> list[str]:
    """Delete every nested ``.git`` below ``root``, never following a link out of the tree. Returns what it removed."""
    removed = find_nested_git(root)
    for rel in removed:
        path = root / rel
        if path.is_symlink() or path.is_file():
            path.unlink()
        else:
            _rmtree(path)
    return removed


def _rmtree(path: Path) -> None:
    for child in sorted(path.rglob("*"), reverse=True):
        if child.is_symlink():
            continue  # never chmod through a link to something outside the tree
        child.chmod(0o700 if child.is_dir() else 0o600)
    shutil.rmtree(path)


def _gc(repo: Path) -> None:
    """``git gc --prune=now``, in the foreground, and checked: a gc that did not run (one another process holds the
    repository for exits 128, "gc is already running") raises HistoryStripError, so a tree whose objects something
    may still be deleting is never handed on."""
    _git(repo, "gc", "--prune=now", "--quiet")


def init_single_commit(repo: Path, *, message: str = "base") -> str:
    """Make ``repo`` (a tree with no .git, such as a repository copied out of an image without its history) a
    repository holding exactly one commit of its tree, with no git process still running in it when this returns.
    Returns the commit's sha."""
    if (repo / ".git").exists() or (repo / ".git").is_symlink():
        raise HistoryStripError(f"{repo} already has a .git; use rebuild_single_commit")
    remove_nested_git(repo)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A", "--force")
    _git(repo, "commit", "-q", "--allow-empty", "--no-verify", "-m", message)
    _gc(repo)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def rebuild_single_commit(repo: Path, *, message: str = "base") -> str:
    """Replace ``repo/.git`` with a brand-new repository holding exactly one commit of the working tree.

    Every tracked and untracked path in the working tree is captured, so a patch already applied to the tree is kept.
    Nothing from the old history survives, so the single commit is all ``git log --all`` can reach.
    """
    if not (repo / ".git").exists():
        raise HistoryStripError(f"{repo} has no .git to rebuild")
    _remove_git_dir(repo)
    _git(repo, "init", "-q", "-b", "main")
    _git(repo, "add", "-A")
    # --allow-empty: a repository with no files still rebuilds to one commit rather than failing.
    _git(repo, "commit", "-q", "--allow-empty", "-m", message)
    _git(repo, "reflog", "expire", "--expire=now", "--all", check=False)
    _gc(repo)
    return _git(repo, "rev-parse", "HEAD").stdout.strip()


def _remove_git_dir(repo: Path) -> None:
    git_path = repo / ".git"
    if git_path.is_symlink() or git_path.is_file():  # a worktree or submodule: .git is a gitdir pointer
        raise HistoryStripError(f"{repo}/.git is a pointer file (worktree or submodule); rebuild needs a full .git")
    _rmtree(git_path)


def strip_refs_in_place(repo: Path) -> None:
    """Keep HEAD's commit sha but remove every other way to reach a commit: branches, tags, remotes, notes,
    ``refs/original``, packed-refs, ORIG_HEAD, FETCH_HEAD, MERGE_HEAD, the stash and all reflogs, then prune."""
    head = _git(repo, "rev-parse", "HEAD").stdout.strip()
    _git(repo, "checkout", "-q", "--detach", head, check=False)
    for ref in list_refs(repo):
        _git(repo, "update-ref", "-d", ref, check=False)
    _git(repo, "stash", "clear", check=False)
    _git(repo, "remote", "remove", "origin", check=False)
    for name in (*_LEFTOVER_HEADS, "packed-refs"):
        (repo / ".git" / name).unlink(missing_ok=True)
    _git(repo, "reflog", "expire", "--expire=now", "--all", check=False)
    _gc(repo)


def assert_history_stripped(repo: Path, *, expected_commits: int = 1) -> None:
    """Raise unless the workspace holds exactly ``expected_commits`` commits, reachable and stored, with no stash, no
    leftover HEAD files and no nested repository.

    This is the gate the spec requires: it fails loudly, and a strip that leaves a ref, tag or reflog entry behind is
    caught because that commit is still reachable from ``--all --reflog``; a commit no ref names is caught because it
    is still stored.
    """
    if not is_git_repo(repo):
        raise HistoryStripError(f"{repo} is not a git work tree")
    stashes = _stash_count(repo)
    if stashes:
        raise HistoryStripError(f"history not stripped: {stashes} stash entr{'y' if stashes == 1 else 'ies'} remain")
    leftover = [n for n in _LEFTOVER_HEADS if (repo / ".git" / n).exists()]
    if leftover:
        raise HistoryStripError(f"history not stripped: leftover {', '.join(leftover)}")
    count = reachable_commit_count(repo)
    if count != expected_commits:
        refs = list_refs(repo)
        raise HistoryStripError(
            f"history not stripped: {count} commits reachable (--all --reflog), expected {expected_commits}; "
            f"refs still present: {refs or 'none'}"
        )
    stored = stored_commit_count(repo)
    if stored != expected_commits:
        raise HistoryStripError(
            f"history not stripped: {stored} commit objects stored, expected {expected_commits} (a dangling commit "
            "is still readable by id)"
        )
    nested = find_nested_git(repo)
    if nested:
        raise HistoryStripError(f"history not stripped: nested repositories remain at {', '.join(nested)}")


def strip_git_history(repo: Path, *, message: str = "base", expected_commits: int = 1) -> str:
    """Drop nested repositories, rebuild the checkout as a single commit and assert the result. Returns the new sha."""
    remove_nested_git(repo)
    sha = rebuild_single_commit(repo, message=message)
    assert_history_stripped(repo, expected_commits=expected_commits)
    return sha


def strip_workspace(root: Path) -> str | None:
    """Strip whatever history a checkout carries before it enters the sandbox. A checkout that is a repository is
    rebuilt to one commit (returns its sha); one that is not has any nested repository removed (returns None). Either
    way no ``.git`` with history is left below ``root``, and that is asserted."""
    if (root / ".git").exists() or (root / ".git").is_symlink():
        return strip_git_history(root)
    remove_nested_git(root)
    nested = find_nested_git(root)
    if nested:
        raise HistoryStripError(f"history not stripped: nested repositories remain at {', '.join(nested)}")
    return None


def git_changed_files(repo: Path, base: str, head: str) -> Sequence[str]:
    """Repo-relative paths changed between two commits, for the classifier's changed-file list. Here for callers that
    still have both commits; the eval usually takes the changed-file list straight from the patch instead."""
    out = _git(repo, "diff", "--name-only", f"{base}..{head}").stdout
    return [line for line in out.splitlines() if line.strip()]
