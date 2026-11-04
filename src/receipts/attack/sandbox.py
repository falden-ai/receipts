"""The adversarial sandbox (RT2): run untrusted agent code and attacker repros under hard isolation.

What a run gets. Every run-phase flag comes from ``contracts.SandboxLimits`` (``RUN_PHASE_LIMITS`` by default) through
its own ``run_flags``, and the build phase's limits through ``limit_flags``, so this sandbox and the eval VM's runsc
check render one list and keep no copy of it: no network, every Linux capability dropped, no-new-privileges, a pid
cap, a memory cap with swap off, a cpu cap, a /dev/shm cap, a private IPC namespace, no POSIX message queues, uid
65534, a read-only root, and RAM-backed workspace and /tmp mounts whose sizes are the disk quota. The run user also
closes the runtime's own /dev tmpfs, which Docker leaves writable to root even under ``--read-only``. Each run's HOME
is a directory inside the /tmp mount, so nothing reaches for nobody's /nonexistent.

The workspace is the image's repository path for a Tier A target (/testbed in the SWE-bench images) and /workspace
otherwise. For a Tier A target with no checkout callable, the base tree is the image's own: ``docker cp`` copies the
repository directory out of a created, never started container, the copy drops every ``.git`` and is rebuilt as one
commit, and that tree, with the side's patch applied, is what the workspace mount holds. The mount covers the image's
directory, so the image's baked history is unreachable and an editable install resolves to the patched tree, compiled
extensions included. An image that holds its own copy of the repository's code instead resolves nothing to the tree:
SWE-bench builds Django 1.4 to 2.2 with ``python setup.py install`` and psf/requests with ``pip install .``, and its
eval script installs again after applying the patch, so its labels describe the patched tree. The run phase cannot
write site-packages, so when the image's own interpreter imports one of the tree's top-level packages from a copy of
the base tree's code (``installed_copies``, asked once per image), every run in that workspace gets the workspace root
first on PYTHONPATH, ahead of the copy. The image's interpreter environment (a conda ``testbed`` environment in the
SWE-bench images) goes first on PATH, since a plain exec does not activate it.

A side's patch applies the way the SWE-bench harness applies it (REG-569; swebench 4.1.0, harness/run_evaluation.py
GIT_APPLY_CMDS), by exactly its chain and in its order: ``git apply --verbose``, then ``git apply --verbose --reject``,
both on the host, where git writes only inside the tree; then ``patch --batch --fuzz=5 -p1 -i``, run in a run-phase
container of the target's own image, never on the host. Each command runs on the tree the previous one left, as the
harness's do: a ``--reject`` that applied some hunks leaves them and its ``.rej`` files, and GNU patch runs on that
tree (and, as ``--batch`` makes it, reverses a hunk that looks already applied). The first exit 0 wins, and the
workspace records which command it was (``Workspace.patch_method``). A patch no command applies is refused (an
attacker's NO_ENV), and the tree it left is thrown away.

No secret and no host environment variable crosses in. A container's environment must equal its image's (the docker
CLI can add proxy settings from its own config file, and that is refused); an image whose environment holds a proxy
setting or anything shaped like a credential is refused; a run's own variables get the same check. The model calls
happen in the orchestrator, outside the container.

Output is read incrementally into bounded buffers: the last ``truncate_bytes`` of each stream for the run's result
(what a model may read), and for a runner that reads its cases from its output, at most ``parse_bytes`` from the
start. What does not fit is counted and dropped as it arrives, so a flood never accumulates in this process.

``run_test`` runs tests with the target's suite runner and returns the classified result (``receipts.attack.classify``).
Each run writes its report to its own path under /tmp, and the output and the report are read before a fresh
container is removed. ``run_cases`` returns the run's cases as well, for the regression attacker.

Every image is digest-pinned (``name@sha256:<64 hex>``, or a local ``sha256:`` image id the build phase committed)
before a runtime is chosen; ``Target.image`` wins when the target carries one. gVisor ``runsc`` whenever the host has
it; otherwise ``runc`` only for a Tier A target, never for a live repository (Tier B or C) or the hosted App. Every
refusal raises ``RuntimeRefused``. gVisor is not installed on this Mac (it is on the x86 eval VM), so local container
tests run under runc on digest-pinned Tier A targets, which is what the rules allow.

The build phase runs setup commands in a throwaway container and commits it. Its default, and the only egress this
lane enforces, is none. ``EgressPolicy.NETWORK`` joins a docker network the caller names (the eval VM's package-index
allowlist, which this lane does not build or verify); nothing goes into the environment for it, and a committed image
whose environment differs from its base image's is refused and removed.
"""

from __future__ import annotations

import hashlib
import io
import json
import math
import posixpath
import re
import shutil
import subprocess
import tarfile
import tempfile
import threading
import time
import uuid
import weakref
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import ExitStack, contextmanager, suppress
from dataclasses import dataclass, field, replace
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import IO, Protocol, runtime_checkable

from pydantic import TypeAdapter, ValidationError

from receipts.attack import git_history
from receipts.attack.classify import MAX_REPORT_BYTES, SuiteRun, classify_suite, with_expect
from receipts.contracts import (
    RUN_PHASE_LIMITS,
    FailureType,
    Outcome,
    PatchMethod,
    RunnerKind,
    RunResult,
    Sandbox,
    SandboxLimits,
    Side,
    SuiteRunner,
    Target,
    Tier,
    Workspace,
)

ImageResolver = Callable[[Target], str]
# A disposable copy of a side's base tree (the sandbox applies the side's patch, the third argument, to it), or None
# for an empty workspace.
Checkout = Callable[[Target, Side, str | None], Path | None]

_TIMEOUT_EXIT = 124  # coreutils `timeout` when its TERM ended the command; also the host-side timeout here
_KILLED_EXIT = 137  # 128 + SIGKILL: `timeout -k` killed a command that ignored TERM (or the OOM killer did)
_KILL_AFTER_S = 5
_BUILD_DIR = "/build"
_TMP = "/tmp"
RUN_HOME = "/tmp/home"  # each run's HOME: inside the /tmp mount, so it exists and is writable
_PATCH_FILE = "receipts-patch.diff"
TIER_A_ROOT = "/testbed"  # the SWE-bench images' repository path
DEFAULT_ROOT = "/workspace"
# Interpreter environments a plain exec does not activate; the first one an image has goes first on PATH.
_ENV_BINS = ("/opt/miniconda3/envs/testbed/bin",)
_OPEN_NETWORKS = frozenset({"bridge", "host", "default", "none"})
_CONTROL_BYTES = 4 * 1024 * 1024  # docker's own answers (inspect, version) are small
_TREE_BYTES = 6 * 1024**3  # a repository copied out of an image, .git included, before .git is dropped
_SOURCE_BYTES = 4 * 1024 * 1024  # a test module read to see which tests it defines

# A repository reference that docker parses as an image, never as a flag: it starts with a lowercase letter or digit.
_REPO = re.compile(r"[a-z0-9][a-z0-9._\-/]*(?::[0-9]{1,5}(?:/[a-z0-9._\-]+)+)?(?::[A-Za-z0-9_][A-Za-z0-9_.\-]{0,127})?")
_HEX64 = re.compile(r"[0-9a-f]{64}")

# Environment entries that never go into a container: proxy settings, and anything shaped like a credential.
_PROXY_NAMES = frozenset({"http_proxy", "https_proxy", "all_proxy", "ftp_proxy", "socks_proxy", "no_proxy"})
_CREDENTIAL_NAME = re.compile(
    r"(?:^|_)(?:API_?KEY|ACCESS_?KEY|SECRET|TOKEN|PASSWORD|PASSWD|CREDENTIALS?|PRIVATE_?KEY|AUTH)(?:_|$)",
    re.IGNORECASE,
)
_HOST_SECRET_PREFIXES = ("FALDEN_", "AWS_", "ANTHROPIC_", "OPENAI_")
_CREDENTIAL_VALUE = re.compile(
    r"[a-z][a-z0-9+.\-]*://[^/\s:@]+:[^/\s@]+@"  # a URL carrying user:password
    r"|sk-[A-Za-z0-9_\-]{16,}|gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|AKIA[0-9A-Z]{16}"
    r"|xox[abprs]-[A-Za-z0-9\-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----"
)


class SandboxError(RuntimeError):
    """A sandbox operation failed (docker returned an error, a build step failed, a patch did not apply)."""


class RuntimeRefused(SandboxError):
    """A runtime or an image was requested that the rules forbid for this target."""


class BuildError(SandboxError):
    """A build-phase setup command failed."""


# ---------------------------------------------------------------- bounded output


@dataclass(frozen=True)
class Keep:
    """How much of a stream to keep: up to ``head`` bytes from its start and the last ``tail`` bytes."""

    head: int = 0
    tail: int = 0


_SMALL = Keep(head=_CONTROL_BYTES)


class Captured:
    """One stream, read incrementally. Only the first ``keep.head`` and the last ``keep.tail`` bytes are held; the
    rest is counted and dropped as it arrives."""

    def __init__(self, keep: Keep) -> None:
        self.keep = keep
        self._head = bytearray()
        self._tail = bytearray()
        self.total = 0

    @classmethod
    def of(cls, data: bytes, keep: Keep) -> Captured:
        captured = cls(keep)
        captured.feed(data)
        return captured

    def feed(self, chunk: bytes) -> None:
        self.total += len(chunk)
        room = self.keep.head - len(self._head)
        if room > 0:
            self._head += chunk[:room]
        if self.keep.tail > 0:
            self._tail += chunk[-self.keep.tail :]
            if len(self._tail) > self.keep.tail:
                del self._tail[: len(self._tail) - self.keep.tail]

    @property
    def dropped(self) -> int:
        """Bytes neither buffer holds."""
        return max(0, self.total - max(len(self._head), len(self._tail)))

    def whole(self) -> bytes | None:
        """Everything the stream carried, when one buffer holds all of it; otherwise None."""
        if self.total <= len(self._head):
            return bytes(self._head)
        if self.total <= len(self._tail):
            return bytes(self._tail)
        return None

    def text(self) -> str | None:
        data = self.whole()
        return None if data is None else data.decode("utf-8", "replace")

    def tail_text(self) -> str:
        """The stream's tail for a run's result: valid UTF-8, marked when anything before it was dropped. A stream
        kept by its head only (docker's own answers) gives its head."""
        if self.keep.tail == 0:
            kept = bytes(self._head)
            marker = "" if self.total <= len(kept) else f"\n[... {self.total - len(kept)} bytes truncated ...]"
            return kept.decode("utf-8", "replace") + marker
        kept = bytes(self._tail)
        if self.total <= len(kept):
            return kept.decode("utf-8", "replace")
        return f"[... {self.total - len(kept)} bytes truncated ...]\n" + kept.decode("utf-8", "replace")


def truncate_tail(data: str | bytes, limit_bytes: int) -> str:
    """Keep only the last ``limit_bytes`` of some output, marked when anything was dropped, as valid UTF-8 text."""
    raw = data.encode("utf-8", "replace") if isinstance(data, str) else data
    return Captured.of(raw, Keep(tail=limit_bytes)).tail_text()


class Runner(Protocol):
    """Runs one command, given as an argument list, and returns (exit code, stdout, stderr), each stream bounded by
    ``keep``. With ``consume``, stdout is handed to it as a pipe instead of being captured."""

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        stdin: bytes | Path | None = None,
        keep: Keep = _SMALL,
        consume: Callable[[IO[bytes]], None] | None = None,
    ) -> tuple[int, Captured, Captured]: ...


def _pump(stream: IO[bytes], sink: Captured) -> None:
    for chunk in iter(lambda: stream.read(64 * 1024), b""):
        sink.feed(chunk)


def _run(
    argv: Sequence[str],
    *,
    timeout: float,
    stdin: bytes | Path | None = None,
    keep: Keep = _SMALL,
    consume: Callable[[IO[bytes]], None] | None = None,
) -> tuple[int, Captured, Captured]:
    """The real runner: subprocess with reader threads that keep only what ``keep`` allows of each stream."""
    out, err = Captured(keep), Captured(keep)
    failures: list[BaseException] = []
    with ExitStack() as stack:
        source = stack.enter_context(stdin.open("rb")) if isinstance(stdin, Path) else None
        proc = subprocess.Popen(
            list(argv),
            stdin=source if source is not None else (subprocess.PIPE if stdin is not None else subprocess.DEVNULL),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        assert proc.stdout is not None and proc.stderr is not None
        stdout = proc.stdout

        def read_stdout() -> None:
            try:
                if consume is not None:
                    consume(stdout)
                else:
                    _pump(stdout, out)
            except BaseException as exc:  # handed back below; the process must not block on a full pipe
                failures.append(exc)
                proc.kill()

        threads = [threading.Thread(target=read_stdout), threading.Thread(target=_pump, args=(proc.stderr, err))]
        if isinstance(stdin, bytes):
            assert proc.stdin is not None
            pipe = proc.stdin

            def feed() -> None:
                try:
                    pipe.write(stdin)
                except BrokenPipeError:
                    pass
                finally:
                    pipe.close()

            threads.append(threading.Thread(target=feed))
        for thread in threads:
            thread.daemon = True
            thread.start()
        try:
            code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()
            code = _TIMEOUT_EXIT
            err.feed(b"\n[host timeout]")
        for thread in threads:
            thread.join(timeout=10)
    if failures:
        raise failures[0]
    return code, out, err


# ---------------------------------------------------------------- policy helpers


def image_is_digest_pinned(image: str) -> bool:
    """True for a reference pinned to an exact image: ``name@sha256:<64 hex>`` (the name may carry a registry and a
    tag), or a bare ``sha256:<64 hex>`` image id (what ``docker commit`` returns). Nothing else pins an image, and a
    name that docker could read as a flag (``-v=...``) is not a name."""
    if image.startswith("sha256:"):
        return _HEX64.fullmatch(image[len("sha256:") :]) is not None
    name, sep, digest = image.partition("@sha256:")
    return bool(sep) and _REPO.fullmatch(name) is not None and _HEX64.fullmatch(digest) is not None


def require_pinned(image: str, what: str = "image") -> str:
    if not image_is_digest_pinned(image):
        raise RuntimeRefused(
            f"{what} {image!r} is not digest-pinned; the sandbox needs name@sha256:<64 hex> or a local sha256: image id"
        )
    return image


def choose_runtime(target: Target, image: str, runsc_available: bool) -> str:
    """Decide the container runtime for a target, or refuse. The image must be digest-pinned for every runtime. gVisor
    when present; otherwise runc only for a Tier A target, never for a live repository or the App."""
    require_pinned(image)
    if runsc_available:
        return "runsc"
    if target.tier is Tier.LIVE:
        raise RuntimeRefused("runc is refused for the hosted App (live PRs); gVisor runsc is required there")
    if target.tier is not Tier.A:
        raise RuntimeRefused(
            f"runc is refused for a live repository (Tier {target.tier.value}); allowed only for digest-pinned Tier A"
        )
    return "runc"


def refused_env_names(env: Mapping[str, str]) -> list[str]:
    """Names (never values) of the entries that must not enter a container: a proxy setting, a name that looks like a
    credential or a host secret, or a value shaped like one (a URL with a password, a token, a private key)."""
    bad: list[str] = []
    for name, value in env.items():
        if (
            name.lower() in _PROXY_NAMES
            or _CREDENTIAL_NAME.search(name) is not None
            or name.upper().startswith(_HOST_SECRET_PREFIXES)
            or _CREDENTIAL_VALUE.search(value) is not None
        ):
            bad.append(name)
    return sorted(bad)


def check_environment(env: Mapping[str, str], where: str) -> None:
    bad = refused_env_names(env)
    if bad:
        raise SandboxError(f"{where} carries proxy or credential variables, refused: {bad}")


def _env_map(entries: Sequence[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for entry in entries:
        name, _, value = entry.partition("=")
        out[name] = value
    return out


def workspace_root(target: Target) -> str:
    """Where a target's repository lives in the sandbox: the image's repository path for Tier A, else /workspace."""
    return TIER_A_ROOT if target.tier is Tier.A else DEFAULT_ROOT


class EgressPolicy(StrEnum):
    """What the build phase may reach.

    - ``NONE`` (the default): ``--network=none``. No egress at all, enforced on every host, this Mac included.
    - ``NETWORK``: the builder joins a docker network the caller names, which on the eval VM is the package-index
      allowlist (not built in this lane). Nothing is put in the environment for it. This lane cannot verify what that
      network allows; it refuses only the names that are plainly open (bridge, host, default) or not a network.
    """

    NONE = "none"
    NETWORK = "network"


# ---------------------------------------------------------------- trees on the host


def _as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
    # The host's uid and user name stay on the host; extraction as any user needs no chown.
    info.uid = info.gid = 0
    info.uname = info.gname = "root"
    return info


def _write_tar_of_dir(root: Path, dest: Path, *, skip_git: bool = False) -> None:
    with tarfile.open(dest, mode="w") as tar:
        for path in sorted(root.rglob("*")):
            rel = path.relative_to(root)
            if skip_git and rel.parts[0] == ".git":
                continue
            tar.add(path, arcname=str(rel), recursive=False, filter=_as_root)


def _tar_of_dir(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path in sorted(root.rglob("*")):
            tar.add(path, arcname=str(path.relative_to(root)), recursive=False, filter=_as_root)
    return buf.getvalue()


def _tar_of_files(files: Mapping[str, str | bytes]) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in files.items():
            pure = PurePosixPath(name)
            if pure.is_absolute() or ".." in pure.parts or not pure.parts or pure.parts[0] == ".git":
                raise SandboxError(f"run file {name!r} must be a relative path inside the workspace")
            data = content if isinstance(content, bytes) else content.encode("utf-8")
            info = _as_root(tarfile.TarInfo(str(pure)))
            info.size = len(data)
            info.mode = 0o644
            tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


def extract_tree(stream: IO[bytes], dest: Path, *, max_bytes: int = _TREE_BYTES) -> int:
    """Extract ``docker cp``'s tar of a directory into ``dest``, dropping the top directory and every ``.git``, and
    refusing anything the stdlib's data filter refuses (absolute names, ``..``, links out of the tree, devices).
    Returns how many members were skipped."""
    skipped = 0
    seen = 0
    with tarfile.open(fileobj=stream, mode="r|") as tar:
        for member in tar:
            seen += member.size
            if seen > max_bytes:
                raise SandboxError(f"the image's repository is larger than {max_bytes} bytes")
            parts = PurePosixPath(member.name).parts[1:]
            if not parts or ".git" in parts:
                continue
            member.name = str(PurePosixPath(*parts))
            if member.islnk():
                link = PurePosixPath(member.linkname).parts[1:]
                if not link:
                    skipped += 1
                    continue
                member.linkname = str(PurePosixPath(*link))
            try:
                tar.extract(member, dest, filter="data")
            except (tarfile.FilterError, OSError):
                skipped += 1
    return skipped


# The SWE-bench harness's chain (swebench 4.1.0, harness/run_evaluation.py GIT_APPLY_CMDS), command for command and in
# its order (REG-569). Each runs on the tree the previous one left, and the first exit 0 wins. The two git commands run
# on the host; GNU patch, the last, runs in a run-phase container of the target's own image (``Fallback``).
HARNESS_CHAIN: tuple[tuple[PatchMethod, tuple[str, ...]], ...] = (
    (PatchMethod.GIT_APPLY, ("git", "apply", "--verbose")),
    (PatchMethod.GIT_APPLY_REJECT, ("git", "apply", "--verbose", "--reject")),
    (PatchMethod.GNU_PATCH_FUZZ, ("patch", "--batch", "--fuzz=5", "-p1", "-i")),
)
GNU_PATCH = HARNESS_CHAIN[2][1]
# GNU patch on a tree, given the diff's exact bytes; it raises SandboxError when the patch did not apply.
Fallback = Callable[[Path, bytes], None]


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[str]:
    # GIT_CEILING_DIRECTORIES keeps git from finding a repository above a tree that has no .git of its own, and no git
    # command here leaves a gc or maintenance running after it returns (git_history.NO_BACKGROUND).
    env = {**git_history.GIT_ENV, "GIT_CEILING_DIRECTORIES": str(repo.parent)}
    return subprocess.run(
        ["git", "-C", str(repo), "-c", "core.hooksPath=/dev/null", *git_history.NO_BACKGROUND, *args],
        capture_output=True,
        text=True,
        env=env,
        check=False,
    )


def _said(text: str, lines: int = 3) -> str:
    """The last few lines a patch command printed, its own error lines first when it marks them (git's 'error:')."""
    every = [line.strip() for line in text.splitlines() if line.strip()]
    errors = [line for line in every if line.startswith("error:")]
    return " | ".join((errors or every)[-lines:])[-400:] or "no output"


def apply_patch(repo: Path, patch: str, *, fallback: Fallback | None = None) -> PatchMethod:
    """Apply a unified diff to a tree exactly the way the SWE-bench harness does (REG-569): ``git apply --verbose``,
    then ``git apply --verbose --reject``, then ``fallback`` (GNU patch, ``patch --batch --fuzz=5 -p1 -i``, run in the
    sandbox). Each command runs on the tree the one before it left, and the first that exits 0 wins; its method is
    returned. Nothing is reset between them: a plain ``git apply`` writes all of a patch or none of it, but a
    ``--reject`` that applied some hunks leaves them and its ``.rej`` files in the tree, and GNU patch then runs on that
    tree, as the harness's own does. The diff goes to every command byte for byte, as the harness writes it.

    A patch no command applies is refused with SandboxError naming what each one said; the tree is then in whatever
    state the last command left it, so the caller throws it away."""
    data = patch.encode("utf-8")
    with tempfile.NamedTemporaryFile("wb", suffix=".diff", delete=False) as fh:
        fh.write(data)
        diff = fh.name
    try:
        said: list[str] = []
        for method, argv in HARNESS_CHAIN[:2]:
            proc = _git(repo, *argv[1:], diff)
            if proc.returncode == 0:
                return method
            output = (proc.stderr or proc.stdout).replace(diff, "the patch")  # the host's temp path stays here
            said.append(f"{' '.join(argv)} exited {proc.returncode}: {_said(output)}")
        if fallback is not None:
            try:
                fallback(repo, data)
            except SandboxError as exc:
                said.append(str(exc))
            else:
                return PatchMethod.GNU_PATCH_FUZZ
        raise SandboxError(f"the patch does not apply: {'; '.join(said)}")
    finally:
        Path(diff).unlink(missing_ok=True)


# ---------------------------------------------------------------- where the image imports the repository from

# Where the image's own interpreter imports each of the tree's top-level packages from. It runs in the prepared
# container as the run phase does (uid 65534, read-only root, no network), from / and with no PYTHONPATH, so the
# workspace is not on its path, and prints each directory a package resolves to with the sha256 of every module
# directly in it. Python 3.5 runs it (the Django 1.11 and 2.2 images' interpreter): no f-strings.
_IMPORTS_PROBE = """\
import hashlib, importlib.util, json, os, sys
out = {}
for name in sys.argv[1:]:
    try:
        spec = importlib.util.find_spec(name)
    except Exception as e:
        sys.stderr.write('find_spec(%r) raised %s\\n' % (name, type(e).__name__))
        sys.exit(3)
    places = []
    for where in list(getattr(spec, 'submodule_search_locations', None) or []):
        files = {}
        if os.path.isdir(where):
            for entry in sorted(os.listdir(where)):
                path = os.path.join(where, entry)
                if entry.endswith('.py') and os.path.isfile(path) and not os.path.islink(path):
                    with open(path, 'rb') as fh:
                        files[entry] = hashlib.sha256(fh.read()).hexdigest()
        places.append([os.path.realpath(where), files])
    out[name] = places
print('RECEIPTS-IMPORTS ' + json.dumps(out, sort_keys=True))
"""
_IMPORTS_LINE = "RECEIPTS-IMPORTS "
_IMPORTS_TIMEOUT_S = 120.0
# Each package's places: [directory, {module: sha256}].
_IMPORTS: TypeAdapter[dict[str, list[tuple[str, dict[str, str]]]]] = TypeAdapter(
    dict[str, list[tuple[str, dict[str, str]]]]
)


def root_packages(tree: Path) -> dict[str, dict[str, str]]:
    """The tree's top-level packages, each a directory at its root with an ``__init__.py``, with the sha256 of every
    module directly in it. Symlinks are left out, as the probe leaves them out."""
    out: dict[str, dict[str, str]] = {}
    for child in sorted(tree.iterdir()):
        init = child / "__init__.py"
        if not child.name.isidentifier() or child.is_symlink() or not child.is_dir():
            continue
        if init.is_symlink() or not init.is_file():
            continue
        out[child.name] = {
            f.name: hashlib.sha256(f.read_bytes()).hexdigest()
            for f in sorted(child.iterdir())
            if f.name.endswith(".py") and not f.is_symlink() and f.is_file()
        }
    return out


def installed_copies(probe_output: str, packages: Mapping[str, Mapping[str, str]], root: str) -> list[str]:
    """The packages the image holds its own copy of, from the probe's output: a package that resolves to a directory
    outside the workspace whose modules are, byte for byte, the base tree's. SWE-bench installs Django 1.4 to 2.2 with
    "python setup.py install" and psf/requests with "python -m pip install ." (swebench 4.1.0,
    harness/constants/python.py, lines 106, 122 and 184), and its eval script runs that install again after the patch
    is applied (harness/test_spec/python.py, make_eval_script_list_py, lines 443 to 444), so the harness tests the
    patched tree. A package the image installs editably (the workspace itself), does not hold, or holds as different
    code is no copy. Output that is not the probe's answer for exactly these packages is refused."""
    lines = [line for line in probe_output.splitlines() if line.startswith(_IMPORTS_LINE)]
    if not lines:
        raise SandboxError("the import probe printed no answer")
    try:
        found = _IMPORTS.validate_json(lines[-1][len(_IMPORTS_LINE) :])
    except ValidationError as exc:
        raise SandboxError(f"the import probe's answer is unreadable: {exc.error_count()} errors") from exc
    if set(found) != set(packages):
        raise SandboxError(f"the import probe answered for {sorted(found)}, not {sorted(packages)}")
    top = PurePosixPath(root)
    return sorted(
        name
        for name, places in found.items()
        for where, files in places
        if not PurePosixPath(where).is_relative_to(top) and files == dict(packages[name])
    )


# ---------------------------------------------------------------- the sandbox


@dataclass
class _ContainerState:
    side: Side
    image: str
    runtime: str
    digest: str
    root: str
    target: Target
    host_dir: Path | None
    tar_path: Path | None
    path_env: str
    has_timeout: bool = False
    patch_method: PatchMethod | None = None
    extra_containers: list[str] = field(default_factory=list)
    # The workspace root when the image holds its own copy of the repository's code (``installed_copies``): every run
    # then gets it first on PYTHONPATH, so the run imports the patched tree, as the harness's reinstall makes it do.
    import_root: str | None = None


@runtime_checkable
class SuiteSandbox(Sandbox, Protocol):
    """A sandbox that also reports every case a run reported, and reads the tree it prepared. The regression attacker
    needs both; contracts.Sandbox has neither yet (see the lane's contract requests)."""

    def run_cases(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: SuiteRunner,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
    ) -> SuiteRun: ...

    def tree_files(self, workspace: Workspace) -> Sequence[str]: ...

    def read_text(self, workspace: Workspace, path: str, *, max_bytes: int = ...) -> str: ...


def docker_available(docker: str = "docker", *, timeout: float = 10.0, runner: Runner | None = None) -> bool:
    """True when a docker daemon answers. Docker-backed tests skip cleanly when this is False."""
    rc, _, _ = (runner or _run)([docker, "version", "--format", "{{.Server.Version}}"], timeout=timeout)
    return rc == 0


def runsc_available(docker: str = "docker", *, timeout: float = 10.0, runner: Runner | None = None) -> bool:
    """True when the daemon has a runtime registered under the name ``runsc``."""
    rc, out, _ = (runner or _run)([docker, "info", "--format", "{{json .Runtimes}}"], timeout=timeout)
    if rc != 0:
        return False
    try:
        runtimes = json.loads(out.text() or "null")
    except ValueError:
        return False
    return isinstance(runtimes, dict) and "runsc" in runtimes


class DockerSandbox(Sandbox):
    """A ``contracts.Sandbox`` (and a ``SuiteSandbox``) backed by the docker CLI through argument lists, never a shell
    string.

    ``run`` returns a coarse result: PASS on exit 0, TIMEOUT when the command was killed for time, ERROR otherwise.
    ``run_test`` runs tests with a suite runner and returns the contract-grade result. ``checkout``, when given, hands
    over a disposable copy of the base tree for a side (its history is stripped before it is copied in, and the
    sandbox applies the side's patch to it; a copy whose patch is refused is left as the chain left it, the caller's
    to throw away); without one, a target with an image gets the image's tree (also for Tier A image resolvers).
    ``runner`` and ``clock`` exist so the
    argument lists and the timeout labels can be tested without Docker.
    """

    def __init__(
        self,
        *,
        image_resolver: ImageResolver | None = None,
        checkout: Checkout | None = None,
        runsc: bool | None = None,
        limits: SandboxLimits | None = None,
        workspace_path: str | None = None,
        strip_history: bool = True,
        container_prefix: str = "receipts",
        wall_s: float = 3600.0,
        truncate_bytes: int = 64 * 1024,
        parse_bytes: int = 4 * 1024 * 1024,
        docker: str = "docker",
        command_timeout_slack_s: float = 10.0,
        work_dir: Path | None = None,
        runner: Runner | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._resolve = image_resolver
        self._checkout = checkout
        self._limits = limits or RUN_PHASE_LIMITS
        self._workspace_path = workspace_path
        self._strip_history = strip_history
        self._prefix = container_prefix
        self._wall_s = wall_s
        self._truncate = truncate_bytes
        self._parse = parse_bytes
        self._docker = docker
        self._slack = command_timeout_slack_s
        self._work = Path(tempfile.mkdtemp(prefix="receipts-sandbox-", dir=work_dir))
        # The trees and tars kept here go when the sandbox does (or at exit), whatever its caller forgets.
        self._finalizer = weakref.finalize(self, shutil.rmtree, self._work, ignore_errors=True)
        self._tree_lock = threading.Lock()
        self._runner: Runner = runner or _run
        self._clock: Callable[[], float] = clock or time.monotonic
        self._runsc = runsc_available(docker, runner=self._runner) if runsc is None else runsc
        self._state: dict[str, _ContainerState] = {}
        self._image_env: dict[str, list[str]] = {}
        self._trees: dict[tuple[str, str], Path] = {}
        # (image, root, the base tree's packages) -> the packages the image holds its own copy of
        self._copies: dict[tuple[str, str, str], tuple[str, ...]] = {}

    # -- the contract ------------------------------------------------------

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        """Make a side's workspace. ``patch`` is the diff for this side; when it is None, Side.PATCH applies the
        target's patch and every other side applies nothing. The workspace records how the diff applied
        (``Workspace.patch_method``), or None when there was none to apply."""
        image = self._image_for(target)
        runtime = choose_runtime(target, image, self._runsc)
        image_env = self._check_image_env(image)
        root = self._workspace_path or workspace_root(target)
        diff = patch if patch is not None else (target.patch if side is Side.PATCH else None)
        host_dir, method, packages = self._tree_for(target, side, diff or None, image, runtime, root)
        tar_path: Path | None = None
        if host_dir is not None:
            tar_path = self._work / f"{host_dir.name}.tar"
            _write_tar_of_dir(host_dir, tar_path)
        name = self._new_name(side.value)
        self._create_and_start(name, image, runtime, root)
        try:
            if tar_path is not None:
                self._inject_tar(name, tar_path, root)
            self._make_home(name)
            path_env = self._path_env(name, image_env)
            has_timeout = self._detect_timeout(name)
            copies = self._copies_of(name, image, root, path_env, packages)
        except BaseException:
            # A failure after the container started (injection, detection) must not leak the container.
            self._force_rm(name)
            raise
        digest = image.split("@", 1)[1] if "@" in image else image
        self._state[name] = _ContainerState(
            side=side,
            image=image,
            runtime=runtime,
            digest=digest,
            root=root,
            target=target,
            host_dir=host_dir,
            tar_path=tar_path,
            path_env=path_env,
            has_timeout=has_timeout,
            patch_method=method,
            import_root=root if copies else None,
        )
        return Workspace(id=name, side=side, root=root, image_digest=digest, patch_method=method)

    def run(
        self,
        workspace: Workspace,
        command: Sequence[str],
        *,
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        env: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
    ) -> RunResult:
        st = self._checked(workspace, command, timeout_s)
        merged = self._env(env, seed)
        with self._container(workspace, st, fresh) as (name, state):
            coarse, _, _ = self._exec(name, state, command, timeout_s, files, merged, seed, workspace.side, fresh, 0)
        return coarse

    def run_test(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: SuiteRunner,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
        expect: Sequence[str] | None = None,
    ) -> RunResult:
        run = self.run_cases(
            workspace,
            tests,
            runner=runner,
            changed_files=changed_files,
            timeout_s=timeout_s,
            files=files,
            seed=seed,
            fresh=fresh,
        )
        homes = self._homes(workspace, runner, tests, files, expect) if expect else None
        # The repository the workspace was prepared from: where swebench 4.1.0's log parser renames a case, the id it
        # records answers for that case (contracts.answers).
        return with_expect(run, expect, homes=homes, repo=self._known(workspace).target.repo)

    def _homes(
        self,
        workspace: Workspace,
        runner: SuiteRunner,
        tests: Sequence[str],
        files: Mapping[str, str] | None,
        expect: Sequence[str],
    ) -> dict[str, tuple[str, ...]]:
        """For a sympy run, which of the run's test files define each bare name in ``expect`` (SWE-bench names a sympy
        test by its function alone). That is what tells an expected case that never ran because its module did not
        import from one no file defines. Read from the tree the workspace was prepared from and the run's own files,
        which are what the run ran; a pytest node id and a Django label name their module themselves."""
        bare = {name for name in expect if "::" not in name}
        if runner.kind is not RunnerKind.SYMPY or not bare:
            return {}
        from receipts.attack.selection import case_ids  # selection reads the runners, which import this module

        st = self._known(workspace)
        homes: dict[str, list[str]] = {}
        for path in dict.fromkeys(test.partition("::")[0] for test in tests):
            for module, text in self._sources(st, path, files or {}):
                try:
                    defined = [i.rpartition("::")[2] for i in case_ids(RunnerKind.SYMPY, module, text)]
                except (SyntaxError, ValueError):
                    continue  # a module that does not parse defines nothing a run could have run
                for name in defined:
                    if name in bare and module not in homes.setdefault(name, []):
                        homes[name].append(module)
        return {name: tuple(paths) for name, paths in homes.items()}

    def _sources(self, st: _ContainerState, path: str, files: Mapping[str, str]) -> list[tuple[str, str]]:
        """The test modules a run's path argument stands for (the file, or the test files under a directory), each with
        its text as the run saw it: the run's own file when it wrote one, else the prepared tree's."""
        pure = PurePosixPath(path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            return []
        out: list[tuple[str, str]] = []
        wanted: list[str] = [path] if path.endswith(".py") else []
        with suppress(OSError):  # a tree that cannot be listed defines nothing here; the case stays unexplained
            if not wanted and st.host_dir is not None and (st.host_dir / pure).is_dir():
                wanted = sorted(
                    p.relative_to(st.host_dir).as_posix()
                    for p in (st.host_dir / pure).rglob("test_*.py")
                    if p.is_file() and ".git" not in p.relative_to(st.host_dir).parts
                )
        wanted += sorted(f for f in files if f.startswith(path.rstrip("/") + "/") and f.endswith(".py"))
        for module in dict.fromkeys(wanted):
            if module in files:
                out.append((module, files[module]))
                continue
            if st.host_dir is None:
                continue
            with suppress(OSError):  # never fail a run over a file it could not read; that file explains nothing
                full = (st.host_dir / module).resolve()
                inside = full.is_relative_to(st.host_dir.resolve()) and full.is_file()
                if inside and full.stat().st_size <= _SOURCE_BYTES:
                    out.append((module, full.read_text(encoding="utf-8", errors="replace")))
        return out

    def cleanup(self) -> None:
        """Close every workspace and remove what the sandbox kept on the host (its trees and tars)."""
        for name in list(self._state):
            st = self._state[name]
            self.close(
                Workspace(id=name, side=st.side, root=st.root, image_digest=st.digest, patch_method=st.patch_method)
            )
        self._finalizer()

    def close(self, workspace: Workspace) -> None:
        st = self._state.pop(workspace.id, None)
        if st is None:
            return
        for name in [*st.extra_containers, workspace.id]:
            self._force_rm(name)
        if st.tar_path is not None:
            st.tar_path.unlink(missing_ok=True)
        if st.host_dir is not None and st.host_dir.is_relative_to(self._work):
            shutil.rmtree(st.host_dir, ignore_errors=True)

    # -- per-case runs and the prepared tree (SuiteSandbox) -----------------

    def run_cases(
        self,
        workspace: Workspace,
        tests: Sequence[str],
        *,
        runner: SuiteRunner,
        changed_files: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None = None,
        seed: int | None = None,
        fresh: bool = False,
    ) -> SuiteRun:
        """Run ``tests`` with ``runner`` and classify the run from what it printed and the report it wrote, both read
        before a fresh container is removed. The report path is new for every run."""
        if not tests:
            raise SandboxError("a test run names at least one test")
        st = self._checked(workspace, list(tests), timeout_s)
        bound = _bind(runner, workspace.root)
        report_path = f"{_TMP}/receipts-report-{uuid.uuid4().hex}.xml"
        command = bound.command(st.target, tests, report_path=report_path)
        stream = getattr(bound, "stream", "both")
        head = 0 if stream == "none" else self._parse
        merged = self._env(None, seed)
        with self._container(workspace, st, fresh) as (name, state):
            coarse, out, err = self._exec(
                name, state, command, timeout_s, files, merged, seed, workspace.side, fresh, head
            )
            # The lane's Django and sympy runners read their output, not a report; any other runner may write one.
            reads_output = stream in ("stdout", "stderr")
            report, problem = (None, None) if reads_output else self._read_report(name, report_path)
        timed_out = coarse.outcome is Outcome.TIMEOUT
        output = _output_for(stream, out, err)
        return classify_suite(
            coarse,
            runner=bound,
            output=output,
            report=report,
            changed_files=changed_files,
            timed_out=timed_out,
            problem=problem,
        )

    def tree_files(self, workspace: Workspace) -> tuple[str, ...]:
        """Every file in the tree the workspace was prepared from, repo-relative, .git excluded."""
        st = self._known(workspace)
        if st.host_dir is None:
            return ()
        out = []
        for path in st.host_dir.rglob("*"):
            rel = path.relative_to(st.host_dir)
            if rel.parts and rel.parts[0] != ".git" and (path.is_file() or path.is_symlink()):
                out.append(rel.as_posix())
        return tuple(sorted(out))

    def read_text(self, workspace: Workspace, path: str, *, max_bytes: int = 4 * 1024 * 1024) -> str:
        """A file from the tree the workspace was prepared from (what was copied in, before any run changed it)."""
        st = self._known(workspace)
        pure = PurePosixPath(path)
        if st.host_dir is None or pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise SandboxError(f"{path!r} must be a relative path inside the workspace")
        full = (st.host_dir / pure).resolve()
        if not full.is_relative_to(st.host_dir.resolve()) or not full.is_file():
            raise SandboxError(f"{path!r} is not a file in the workspace")
        if full.stat().st_size > max_bytes:
            raise SandboxError(f"{path!r} is larger than {max_bytes} bytes; refusing to read it")
        return full.read_text(encoding="utf-8", errors="replace")

    # -- extras used by the attack lane and its tests --------------------

    def copy_out_text(self, workspace: Workspace, rel_path: str, *, max_bytes: int = 8 * 1024 * 1024) -> str:
        """Read a file a run produced in the prepared container. The content is untrusted: a file over ``max_bytes``
        is refused, and nothing read here goes to a model without ``truncate_tail``."""
        st = self._known(workspace)
        pure = PurePosixPath(rel_path)
        if pure.is_absolute() or ".." in pure.parts or not pure.parts:
            raise SandboxError(f"{rel_path!r} must be a relative path inside the workspace")
        text, problem = self._read_file(workspace.id, f"{st.root}/{pure}", max_bytes)
        if problem is not None or text is None:
            raise SandboxError(f"could not read {rel_path!r}: {problem or 'no such file'}")
        return text

    def build_phase(
        self,
        target: Target,
        setup_commands: Sequence[Sequence[str]],
        *,
        egress: EgressPolicy = EgressPolicy.NONE,
        base_image: str | None = None,
        egress_network: str | None = None,
        context: Path | None = None,
        disk_quota: str | None = None,
        timeout_s: float = 600.0,
    ) -> str:
        """Run setup commands in a throwaway container, then commit it to a new image the run phase can use with no
        network. Returns the committed image id (``sha256:...``).

        ``context`` is a host directory copied to /build before the setup commands run, so a build with no egress can
        still install from a local wheelhouse. ``disk_quota`` adds ``--storage-opt size=`` for a storage driver that
        supports it; without one the build's disk is not capped. The build phase runs as the image's own user, since
        installing writes to the image.
        """
        image = require_pinned(base_image, "base image") if base_image is not None else self._image_for(target)
        runtime = choose_runtime(target, image, self._runsc)
        self._check_image_env(image)
        net_flags = self._egress_flags(egress, egress_network)
        name = self._new_name("build")
        create = [
            self._docker,
            "create",
            "--name",
            name,
            f"--runtime={runtime}",
            "--pull=never",
            *net_flags,
            *self._limits.limit_flags(),
            *(["--storage-opt", f"size={disk_quota}"] if disk_quota is not None else []),
            "-w",
            _BUILD_DIR,
            image,
            "sleep",
            str(int(self._wall_s)),
        ]
        rc, _, err = self._call(create, timeout=60.0)
        if rc != 0:
            raise BuildError(f"could not create builder: {err.tail_text().strip()}")
        try:
            self._check_container_env(name, image)
            self._start(name)
            if context is not None:
                self._inject_bytes(name, _tar_of_dir(context), _BUILD_DIR)
            for cmd in setup_commands:
                keep = Keep(tail=self._truncate)
                rc, out, berr = self._call(
                    [self._docker, "exec", "-w", _BUILD_DIR, name, *cmd], timeout=timeout_s, keep=keep
                )
                if rc != 0:
                    raise BuildError(
                        f"build step {list(cmd)} failed (exit {rc}):\n{out.tail_text()}\n{berr.tail_text()}"
                    )
            rc, out, cerr = self._call([self._docker, "commit", name], timeout=120.0)
            if rc != 0:
                raise BuildError(f"commit failed: {cerr.tail_text().strip()}")
            # The image id is the last non-empty line; any deprecation notice prints above it.
            lines = [line.strip() for line in (out.text() or "").splitlines() if line.strip()]
            if not lines or not lines[-1].startswith("sha256:") or not image_is_digest_pinned(lines[-1]):
                raise BuildError(f"commit did not return an image id: {(out.text() or '').strip()!r}")
            built = lines[-1]
            try:
                self._check_committed_env(built, image)
            except BaseException:
                self._call([self._docker, "rmi", "-f", built], timeout=60.0)
                raise
            return built
        finally:
            self._force_rm(name)

    def _egress_flags(self, egress: EgressPolicy, egress_network: str | None) -> list[str]:
        if egress is EgressPolicy.NONE:
            if egress_network is not None:
                raise BuildError("EgressPolicy.NONE takes no egress_network")
            return ["--network=none"]
        if not egress_network:
            raise BuildError("EgressPolicy.NETWORK needs the name of the caller's allowlisting network")
        if egress_network in _OPEN_NETWORKS or ":" in egress_network or egress_network.startswith("-"):
            raise BuildError(f"{egress_network!r} is not an allowlisting network")
        return [f"--network={egress_network}"]

    # -- trees -----------------------------------------------------------

    def _tree_for(
        self, target: Target, side: Side, diff: str | None, image: str, runtime: str, root: str
    ) -> tuple[Path | None, PatchMethod | None, dict[str, dict[str, str]]]:
        """The side's tree on the host, with ``diff`` applied, how it applied, and the base tree's top-level packages
        (``root_packages``, read before the diff applies). The base tree is the checkout's (its history stripped) or,
        for a target with an image or a Tier A image resolver, the image's own; a side with a diff needs a tree."""
        if self._checkout is not None:
            host_dir = self._checkout(target, side, diff)
            if host_dir is not None and self._strip_history:
                git_history.strip_workspace(host_dir)
        elif target.tier is Tier.A or target.image is not None:
            base = self._image_tree(image, root)
            host_dir = Path(tempfile.mkdtemp(prefix=f"ws-{side.value}-", dir=self._work))
            shutil.copytree(base, host_dir, symlinks=True, dirs_exist_ok=True)
        else:
            host_dir = None
        packages = root_packages(host_dir) if host_dir is not None else {}
        method: PatchMethod | None = None
        if diff:
            if host_dir is None:
                raise SandboxError(f"the {side.value} side has a patch but no tree to apply it to")

            def gnu_patch(repo: Path, data: bytes) -> None:
                self._gnu_patch(image, runtime, root, repo, data)

            try:
                method = apply_patch(host_dir, diff, fallback=gnu_patch)
            except BaseException:
                # A refused patch leaves the tree as its last command left it; a tree this sandbox made goes with it.
                if host_dir.is_relative_to(self._work):
                    shutil.rmtree(host_dir, ignore_errors=True)
                raise
        stripped = self._checkout is None or self._strip_history
        if host_dir is not None and stripped and (host_dir / ".git").exists():
            git_history.assert_history_stripped(host_dir)
        return host_dir, method, packages

    def _gnu_patch(self, image: str, runtime: str, root: str, repo: Path, diff: bytes) -> None:
        """GNU patch, the SWE-bench harness's last command (``patch --batch --fuzz=5 -p1 -i``), run in a run-phase
        container of the target's own image: the same patch program the labels were graded with, and never on the
        host, since GNU patch follows symlinks and runs ed scripts. The container gets a copy of the tree as the
        previous command left it (with any hunks and ``.rej`` files a ``--reject`` left), as the harness's GNU patch
        does. Only on exit 0 does the tree come back out, whole, and replace ``repo`` (its own .git kept)."""
        name = self._new_name("gnupatch")
        tree = self._work / f"{name}.tar"
        dest = Path(tempfile.mkdtemp(prefix="patched-", dir=self._work))
        try:
            _write_tar_of_dir(repo, tree, skip_git=True)
            self._create_and_start(name, image, runtime, root)
            self._inject_tar(name, tree, root)
            self._inject_bytes(name, _tar_of_files({_PATCH_FILE: diff}), _TMP)
            keep = Keep(tail=4096)
            argv = [self._docker, "exec", "-w", root, name, *GNU_PATCH, f"{_TMP}/{_PATCH_FILE}"]
            rc, out, err = self._call(argv, timeout=120.0, keep=keep)
            if rc != 0:
                said = _said(out.tail_text() + "\n" + err.tail_text())
                raise SandboxError(f"{' '.join(GNU_PATCH)} exited {rc}: {said}")
            unreadable: list[str] = []

            def consume(pipe: IO[bytes]) -> None:
                try:
                    extract_tree(pipe, dest)
                except tarfile.TarError as exc:
                    unreadable.append(str(exc))
                    pipe.read()

            parent, top = posixpath.split(root.rstrip("/"))
            copy = [self._docker, "exec", name, "tar", "-c", "-f", "-", "-C", parent or "/", top]
            rc, _, err = self._call(copy, timeout=600.0, keep=_SMALL, consume=consume)
            if rc != 0 or unreadable:
                why = err.tail_text().strip() or "; ".join(unreadable)
                raise SandboxError(f"could not copy the patched tree out of {name}: {why}")
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        finally:
            self._force_rm(name)
            tree.unlink(missing_ok=True)
        if (repo / ".git").exists():
            (repo / ".git").rename(dest / ".git")
        shutil.rmtree(repo)
        dest.rename(repo)

    def _image_tree(self, image: str, root: str) -> Path:
        """The image's repository directory on the host, with no .git but a rebuilt single commit; cached, and made
        once per image even when attackers share the sandbox across threads."""
        with self._tree_lock:
            return self._image_tree_locked(image, root)

    def _image_tree_locked(self, image: str, root: str) -> Path:
        key = (image, root)
        if key in self._trees:
            return self._trees[key]
        source = self._new_name("tree")
        rc, _, err = self._call([self._docker, "create", "--pull=never", "--name", source, image, "true"], timeout=60.0)
        if rc != 0:
            raise SandboxError(f"could not create {source} to copy {root} out: {err.tail_text().strip()}")
        dest = Path(tempfile.mkdtemp(prefix="tree-", dir=self._work))
        skipped: list[int] = []
        unreadable: list[str] = []
        try:

            def consume(pipe: IO[bytes]) -> None:
                try:
                    skipped.append(extract_tree(pipe, dest))
                except tarfile.TarError as exc:
                    unreadable.append(str(exc))
                    pipe.read()  # drain, so docker cp exits on its own

            rc, _, err = self._call(
                [self._docker, "cp", f"{source}:{root}", "-"], timeout=900.0, keep=_SMALL, consume=consume
            )
            if rc != 0 or not skipped:
                why = err.tail_text().strip() or "; ".join(unreadable) or "no tree"
                raise SandboxError(f"could not copy {root} out of {image}: {why}")
        except BaseException:
            shutil.rmtree(dest, ignore_errors=True)
            raise
        finally:
            self._force_rm(source)
        git_history.init_single_commit(dest)
        git_history.assert_history_stripped(dest)
        self._trees[key] = dest
        return dest

    # -- docker helpers --------------------------------------------------

    def _call(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        stdin: bytes | Path | None = None,
        keep: Keep = _SMALL,
        consume: Callable[[IO[bytes]], None] | None = None,
    ) -> tuple[int, Captured, Captured]:
        return self._runner(argv, timeout=timeout, stdin=stdin, keep=keep, consume=consume)

    def _new_name(self, kind: str) -> str:
        return f"{self._prefix}-{kind}-{uuid.uuid4().hex[:10]}"

    def _known(self, workspace: Workspace) -> _ContainerState:
        if workspace.id not in self._state:
            raise SandboxError(f"unknown workspace {workspace.id!r}; call prepare first")
        return self._state[workspace.id]

    def _checked(self, workspace: Workspace, command: Sequence[str], timeout_s: float) -> _ContainerState:
        st = self._known(workspace)
        if not command:
            raise SandboxError("a run needs a command")
        if not timeout_s > 0:
            raise SandboxError("a run needs a positive timeout")
        return st

    def _env(self, env: Mapping[str, str] | None, seed: int | None) -> dict[str, str]:
        merged = {"HOME": RUN_HOME, **(env or {})}
        if seed is not None:
            merged.setdefault("PYTHONHASHSEED", str(seed))
        check_environment(merged, "the run's environment")
        return merged

    def _image_for(self, target: Target) -> str:
        if target.image is not None:
            return require_pinned(target.image, "target image")
        if self._resolve is None:
            raise SandboxError(f"target {target.id!r} names no image and the sandbox has no image resolver")
        return require_pinned(self._resolve(target), "resolved image")

    def _inspect_env(self, kind: str, ref: str) -> list[str]:
        rc, out, err = self._call(
            [self._docker, kind, "inspect", "--format", "{{json .Config.Env}}", ref], timeout=30.0
        )
        if rc != 0:
            raise SandboxError(f"could not inspect {kind} {ref}: {err.tail_text().strip()}")
        try:
            env = json.loads(out.text() or "null")
        except ValueError as exc:
            raise SandboxError(f"unreadable environment for {kind} {ref}") from exc
        if env is None:
            return []
        if not isinstance(env, list) or not all(isinstance(e, str) for e in env):
            raise SandboxError(f"unreadable environment for {kind} {ref}")
        return list(env)

    def _check_image_env(self, image: str) -> list[str]:
        """An image whose own environment carries a proxy setting or a credential is refused before it runs."""
        if image not in self._image_env:
            env = self._inspect_env("image", image)
            check_environment(_env_map(env), f"image {image}")
            self._image_env[image] = env
        return self._image_env[image]

    def _check_container_env(self, name: str, image: str) -> None:
        """A container's environment must be exactly its image's: nothing from the host or the docker CLI's config."""
        extra = set(self._inspect_env("container", name)) - set(self._check_image_env(image))
        if extra:
            names = sorted({entry.partition("=")[0] for entry in extra})
            raise SandboxError(f"container {name} carries environment its image does not, refused: {names}")

    def _check_committed_env(self, built: str, base: str) -> None:
        """The build phase adds no environment: the committed image's must equal its base image's, entry for entry."""
        committed = self._inspect_env("image", built)
        base_env = self._check_image_env(base)
        if committed != base_env:
            added = sorted({e.partition("=")[0] for e in set(committed) ^ set(base_env)})
            raise BuildError(f"the committed image's environment differs from its base image's: {added}")

    def _create_and_start(self, name: str, image: str, runtime: str, root: str) -> None:
        create = [
            self._docker,
            "create",
            "--name",
            name,
            "--init",
            f"--runtime={runtime}",
            "--pull=never",
            *self._limits.run_flags(workspace=root, tmp=_TMP),
            "-w",
            root,
            image,
            "sleep",
            str(int(self._wall_s)),
        ]
        rc, _, err = self._call(create, timeout=60.0)
        if rc != 0:
            raise SandboxError(f"could not create container {name}: {err.tail_text().strip()}")
        try:
            self._check_container_env(name, image)
        except BaseException:
            self._force_rm(name)
            raise
        self._start(name)

    def _start(self, name: str) -> None:
        rc, _, err = self._call([self._docker, "start", name], timeout=60.0)
        if rc != 0:
            self._force_rm(name)
            raise SandboxError(f"could not start container {name}: {err.tail_text().strip()}")

    def _inject_tar(self, name: str, tar_path: Path, dest: str) -> None:
        self._inject(name, tar_path, dest)

    def _inject_bytes(self, name: str, data: bytes, dest: str) -> None:
        self._inject(name, data, dest)

    def _inject(self, name: str, source: bytes | Path, where: str) -> None:
        """Extract a tar into ``where`` by running the extractor inside the container.

        ``docker cp`` is refused into a container whose root filesystem is read-only, even when the destination is a
        writable mount, so the stream goes to ``tar`` (or a stdlib fallback) over ``docker exec`` instead, which writes
        to the tmpfs. The tar is one this process built, with safe names and root ownership.
        """
        tar_cmd = [self._docker, "exec", "-i", "-w", where, name, "tar", "-x", "-f", "-", "-C", where]
        rc, _, err = self._call(tar_cmd, timeout=600.0, stdin=source)
        if rc == 0:
            return
        extract = f"import sys, tarfile; tarfile.open(fileobj=sys.stdin.buffer, mode='r|').extractall({where!r})"
        py_cmd = [self._docker, "exec", "-i", "-w", where, name, "python", "-I", "-S", "-c", extract]
        rc2, _, err2 = self._call(py_cmd, timeout=600.0, stdin=source)
        if rc2 != 0:
            tar_err, py_err = err.tail_text().strip(), err2.tail_text().strip()
            raise SandboxError(f"could not inject files into {name}: tar: {tar_err}; python: {py_err}")

    def _make_home(self, name: str) -> None:
        rc, _, err = self._call([self._docker, "exec", name, "mkdir", "-p", RUN_HOME], timeout=30.0)
        if rc != 0:
            raise SandboxError(f"could not make the run's HOME in {name}: {err.tail_text().strip()}")

    def _path_env(self, name: str, image_env: Sequence[str]) -> str:
        """The image's PATH, with its interpreter environment's bin directory first when it has one."""
        path = _env_map(image_env).get("PATH", "/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin")
        for bin_dir in _ENV_BINS:
            if bin_dir in path.split(":"):
                return path  # the image already puts it on PATH
            rc, _, _ = self._call([self._docker, "exec", name, "test", "-x", f"{bin_dir}/python"], timeout=20.0)
            if rc == 0:
                return f"{bin_dir}:{path}"
        return path

    def _detect_timeout(self, name: str) -> bool:
        rc, _, _ = self._call([self._docker, "exec", name, "timeout", "--version"], timeout=20.0)
        return rc == 0

    def _copies_of(
        self, name: str, image: str, root: str, path_env: str, packages: Mapping[str, Mapping[str, str]]
    ) -> tuple[str, ...]:
        """Which of the base tree's top-level packages the image holds its own copy of (``installed_copies``), asked of
        the image's interpreter in the prepared container (``_IMPORTS_PROBE``) once per image, root and base tree. A
        tree with no top-level package asks nothing. A probe that does not answer refuses the workspace, since a run
        could then be testing the image's copy instead of the patched tree."""
        if not packages:
            return ()
        key = (image, root, json.dumps(packages, sort_keys=True))
        if key not in self._copies:
            argv = [
                self._docker,
                "exec",
                "-w",
                "/",
                "-e",
                f"PATH={path_env}",
                "-e",
                f"HOME={RUN_HOME}",
                name,
                "python",
                "-c",
                _IMPORTS_PROBE,
                *sorted(packages),
            ]
            rc, out, err = self._call(argv, timeout=_IMPORTS_TIMEOUT_S)
            if rc != 0:
                said = _said(err.tail_text() or out.tail_text())
                raise SandboxError(
                    f"could not ask the image's python where it imports {sorted(packages)} from (exit {rc}): {said}"
                )
            self._copies[key] = tuple(installed_copies(out.text() or "", packages, root))
        return self._copies[key]

    @contextmanager
    def _container(
        self, workspace: Workspace, st: _ContainerState, fresh: bool
    ) -> Iterator[tuple[str, _ContainerState]]:
        """The prepared container, or a fresh one made the same way and removed when the block ends."""
        if not fresh:
            yield workspace.id, st
            return
        name = self._new_name(f"{workspace.side.value}-fresh")
        self._create_and_start(name, st.image, st.runtime, st.root)
        st.extra_containers.append(name)
        try:
            if st.tar_path is not None:
                self._inject_tar(name, st.tar_path, st.root)
            self._make_home(name)
            yield name, replace(st, has_timeout=self._detect_timeout(name), extra_containers=[])
        finally:
            self._force_rm(name)
            st.extra_containers.remove(name)

    def _exec(
        self,
        name: str,
        st: _ContainerState,
        command: Sequence[str],
        timeout_s: float,
        files: Mapping[str, str] | None,
        env: Mapping[str, str],
        seed: int | None,
        side: Side,
        fresh: bool,
        head: int,
    ) -> tuple[RunResult, Captured, Captured]:
        if files:
            self._inject_bytes(name, _tar_of_files(files), st.root)
        variables = {"PATH": st.path_env, **env}
        if st.import_root is not None:
            # The patched tree ahead of the image's own copy of it, then the run's PYTHONPATH, else the image's.
            rest = variables.get("PYTHONPATH", _env_map(self._image_env.get(st.image, [])).get("PYTHONPATH", ""))
            variables["PYTHONPATH"] = ":".join(p for p in (st.import_root, rest) if p)
        env_flags: list[str] = []
        for key, value in variables.items():
            env_flags += ["-e", f"{key}={value}"]
        inner: list[str] = [self._docker, "exec", "-w", st.root, *env_flags, name]
        if st.has_timeout:
            inner += ["timeout", "-k", str(_KILL_AFTER_S), str(math.ceil(timeout_s))]
        inner += list(command)
        started = self._clock()
        keep = Keep(head=head, tail=self._truncate)
        rc, out, err = self._call(inner, timeout=timeout_s + _KILL_AFTER_S + self._slack, keep=keep)
        duration = max(0.0, self._clock() - started)
        # Killed for time only when the timeout's exit code comes after the timeout: a command that exits 124 on its
        # own, or one the OOM killer ends (137) early, is an error, not a timeout.
        timed_out = rc in (_TIMEOUT_EXIT, _KILLED_EXIT) and duration >= timeout_s
        outcome = Outcome.PASS if rc == 0 else (Outcome.TIMEOUT if timed_out else Outcome.ERROR)
        result = RunResult(
            side=side,
            outcome=outcome,
            failure_type=FailureType.TIMEOUT if outcome is Outcome.TIMEOUT else None,
            exit_code=rc,
            seed=seed,
            fresh_container=fresh,
            duration_s=duration,
            stdout_tail=out.tail_text(),
            stderr_tail=err.tail_text(),
            image_digest=st.digest,
        )
        return result, out, err

    def _read_file(self, name: str, path: str, max_bytes: int) -> tuple[str | None, str | None]:
        """A file in a container, read with ``head`` so at most ``max_bytes`` plus one cross; (text, problem)."""
        argv = [self._docker, "exec", name, "head", "-c", str(max_bytes + 1), path]
        rc, out, _ = self._call(argv, timeout=60.0, keep=Keep(head=max_bytes + 1))
        if rc != 0:
            return None, None
        if out.total > max_bytes:
            return None, f"{posixpath.basename(path)} is larger than {max_bytes} bytes"
        return out.text(), None

    def _read_report(self, name: str, report_path: str) -> tuple[str | None, str | None]:
        report, problem = self._read_file(name, report_path, MAX_REPORT_BYTES)
        self._call([self._docker, "exec", name, "rm", "-f", report_path], timeout=30.0)
        return report, problem

    def _force_rm(self, name: str) -> None:
        self._call([self._docker, "rm", "-f", "-v", name], timeout=60.0)


def _bind(runner: SuiteRunner, root: str) -> SuiteRunner:
    """The lane's own runners resolve frames against the workspace root they are bound to."""
    with_root = getattr(runner, "with_root", None)
    if callable(with_root):
        bound = with_root(root)
        if isinstance(bound, type(runner)):
            return bound
    return runner


def _output_for(stream: str, out: Captured, err: Captured) -> str | None:
    """What a runner reads its cases from: one stream, or both; None when what it needs did not fit."""
    if stream == "none":
        return ""
    if stream == "stdout":
        return out.text()
    if stream == "stderr":
        return err.text()
    texts = (out.text(), err.text())
    return None if None in texts else "\n".join(t for t in texts if t is not None)
