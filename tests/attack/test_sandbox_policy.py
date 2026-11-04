"""The sandbox without Docker: the run flags, the runtime and pin rules, the exact docker argument lists, the Tier A
tree, the patched tree ahead of an image's own copy of the repository, run_test's report handling, and the bounded
output reader.

A fake runner stands in for the docker CLI. It records every argument list and answers like a daemon would, so these
tests can assert the runtime flag, every isolation flag, --pull=never, the environment checks, truncation, the timeout
labels, the history strip, the report path and the order of reads and removals on each code path. Deleting any one
of those from the sandbox fails a test here.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import IO

import pytest

import receipts.attack.sandbox as sandbox_mod
from receipts.attack.classify import SuiteRun
from receipts.attack.git_history import init_single_commit, reachable_commit_count
from receipts.attack.runners import DjangoRunner, PytestRunner, SympyRunner
from receipts.attack.sandbox import (
    RUN_HOME,
    BuildError,
    Captured,
    DockerSandbox,
    EgressPolicy,
    Keep,
    RuntimeRefused,
    SandboxError,
    SuiteSandbox,
    apply_patch,
    choose_runtime,
    docker_available,
    image_is_digest_pinned,
    refused_env_names,
    runsc_available,
    truncate_tail,
)
from receipts.contracts import (
    RUN_PHASE_LIMITS,
    FailureType,
    Outcome,
    PatchMethod,
    RunResult,
    SandboxLimits,
    Side,
    Target,
    Tier,
)

PINNED = "python@sha256:" + "a" * 64
OTHER_PINNED = "python@sha256:" + "d" * 64
ID_PINNED = "sha256:" + "b" * 64
COMMITTED = "sha256:" + "e" * 64
IMAGE_ENV = ["PATH=/usr/local/bin:/usr/bin:/bin", "LANG=C.UTF-8", "PYTHON_VERSION=3.13.9"]
CONDA = "/opt/miniconda3/envs/testbed/bin"
JUNIT_PASS = (Path(__file__).parent / "fixtures" / "junit" / "pass.xml").read_text()
JUNIT_FAIL = (Path(__file__).parent / "fixtures" / "junit" / "assertion.xml").read_text()

# The spec's run-phase values, in docker's own syntax, as a check on what the contract renders (RT2, REG-565).
SPEC_RUN = [
    "--network=none",
    "--cap-drop=ALL",
    "--security-opt=no-new-privileges",
    "--pids-limit=512",
    "--memory=4g",
    "--memory-swap=4g",
    "--cpus=2",
    "--user=65534:65534",
    "--read-only",
]
_KEEP_SMALL = Keep(head=4 * 1024 * 1024)
# Flags that would open a hole; none may appear on any create.
FORBIDDEN = ("-e", "--env", "--env-file", "-v", "--volume", "--mount", "--privileged", "--cap-add", "-p", "--publish")


def tar_of(entries: dict[str, str | None]) -> bytes:
    """A tar like `docker cp` writes: every name under one top directory. None makes a directory."""
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for name, content in entries.items():
            info = tarfile.TarInfo(name)
            if content is None:
                info.type, info.mode = tarfile.DIRTYPE, 0o755
                tar.addfile(info)
            else:
                data = content.encode()
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
    return buf.getvalue()


IMAGE_TREE = tar_of(
    {
        "testbed": None,
        "testbed/src": None,
        "testbed/src/mod.py": "def f():\n    return 1\n",
        "testbed/tests": None,
        "testbed/tests/test_mod.py": "from mod import f\n\n\ndef test_f():\n    assert f() == 1\n",
        "testbed/.git": None,
        "testbed/.git/HEAD": "ref: refs/heads/main\n",
        "testbed/.git/refs/heads/future": "f" * 40 + "\n",
        "testbed/vendor/.git": None,
        "testbed/vendor/.git/config": "[core]\n",
        "testbed/vendor/lib.py": "X = 1\n",
    }
)
PATCH = """diff --git a/src/mod.py b/src/mod.py
--- a/src/mod.py
+++ b/src/mod.py
@@ -1,2 +1,2 @@
 def f():
-    return 1
+    return 2
"""


class FakeDocker:
    """Records every docker argument list and answers like a daemon would."""

    def __init__(
        self,
        *,
        image_env: list[str] | None = None,
        container_env: list[str] | None = None,
        committed_env: list[str] | None = None,
        exec_result: tuple[int, bytes, bytes] = (0, b"ok\n", b""),
        exec_seconds: float = 0.1,
        has_timeout: bool = True,
        has_conda: bool = False,
        runtimes: object = None,
        reports: Sequence[str | None] = (),
        tree: bytes = IMAGE_TREE,
        gnu_patch: tuple[int, bytes, bytes] = (1, b"", b"patch: **** Only garbage was found in the patch input.\n"),
        patched_tree: bytes | None = None,
        imports: Callable[[list[str]], tuple[int, bytes, bytes]] | None = None,
    ) -> None:
        self.image_env = IMAGE_ENV if image_env is None else image_env
        self.container_env = container_env
        self.committed_env = committed_env
        self.exec_result = exec_result
        self.exec_seconds = exec_seconds
        self.has_timeout = has_timeout
        self.has_conda = has_conda
        self.runtimes = {"runc": {}} if runtimes is None else runtimes
        self.reports = list(reports)
        self.tree = tree
        self.gnu_patch = gnu_patch
        self.patched_tree = patched_tree
        self.imports = imports
        self.files: dict[str, bytes] = {}
        self.calls: list[list[str]] = []
        self.stdins: list[bytes | None] = []
        self.now = 1000.0

    def clock(self) -> float:
        return self.now

    def __call__(
        self,
        argv: Sequence[str],
        *,
        timeout: float,
        stdin: bytes | Path | None = None,
        keep: Keep = _KEEP_SMALL,
        consume: Callable[[IO[bytes]], None] | None = None,
    ) -> tuple[int, Captured, Captured]:
        rc, out, err = self.answer(list(argv), stdin.read_bytes() if isinstance(stdin, Path) else stdin, consume)
        return rc, Captured.of(out, keep), Captured.of(err, keep)

    def answer(
        self, argv: list[str], stdin: bytes | None, consume: Callable[[IO[bytes]], None] | None
    ) -> tuple[int, bytes, bytes]:
        self.calls.append(argv)
        self.stdins.append(stdin)
        sub = argv[1]
        if sub == "version":
            return 0, b"29.4.0\n", b""
        if sub == "info":
            return 0, json.dumps(self.runtimes).encode(), b""
        if sub in ("image", "container") and argv[2] == "inspect":
            if sub == "container":
                env = self.container_env if self.container_env is not None else self.image_env
            elif argv[-1] == COMMITTED:
                env = self.committed_env if self.committed_env is not None else self.image_env
            else:
                env = self.image_env
            return 0, json.dumps(env).encode(), b""
        if sub in ("create", "start", "rm", "rmi"):
            return 0, b"", b""
        if sub == "cp":
            assert consume is not None
            consume(io.BytesIO(self.tree))
            return 0, b"", b""
        if sub == "commit":
            return 0, f"{COMMITTED}\n".encode(), b""
        if sub == "exec" and "tar" in argv and "-c" in argv:
            assert consume is not None and self.patched_tree is not None, argv
            consume(io.BytesIO(self.patched_tree))  # the patched tree, streamed back out of the patch container
            return 0, b"", b""
        if sub == "exec":
            return self.answer_exec(argv[2:], stdin)
        raise AssertionError(f"unexpected docker call: {argv}")

    def answer_exec(self, rest: list[str], stdin: bytes | None) -> tuple[int, bytes, bytes]:
        if "--fuzz=5" in rest:
            return self.gnu_patch
        if sandbox_mod._IMPORTS_PROBE in rest:
            assert self.imports is not None, f"the import probe ran on a fake with no answer for it: {rest}"
            return self.imports(rest)
        if rest[-2:] == ["timeout", "--version"]:
            return (0 if self.has_timeout else 127), b"", b""
        if rest[-3:-1] == ["test", "-x"]:
            return (0 if self.has_conda and rest[-1] == f"{CONDA}/python" else 1), b"", b""
        if "tar" in rest and "-x" in rest:
            return 0, b"", b""
        if rest[-3:-1] == ["mkdir", "-p"] or rest[-3:-1] == ["rm", "-f"]:
            return 0, b"", b""
        if "head" in rest:
            path = rest[-1]
            if path in self.files:
                return 0, self.files[path], b""
            return 1, b"", b"head: cannot open: No such file or directory"
        self.now += self.exec_seconds
        junit = next((a.split("=", 1)[1] for a in rest if a.startswith("--junitxml=")), None)
        if junit is not None and self.reports:
            report = self.reports.pop(0)
            if report is not None:
                self.files[junit] = report.encode()
        return self.exec_result

    def of(self, sub: str) -> list[list[str]]:
        return [c for c in self.calls if c[1] == sub]

    def command_execs(self) -> list[list[str]]:
        def control(c: list[str]) -> bool:
            return (
                c[-2:] == ["timeout", "--version"]
                or ("tar" in c and ("-x" in c or "-c" in c))
                or "head" in c
                or "--fuzz=5" in c
                or sandbox_mod._IMPORTS_PROBE in c
                or c[-3:-1] in (["mkdir", "-p"], ["rm", "-f"], ["test", "-x"])
            )

        return [c for c in self.of("exec") if not control(c)]


def _target(
    tier: Tier = Tier.A, image: str | None = None, patch: str = "diff --git a b\n", repo: str = "o/r"
) -> Target:
    return Target(id="entry~repo-1", tier=tier, repo=repo, base_sha="a" * 40, patch=patch, image=image)


def _sandbox(fake: FakeDocker, tmp_path: Path | None = None, **kw: object) -> DockerSandbox:
    kw.setdefault("image_resolver", lambda _t: PINNED)
    kw.setdefault("runsc", False)
    kw.setdefault("strip_history", False)
    if "checkout" not in kw:
        tree = (tmp_path or Path("/nonexistent")) / "checkout"
        kw.setdefault("checkout", None if tmp_path is None else (lambda _t, _s, _p: _plain_tree(tree)))
    return DockerSandbox(runner=fake, clock=fake.clock, **kw)  # type: ignore[arg-type]


def _plain_tree(root: Path) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / "a.py").write_text("x = 1\n")
    return root


# A patch that applies to the plain checkout tree, for tests that prepare the PATCH side.
PLAIN_PATCH = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-x = 1\n+x = 2\n"


def _flags_before_image(argv: list[str], image: str, flags: list[str]) -> None:
    at = argv.index(image)
    for flag in flags:
        assert flag in argv[:at], f"{flag} missing before the image in {argv}"
    for bad in FORBIDDEN:
        assert bad not in argv[:at], f"{bad} must not be on a create: {argv}"


def _env_of(cmd: list[str]) -> dict[str, str]:
    return dict(cmd[i + 1].split("=", 1) for i, part in enumerate(cmd) if part == "-e")


# -- the run phase's argument lists ---------------------------------------------------------------------------------


@pytest.mark.parametrize(("runsc", "runtime"), [(True, "runsc"), (False, "runc")])
def test_run_phase_create_carries_the_runtime_and_the_contracts_run_flags(
    runsc: bool, runtime: str, tmp_path: Path
) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path, runsc=runsc)
    ws = sb.prepare(_target(), Side.BASE)
    (create,) = fake.of("create")
    flags = RUN_PHASE_LIMITS.run_flags(workspace="/testbed", tmp="/tmp")
    _flags_before_image(create, PINNED, [f"--runtime={runtime}", "--init", "--pull=never", *flags, *SPEC_RUN])
    at = create.index(PINNED)
    # Exactly the contract's rendering, in its order, with nothing of the sandbox's own in between.
    start = create.index(flags[0])
    assert create[start : start + len(flags)] == flags
    assert create[at + 1 :] == ["sleep", "3600"]
    assert ws.root == "/testbed" and ws.image_digest == "sha256:" + "a" * 64


@pytest.mark.parametrize(("tier", "root"), [(Tier.A, "/testbed"), (Tier.B, "/workspace")])
def test_the_workspace_is_the_images_repository_path_for_tier_a(tier: Tier, root: str, tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path, runsc=True)
    ws = sb.prepare(_target(tier), Side.BASE)
    (create,) = fake.of("create")
    assert ws.root == root
    assert f"--tmpfs={root}:rw,exec,nosuid,nodev,size=1073741824,uid=65534,gid=65534,mode=0755" in create
    assert create[create.index("-w") + 1] == root


def test_a_fresh_run_creates_its_container_with_the_same_flags(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path, runsc=True)
    ws = sb.prepare(_target(patch=PLAIN_PATCH), Side.PATCH)
    assert ws.patch_method is PatchMethod.GIT_APPLY
    result = sb.run(ws, ["python", "-c", "pass"], timeout_s=10, fresh=True)
    assert result.fresh_container is True
    creates = fake.of("create")
    assert len(creates) == 2
    assert creates[1][creates[1].index("--init") :] == creates[0][creates[0].index("--init") :]
    assert fake.of("rm")  # the fresh container is removed after its run


def test_the_flags_come_from_the_sandboxs_limits(tmp_path: Path) -> None:
    limits = SandboxLimits(pids_limit=256, memory="8g", cpus=1.5, shm_size="32m", workspace_bytes=2 * 1024**3)
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path, limits=limits)
    sb.prepare(_target(), Side.BASE)
    (create,) = fake.of("create")
    for flag in limits.run_flags(workspace="/testbed"):
        assert flag in create
    assert "--pids-limit=256" in create and "--memory-swap=8g" in create and "--cpus=1.5" in create
    sb.build_phase(_target(), [["true"]], base_image=PINNED)
    build = fake.of("create")[-1]
    assert build[build.index("--network=none") + 1 : build.index("-w")] == limits.limit_flags()


def test_the_default_limits_are_the_contracts_and_are_read_at_construction(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    limits = SandboxLimits(pids_limit=128)
    monkeypatch.setattr(sandbox_mod, "RUN_PHASE_LIMITS", limits)
    fake = FakeDocker()
    _sandbox(fake, tmp_path).prepare(_target(), Side.BASE)
    assert "--pids-limit=128" in fake.of("create")[0]
    assert RUN_PHASE_LIMITS.pids_limit == 512  # the real contract is untouched


def test_a_run_execs_with_the_workspace_the_timeout_home_and_its_env(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    sb.run(ws, ["python", "-m", "pytest"], timeout_s=2.5, env={"RECEIPTS_PROBE_RUN": "1"}, seed=3)
    (cmd,) = fake.command_execs()
    assert cmd[:4] == ["docker", "exec", "-w", "/testbed"]
    assert cmd[cmd.index(ws.id) + 1 :] == ["timeout", "-k", "5", "3", "python", "-m", "pytest"]  # ceil(2.5) = 3
    env = _env_of(cmd)
    assert env["HOME"] == RUN_HOME == "/tmp/home"  # inside the /tmp mount, never nobody's /nonexistent
    assert env["RECEIPTS_PROBE_RUN"] == "1" and env["PYTHONHASHSEED"] == "3"
    assert env["PATH"] == "/usr/local/bin:/usr/bin:/bin"  # the image's own PATH
    assert ["docker", "exec", ws.id, "mkdir", "-p", "/tmp/home"] in fake.calls  # HOME exists before any run


def test_the_images_conda_environment_goes_first_on_path(tmp_path: Path) -> None:
    fake = FakeDocker(has_conda=True)
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    sb.run(ws, ["python", "-V"], timeout_s=5)
    assert _env_of(fake.command_execs()[0])["PATH"] == f"{CONDA}:/usr/local/bin:/usr/bin:/bin"


def test_an_image_that_puts_its_environment_on_path_keeps_its_path(tmp_path: Path) -> None:
    image_env = [f"PATH={CONDA}:/opt/miniconda3/bin:/usr/bin:/bin", "LANG=C.UTF-8"]
    fake = FakeDocker(image_env=image_env, has_conda=True)
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    sb.run(ws, ["python", "-V"], timeout_s=5)
    assert _env_of(fake.command_execs()[0])["PATH"] == f"{CONDA}:/opt/miniconda3/bin:/usr/bin:/bin"  # no duplicate


def test_cleanup_closes_every_workspace_and_removes_the_hosts_copies(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    first, second = sb.prepare(_target(patch=PATCH), Side.BASE), sb.prepare(_target(patch=PATCH), Side.PATCH)
    work = next(tmp_path.iterdir())
    assert work.name.startswith("receipts-sandbox-") and any(work.iterdir())
    sb.cleanup()
    assert not work.exists() and sb._state == {}
    removed = {c[-1] for c in fake.of("rm")}
    assert {first.id, second.id} <= removed


def test_a_test_run_names_a_test(tmp_path: Path) -> None:
    sb = _sandbox(FakeDocker(), tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    with pytest.raises(SandboxError, match="at least one test"):
        sb.run_cases(ws, [], runner=PytestRunner(), changed_files=[], timeout_s=5)


# -- the Tier A tree ------------------------------------------------------------------------------------------------


def _injected(fake: FakeDocker, name: str) -> dict[str, str]:
    """The files of the tar the sandbox streamed into container ``name``'s workspace."""
    for argv, data in zip(fake.calls, fake.stdins, strict=True):
        if argv[1] == "exec" and name in argv and "tar" in argv and data is not None:
            with tarfile.open(fileobj=io.BytesIO(data)) as tar:
                return {
                    m.name: (tar.extractfile(m).read().decode(errors="replace") if m.isfile() else "")  # type: ignore[union-attr]
                    for m in tar.getmembers()
                }
    raise AssertionError(f"nothing was injected into {name}")


def test_a_tier_a_tree_is_the_images_own_without_its_history_and_with_the_patch(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    target = _target(patch=PATCH)
    base = sb.prepare(target, Side.BASE)
    patched = sb.prepare(target, Side.PATCH)
    # The tree is copied out of a created, never started container, and that container is removed.
    (source,) = [c for c in fake.of("create") if c[-1] == "true"]
    assert "start" not in [c[1] for c in fake.calls if source[source.index("--name") + 1] in c]
    assert fake.of("cp")[0][-2:] == [f"{source[source.index('--name') + 1]}:/testbed", "-"]
    assert len(fake.of("cp")) == 1  # cached: copied once per image
    base_files, patch_files = _injected(fake, base.id), _injected(fake, patched.id)
    assert base_files["src/mod.py"] == "def f():\n    return 1\n"
    assert patch_files["src/mod.py"] == "def f():\n    return 2\n"
    assert "vendor/lib.py" in base_files and "vendor/.git/config" not in base_files
    assert not any(name.startswith(".git/refs/heads/future") for name in base_files)
    # The workspace's .git is one commit, rebuilt on the host; the image's refs never cross.
    st = sb._state[base.id]
    assert st.host_dir is not None and reachable_commit_count(st.host_dir) == 1
    assert sb.read_text(base, "src/mod.py") == "def f():\n    return 1\n"
    assert "tests/test_mod.py" in sb.tree_files(base) and not any(f.startswith(".git") for f in sb.tree_files(base))


def test_a_tree_that_would_escape_is_not_extracted(tmp_path: Path) -> None:
    hostile = tar_of({"testbed": None, "testbed/../../escape.py": "x = 1\n", "testbed/ok.py": "y = 1\n"})
    fake = FakeDocker(tree=hostile)
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    assert sb.tree_files(ws) == ("ok.py",)
    assert not (tmp_path.parent / "escape.py").exists()


def test_read_text_stays_inside_the_tree(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    for bad in ("../x", "/etc/passwd", ""):
        with pytest.raises(SandboxError):
            sb.read_text(ws, bad)
    with pytest.raises(SandboxError, match="larger than"):
        sb.read_text(ws, "src/mod.py", max_bytes=5)


def test_the_sandbox_is_a_suite_sandbox(tmp_path: Path) -> None:
    assert isinstance(_sandbox(FakeDocker(), tmp_path), SuiteSandbox)


# -- an image's own copy of the repository: the patched tree goes ahead of it on PYTHONPATH ---------------------------

ACME_INIT = "VERSION = (1, 11)\n"
ACME_MOD = "def shortcut():\n    return 1\n"
ACME_HASHES = {
    "__init__.py": hashlib.sha256(ACME_INIT.encode()).hexdigest(),
    "shortcuts.py": hashlib.sha256(ACME_MOD.encode()).hexdigest(),
}
# Where an image built with "python setup.py install" keeps its copy (the egg), as in the Django 1.11 and 2.2 images.
EGG = "/opt/miniconda3/envs/testbed/lib/python3.5/site-packages/Acme-1.11-py3.5.egg/acme"
ACME_TREE = tar_of(
    {
        "testbed": None,
        "testbed/acme": None,
        "testbed/acme/__init__.py": ACME_INIT,
        "testbed/acme/shortcuts.py": ACME_MOD,
        "testbed/acme/sub": None,
        "testbed/acme/sub/__init__.py": "",
        "testbed/docs": None,  # a directory with no __init__.py is no package
        "testbed/docs/conf.py": "x = 1\n",
        "testbed/tests": None,
        "testbed/tests/runtests.py": "print('ran')\n",
        "testbed/setup.py": "from setuptools import setup\n",
    }
)
ACME_INIT_PATCH = (
    "diff --git a/acme/__init__.py b/acme/__init__.py\n--- a/acme/__init__.py\n+++ b/acme/__init__.py\n"
    "@@ -1 +1 @@\n-VERSION = (1, 11)\n+VERSION = (1, 12)\n"
)


def _image_holding(
    where: str | None, files: dict[str, str] | None = None
) -> Callable[[list[str]], tuple[int, bytes, bytes]]:
    """The import probe's answer from an image whose ``acme`` resolves to ``where`` (None: to nothing) and holds
    ``files``. Run from the workspace instead, a plain ``python -c`` has the workspace on its path ('' first), and the
    package resolves to the workspace's own, as it would in a real container."""

    def answer(rest: list[str]) -> tuple[int, bytes, bytes]:
        names = rest[rest.index(sandbox_mod._IMPORTS_PROBE) + 1 :]
        from_workspace = rest[rest.index("-w") + 1] == "/testbed"
        out: dict[str, list[list[object]]] = {}
        for name in names:
            if from_workspace:
                out[name] = [[f"/testbed/{name}", ACME_HASHES]]
            else:
                out[name] = [] if where is None else [[where, ACME_HASHES if files is None else files]]
        return 0, f"RECEIPTS-IMPORTS {json.dumps(out)}\n".encode(), b""

    return answer


def _acme_sandbox(fake: FakeDocker, tmp_path: Path) -> DockerSandbox:
    return DockerSandbox(
        runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path
    )


def _probes(fake: FakeDocker) -> list[list[str]]:
    return [c for c in fake.of("exec") if sandbox_mod._IMPORTS_PROBE in c]


def test_a_run_imports_the_patched_tree_ahead_of_the_images_own_copy_of_it(tmp_path: Path) -> None:
    # PLANT (the round-5 attack review's blocker): SWE-bench builds Django 1.11 and 2.2 with "python setup.py install"
    # and its eval script installs again after applying the patch, so the harness tests the patched tree. The run
    # phase only mounted the patched tree, so `import django` read the image's egg: base code, on every side. The
    # image's own interpreter now says where it imports the tree's top-level packages from, and a copy of the base
    # tree's code puts the workspace root first on PYTHONPATH for every run in the workspace, fresh ones included.
    fake = FakeDocker(tree=ACME_TREE, imports=_image_holding(EGG))
    sb = _acme_sandbox(fake, tmp_path)
    target = _target(patch=ACME_INIT_PATCH)
    base, patched = sb.prepare(target, Side.BASE), sb.prepare(target, Side.PATCH)  # the patch changes __init__.py
    for ws in (base, patched):
        sb.run(ws, ["python", "-c", "import acme"], timeout_s=5)
        sb.run(ws, ["python", "-c", "import acme"], timeout_s=5, fresh=True)
    runs = fake.command_execs()
    assert len(runs) == 4 and [_env_of(c)["PYTHONPATH"] for c in runs] == ["/testbed"] * 4
    # Asked once per image, root and base tree, of the image's python, from / so the workspace is not on its path.
    (probe,) = _probes(fake)
    assert probe[2:4] == ["-w", "/"] and _env_of(probe) == {"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": RUN_HOME}
    assert probe[probe.index(sandbox_mod._IMPORTS_PROBE) - 2 :] == ["python", "-c", sandbox_mod._IMPORTS_PROBE, "acme"]
    assert probe[probe.index(sandbox_mod._IMPORTS_PROBE) - 3] == base.id


@pytest.mark.parametrize(
    ("imports", "why"),
    [
        (_image_holding("/testbed/acme"), "an editable install: the image imports the workspace itself"),
        (_image_holding(None), "the image holds no acme"),
        (_image_holding(EGG, {"__init__.py": "0" * 64}), "a package of that name, but not the tree's code"),
        (_image_holding(EGG, {**ACME_HASHES, "extra.py": "1" * 64}), "the tree's modules and one more"),
    ],
)
def test_an_image_with_no_copy_of_the_trees_code_runs_as_before(
    imports: Callable[[list[str]], tuple[int, bytes, bytes]], why: str, tmp_path: Path
) -> None:
    fake = FakeDocker(tree=ACME_TREE, imports=imports, image_env=[*IMAGE_ENV, "PYTHONPATH=/opt/extra"])
    sb = _acme_sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    sb.run(ws, ["python", "-V"], timeout_s=5)
    (cmd,) = fake.command_execs()
    assert "PYTHONPATH" not in _env_of(cmd), why  # the image's own PYTHONPATH stays exactly as the image sets it
    assert len(_probes(fake)) == 1


def test_the_workspace_goes_ahead_of_the_images_own_pythonpath_and_a_runs_own(tmp_path: Path) -> None:
    fake = FakeDocker(tree=ACME_TREE, imports=_image_holding(EGG), image_env=[*IMAGE_ENV, "PYTHONPATH=/opt/extra"])
    sb = _acme_sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    sb.run(ws, ["python", "-V"], timeout_s=5)
    sb.run(ws, ["python", "-V"], timeout_s=5, env={"PYTHONPATH": "/tmp/mine"})
    first, second = fake.command_execs()
    assert _env_of(first)["PYTHONPATH"] == "/testbed:/opt/extra"
    assert _env_of(second)["PYTHONPATH"] == "/testbed:/tmp/mine"


@pytest.mark.parametrize(
    "answer",
    [
        (1, b"", b"/bin/sh: python: not found\n"),
        (3, b"", b"find_spec('acme') raised RuntimeError\n"),
        (0, b"no answer at all\n", b""),
        (0, b'RECEIPTS-IMPORTS {"other": []}\n', b""),  # an answer for packages it was not asked about
        (0, b"RECEIPTS-IMPORTS [1, 2]\n", b""),
        (0, b'RECEIPTS-IMPORTS {"acme": [["/x", {"__init__.py": 7}]]}\n', b""),
    ],
)
def test_a_probe_that_does_not_answer_refuses_the_workspace(answer: tuple[int, bytes, bytes], tmp_path: Path) -> None:
    # Fail closed: without the answer a run could be testing the image's copy instead of the patched tree.
    fake = FakeDocker(tree=ACME_TREE, imports=lambda _rest: answer)
    sb = _acme_sandbox(fake, tmp_path)
    with pytest.raises(SandboxError, match="import"):
        sb.prepare(_target(), Side.BASE)
    (workspace,) = [c[c.index("--name") + 1] for c in fake.of("create") if c[-1] != "true"]
    assert any(c[1] == "rm" and c[-1] == workspace for c in fake.calls)  # the started container never leaks
    assert sb._state == {}


def test_a_tree_with_no_top_level_package_asks_nothing(tmp_path: Path) -> None:
    fake = FakeDocker()  # IMAGE_TREE: src/mod.py, tests/ and vendor/, none of them a package
    ws = _acme_sandbox(fake, tmp_path).prepare(_target(), Side.BASE)
    assert _probes(fake) == [] and ws.root == "/testbed"


def test_root_packages_reads_each_top_level_package_and_its_own_modules(tmp_path: Path) -> None:
    tree = tmp_path / "tree"
    (tree / "acme" / "sub").mkdir(parents=True)
    (tree / "acme" / "__init__.py").write_text(ACME_INIT)
    (tree / "acme" / "shortcuts.py").write_text(ACME_MOD)
    (tree / "acme" / "sub" / "__init__.py").write_text("")
    (tree / "acme" / "data.txt").write_text("not a module")
    (tree / "docs").mkdir()
    (tree / "docs" / "conf.py").write_text("x = 1\n")  # no __init__.py: no package
    (tree / "not-a-name").mkdir()
    (tree / "not-a-name" / "__init__.py").write_text("")
    (tree / "linked").symlink_to(tree / "acme")
    (tree / "acme" / "alias.py").symlink_to(tree / "acme" / "shortcuts.py")
    assert sandbox_mod.root_packages(tree) == {"acme": ACME_HASHES}


@pytest.mark.parametrize(
    ("python_path", "copies"),
    [
        ("{egg}", ["acme"]),  # the image's own copy, where an egg's easy-install.pth puts it
        ("{workspace}", []),  # an editable install: the workspace itself
        ("", []),  # the image holds no acme
    ],
)
def test_the_import_probe_tells_a_copy_from_the_workspace_in_a_real_python(
    python_path: str, copies: list[str], tmp_path: Path
) -> None:
    # The probe's own code, run by this environment's interpreter with no Docker: what it prints, read the way the
    # sandbox reads it, tells a copy of the base tree's code from the workspace itself.
    workspace = tmp_path / "testbed"
    (workspace / "acme").mkdir(parents=True)
    (workspace / "acme" / "__init__.py").write_text(ACME_INIT)
    (workspace / "acme" / "shortcuts.py").write_text(ACME_MOD)
    egg = tmp_path / "site" / "Acme-1.11-py3.5.egg"
    shutil.copytree(workspace / "acme", egg / "acme")
    env = {"PATH": os.environ.get("PATH", ""), "PYTHONPATH": python_path.format(egg=egg, workspace=workspace)}
    proc = subprocess.run(
        [sys.executable, "-c", sandbox_mod._IMPORTS_PROBE, "acme"],
        cwd="/",
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    packages = sandbox_mod.root_packages(workspace)
    assert sandbox_mod.installed_copies(proc.stdout, packages, str(workspace)) == copies
    (egg / "acme" / "shortcuts.py").write_text("def shortcut():\n    return 2\n")  # the copy is no longer the tree's
    proc = subprocess.run(
        [sys.executable, "-c", sandbox_mod._IMPORTS_PROBE, "acme"], cwd="/", env=env, capture_output=True, text=True,
        timeout=60, check=False,
    )  # fmt: skip
    assert sandbox_mod.installed_copies(proc.stdout, packages, str(workspace)) == []


# -- REG-569: a patch applies exactly the way the SWE-bench harness applies it ----------------------------------------

_LINES = "".join(f"line {i}\n" for i in range(1, 21))


def _hunk(
    context: tuple[str, str, str, str, str, str], *, removed: str = "line 10", header: str = "@@ -7,7 +7,7 @@"
) -> str:
    before, after = context[:3], context[3:]
    body = "".join(f" {c}\n" for c in before) + f"-{removed}\n+line ten\n" + "".join(f" {c}\n" for c in after)
    return f"diff --git a/src/lines.py b/src/lines.py\n--- a/src/lines.py\n+++ b/src/lines.py\n{header}\n{body}"


_RIGHT = ("line 7", "line 8", "line 9", "line 11", "line 12", "line 13")
EXACT = _hunk(_RIGHT)
FUZZ = _hunk(("wrong 7", "wrong 8", "wrong 9", "wrong 11", "wrong 12", "wrong 13"))  # no context matches at all
NONE = _hunk(("wrong 7", "wrong 8", "wrong 9", "wrong 11", "wrong 12", "wrong 13"), removed="no such line")
# GNU patch's own report of FUZZ, recorded in the sympy__sympy-13974 image (fixtures/runners/gnu_patch_fuzz.json).
GNU_FUZZ = json.loads((Path(__file__).parent / "fixtures" / "runners" / "gnu_patch_fuzz.json").read_text())
# The harness's own chain, run as the harness runs it (as root, in the image's /testbed, each command on the tree the
# one before left) on seven patches, with each command's exit code, output and the tree it left
# (fixtures/runners/harness_chain_13974.json, recorded by record.py round4).
HARNESS = json.loads((Path(__file__).parent / "fixtures" / "runners" / "harness_chain_13974.json").read_text())
_METHOD_OF = {" ".join(argv): method for method, argv in sandbox_mod.HARNESS_CHAIN}


def _committed_tree(root: Path, files: dict[str, str]) -> Path:
    for rel, text in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(text)
    init_single_commit(root)
    return root


def _tree(root: Path) -> dict[str, str]:
    return {
        p.relative_to(root).as_posix(): p.read_text()
        for p in sorted(root.rglob("*"))
        if p.is_file() and ".git" not in p.relative_to(root).parts
    }


def _never(_tree: Path, _data: bytes) -> None:
    raise AssertionError("GNU patch must not run: an earlier command applied the patch")


def test_the_chain_is_the_harnesss_three_commands_in_its_order(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # PLANT (REG-569 conformance): the chain was git apply exact, then --recount, then --recount -C1, each from the
    # clean tree, then GNU patch on a clean copy. --recount and -C1 apply patches the harness refuses, so an instance
    # could pass here on code its label was never graded on.
    assert sandbox_mod.HARNESS_CHAIN == (
        (PatchMethod.GIT_APPLY, ("git", "apply", "--verbose")),
        (PatchMethod.GIT_APPLY_REJECT, ("git", "apply", "--verbose", "--reject")),
        (PatchMethod.GNU_PATCH_FUZZ, ("patch", "--batch", "--fuzz=5", "-p1", "-i")),
    )
    assert [method for method, _ in sandbox_mod.HARNESS_CHAIN] == list(PatchMethod)  # the contract's own order
    assert list(_METHOD_OF) == HARNESS["cmds"]  # what the recording ran in the image, word for word
    calls: list[tuple[str, ...]] = []
    real = sandbox_mod._git

    def recording(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        calls.append(args[:-1])
        return real(tree, *args)

    monkeypatch.setattr(sandbox_mod, "_git", recording)
    gnu: list[bytes] = []
    repo = _committed_tree(tmp_path / "t", {"src/lines.py": _LINES})
    assert apply_patch(repo, FUZZ, fallback=lambda _t, data: gnu.append(data)) is PatchMethod.GNU_PATCH_FUZZ
    assert calls == [("apply", "--verbose"), ("apply", "--verbose", "--reject")] and gnu == [FUZZ.encode()]
    calls.clear()
    assert apply_patch(_committed_tree(tmp_path / "u", {"src/lines.py": _LINES}), EXACT, fallback=_never) is (
        PatchMethod.GIT_APPLY
    )
    assert calls == [("apply", "--verbose")]  # the first exit 0 wins: nothing after it runs


@pytest.mark.parametrize("case", sorted(HARNESS["cases"]))
def test_each_patch_ends_as_the_harnesss_chain_ended_it_in_the_image(case: str, tmp_path: Path) -> None:
    # The two git commands run here on the host (git 2.54 or the VM's), GNU patch in the image: the tree each one
    # hands on, the command that applied the patch and the tree the run gets must be the harness's own. The cases:
    # exact (git apply); fuzz (GNU patch with fuzz); reject_then_gnu (--reject applies a.py and leaves b.py.rej, then
    # GNU patch, seeing a.py's hunk already applied, reverses it and fuzzes b.py: a.py ends at base); reverse (a hunk
    # whose new side the file already has, which GNU patch --batch reverses); exact_then_fuzz (--reject applies hunk
    # 1, GNU patch reverses it and fails hunk 2: refused, though GNU patch alone would have applied it); none; and
    # no_newline (an exact patch without its last newline, which both git commands call corrupt).
    # PLANT: with the round-3 chain, reject_then_gnu ran a.py's change, exact_then_fuzz applied, and no_newline was
    # git_apply (a newline was added to the diff), so three instances would run code their labels were not graded on.
    rec = HARNESS["cases"][case]
    top = f"receipts_chain/{case}"
    steps = rec["steps"]
    gnu = steps[2] if len(steps) == 3 else None
    patched = {f"testbed/{top}/{rel}": text for rel, text in (gnu["tree"] if gnu else {}).items()}
    fake = FakeDocker(
        gnu_patch=(gnu["exit_code"], gnu["stdout"].encode(), gnu["stderr"].encode()) if gnu else (1, b"", b"unused"),
        patched_tree=tar_of({"testbed": None, "testbed/receipts_chain": None, f"testbed/{top}": None, **patched}),
    )

    def checkout(_t: Target, _s: Side, _p: str | None) -> Path:
        return _committed_tree(
            tmp_path / f"tree-{len(list(tmp_path.iterdir()))}", {f"{top}/{k}": v for k, v in rec["files"].items()}
        )

    sb = _sandbox(fake, tmp_path, checkout=checkout)
    target = _target(patch=rec["diff"])
    if rec["applied_by"] is None:
        with pytest.raises(SandboxError, match="the patch does not apply: git apply --verbose exited"):
            sb.prepare(target, Side.PATCH)
        assert not [c for c in fake.of("exec") if "tar" in c and "-c" in c]  # a refused tree is never read back
        return
    ws = sb.prepare(target, Side.PATCH)
    assert ws.patch_method is _METHOD_OF[rec["applied_by"]]
    if gnu is not None:
        (helper,) = [n for c in fake.of("create") if "receipts-gnupatch-" in (n := c[c.index("--name") + 1])]
        handed = {k[len(top) + 1 :]: v for k, v in _injected(fake, helper).items() if k.startswith(top + "/")}
        assert handed == steps[1]["tree"]  # what the harness's --reject left, .rej files and all
        (data,) = [d for c, d in zip(fake.calls, fake.stdins, strict=True) if helper in c and "/tmp" in c and d]
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            member = tar.extractfile("receipts-patch.diff")
            assert member is not None and member.read() == rec["diff"].encode()  # byte for byte, as the harness's
    got = {rel[len(top) + 1 :]: sb.read_text(ws, rel) for rel in sb.tree_files(ws) if rel.startswith(top + "/")}
    assert got == steps[-1]["tree"]  # the tree the tests run on is the harness's


def test_a_patch_git_takes_only_with_reject_is_named_by_that_command(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # git apply --verbose --reject exits 0 only where git apply --verbose does (every case in the recording, and 14
    # patch shapes probed on git 2.54.0 this round), so the second command wins only when the first failed for a
    # reason the second did not share. Scripted here: the first command fails, the second is real git.
    real = sandbox_mod._git

    def first_fails(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:-1] == ("apply", "--verbose"):
            return subprocess.CompletedProcess(list(args), 1, "", "error: scripted")
        return real(tree, *args)

    monkeypatch.setattr(sandbox_mod, "_git", first_fails)
    repo = _committed_tree(tmp_path / "t", {"src/lines.py": _LINES})
    assert apply_patch(repo, EXACT, fallback=_never) is PatchMethod.GIT_APPLY_REJECT
    assert (repo / "src/lines.py").read_text() == _LINES.replace("line 10\n", "line ten\n")


def test_each_command_runs_on_the_tree_the_one_before_it_left(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # PLANT (the reset is gone): each attempt started from the clean tree, and a changed tree stopped the chain. The
    # harness resets nothing, so what the first command wrote is still there for the second, and for GNU patch.
    real = sandbox_mod._git

    def first_writes(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        if args[:-1] == ("apply", "--verbose"):
            (tree / "left_by_the_first.txt").write_text("x\n")
            return subprocess.CompletedProcess(list(args), 1, "", "error: patch failed")
        assert (tree / "left_by_the_first.txt").exists()  # the second command sees it
        return real(tree, *args)

    monkeypatch.setattr(sandbox_mod, "_git", first_writes)
    seen: list[dict[str, str]] = []
    repo = _committed_tree(tmp_path / "t", {"src/lines.py": _LINES})
    assert apply_patch(repo, FUZZ, fallback=lambda tree, _d: seen.append(_tree(tree))) is PatchMethod.GNU_PATCH_FUZZ
    (handed,) = seen
    assert handed["left_by_the_first.txt"] == "x\n" and "src/lines.py.rej" in handed  # and what --reject left


def test_a_tree_with_no_commit_still_reaches_gnu_patch(tmp_path: Path) -> None:
    repo = tmp_path / "t"
    (repo / "src").mkdir(parents=True)
    (repo / "src/lines.py").write_text(_LINES)
    subprocess.run(["git", "-C", str(repo), "init", "-q"], check=True, capture_output=True)
    used: list[bytes] = []
    assert apply_patch(repo, FUZZ, fallback=lambda _t, data: used.append(data)) is PatchMethod.GNU_PATCH_FUZZ
    assert used == [FUZZ.encode()]


def test_a_patch_no_command_applies_says_what_each_one_said(tmp_path: Path) -> None:
    repo = _committed_tree(tmp_path / "t", {"src/lines.py": _LINES})

    def refuses(_tree: Path, _data: bytes) -> None:
        raise SandboxError("patch --batch --fuzz=5 -p1 -i exited 1: Hunk #1 FAILED at 7.")

    with pytest.raises(SandboxError) as refused:
        apply_patch(repo, NONE, fallback=refuses)
    said = str(refused.value)
    first, second, third = said.removeprefix("the patch does not apply: ").split("; ")
    assert first.startswith("git apply --verbose exited 1: ") and "error: patch failed: src/lines.py:7" in first
    assert second.startswith("git apply --verbose --reject exited 1: ") and "src/lines.py:7" in second
    assert third == "patch --batch --fuzz=5 -p1 -i exited 1: Hunk #1 FAILED at 7."
    with pytest.raises(SandboxError, match="git apply --verbose exited 128: error: corrupt patch at "):
        apply_patch(repo, "diff --git a/x b/x\n--- a/x\n+++ b/x\n@@ -1 +1 @@\n-a")  # no GNU patch here


def test_the_hosts_temp_path_never_reaches_a_reason(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # git 2.4x and later name the diff file they read ("corrupt patch at /tmp/tmpab12.diff:12"; older git says "at line
    # 12"). The file is the host's temp path, which must not travel to a receipt. Scripted, so it holds on any git.
    def names_its_file(_tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(list(args), 128, "", f"error: corrupt patch at {args[-1]}:12\n")

    monkeypatch.setattr(sandbox_mod, "_git", names_its_file)
    with pytest.raises(SandboxError) as corrupt:
        apply_patch(_committed_tree(tmp_path / "t", {"a.py": "x\n"}), EXACT)
    said = str(corrupt.value)
    assert said.count("error: corrupt patch at the patch:12") == 2
    assert tempfile.gettempdir() not in said and ".diff" not in said


def _patched_tree(content: str) -> bytes:
    entries: dict[str, str | None] = {
        "testbed": None,
        "testbed/src": None,
        "testbed/src/mod.py": content,
        "testbed/tests": None,
        "testbed/tests/test_mod.py": "from mod import f\n\n\ndef test_f():\n    assert f() == 1\n",
        "testbed/.git": None,
        "testbed/.git/HEAD": "ref: refs/heads/planted\n",  # a .git from the container never comes back
    }
    return tar_of(entries)


# src/mod.py in IMAGE_TREE is 'def f():\n    return 1\n'; this hunk's one line of context is wrong, so neither git
# command takes it and only GNU patch with fuzz does.
MOD_FUZZ = (
    "diff --git a/src/mod.py b/src/mod.py\n--- a/src/mod.py\n+++ b/src/mod.py\n"
    "@@ -1,2 +1,2 @@\n def g():\n-    return 1\n+    return 2\n"
)


def test_gnu_patch_runs_in_a_run_phase_container_of_the_image_on_the_tree_reject_left(tmp_path: Path) -> None:
    fake = FakeDocker(
        gnu_patch=(0, GNU_FUZZ["fuzz"]["stdout"].encode(), b""), patched_tree=_patched_tree("def f():\n    return 2\n")
    )
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=True, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    ws = sb.prepare(_target(patch=MOD_FUZZ), Side.PATCH)
    assert ws.patch_method is PatchMethod.GNU_PATCH_FUZZ
    helper = next(
        c[c.index("--name") + 1] for c in fake.of("create") if "receipts-gnupatch-" in c[c.index("--name") + 1]
    )
    (create,) = [c for c in fake.of("create") if helper in c]
    # The patch runs in a run-phase container of the target's own image: no network, uid 65534, read-only root.
    flags = RUN_PHASE_LIMITS.run_flags(workspace="/testbed", tmp="/tmp")
    assert create[create.index(flags[0]) : create.index(flags[0]) + len(flags)] == flags and "--runtime=runsc" in create
    (gnu,) = [c for c in fake.of("exec") if "--fuzz=5" in c]
    assert gnu[gnu.index(helper) + 1 :] == ["patch", "--batch", "--fuzz=5", "-p1", "-i", "/tmp/receipts-patch.diff"]
    assert gnu[gnu.index("-w") + 1] == "/testbed"
    handed = _injected(fake, helper)
    # PLANT (the reset is gone): GNU patch got a copy of the clean tree. It gets the tree --reject left: the file as it
    # was, since no hunk applied, and the hunk git rejected.
    assert handed["src/mod.py"] == "def f():\n    return 1\n" and "+    return 2" in handed["src/mod.py.rej"]
    assert not any(name.startswith(".git") for name in handed)
    assert any(c[1] == "rm" and c[-1] == helper for c in fake.calls)  # the patch container is removed
    # The run's workspace holds the patched tree, with the host's own one-commit history, not the container's .git.
    assert _injected(fake, ws.id)["src/mod.py"] == "def f():\n    return 2\n"
    st = sb._state[ws.id]
    assert st.host_dir is not None and reachable_commit_count(st.host_dir) == 1
    assert (st.host_dir / ".git" / "HEAD").read_text() != "ref: refs/heads/planted\n"
    base = sb.prepare(_target(patch=MOD_FUZZ), Side.BASE)
    assert base.patch_method is None and sb.read_text(base, "src/mod.py") == "def f():\n    return 1\n"


def test_a_patch_no_command_applies_is_refused_and_its_tree_is_thrown_away(tmp_path: Path) -> None:
    fake = FakeDocker()  # GNU patch answers "Only garbage was found in the patch input."
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    bad = PATCH.replace("-    return 1", "-    return 9")
    with pytest.raises(
        SandboxError, match=r"does not apply: .*patch --batch --fuzz=5 -p1 -i exited 1: patch: \*\*\*\*"
    ):
        sb.prepare(_target(patch=bad), Side.PATCH)
    names = [c[c.index("--name") + 1] for c in fake.of("create")]
    assert all("receipts-tree-" in n or "receipts-gnupatch-" in n for n in names)  # no run container was made
    assert all(any(c[1] == "rm" and c[-1] == n for c in fake.calls) for n in names)
    assert not [c for c in fake.of("exec") if "tar" in c and "-c" in c]  # nothing was copied back
    assert sb._state == {}
    # PLANT: the tree --reject left (with its .rej) stayed on the host; a refused side's tree is removed.
    assert not list(sb._work.glob("ws-patch-*"))
    base = sb.prepare(_target(patch=bad), Side.BASE)  # the cached base tree is untouched
    assert sb.read_text(base, "src/mod.py") == "def f():\n    return 1\n" and "src/mod.py.rej" not in sb.tree_files(
        base
    )


def test_a_gnu_patch_that_rejects_a_hunk_never_lands_in_part(tmp_path: Path) -> None:
    rejected = (1, b"patching file src/mod.py\nHunk #2 FAILED at 9.\n1 out of 2 hunks FAILED -- saving rejects\n", b"")
    fake = FakeDocker(gnu_patch=rejected, patched_tree=_patched_tree("half patched\n"))
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    with pytest.raises(SandboxError, match=r"exited 1: .*1 out of 2 hunks FAILED"):
        sb.prepare(_target(patch=MOD_FUZZ), Side.PATCH)
    assert not [c for c in fake.of("exec") if "tar" in c and "-c" in c]  # the partial tree is never read back


def test_every_prepared_workspace_names_how_its_patch_applied(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    target = _target(patch=PATCH)
    assert sb.prepare(target, Side.BASE).patch_method is None
    assert sb.prepare(target, Side.PATCH).patch_method is PatchMethod.GIT_APPLY
    assert sb.prepare(target, Side.GOLD, patch=PATCH).patch_method is PatchMethod.GIT_APPLY
    assert sb.prepare(target, Side.GOLD).patch_method is None  # a gold side with no patch given applies nothing
    with pytest.raises(SandboxError, match="no tree to apply it to"):
        _sandbox(FakeDocker(), checkout=lambda _t, _s, _p: None).prepare(_target(patch=PATCH), Side.PATCH)


def test_an_attacker_whose_patch_nothing_applies_says_no_env(tmp_path: Path) -> None:
    from receipts.attack.correctness import CorrectnessAttacker
    from receipts.contracts import Arm, Attack, ContextMask, Lens, UncheckedReason

    sb = DockerSandbox(runner=FakeDocker(), runsc=False, image_resolver=lambda _t: PINNED, work_dir=tmp_path)
    attack = Attack(
        id="a1", claim_id="fixed_x", lens=Lens.CORRECTNESS, hypothesis="h", mask=ContextMask.DIFF, temperature=0.0
    )
    outcome = CorrectnessAttacker().attack(
        _target(patch=PATCH.replace("-    return 1", "-    return 9")), attack, arm=Arm(name="spark"),
        client=_NoModel(), sandbox=sb, runner=PytestRunner(),
    )  # fmt: skip
    (item,) = outcome.unchecked
    assert item.reason is UncheckedReason.NO_ENV and "does not apply" in item.detail and outcome.repros == ()


class _NoModel:
    def complete(self, request: object) -> object:
        raise AssertionError("no model is called when the patch does not apply")


# -- run_test: the report is the run's own, read before a fresh container goes ---------------------------------------


def _report_paths(fake: FakeDocker) -> list[str]:
    return [a.split("=", 1)[1] for c in fake.command_execs() for a in c if a.startswith("--junitxml=")]


# The test modules of fixtures/runners/sympy_modules.json, recorded in the sympy__sympy-13974 image (record.py).
_SYMPY_MODULES = {
    "sympy/utilities/tests/test_receipts_a_ok.py": "def test_ok():\n    assert True\n",
    "sympy/utilities/tests/test_receipts_b_import_error.py": "import receipts_no_such_module  # noqa: F401\n\n\n"
    "def test_never_runs():\n    assert True\n",
    "sympy/utilities/tests/test_receipts_c_module_error.py": "def test_defined_before_the_error():\n"
    '    assert True\n\n\nraise ValueError("raised by the module\'s own code")\n',
    "sympy/utilities/tests/test_receipts_d_module_error_first.py": 'raise ValueError("raised before any test")\n\n\n'
    "def test_never_defined():\n    assert True\n",
}


def test_run_test_reads_which_file_defines_a_bare_sympy_name_so_its_failed_import_is_import(tmp_path: Path) -> None:
    # SWE-bench names a sympy test by its function alone, and R24 expects those names. PLANT: a bare name has no file
    # in it, so a gold whose test module stopped importing read as infra (retried, then left unchecked) instead of
    # the gold's own failure. run_test reads the run's test files, the tree's or the run's own, for the names.
    rec = json.loads((Path(__file__).parent / "fixtures" / "runners" / "sympy_modules.json").read_text())
    fake = FakeDocker(exec_result=(int(rec["exit_code"]), str(rec["stdout"]).encode(), b""))

    def checkout(_t: Target, _s: Side, _p: str | None) -> Path:
        return _committed_tree(tmp_path / f"tree-{len(list(tmp_path.iterdir()))}", _SYMPY_MODULES)

    sb = _sandbox(fake, tmp_path, checkout=checkout)
    ws = sb.prepare(_target(), Side.BASE)
    tests = list(_SYMPY_MODULES)
    got = sb.run_test(ws, tests, runner=SympyRunner(), changed_files=[], timeout_s=30, expect=["test_never_runs"])
    assert (got.outcome, got.failure_type) == (Outcome.ERROR, FailureType.IMPORT)
    assert got.problem.startswith(
        "expected case test_never_runs never ran: its test module "
        "sympy/utilities/tests/test_receipts_b_import_error.py failed to import (ModuleNotFoundError"
    )
    nowhere = sb.run_test(ws, tests, runner=SympyRunner(), changed_files=[], timeout_s=30, expect=["test_nowhere"])
    assert (nowhere.outcome, nowhere.failure_type) == (Outcome.ERROR, FailureType.INFRA)
    # The run's own copy of a file is what it ran: one that no longer defines the name leaves it unexplained.
    rewritten = {tests[1]: "import receipts_no_such_module  # noqa: F401\n"}
    own = sb.run_test(
        ws, tests, runner=SympyRunner(), changed_files=[], timeout_s=30, files=rewritten, expect=["test_never_runs"]
    )
    assert own.failure_type is FailureType.INFRA
    # A directory argument stands for the test files under it.
    under = sb.run_test(
        ws, ["sympy/utilities/tests"], runner=SympyRunner(), changed_files=[], timeout_s=30, expect=["test_never_runs"]
    )
    assert under.failure_type is FailureType.IMPORT
    # A file the sandbox cannot read explains nothing, and never fails the run: the case stays unexplained (infra).
    st = sb._state[ws.id]
    assert st.host_dir is not None
    locked = st.host_dir / tests[1]
    locked.chmod(0)
    try:
        blind = sb.run_test(ws, tests, runner=SympyRunner(), changed_files=[], timeout_s=30, expect=["test_never_runs"])
    finally:
        locked.chmod(0o644)
    assert (blind.outcome, blind.failure_type) == (Outcome.ERROR, FailureType.INFRA)


def test_each_run_writes_its_own_report_path(tmp_path: Path) -> None:
    fake = FakeDocker(reports=[JUNIT_PASS, None], exec_result=(0, b"", b""))
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    runner = PytestRunner()
    first = sb.run_test(ws, ["tests/test_pass.py"], runner=runner, changed_files=[], timeout_s=30)
    assert first.outcome is Outcome.PASS
    # PLANT (review blocker): the second run dies before writing a report. With one shared path it read the first
    # run's report and passed; with its own path there is no report, so it is infra.
    second = sb.run_test(
        ws,
        ["tests/test_pass.py"],
        runner=runner,
        changed_files=[],
        timeout_s=30,
        expect=["tests/test_pass.py::test_pass"],
    )
    assert (second.outcome, second.failure_type) == (Outcome.ERROR, FailureType.INFRA)
    paths = _report_paths(fake)
    assert len(set(paths)) == 2 and all(p.startswith("/tmp/receipts-report-") for p in paths)


def test_a_fresh_runs_report_is_read_before_its_container_is_removed(tmp_path: Path) -> None:
    fake = FakeDocker(reports=[JUNIT_FAIL], exec_result=(1, b"", b""))
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(patch=PLAIN_PATCH), Side.PATCH)
    result = sb.run_test(
        ws,
        ["tests/test_assn.py"],
        runner=PytestRunner(),
        changed_files=[],
        timeout_s=30,
        fresh=True,
        expect=["tests/test_assn.py::test_assertion"],
    )
    assert (result.outcome, result.failure_type, result.fresh_container) == (
        Outcome.FAIL,
        FailureType.ASSERTION,
        True,
    )
    fresh = next(c[c.index("--name") + 1] for c in fake.of("create")[1:])
    read = next(i for i, c in enumerate(fake.calls) if c[1] == "exec" and fresh in c and "head" in c)
    removed = next(i for i, c in enumerate(fake.calls) if c[1] == "rm" and fresh in c)
    assert read < removed  # PLANT (review blocker): the fresh container was removed first, so no report was read
    assert fake.calls[read][-1] == _report_paths(fake)[0]


def test_run_test_checks_the_exit_code_with_expect(tmp_path: Path) -> None:
    fake = FakeDocker(reports=[JUNIT_PASS], exec_result=(137, b"", b"Killed"))
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    result = sb.run_test(
        ws, ["tests/test_pass.py"], runner=PytestRunner(), changed_files=[], timeout_s=30,
        expect=["tests/test_pass.py::test_pass"],
    )  # fmt: skip
    assert (result.outcome, result.failure_type, result.exit_code) == (Outcome.ERROR, FailureType.INFRA, 137)


# pylint-dev__pylint-7080's report from R24's image, trimmed (fixtures/swebench/, see test_classify.py), and its
# PASS_TO_PASS id as swebench 4.1.0's pylint parser records that report's first case.
_PYLINT_7080 = (Path(__file__).parent / "fixtures" / "swebench" / "pylint-dev__pylint-7080.xml").read_text()
_STDIN = "tests/test_self.py::TestRunTC::test_stdin[/mymodule.py]"


@pytest.mark.parametrize(
    ("repo", "outcome", "ftype"),
    [("pylint-dev/pylint", Outcome.PASS, None), ("o/r", Outcome.ERROR, FailureType.INFRA)],
)
def test_plant_run_test_matches_in_the_repository_the_workspace_was_prepared_from(
    tmp_path: Path, repo: str, outcome: Outcome, ftype: FailureType | None
) -> None:
    """R24 left pylint-dev__pylint-7080 and 7277 unchecked: their expected test_stdin[/mymodule.py] "never ran", though
    the run reported it under its own node id. run_test matches expect with contracts.answers in the repository of the
    target the workspace was prepared from, so pylint's parser's id answers there and nowhere else."""
    fake = FakeDocker(reports=[_PYLINT_7080], exec_result=(0, b"", b""))
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(repo=repo), Side.GOLD)
    result = sb.run_test(
        ws, ["tests/test_self.py"], runner=PytestRunner(), changed_files=[], timeout_s=30, expect=[_STDIN]
    )
    assert (result.outcome, result.failure_type) == (outcome, ftype)


def test_run_cases_returns_every_case_and_binds_the_runner_to_the_root(tmp_path: Path) -> None:
    fake = FakeDocker(reports=[(Path(__file__).parent / "fixtures" / "junit" / "crash.xml").read_text()])
    fake.exec_result = (1, b"", b"")
    sb = _sandbox(fake, tmp_path, workspace_path="/workspace")
    ws = sb.prepare(_target(patch=PLAIN_PATCH), Side.PATCH)
    run = sb.run_cases(ws, ["tests"], runner=PytestRunner("/testbed"), changed_files=["src/pkg/mod.py"], timeout_s=30)
    assert isinstance(run, SuiteRun) and run.problem is None
    (case,) = run.cases
    # The report's frames are under /workspace; a runner left at /testbed would never match a changed file.
    assert (case.failure_type, case.raising_file) == (FailureType.CRASH_IN_CHANGED, "src/pkg/mod.py")


def test_an_output_runner_reads_its_stream_whole_or_not_at_all(tmp_path: Path) -> None:
    big = b"x" * (200 * 1024)  # more than the parse buffer (1 KiB here) and the tail (64 KiB) together
    fake = FakeDocker(exec_result=(0, b"", big))
    sb = _sandbox(fake, tmp_path, parse_bytes=1024)
    ws = sb.prepare(_target(), Side.BASE)
    run = sb.run_cases(ws, ["tests/x/tests.py"], runner=DjangoRunner(), changed_files=[], timeout_s=30)
    assert run.problem is not None and "too large" in run.problem
    assert run.result.failure_type is FailureType.INFRA


def test_a_test_id_that_looks_like_an_option_is_refused(tmp_path: Path) -> None:
    sb = _sandbox(FakeDocker(), tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    with pytest.raises(ValueError, match="not a test path"):
        sb.run_test(ws, ["--rootdir=/"], runner=PytestRunner(), changed_files=[], timeout_s=30)


# -- truncation and the bounded reader ------------------------------------------------------------------------------


def test_run_truncates_a_megabyte_of_output_before_it_leaves_the_sandbox(tmp_path: Path) -> None:
    big_out = b"A" + b"B" * 1_000_000
    big_err = b"\xff" * 300_000  # not even UTF-8
    fake = FakeDocker(exec_result=(1, big_out, big_err))
    sb = _sandbox(fake, tmp_path, truncate_bytes=64 * 1024)
    ws = sb.prepare(_target(patch=PLAIN_PATCH), Side.PATCH)
    result = sb.run(ws, ["python", "-c", "print()"], timeout_s=10)
    assert len(result.stdout_tail.encode("utf-8")) <= 64 * 1024 + 64
    assert result.stdout_tail.endswith("B" * 1000) and "1000001" not in result.stdout_tail
    assert "[... 934465 bytes truncated ...]" in result.stdout_tail and "A" not in result.stdout_tail
    assert len(result.stderr_tail.encode("utf-8")) <= 3 * 64 * 1024 + 64  # each U+FFFD is 3 bytes
    assert "truncated" in result.stderr_tail
    assert result.outcome is Outcome.ERROR


def test_captured_keeps_only_its_head_and_tail_and_counts_the_rest() -> None:
    cap = Captured(Keep(head=4, tail=3))
    for chunk in (b"abc", b"defgh", b"ij"):
        cap.feed(chunk)
    assert (cap.total, cap.dropped, cap.whole(), cap.text()) == (10, 6, None, None)
    assert cap.tail_text() == "[... 7 bytes truncated ...]\nhij"
    small = Captured.of(b"hi", Keep(head=4, tail=3))
    assert (small.whole(), small.tail_text(), small.dropped) == (b"hi", "hi", 0)
    head_only = Captured.of(b"x" * 10, Keep(head=4))
    assert head_only.tail_text() == "xxxx\n[... 6 bytes truncated ...]"


def test_truncate_keeps_a_bounded_tail() -> None:
    assert truncate_tail("short", 1024) == "short"
    out = truncate_tail("A" + "B" * 5000, 100)
    assert len(out.encode("utf-8")) <= 100 + 64
    assert out.endswith("B" * 100) and "truncated" in out


def test_truncated_output_is_always_valid_utf8_for_the_run_result() -> None:
    cut_mid_char = ("é" * 100).encode("utf-8")  # 2 bytes each; an odd limit cuts one in half
    lone_surrogate = "ok \udc80 tail"
    for tail in (truncate_tail(cut_mid_char, 51), truncate_tail(lone_surrogate, 1024), truncate_tail(b"\xff\xfe", 8)):
        tail.encode("utf-8")  # does not raise
        RunResult(side=Side.BASE, outcome=Outcome.PASS, fresh_container=True, duration_s=0, stdout_tail=tail)


_FLOOD = (
    "import os, sys\n"
    "chunk = os.urandom(1024 * 1024)\n"
    "for _ in range({mib}):\n"
    "    sys.stdout.buffer.write(chunk)\n"
    "    sys.stderr.buffer.write(chunk)\n"
)
# What this process's Python allocated at its peak (tracemalloc), and its peak RSS. RSS alone cannot show a leak on
# macOS: held in full, 512 MiB of fresh random bytes showed a ps RSS of 320 MiB, and 512 MiB held in the reader's own
# buffer a peak RSS under 100 MiB. tracemalloc counts every byte the interpreter allocated, wherever it ends up.
_MEASURE = (
    "import resource, sys, tracemalloc\n"
    "from receipts.attack.sandbox import Keep, _run\n"
    "flood = sys.argv[1]\n"
    "tracemalloc.start()\n"
    "rc, out, err = _run([sys.executable, '-c', flood], timeout=120, keep=Keep(head=1024 * 1024, tail=64 * 1024))\n"
    "_, traced = tracemalloc.get_traced_memory()\n"
    "peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss\n"
    "peak = peak if sys.platform == 'darwin' else peak * 1024\n"
    "print(rc, out.total, err.total, len(out.tail_text()), traced, peak)\n"
)


def test_a_flood_of_output_never_accumulates_in_this_process() -> None:
    # PLANT (review blocker): subprocess.run(capture_output=True) held all of it; 512 MiB raised peak RSS by over
    # 1 GiB. The real reader, in a child process so its peak is its own: 256 MiB per stream, of which it keeps a 1 MiB
    # head and a 64 KiB tail. PLANT (review test gap): with the head buffer unbounded the reader held all 512 MiB,
    # yet peak RSS stayed under this test's bound, so the bound on what Python allocated is the one with teeth.
    mib = 256
    proc = subprocess.run(
        [sys.executable, "-c", _MEASURE, _FLOOD.format(mib=mib)],
        capture_output=True,
        text=True,
        timeout=300,
        check=True,
        env={**os.environ, "PYTHONPATH": str(Path(sandbox_mod.__file__).parents[2])},
    )
    rc, out_total, err_total, tail_len, traced, peak = (int(x) for x in proc.stdout.split())
    assert (rc, out_total, err_total) == (0, mib * 1024**2, mib * 1024**2)
    assert tail_len <= 64 * 1024 + 64
    assert traced < 16 * 1024**2, f"Python allocated {traced / 1024**2:.0f} MiB at its peak for {2 * mib} MiB of output"
    assert peak < 150 * 1024**2, f"peak RSS {peak / 1024**2:.0f} MiB for {2 * mib} MiB of output"


def test_the_real_reader_hands_a_consumer_the_pipe_and_raises_its_error() -> None:
    seen: list[bytes] = []
    rc, _, _ = sandbox_mod._run(
        [sys.executable, "-c", "print('streamed')"], timeout=60, consume=lambda p: seen.append(p.read())
    )
    assert rc == 0 and seen == [b"streamed\n"]

    def refuse(_pipe: IO[bytes]) -> None:
        raise SandboxError("too big")

    with pytest.raises(SandboxError, match="too big"):
        sandbox_mod._run([sys.executable, "-c", "print('x' * 100000)"], timeout=60, consume=refuse)


# -- timeout labels -----------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("rc", "seconds", "outcome"),
    [
        (0, 0.1, Outcome.PASS),
        (1, 0.1, Outcome.ERROR),
        (124, 0.07, Outcome.ERROR),  # a command that exits 124 by itself is not a timeout
        (124, 10.2, Outcome.TIMEOUT),  # TERM after the timeout
        (137, 15.1, Outcome.TIMEOUT),  # it ignored TERM; KILL after -k 5
        (137, 0.5, Outcome.ERROR),  # killed early: the OOM killer, not the clock
    ],
)
def test_timeout_labels_need_the_exit_code_and_the_elapsed_time(
    rc: int, seconds: float, outcome: Outcome, tmp_path: Path
) -> None:
    fake = FakeDocker(exec_result=(rc, b"", b""), exec_seconds=seconds)
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    result = sb.run(ws, ["sleep", "100"], timeout_s=10)
    assert result.outcome is outcome
    assert (result.failure_type is not None) == (outcome is Outcome.TIMEOUT)


# -- pins, runtimes and refusals ------------------------------------------------------------------------------------


def test_image_pin_check() -> None:
    assert image_is_digest_pinned(PINNED)
    assert image_is_digest_pinned(ID_PINNED)
    assert image_is_digest_pinned("python:3.13-slim@sha256:" + "a" * 64)  # a tag plus a digest
    assert image_is_digest_pinned("localhost:5000/team/img@sha256:" + "a" * 64)
    assert not image_is_digest_pinned("python:3.13-slim")
    assert not image_is_digest_pinned("python@sha256:tooshort")
    assert not image_is_digest_pinned("python@sha256:" + "g" * 64)  # non-hex
    assert not image_is_digest_pinned("Python@sha256:" + "a" * 64)  # docker refuses upper-case names
    assert not image_is_digest_pinned("-v=/:/host@sha256:" + "a" * 64)  # a flag, not a name


@pytest.mark.parametrize("tier", [Tier.A, Tier.B, Tier.C, Tier.LIVE])
def test_runsc_is_chosen_whenever_available(tier: Tier) -> None:
    assert choose_runtime(_target(tier), PINNED, runsc_available=True) == "runsc"


@pytest.mark.parametrize("runsc", [True, False])
def test_an_unpinned_image_is_refused_for_every_runtime(runsc: bool) -> None:
    with pytest.raises(RuntimeRefused, match="not digest-pinned"):
        choose_runtime(_target(), "python:3.13-slim", runsc_available=runsc)


def test_runc_is_allowed_only_for_a_tier_a_target() -> None:
    assert choose_runtime(_target(Tier.A), PINNED, runsc_available=False) == "runc"
    assert choose_runtime(_target(Tier.A), ID_PINNED, runsc_available=False) == "runc"
    with pytest.raises(RuntimeRefused, match="App"):
        choose_runtime(_target(Tier.LIVE), PINNED, runsc_available=False)
    for tier in (Tier.B, Tier.C):
        with pytest.raises(RuntimeRefused, match="live repository"):
            choose_runtime(_target(tier), PINNED, runsc_available=False)


def test_an_unpinned_resolved_image_is_refused_before_any_docker_call(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path, image_resolver=lambda _t: "python:3.13-slim", runsc=True)
    with pytest.raises(RuntimeRefused, match=r"^resolved image .* not digest-pinned"):
        sb.prepare(_target(), Side.BASE)
    assert fake.calls == []


def test_target_image_wins_over_the_resolver(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path, image_resolver=lambda _t: PINNED)
    sb.prepare(_target(image=OTHER_PINNED), Side.BASE)
    (create,) = fake.of("create")
    assert OTHER_PINNED in create and PINNED not in create


def test_no_image_and_no_resolver_is_refused() -> None:
    fake = FakeDocker()
    sb = DockerSandbox(runner=fake, clock=fake.clock, runsc=False)
    with pytest.raises(SandboxError, match="no image"):
        sb.prepare(_target(), Side.BASE)


@pytest.mark.parametrize("runsc", [True, False])
def test_an_unpinned_base_image_is_refused_in_the_build_phase(runsc: bool) -> None:
    for tier in (Tier.A, Tier.C, Tier.LIVE):
        fake = FakeDocker()
        sb = _sandbox(fake, runsc=runsc)
        with pytest.raises(RuntimeRefused, match=r"^base image .* not digest-pinned"):
            sb.build_phase(_target(tier), [["true"]], base_image="python:3.13-slim")
        assert fake.calls == []


# -- the build phase --------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("runsc", "runtime"), [(True, "runsc"), (False, "runc")])
def test_build_phase_argv_has_the_runtime_no_network_and_no_environment(runsc: bool, runtime: str) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, runsc=runsc)
    built = sb.build_phase(_target(), [["python", "-c", "pass"]], base_image=PINNED)
    assert built == COMMITTED
    (create,) = fake.of("create")
    _flags_before_image(create, PINNED, [f"--runtime={runtime}", "--pull=never", "--network=none"])
    assert create[create.index("--network=none") + 1 : create.index("-w")] == RUN_PHASE_LIMITS.limit_flags()
    setup = [c for c in fake.of("exec") if c[-3:] == ["python", "-c", "pass"]]
    assert len(setup) == 1 and "-e" not in setup[0]
    assert fake.of("commit") and fake.of("rm")


def test_build_phase_copies_a_context_directory_to_build(tmp_path: Path) -> None:
    (tmp_path / "wheels").mkdir()
    (tmp_path / "wheels" / "pkg.whl").write_bytes(b"not really a wheel")
    fake = FakeDocker()
    sb = _sandbox(fake)
    sb.build_phase(_target(), [["true"]], base_image=PINNED, context=tmp_path)
    (argv, stdin), *_ = [(c, fake.stdins[i]) for i, c in enumerate(fake.calls) if c[1] == "exec" and "tar" in c]
    assert argv[-2:] == ["-C", "/build"]
    assert stdin is not None
    with tarfile.open(fileobj=io.BytesIO(stdin)) as tar:
        assert {m.uid for m in tar.getmembers()} == {0} and {m.uname for m in tar.getmembers()} == {"root"}
        assert "wheels/pkg.whl" in tar.getnames()


def test_tar_fallback_cannot_import_a_workspace_module_or_expose_run_files(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    workspace.mkdir()
    stolen = workspace / "stolen-tar"
    (workspace / "tarfile.py").write_text(
        "import sys\n"
        "from pathlib import Path\n"
        "Path('stolen-tar').write_bytes(sys.stdin.buffer.read())\n"
        "raise RuntimeError('workspace tarfile imported')\n"
    )
    repro_path = "test_predictable_repro.py"
    (workspace / repro_path).mkdir()
    (workspace / repro_path / "keep").write_text("nonempty directory forces tar to fail\n")

    class LocalExtractors(FakeDocker):
        def answer_exec(self, rest: list[str], stdin: bytes | None) -> tuple[int, bytes, bytes]:
            assert rest[:4] == ["-i", "-w", str(workspace), "local"]
            command = rest[4:]
            assert command[0] in ("tar", "python")
            if command[0] == "python":
                command[0] = sys.executable
            done = subprocess.run(command, cwd=workspace, input=stdin, capture_output=True, timeout=30, check=False)
            return done.returncode, done.stdout, done.stderr

    fake = LocalExtractors()
    sb = _sandbox(fake, tmp_path)
    try:
        with pytest.raises(SandboxError, match="could not inject"):
            sb._inject_bytes("local", tar_of({repro_path: "synthetic nonce\n"}), str(workspace))
        assert len(fake.of("exec")) == 2
        assert (workspace / repro_path / "keep").is_file()
        assert not stolen.exists()
    finally:
        sb.cleanup()


def test_a_committed_image_with_new_environment_is_refused_and_removed() -> None:
    leaked = [*IMAGE_ENV, "HTTP_PROXY=http://builduser:hunter2-not-real@proxy.invalid:3128"]
    fake = FakeDocker(committed_env=leaked)
    sb = _sandbox(fake)
    with pytest.raises(BuildError, match="HTTP_PROXY") as info:
        sb.build_phase(_target(), [["true"]], base_image=PINNED)
    assert "hunter2" not in str(info.value)  # names only, never values
    assert any(c[1] == "rmi" and c[-1] == COMMITTED for c in fake.calls)
    assert fake.of("rm")


def test_a_builder_whose_environment_differs_from_its_image_is_refused_before_it_starts() -> None:
    # The review's survivor: the pre-start check had no test. The docker CLI can add proxies to a builder too.
    fake = FakeDocker(container_env=[*IMAGE_ENV, "HTTPS_PROXY=http://u:pw@proxy.invalid:3128"])
    sb = _sandbox(fake)
    with pytest.raises(SandboxError, match="HTTPS_PROXY"):
        sb.build_phase(_target(), [["true"]], base_image=PINNED)
    assert not fake.of("start") and fake.of("rm")


def test_egress_network_mode_joins_the_named_network_and_sets_nothing() -> None:
    fake = FakeDocker()
    sb = _sandbox(fake)
    sb.build_phase(_target(), [["true"]], base_image=PINNED, egress=EgressPolicy.NETWORK, egress_network="allowlist")
    (create,) = fake.of("create")
    assert "--network=allowlist" in create and "--network=none" not in create
    assert "-e" not in create


@pytest.mark.parametrize("network", ["bridge", "host", "default", "none", "container:abc", "", "-x"])
def test_egress_network_mode_refuses_an_open_or_missing_network(network: str) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake)
    with pytest.raises(BuildError):
        sb.build_phase(_target(), [["true"]], base_image=PINNED, egress=EgressPolicy.NETWORK, egress_network=network)
    assert fake.of("create") == []


def test_no_egress_takes_no_network_name() -> None:
    sb = _sandbox(FakeDocker())
    with pytest.raises(BuildError):
        sb.build_phase(_target(), [["true"]], base_image=PINNED, egress=EgressPolicy.NONE, egress_network="x")


def test_a_failed_build_step_removes_the_builder() -> None:
    fake = FakeDocker(exec_result=(2, b"", b"boom"))
    sb = _sandbox(fake)
    with pytest.raises(BuildError, match="exit 2"):
        sb.build_phase(_target(), [["make"]], base_image=PINNED)
    assert fake.of("rm") and not fake.of("commit")


# -- no secret and no host environment inside -----------------------------------------------------------------------


@pytest.mark.parametrize(
    "env",
    [
        {"HTTP_PROXY": "http://proxy.invalid:3128"},
        {"https_proxy": "http://proxy.invalid:3128"},
        {"FALDEN_META_API_KEY": "planted-value-1"},
        {"AWS_SECRET_ACCESS_KEY": "planted-value-2"},
        {"GITHUB_TOKEN": "planted-value-3"},
        {"INNOCENT": "https://user:pass@example.invalid/"},
        {"INNOCENT": "ghp_" + "a" * 36},
    ],
)
def test_a_run_environment_with_a_proxy_or_credential_is_refused(env: dict[str, str], tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    with pytest.raises(SandboxError, match="refused") as info:
        sb.run(ws, ["true"], timeout_s=5, env=env)
    for value in env.values():
        assert value not in str(info.value)
    assert fake.command_execs() == []


def test_benign_environment_passes_the_check() -> None:
    benign = {"PYTHONHASHSEED": "0", "RECEIPTS_PROBE_RUN": "1", "GPG_KEY": "public", "TOKENIZERS_PARALLELISM": "0"}
    assert refused_env_names(benign) == []


def test_a_container_whose_environment_differs_from_its_image_is_refused(tmp_path: Path) -> None:
    fake = FakeDocker(container_env=[*IMAGE_ENV, "HTTPS_PROXY=http://u:secretpw@proxy.invalid:3128"])
    sb = _sandbox(fake, tmp_path)
    with pytest.raises(SandboxError, match="HTTPS_PROXY") as info:
        sb.prepare(_target(), Side.BASE)
    assert "secretpw" not in str(info.value)
    assert fake.of("rm") and not fake.of("start")


def test_an_image_that_bakes_in_a_proxy_is_refused_before_create(tmp_path: Path) -> None:
    fake = FakeDocker(image_env=[*IMAGE_ENV, "http_proxy=http://proxy.invalid:3128"])
    sb = _sandbox(fake, tmp_path)
    with pytest.raises(SandboxError, match="http_proxy"):
        sb.prepare(_target(), Side.BASE)
    assert fake.of("create") == []


# -- the history strip on the checkout path ----------------------------------------------------------------------------


def _three_commit_repo(root: Path) -> Path:
    env = {"GIT_CONFIG_GLOBAL": "/dev/null", "GIT_CONFIG_SYSTEM": "/dev/null", "PATH": os.environ["PATH"]}
    # No background maintenance from the test's own commits runs under the strip (git_history.NO_BACKGROUND).
    who = ["-c", "user.name=t", "-c", "user.email=t@localhost", "-c", "commit.gpgsign=false", "-c", "gc.auto=0"]
    who += ["-c", "maintenance.auto=false"]
    subprocess.run(["git", "-C", str(root), "init", "-q", "-b", "main"], check=True, env=env)
    for i in range(3):
        (root / "f.txt").write_text(f"{i}\n")
        subprocess.run(["git", "-C", str(root), "add", "-A"], check=True, env=env)
        subprocess.run(["git", "-C", str(root), *who, "commit", "-q", "-m", f"c{i}"], check=True, env=env)
    return root


@pytest.mark.parametrize(("strip", "commits"), [(True, 1), (False, 3)])
def test_prepare_strips_the_checkout_history(tmp_path: Path, strip: bool, commits: int) -> None:
    repo = _three_commit_repo(tmp_path)
    fake = FakeDocker()
    sb = _sandbox(fake, checkout=lambda _t, _s, _p: repo, strip_history=strip)
    sb.prepare(_target(), Side.BASE)
    assert reachable_commit_count(repo) == commits  # the control (strip off) keeps all 3


def test_prepare_cleans_up_the_container_when_injection_fails(tmp_path: Path) -> None:
    class FailingInject(FakeDocker):
        def answer_exec(self, rest: list[str], stdin: bytes | None) -> tuple[int, bytes, bytes]:
            if "tar" in rest or "python" in rest:
                return 1, b"", b"injection failed"
            return super().answer_exec(rest, stdin)

    fake = FailingInject()
    sb = _sandbox(fake, tmp_path)
    with pytest.raises(SandboxError, match="could not inject"):
        sb.prepare(_target(), Side.BASE)
    assert len(fake.of("rm")) == 1  # the created container was force-removed, not leaked
    assert sb._state == {}


# -- files in and out -------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("name", ["../escape.py", "/etc/passwd", "", ".git/config"])
def test_run_files_must_stay_inside_the_workspace(name: str, tmp_path: Path) -> None:
    sb = _sandbox(FakeDocker(), tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    with pytest.raises(SandboxError, match="relative path"):
        sb.run(ws, ["true"], timeout_s=5, files={name: "x"})


def test_copy_out_text_is_size_capped_and_relative_only(tmp_path: Path) -> None:
    fake = FakeDocker()
    sb = _sandbox(fake, tmp_path)
    ws = sb.prepare(_target(), Side.BASE)
    fake.files["/testbed/report.json"] = b"x" * 11
    assert sb.copy_out_text(ws, "report.json", max_bytes=11) == "x" * 11
    fake.files["/testbed/report.json"] = b"x" * 12
    with pytest.raises(SandboxError, match="larger than"):
        sb.copy_out_text(ws, "report.json", max_bytes=11)
    with pytest.raises(SandboxError, match="relative path"):
        sb.copy_out_text(ws, "../../etc/shadow")


def test_docker_and_runsc_detection_use_the_runner() -> None:
    assert docker_available(runner=FakeDocker())
    assert runsc_available(runner=FakeDocker(runtimes={"runc": {}, "runsc": {"path": "/usr/local/bin/runsc"}}))
    assert not runsc_available(runner=FakeDocker(runtimes={"runc": {}, "io.containerd.runc.v2": {}}))
    assert not runsc_available(runner=FakeDocker(runtimes=["runsc"]))  # not the shape docker info prints
