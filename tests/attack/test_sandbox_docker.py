"""Docker-backed tests (marker ``docker``): they skip cleanly without a daemon or without ``RECEIPTS_RUN_DOCKER=1``, and
run under the shared heavy lock. They use digest-pinned local images and runc, which is what the rules allow for a
Tier A target when gVisor is absent (as it is on this Mac). Nothing here reaches the internet: the sandbox has no
network, the one networked container joins an ``--internal`` docker network whose only other member is a listener
this module starts, and no image is pulled (a test whose image is not present skips).

The headline is the hostile-conftest test. The build phase makes a pytest-bearing image offline from this
environment's own pure-Python pytest packages; pytest then runs inside the sandbox through ``run_test``, so the
hostile ``conftest.py`` executes at collection, aims at a listener shown reachable from a networked container, and
must find no secret, resolve no name, reach nothing, see only loopback and find no credential under any home. The
same run in the networked container must fail, and a credential planted in the sandbox's own HOME must be found.

The Tier A tests run the SWE-bench images' own trees (/testbed copied out, history dropped, patched on the host) with
the Django and sympy runners: an instance's FAIL_TO_PASS test fails on its base with the test patch and passes with
the gold patch, which is the eval lane's R24 check on the run phase's real path. A Django target whose command
exports SWE-bench's locale runs its tests under it (a probe reads the encodings Python chose), and an expected failure
comes back from each image flagged as one. From round 6: an image that holds Django as its own egg (django__django-7530
and 10097, built with python setup.py install) runs the patched tree, which runtests.py names on its first line; an
image with an editable Django (11999) runs as before; and 11141's test patch runs as the harness's own labels.
"""

from __future__ import annotations

import difflib
import importlib.util
import io
import json
import re
import shutil
import subprocess
import tarfile
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

from receipts.attack.classify import SuiteRun, junit_cases, with_expect
from receipts.attack.correctness import CorrectnessAttacker
from receipts.attack.git_history import init_single_commit
from receipts.attack.regression import RegressionAttacker
from receipts.attack.runners import PYTEST_OPTIONS, DjangoRunner, PytestRunner, SympyRunner, django_label_of
from receipts.attack.sandbox import Checkout, DockerSandbox, SandboxError, image_is_digest_pinned, refused_env_names
from receipts.contracts import (
    RUN_PHASE_LIMITS,
    Arm,
    Attack,
    CaseResult,
    ContextMask,
    FailureType,
    Lens,
    ModelRequest,
    ModelResponse,
    Outcome,
    PatchMethod,
    RunnerKind,
    Side,
    Target,
    Tier,
    UncheckedReason,
    Usage,
)

pytestmark = pytest.mark.docker

_FIXTURE = Path(__file__).parent / "fixtures" / "conftest_secret"
_RUNNERS = Path(__file__).parent / "fixtures" / "runners"
_PORT = 8080
_LISTENER = (
    "import socket\n"
    "s = socket.socket()\n"
    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
    f"s.bind(('0.0.0.0', {_PORT}))\n"
    "s.listen(16)\n"
    "while True:\n"
    "    c, _ = s.accept()\n"
    "    c.close()\n"
)
# pytest and what it imports, all pure Python (py3-none-any), copied from this environment into the build context.
_PYTEST_MODULES = ("pytest", "_pytest", "py", "pluggy", "iniconfig", "packaging", "pygments")
_PYTEST_DISTS = ("pytest", "pluggy", "iniconfig", "packaging", "pygments")
_COPY_INTO_SITE = (
    "import shutil, sysconfig; shutil.copytree('/build/site', sysconfig.get_paths()['purelib'], dirs_exist_ok=True)"
)
DJANGO = (
    "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11999"
    "@sha256:6bf502e69a74a0a78a87565117d04369ec89fbbca71e5929110f65e3ee1f5bc6"
)
SYMPY = (
    "ghcr.io/epoch-research/swe-bench.eval.x86_64.sympy__sympy-13974"
    "@sha256:49c852aecff5447405c1f9413d2d90113bd5ba58fdbd3efe6ba2f4d1f882fbc4"
)


def _docker(*args: str, stdin: bytes | None = None, timeout: float = 120.0) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["docker", *args], input=stdin, capture_output=True, timeout=timeout, check=False)


def _target(**kw: object) -> Target:
    fields: dict[str, object] = {"id": "probe", "tier": Tier.A, "repo": "o/r", "base_sha": "a" * 40, "patch": "x\n"}
    return Target.model_validate({**fields, **kw})


def _no_tree(_t: Target, _s: Side, _p: str | None) -> None:
    """A checkout with no tree: the workspace mount starts empty (a plain image has no repository to copy out)."""
    return None


def _present(image: str) -> str:
    if _docker("image", "inspect", image).returncode != 0:
        pytest.skip(f"{image.split('@')[0]} is not present locally (tests never pull)")
    return image


@pytest.fixture(scope="module")
def python_image() -> str:
    out = _docker("images", "--no-trunc", "--format", "{{.Repository}}@{{.Digest}}", "python").stdout.decode()
    for line in out.splitlines():
        if image_is_digest_pinned(line.strip()):
            return line.strip()
    pytest.skip("no digest-pinned python image present locally")


@pytest.fixture(scope="module")
def pytest_image(python_image: str, tmp_path_factory: pytest.TempPathFactory) -> Iterator[str]:
    """A digest-pinned image with pytest, made by the build phase with no egress: the packages arrive as a context
    directory, and a setup step copies them into the image's site-packages."""
    context = tmp_path_factory.mktemp("pytest-context")
    site = context / "site"
    site.mkdir()
    for module in _PYTEST_MODULES:
        spec = importlib.util.find_spec(module)
        assert spec is not None and spec.origin is not None
        origin = Path(spec.origin)
        if origin.name == "__init__.py":
            shutil.copytree(origin.parent, site / origin.parent.name, ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(origin, site / origin.name)
    packages_dir = Path(importlib.util.find_spec("pytest").origin).parent.parent  # type: ignore[union-attr, arg-type]
    for dist in _PYTEST_DISTS:
        for info in packages_dir.glob(f"{dist}-*.dist-info"):
            shutil.copytree(info, site / info.name)
    builder = DockerSandbox(image_resolver=lambda _t: python_image, runsc=False, strip_history=False)
    built = builder.build_phase(
        _target(),
        [["python", "-c", _COPY_INTO_SITE], ["python", "-c", "import pytest; print(pytest.__version__)"]],
        base_image=python_image,
        context=context,
    )
    assert image_is_digest_pinned(built), built
    yield built
    _docker("rmi", "-f", built)


@pytest.fixture(scope="module")
def listener(python_image: str) -> Iterator[tuple[str, str, str]]:
    """A TCP listener on an --internal network (no route off the host). Yields (network, listener name, address)."""
    tag = uuid.uuid4().hex[:10]
    network, name = f"receipts-probe-net-{tag}", f"receipts-listener-{tag}"
    assert _docker("network", "create", "--internal", network).returncode == 0
    try:
        started = _docker(
            "run", "-d", "--name", name, "--network", network, "--cap-drop=ALL", "--pull=never", "--memory=256m",
            python_image, "python", "-c", _LISTENER,
        )  # fmt: skip
        assert started.returncode == 0, started.stderr
        ready = f"import socket; socket.create_connection(('127.0.0.1', {_PORT}), timeout=1).close()"
        for _ in range(50):
            if _docker("exec", name, "python", "-c", ready).returncode == 0:
                break
        else:
            pytest.fail("the listener never came up")
        inspect = _docker("inspect", "-f", "{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}", name)
        address = inspect.stdout.decode().strip()
        assert address, inspect.stderr
        yield network, name, address
    finally:
        _docker("rm", "-f", "-v", name)
        _docker("network", "rm", network)


def _probe_config(listener: tuple[str, str, str]) -> str:
    """The probe's settings, written next to the hostile conftest as a run file; the file is also its gate."""
    _, name, address = listener
    return json.dumps(
        {
            "PROBE_TARGETS": f"{address}:{_PORT}",
            "PROBE_DNS": name,
            "PROBE_HOMES": str(Path.home()),  # the host's own home path must not exist inside
        }
    )


_HOSTILE_CASE = ["test_probe.py::test_sandbox_leaked_nothing"]


def _as_root(info: tarfile.TarInfo) -> tarfile.TarInfo:
    info.uid = info.gid = 0  # tar as root with every capability dropped cannot chown to the host's uid
    info.uname = info.gname = "root"
    return info


def _tar_of(root: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tar:
        for path in sorted(root.rglob("*")):
            if "__pycache__" not in path.parts:
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False, filter=_as_root)
    return buf.getvalue()


def _hostile_sandbox(tmp_path: Path, image: str) -> DockerSandbox:
    def checkout(_t: Target, _s: Side, _p: str | None) -> Path:
        dst = tmp_path / f"ws-{uuid.uuid4().hex[:6]}"
        shutil.copytree(_FIXTURE, dst, ignore=shutil.ignore_patterns("__pycache__"))
        return dst

    return DockerSandbox(image_resolver=lambda _t: image, checkout=checkout, runsc=False, workspace_path="/workspace")


def _networked_run(image: str, network: str, config: str, argv: list[str]) -> tuple[str, str]:
    """Run argv in a normally networked container on ``network``, as root, with the hostile fixture and its config in
    /workspace and a credential planted in root's home; return the junit report and the probe report. This is the
    control: not the sandbox."""
    name = f"receipts-control-{uuid.uuid4().hex[:10]}"
    created = _docker(
        "create", "--name", name, "--network", network, "--cap-drop=ALL", "--security-opt=no-new-privileges",
        "--pull=never", "-w", "/workspace", image, "sleep", "600",
    )  # fmt: skip
    assert created.returncode == 0, created.stderr
    try:
        assert _docker("start", name).returncode == 0
        copied = _docker("exec", "-i", name, "tar", "-x", "-f", "-", "-C", "/workspace", stdin=_tar_of(_FIXTURE))
        assert copied.returncode == 0, copied.stderr
        wrote = _docker("exec", "-i", name, "sh", "-c", "cat > /workspace/probe_config.json", stdin=config.encode())
        assert wrote.returncode == 0, wrote.stderr
        plant = "mkdir -p /root/.aws && echo '[default]' > /root/.aws/credentials"
        assert _docker("exec", name, "sh", "-c", plant).returncode == 0
        _docker("exec", "-w", "/workspace", name, *argv)
        junit, report = (
            _docker("exec", name, "cat", f"/workspace/{rel}").stdout.decode()
            for rel in (".junit-control.xml", "probe_report.json")
        )
        return junit, report
    finally:
        _docker("rm", "-f", "-v", name)


def test_the_listener_is_reachable_from_a_networked_container(
    python_image: str, listener: tuple[str, str, str]
) -> None:
    # Proof first: the target the sandboxed probe aims at is really there for a container that has a network.
    network, name, address = listener
    check = (
        "import os, socket\n"
        f"socket.create_connection(('{address}', {_PORT}), timeout=3).close()\n"
        f"socket.getaddrinfo('{name}', None)\n"
        "print('REACHED', sorted(os.listdir('/sys/class/net')))\n"
    )
    out = _docker(
        "run", "--rm", "--network", network, "--cap-drop=ALL", "--pull=never", python_image, "python", "-c", check
    )
    assert out.returncode == 0, out.stderr
    assert "REACHED" in out.stdout.decode() and "eth0" in out.stdout.decode()


def test_hostile_conftest_runs_under_pytest_in_the_sandbox_and_finds_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, pytest_image: str, listener: tuple[str, str, str]
) -> None:
    # A fake host secret in this process's environment. If the sandbox leaked the host env, it would show inside.
    monkeypatch.setenv("FALDEN_SECRET_SENTINEL", "sk-fake-sentinel-not-a-real-key")
    sandbox = _hostile_sandbox(tmp_path, pytest_image)
    ws = sandbox.prepare(_target(), Side.BASE)
    try:
        result = sandbox.run_test(
            ws, ["test_probe.py"], runner=PytestRunner(), changed_files=[], timeout_s=120,
            files={"probe_config.json": _probe_config(listener)}, expect=_HOSTILE_CASE,
        )  # fmt: skip
        assert result.outcome is Outcome.PASS, (result.stdout_tail, result.stderr_tail)
        report = json.loads(sandbox.copy_out_text(ws, "probe_report.json"))
        assert report["ran_from"] == "conftest"  # pytest imported the hostile conftest at collection
        assert report["home"] == "/tmp/home"  # the run's HOME exists inside the /tmp mount
        assert "FALDEN_SECRET_SENTINEL" not in report["env_names"]
        targets = {c["target"]: c["connected"] for c in report["connections"]}
        assert targets[f"{listener[2]}:{_PORT}"] is False  # the listener a networked container reaches
        assert not any(targets.values()), report["connections"]
        assert report["dns"] and not any(d["resolved"] for d in report["dns"]), report["dns"]
        assert report["interfaces"] == ["lo"], report["interfaces"]
        assert report["home_secrets"] == []
    finally:
        sandbox.close(ws)


def test_a_credential_in_the_sandboxs_own_home_is_found(
    tmp_path: Path, pytest_image: str, listener: tuple[str, str, str]
) -> None:
    # PLANT for the home detector, inside the sandbox: a credentials file in the run's HOME must be reported, and the
    # hostile test must then fail on its home assertion.
    sandbox = _hostile_sandbox(tmp_path, pytest_image)
    ws = sandbox.prepare(_target(), Side.BASE)
    try:
        plant = ["sh", "-c", "mkdir -p $HOME/.aws && echo '[default]' > $HOME/.aws/credentials"]
        assert sandbox.run(ws, plant, timeout_s=30).outcome is Outcome.PASS
        result = sandbox.run_test(
            ws, ["test_probe.py"], runner=PytestRunner(), changed_files=[], timeout_s=120,
            files={"probe_config.json": _probe_config(listener)}, expect=_HOSTILE_CASE,
        )  # fmt: skip
        assert (result.outcome, result.failure_type) == (Outcome.FAIL, FailureType.ASSERTION)
        report = json.loads(sandbox.copy_out_text(ws, "probe_report.json"))
        assert report["home_secrets"] == ["/tmp/home/.aws/credentials"]
    finally:
        sandbox.close(ws)


def test_the_same_hostile_run_with_a_network_is_caught(pytest_image: str, listener: tuple[str, str, str]) -> None:
    # PLANT: the identical pytest run, in a container that has a network and runs as root with a credential in its
    # home. The hostile test must fail on it, so the sandboxed pass means the isolation held, not a blind probe.
    argv = ["python", "-m", "pytest", *PYTEST_OPTIONS, "--junitxml=/workspace/.junit-control.xml", "test_probe.py"]
    xml, raw = _networked_run(pytest_image, listener[0], _probe_config(listener), argv)
    report = json.loads(raw)
    assert report["ran_from"] == "conftest"
    assert any(c["connected"] for c in report["connections"]), report["connections"]
    assert any(d["resolved"] for d in report["dns"]), report["dns"]
    assert "eth0" in report["interfaces"]
    assert "/root/.aws/credentials" in report["home_secrets"]  # the detector fires on a real home
    (case,) = junit_cases(xml, root="/workspace", changed_files=[])
    assert (case.outcome, case.failure_type) == (Outcome.FAIL, FailureType.ASSERTION)


# -- run_test: each run's own report, read before a fresh container goes --------------------------------------------

_CASES = {
    "tests/test_ok.py": "def test_ok():\n    assert True\n",
    "tests/test_bad.py": "def test_bad():\n    assert 1 == 2\n",
    "tests/test_dies.py": "import os\n\n\ndef test_dies():\n    os._exit(137)\n",
}


def test_run_test_classifies_each_run_from_its_own_report(tmp_path: Path, pytest_image: str) -> None:
    def checkout(_t: Target, _s: Side, _p: str | None) -> Path:
        dst = tmp_path / f"ws-{uuid.uuid4().hex[:6]}"
        for rel, content in _CASES.items():
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            (dst / rel).write_text(content)
        return dst

    prefix = f"receipts-rt-{uuid.uuid4().hex[:6]}"
    sandbox = DockerSandbox(
        image_resolver=lambda _t: pytest_image, checkout=checkout, runsc=False, container_prefix=prefix
    )
    ws = sandbox.prepare(_target(), Side.BASE)
    runner = PytestRunner()
    try:
        bad = sandbox.run_test(
            ws, ["tests/test_bad.py"], runner=runner, changed_files=[], timeout_s=60, fresh=True,
            expect=["tests/test_bad.py::test_bad"],
        )  # fmt: skip
        assert (bad.outcome, bad.failure_type, bad.fresh_container) == (Outcome.FAIL, FailureType.ASSERTION, True)
        ok = sandbox.run_test(
            ws, ["tests/test_ok.py"], runner=runner, changed_files=[], timeout_s=60, fresh=True,
            expect=["tests/test_ok.py::test_ok"],
        )  # fmt: skip
        assert ok.outcome is Outcome.PASS
        # PLANT (review blockers 1 and 2): a run that dies before pytest writes its report. With one shared path it
        # read an earlier run's report; with its own path, and the exit code checked, it is infra.
        for fresh in (False, True):
            dead = sandbox.run_test(
                ws, ["tests/test_dies.py"], runner=runner, changed_files=[], timeout_s=60, fresh=fresh,
                expect=["tests/test_dies.py::test_dies"],
            )  # fmt: skip
            assert (dead.outcome, dead.failure_type, dead.exit_code) == (Outcome.ERROR, FailureType.INFRA, 137)
    finally:
        sandbox.close(ws)
    # Every container this sandbox made, fresh ones included, is gone.
    assert _docker("ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}").stdout.strip() == b""


# -- Tier A: the SWE-bench images' own trees, with the Django and sympy runners -------------------------------------


def _instance(image: str, runner: RunnerKind, command: str) -> Target:
    return _target(id="r2~instance", image=image, test_runner=runner, test_command=command)


def _diff(name: str) -> str:
    return (_RUNNERS / name).read_text()


def test_sympy_f2p_fails_on_base_and_passes_with_gold_in_the_image(tmp_path: Path) -> None:
    image = _present(SYMPY)
    target = _instance(
        image, RunnerKind.SYMPY, "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
    )
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    tests = ["sympy/physics/quantum/tests/test_tensorproduct.py"]
    changed = ["sympy/physics/quantum/tensorproduct.py"]
    base = sandbox.prepare(target, Side.BASE, patch=_diff("sympy_13974_test.diff"))
    gold = sandbox.prepare(target, Side.GOLD, patch=_diff("sympy_13974_gold.diff") + _diff("sympy_13974_test.diff"))
    try:
        assert base.root == "/testbed"
        assert (base.patch_method, gold.patch_method) == (PatchMethod.GIT_APPLY, PatchMethod.GIT_APPLY)
        before = sandbox.run_test(
            base, tests, runner=SympyRunner(), changed_files=changed, timeout_s=600, expect=["test_tensor_product_simp"]
        )
        assert (before.outcome, before.failure_type) == (Outcome.FAIL, FailureType.ASSERTION), before.stdout_tail
        after = sandbox.run_test(
            gold, tests, runner=SympyRunner(), changed_files=changed, timeout_s=600, fresh=True,
            expect=["test_tensor_product_simp", "test_tensor_product_abstract", "test_issue_5923"],
        )  # fmt: skip
        assert after.outcome is Outcome.PASS, after.stdout_tail
        history = sandbox.run(base, ["git", "rev-list", "--all", "--count"], timeout_s=60)
        assert history.stdout_tail.strip() == "1"  # the image's own history is unreachable
    finally:
        sandbox.close(base)
        sandbox.close(gold)


def test_django_f2p_fails_on_base_and_passes_with_gold_in_the_image(tmp_path: Path) -> None:
    image = _present(DJANGO)
    target = _instance(
        image, RunnerKind.DJANGO, "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
    )
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    tests = ["tests/model_fields/tests.py"]
    changed = ["django/db/models/fields/__init__.py"]
    f2p = ["model_fields.tests.GetFieldDisplayTests.test_overriding_FIELD_display"]
    base = sandbox.prepare(target, Side.BASE, patch=_diff("django_11999_test.diff"))
    gold = sandbox.prepare(target, Side.GOLD, patch=_diff("django_11999_gold.diff") + _diff("django_11999_test.diff"))
    try:
        before = sandbox.run_test(base, tests, runner=DjangoRunner(), changed_files=changed, timeout_s=900, expect=f2p)
        assert (before.outcome, before.failure_type) == (Outcome.FAIL, FailureType.ASSERTION), before.stderr_tail
        after = sandbox.run_test(
            gold, tests, runner=DjangoRunner(), changed_files=changed, timeout_s=900, fresh=True, expect=f2p
        )
        assert after.outcome is Outcome.PASS, after.stderr_tail
    finally:
        sandbox.close(base)
        sandbox.close(gold)


# -- the run phase itself ---------------------------------------------------------------------------------------------

_WRITABLE_PROBE = """
import ctypes, errno, json, os, uuid
points = set()
with open('/proc/self/mountinfo') as fh:
    for line in fh:
        points.add(line.split()[4])
points |= {'/', '/dev', '/dev/shm', '/dev/mqueue', '/tmp', '/workspace', '/run', '/var/tmp', '/root', '/etc', '/home',
           '/usr', '/opt', '/srv', '/mnt', '/var', '/sys/fs/cgroup'}
writable = {}
for p in sorted(points):
    if not os.path.isdir(p):
        continue
    probe = os.path.join(p, '.w' + uuid.uuid4().hex[:6])
    try:
        open(probe, 'w').close()
    except OSError:
        continue
    os.unlink(probe)
    st = os.statvfs(p)
    writable[p] = round(st.f_blocks * st.f_frsize / 2**20)
libc = ctypes.CDLL(None, use_errno=True)
mq = libc.mq_open(b'/receipts_q', os.O_CREAT | os.O_RDWR, 0o600, None)
print(json.dumps({'writable': writable, 'mq_open': mq, 'mq_errno': errno.errorcode.get(ctypes.get_errno(), ''),
                  'uid': os.getuid(), 'net': sorted(os.listdir('/sys/class/net')), 'home': os.environ.get('HOME')}))
"""


def test_the_run_phase_writes_only_to_its_three_capped_tmpfs_mounts(python_image: str) -> None:
    sandbox = DockerSandbox(
        image_resolver=lambda _t: python_image, runsc=False, workspace_path="/workspace", checkout=_no_tree
    )
    ws = sandbox.prepare(_target(tier=Tier.A), Side.BASE)
    try:
        result = sandbox.run(ws, ["python", "-c", _WRITABLE_PROBE], timeout_s=30)
        assert result.outcome is Outcome.PASS, (result.exit_code, result.stderr_tail)
        data = json.loads(result.stdout_tail.splitlines()[-1])
        shm_mib = int(RUN_PHASE_LIMITS.shm_size.rstrip("m"))
        assert data["writable"] == {"/workspace": 1024, "/tmp": 256, "/dev/shm": shm_mib}, data["writable"]
        assert data["mq_open"] == -1, "a POSIX message queue was created"
        assert data["uid"] == 65534 and data["net"] == ["lo"] and data["home"] == "/tmp/home"
    finally:
        sandbox.close(ws)


def test_a_flood_of_output_comes_back_as_a_bounded_tail(python_image: str) -> None:
    sandbox = DockerSandbox(
        image_resolver=lambda _t: python_image, runsc=False, workspace_path="/workspace", checkout=_no_tree
    )
    ws = sandbox.prepare(_target(), Side.BASE)
    try:
        flood = "import sys; sys.stdout.write('A' + 'B' * 3000000); sys.stderr.write('E' * 3000000)"
        result = sandbox.run(ws, ["python", "-c", flood], timeout_s=60)
        assert result.outcome is Outcome.PASS
        assert len(result.stdout_tail.encode()) <= 64 * 1024 + 64 and "truncated" in result.stdout_tail
        assert result.stdout_tail.endswith("B" * 1000) and "A" not in result.stdout_tail
        assert len(result.stderr_tail.encode()) <= 64 * 1024 + 64
    finally:
        sandbox.close(ws)


def test_build_phase_commits_an_image_with_its_base_environment_only(python_image: str) -> None:
    builder = DockerSandbox(image_resolver=lambda _t: python_image, runsc=False, strip_history=False)
    built = builder.build_phase(
        _target(), [["python", "-c", "import pathlib; pathlib.Path('/build/marker').write_text('ok')"]]
    )
    try:
        assert image_is_digest_pinned(built), built
        env_of = {
            ref: json.loads(_docker("image", "inspect", "--format", "{{json .Config.Env}}", ref).stdout)
            for ref in (python_image, built)
        }
        assert env_of[built] == env_of[python_image]
        assert refused_env_names(dict(e.partition("=")[::2] for e in env_of[built])) == []
        run_sb = DockerSandbox(
            image_resolver=lambda _t: built, runsc=False, workspace_path="/workspace", checkout=_no_tree
        )
        ws = run_sb.prepare(_target(), Side.BASE)
        try:
            marker = run_sb.run(ws, ["cat", "/build/marker"], timeout_s=20)
            assert marker.outcome is Outcome.PASS and "ok" in marker.stdout_tail
            env = run_sb.run(ws, ["env"], timeout_s=20)
            names = {line.partition("=")[0] for line in env.stdout_tail.splitlines()}
            assert not {n for n in names if "proxy" in n.lower()}, names
        finally:
            run_sb.close(ws)
    finally:
        _docker("rmi", "-f", built)


# -- AT3 end to end on a SWE-bench image ----------------------------------------------------------------------------


class _NotIntended:
    """A model client that marks every candidate unintended (no model is called)."""

    def __init__(self) -> None:
        self.prompts: list[str] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.prompts.append(request.messages[-1].content)
        lines = request.messages[-1].content.splitlines()
        tests = [line.removeprefix("Test: ") for line in lines if line.startswith("Test: ")]
        verdicts = [{"test": t, "intended": False, "reason": "not asked for"} for t in tests]
        return ModelResponse(
            model=request.model, text=json.dumps({"verdicts": verdicts}), finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )  # fmt: skip


def _replace(text: str, old: str, new: str) -> str:
    assert old in text, old
    return text.replace(old, new, 1)


def _diff_of(path: str, before: str, after: str) -> str:
    lines = difflib.unified_diff(
        before.splitlines(keepends=True), after.splitlines(keepends=True), f"a/{path}", f"b/{path}"
    )
    return f"diff --git a/{path} b/{path}\n" + "".join(lines)


def test_at3_finds_the_regression_an_agent_hid_by_editing_the_test(tmp_path: Path) -> None:
    # The agent changes what django.utils.text.get_text_list returns for an empty list and edits the existing test to
    # expect the new value, so the test passes on its patch. AT3 selects that test (rule 1: a changed existing test),
    # runs it on base and, with the base version written back, on the patch, and returns the regression as a repro.
    image = _present(DJANGO)
    plain = _instance(image, RunnerKind.DJANGO, "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1")
    probe = DockerSandbox(runsc=False, work_dir=tmp_path)
    ws = probe.prepare(plain, Side.BASE)
    source, test = probe.read_text(ws, "django/utils/text.py"), probe.read_text(ws, "tests/utils_tests/test_text.py")
    probe.cleanup()
    broken = _replace(source, "    if not list_:\n        return ''\n", "    if not list_:\n        return 'none'\n")
    hidden = _replace(
        test, "self.assertEqual(text.get_text_list([]), '')", "self.assertEqual(text.get_text_list([]), 'none')"
    )
    patch = _diff_of("django/utils/text.py", source, broken) + _diff_of("tests/utils_tests/test_text.py", test, hidden)
    target = plain.model_copy(update={"patch": patch})
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    attack = Attack(
        id="r2-at3", claim_id="nothing_else_changed", lens=Lens.REGRESSION, hypothesis="the change breaks other output",
        mask=ContextMask.DIFF, temperature=0.0,
    )  # fmt: skip
    client = _NotIntended()
    try:
        outcome = RegressionAttacker(max_test_files=1).attack(
            target, attack, arm=Arm(name="spark"), client=client, sandbox=sandbox, runner=DjangoRunner()
        )
    finally:
        sandbox.cleanup()
    assert all(u.reason is UncheckedReason.BUDGET for u in outcome.unchecked), outcome  # other files over text.py
    (repro,) = outcome.repros
    assert repro.command == ("utils_tests.test_text.TestUtilsText.test_get_text_list",), outcome
    assert repro.files == {"tests/utils_tests/test_text.py": test}  # the test as it was before the agent's edit
    assert "AssertionError" in client.prompts[0]


# -- sympy__sympy-13974, the pilot instance: the patch chain, AT2's two-test candidates, AT3's suite runs --------------

_SYMPY_COMMAND = "PYTHONWARNINGS='ignore::UserWarning,ignore::SyntaxWarning' bin/test -C --verbose"
_TENSOR = "sympy/physics/quantum/tensorproduct.py"


def _sympy_source(tmp_path: Path, path: str) -> str:
    probe = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        ws = probe.prepare(_instance(_present(SYMPY), RunnerKind.SYMPY, _SYMPY_COMMAND), Side.BASE)
        return probe.read_text(ws, path)
    finally:
        probe.cleanup()


def _fuzz_only(source: str, line: str, new: str) -> str:
    """A hunk that changes ``line`` with three lines of context on each side, none of which is in the file: no
    git apply way takes it, and GNU patch takes it with fuzz 3."""
    at = source.splitlines().index(line) + 1
    before = "".join(f" # receipts: not in the file {i}\n" for i in (1, 2, 3))
    after = "".join(f" # receipts: not in the file {i}\n" for i in (4, 5, 6))
    return (
        f"diff --git a/{_TENSOR} b/{_TENSOR}\n--- a/{_TENSOR}\n+++ b/{_TENSOR}\n@@ -{at - 3},7 +{at - 3},7 @@\n"
        f"{before}-{line}\n+{new}\n{after}"
    )


def test_a_patch_only_gnu_patch_applies_is_applied_with_fuzz_in_the_image(tmp_path: Path) -> None:
    # REG-569: the SWE-bench harness's last fallback, GNU patch --batch --fuzz=5 -p1, run in a run-phase container of
    # the image itself (GNU patch 2.7.6 there). A patch no way applies is refused, and an attacker says NO_ENV.
    source = _sympy_source(tmp_path, _TENSOR)
    fuzz = _fuzz_only(source, "class TensorProduct(Expr):", "class TensorProduct(Expr):  # receipts fuzz")
    target = _instance(_present(SYMPY), RunnerKind.SYMPY, _SYMPY_COMMAND).model_copy(update={"patch": fuzz})
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path, container_prefix=f"receipts-fz-{uuid.uuid4().hex[:6]}")
    try:
        ws = sandbox.prepare(target, Side.PATCH)
        assert ws.patch_method is PatchMethod.GNU_PATCH_FUZZ
        assert "class TensorProduct(Expr):  # receipts fuzz\n" in sandbox.read_text(ws, _TENSOR)
        seen = sandbox.run(ws, ["grep", "-c", "receipts fuzz", _TENSOR], timeout_s=60)
        assert seen.stdout_tail.strip() == "1"  # the workspace the tests run in holds the patched file
        history = sandbox.run(ws, ["git", "rev-list", "--all", "--count"], timeout_s=60)
        assert history.stdout_tail.strip() == "1"
        assert sandbox.prepare(target, Side.BASE).patch_method is None
        none = fuzz.replace("-class TensorProduct(Expr):", "-class NoSuchClass(Expr):")
        with pytest.raises(SandboxError, match=r"does not apply: .*patch --batch --fuzz=5 -p1 -i exited 1"):
            sandbox.prepare(target.model_copy(update={"patch": none}), Side.PATCH)
    finally:
        sandbox.cleanup()
    prefix = sandbox._prefix  # every container this sandbox made, the patch containers included, is gone
    assert _docker("ps", "-a", "--filter", f"name={prefix}", "--format", "{{.Names}}").stdout.strip() == b""
    attacker_box = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        outcome = CorrectnessAttacker().attack(
            target.model_copy(update={"patch": none}),
            Attack(id="r3-at2-none", claim_id="fixed_x", lens=Lens.CORRECTNESS, hypothesis="h", mask=ContextMask.DIFF,
                   temperature=0.0),
            arm=Arm(name="spark"), client=_NotIntended(), sandbox=attacker_box, runner=SympyRunner(),
        )  # fmt: skip
    finally:
        attacker_box.cleanup()
    (item,) = outcome.unchecked
    assert item.reason is UncheckedReason.NO_ENV and "does not apply" in item.detail


_TWO = "def test_two_a():\n    assert 1 == 2\n\n\ndef test_two_b():\n    assert 1 == 3\n"
_ONE = "def test_one_c():\n    assert 1 == 2\n"


class _TwoCandidates:
    """A model client that proposes two sympy candidates: a file with two failing tests and one with one."""

    def complete(self, request: ModelRequest) -> ModelResponse:
        tests = [
            {"path": "sympy/physics/quantum/tests/test_two.py", "code": _TWO, "why": "two functions"},
            {"path": "sympy/physics/quantum/tests/test_one.py", "code": _ONE, "why": "one function"},
        ]
        return ModelResponse(
            model=request.model, text=json.dumps({"tests": tests}), finish_reason="stop",
            usage=Usage(input_tokens=1, output_tokens=1),
        )  # fmt: skip


def test_at2_keeps_a_sympy_candidate_with_two_failing_tests_in_the_image(tmp_path: Path) -> None:
    # PLANT (review blockers 1 and 2, the reviewer's real run): with one -k per name only test_two_b ran, run_test was
    # infra, and the two-test candidate vanished with no record: AT2 kept 1 of 2.
    target = _instance(_present(SYMPY), RunnerKind.SYMPY, _SYMPY_COMMAND).model_copy(
        update={"patch": _diff("sympy_13974_gold.diff")}
    )
    attack = Attack(
        id="r3-at2", claim_id="fixed_x", lens=Lens.CORRECTNESS, hypothesis="h", mask=ContextMask.DIFF, temperature=0.0
    )
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        outcome = CorrectnessAttacker(run_timeout_s=600).attack(
            target, attack, arm=Arm(name="spark"), client=_TwoCandidates(), sandbox=sandbox, runner=SympyRunner()
        )
    finally:
        sandbox.cleanup()
    assert outcome.unchecked == (), outcome.unchecked
    assert sorted(r.command for r in outcome.repros) == [
        ("sympy/physics/quantum/tests/test_receipts_r3_at2_0.py::test_two_a",
         "sympy/physics/quantum/tests/test_receipts_r3_at2_0.py::test_two_b"),
        ("sympy/physics/quantum/tests/test_receipts_r3_at2_1.py::test_one_c",),
    ]  # fmt: skip


def test_at3_reads_the_13974_suite_on_base_and_patch_and_finds_the_regression(tmp_path: Path) -> None:
    # PLANT (review blockers 3 and 4): AT3's base run over the 7 files it selects here was unreadable, so AT3 could
    # only return unchecked for this pilot target. The agent's change stops tensor_product_simp simplifying products.
    source = _sympy_source(tmp_path, _TENSOR)
    old = "    elif isinstance(e, Mul):\n        return tensor_product_simp_Mul(e)\n"
    broken = _replace(source, old, "    elif isinstance(e, Mul):\n        return e\n")
    target = _instance(_present(SYMPY), RunnerKind.SYMPY, _SYMPY_COMMAND).model_copy(
        update={"patch": _diff_of(_TENSOR, source, broken)}
    )
    attack = Attack(
        id="r3-at3", claim_id="nothing_else_changed", lens=Lens.REGRESSION, hypothesis="the change breaks simp",
        mask=ContextMask.DIFF, temperature=0.0,
    )  # fmt: skip
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        outcome = RegressionAttacker().attack(
            target, attack, arm=Arm(name="spark"), client=_NotIntended(), sandbox=sandbox, runner=SympyRunner()
        )
    finally:
        sandbox.cleanup()
    assert not [u for u in outcome.unchecked if u.reason is not UncheckedReason.BUDGET], outcome.unchecked
    commands = {r.command for r in outcome.repros}
    assert ("sympy/physics/quantum/tests/test_tensorproduct.py::test_tensor_product_simp",) in commands, outcome
    for repro in outcome.repros:  # each repro is the file its command names, at its base content
        (path,) = repro.files
        assert repro.command == (f"{path}::{repro.command[0].rsplit('::', 1)[1]}",)


# -- round 4: the harness's own chain, command for command, against the run phase's ------------------------------------

_HARNESS_CMDS = ("git apply --verbose", "git apply --verbose --reject", "patch --batch --fuzz=5 -p1 -i")
_CHAIN = json.loads((_RUNNERS / "harness_chain_13974.json").read_text())


def _files_in(name: str, top: str) -> dict[str, str]:
    out = _docker("exec", name, "tar", "-c", "-f", "-", "-C", "/testbed", top)
    assert out.returncode == 0, out.stderr
    files: dict[str, str] = {}
    with tarfile.open(fileobj=io.BytesIO(out.stdout)) as tar:
        for member in tar:
            handle = tar.extractfile(member) if member.isfile() else None
            if handle is not None:
                files[member.name[len(top) + 1 :]] = handle.read().decode("utf-8", "replace")
    return dict(sorted(files.items()))


def _harness_applies(image: str, top: str, files: dict[str, str], diff: str) -> tuple[str | None, dict[str, str]]:
    """The SWE-bench harness's chain run as it runs it: as root, in the image's own /testbed, each command on the
    tree the one before left, until one exits 0. A control container with no network, not the sandbox."""
    name = f"receipts-harness-{uuid.uuid4().hex[:10]}"
    made = _docker("create", "--pull=never", "--network=none", "--name", name, "-w", "/testbed", image, "sleep", "600")
    assert made.returncode == 0, made.stderr
    try:
        assert _docker("start", name).returncode == 0
        planted = {f"{top}/{rel}": text for rel, text in files.items()}
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel, text in planted.items():
                data = text.encode()
                info = tarfile.TarInfo(rel)
                info.size, info.mode = len(data), 0o644
                tar.addfile(info, io.BytesIO(data))
        assert (
            _docker("exec", "-i", name, "tar", "-x", "-f", "-", "-C", "/testbed", stdin=buf.getvalue()).returncode == 0
        )
        assert _docker("exec", "-i", name, "sh", "-c", "cat > /tmp/p.diff", stdin=diff.encode()).returncode == 0
        winner = None
        for cmd in _HARNESS_CMDS:
            if _docker("exec", "-w", "/testbed", name, "sh", "-c", f"{cmd} /tmp/p.diff").returncode == 0:
                winner = cmd
                break
        return winner, _files_in(name, top)
    finally:
        _docker("rm", "-f", "-v", name)


def _checkout_of(root: Path, top: str, files: dict[str, str]) -> Checkout:
    """A checkout that hands over a fresh copy of ``files`` under ``top`` each time, as a one-commit tree."""

    def checkout(_t: Target, _s: Side, _p: str | None) -> Path:
        dst = root / uuid.uuid4().hex[:8]
        for rel, text in files.items():
            (dst / top / rel).parent.mkdir(parents=True, exist_ok=True)
            (dst / top / rel).write_text(text)
        init_single_commit(dst)  # as a Tier A tree is: the image's files as one commit
        return dst

    return checkout


def test_every_patch_lands_as_the_harnesss_own_chain_lands_it_in_the_image(tmp_path: Path) -> None:
    # REG-569 conformance, live: for each of the seven recorded patches (exact, fuzz-only, a partial --reject then
    # GNU patch, a hunk that looks already applied, an exact hunk then a fuzz-only one in one file, one nothing
    # applies, one without its last newline), the run phase's chain (git on the host, GNU patch in a run-phase
    # container of the image) names the command the harness's chain stopped at and leaves the harness's tree, .rej and
    # .orig files included. PLANT: the round-3 chain (exact, --recount, -C1, then GNU patch on the clean tree)
    # differs on reject_then_gnu (a.py kept its change; the harness's GNU patch reverses it), exact_then_fuzz (it
    # applied; the harness's chain refuses it) and no_newline (git_apply, with a newline the harness never adds).
    image = _present(SYMPY)
    method_of = {cmd: method for cmd, method in zip(_HARNESS_CMDS, PatchMethod, strict=True)}
    for case, rec in sorted(_CHAIN["cases"].items()):
        top = f"receipts_chain/{case}"
        winner, harness_tree = _harness_applies(image, top, rec["files"], rec["diff"])
        assert (winner, harness_tree) == (rec["applied_by"], rec["steps"][-1]["tree"]), case  # as recorded

        checkout = _checkout_of(tmp_path / case, top, rec["files"])
        sandbox = DockerSandbox(runsc=False, work_dir=tmp_path, checkout=checkout, image_resolver=lambda _t: image)
        target = _instance(image, RunnerKind.SYMPY, _SYMPY_COMMAND).model_copy(update={"patch": rec["diff"]})
        try:
            if winner is None:
                with pytest.raises(SandboxError, match="does not apply"):
                    sandbox.prepare(target, Side.PATCH)
                continue
            ws = sandbox.prepare(target, Side.PATCH)
            assert ws.patch_method is method_of[winner], case
            ours = {
                r[len(top) + 1 :]: sandbox.read_text(ws, r) for r in sandbox.tree_files(ws) if r.startswith(top + "/")
            }
            assert ours == harness_tree, case
            inside = sandbox.run(ws, ["sh", "-c", f"cat {top}/a.py"], timeout_s=60)
            assert inside.stdout_tail == harness_tree["a.py"], case  # what the run phase's tests see
        finally:
            sandbox.cleanup()


def test_gnu_patch_reverses_a_hunk_the_sympy_tree_already_has_as_the_harness_does(tmp_path: Path) -> None:
    # The round-3 review's case, kept and pinned (REG-569: the harness's own behavior): a hunk that adds density.py's
    # existing import line looks already applied, so GNU patch --batch applies it in reverse and the line goes.
    image = _present(SYMPY)
    path = "sympy/physics/quantum/density.py"
    source = _sympy_source(tmp_path, path)
    line = "from sympy.physics.quantum.tensorproduct import TensorProduct, tensor_product_simp"
    lines = source.splitlines()
    at = lines.index(line)
    before, after = lines[at - 3 : at], lines[at + 1 : at + 4]
    hunk = f"@@ -{at - 2},6 +{at - 2},7 @@\n" + "".join(f" {c}\n" for c in before) + f"+{line}\n"
    diff = f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n" + hunk + "".join(f" {c}\n" for c in after)
    target = _instance(image, RunnerKind.SYMPY, _SYMPY_COMMAND).model_copy(update={"patch": diff})
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        ws = sandbox.prepare(target, Side.PATCH)
        assert ws.patch_method is PatchMethod.GNU_PATCH_FUZZ
        assert source.count(line) == 1 and sandbox.read_text(ws, path).count(line) == 0
        assert f"{path}.rej" in sandbox.tree_files(ws) and f"{path}.orig" in sandbox.tree_files(ws)
    finally:
        sandbox.cleanup()


def test_a_case_whose_test_module_stopped_importing_is_an_import_error_in_each_image(tmp_path: Path) -> None:
    # Contract v3, through run_test in the images: an expected case that never ran because its module failed to
    # import is ERROR typed IMPORT, with the reason on the result; one nothing explains is INFRA.
    sympy = _instance(_present(SYMPY), RunnerKind.SYMPY, _SYMPY_COMMAND)
    mod = "sympy/physics/quantum/tests/test_receipts_r4.py"
    files = {mod: "import receipts_no_such_module  # noqa: F401\n\n\ndef test_r4():\n    assert True\n"}
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        ws = sandbox.prepare(sympy, Side.BASE)
        for expect, ftype in (([f"{mod}::test_r4"], FailureType.IMPORT), (["test_r4"], FailureType.IMPORT),
                              (["test_r4_nowhere"], FailureType.INFRA)):  # fmt: skip
            got = sandbox.run_test(
                ws, [mod], runner=SympyRunner(), changed_files=[], timeout_s=600, files=files, expect=expect
            )
            assert (got.outcome, got.failure_type) == (Outcome.ERROR, ftype), (expect, got.problem, got.stdout_tail)
            assert got.problem.startswith(f"expected case {expect[0]} never ran")
    finally:
        sandbox.cleanup()
    django = _instance(_present(DJANGO), RunnerKind.DJANGO, "./tests/runtests.py --verbosity 2 --settings=test_sqlite")
    app = {
        "tests/receipts_r4/__init__.py": "",
        "tests/receipts_r4/tests.py": "import receipts_no_such_module  # noqa: F401\nfrom django.test import "
        "SimpleTestCase\n\n\nclass T(SimpleTestCase):\n    def test_x(self):\n        pass\n",
    }
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        ws = sandbox.prepare(django, Side.BASE)
        got = sandbox.run_test(
            ws, ["receipts_r4.tests"], runner=DjangoRunner(), changed_files=[], timeout_s=600, files=app,
            expect=["receipts_r4.tests.T.test_x"],
        )  # fmt: skip
        assert (got.outcome, got.failure_type) == (Outcome.ERROR, FailureType.IMPORT), (got.problem, got.stderr_tail)
        assert "its test module receipts_r4.tests failed to import (ModuleNotFoundError" in got.problem
    finally:
        sandbox.cleanup()


# -- round 5: a Django target's exported locale reaches its tests; an expected failure says so ----------------------

# SWE-bench 4.1.0's eval_commands exports as leading assignments of the target's command: Django 3.0 to 3.2's three
# (11999 is a 3.1 instance; the root-only locale-gen line before them is not needed, since SWE-bench's base image
# installs locales-all) and Django 1.11 and 2.2's four.
_DJANGO_COMMAND = "./tests/runtests.py --verbosity 2 --settings=test_sqlite --parallel 1"
_DJANGO_31_EXPORTS = "LANG=en_US.UTF-8 LANGUAGE=en_US:en LC_ALL=en_US.UTF-8"
_DJANGO_22_EXPORTS = "LANG=en_US.UTF-8 LC_ALL=en_US.UTF-8 PYTHONIOENCODING=utf8 LANGUAGE=en_US:en"
_LOCALE_PROBE = """import locale
import sys

from django.test import SimpleTestCase


class LocaleProbe(SimpleTestCase):
    def test_sees_a_utf8_locale(self):
        seen = (locale.getpreferredencoding(), sys.getfilesystemencoding(), sys.stdout.encoding)
        sys.stderr.write("receipts-locale %r\\n" % (seen,))
        self.assertEqual(seen[:2], ("UTF-8", "utf-8"))
"""


def _seen(tail: str) -> str:
    """The probe's own line. It lands after unittest's 'test_... ... ', which the runner prints with no newline."""
    found = re.search(r"receipts-locale \(.*\)", tail)
    return found.group(0) if found is not None else "no probe line"


def test_a_django_target_runs_under_the_locale_its_command_exports_in_the_image(tmp_path: Path) -> None:
    image = _present(DJANGO)
    app = {"tests/receipts_locale/__init__.py": "", "tests/receipts_locale/tests.py": _LOCALE_PROBE}
    expect = ["receipts_locale.tests.LocaleProbe.test_sees_a_utf8_locale"]
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    seen: dict[str, tuple[Outcome, FailureType | None, str]] = {}
    try:
        for name, command in (
            ("3.1", f"{_DJANGO_31_EXPORTS} {_DJANGO_COMMAND}"),
            ("2.2", f"{_DJANGO_22_EXPORTS} {_DJANGO_COMMAND}"),
            ("none", _DJANGO_COMMAND),
        ):
            ws = sandbox.prepare(_instance(image, RunnerKind.DJANGO, command), Side.BASE)
            got = sandbox.run_test(
                ws, ["receipts_locale.tests"], runner=DjangoRunner(), changed_files=[], timeout_s=600, files=app,
                expect=expect,
            )  # fmt: skip
            seen[name] = (got.outcome, got.failure_type, _seen(got.stderr_tail))
            sandbox.close(ws)
        print(json.dumps({name: list(v) for name, v in seen.items()}, indent=1))
        assert seen["3.1"] == (Outcome.PASS, None, "receipts-locale ('UTF-8', 'utf-8', 'UTF-8')"), seen
        assert seen["2.2"] == (Outcome.PASS, None, "receipts-locale ('UTF-8', 'utf-8', 'utf8')"), seen
        # PLANT (round 4's verify:main): the command as round 4's DjangoRunner ran every Django target, with no export.
        assert seen["none"] == (
            Outcome.FAIL, FailureType.ASSERTION, "receipts-locale ('ANSI_X3.4-1968', 'ascii', 'ANSI_X3.4-1968')",
        ), seen  # fmt: skip
    finally:
        sandbox.cleanup()


_DJANGO_XFAIL = """import unittest

from django.test import SimpleTestCase


class T(SimpleTestCase):
    @unittest.expectedFailure
    def test_fails_as_expected(self):
        self.assertEqual(1, 2)

    @unittest.skip("not here")
    def test_skipped(self):
        pass
"""
_SYMPY_XFAIL = """from sympy.utilities.pytest import XFAIL, skip


@XFAIL
def test_fails_as_expected():
    assert 1 == 2


def test_skipped():
    skip("not here")
"""


def test_an_expected_failure_comes_back_flagged_from_each_image(tmp_path: Path) -> None:
    # Contract v3.1 on the run phase's real path: the sandbox's own run, the runner's command and its parser.
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        django = sandbox.prepare(_instance(_present(DJANGO), RunnerKind.DJANGO, _DJANGO_COMMAND), Side.BASE)
        run = sandbox.run_cases(
            django, ["receipts_xf5.tests"], runner=DjangoRunner(), changed_files=[], timeout_s=600,
            files={"tests/receipts_xf5/__init__.py": "", "tests/receipts_xf5/tests.py": _DJANGO_XFAIL},
        )  # fmt: skip
        assert run.problem is None, run.problem
        assert {c.name: (c.outcome, c.expected_failure) for c in run.cases} == {
            "receipts_xf5.tests.T.test_fails_as_expected": (Outcome.SKIPPED, True),
            "receipts_xf5.tests.T.test_skipped": (Outcome.SKIPPED, False),
        }
        assert run.result.outcome is Outcome.SKIPPED  # never a pass in the reduced result
        mod = "sympy/utilities/tests/test_receipts_xf5.py"
        sympy = sandbox.prepare(_instance(_present(SYMPY), RunnerKind.SYMPY, _SYMPY_COMMAND), Side.BASE)
        run = sandbox.run_cases(
            sympy, [mod], runner=SympyRunner(), changed_files=[], timeout_s=600, files={mod: _SYMPY_XFAIL}
        )
        assert run.problem is None, run.problem
        assert {c.name: (c.outcome, c.expected_failure) for c in run.cases} == {
            "test_fails_as_expected": (Outcome.SKIPPED, True),
            "test_skipped": (Outcome.SKIPPED, False),
        }
    finally:
        sandbox.cleanup()


# -- round 6: an image that holds its own copy of Django tests the patched tree; the harness's labels run ------------

# SWE-bench builds Django 1.7 to 2.2 with "python setup.py install" (swebench 4.1.0, harness/constants/python.py, line
# 106) and its eval script runs that install again after applying the patch (harness/test_spec/python.py, lines 443
# to 444), so the harness tests the patched tree. These two images hold Django as an egg in site-packages.
DJANGO_1_11 = (
    "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-7530"
    "@sha256:53920abeaaf9eee5cd7ac3cefdd7b94cb3e600268be0c90de9ed60327c20c987"
)
DJANGO_2_2 = (
    "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-10097"
    "@sha256:222e91d800a2914c5e996ef854c209911481f6da3d6cad3b2efce3d698526750"
)
DJANGO_11141 = (
    "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11141"
    "@sha256:cec14a5eaa9a94504a4affecb91cc027c8672039d9eb9df2b6546efd6b8e7875"
)
_ROWS = json.loads((_RUNNERS / "django_rows.json").read_text())["rows"]
_TESTED_TREE = "Testing against Django installed in '/testbed/django'"  # runtests.py's own line, on stdout


def _changed(patch: str) -> list[str]:
    return re.findall(r"diff --git a/.* b/(.*)", patch)


def _named(ids: list[str]) -> list[str]:
    return [django_label_of(i) for i in ids if re.fullmatch(r"\w+ \([\w.]+\)", i)]


# The cases each gold fixes, as the run reports them. 7530: the test its test patch changes, which SWE-bench names by
# its docstring and lists in neither FAIL_TO_PASS nor PASS_TO_PASS (on base its one failure, here; the listed
# FAIL_TO_PASS test, test_squashmigrations_initial_attribute, passes on base in this run, and every listed case must
# pass with gold, as R24 checks). 10097: the six URLs its test patch adds to invalid_urls.txt, each a test that
# TestSimpleValidators makes from that file and names by its place in the data (test_URLValidator_raises_error_<n>), so
# they are told by the message base code fails them with.
_ROUTERS = "migrations.test_commands.MakeMigrationsTests.test_makemigrations_consistency_checks_respect_routers"
_EGG_CASES = {
    "django__django-7530": ["tests/migrations/test_commands.py"],
    "django__django-10097": ["validators"],
}
_NEW_INVALID_URLS = (
    "http://foo@bar@example.com",
    "http://foo/bar@example.com",
    "http://foo:bar:baz@example.com",
    "http://foo:bar@baz@example.com",
    "http://foo:bar/baz@example.com",
    "http://invalid-.com/?m=foo@example.com",
)


def _failing(run: SuiteRun) -> list[CaseResult]:
    return [c for c in run.cases if c.outcome in (Outcome.FAIL, Outcome.ERROR)]


@pytest.mark.parametrize(
    ("instance", "image"), [("django__django-7530", DJANGO_1_11), ("django__django-10097", DJANGO_2_2)]
)
def test_an_image_that_holds_its_own_egg_of_django_tests_the_patched_tree(
    instance: str, image: str, tmp_path: Path
) -> None:
    # PLANT (the round-5 attack review's blocker): the run phase only mounted the patched tree, and runtests.py put
    # only tests/ on its path, so `import django` read the image's egg on both sides, which runtests.py's first line
    # names ("Testing against Django installed in '.../site-packages/Django-...egg/django'"). The workspace root now
    # goes first on PYTHONPATH for a workspace whose image holds such a copy, so each side tests its own tree, and
    # the cases the gold fixes fail on base and pass with gold.
    row = _ROWS[instance]
    target = _instance(_present(image), RunnerKind.DJANGO, f"{_DJANGO_22_EXPORTS} {_DJANGO_COMMAND}")
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    tests, changed = _EGG_CASES[instance], _changed(row["patch"])
    try:
        base = sandbox.prepare(target, Side.BASE, patch=row["test_patch"])
        gold = sandbox.prepare(target, Side.GOLD, patch=row["patch"] + row["test_patch"])
        before = sandbox.run_cases(base, tests, runner=DjangoRunner(), changed_files=changed, timeout_s=1800)
        after = sandbox.run_cases(gold, tests, runner=DjangoRunner(), changed_files=changed, timeout_s=1800, fresh=True)
        for run in (before, after):
            assert run.problem is None, run.problem
            tested = re.findall(r"Testing against Django installed in '[^']*'", run.result.stdout_tail)
            assert tested == [_TESTED_TREE], tested
        assert sandbox._state[base.id].import_root == sandbox._state[gold.id].import_root == "/testbed"
        assert _failing(after) == [], [(c.name, c.message) for c in _failing(after)]
        if instance == "django__django-7530":
            assert [c.name for c in _failing(before)] == [_ROUTERS], [(c.name, c.message) for c in _failing(before)]
            listed = [*_named(row["FAIL_TO_PASS"]), *_named(row["PASS_TO_PASS"])]
            assert with_expect(after, [*listed, _ROUTERS]).outcome is Outcome.PASS
        else:
            fixed = [c for c in _failing(before) if any(f"'{url}'" in c.message for url in _NEW_INVALID_URLS)]
            assert len(fixed) == len(_NEW_INVALID_URLS) == len(_failing(before)), [c.message for c in _failing(before)]
            assert all(c.failure_type is FailureType.ASSERTION for c in fixed)
            assert with_expect(after, [c.name for c in fixed]).outcome is Outcome.PASS
    finally:
        sandbox.cleanup()


def test_an_image_with_an_editable_django_needs_no_path_of_its_own(tmp_path: Path) -> None:
    # The 11999 image installs Django editably (its egg-link is the workspace): its runs keep their environment.
    target = _instance(_present(DJANGO), RunnerKind.DJANGO, f"{_DJANGO_31_EXPORTS} {_DJANGO_COMMAND}")
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    try:
        ws = sandbox.prepare(target, Side.BASE)
        assert sandbox._state[ws.id].import_root is None
        got = sandbox.run(ws, ["python", "-c", "import django; print(django.__file__)"], timeout_s=300)
        assert got.stdout_tail.strip() == "/testbed/django/__init__.py", got.stdout_tail
    finally:
        sandbox.cleanup()


def test_the_harnesss_labels_for_the_test_patch_of_11141_run_in_its_image(tmp_path: Path) -> None:
    # PLANT (round 5's verify:main): the runner refused the test patch's migration module 0001_initial as a label, so
    # R24 read django__django-11141 as infra on every attempt. Its labels are now swebench 4.1.0's own (runners.py,
    # harness_label), and runtests.py runs them: the FAIL_TO_PASS test, named by its docstring in SWE-bench, fails on
    # base and passes with gold, beside the instance's named PASS_TO_PASS tests.
    row = _ROWS["django__django-11141"]
    target = _instance(_present(DJANGO_11141), RunnerKind.DJANGO, f"{_DJANGO_31_EXPORTS} {_DJANGO_COMMAND}")
    files = _changed(row["test_patch"])
    f2p = ["migrations.test_loader.LoaderTests.test_loading_namespace_package"]  # its docstring, by its source
    expect = f2p + _named(row["PASS_TO_PASS"])
    sandbox = DockerSandbox(runsc=False, work_dir=tmp_path)
    changed = _changed(row["patch"])
    try:
        base = sandbox.prepare(target, Side.BASE, patch=row["test_patch"])
        gold = sandbox.prepare(target, Side.GOLD, patch=row["patch"] + row["test_patch"])
        before = sandbox.run_test(base, files, runner=DjangoRunner(), changed_files=changed, timeout_s=900, expect=f2p)
        assert before.outcome in (Outcome.FAIL, Outcome.ERROR) and before.failure_type is not FailureType.INFRA, (
            before.problem,
            before.stderr_tail[-1500:],
        )
        after = sandbox.run_test(
            gold, files, runner=DjangoRunner(), changed_files=changed, timeout_s=900, fresh=True, expect=expect
        )
        assert after.outcome is Outcome.PASS, (after.problem, after.stderr_tail[-2000:])
    finally:
        sandbox.cleanup()
