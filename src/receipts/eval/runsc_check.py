"""The SWE-bench harness under gVisor on the eval VM: one fresh container per run, the patch applied the harness's way,
the harness's own eval script, graded by the harness's own log parser. Two things use it: the label runner
(`receipts.eval.labelrun`, the SWE-ABS tests on agent patches and on gold) and the harness gold check here
(`runsc-check`), which runs gold against FAIL_TO_PASS and PASS_TO_PASS as root, the way SWE-bench does.

R24 itself (gold must pass in the run phase) is `receipts.eval.r24`, on the sandbox's real path: uid 65534, a read-only
root, RAM-backed workspace. The harness gold check explains R24's drops: an instance whose gold passes here but fails
there needs root or a writable root, and goes back to the founder (REG-565).

This runs on the eval VM, never on the Mac and never in tests. Docker and the harness are reached through two small
interfaces, `Docker` and `Grader`, which the tests replace with fakes. The grader is swebench 4.1.0's own code:
`make_test_spec` builds the eval script and `get_logs_eval` reads its output, so a pass here means what it means in
the harness.

Flags: the runtime, then contracts.RUN_PHASE_LIMITS.limit_flags() (no capabilities, no new privileges, 512 pids, 4g
and no swap, 2 CPUs, 64m of /dev/shm, private IPC, no message queues) and the run phase's --network=none. The harness
runs as root and writes the image's filesystem (it resets and patches test files and, for Django, regenerates
locales), so the run phase's --user, --read-only and RAM-backed mounts are not applied here. A failure is rerun twice
to explain it, under runsc with network and default capabilities, and under runc with the same limits, so the log can
tell a gVisor incompatibility from a broken instance.

Every step that Docker itself performs is checked: create, start, the kernel probe, each file written into the
container, each patch attempt and the eval script. A step Docker cannot finish (a non-zero exit from a write, a
negative exit, exit 125, 126 or 127 from `docker exec`, 255 when the CLI loses the daemon, a timeout outside the eval
script) makes the run INFRA: rerun it, never read it as a verdict on the instance. `docker exec` also exits 1 for a
container that is gone or stopped, the same code a failing command gives, so after the kernel probe or a patch
attempt exits non-zero, and after the eval script however it ended, a checked `docker inspect -f {{.State.Running}}`
must say the container is still running; if it does not, or the inspect fails, the run is INFRA.

Docker's own report of a container's runtime only echoes the --runtime flag it was given, so it proves nothing. The
check asks the kernel instead: gVisor's sentry answers `dmesg` with a log that starts "Starting gVisor..." (gVisor
release-20261005.0, pkg/sentry/kernel/syslog.go, and its syslog(2) needs no capability), while a container on the
host kernel with every capability dropped cannot read the kernel log at all. Under runsc, only a kernel log that
reads (dmesg exits 0) and lacks the banner stops the whole check; a probe that exits non-zero read nothing (the
container died right after start, say) and is that instance's INFRA. `preflight` still proves the VM's runsc is
gVisor before any instance runs.

A test passes only when the harness's log parser reads it as PASSED. swebench also counts XFAIL as a pass and leaves
SKIPPED out of both lists; here a skipped or missing test always fails, and an XFAIL is accepted only for a
PASS_TO_PASS test (a test the repository marks xfail is XFAIL on every runtime, so XFAIL is its unchanged state).
"""

from __future__ import annotations

import fcntl
import importlib
import json
import os
import subprocess
import tempfile
import threading
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field

from receipts.contracts import RUN_PHASE_LIMITS, Frozen, PatchMethod, SandboxLimits
from receipts.eval.hashing import sha256_text, write_text_atomic
from receipts.eval.images import ImageRef

# swebench 4.1.0, harness/run_evaluation.py GIT_APPLY_CMDS: the harness's own order of patch attempts, each named by
# the contract's PatchMethod (REG-569). Each runs on the tree the previous attempt left, and the first exit 0 wins.
PATCH_CHAIN: tuple[tuple[PatchMethod, str], ...] = (
    (PatchMethod.GIT_APPLY, "git apply --verbose"),
    (PatchMethod.GIT_APPLY_REJECT, "git apply --verbose --reject"),
    (PatchMethod.GNU_PATCH_FUZZ, "patch --batch --fuzz=5 -p1 -i"),
)
DOCKER_PATCH = "/tmp/patch.diff"
DOCKER_WORKDIR = "/testbed"
# `docker exec` and `docker run` exit 125 when Docker or the runtime fails, 126 when the command cannot be invoked and
# 127 when it is not found, and the CLI exits 255 when its connection to the daemon drops (docker/cli,
# cli/command/container/exec.go: getExecExitStatus returns StatusError{-1}): Docker's failure, not the command's.
DOCKER_SIDE_EXITS = frozenset({125, 126, 127, 255})
OUTPUT_LIMIT = 2_000_000  # characters of eval output kept for grading; the harness's logs are far smaller

GVISOR_BANNER = "Starting gVisor..."


class RunConfig(Frozen):
    name: str
    runtime: Literal["runsc", "runc"]
    network: bool  # False: the run phase's --network=none
    cap_drop_all: bool  # True: the run phase's --cap-drop=ALL
    limits: SandboxLimits = RUN_PHASE_LIMITS


PRIMARY = RunConfig(name="runsc-rt2", runtime="runsc", network=False, cap_drop_all=True)
DIAGNOSTICS = (
    RunConfig(name="runsc-open", runtime="runsc", network=True, cap_drop_all=False),
    RunConfig(name="runc-rt2", runtime="runc", network=False, cap_drop_all=True),
)


def network_none(limits: SandboxLimits) -> str:
    """The run phase's network flag, as contracts.SandboxLimits renders it."""
    return next(f for f in limits.run_flags() if f.startswith("--network="))


def limit_flags(config: RunConfig) -> list[str]:
    """The flags for one configuration: its runtime, then the contract's shared limits and its network flag."""
    lim = config.limits
    shared = [f for f in lim.limit_flags() if config.cap_drop_all or not f.startswith("--cap-drop=")]
    return [f"--runtime={config.runtime}", *shared, *([] if config.network else [network_none(lim)])]


def create_args(config: RunConfig, name: str, image: ImageRef) -> list[str]:
    """`docker create` for one run. --pull=never: only the pinned images already on disk."""
    return [
        "docker",
        "create",
        "--pull=never",
        "--name",
        name,
        *limit_flags(config),
        "--label",
        "receipts.eval=harness",
        image.ref,
        "tail",
        "-f",
        "/dev/null",
    ]


def probe_args(config: RunConfig, image: ImageRef) -> list[str]:
    """`docker run` of `dmesg` in a throwaway container: under gVisor its first line is GVISOR_BANNER."""
    return ["docker", "run", "--rm", "--pull=never", *limit_flags(config), image.ref, "dmesg"]


def shows_gvisor(output: str) -> bool:
    return any(line.rstrip().endswith(GVISOR_BANNER) for line in output.splitlines())


class Completed(Frozen):
    returncode: int
    stdout: str = ""
    stderr: str = ""
    timed_out: bool = False


class Docker(Protocol):
    def __call__(self, args: Sequence[str], *, timeout_s: float, stdin: str | None = None) -> Completed: ...


def _text(data: bytes | str | None) -> str:
    """Bytes from a container, decoded so that no byte sequence a test prints can stop the run."""
    if data is None:
        return ""
    return data if isinstance(data, str) else data.decode("utf-8", "replace")


def subprocess_docker(args: Sequence[str], *, timeout_s: float, stdin: str | None = None) -> Completed:
    try:
        p = subprocess.run(
            list(args),
            input=None if stdin is None else stdin.encode("utf-8"),
            capture_output=True,
            timeout=timeout_s,
            check=False,
        )
    except subprocess.TimeoutExpired as exc:
        return Completed(returncode=-1, stdout=_text(exc.stdout), stderr=_text(exc.stderr), timed_out=True)
    return Completed(returncode=p.returncode, stdout=_text(p.stdout), stderr=_text(p.stderr))


class Grade(Frozen):
    """One run, read by the harness's own log parser."""

    log_found: bool  # the harness found its test-output markers and no apply, reset, error or timeout code
    harness_resolved: bool  # swebench's verdict, which counts XFAIL as a pass and SKIPPED as neither
    statuses: dict[str, str]  # test -> PASSED, FAILED, SKIPPED, ERROR or XFAIL, as the log parser read it
    fail_to_pass: tuple[str, ...]
    pass_to_pass: tuple[str, ...]

    @property
    def fail_to_pass_failed(self) -> tuple[str, ...]:
        """Every FAIL_TO_PASS test the run did not show as PASSED: failed, errored, skipped, xfailed or missing."""
        return tuple(t for t in self.fail_to_pass if self.statuses.get(t) != "PASSED")

    @property
    def pass_to_pass_failed(self) -> tuple[str, ...]:
        """Every PASS_TO_PASS test that is neither PASSED nor XFAIL (its unchanged state when the repo marks it)."""
        return tuple(t for t in self.pass_to_pass if self.statuses.get(t) not in ("PASSED", "XFAIL"))

    @property
    def passed(self) -> bool:
        return self.log_found and not self.fail_to_pass_failed and not self.pass_to_pass_failed


class Grader(Protocol):
    def eval_script(self, row: Mapping[str, Any]) -> str: ...

    def grade(self, row: Mapping[str, Any], test_output: str) -> Grade: ...


class SwebenchGrader:
    """swebench==4.1.0's test spec and grading, imported only where the harness is installed (the VM)."""

    def __init__(self) -> None:
        self._spec: Any = importlib.import_module("swebench.harness.test_spec.test_spec")
        self._grading: Any = importlib.import_module("swebench.harness.grading")

    def eval_script(self, row: Mapping[str, Any]) -> str:
        script: str = self._spec.make_test_spec(dict(row)).eval_script
        return script

    def grade(self, row: Mapping[str, Any], test_output: str) -> Grade:
        spec = self._spec.make_test_spec(dict(row))
        iid = str(row["instance_id"])
        with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False, encoding="utf-8") as fh:
            fh.write(test_output)
            path = fh.name
        try:
            statuses, found = self._grading.get_logs_eval(spec, path)
            prediction = {"instance_id": iid, "model_name_or_path": "gold", "model_patch": row["patch"]}
            report: dict[str, Any] = self._grading.get_eval_report(spec, prediction, path, include_tests_status=True)
        finally:
            Path(path).unlink(missing_ok=True)
        return Grade(
            log_found=bool(found),
            harness_resolved=bool(report[iid]["resolved"]),
            statuses={str(k): str(v) for k, v in dict(statuses).items()},
            fail_to_pass=tuple(spec.FAIL_TO_PASS),
            pass_to_pass=tuple(spec.PASS_TO_PASS),
        )


class RuntimeMismatchError(RuntimeError):
    """A container that did not run under the runtime asked for. The whole check stops: the VM is misconfigured."""


class ContainerRun(Frozen):
    """One harness run in one fresh container, before any grading."""

    gvisor: bool | None = None  # the container's kernel log showed GVISOR_BANNER; None when it never got that far
    patch_applied: bool = False
    patch_method: PatchMethod | None = None  # the step of PATCH_CHAIN that applied it; None when none did
    timed_out: bool = False  # the eval script ran out of time
    output: str = ""
    seconds: float = Field(default=0.0, ge=0)
    infra: str = ""  # a step Docker could not finish: rerun, never a verdict on the instance
    error: str = ""  # what went wrong on the instance itself: the patch did not apply


def _tail(c: Completed) -> str:
    return (c.stderr or c.stdout).strip()[-500:]


class _InfraError(Exception):
    pass


def _checked(
    docker: Docker, args: Sequence[str], what: str, *, timeout_s: float, stdin: str | None = None
) -> Completed:
    """A Docker step that must finish: a timeout or a Docker-side exit code is INFRA."""
    r = docker(args, timeout_s=timeout_s, stdin=stdin)
    if r.timed_out:
        raise _InfraError(f"{what} timed out after {timeout_s:g} s")
    if r.returncode < 0 or r.returncode in DOCKER_SIDE_EXITS:
        raise _InfraError(f"{what}: docker exited {r.returncode}: {_tail(r)}")
    return r


def _write(docker: Docker, name: str, path: str, content: str, what: str) -> None:
    """Write one file into a running container; anything but exit 0 means it is not there."""
    r = _checked(
        docker, ["docker", "exec", "-i", name, "bash", "-c", f"cat > {path}"], what, timeout_s=60, stdin=content
    )
    if r.returncode != 0:
        raise _InfraError(f"{what}: exit {r.returncode}: {_tail(r)}")


def inspect_args(name: str) -> list[str]:
    """The one thing this module asks `docker inspect` about a container: whether it is running. Never its runtime,
    which only echoes the --runtime flag."""
    return ["docker", "inspect", "-f", "{{.State.Running}}", name]


def _alive(docker: Docker, name: str, after: str) -> None:
    """The container still runs after `after`. docker exec exits 1 for a container that is gone or stopped, which is
    also what a failing command exits, so a non-zero exit is the command's own only once this holds."""
    r = _checked(docker, inspect_args(name), f"inspecting the container after {after}", timeout_s=60)
    if r.returncode != 0:
        raise _InfraError(f"after {after}, docker inspect exited {r.returncode}: {_tail(r)}")
    state = r.stdout.strip()
    if state != "true":
        raise _InfraError(f"after {after}, the container is not running (State.Running is {state or 'empty'})")


def run_in_container(
    docker: Docker,
    image: ImageRef,
    config: RunConfig,
    *,
    name: str,
    patch: str,
    eval_script: str,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> ContainerRun:
    """Create, start, prove the kernel, apply the patch the harness's way (PATCH_CHAIN, recording the step that
    applied it), run the eval script, remove."""
    start = clock()

    def done(**kw: Any) -> ContainerRun:
        return ContainerRun(seconds=round(max(0.0, clock() - start), 3), **kw)

    gvisor: bool | None = None
    applied = False
    method: PatchMethod | None = None
    try:
        made = docker(create_args(config, name, image), timeout_s=300)
        if made.timed_out or made.returncode != 0:
            raise _InfraError(f"docker create failed: {_tail(made)}")
        started = docker(["docker", "start", name], timeout_s=120)
        if started.timed_out or started.returncode != 0:
            raise _InfraError(f"docker start failed: {_tail(started)}")
        kernel = _checked(docker, ["docker", "exec", name, "dmesg"], "the kernel probe", timeout_s=60)
        if kernel.returncode != 0:
            _alive(docker, name, "the kernel probe")
        gvisor = kernel.returncode == 0 and shows_gvisor(kernel.stdout)
        if config.runtime == "runsc" and not gvisor:
            if kernel.returncode != 0:  # nothing was read, so nothing says which kernel answered: this run's INFRA
                raise _InfraError(
                    f"the kernel probe exited {kernel.returncode} and read no kernel log: {_tail(kernel)}"
                )
            raise RuntimeMismatchError(
                f"{name}: asked for runsc, but the kernel log the container read does not show {GVISOR_BANNER!r}"
            )
        if config.runtime == "runc" and gvisor:
            raise RuntimeMismatchError(
                f"{name}: asked for runc, but the container's kernel log shows {GVISOR_BANNER!r}"
            )
        _write(docker, name, DOCKER_PATCH, patch, "writing the patch")
        for step, cmd in PATCH_CHAIN:
            r = _checked(
                docker,
                ["docker", "exec", "-w", DOCKER_WORKDIR, name, "bash", "-c", f"{cmd} {DOCKER_PATCH}"],
                f"`{cmd}`",
                timeout_s=120,
            )
            if r.returncode == 0:
                applied, method = True, step
                break
            _alive(docker, name, f"`{cmd}` exited {r.returncode}")
        if not applied:
            return done(gvisor=gvisor, error="the patch did not apply")
        _write(docker, name, "/eval.sh", eval_script, "writing the eval script")
        out = docker(["docker", "exec", name, "bash", "-c", "/bin/bash /eval.sh 2>&1"], timeout_s=timeout_s)
        if not out.timed_out and (out.returncode < 0 or out.returncode in DOCKER_SIDE_EXITS):
            raise _InfraError(f"the eval script: docker exited {out.returncode}: {_tail(out)}")
        _alive(docker, name, "the eval script")  # a container that died mid-run left a cut-off log, not a verdict
        if out.timed_out:
            return done(
                gvisor=gvisor,
                patch_applied=True,
                patch_method=method,
                timed_out=True,
                output=out.stdout[-OUTPUT_LIMIT:],
            )
        return done(gvisor=gvisor, patch_applied=True, patch_method=method, output=out.stdout[-OUTPUT_LIMIT:])
    except _InfraError as e:
        return done(gvisor=gvisor, patch_applied=applied, patch_method=method, infra=str(e))
    finally:
        docker(["docker", "rm", "-f", name], timeout_s=120)


class Attempt(Frozen):
    config: str
    gvisor: bool | None = None  # the container's kernel log showed GVISOR_BANNER; None when it never started
    patch_applied: bool = False  # the harness's chain applied the gold patch
    patch_method: PatchMethod | None = None  # which step of the chain applied it (REG-569)
    log_parsed: bool = False  # the harness found its test-output markers and no apply, reset, error or timeout code
    timed_out: bool = False
    harness_resolved: bool = False  # swebench's own verdict, kept for the log
    resolved: bool = False  # every FAIL_TO_PASS test PASSED and every PASS_TO_PASS test PASSED or XFAIL
    fail_to_pass_failed: tuple[str, ...] = ()
    pass_to_pass_failed: tuple[str, ...] = ()
    seconds: float = Field(default=0.0, ge=0)
    infra: str = ""  # Docker could not finish a step: rerun
    error: str = ""


class Verdict(StrEnum):
    KEPT = "kept"
    NEEDS_NETWORK_OR_CAPS = "needs_network_or_caps"  # passes under runsc only with network and default capabilities
    RUNSC_INCOMPATIBLE = "runsc_incompatible"  # fails under runsc either way, passes under runc with RT2's flags
    GOLD_FAILS = "gold_fails"  # fails under every configuration: the instance is broken in this image
    PATCH_DOES_NOT_APPLY = "patch_does_not_apply"
    TIMEOUT = "timeout"
    INFRA = "infra"  # docker itself failed; rerun before reading anything into it


class InstanceResult(Frozen):
    instance_id: str
    image: str
    verdict: Verdict
    attempts: tuple[Attempt, ...]

    @property
    def kept(self) -> bool:
        return self.verdict is Verdict.KEPT


def run_gold(
    row: Mapping[str, Any],
    image: ImageRef,
    config: RunConfig,
    docker: Docker,
    grader: Grader,
    *,
    timeout_s: float,
    run_id: str,
    clock: Callable[[], float] = time.monotonic,
) -> Attempt:
    iid = str(row["instance_id"])
    name = f"receipts-gold-{run_id}-{config.name}-{sha256_text(iid)[:12]}"
    run = run_in_container(
        docker,
        image,
        config,
        name=name,
        patch=str(row["patch"]),
        eval_script=grader.eval_script(row),
        timeout_s=timeout_s,
        clock=clock,
    )
    base = {"config": config.name, "gvisor": run.gvisor, "seconds": run.seconds, "patch_method": run.patch_method}
    if run.infra:
        return Attempt(**base, patch_applied=run.patch_applied, infra=run.infra)
    if not run.patch_applied:
        return Attempt(**base, error=run.error)
    if run.timed_out:
        return Attempt(**base, patch_applied=True, timed_out=True)
    g = grader.grade(row, run.output)
    return Attempt(
        **base,
        patch_applied=True,
        log_parsed=g.log_found,
        harness_resolved=g.harness_resolved,
        resolved=g.passed,
        fail_to_pass_failed=g.fail_to_pass_failed,
        pass_to_pass_failed=g.pass_to_pass_failed,
    )


def classify(primary: Attempt, diagnostics: Sequence[Attempt]) -> Verdict:
    if primary.resolved:
        return Verdict.KEPT
    if primary.infra or any(a.infra for a in diagnostics):
        return Verdict.INFRA  # a drop's reason must be read from runs that finished
    if not primary.patch_applied:
        return Verdict.PATCH_DOES_NOT_APPLY
    by = {a.config: a for a in diagnostics}
    runc = by.get("runc-rt2")
    open_runsc = by.get("runsc-open")
    if open_runsc is not None and open_runsc.resolved:
        return Verdict.NEEDS_NETWORK_OR_CAPS
    if runc is not None and runc.resolved:
        return Verdict.RUNSC_INCOMPATIBLE
    if primary.timed_out:
        return Verdict.TIMEOUT
    return Verdict.GOLD_FAILS


def check_instance(
    row: Mapping[str, Any], image: ImageRef, docker: Docker, grader: Grader, *, timeout_s: float, run_id: str
) -> InstanceResult:
    if str(row["instance_id"]) != image.instance_id:
        raise ValueError(f"row {row['instance_id']} paired with the image for {image.instance_id}")
    primary = run_gold(row, image, PRIMARY, docker, grader, timeout_s=timeout_s, run_id=run_id)
    diagnostics: list[Attempt] = []
    if not primary.resolved and primary.patch_applied and not primary.infra:
        diagnostics = [run_gold(row, image, c, docker, grader, timeout_s=timeout_s, run_id=run_id) for c in DIAGNOSTICS]
    return InstanceResult(
        instance_id=image.instance_id,
        image=image.ref,
        verdict=classify(primary, diagnostics),
        attempts=(primary, *diagnostics),
    )


class PreflightError(RuntimeError):
    """The VM is not ready: Docker unreachable, runsc missing or not gVisor, or an image not on disk."""


def preflight(docker: Docker, images: Sequence[ImageRef]) -> None:
    """Docker answers, runsc is registered, every image is on disk, a probe under runsc really runs on gVisor and a
    probe under runc really does not. Nothing runs on an instance until all of that holds."""
    info = docker(["docker", "info", "--format", "{{json .Runtimes}}"], timeout_s=60)
    if info.returncode != 0:
        raise PreflightError(f"docker info failed (is this user in the docker group?): {info.stderr.strip()[-500:]}")
    if '"runsc"' not in info.stdout:
        raise PreflightError("runsc is not registered with Docker; run the cloud-init bootstrap first")
    if not images:
        raise PreflightError("no images to check")
    for image in images:
        if docker(["docker", "image", "inspect", image.ref], timeout_s=60).returncode != 0:
            raise PreflightError(f"{image.ref} is not on this machine; the check never pulls")
    runc = next(c for c in DIAGNOSTICS if c.runtime == "runc")
    for config, want in ((PRIMARY, True), (runc, False)):
        probe = docker(probe_args(config, images[0]), timeout_s=300)
        # docker run exits 125 when the daemon or runtime fails, 126 or 127 when the command cannot run (no dmesg);
        # any other code is dmesg's own, and dmesg failing is what a host kernel with no capabilities answers.
        if probe.timed_out or probe.returncode in DOCKER_SIDE_EXITS:
            raise PreflightError(
                f"the {config.runtime} probe did not run (exit {probe.returncode}): "
                f"{(probe.stderr or probe.stdout).strip()[-500:]}"
            )
        if shows_gvisor(probe.stdout) != want:
            raise RuntimeMismatchError(
                f"the {config.runtime} probe's kernel log {'lacks' if want else 'shows'} {GVISOR_BANNER!r}: "
                f"{'runsc is not running gVisor' if want else 'runc is not the host runtime'}"
            )


class ResultLog:
    """An append-only JSONL file of finished results, one line each, flushed and fsynced as each one lands, so a crash
    or a stop keeps every result already written. Threads share one log; the file lock keeps two processes apart."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self._lock = threading.Lock()
        path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, result: BaseModel) -> None:
        line = result.model_dump_json() + "\n"
        with self._lock, self.path.open("a", encoding="utf-8") as fh:
            fcntl.flock(fh, fcntl.LOCK_EX)
            try:
                fh.write(line)
                fh.flush()
                os.fsync(fh.fileno())
            finally:
                fcntl.flock(fh, fcntl.LOCK_UN)

    def lines(self) -> list[str]:
        if not self.path.is_file():
            return []
        return [line for line in self.path.read_text(encoding="utf-8").splitlines() if line.strip()]


class Report(Frozen):
    run_id: str
    instances: int
    kept: int
    verdicts: dict[str, int]
    dropped: dict[str, str]  # instance -> verdict
    max_drop_pct: int

    @property
    def drop_pct_over_limit(self) -> bool:
        return (self.instances - self.kept) * 100 > self.max_drop_pct * self.instances


def run_check(
    rows: Mapping[str, Mapping[str, Any]],
    images: Sequence[ImageRef],
    docker: Docker,
    grader: Grader,
    out_dir: Path,
    *,
    run_id: str,
    timeout_s: float = 1800,
    workers: int = 16,
    max_drop_pct: int = 25,
) -> Report:
    """Check every image's instance; results.jsonl gets each result as it finishes, then kept.txt, dropped.tsv and
    report.json. Never pulls, never keeps a failure. An instance whose check raises anything but a runtime mismatch
    is recorded as INFRA, so one bad instance cannot lose the others' work."""
    missing = [i.instance_id for i in images if i.instance_id not in rows]
    if missing:
        raise ValueError(f"no dataset row for {len(missing)} instances, e.g. {missing[:3]}")
    preflight(docker, images)
    out_dir.mkdir(parents=True, exist_ok=True)
    log = ResultLog(out_dir / "results.jsonl")
    if log.lines():
        raise ValueError(f"{log.path} already has results; use a new --out directory for a new run")

    def one(image: ImageRef) -> InstanceResult:
        try:
            return check_instance(rows[image.instance_id], image, docker, grader, timeout_s=timeout_s, run_id=run_id)
        except RuntimeMismatchError:
            raise
        except Exception as e:  # anything else is this instance's infrastructure, recorded, never fatal
            infra = Attempt(config=PRIMARY.name, infra=f"{type(e).__name__}: {e}"[-500:])
            return InstanceResult(
                instance_id=image.instance_id, image=image.ref, verdict=Verdict.INFRA, attempts=(infra,)
            )

    results: list[InstanceResult] = []
    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(one, image) for image in images]
        for future in as_completed(futures):
            r = future.result()  # a RuntimeMismatchError stops everything still queued
            log.append(r)
            results.append(r)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    results.sort(key=lambda r: r.instance_id)
    write_text_atomic(out_dir / "kept.txt", "".join(f"{r.instance_id}\n" for r in results if r.kept))
    dropped_lines = []
    for r in results:
        if r.kept:
            continue
        primary = r.attempts[0]
        failing = ",".join((*primary.fail_to_pass_failed, *primary.pass_to_pass_failed)[:5])
        dropped_lines.append(f"{r.instance_id}\t{r.verdict.value}\t{primary.infra or primary.error or failing}\n")
    write_text_atomic(out_dir / "dropped.tsv", "".join(dropped_lines))
    report = Report(
        run_id=run_id,
        instances=len(results),
        kept=sum(1 for r in results if r.kept),
        verdicts=dict(sorted(Counter(r.verdict.value for r in results).items())),
        dropped={r.instance_id: r.verdict.value for r in results if not r.kept},
        max_drop_pct=max_drop_pct,
    )
    write_text_atomic(out_dir / "report.json", json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    return report


def load_rows(path: Path, *, expect_sha256: str | None = None) -> dict[str, dict[str, Any]]:
    """SWE-bench Verified rows by instance id, from JSONL or from the pinned parquet (pyarrow comes with swebench)."""
    from receipts.eval.hashing import sha256_file

    if expect_sha256 is not None and sha256_file(path) != expect_sha256:
        raise ValueError(f"{path}: sha256 does not match the pinned dataset")
    if path.suffix == ".parquet":
        pq: Any = importlib.import_module("pyarrow.parquet")
        rows: list[dict[str, Any]] = pq.read_table(path).to_pylist()
    else:
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    out = {str(r["instance_id"]): r for r in rows}
    if len(out) != len(rows):
        raise ValueError(f"{path}: duplicate instance ids")
    return out
