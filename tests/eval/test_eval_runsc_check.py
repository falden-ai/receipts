"""The harness gold check keeps only gold that passes under gVisor with the run phase's limits, proves the kernel it ran
on, explains every drop, cleans up every container, turns every step Docker could not finish into INFRA (never a drop),
and stops when the VM is not ready. Docker and the harness are fakes: no container runs here."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from receipts.contracts import RUN_PHASE_LIMITS, PatchMethod
from receipts.eval.images import ImageRef
from receipts.eval.runsc_check import (
    DIAGNOSTICS,
    GVISOR_BANNER,
    PATCH_CHAIN,
    PRIMARY,
    Attempt,
    Completed,
    Grade,
    InstanceResult,
    PreflightError,
    Report,
    RunConfig,
    RuntimeMismatchError,
    Verdict,
    check_instance,
    classify,
    create_args,
    inspect_args,
    load_rows,
    preflight,
    probe_args,
    run_check,
    run_gold,
    subprocess_docker,
)

PASS = "TESTS PASS"
FAIL = "TESTS FAIL"
GVISOR_DMESG = f"[    0.000000] {GVISOR_BANNER}\n[    0.412345] Waiting for children...\n[    1.023456] Ready!\n"
HOST_DMESG_DENIED = "dmesg: read kernel buffer failed: Operation not permitted\n"
HOST_DMESG_READ = "[    0.000000] Linux version 6.8.0-1015-aws (buildd@lcy02-amd64-001)\n[    0.1] Command line: ro\n"


def image(iid: str) -> ImageRef:
    return ImageRef(
        instance_id=iid,
        image=f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}",
        digest="sha256:" + "e" * 64,
        media_type="application/vnd.docker.distribution.manifest.v2+json",
    )


def row(iid: str) -> dict[str, Any]:
    return {"instance_id": iid, "patch": "diff --git a/x b/x\n", "repo": "django/django"}


Kernel = bool | str  # True: gVisor's banner; False: a host kernel refusing syslog (exit 1); "readable": a host log


class FakeDocker:
    """A Docker CLI stand-in that answers like real Docker. `outcome[(instance, config)]` is what the eval script
    prints; anything else passes. The kernel a container really runs on is its runtime's, unless a plant says not. A
    container can die after a step (`gone_after`): from then on `docker exec` on it exits 1, as Docker's own CLI does
    for a container that is gone or not running, and `docker inspect` says it is stopped (or, removed, exits 1)."""

    def __init__(
        self,
        outcome: Mapping[tuple[str, str], str] | None = None,
        *,
        kernel: Mapping[str, Kernel] | None = None,  # runtime -> what dmesg answers there (a misconfigured VM)
        probe_fails: bool = False,
        apply_fails: bool = False,
        create_fails: bool = False,
        runsc_registered: bool = True,
        info_fails: bool = False,
        missing_images: Sequence[str] = (),
        step: Mapping[str, Completed] | None = None,  # a Docker step's answer, by step name (see _step)
        gone_after: str | None = None,  # a step name, or "start": the container dies once it has run
        gone: str = "stopped",  # "stopped": inspect says false; "removed": inspect exits 1, no such object
        inspect: Completed | None = None,  # a planted answer to every docker inspect
    ) -> None:
        self.outcome = dict(outcome or {})
        self.step = dict(step or {})
        self.kernel: dict[str, Kernel] = {"runsc": True, "runc": False} | dict(kernel or {})
        self.probe_fails = probe_fails
        self.apply_fails = apply_fails
        self.create_fails = create_fails
        self.runsc_registered = runsc_registered
        self.info_fails = info_fails
        self.missing_images = set(missing_images)
        self.gone_after = gone_after
        self.gone = gone
        self.inspect = inspect
        self.calls: list[list[str]] = []
        self.containers: dict[str, tuple[str, str, str]] = {}  # name -> (instance, config, runtime)
        self.removed: list[str] = []
        self.dead: set[str] = set()

    def _dmesg(self, runtime: str) -> Completed:
        k = self.kernel[runtime]
        if k is True:
            return Completed(returncode=0, stdout=GVISOR_DMESG)
        if k == "readable":
            return Completed(returncode=0, stdout=HOST_DMESG_READ)
        return Completed(returncode=1, stderr=HOST_DMESG_DENIED)

    def __call__(self, args: Sequence[str], *, timeout_s: float, stdin: str | None = None) -> Completed:
        a = list(args)
        self.calls.append(a)
        verb = a[1]
        runtime = next((x.split("=", 1)[1] for x in a if x.startswith("--runtime=")), "")
        if verb == "info":
            if self.info_fails:
                return Completed(returncode=1, stderr="permission denied while trying to connect to the Docker daemon")
            runtimes = {"runc": {}, **({"runsc": {}} if self.runsc_registered else {})}
            return Completed(returncode=0, stdout=json.dumps(runtimes))
        if verb == "image":
            ref = a[-1]
            return Completed(returncode=1 if any(m in ref for m in self.missing_images) else 0)
        if verb == "run":  # the preflight probe: docker run --rm ... <image> dmesg
            assert a[-1] == "dmesg" and "--rm" in a
            if self.probe_fails and runtime == "runsc":
                return Completed(returncode=125, stderr="OCI runtime create failed: runsc: exit status 1")
            return self._dmesg(runtime)
        if verb == "create":
            if self.create_fails:
                return Completed(returncode=1, stderr="no space left on device")
            name = a[a.index("--name") + 1]
            config = next((c.name for c in (PRIMARY, *DIAGNOSTICS) if f"-{c.name}-" in name), PRIMARY.name)
            iid = a[-4].split("x86_64.")[1].split("@")[0]
            self.containers[name] = (iid, config, runtime)
            return Completed(returncode=0)
        if verb == "start":
            if self.gone_after == "start":
                self.dead.add(a[2])
            return Completed(returncode=0)
        if verb == "rm":
            self.removed.append(a[3])
            return Completed(returncode=0)
        if verb == "inspect":
            assert a[2:4] == ["-f", "{{.State.Running}}"], a  # the runtime field is never asked for
            name = a[4]
            if self.inspect is not None:
                return self.inspect
            if name not in self.containers or name in self.removed or (name in self.dead and self.gone == "removed"):
                return Completed(returncode=1, stderr=f"Error: No such object: {name}")
            return Completed(returncode=0, stdout="false\n" if name in self.dead else "true\n")
        if verb == "exec":
            name = self._exec_name(a)
            if name in self.dead:
                return Completed(returncode=1, stderr=f"Error response from daemon: container {name} is not running")
            answer = self._exec(name, a[-1])
            if self._step(a[-1]) == self.gone_after:
                self.dead.add(name)
            return answer
        raise AssertionError(f"unexpected docker call {a}")

    def _exec(self, name: str, command: str) -> Completed:
        iid, config, container_runtime = self.containers[name]
        planted = self.step.get(self._step(command))
        if planted is not None:
            return planted
        if command == "dmesg":
            return self._dmesg(container_runtime)
        if "git apply" in command or "patch --batch" in command:
            return Completed(returncode=1 if self.apply_fails else 0)
        if "cat >" in command:
            return Completed(returncode=0)
        out = self.outcome.get((iid, config), PASS)
        if out == "TIMEOUT":
            return Completed(returncode=-1, timed_out=True)
        return Completed(returncode=0, stdout=out)

    @staticmethod
    def _step(command: str) -> str:
        if command == "dmesg":
            return "dmesg"
        if "cat > /tmp/patch.diff" in command:
            return "write-patch"
        if "cat > /eval.sh" in command:
            return "write-eval"
        if command.startswith("git apply --verbose /") or command.startswith("git apply --verbose --reject"):
            return "apply" if "--reject" not in command else "apply-reject"
        if command.startswith("patch --batch"):
            return "apply-patch"
        if "/eval.sh 2>&1" in command:
            return "eval"
        return command

    @staticmethod
    def _exec_name(a: list[str]) -> str:
        rest = a[2:]
        while rest and rest[0].startswith("-"):
            rest = rest[2:] if rest[0] == "-w" else rest[1:]
        return rest[0]


class FakeGrader:
    """The harness's log parser, reduced to what the tests need: the eval output names each test's status."""

    F2P = ("t::fix",)
    P2P = ("t::old",)

    def eval_script(self, row: Mapping[str, Any]) -> str:
        return "echo run the tests"

    def grade(self, row: Mapping[str, Any], test_output: str) -> Grade:
        statuses = {"t::fix": "PASSED", "t::old": "PASSED"}
        if test_output == FAIL:
            statuses["t::fix"] = "FAILED"
        elif test_output.startswith("STATUS "):  # "STATUS t::fix=XFAIL"
            test, status = test_output.removeprefix("STATUS ").split("=")
            statuses[test] = status
        return Grade(
            log_found=True,
            harness_resolved=all(v in ("PASSED", "XFAIL") for v in statuses.values()),
            statuses=statuses,
            fail_to_pass=self.F2P,
            pass_to_pass=self.P2P,
        )


def check(docker: FakeDocker, iid: str = "django__django-1") -> InstanceResult:
    return check_instance(row(iid), image(iid), docker, FakeGrader(), timeout_s=60, run_id="t")


def test_the_primary_configuration_carries_the_contracts_run_phase_limits() -> None:
    args = create_args(PRIMARY, "n", image("django__django-1"))
    lim = RUN_PHASE_LIMITS
    assert "--runtime=runsc" in args and "--network=none" in args
    for flag in lim.limit_flags():  # every shared limit, as the contract renders it
        assert flag in args
    assert "--pull=never" in args and args[-4].endswith("@sha256:" + "e" * 64)
    names = {c.name: c for c in DIAGNOSTICS}
    assert "--network=none" not in create_args(names["runsc-open"], "n", image("django__django-1"))
    assert "--runtime=runc" in create_args(names["runc-rt2"], "n", image("django__django-1"))


def test_the_flags_come_from_the_contract_not_from_a_copy() -> None:
    tighter = PRIMARY.model_copy(update={"limits": RUN_PHASE_LIMITS.model_copy(update={"pids_limit": 99})})
    assert "--pids-limit=99" in create_args(tighter, "n", image("django__django-1"))
    assert PRIMARY.limits is RUN_PHASE_LIMITS
    open_args = create_args(DIAGNOSTICS[0], "n", image("django__django-1"))
    assert not any(f.startswith("--cap-drop=") or f.startswith("--network=") for f in open_args)
    assert all(f in open_args for f in RUN_PHASE_LIMITS.limit_flags() if not f.startswith("--cap-drop="))


def test_gold_that_passes_on_gvisor_is_kept_without_extra_runs() -> None:
    d = FakeDocker()
    r = check(d)
    assert r.verdict is Verdict.KEPT and len(r.attempts) == 1 and r.attempts[0].gvisor is True
    # Docker's own runtime field is never trusted: the one inspect, after the eval script, asks whether it runs
    assert [c for c in d.calls if c[1] == "inspect"] == [inspect_args(next(iter(d.containers)))]


@pytest.mark.parametrize(
    ("outcome", "verdict"),
    [
        ({"runsc-rt2": FAIL, "runsc-open": FAIL}, Verdict.RUNSC_INCOMPATIBLE),
        ({"runsc-rt2": FAIL}, Verdict.NEEDS_NETWORK_OR_CAPS),
        ({"runsc-rt2": FAIL, "runsc-open": FAIL, "runc-rt2": FAIL}, Verdict.GOLD_FAILS),
        ({"runsc-rt2": "TIMEOUT", "runsc-open": FAIL, "runc-rt2": FAIL}, Verdict.TIMEOUT),
    ],
)
def test_plant_a_failing_gold_is_dropped_with_its_reason(outcome: dict[str, str], verdict: Verdict) -> None:
    iid = "django__django-1"
    d = FakeDocker({(iid, k): v for k, v in outcome.items()})
    r = check(d, iid)
    assert r.verdict is verdict and not r.kept
    assert [a.config for a in r.attempts] == ["runsc-rt2", "runsc-open", "runc-rt2"]
    assert [a.gvisor for a in r.attempts] == [True, True, False]


@pytest.mark.parametrize(
    ("status", "kept"),
    [
        ("t::fix=XFAIL", False),  # swebench counts XFAIL as a pass; a FAIL_TO_PASS test must really pass
        ("t::fix=SKIPPED", False),  # skipped never counts as a pass
        ("t::old=SKIPPED", False),
        ("t::old=XFAIL", True),  # a PASS_TO_PASS test the repository marks xfail: its unchanged state
    ],
)
def test_plant_a_skipped_or_xfailed_test_never_counts_as_a_pass(status: str, kept: bool) -> None:
    iid = "django__django-1"
    d = FakeDocker({(iid, c.name): f"STATUS {status}" for c in (PRIMARY, *DIAGNOSTICS)})
    assert check(d, iid).kept is kept


def test_plant_a_patch_that_does_not_apply_is_dropped_without_diagnostics() -> None:
    r = check(FakeDocker(apply_fails=True))
    assert r.verdict is Verdict.PATCH_DOES_NOT_APPLY and len(r.attempts) == 1


@pytest.mark.parametrize(
    ("kernel", "match"),
    [
        ({"runsc": "readable"}, "asked for runsc, but the kernel log the container read does not show"),  # a host log
        ({"runc": True}, "asked for runc, but the container's kernel log shows"),  # runc is not the host runtime
    ],
)
def test_plant_a_container_on_the_wrong_kernel_stops_the_check(kernel: dict[str, Kernel], match: str) -> None:
    iid = "django__django-1"
    d = FakeDocker({(iid, "runsc-rt2"): FAIL, (iid, "runsc-open"): FAIL}, kernel=kernel)
    with pytest.raises(RuntimeMismatchError, match=match):
        check(d, iid)
    assert sorted(d.removed) == sorted(d.containers)  # every container was removed on the way out


@pytest.mark.parametrize(
    ("docker", "why"),
    [
        (FakeDocker(gone_after="start"), "not running"),  # died right after start: the dmesg exec exits 1
        (FakeDocker(gone_after="start", gone="removed"), "docker inspect exited 1"),
        (FakeDocker(kernel={"runsc": False}), "read no kernel log"),  # running, but dmesg read nothing
    ],
)
def test_plant_a_runsc_kernel_probe_that_exits_1_is_this_instances_infra_not_a_stop(
    docker: FakeDocker, why: str
) -> None:
    """The integrator's call: under runsc, a container that dies right after start (the dmesg exec exits 1) is INFRA
    for that instance, never a RuntimeMismatchError for the whole run; only a log that reads and is not gVisor's stops
    it."""
    r = check(docker)
    assert r.verdict is Verdict.INFRA and why in r.attempts[0].infra, r.attempts[0]
    assert len(r.attempts) == 1 and sorted(docker.removed) == sorted(docker.containers)


def test_plant_a_container_that_dies_right_after_start_never_stops_the_whole_check(tmp_path: Path) -> None:
    """The VM itself is proven by the preflight probes; one container dying at start is that instance's INFRA, and
    every other instance still runs."""
    ids = ["django__django-1", "django__django-2"]
    d = FakeDocker(gone_after="start")
    report = run_check({i: row(i) for i in ids}, [image(i) for i in ids], d, FakeGrader(), tmp_path, run_id="t")
    assert report.verdicts == {"infra": 2} and len((tmp_path / "results.jsonl").read_text().splitlines()) == 2


def test_a_docker_failure_is_infra_not_a_verdict_on_the_instance() -> None:
    r = check(FakeDocker(create_fails=True))
    assert r.verdict is Verdict.INFRA and "no space left" in r.attempts[0].infra and not r.attempts[0].error


def test_every_created_container_is_removed() -> None:
    iid = "django__django-1"
    d = FakeDocker({(iid, "runsc-rt2"): FAIL})
    check(d, iid)
    assert sorted(d.removed) == sorted(d.containers)


def test_preflight_probes_both_runtimes_by_running_dmesg() -> None:
    d = FakeDocker()
    preflight(d, [image("django__django-1")])
    probes = [c for c in d.calls if c[1] == "run"]
    assert probes == [
        probe_args(PRIMARY, image("django__django-1")),
        probe_args(DIAGNOSTICS[1], image("django__django-1")),
    ]
    assert "--cap-drop=ALL" in probes[0] and "--network=none" in probes[0]


@pytest.mark.parametrize(
    ("docker", "error", "match"),
    [
        (FakeDocker(info_fails=True), PreflightError, "docker info failed .*permission denied"),
        (FakeDocker(runsc_registered=False), PreflightError, "runsc is not registered"),
        (FakeDocker(missing_images=["django__django-1"]), PreflightError, "never pulls"),
        (
            FakeDocker(probe_fails=True),
            PreflightError,
            r"the runsc probe did not run \(exit 125\): OCI runtime create failed",
        ),
        (FakeDocker(kernel={"runsc": False}), RuntimeMismatchError, "runsc is not running gVisor"),
        (FakeDocker(kernel={"runc": True}), RuntimeMismatchError, "runc is not the host runtime"),
    ],
)
def test_plant_a_vm_that_is_not_ready_fails_before_any_instance_runs(
    tmp_path: Path, docker: FakeDocker, error: type[Exception], match: str
) -> None:
    ids = ["django__django-1", "django__django-2"]
    with pytest.raises(error, match=match):
        run_check({i: row(i) for i in ids}, [image(i) for i in ids], docker, FakeGrader(), tmp_path, run_id="t")
    assert not any(c[1] == "create" for c in docker.calls)
    assert not (tmp_path / "results.jsonl").exists()


def test_run_check_writes_the_drop_log_and_flags_a_systemic_failure(tmp_path: Path) -> None:
    ids = ["django__django-1", "django__django-2", "sympy__sympy-3", "sympy__sympy-4"]
    fails = {(i, c): FAIL for i in ids[1:] for c in ("runsc-rt2", "runsc-open")}
    report = run_check(
        {i: row(i) for i in ids},
        [image(i) for i in ids],
        FakeDocker(fails),
        FakeGrader(),
        tmp_path,
        run_id="t",
        workers=2,
    )
    assert (report.instances, report.kept) == (4, 1)
    assert report.verdicts == {"kept": 1, "runsc_incompatible": 3}
    assert report.drop_pct_over_limit
    assert (tmp_path / "kept.txt").read_text() == "django__django-1\n"
    dropped = (tmp_path / "dropped.tsv").read_text().splitlines()
    assert len(dropped) == 3 and dropped[0].split("\t")[1] == "runsc_incompatible"
    assert len((tmp_path / "results.jsonl").read_text().splitlines()) == 4


def test_plant_a_row_paired_with_another_instances_image_is_refused() -> None:
    with pytest.raises(ValueError, match="paired with the image"):
        check_instance(
            row("django__django-1"), image("django__django-2"), FakeDocker(), FakeGrader(), timeout_s=1, run_id="t"
        )


def test_run_gold_reports_the_kernel_it_saw() -> None:
    a = run_gold(
        row("django__django-1"), image("django__django-1"), PRIMARY, FakeDocker(), FakeGrader(), timeout_s=1, run_id="t"
    )
    assert a.gvisor is True and a.patch_applied and a.resolved and a.harness_resolved


def test_classify_never_keeps_a_primary_failure() -> None:
    primary = Attempt(config="runsc-rt2", patch_applied=True, log_parsed=True, resolved=False)
    for diag in ([Attempt(config="runc-rt2", resolved=True)], [Attempt(config="runsc-open", resolved=True)], []):
        assert classify(primary, diag) is not Verdict.KEPT


def test_load_rows_checks_the_pin_and_refuses_duplicates(tmp_path: Path) -> None:
    from receipts.eval.hashing import sha256_file

    path = tmp_path / "rows.jsonl"
    path.write_text(json.dumps(row("django__django-1")) + "\n" + json.dumps(row("django__django-2")) + "\n")
    assert set(load_rows(path, expect_sha256=sha256_file(path))) == {"django__django-1", "django__django-2"}
    with pytest.raises(ValueError, match="sha256"):
        load_rows(path, expect_sha256="0" * 64)
    path.write_text(json.dumps(row("django__django-1")) + "\n" + json.dumps(row("django__django-1")) + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        load_rows(path)


def test_a_config_can_only_name_the_two_runtimes() -> None:
    with pytest.raises(ValueError):
        RunConfig.model_validate({"name": "x", "runtime": "kata", "network": False, "cap_drop_all": True})


# ---------------------------------------------------------------- blocker: every Docker step is checked


@pytest.mark.parametrize(
    ("step", "answer"),
    [
        ("write-patch", Completed(returncode=1, stderr="bash: /tmp/patch.diff: No space left on device")),
        ("write-patch", Completed(returncode=126, stderr="OCI runtime exec failed")),
        ("write-patch", Completed(returncode=-1, timed_out=True)),
        ("apply", Completed(returncode=-1, timed_out=True)),  # a timed-out apply is Docker's, not the patch's
        ("apply", Completed(returncode=137, timed_out=True)),  # killed at the timeout: still not the patch's
        ("apply", Completed(returncode=125, stderr="Error response from daemon: container is not running")),
        ("write-eval", Completed(returncode=1, stderr="bash: /eval.sh: Read-only file system")),
        ("write-eval", Completed(returncode=-1, timed_out=True)),
        ("eval", Completed(returncode=126, stderr="OCI runtime exec failed: exec failed")),
        ("eval", Completed(returncode=137 - 10, stderr="exec: bash: not found")),  # 127
        ("dmesg", Completed(returncode=125, stderr="Error response from daemon: container is restarting")),
        ("dmesg", Completed(returncode=-1, timed_out=True)),
    ],
)
def test_plant_a_docker_step_that_fails_after_the_container_starts_is_infra_never_a_drop(
    step: str, answer: Completed
) -> None:
    """The re-review's blocker: a lost patch or eval.sh write, or a timed-out apply, became a false
    PATCH_DOES_NOT_APPLY or GOLD_FAILS drop. Every one is INFRA (rerun), with the step named."""
    d = FakeDocker(step={step: answer})
    r = check(d)
    assert r.verdict is Verdict.INFRA, r
    assert r.attempts[0].infra and not r.kept
    assert len(r.attempts) == 1  # no diagnostics are run to explain a run that never finished
    assert sorted(d.removed) == sorted(d.containers)


def test_a_patch_that_really_does_not_apply_is_still_dropped_as_such() -> None:
    d = FakeDocker(apply_fails=True)
    r = check(d)
    assert r.verdict is Verdict.PATCH_DOES_NOT_APPLY and not r.attempts[0].infra
    # each failed attempt was read as the patch's only after Docker said the container was still running
    assert sum(1 for c in d.calls if c[1] == "inspect") == 3


@pytest.mark.parametrize(
    ("docker", "old_verdict"),
    [
        # the exit codes Docker really gives: 1 for exec on a container that is gone or not running ...
        (FakeDocker(gone_after="write-patch"), "patch_does_not_apply"),
        (FakeDocker(gone_after="write-patch", gone="removed"), "patch_does_not_apply"),
        (FakeDocker(gone_after="write-eval"), "needs_network_or_caps"),  # the eval exec exits 1, prints nothing
        (FakeDocker(gone_after="eval"), "kept"),  # died during the eval script: a cut-off log
        # ... and 255 when the CLI's connection to the daemon drops
        (FakeDocker(step={"eval": Completed(returncode=255, stderr="error during connect: EOF")}), "gold_fails"),
        (FakeDocker(step={"apply": Completed(returncode=255, stderr="error during connect: EOF")}), "kept"),
        # an inspect that cannot answer is never read as a running container
        (FakeDocker(apply_fails=True, inspect=Completed(returncode=255, stderr="EOF")), "patch_does_not_apply"),
        (FakeDocker(apply_fails=True, inspect=Completed(returncode=0, stdout="")), "patch_does_not_apply"),
    ],
)
def test_plant_a_container_docker_lost_is_infra_never_a_verdict(docker: FakeDocker, old_verdict: str) -> None:
    """The prior blocker, still open after round 2: docker's own exit codes (1 for a gone container, 255 for a lost
    daemon) were read as the command's, so a dead container became a false drop. `old_verdict` is what the round-2
    code said; every one is INFRA now, with no diagnostics, and every container is removed."""
    r = check(docker)
    assert r.verdict is Verdict.INFRA, (r.verdict, old_verdict)
    assert r.attempts[0].infra and len(r.attempts) == 1
    assert sorted(docker.removed) == sorted(docker.containers)


@pytest.mark.parametrize(
    ("step", "answer"),
    [
        ("apply", Completed(returncode=-9, stderr="")),  # the docker CLI killed by a signal, not by our timeout
        ("write-patch", Completed(returncode=-9)),
        ("dmesg", Completed(returncode=-15)),
        ("eval", Completed(returncode=-15, stdout="Ran 3 tests\nOK\n")),
    ],
)
def test_plant_a_negative_docker_exit_is_infra_even_without_a_timeout(step: str, answer: Completed) -> None:
    """R1: a negative exit with no timeout (the CLI killed by a signal) is Docker's, never the command's; every planted
    timeout used to carry timed_out too, so the negative check itself went untested."""
    d = FakeDocker(step={step: answer})
    r = check(d)
    assert r.verdict is Verdict.INFRA and f"exited {answer.returncode}" in r.attempts[0].infra, r.attempts[0]


def test_plant_a_new_check_never_appends_to_an_old_results_file(tmp_path: Path) -> None:
    """R9: a second run into the same --out would mix two runs' results in one results.jsonl."""
    ids = ["django__django-1"]
    run_check({i: row(i) for i in ids}, [image(i) for i in ids], FakeDocker(), FakeGrader(), tmp_path, run_id="a")
    before = (tmp_path / "results.jsonl").read_text()
    with pytest.raises(ValueError, match="already has results"):
        run_check({i: row(i) for i in ids}, [image(i) for i in ids], FakeDocker(), FakeGrader(), tmp_path, run_id="b")
    assert (tmp_path / "results.jsonl").read_text() == before


@pytest.mark.parametrize(("dropped", "over"), [(1, False), (2, True)])
def test_the_harness_checks_drop_limit_is_more_than_a_quarter(dropped: int, over: bool) -> None:
    """Exactly 25% dropped is within the limit; more is a systemic failure (exit 3)."""
    report = Report(run_id="t", instances=4, kept=4 - dropped, verdicts={}, dropped={}, max_drop_pct=25)
    assert report.drop_pct_over_limit is over


def test_plant_a_diagnostic_run_that_hits_infra_keeps_the_drop_reason_unread() -> None:
    iid = "django__django-1"
    d = FakeDocker({(iid, "runsc-rt2"): FAIL})
    calls = 0

    real = d.__call__

    def flaky(args: Sequence[str], *, timeout_s: float, stdin: str | None = None) -> Completed:
        nonlocal calls
        if args[1] == "create" and "-runc-rt2-" in args[args.index("--name") + 1]:
            calls += 1
            return Completed(returncode=1, stderr="Cannot connect to the Docker daemon")
        return real(args, timeout_s=timeout_s, stdin=stdin)

    r = check_instance(row(iid), image(iid), flaky, FakeGrader(), timeout_s=60, run_id="t")
    assert calls == 1 and r.verdict is Verdict.INFRA  # not GOLD_FAILS: the runc run never finished


def test_an_instance_whose_check_raises_is_recorded_as_infra_and_the_others_keep_their_results(
    tmp_path: Path,
) -> None:
    ids = ["django__django-1", "django__django-2"]

    class Exploding(FakeGrader):
        def grade(self, row: Mapping[str, Any], test_output: str) -> Grade:
            if row["instance_id"] == "django__django-2":
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            return super().grade(row, test_output)

    report = run_check(
        {i: row(i) for i in ids}, [image(i) for i in ids], FakeDocker(), Exploding(), tmp_path, run_id="t", workers=1
    )
    assert report.verdicts == {"infra": 1, "kept": 1}
    lines = (tmp_path / "results.jsonl").read_text().splitlines()
    assert len(lines) == 2  # written as each instance finished


def test_subprocess_docker_survives_bytes_that_are_not_utf8(tmp_path: Path) -> None:
    script = tmp_path / "noise.sh"
    script.write_bytes(b"#!/bin/sh\nprintf 'ok \\377\\376 done'\n")
    script.chmod(0o755)
    out = subprocess_docker([str(script)], timeout_s=30)
    assert out.returncode == 0 and out.stdout.startswith("ok ") and out.stdout.endswith(" done")


# ---------------------------------------------------------------- round 4: the harness's chain in contract v3's names


def test_the_harness_patch_chain_is_the_contracts_patch_methods_in_order() -> None:
    """REG-569: the label runs and the harness check apply a patch the way the SWE-bench harness does, and the names
    are contract v3's PatchMethod, in its order."""
    assert [m for m, _ in PATCH_CHAIN] == list(PatchMethod)
    assert [c for _, c in PATCH_CHAIN] == [
        "git apply --verbose",
        "git apply --verbose --reject",
        "patch --batch --fuzz=5 -p1 -i",
    ]


@pytest.mark.parametrize(
    ("failing", "method"),
    [
        ((), PatchMethod.GIT_APPLY),
        (("apply",), PatchMethod.GIT_APPLY_REJECT),
        (("apply", "apply-reject"), PatchMethod.GNU_PATCH_FUZZ),
        (("apply", "apply-reject", "apply-patch"), None),
    ],
)
def test_plant_the_harness_records_which_step_of_its_chain_applied_the_patch(
    failing: tuple[str, ...], method: PatchMethod | None
) -> None:
    d = FakeDocker(step={s: Completed(returncode=1, stderr="error: patch failed") for s in failing})
    r = check(d)
    assert r.attempts[0].patch_method is method
    assert r.verdict is (Verdict.KEPT if method else Verdict.PATCH_DOES_NOT_APPLY)
    # every step runs in the same container, on the tree the step before it left: no reset in between
    applied = [c[-1] for c in d.calls if c[1] == "exec" and FakeDocker._step(c[-1]).startswith("apply")]
    assert applied == [f"{cmd} /tmp/patch.diff" for _, cmd in PATCH_CHAIN][: len(failing) + (1 if method else 0)]
    assert len({FakeDocker._exec_name(c) for c in d.calls if c[1] == "exec"}) == 1
