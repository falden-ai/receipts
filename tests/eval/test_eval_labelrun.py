"""The label runner (REG-564): SWE-ABS's evaluator reproduced (chunk removal, the swapped test patch, the status-map
rule) with our two stricter checks, a stable plan from the manifest, resumable append-only results, and verdicts that
wait for gold. Docker and the harness are fakes: no container runs here."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pytest

from receipts.contracts import PatchMethod
from receipts.eval.hashing import sha256_text
from receipts.eval.images import ImageRef
from receipts.eval.labelrun import (
    GOLD,
    Job,
    JobKind,
    JobResult,
    JobStatus,
    LabelDrop,
    LabelResultsError,
    Material,
    PlanInputs,
    evidence,
    gold_passes,
    land_drops,
    materials,
    parse_label_drops,
    patch_passes,
    pending_infra,
    plan,
    read_results,
    remove_conflicting_chunks,
    run_job,
    run_labels,
    swe_abs_patch_text,
    swe_abs_rule,
)
from receipts.eval.rows import AbsRow, Row
from receipts.eval.runsc_check import GVISOR_BANNER, Completed, Grade, ResultLog

ABS_TEST = (
    "diff --git a/tests/test_x.py b/tests/test_x.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_x.py\n"
    "@@ -0,0 +1 @@\n+def test_strong(): assert f(2) == 4\n"
)
AGENT = (
    "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n"
    "diff --git a/tests/test_x.py b/tests/test_x.py\nnew file mode 100644\n--- /dev/null\n+++ b/tests/test_x.py\n"
    "@@ -0,0 +1 @@\n+def test_mine(): pass\n"
)


def test_conflicting_chunks_are_removed_the_way_swe_abs_removes_them() -> None:
    kept = remove_conflicting_chunks(AGENT, ABS_TEST)
    assert "src/x.py" in kept and "tests/test_x.py" not in kept
    unified = (
        "--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+b\n--- /dev/null\n+++ b/tests/test_x.py\n@@ -0,0 +1 @@\n+x\n"
    )
    assert "tests/test_x.py" not in remove_conflicting_chunks(unified, ABS_TEST)
    assert swe_abs_patch_text(AGENT, abs_row()).endswith("\n")
    two_files = AGENT.replace("tests/test_x.py", "src/y.py")  # nothing to remove: the graded text, byte for byte
    assert swe_abs_patch_text(two_files, abs_row()) == two_files


def test_the_swe_abs_rule_and_ours() -> None:
    gold = {"t1": "PASSED", "t2": "PASSED", "t3": "SKIPPED"}
    assert gold_passes(gold, True) and patch_passes(gold, True, gold)
    assert swe_abs_rule({}) is True  # theirs passes an empty map
    assert patch_passes({}, True, gold) is False  # ours: a run that reports no tests never passes
    assert patch_passes({}, True, {}) is False  # not even against a gold that reported none
    assert gold_passes({"t3": "SKIPPED"}, True) is False  # a gold with nothing PASSED is no oracle
    assert gold_passes(gold, False) is False


@pytest.mark.parametrize(
    "patch",
    [
        {"t1": "PASSED", "t3": "SKIPPED"},  # t2 passed on gold and disappeared here
        {"t1": "PASSED", "t2": "SKIPPED", "t3": "SKIPPED"},  # t2 passed on gold and was skipped here
        {"t1": "PASSED", "t2": "XFAIL", "t3": "SKIPPED"},  # SWE-ABS fails an XFAIL too
        {"t1": "PASSED", "t2": "FAILED", "t3": "SKIPPED"},
        {"t1": "PASSED", "t2": "PASSED", "t3": "ERROR"},
    ],
)
def test_plant_a_patch_run_that_must_fail(patch: dict[str, str]) -> None:
    gold = {"t1": "PASSED", "t2": "PASSED", "t3": "SKIPPED"}
    assert patch_passes(patch, True, gold) is False


def test_plant_an_xfail_never_counts_as_passing() -> None:
    """L2: with XFAIL among the passing statuses, a gold whose SWE-ABS test xfails became an oracle, and a patch whose
    test xfails passed. SWE-ABS's own rule fails an XFAIL too."""
    assert gold_passes({"t1": "PASSED", "t2": "XFAIL"}, True) is False
    gold = {"t1": "PASSED", "t3": "SKIPPED"}
    assert patch_passes({"t1": "PASSED", "t3": "XFAIL"}, True, gold) is False  # t3 never passed on gold
    assert patch_passes({"t1": "PASSED", "t4": "XFAIL"}, True, gold) is False  # a test gold never reported
    assert swe_abs_rule({"t1": "PASSED", "t2": "XFAIL"}) is False
    rows = {"django__django-1": verified()}
    g, pj = job(JobKind.SWE_ABS_GOLD), job(JobKind.SWE_ABS_PATCH)
    ev = evidence({g.id: res(g, JobStatus.DONE, {"t1": "PASSED", "t2": "XFAIL"}),
                   pj.id: res(pj, JobStatus.DONE, {"t1": "PASSED", "t2": "XFAIL"})}, rows)  # fmt: skip
    assert ev.gold == {"django__django-1": False}  # no oracle, so the record's label waits on NO_ABS_ORACLE


# ---------------------------------------------------------------- the plan

INPUTS = PlanInputs(
    pool=("django__django-1", "django__django-2", "sympy__sympy-3"),
    dropped=frozenset({"sympy__sympy-3"}),
    resolved=(
        ("e1", "django__django-1", "a" * 64),
        ("e1", "sympy__sympy-3", "b" * 64),
        ("e2", "django__django-2", "c" * 64),
    ),
    no_report=(("e2", "django__django-1", "d" * 64),),
    gold_sha256={"django__django-1": "1" * 64, "django__django-2": "2" * 64, "sympy__sympy-3": "3" * 64},
)


def test_the_plan_is_every_gold_resolved_patch_and_missing_report_minus_r24_drops() -> None:
    jobs = plan(INPUTS)
    assert [(j.kind, j.entry, j.instance_id) for j in jobs] == [
        (JobKind.SWE_ABS_GOLD, GOLD, "django__django-1"),
        (JobKind.SWE_ABS_GOLD, GOLD, "django__django-2"),
        (JobKind.SWE_ABS_PATCH, "e1", "django__django-1"),
        (JobKind.SWE_ABS_PATCH, "e2", "django__django-2"),
        (JobKind.F2P_PATCH, "e2", "django__django-1"),
    ]
    assert plan(INPUTS) == jobs  # stable
    assert all(j.instance_id != "sympy__sympy-3" for j in jobs)  # dropped by R24: no runs
    assert [j.id for j in plan(INPUTS, only=["django__django-2"])] == [
        "swe_abs_gold~gold~django__django-2",
        "swe_abs_patch~e2~django__django-2",
    ]


# ---------------------------------------------------------------- materials and runs

GOLD_PATCH = "diff --git a/src/x.py b/src/x.py\n--- a/src/x.py\n+++ b/src/x.py\n@@ -1 +1 @@\n-a\n+c\n"


def verified(iid: str = "django__django-1") -> Row:
    return Row(
        FAIL_TO_PASS=("t::fix",),
        PASS_TO_PASS=("t::old",),
        base_commit="a" * 40,
        created_at="2020-01-01T00:00:00Z",
        difficulty="<15 min fix",
        environment_setup_commit="b" * 40,
        hints_text="",
        instance_id=iid,
        patch=GOLD_PATCH,
        problem_statement="It breaks.",
        repo="django/django",
        test_patch="diff --git a/tests/test_orig.py b/tests/test_orig.py\n+orig\n",
        version="3.0",
    )


def abs_row(iid: str = "django__django-1") -> AbsRow:
    v = verified(iid)
    return AbsRow(
        FAIL_TO_PASS=v.FAIL_TO_PASS,
        PASS_TO_PASS=v.PASS_TO_PASS,
        base_commit=v.base_commit,
        instance_id=iid,
        original_test_patch=v.test_patch,
        patch=v.patch,
        repo=v.repo,
        test_patch=ABS_TEST,
        version=v.version,
    )


def image(iid: str = "django__django-1") -> ImageRef:
    return ImageRef(
        instance_id=iid,
        image=f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}",
        digest="sha256:" + "e" * 64,
        media_type="application/vnd.docker.distribution.manifest.v2+json",
    )


def job(kind: JobKind, entry: str = "e1", iid: str = "django__django-1", text: str = AGENT) -> Job:
    sha = sha256_text(GOLD_PATCH if kind is JobKind.SWE_ABS_GOLD else text)
    e = GOLD if kind is JobKind.SWE_ABS_GOLD else entry
    return Job(id=f"{kind.value}~{e}~{iid}", kind=kind, entry=e, instance_id=iid, patch_sha256=sha)


def material(kind: JobKind) -> Material:
    [m] = materials([job(kind)], {"django__django-1": verified()}, {"django__django-1": abs_row()},
                    {"django__django-1": image()}, lambda e, i: AGENT)  # fmt: skip
    return m


def test_materials_swap_in_the_swe_abs_test_patch_and_check_every_patch_hash() -> None:
    gold, mine, f2p = material(JobKind.SWE_ABS_GOLD), material(JobKind.SWE_ABS_PATCH), material(JobKind.F2P_PATCH)
    assert gold.harness_row["test_patch"] == ABS_TEST and gold.patch == GOLD_PATCH
    assert mine.harness_row["test_patch"] == ABS_TEST and "tests/test_x.py" not in mine.patch
    assert f2p.harness_row["test_patch"] == verified().test_patch and f2p.patch == AGENT  # SWE-bench's own run
    with pytest.raises(LabelResultsError, match="not the one the manifest pinned"):
        materials([job(JobKind.SWE_ABS_PATCH)], {"django__django-1": verified()}, {"django__django-1": abs_row()},
                  {"django__django-1": image()}, lambda e, i: AGENT + "+tampered\n")  # fmt: skip


class FakeDocker:
    """Like the runsc check's fake: `eval_out` is what the eval script prints, per call, in order; `steps` plants a
    Docker step's answer. A container can die once a step whose command contains `gone_after` has run: then `docker
    exec` on it exits 1, as Docker's CLI does for a container that is not running, and `docker inspect` says false."""

    def __init__(
        self,
        eval_out: Sequence[Completed] = (),
        *,
        apply_rc: int = 0,
        steps: Mapping[str, Completed] = {},
        gone_after: str | None = None,
    ):
        self.eval_out = list(eval_out)
        self.apply_rc = apply_rc
        self.steps = dict(steps)
        self.gone_after = gone_after
        self.created: list[str] = []
        self.removed: list[str] = []
        self.dead: set[str] = set()

    def __call__(self, args: Sequence[str], *, timeout_s: float, stdin: str | None = None) -> Completed:
        a = list(args)
        if a[1] == "create":
            self.created.append(a[a.index("--name") + 1])
            return Completed(returncode=0)
        if a[1] in ("start",):
            return Completed(returncode=0)
        if a[1] == "rm":
            self.removed.append(a[3])
            return Completed(returncode=0)
        if a[1] == "inspect":
            assert a[2:4] == ["-f", "{{.State.Running}}"]
            return Completed(returncode=0, stdout="false\n" if a[4] in self.dead else "true\n")
        name = next(x for x in a[2:] if x in self.created)
        if name in self.dead:
            return Completed(returncode=1, stderr=f"Error response from daemon: container {name} is not running")
        answer = self._exec(a)
        if self.gone_after is not None and self.gone_after in a[-1]:
            self.dead.add(name)
        return answer

    def _exec(self, a: list[str]) -> Completed:
        cmd = a[-1]
        for key, answer in self.steps.items():
            if key in cmd:
                return answer
        if cmd == "dmesg":
            return Completed(returncode=0, stdout=f"[0.0] {GVISOR_BANNER}\n")
        if "git apply" in cmd or "patch --batch" in cmd:
            return Completed(returncode=self.apply_rc)
        if cmd.startswith("cat >"):
            return Completed(returncode=0)
        if "/eval.sh" in cmd:
            return self.eval_out.pop(0) if len(self.eval_out) > 1 else (self.eval_out or [Completed(returncode=0)])[0]
        raise AssertionError(a)


def out(statuses: dict[str, str]) -> Completed:
    return Completed(returncode=0, stdout="STATUSES " + json.dumps(statuses))


class FakeGrader:
    def eval_script(self, row: Mapping[str, Any]) -> str:
        return f"run the tests of {row['test_patch'][:20]}"

    def grade(self, row: Mapping[str, Any], test_output: str) -> Grade:
        found = test_output.startswith("STATUSES ")
        st = json.loads(test_output.removeprefix("STATUSES ")) if found else {}
        return Grade(log_found=found, harness_resolved=False, statuses=st, fail_to_pass=(), pass_to_pass=())


def test_a_job_records_the_harness_status_map() -> None:
    d = FakeDocker([out({"tests/test_x.py::test_strong": "PASSED"})])
    r = run_job(material(JobKind.SWE_ABS_PATCH), d, FakeGrader(), run_id="d1", timeout_s=60)
    assert (r.status, r.log_found, r.statuses, r.gvisor) == (
        JobStatus.DONE,
        True,
        {"tests/test_x.py::test_strong": "PASSED"},
        True,
    )
    assert d.removed == d.created


@pytest.mark.parametrize(
    ("docker", "status", "attempts"),
    [
        (FakeDocker(apply_rc=1), JobStatus.PATCH_DOES_NOT_APPLY, 1),
        (FakeDocker([Completed(returncode=-1, timed_out=True)] * 2), JobStatus.TIMEOUT, 2),
        (FakeDocker([Completed(returncode=-1, timed_out=True), out({"t": "PASSED"})]), JobStatus.DONE, 2),
        (
            FakeDocker(steps={"cat > /tmp/patch.diff": Completed(returncode=1, stderr="no space left")}),
            JobStatus.INFRA,
            1,
        ),
        (FakeDocker(steps={"cat > /eval.sh": Completed(returncode=126)}), JobStatus.INFRA, 1),
        (FakeDocker([Completed(returncode=125, stderr="container is not running")]), JobStatus.INFRA, 1),
        (FakeDocker([Completed(returncode=255, stderr="error during connect: EOF")]), JobStatus.INFRA, 1),
        (FakeDocker(steps={"git apply --verbose /": Completed(returncode=255)}), JobStatus.INFRA, 1),
    ],
)
def test_plant_each_way_a_job_can_end(docker: FakeDocker, status: JobStatus, attempts: int) -> None:
    r = run_job(material(JobKind.SWE_ABS_PATCH), docker, FakeGrader(), run_id="d1", timeout_s=60)
    assert (r.status, r.attempts) == (status, attempts)
    assert docker.removed == docker.created


@pytest.mark.parametrize(
    ("gone_after", "old_status"),
    [
        (
            "cat > /tmp/patch.diff",
            "patch_does_not_apply",
        ),  # every apply then exits 1, Docker's code for a gone container
        ("cat > /eval.sh", "done"),  # the eval exec exits 1 and prints nothing: log_found False
        ("/eval.sh 2>&1", "done"),  # the container died during the eval script: a cut-off log
    ],
)
def test_plant_a_gold_job_whose_container_died_never_ends_final_with_no_oracle(
    tmp_path: Path, gone_after: str, old_status: str
) -> None:
    """The prior blocker on the label runner's path: a gold job whose container died ended final (the round-2 code
    said `old_status`), gold's oracle was False, every resolved record on the instance became NO_ABS_ORACLE, and the
    job was never run again. Now it is INFRA: not final, no oracle verdict, rerun by the next `labels`."""
    rows = {"django__django-1": verified()}
    gold = material(JobKind.SWE_ABS_GOLD)
    d = FakeDocker([out({"tests/test_x.py::test_strong": "PASSED"})], gone_after=gone_after)
    r = run_job(gold, d, FakeGrader(), run_id="d1", timeout_s=60)
    assert r.status is JobStatus.INFRA and "not running" in r.detail, (r.status, old_status, r.detail)
    assert read_results([r.model_dump_json()]) == {}  # never final
    assert evidence(read_results([r.model_dump_json()]), rows).gold == {}  # no oracle verdict, False or otherwise
    log = ResultLog(tmp_path / "results.jsonl")
    log.append(r)
    again = run_labels([gold], FakeDocker([out({"tests/test_x.py::test_strong": "PASSED"})]), FakeGrader(), log,
                       run_id="d2", workers=1)  # fmt: skip
    assert (again.already_final, again.ran, again.statuses) == (0, 1, {"done": 1})
    assert evidence(read_results(log.lines()), rows).gold == {"django__django-1": True}


def test_runs_resume_skip_final_jobs_and_retry_infra(tmp_path: Path) -> None:
    log = ResultLog(tmp_path / "results.jsonl")
    work = [material(JobKind.SWE_ABS_GOLD), material(JobKind.SWE_ABS_PATCH)]
    broken = FakeDocker(steps={"cat > /eval.sh": Completed(returncode=1, stderr="read-only file system")})
    first = run_labels(work, broken, FakeGrader(), log, run_id="d1", workers=1)
    assert first.statuses == {"infra": 2} and len(first.infra) == 2
    fixed = FakeDocker([out({"tests/test_x.py::test_strong": "PASSED"})])
    second = run_labels(work, fixed, FakeGrader(), log, run_id="d2", workers=1)
    assert (second.already_final, second.ran, second.statuses) == (0, 2, {"done": 2})
    third = run_labels(work, fixed, FakeGrader(), log, run_id="d3", workers=1)
    assert (third.already_final, third.ran) == (2, 0)  # nothing final is ever run again
    assert len(log.lines()) == 4


def test_an_exception_in_one_job_is_its_infra_and_the_others_finish(tmp_path: Path) -> None:
    class Exploding(FakeGrader):
        def grade(self, row: Mapping[str, Any], test_output: str) -> Grade:
            if "tests/test_x" in row["test_patch"] and row["patch"] == GOLD_PATCH and "STATUSES" in test_output:
                raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")
            return super().grade(row, test_output)

    log = ResultLog(tmp_path / "results.jsonl")
    work = [material(JobKind.SWE_ABS_PATCH), material(JobKind.F2P_PATCH)]
    summary = run_labels(work, FakeDocker([out({"t": "PASSED"})]), Exploding(), log, run_id="d1", workers=1)
    assert summary.ran == 2 and summary.statuses.get("infra") == 1


# ---------------------------------------------------------------- reading results and making evidence


def res(j: Job, status: JobStatus, statuses: dict[str, str] | None = None, *, log_found: bool = True) -> JobResult:
    return JobResult(run_id="d1", job=j, image=image().ref, status=status, log_found=log_found, statuses=statuses or {})


def test_read_results_keeps_the_first_final_and_refuses_disagreement_or_strangers() -> None:
    g = job(JobKind.SWE_ABS_GOLD)
    lines = [res(g, JobStatus.INFRA).model_dump_json(), res(g, JobStatus.DONE, {"t": "PASSED"}).model_dump_json()]
    assert read_results(lines)[g.id].status is JobStatus.DONE
    with pytest.raises(LabelResultsError, match="two final results that disagree"):
        read_results([*lines, res(g, JobStatus.DONE, {"t": "FAILED"}).model_dump_json()])
    with pytest.raises(LabelResultsError, match="not a planned job"):
        read_results(lines, {})
    other = g.model_copy(update={"patch_sha256": "0" * 64})
    with pytest.raises(LabelResultsError, match="not a planned job"):
        read_results(lines, {other.id: other})


def test_evidence_waits_for_gold_and_follows_it() -> None:
    rows = {"django__django-1": verified()}
    g, p = job(JobKind.SWE_ABS_GOLD), job(JobKind.SWE_ABS_PATCH)
    gold_ok = res(g, JobStatus.DONE, {"t1": "PASSED", "t2": "PASSED"})
    patch_bad = res(p, JobStatus.DONE, {"t1": "PASSED"})  # t2 vanished
    assert evidence({p.id: patch_bad}, rows).records == {}  # no gold yet: no verdict
    ev = evidence({g.id: gold_ok, p.id: patch_bad}, rows)
    assert ev.gold == {"django__django-1": True}
    rec = ev.records["e1~django__django-1"]
    assert (rec.swe_abs_passed, rec.swe_abs_rule, rec.patch_applies) == (False, True, True)  # ours is stricter
    gold_bad = res(g, JobStatus.DONE, {"t1": "FAILED"})
    ev = evidence({g.id: gold_bad, p.id: patch_bad}, rows)
    assert ev.gold == {"django__django-1": False} and ev.records["e1~django__django-1"].swe_abs_passed is None
    timed_out = evidence({g.id: gold_ok, p.id: res(p, JobStatus.TIMEOUT)}, rows)
    assert timed_out.records["e1~django__django-1"].swe_abs_passed is False
    not_applied = evidence({p.id: res(p, JobStatus.PATCH_DOES_NOT_APPLY)}, rows)
    assert not_applied.records["e1~django__django-1"].patch_applies is False
    no_log = evidence({g.id: gold_ok, p.id: res(p, JobStatus.DONE, {}, log_found=False)}, rows)
    rec = no_log.records["e1~django__django-1"]
    assert (rec.swe_abs_passed, rec.swe_abs_rule) == (False, True)  # theirs passes an empty map; ours never does


def test_evidence_from_a_fail_to_pass_run() -> None:
    rows = {"django__django-1": verified()}
    f = job(JobKind.F2P_PATCH)
    bad = evidence({f.id: res(f, JobStatus.DONE, {"t::fix": "FAILED", "t::old": "PASSED"})}, rows)
    assert (
        bad.records["e1~django__django-1"].fail_to_pass_passed,
        bad.records["e1~django__django-1"].pass_to_pass_passed,
    ) == (False, True)
    regression = evidence({f.id: res(f, JobStatus.DONE, {"t::fix": "PASSED", "t::old": "ERROR"})}, rows)
    assert regression.records["e1~django__django-1"].fail_to_pass_passed is True
    assert regression.records["e1~django__django-1"].pass_to_pass_passed is False
    skipped = evidence({f.id: res(f, JobStatus.DONE, {"t::fix": "SKIPPED", "t::old": "PASSED"})}, rows)
    assert skipped.records["e1~django__django-1"].fail_to_pass_passed is False  # a skip never passes


# ---------------------------------------------------------------- round 4: the patch method, the founder's label drop


def test_a_job_records_which_step_of_the_harness_chain_applied_its_patch() -> None:
    """REG-569: a label run's result says how its patch applied, in contract v3's names."""
    refused = {
        "git apply --verbose /": Completed(returncode=1),
        "git apply --verbose --reject": Completed(returncode=1),
    }
    fuzz = FakeDocker([out({"t": "PASSED"})], steps=refused)
    r = run_job(material(JobKind.SWE_ABS_PATCH), fuzz, FakeGrader(), run_id="d1", timeout_s=60)
    assert (r.status, r.patch_method) == (JobStatus.DONE, PatchMethod.GNU_PATCH_FUZZ)
    plain = run_job(material(JobKind.SWE_ABS_PATCH), FakeDocker([out({"t": "PASSED"})]), FakeGrader(), run_id="d1",
                    timeout_s=60)  # fmt: skip
    assert plain.patch_method is PatchMethod.GIT_APPLY
    none = run_job(material(JobKind.SWE_ABS_PATCH), FakeDocker(apply_rc=1), FakeGrader(), run_id="d1", timeout_s=60)
    assert (none.status, none.patch_method) == (JobStatus.PATCH_DOES_NOT_APPLY, None)
    assert JobResult.model_validate_json(r.model_dump_json()).patch_method is PatchMethod.GNU_PATCH_FUZZ


def drop_jobs() -> list[Job]:
    """django__django-1's plan: its gold job, two resolved patches (e1, e2) and a FAIL_TO_PASS run (e3); then
    django__django-2's gold job."""
    return [
        job(JobKind.SWE_ABS_GOLD),
        job(JobKind.SWE_ABS_PATCH, "e1"),
        job(JobKind.SWE_ABS_PATCH, "e2"),
        job(JobKind.F2P_PATCH, "e3"),
        job(JobKind.SWE_ABS_GOLD, iid="django__django-2"),
    ]


def as_lines(*results: JobResult) -> list[str]:
    return [r.model_dump_json() for r in results]


def test_pending_infra_is_every_unfinished_job_that_ran_into_infra_every_time() -> None:
    g, p1, _p2, f, _g2 = drop_jobs()
    lines = as_lines(res(g, JobStatus.INFRA), res(g, JobStatus.INFRA), res(p1, JobStatus.INFRA),
                     res(p1, JobStatus.DONE, {"t": "PASSED"}), res(f, JobStatus.INFRA))  # fmt: skip
    jobs = {j.id: j for j in drop_jobs()}
    assert sorted(pending_infra(lines, jobs)) == sorted([g.id, f.id])  # p1 finished; p2 never ran
    with pytest.raises(LabelResultsError, match="not a planned job"):
        pending_infra(lines, {})


KEPT = {"django__django-1": True, "django__django-2": True}
REASON = "gVisor kills the sandbox at the 4g cap on every run (founder, 10/28)"


def test_plant_a_label_drop_lands_only_where_every_unfinished_job_was_infra_on_every_run() -> None:
    """Persistent INFRA's exit path (round-3 review): a label job that is INFRA on every run stays non-final forever and
    the manifest never freezes. The founder's drop lands only on an instance R24 kept, every one of whose unfinished
    label jobs ran and hit INFRA every time, and it names exactly the records whose label waited on those jobs."""
    jobs = drop_jobs()
    g, p1, p2, f, g2 = jobs
    by = {j.id: j for j in jobs}
    lines = as_lines(res(g, JobStatus.INFRA), res(g, JobStatus.INFRA), res(p1, JobStatus.DONE, {"t": "PASSED"}),
                     res(p2, JobStatus.PATCH_DOES_NOT_APPLY), res(f, JobStatus.INFRA),
                     res(g2, JobStatus.DONE, {"t": "PASSED"}))  # fmt: skip
    final = read_results(lines, by)
    [landed] = land_drops([LabelDrop(instance_id="django__django-1", reason=REASON)], jobs, final,
                          pending_infra(lines, by), KEPT)  # fmt: skip
    assert landed.jobs == tuple(sorted([g.id, f.id])) and landed.reason == REASON
    # gold's run holds up e1's verdict and the FAIL_TO_PASS run holds up e3's; e2 did not apply, so it stays UNUSABLE
    assert landed.records == ("e1~django__django-1", "e3~django__django-1")
    rows = {"django__django-1": verified(), "django__django-2": verified("django__django-2")}
    assert evidence(final, rows, [landed]).dropped == dict.fromkeys(landed.records, REASON)
    assert evidence(final, rows).dropped == {}  # nothing is dropped without the founder's drop

    def refused(drops: list[LabelDrop], match: str, *, gold: Mapping[str, bool | None] = KEPT,
                text: list[str] = lines) -> None:  # fmt: skip
        with pytest.raises(LabelResultsError, match=match):
            land_drops(drops, jobs, read_results(text, by), pending_infra(text, by), gold)

    d = LabelDrop(instance_id="django__django-1", reason="x")
    refused([d, d], "appears twice")
    refused([LabelDrop(instance_id="sympy__sympy-9", reason="x")], "no label job is planned on it")
    refused([LabelDrop(instance_id="django__django-2", reason="x")], "every label job on it has a final result")
    refused([d], "R24 already dropped its gold", gold={"django__django-1": False})
    refused([d], "R24 has not checked its gold", gold={"django__django-1": None})
    refused([d], "has not run yet", text=[x for x in lines if f.id not in x])  # the FAIL_TO_PASS job never ran


def test_label_drops_parse_as_instance_equals_reason() -> None:
    assert parse_label_drops(["django__django-1=it dies = every run"]) == [
        LabelDrop(instance_id="django__django-1", reason="it dies = every run")
    ]
    for bad in ("django__django-1", "=reason", "django__django-1=  "):
        with pytest.raises(LabelResultsError, match="want <instance_id>=<reason>"):
            parse_label_drops([bad])
