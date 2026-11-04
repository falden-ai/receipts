"""The label runner (REG-564): our own harness runs that make Tier A labels final, on the eval VM under gVisor.

Three kinds of job, each one harness run in a fresh container (`receipts.eval.runsc_check.run_in_container`: runsc,
the contract's limits, no network, every Docker step checked):
- `swe_abs_gold`: the SWE-ABS tests on the instance's gold patch, once per pool instance R24 has not dropped. Gold
  must pass them here before they can label anyone's patch (NO_ABS_ORACLE otherwise).
- `swe_abs_patch`: the SWE-ABS tests on each resolved agent patch (GOOD or PLAUSIBLE_WRONG).
- `f2p_patch`: SWE-bench's own FAIL_TO_PASS and PASS_TO_PASS on an unresolved patch that has no published report (BAD
  or REGRESSION_ONLY).

The SWE-ABS run is SWE-ABS's own evaluator, reproduced (OpenAgentEval/SWE-ABS at 26b4c736,
swe-bench/swebench/runtest/run_evaluation_test.py): the eval script is the harness's with the SWE-ABS test patch in
place of SWE-bench's; an agent patch first loses every file chunk that the SWE-ABS test patch also touches
(`remove_conflicting_chunks`); the run is graded from the harness's status map of every test the SWE-ABS test files
report. SWE-ABS fails a run when any test is neither OK, PASSED nor SKIPPED. Our rule adds two things, both in the
direction of failing: a run that reports no tests never passes, and a test that passed on gold must pass on the patch
(a test that disappears, is skipped or xfails there fails it). Both verdicts are kept, so a disagreement with
SWE-ABS's published verdict can be counted.

Jobs are a pure function of the manifest, so the plan is the same on every machine. Results go to an append-only
results.jsonl as each job finishes; a job with a final result is never run again, so a stopped run resumes where it
stopped, and `--only` reruns one instance's unfinished jobs. INFRA (Docker could not finish a step) is never final. A
timeout is final after it repeats in a fresh container (the harness counts a timeout as a failure). Verdicts are
computed when the results are merged into the manifest, because a patch's verdict needs gold's status map.

A job that is INFRA on every run would keep the manifest from freezing forever. Its one way out is the founder's:
`manifest --label-drop <instance>=<reason>` (`land_drops`), which lands only on an instance R24 kept whose unfinished
jobs have all run into INFRA, and marks DROPPED (never sampled) exactly the records whose label waited on them. The
manifest records the reason, the jobs and the records. Nothing drops a label on its own.
"""

from __future__ import annotations

import re
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from enum import StrEnum
from pathlib import Path
from typing import Any, Literal

from pydantic import Field

from receipts.contracts import Frozen, PatchMethod
from receipts.eval.hashing import SHA256, sha256_text
from receipts.eval.images import ImageRef
from receipts.eval.rows import AbsRow, Row
from receipts.eval.runsc_check import (
    PRIMARY,
    ContainerRun,
    Docker,
    Grader,
    ResultLog,
    RuntimeMismatchError,
    run_in_container,
)

RESULT_FORMAT: Literal["receipts.eval.labelrun/1"] = "receipts.eval.labelrun/1"
GOLD = "gold"


class JobKind(StrEnum):
    SWE_ABS_GOLD = "swe_abs_gold"
    SWE_ABS_PATCH = "swe_abs_patch"
    F2P_PATCH = "f2p_patch"


class Job(Frozen):
    id: str
    kind: JobKind
    entry: str  # "gold" for a gold job
    instance_id: str
    patch_sha256: str = Field(pattern=SHA256)  # the patch text the job applies, before any chunk is removed


def job_id(kind: JobKind, entry: str, instance_id: str) -> str:
    return f"{kind.value}~{entry}~{instance_id}"


class JobStatus(StrEnum):
    DONE = "done"  # the eval script ran; the status map is the harness's reading of it
    PATCH_DOES_NOT_APPLY = "patch_does_not_apply"
    TIMEOUT = "timeout"  # twice, in fresh containers
    INFRA = "infra"  # Docker could not finish a step: never final


FINAL = frozenset({JobStatus.DONE, JobStatus.PATCH_DOES_NOT_APPLY, JobStatus.TIMEOUT})


class JobResult(Frozen):
    format: Literal["receipts.eval.labelrun/1"] = RESULT_FORMAT
    run_id: str = Field(min_length=1)
    job: Job
    image: str
    status: JobStatus
    gvisor: bool | None = None
    log_found: bool = False
    statuses: dict[str, str] = Field(default_factory=dict)  # test -> PASSED, FAILED, SKIPPED, ERROR or XFAIL
    seconds: float = Field(default=0.0, ge=0)
    attempts: int = Field(default=1, ge=1)
    detail: str = ""
    patch_method: PatchMethod | None = None  # the harness's step that applied the patch (REG-569)


# ---------------------------------------------------------------- the SWE-ABS run, reproduced


def apply_files(code_patch: str) -> list[str]:
    """SWE-ABS's get_apply_files: the a/ path of every `diff --git` header."""
    out = []
    for line in code_patch.splitlines():
        if line.startswith("diff --git"):
            parts = line.strip().split()
            out.append(parts[2][2:])
    return out


def remove_conflicting_chunks(model_patch: str, model_test_patch: str) -> str:
    """SWE-ABS's remove_conflicting_chunks (swe-bench/swebench/harness/utils.py at 26b4c736): drop every file chunk of
    the agent patch that touches a file the SWE-ABS test patch also touches, so the test patch applies cleanly."""
    test_files = apply_files(model_test_patch)
    chunks: list[str] = []
    if "diff --git a/" in model_patch:
        for m in re.finditer(r"(diff --git a/(?P<file>.+?) b/.+?)(?=diff --git a/|\Z)", model_patch, flags=re.DOTALL):
            if m.group("file") not in test_files:
                chunks.append(m.group(0))
    else:
        pattern = r"(--- (?P<old>.+?)\n\+\+\+ (?P<new>.+?)\n.+?)(?=\n--- |\Z)"
        for m in re.finditer(pattern, model_patch, flags=re.DOTALL):
            old, new = m.group("old"), m.group("new")
            path = new if old == "/dev/null" else old
            if path.startswith(("a/", "b/")):
                path = path[2:]
            if path not in test_files:
                chunks.append(m.group(0))
    return "\n".join(chunks)


def swe_abs_patch_text(model_patch: str, abs_row: AbsRow) -> str:
    """The agent patch as SWE-ABS applies it: conflicting chunks removed, ending in a newline. When no chunk is
    removed it is the patch exactly as the harness graded it (SWE-ABS's rejoin would add a blank line between file
    diffs, which changes nothing it applies but is not the graded text)."""
    if not set(apply_files(model_patch)) & set(apply_files(abs_row.test_patch)):
        text = model_patch
    else:
        text = remove_conflicting_chunks(model_patch, abs_row.test_patch)
    return text if text.endswith("\n") else text + "\n"


PASSING = frozenset(
    {"PASSED", "SKIPPED"}
)  # SWE-ABS's analyze_eval_status_map also accepts "OK"; the parsers say PASSED


def swe_abs_rule(statuses: Mapping[str, str]) -> bool:
    """SWE-ABS's own verdict: no test is anything but OK, PASSED or SKIPPED (an empty map passes, as theirs does)."""
    return all(s.upper() in ("OK", "PASSED", "SKIPPED") for s in statuses.values())


def gold_passes(statuses: Mapping[str, str], log_found: bool) -> bool:
    """Gold is a usable oracle: the log parsed, at least one test PASSED, and none failed, errored or xfailed."""
    return log_found and any(s == "PASSED" for s in statuses.values()) and all(s in PASSING for s in statuses.values())


def patch_passes(statuses: Mapping[str, str], log_found: bool, gold: Mapping[str, str]) -> bool:
    """Our verdict: SWE-ABS's rule, plus every test that PASSED on gold PASSED here and the run reported tests."""
    if not log_found or not statuses:
        return False
    if any(s not in PASSING for s in statuses.values()):
        return False
    return all(statuses.get(t) == "PASSED" for t, s in gold.items() if s == "PASSED")


# ---------------------------------------------------------------- the plan


class PlanInputs(Frozen):
    """What a plan is built from, by instance id."""

    pool: tuple[str, ...]
    dropped: frozenset[str]  # instances R24 dropped: no runs at all
    resolved: tuple[tuple[str, str, str], ...]  # (entry, instance, patch sha256) of every resolved non-empty patch
    no_report: tuple[tuple[str, str, str], ...]  # the same for unresolved patches with no published report
    gold_sha256: dict[str, str]  # instance -> sha256 of its gold patch


def plan(inputs: PlanInputs, *, only: Iterable[str] | None = None) -> list[Job]:
    """Every job the manifest calls for, in a stable order; `only` keeps one set of instances."""
    keep = None if only is None else set(only)
    jobs: list[Job] = []

    def add(kind: JobKind, entry: str, iid: str, sha: str) -> None:
        if iid in inputs.dropped or (keep is not None and iid not in keep):
            return
        jobs.append(Job(id=job_id(kind, entry, iid), kind=kind, entry=entry, instance_id=iid, patch_sha256=sha))

    for iid in sorted(inputs.pool):
        add(JobKind.SWE_ABS_GOLD, GOLD, iid, inputs.gold_sha256[iid])
    for entry, iid, sha in sorted(inputs.resolved):
        add(JobKind.SWE_ABS_PATCH, entry, iid, sha)
    for entry, iid, sha in sorted(inputs.no_report):
        add(JobKind.F2P_PATCH, entry, iid, sha)
    ids = [j.id for j in jobs]
    if len(ids) != len(set(ids)):
        raise ValueError("two jobs share an id")
    return jobs


class LabelResultsError(ValueError):
    """A results file that cannot be trusted."""


def read_results(lines: Iterable[str], jobs: Mapping[str, Job] | None = None) -> dict[str, JobResult]:
    """The first final result of every job, by job id. Later INFRA lines are ignored; two final results for one job
    must agree on everything a verdict reads, or the file is refused. With `jobs`, every line must be a planned job
    (same patch hash): a result for anything else is refused."""
    final: dict[str, JobResult] = {}
    for n, line in enumerate(lines, 1):
        r = JobResult.model_validate_json(line)
        if jobs is not None:
            want = jobs.get(r.job.id)
            if want is None or want != r.job:
                raise LabelResultsError(f"line {n}: {r.job.id} is not a planned job (or its patch hash differs)")
        if r.status not in FINAL:
            continue
        seen = final.get(r.job.id)
        if seen is None:
            final[r.job.id] = r
        elif (seen.status, seen.log_found, seen.statuses) != (r.status, r.log_found, r.statuses):
            raise LabelResultsError(f"line {n}: {r.job.id} has two final results that disagree")
    return final


# ---------------------------------------------------------------- the founder's way out of an INFRA that persists


def pending_infra(lines: Iterable[str], jobs: Mapping[str, Job]) -> dict[str, JobResult]:
    """Every planned job that has no final result and has run, every time, into INFRA: its last result. Every line must
    be a planned job (same patch hash), as in read_results."""
    final: set[str] = set()
    infra: dict[str, JobResult] = {}
    for n, line in enumerate(lines, 1):
        r = JobResult.model_validate_json(line)
        want = jobs.get(r.job.id)
        if want is None or want != r.job:
            raise LabelResultsError(f"line {n}: {r.job.id} is not a planned job (or its patch hash differs)")
        if r.status in FINAL:
            final.add(r.job.id)
        else:
            infra[r.job.id] = r
    return {k: v for k, v in sorted(infra.items()) if k not in final}


class LabelDrop(Frozen):
    """The founder's override (`manifest --label-drop <instance>=<reason>`): stop waiting on an instance's label runs
    that keep failing on infrastructure. Never automatic."""

    instance_id: str = Field(min_length=1)
    reason: str = Field(min_length=1)


def parse_label_drops(values: Iterable[str]) -> list[LabelDrop]:
    """`<instance_id>=<reason>` per --label-drop; an instance id never holds '='."""
    out = []
    for v in values:
        iid, sep, reason = v.partition("=")
        if not sep or not iid.strip() or not reason.strip():
            raise LabelResultsError(f"--label-drop {v!r}: want <instance_id>=<reason>")
        out.append(LabelDrop(instance_id=iid.strip(), reason=reason.strip()))
    return out


class LandedLabelDrop(Frozen):
    """A label drop as the manifest records it: the founder's reason, the jobs that were INFRA on every run, and the
    records whose label waited on them, which are DROPPED (never sampled). Records on the instance whose label was
    already final keep it."""

    instance_id: str
    reason: str = Field(min_length=1)
    jobs: tuple[str, ...] = Field(min_length=1)
    records: tuple[str, ...]  # "<entry>~<instance>"


def land_drops(
    drops: Sequence[LabelDrop],
    jobs: Sequence[Job],
    final: Mapping[str, JobResult],
    pending: Mapping[str, JobResult],
    gold_runsc: Mapping[str, bool | None],
) -> tuple[LandedLabelDrop, ...]:
    """Each drop lands only on an instance R24 kept, every one of whose unfinished label jobs has run and been INFRA
    every time; anything else is refused. A gold job that never finished holds up every resolved record on its
    instance (their SWE-ABS verdict needs gold's), and a patch job holds up its own record."""
    by_instance: dict[str, list[Job]] = {}
    for j in jobs:
        by_instance.setdefault(j.instance_id, []).append(j)
    seen: set[str] = set()
    out = []
    for d in drops:
        iid = d.instance_id
        if iid not in by_instance:
            raise LabelResultsError(f"label drop {iid}: no label job is planned on it (not a pool instance?)")
        if iid in seen:
            raise LabelResultsError(f"label drop {iid}: appears twice")
        seen.add(iid)
        kept = gold_runsc.get(iid)
        if kept is not True:
            raise LabelResultsError(
                f"label drop {iid}: "
                + (
                    "R24 already dropped its gold, so its records need no label run"
                    if kept is False
                    else "R24 has not checked its gold yet; settle R24 first (a rerun, or --gold-drop)"
                )
            )
        unfinished = [j for j in by_instance[iid] if j.id not in final]
        if not unfinished:
            raise LabelResultsError(f"label drop {iid}: every label job on it has a final result; nothing to drop")
        never = [j.id for j in unfinished if j.id not in pending]
        if never:
            raise LabelResultsError(
                f"label drop {iid}: {never[0]} has not run yet; run `labels` again first (a drop lands only on jobs "
                "that ran into INFRA every time)"
            )
        records: set[str] = set()
        for j in unfinished:
            if j.kind is JobKind.SWE_ABS_GOLD:
                records.update(
                    f"{p.entry}~{iid}"
                    for p in by_instance[iid]
                    if p.kind is JobKind.SWE_ABS_PATCH
                    and (p.id not in final or final[p.id].status is not JobStatus.PATCH_DOES_NOT_APPLY)
                )
            else:
                records.add(f"{j.entry}~{iid}")
        out.append(
            LandedLabelDrop(
                instance_id=iid,
                reason=d.reason,
                jobs=tuple(sorted(j.id for j in unfinished)),
                records=tuple(sorted(records)),
            )
        )
    return tuple(sorted(out, key=lambda x: x.instance_id))


# ---------------------------------------------------------------- running on the VM


class Material(Frozen):
    """What one job needs, resolved from the pinned inputs."""

    job: Job
    image: ImageRef
    harness_row: dict[str, Any]  # the row the grader builds the eval script from (test patch swapped for SWE-ABS)
    patch: str  # what is applied, after SWE-ABS's chunk removal for a patch job


def materials(
    jobs: Sequence[Job],
    rows: Mapping[str, Row],
    abs_rows: Mapping[str, AbsRow],
    images: Mapping[str, ImageRef],
    patches: Callable[[str, str], str],
) -> list[Material]:
    """Resolve every job's image, row and patch text, checking each patch against the hash the plan recorded."""
    out = []
    for job in jobs:
        row = rows[job.instance_id]
        if job.kind is JobKind.F2P_PATCH:
            text, harness_row = patches(job.entry, job.instance_id), row.harness_dict()
            applied = text
        else:
            abs_row = abs_rows[job.instance_id]
            harness_row = abs_row.harness_row(row)
            text = row.patch if job.kind is JobKind.SWE_ABS_GOLD else patches(job.entry, job.instance_id)
            applied = text if job.kind is JobKind.SWE_ABS_GOLD else swe_abs_patch_text(text, abs_row)
        if sha256_text(text) != job.patch_sha256:
            raise LabelResultsError(f"{job.id}: the patch on disk is not the one the manifest pinned")
        image = images.get(job.instance_id)
        if image is None:
            raise LabelResultsError(f"{job.instance_id}: no pinned image")
        out.append(Material(job=job, image=image, harness_row=harness_row, patch=applied))
    return out


def run_job(
    m: Material,
    docker: Docker,
    grader: Grader,
    *,
    run_id: str,
    timeout_s: float,
    clock: Callable[[], float] = time.monotonic,
) -> JobResult:
    """One job: a fresh container, and one more if the eval script times out. INFRA is never retried here (it is
    recorded, and the next run picks the job up again)."""
    script = grader.eval_script(m.harness_row)
    run: ContainerRun | None = None
    attempts = 0
    for _ in range(2):
        attempts += 1
        name = f"receipts-label-{run_id}-{sha256_text(m.job.id)[:16]}-{attempts}"
        run = run_in_container(
            docker, m.image, PRIMARY, name=name, patch=m.patch, eval_script=script, timeout_s=timeout_s, clock=clock
        )
        if not run.timed_out:
            break
    assert run is not None
    base: dict[str, Any] = {
        "run_id": run_id,
        "job": m.job,
        "image": m.image.ref,
        "gvisor": run.gvisor,
        "seconds": run.seconds,
        "attempts": attempts,
        "patch_method": run.patch_method,
    }
    if run.infra:
        return JobResult(**base, status=JobStatus.INFRA, detail=run.infra)
    if not run.patch_applied:
        return JobResult(**base, status=JobStatus.PATCH_DOES_NOT_APPLY, detail=run.error)
    if run.timed_out:
        return JobResult(**base, status=JobStatus.TIMEOUT, detail=f"the eval script ran past {timeout_s:g} s twice")
    g = grader.grade(m.harness_row, run.output)
    return JobResult(**base, status=JobStatus.DONE, log_found=g.log_found, statuses=g.statuses)


class RunSummary(Frozen):
    run_id: str
    planned: int
    already_final: int
    ran: int
    statuses: dict[str, int]
    infra: tuple[str, ...]  # job ids still waiting on a rerun


def run_labels(
    work: Sequence[Material],
    docker: Docker,
    grader: Grader,
    log: ResultLog,
    *,
    run_id: str,
    timeout_s: float = 1800,
    workers: int = 16,
) -> RunSummary:
    """Run every job without a final result in the log, appending each result as it lands. A runtime mismatch stops
    everything still queued; any other exception in one job is that job's INFRA."""
    done = read_results(log.lines())
    todo = [m for m in work if m.job.id not in done]
    lock = threading.Lock()
    ran: list[JobResult] = []

    def one(m: Material) -> JobResult:
        try:
            return run_job(m, docker, grader, run_id=run_id, timeout_s=timeout_s)
        except RuntimeMismatchError:
            raise
        except Exception as e:
            return JobResult(
                run_id=run_id,
                job=m.job,
                image=m.image.ref,
                status=JobStatus.INFRA,
                detail=f"{type(e).__name__}: {e}"[-500:],
            )

    pool = ThreadPoolExecutor(max_workers=workers)
    try:
        futures = [pool.submit(one, m) for m in todo]
        for future in as_completed(futures):
            r = future.result()
            log.append(r)
            with lock:
                ran.append(r)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    return RunSummary(
        run_id=run_id,
        planned=len(work),
        already_final=len(work) - len(todo),
        ran=len(ran),
        statuses=dict(sorted(Counter(r.status.value for r in ran).items())),
        infra=tuple(sorted(r.job.id for r in ran if r.status is JobStatus.INFRA)),
    )


# ---------------------------------------------------------------- from results to evidence


class RecordLabelEvidence(Frozen):
    swe_abs_passed: bool | None = None  # our rule
    swe_abs_rule: bool | None = None  # SWE-ABS's own rule on the same run
    fail_to_pass_passed: bool | None = None
    pass_to_pass_passed: bool | None = None
    patch_applies: bool | None = None


class LabelEvidence(Frozen):
    gold: dict[str, bool]  # instance -> gold passes the SWE-ABS tests here
    records: dict[str, RecordLabelEvidence]  # "<entry>~<instance>" -> evidence
    dropped: dict[str, str] = Field(default_factory=dict)  # "<entry>~<instance>" -> the founder's reason


def evidence(
    results: Mapping[str, JobResult], rows: Mapping[str, Row], drops: Sequence[LandedLabelDrop] = ()
) -> LabelEvidence:
    """Verdicts from final results. A patch's SWE-ABS verdict needs its instance's gold result; without one it stays
    unknown. A gold that timed out or did not apply is not an oracle. `drops` (the founder's) mark the records whose
    label waited on a job that was INFRA on every run."""
    gold_status: dict[str, Mapping[str, str]] = {}
    gold: dict[str, bool] = {}
    for r in results.values():
        if r.job.kind is JobKind.SWE_ABS_GOLD:
            ok = r.status is JobStatus.DONE and gold_passes(r.statuses, r.log_found)
            gold[r.job.instance_id] = ok
            if ok:
                gold_status[r.job.instance_id] = r.statuses
    records: dict[str, RecordLabelEvidence] = {}
    for r in results.values():
        key = f"{r.job.entry}~{r.job.instance_id}"
        if r.job.kind is JobKind.SWE_ABS_PATCH:
            iid = r.job.instance_id
            if r.status is JobStatus.PATCH_DOES_NOT_APPLY:
                records[key] = RecordLabelEvidence(patch_applies=False)
            elif iid not in gold:
                continue  # gold has not run: the patch's verdict waits
            elif not gold[iid]:
                records[key] = RecordLabelEvidence(patch_applies=True)  # NO_ABS_ORACLE comes from the gold side
            elif r.status is JobStatus.TIMEOUT:
                records[key] = RecordLabelEvidence(swe_abs_passed=False, swe_abs_rule=False, patch_applies=True)
            else:
                records[key] = RecordLabelEvidence(
                    swe_abs_passed=patch_passes(r.statuses, r.log_found, gold_status[iid]),
                    swe_abs_rule=swe_abs_rule(r.statuses),  # theirs reads the status map alone
                    patch_applies=True,
                )
        elif r.job.kind is JobKind.F2P_PATCH:
            row = rows[r.job.instance_id]
            if r.status is JobStatus.PATCH_DOES_NOT_APPLY:
                records[key] = RecordLabelEvidence(patch_applies=False)
            elif r.status is JobStatus.TIMEOUT or not r.log_found:
                records[key] = RecordLabelEvidence(fail_to_pass_passed=False, patch_applies=True)
            else:
                f2p = all(r.statuses.get(t) == "PASSED" for t in row.FAIL_TO_PASS)
                p2p = all(r.statuses.get(t) in ("PASSED", "XFAIL") for t in row.PASS_TO_PASS)
                records[key] = RecordLabelEvidence(fail_to_pass_passed=f2p, pass_to_pass_passed=p2p, patch_applies=True)
    dropped = {key: d.reason for d in drops for key in d.records}
    return LabelEvidence(
        gold=dict(sorted(gold.items())), records=dict(sorted(records.items())), dropped=dict(sorted(dropped.items()))
    )


def results_path(out_dir: Path) -> Path:
    return out_dir / "results.jsonl"
