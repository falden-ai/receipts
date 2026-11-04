"""The contracts refuse what the spec's hard rules forbid. Each test feeds broken input and expects a refusal."""

from __future__ import annotations

import json
from collections.abc import Sequence

import pytest
from pydantic import ValidationError

from receipts.cli import main
from receipts.contracts import (
    RUN_PHASE_LIMITS,
    SCHEMA_VERSION,
    Arm,
    Attack,
    Attacker,
    AttackOutcome,
    CaseResult,
    CatchKind,
    Claim,
    ClaimType,
    FailureType,
    Finding,
    LedgerEntry,
    Lens,
    Message,
    ModelCallError,
    ModelClient,
    ModelRefusedError,
    ModelRequest,
    ModelResponse,
    Outcome,
    PatchMethod,
    ReplayKey,
    Repro,
    RunnerKind,
    RunResult,
    Sandbox,
    SandboxLimits,
    Side,
    Span,
    SpanSource,
    StageCost,
    Status,
    SuiteRunner,
    Target,
    Tier,
    UncheckedReason,
    Usage,
    Workspace,
    check_model,
    docker_bytes,
)

SHA_A = "a" * 40
SHA_B = "b" * 40


def run(side: Side, outcome: Outcome, failure: FailureType | None = None) -> RunResult:
    return RunResult(
        side=side,
        outcome=outcome,
        failure_type=failure,
        exit_code=0 if outcome is Outcome.PASS else 1,
        fresh_container=True,
        duration_s=1.0,
    )


def repro(kind: CatchKind = CatchKind.REGRESSION) -> Repro:
    return Repro(
        id="r1",
        attack_id="a1",
        claim_id="c1",
        lens=Lens.CORRECTNESS,
        kind=kind,
        files={"tests/test_receipts_r1.py": "def test_x():\n    assert False\n"},
        command=("python", "-m", "pytest", "tests/test_receipts_r1.py"),
    )


def blocker(runs: tuple[RunResult, ...], kind: CatchKind = CatchKind.REGRESSION, **kw: object) -> Finding:
    fields: dict[str, object] = dict(
        id="t1/c1/a1",
        target_id="t1",
        claim_id="c1",
        lens=Lens.CORRECTNESS,
        status=Status.BLOCKER,
        kind=kind,
        repro=repro(kind),
        runs=runs,
        quote="Fixes the crash when the list is empty.",
        replay=ReplayKey(base_sha=SHA_A, patch_sha=SHA_B, harness_hash="h", seed=0, image_digest="sha256:x"),
    )
    fields.update(kw)
    return Finding.model_validate(fields)


FAILS = tuple(run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION) for _ in range(5))
PASSES = tuple(run(Side.BASE, Outcome.PASS) for _ in range(5))


def test_contributor_and_unknown_model_ids_are_refused() -> None:
    for bad in ("muse-spark-1.3-contributor", "muse-spark-1.2-contributor", "MUSE-SPARK-1.3-CONTRIBUTOR", "gpt-x"):
        with pytest.raises(ModelRefusedError):
            check_model(bad)
    with pytest.raises(ValidationError):
        ModelRequest(
            stage="plan",
            model="muse-spark-1.3-contributor",
            messages=(Message(role="user", content="x"),),
            temperature=0,
            max_tokens=10,
        )
    assert check_model("muse-spark-1.3") == "muse-spark-1.3"


def test_a_regression_blocker_with_full_evidence_is_accepted() -> None:
    assert blocker(FAILS + PASSES).status is Status.BLOCKER


def test_a_blocker_without_a_repro_is_refused() -> None:
    with pytest.raises(ValidationError, match="needs a repro"):
        blocker(FAILS + PASSES, repro=None)


def test_a_blocker_without_the_quoted_sentence_is_refused() -> None:
    with pytest.raises(ValidationError, match="quoted claim"):
        blocker(FAILS + PASSES, quote="")


def test_four_failing_patch_runs_are_not_enough() -> None:
    with pytest.raises(ValidationError, match="every run"):
        blocker(FAILS[:4] + PASSES)


def test_a_mixed_patch_result_is_not_a_blocker() -> None:
    mixed = (*FAILS[:4], run(Side.PATCH, Outcome.PASS), *FAILS[:1])
    with pytest.raises(ValidationError, match="every run"):
        blocker(mixed + PASSES)


def test_an_import_error_does_not_count_as_a_failure() -> None:
    imports = tuple(run(Side.PATCH, Outcome.FAIL, FailureType.IMPORT) for _ in range(5))
    with pytest.raises(ValidationError, match="assertion or a crash"):
        blocker(imports + PASSES)


def test_a_regression_catch_must_pass_on_base() -> None:
    with pytest.raises(ValidationError, match="passes on base"):
        blocker(FAILS + PASSES[:4] + (run(Side.BASE, Outcome.FAIL, FailureType.ASSERTION),))


def test_a_fix_validity_catch_needs_a_passing_surrogate() -> None:
    with pytest.raises(ValidationError, match="surrogate"):
        blocker(FAILS, kind=CatchKind.FIX_VALIDITY)
    ok = blocker((*FAILS, run(Side.SURROGATE, Outcome.PASS)), kind=CatchKind.FIX_VALIDITY)
    assert ok.kind is CatchKind.FIX_VALIDITY


def test_an_unchecked_finding_names_its_reason() -> None:
    with pytest.raises(ValidationError, match="reason"):
        Finding(id="t1/c1/mutation", target_id="t1", claim_id="c1", lens=Lens.MUTATION, status=Status.UNCHECKED)


def test_repro_paths_stay_inside_the_repository() -> None:
    for bad in ("/etc/passwd", "../outside.py", "tests/../../x.py"):
        with pytest.raises(ValidationError):
            Repro(
                id="r",
                attack_id="a",
                claim_id="c",
                lens=Lens.CORRECTNESS,
                kind=CatchKind.REGRESSION,
                files={bad: "x"},
                command=("true",),
            )


def test_extra_fields_are_refused() -> None:
    with pytest.raises(ValidationError):
        RunResult.model_validate(
            {"side": "patch", "outcome": "pass", "fresh_container": True, "duration_s": 0, "surprise": 1}
        )


def test_a_failed_run_names_its_failure_type() -> None:
    with pytest.raises(ValidationError):
        run(Side.PATCH, Outcome.FAIL)


def test_cli_version(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["version"]) == 0
    assert capsys.readouterr().out.strip() == "0.0.1"


def test_a_finding_id_starts_with_its_target() -> None:
    with pytest.raises(ValidationError, match="target id"):
        blocker(FAILS + PASSES, id="c1/a1")


def test_skipped_base_runs_never_count_as_passes() -> None:
    skipped = tuple(run(Side.BASE, Outcome.SKIPPED) for _ in range(5))
    with pytest.raises(ValidationError, match="passes on base"):
        blocker(FAILS + skipped)


def test_output_that_is_not_utf8_is_refused_where_it_is_made() -> None:
    with pytest.raises(ValidationError, match="UTF-8"):
        RunResult(side=Side.PATCH, outcome=Outcome.PASS, fresh_container=True, duration_s=0, stdout_tail="ok \udcff")
    with pytest.raises(ValidationError, match="UTF-8"):
        Repro(
            id="r",
            attack_id="a",
            claim_id="c",
            lens=Lens.CORRECTNESS,
            kind=CatchKind.REGRESSION,
            files={"tests/t.py": "x = '\udcff'"},
            command=("true",),
        )


def test_a_target_image_must_be_pinned_by_digest() -> None:
    base = dict(id="t", tier=Tier.A, repo="o/r", base_sha="a" * 40, patch="diff")
    with pytest.raises(ValidationError):
        Target.model_validate({**base, "image": "python:3.13-slim"})
    pinned = "docker.io/library/python@sha256:" + "0" * 64
    assert Target.model_validate({**base, "image": pinned}).image == pinned


def test_model_call_error_carries_its_reason() -> None:
    e = ModelCallError("budget reached", UncheckedReason.BUDGET)
    assert e.reason is UncheckedReason.BUDGET and isinstance(e, RuntimeError)


def test_run_phase_limits_are_the_spec_values() -> None:
    lim = RUN_PHASE_LIMITS
    assert (lim.network, lim.cap_drop, lim.pids_limit, lim.memory, lim.cpus) == ("none", "ALL", 512, "4g", 2.0)


# ---------------------------------------------------------------- contract version 2

TARGET = dict(tier=Tier.A, repo="o/r", base_sha="a" * 40, patch="diff")
DIGEST = "@sha256:" + "0" * 64


def test_a_target_id_never_holds_a_slash_or_whitespace() -> None:
    for bad in ("t1/x", "owner/repo#12", "t 1", "t\n1", "t\x001"):
        with pytest.raises(ValidationError, match="target id"):
            Target.model_validate({**TARGET, "id": bad})
    assert Target.model_validate({**TARGET, "id": "20260217_mini-v2.0.0_gemini-3-flash-high~django__django-11099"})


def test_a_target_image_follows_dockers_reference_grammar() -> None:
    for good in (
        "docker.io/library/python:3.12" + DIGEST,
        "localhost:5000/a/b" + DIGEST,
        "swebench/sweb.eval.x86_64.astropy__astropy-12907:latest" + DIGEST,
        "ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-11099" + DIGEST,
    ):
        assert Target.model_validate({**TARGET, "id": "t", "image": good}).image == good
    for bad in (
        "-v=/:/host" + DIGEST,  # docker would read it as a flag
        "--privileged" + DIGEST,
        "Python" + DIGEST,  # repository paths are lowercase
        "a//b" + DIGEST,
        "python:3.12@sha256:" + "0" * 63,
        "python" + DIGEST + " ",
        "a" * 256 + DIGEST,
    ):
        with pytest.raises(ValidationError):
            Target.model_validate({**TARGET, "id": "t", "image": bad})


def claim(claim_id: str, kind: ClaimType = ClaimType.FIXED_X) -> Claim:
    return Claim(id=claim_id, type=kind, text="Fixes it.", span=Span(source=SpanSource.TEMPLATE, text="Fixes it."))


def test_claim_ids_are_relative_to_their_target() -> None:
    assert claim("fixed_x").id == "fixed_x"
    assert claim("tests_pass:0123456789", ClaimType.TESTS_PASS).id == "tests_pass:0123456789"
    for bad, kind in (
        ("t1:fixed_x", ClaimType.FIXED_X),  # the day-1 form, with the target id inside
        ("t1:tests_pass:0123456789", ClaimType.TESTS_PASS),
        ("fixed_x:", ClaimType.FIXED_X),
        ("fixed_x:012345678", ClaimType.FIXED_X),
        ("fixed_x:0123456789a", ClaimType.FIXED_X),
        ("fixed_x:ABCDEF0123", ClaimType.FIXED_X),
        ("perf", ClaimType.FIXED_X),  # another type's id
    ):
        with pytest.raises(ValidationError, match="claim id"):
            claim(bad, kind)


def test_a_finding_id_starts_with_its_target_and_its_claim() -> None:
    with pytest.raises(ValidationError, match="then its claim id"):
        blocker(FAILS + PASSES, id="t1/c2/a1")
    with pytest.raises(ValidationError, match="then its claim id"):
        blocker(FAILS + PASSES, id="t1/c1/")
    with pytest.raises(ValidationError, match="target id"):
        blocker(FAILS + PASSES, id="t1/x/c1/a1", target_id="t1/x")
    with pytest.raises(ValidationError, match="claim id"):
        blocker(FAILS + PASSES, id="t1/c/1/a1", claim_id="c/1")


def test_claim_references_carry_no_slash() -> None:
    with pytest.raises(ValidationError, match="claim id"):
        Repro(
            id="r",
            attack_id="a",
            claim_id="c/1",
            lens=Lens.CORRECTNESS,
            kind=CatchKind.REGRESSION,
            files={"tests/t.py": "x"},
            command=("true",),
        )


def test_a_failed_model_call_carries_what_it_spent() -> None:
    spent = StageCost(stage="plan", model="muse-spark-1.3", input_tokens=10, reasoning_tokens=4, micro_usd=12)
    e = ModelCallError("timed out after the reply started", UncheckedReason.TIMEOUT, cost=spent)
    assert e.cost == spent and ModelCallError("x", UncheckedReason.BUDGET).cost is None
    with pytest.raises(ValidationError):
        StageCost(stage="plan", reasoning_tokens=-1)
    assert UncheckedReason("no_input") is UncheckedReason.NO_INPUT


def response(text: str, finish: str) -> ModelResponse:
    return ModelResponse(
        model="muse-spark-1.3", text=text, finish_reason=finish, usage=Usage(input_tokens=1, output_tokens=1)
    )


def test_an_empty_answer_counts_as_truncated() -> None:
    assert response("", "stop").truncated
    assert response(" \n", "stop").truncated
    assert response("ok", "length").truncated
    assert not response("ok", "stop").truncated


def test_ledger_entries_are_version_2() -> None:
    finding = Finding(
        id="t1/c1/mutation",
        target_id="t1",
        claim_id="c1",
        lens=Lens.MUTATION,
        status=Status.UNCHECKED,
        unchecked_reason=UncheckedReason.BUDGET,
    )
    assert LedgerEntry(at="2026-10-27T12:00:00Z", receipts_version="0.0.1", finding=finding).v == SCHEMA_VERSION == 2
    with pytest.raises(ValidationError, match="Input should be 2"):
        LedgerEntry.model_validate(
            {"v": 1, "at": "2026-10-27T12:00:00Z", "receipts_version": "0.0.1", "finding": finding.model_dump()}
        )


def test_run_flags_render_every_isolation_flag() -> None:
    assert RUN_PHASE_LIMITS.run_flags(workspace="/testbed") == [
        "--network=none",
        "--cap-drop=ALL",
        "--security-opt=no-new-privileges",
        "--pids-limit=512",
        "--memory=4g",
        "--memory-swap=4g",
        "--cpus=2",
        "--shm-size=64m",
        "--ipc=private",
        "--ulimit=msgqueue=0",
        "--user=65534:65534",
        "--read-only",
        "--tmpfs=/testbed:rw,exec,nosuid,nodev,size=1073741824,uid=65534,gid=65534,mode=0755",
        "--tmpfs=/tmp:rw,exec,nosuid,nodev,size=268435456,mode=1777",
    ]
    assert RUN_PHASE_LIMITS.limit_flags() == RUN_PHASE_LIMITS.run_flags()[1:10]


def test_run_flags_refuse_a_mount_that_is_not_a_plain_path() -> None:
    for bad in ("/", "testbed", "/dev/x", "/proc", "/sys/fs", "/a/../dev", "/w,uid=0", "/w:ro", "/w x", "/tmp"):
        with pytest.raises(ValueError):
            RUN_PHASE_LIMITS.run_flags(workspace=bad)


def test_sandbox_limits_refuse_root_and_mounts_over_the_memory_cap() -> None:
    for user in ("0:0", "0:65534", "65534:0"):
        with pytest.raises(ValidationError, match="never root"):
            SandboxLimits(user=user)
    with pytest.raises(ValidationError):
        SandboxLimits(user="nobody")
    with pytest.raises(ValidationError, match="memory cap"):
        SandboxLimits(memory="1g")  # 1 GiB workspace + 256 MiB /tmp + 64 MiB /dev/shm do not fit
    with pytest.raises(ValidationError):
        SandboxLimits(memory="4 g")
    assert docker_bytes("64m") == 64 * 1024**2 and docker_bytes("4g") == 4 * 1024**3
    with pytest.raises(ValueError):
        docker_bytes("0g")


def test_model_call_error_crosses_a_process_boundary() -> None:
    import pickle

    spent = StageCost(stage="attack", model="muse-glimmer-30b", output_tokens=7)
    back = pickle.loads(pickle.dumps(ModelCallError("timed out", UncheckedReason.TIMEOUT, cost=spent)))
    assert (str(back), back.reason, back.cost) == ("timed out", UncheckedReason.TIMEOUT, spent)


def test_floats_are_finite() -> None:
    for bad in (float("inf"), float("nan")):
        with pytest.raises(ValidationError):
            StageCost(stage="plan", seconds=bad)
        with pytest.raises(ValidationError):
            RunResult(side=Side.BASE, outcome=Outcome.PASS, fresh_container=True, duration_s=bad)
    assert check_model("muse-glimmer-30b") == "muse-glimmer-30b"


# ---------------------------------------------------------------- the test-runner seam and the pipeline's seams


def test_a_target_names_its_runner() -> None:
    assert Target.model_validate({**TARGET, "id": "t"}).test_runner is RunnerKind.PYTEST
    django = Target.model_validate(
        {**TARGET, "id": "t", "test_runner": "django", "test_command": "./tests/runtests.py --verbosity 2"}
    )
    assert django.test_runner is RunnerKind.DJANGO
    with pytest.raises(ValidationError):
        Target.model_validate({**TARGET, "id": "t", "test_runner": "nose"})


def test_a_case_result_follows_the_run_result_rules() -> None:
    assert CaseResult(name="tests/t.py::test_x", outcome=Outcome.PASS).failure_type is None
    with pytest.raises(ValidationError, match="names its failure type"):
        CaseResult(name="test_x (app.tests.T)", outcome=Outcome.FAIL)
    with pytest.raises(ValidationError, match="no failure type"):
        CaseResult(name="t", outcome=Outcome.PASS, failure_type=FailureType.ASSERTION)
    with pytest.raises(ValidationError, match="UTF-8"):
        CaseResult(name="t", outcome=Outcome.ERROR, message="bad \udcff")


def test_a_case_result_names_its_file_where_names_repeat() -> None:
    assert CaseResult(name="t", outcome=Outcome.PASS).file is None
    case = CaseResult(name="test_x", outcome=Outcome.PASS, file="sympy/core/tests/test_a.py")
    assert (case.name, case.file) == ("test_x", "sympy/core/tests/test_a.py")
    for bad in ("", "/testbed/sympy/core/tests/test_a.py", "../test_a.py", "sympy/../../test_a.py"):
        with pytest.raises(ValidationError, match="relative, inside the repository"):
            CaseResult(name="test_x", outcome=Outcome.PASS, file=bad)
    with pytest.raises(ValidationError, match="UTF-8"):
        CaseResult(name="test_x", outcome=Outcome.PASS, file="test_\udcff.py")


def test_a_workspace_names_how_its_patch_applied() -> None:
    base = {"id": "w", "side": "patch", "root": "/testbed", "image_digest": "sha256:" + "0" * 64}
    assert Workspace.model_validate(base).patch_method is None
    fuzzed = Workspace.model_validate({**base, "patch_method": "gnu_patch_fuzz"})
    assert fuzzed.patch_method is PatchMethod.GNU_PATCH_FUZZ


def test_an_arm_uses_allowed_models_only() -> None:
    assert Arm(name="glimmer", attacker_model="muse-glimmer-30b").step_limit == 20
    for bad in (
        {"name": "Spark"},
        {"name": "s", "attacker_model": "muse-spark-1.3-contributor"},
        {"name": "s", "step_limit": 31},
    ):
        with pytest.raises(ValidationError):
            Arm.model_validate(bad)


def test_an_attack_outcome_holds_only_its_own_repros() -> None:
    mine = repro()  # attack id a1
    assert AttackOutcome(attack_id="a1", repros=(mine,)).repros == (mine,)
    with pytest.raises(ValidationError, match="its own attack"):
        AttackOutcome(attack_id="a2", repros=(mine,))


class _Runner:
    kind = RunnerKind.PYTEST

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["python", "-m", "pytest", f"--junitxml={report_path}", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return ()


class _Attacker:
    lens = Lens.CORRECTNESS

    def attack(
        self,
        target: Target,
        attack: Attack,
        *,
        arm: Arm,
        client: ModelClient,
        sandbox: Sandbox,
        runner: SuiteRunner,
    ) -> AttackOutcome:
        return AttackOutcome(attack_id=attack.id)


def test_fakes_satisfy_the_seams() -> None:
    # `mypy --strict tests/test_contracts.py` checks the fakes' structure; the asserts check they behave.
    runner: SuiteRunner = _Runner()
    attacker: Attacker = _Attacker()
    target = Target.model_validate({**TARGET, "id": "t"})
    assert runner.command(target, ["tests/t.py"], report_path="/tmp/r1.xml")[-1] == "tests/t.py"
    assert attacker.lens is Lens.CORRECTNESS


# ---------------------------------------------------------------- contract version 3

WORKSPACE = {"id": "w", "side": "patch", "root": "/testbed", "image_digest": "sha256:" + "0" * 64}


def test_patch_methods_are_the_harness_chain_in_its_order() -> None:
    # swebench 4.1.0, harness/run_evaluation.py GIT_APPLY_CMDS: git apply --verbose, then --reject, then GNU patch.
    assert [m.value for m in PatchMethod] == ["git_apply", "git_apply_reject", "gnu_patch_fuzz"]
    rejected = Workspace.model_validate({**WORKSPACE, "patch_method": "git_apply_reject"})
    assert rejected.patch_method is PatchMethod.GIT_APPLY_REJECT
    for gone in ("git_apply_recount", "git_apply_c1"):  # either could apply a patch the harness refused
        with pytest.raises(ValidationError):
            Workspace.model_validate({**WORKSPACE, "patch_method": gone})


def repro_with(cases: object) -> Repro:
    return Repro.model_validate({**repro().model_dump(), "cases": cases})


def test_a_repro_names_the_cases_it_runs() -> None:
    assert repro().cases == ()  # every case its test files define
    cases = (
        "tests/test_receipts_r1.py::test_x",  # a pytest node id
        "test_x (validators.tests.TestValidators)",  # a Django label
        "sympy/core/tests/test_receipts_r1.py::test_x",  # sympy, path::name
        "test_y",  # sympy, a bare name
    )
    assert repro_with(cases).cases == cases
    for bad, message in (
        (("",), "blank"),
        ((" \t\n",), "blank"),
        (("tests/test_receipts_r1.py::test_x", "   "), "blank"),
        (("test_\udcff",), "UTF-8"),
        (("test_x", "test_y", "test_x"), "names case 'test_x' twice"),
    ):
        with pytest.raises(ValidationError, match=message):
            repro_with(bad)


def test_a_run_says_why_it_could_not_be_read() -> None:
    assert run(Side.PATCH, Outcome.PASS).problem == ""
    never_ran = RunResult(
        side=Side.PATCH,
        outcome=Outcome.ERROR,
        failure_type=FailureType.IMPORT,
        fresh_container=True,
        duration_s=1.0,
        problem="expected case tests/test_receipts_r1.py::test_x never ran: its module raised ImportError",
    )
    assert never_ran.failure_type is FailureType.IMPORT and "never ran" in never_ran.problem
    with pytest.raises(ValidationError, match="a run's problem is not valid UTF-8"):
        RunResult(side=Side.PATCH, outcome=Outcome.ERROR, fresh_container=True, duration_s=0, problem="bad \udcff")


def test_a_finding_says_why_it_has_its_status_apart_from_a_refutation() -> None:
    unreadable = RunResult(
        side=Side.PATCH,
        outcome=Outcome.ERROR,
        failure_type=FailureType.INFRA,
        fresh_container=True,
        duration_s=1.0,
        problem="the run wrote no report",
    )
    flaky = Finding(
        id="t1/c1/a1",
        target_id="t1",
        claim_id="c1",
        lens=Lens.CORRECTNESS,
        status=Status.FLAKY,
        kind=CatchKind.REGRESSION,
        repro=repro_with(("tests/test_receipts_r1.py::test_x",)),
        runs=(run(Side.PATCH, Outcome.FAIL, FailureType.ASSERTION), run(Side.PATCH, Outcome.PASS), unreadable),
        detail="failed on the patch 3 of 5 times",
    )
    assert flaky.refutation is None and flaky.detail == "failed on the patch 3 of 5 times"
    # The new fields survive the ledger's strict JSON read.
    entry = LedgerEntry(at="2026-10-27T12:00:00Z", receipts_version="0.0.1", finding=flaky)
    assert LedgerEntry.model_validate_json(entry.model_dump_json(), strict=True) == entry


# A ledger line exactly as contract v2 (65a0d21) wrote it: a flaky finding, its repro and two runs, with no v3 field.
V2_LINE = (
    r'{"v":2,"at":"2026-10-27T12:00:00Z","receipts_version":"0.0.1","finding":{"id":"t1/c1/a1","target_id":"t1"'
    r',"claim_id":"c1","lens":"correctness","status":"flaky","kind":"regression","unchecked_reason":null'
    r',"repro":{"id":"r1","attack_id":"a1","claim_id":"c1","lens":"correctness","kind":"regression"'
    r',"files":{"tests/test_receipts_r1.py":"def test_x():\n    assert False\n"},"command":["python","-m","pytest"'
    r',"tests/test_receipts_r1.py"]},"runs":[{"side":"patch","outcome":"fail","failure_type":"assertion"'
    r',"exit_code":1,"seed":null,"fresh_container":true,"duration_s":1.0,"stdout_tail":"","stderr_tail":""'
    r',"image_digest":""},{"side":"patch","outcome":"pass","failure_type":null,"exit_code":0,"seed":null'
    r',"fresh_container":true,"duration_s":1.0,"stdout_tail":"","stderr_tail":"","image_digest":""}],"quote":null'
    r',"refutation":null,"replay":null,"costs":[],"critic_rounds":0,"wall_seconds":null}}'
)


def test_v2_data_still_validates() -> None:
    data = json.loads(V2_LINE)["finding"]
    assert "detail" not in data and "cases" not in data["repro"] and "problem" not in data["runs"][0]
    assert Repro.model_validate(data["repro"]).cases == ()
    assert RunResult.model_validate(data["runs"][0]).problem == ""
    assert Finding.model_validate(data).detail is None
    finding = LedgerEntry.model_validate_json(V2_LINE, strict=True).finding  # as the ledger reads a line
    assert finding.repro is not None and finding.repro.cases == ()
    assert (finding.detail, [r.problem for r in finding.runs]) == (None, ["", ""])


# ---------------------------------------------------------------- contract version 3.1


def test_a_case_says_when_it_failed_as_expected() -> None:
    assert CaseResult(name="t", outcome=Outcome.SKIPPED).expected_failure is False  # a plain skip
    for case in (
        CaseResult(name="tests/t.py::test_x", outcome=Outcome.SKIPPED, message="known bug", expected_failure=True),
        CaseResult(name="app.tests.T.test_x", outcome=Outcome.SKIPPED, expected_failure=True),  # expectedFailure
        CaseResult(name="test_x", outcome=Outcome.SKIPPED, file="sympy/core/tests/test_a.py", expected_failure=True),
    ):
        assert case.expected_failure
        assert CaseResult.model_validate_json(case.model_dump_json(), strict=True) == case
    # It never counts as a pass, so a case that failed as expected is SKIPPED and nothing else.
    for outcome, failure in (
        (Outcome.PASS, None),
        (Outcome.FAIL, FailureType.ASSERTION),
        (Outcome.ERROR, FailureType.INFRA),
        (Outcome.TIMEOUT, FailureType.TIMEOUT),
    ):
        with pytest.raises(ValidationError, match="failed as expected is skipped"):
            CaseResult(name="t", outcome=outcome, failure_type=failure, expected_failure=True)


# Two case results and a ledger line exactly as contract v3 (11f3f0a) wrote them, with no v3.1 field.
V3_CASES = (
    r'{"name":"tests/test_x.py::test_marked","outcome":"skipped","failure_type":null,"raising_file":null'
    r',"message":"known bug","file":null}',
    r'{"name":"test_x","outcome":"pass","failure_type":null,"raising_file":null,"message":""'
    r',"file":"sympy/core/tests/test_a.py"}',
)
V3_LINE = (
    r'{"v":2,"at":"2026-10-28T06:00:00Z","receipts_version":"0.0.1","finding":{"id":"t1/c1/a1","target_id":"t1"'
    r',"claim_id":"c1","lens":"correctness","status":"flaky","kind":"regression","unchecked_reason":null'
    r',"repro":{"id":"r1","attack_id":"a1","claim_id":"c1","lens":"correctness","kind":"regression"'
    r',"files":{"tests/test_receipts_r1.py":"def test_x():\n    assert False\n"},"command":["python","-m","pytest"'
    r',"tests/test_receipts_r1.py"],"cases":["tests/test_receipts_r1.py::test_x"]},"runs":[{"side":"patch"'
    r',"outcome":"fail","failure_type":"assertion","exit_code":1,"seed":null,"fresh_container":true,"duration_s":1.0'
    r',"stdout_tail":"","stderr_tail":"","image_digest":"","problem":""},{"side":"patch","outcome":"error"'
    r',"failure_type":"infra","exit_code":null,"seed":null,"fresh_container":true,"duration_s":1.0,"stdout_tail":""'
    r',"stderr_tail":"","image_digest":"","problem":"the run wrote no report"}],"quote":null,"refutation":null'
    r',"detail":"failed on the patch 3 of 5 times","replay":null,"costs":[],"critic_rounds":0,"wall_seconds":null}}'
)


def test_v3_data_still_validates() -> None:
    for line in V3_CASES:
        assert "expected_failure" not in json.loads(line)
        assert CaseResult.model_validate_json(line, strict=True).expected_failure is False
    finding = LedgerEntry.model_validate_json(V3_LINE, strict=True).finding  # as the ledger reads a line
    assert finding.repro is not None and finding.repro.cases == ("tests/test_receipts_r1.py::test_x",)
    assert finding.detail == "failed on the patch 3 of 5 times"
    assert [r.problem for r in finding.runs] == ["", "the run wrote no report"]


# ---------------------------------------------------------------- contract version 3.2


def test_a_repro_carries_the_sentence_it_enforces_and_an_attack_says_when_it_reruns_the_existing_tests() -> None:
    fields: dict[str, object] = {
        "id": "r1",
        "attack_id": "correctness-1",
        "claim_id": "fixed_x",
        "lens": Lens.CORRECTNESS,
        "kind": CatchKind.FIX_VALIDITY,
        "files": {"tests/test_receipts_r1.py": "def test_x():\n    assert False\n"},
        "command": ("tests/test_receipts_r1.py::test_x",),
    }
    assert Repro.model_validate(fields).quote == ""  # a regression repro's oracle is base: it quotes nothing
    quoted = Repro.model_validate({**fields, "quote": "render crashes on an empty list"})
    assert Repro.model_validate_json(quoted.model_dump_json(), strict=True) == quoted
    with pytest.raises(ValidationError, match="a repro's quote is not valid UTF-8"):
        Repro.model_validate({**fields, "quote": "bad \ud800"})
    plain = {"id": "regression-1", "claim_id": "nothing_else_changed", "hypothesis": "h", "mask": "diff"}
    assert Attack.model_validate({**plain, "lens": "regression", "temperature": 0}).existing_tests is False
    suite = Attack.model_validate({**plain, "lens": "regression", "temperature": 0, "existing_tests": True})
    assert Attack.model_validate_json(suite.model_dump_json(), strict=True) == suite
    with pytest.raises(ValidationError, match="only a regression attack reruns the existing tests"):
        Attack.model_validate({**plain, "lens": "correctness", "temperature": 0, "existing_tests": True})


# An attack exactly as contract v3.1 (be1787a) wrote it, with no v3.2 field; its repros are V3_LINE's.
V31_ATTACK = (
    r'{"id":"regression-1","claim_id":"nothing_else_changed","lens":"regression","hypothesis":"h","mask":"diff"'
    r',"temperature":0.0}'
)


def test_v3_1_data_still_validates() -> None:
    attack = Attack.model_validate_json(V31_ATTACK, strict=True)
    assert attack.existing_tests is False
    finding = LedgerEntry.model_validate_json(V3_LINE, strict=True).finding
    assert finding.repro is not None and "quote" not in json.loads(V3_LINE)["finding"]["repro"]
    assert finding.repro.quote == ""
