"""The composition root: `receipts` has every merged lane's commands, wired to the real implementations, and builds
nothing that needs a key, the network or Docker until a run starts. Every test here runs with no key in the
environment, with sockets and child processes refused, so none can call a model, open a connection or start Docker."""

from __future__ import annotations

import argparse
import functools
import json
import secrets
import shlex
import socket
import subprocess
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import suppress
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path
from typing import Any

import httpx
import pytest

from receipts import cli
from receipts.agentloop.attacker import AttackTrace, LoopAttacker
from receipts.agentloop.fix_command import FixWiring
from receipts.agentloop.fixloop import (
    FIX_CAP_MICRO_USD,
    FIXER_MODEL,
    REATTACK_CAP_MICRO_USD,
    FixRecord,
    FixStageError,
)
from receipts.agentloop.surrogate import SURROGATE_MODEL, LoopSurrogateBuilder
from receipts.attack import (
    CorrectnessAttacker,
    DifferentialAttacker,
    DjangoRunner,
    DockerSandbox,
    PytestRunner,
    RegressionAttacker,
    SympyRunner,
)
from receipts.contracts import (
    Arm,
    Attack,
    AttackOutcome,
    AttackPlan,
    ContextMask,
    Finding,
    Lens,
    Message,
    ModelRequest,
    ModelResponse,
    RunnerKind,
    RunResult,
    Side,
    StageCost,
    Status,
    SuiteRunner,
    Target,
    Tier,
    UncheckedReason,
    Workspace,
)
from receipts.core.cases import expected_cases
from receipts.core.glimmer_client import GlimmerClient, GlimmerConfigError
from receipts.core.meta_client import (
    HIGHEST_PRICE,
    MetaClient,
    MissingKeyError,
    SpendCapError,
    SpendMeter,
    encode_request,
)
from receipts.core.pipeline import TARGET_CAP_MICRO_USD, ArmSummary, Seams, TargetSummary, check_jobs, ledger_entry
from receipts.core.planner import TIER_A_LENSES, PlanCaps, TemperatureMode
from receipts.core.refuter import Refuter
from receipts.core.routing import ModelRouter
from receipts.core.skeptic import RerunPolicy
from receipts.eval import cli as eval_cli
from receipts.eval import vmplan
from receipts.surface.ledger import JsonlLedger

ROOT = Path(__file__).resolve().parents[1]

# Every command a merged lane adds, as `receipts <words>`: core (prices, spend, spend arm, claims, run), surface
# (replay, ledger, ledger rollup, ticket, check post, receipt html), agentloop (fix) and eval (eval and its
# subcommands), beside the root's own version.
EVAL_SUBCOMMANDS = (
    "cloud-init",
    "fetch",
    "images",
    "labels",
    "manifest",
    "patches",
    "pilot",
    "pool",
    "quote",
    "r24",
    "report",
    "rows",
    "runsc-check",
    "sample",
    "score",
    "targets",
    "test-specs",
    "vm-files",
)
COMMANDS = frozenset(
    {"version", "prices", "spend", "spend arm", "claims", "run", "replay", "ledger", "ledger rollup", "eval"}
    | {"ticket", "check", "check post", "receipt", "receipt html", "demo", "fix"}
    | {f"eval {name}" for name in EVAL_SUBCOMMANDS}
    | {"eval demo", "eval demo target", "eval demo fixcheck"}
)
# The attack lane's runner for each kind.
RUNNERS: dict[RunnerKind, type[SuiteRunner]] = {
    RunnerKind.PYTEST: PytestRunner,
    RunnerKind.DJANGO: DjangoRunner,
    RunnerKind.SYMPY: SympyRunner,
}
SPARK = Arm(name="spark")
GLIMMER = Arm(name="glimmer", attacker_model="muse-glimmer-30b")
TARGET = Target(
    id="swe-1",
    tier=Tier.A,
    repo="octo/widgets",
    base_sha="a" * 40,
    patch="--- a/w.py\n+++ b/w.py\n",
    issue="Widgets crash when the list is empty.",
)
ATTACK = Attack(
    id="correctness-1",
    claim_id="fixed_x",
    lens=Lens.CORRECTNESS,
    hypothesis="an empty list still crashes",
    mask=ContextMask.DIFF,
    temperature=0.0,
)


@pytest.fixture(autouse=True)
def offline(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """No key, no Glimmer server, a spend file of the test's own, no socket and no child process."""
    monkeypatch.delenv("FALDEN_META_API_KEY", raising=False)
    monkeypatch.delenv("RECEIPTS_GLIMMER_URL", raising=False)
    monkeypatch.setenv("RECEIPTS_SPEND_FILE", str(tmp_path / "spend" / "spend.jsonl"))

    def no_network(*args: object, **kwargs: object) -> None:
        raise AssertionError("a test reached for the network")

    def no_process(*args: object, **kwargs: object) -> None:
        raise AssertionError(f"a test started a process: {args[:1]}")

    monkeypatch.setattr(socket.socket, "connect", no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", no_network)
    monkeypatch.setattr(socket, "create_connection", no_network)
    monkeypatch.setattr(socket, "getaddrinfo", no_network)
    monkeypatch.setattr(subprocess, "Popen", no_process)


def stand_in_key(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FALDEN_META_API_KEY", f"stand-in-{secrets.token_hex(8)}")  # never a real key


def commands(parser: argparse.ArgumentParser, prefix: str = "") -> Iterator[tuple[str, argparse.ArgumentParser]]:
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                yield prefix + name, sub
                yield from commands(sub, prefix + name + " ")


class NotBuilt:
    """Stands in for a client or sandbox class that must not be built."""

    def __init__(self, *args: object, **kwargs: object) -> None:
        raise AssertionError("built before a run started")


class NoCalls:
    def complete(self, request: ModelRequest) -> ModelResponse:
        raise AssertionError("no model call in this test")


class NoWorkspace:
    """A sandbox that cannot prepare a workspace, as on a host without Docker."""

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        raise RuntimeError("no Docker in this test")

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
        raise AssertionError("nothing runs in this test")

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
        raise AssertionError("nothing runs in this test")

    def close(self, workspace: Workspace) -> None:
        raise AssertionError("nothing was prepared in this test")


# ---------------------------------------------------------------- the commands


def test_every_merged_lane_s_commands_are_registered_with_a_handler() -> None:
    found = dict(commands(cli.build_parser()))
    assert set(found) == COMMANDS
    for name, parser in found.items():
        if name not in ("spend", "ledger", "check", "receipt"):  # each has subcommands; `spend` alone also runs
            assert callable(parser.get_default("func")), name
    assert callable(found["spend"].get_default("func"))


def test_product_commands_reach_their_exact_handlers_and_default_to_preview() -> None:
    parser = cli.build_parser()
    ticket = parser.parse_args(
        [
            "ticket",
            "finding",
            "--ledger",
            "ledger",
            "--team-id",
            "11111111-1111-1111-1111-111111111111",
            "--receipt-base",
            "https://example.test",
            "--expect-org",
            "receipts",
        ]
    )
    post = parser.parse_args(
        [
            "check",
            "post",
            "--ledger",
            "ledger",
            "--targets",
            "targets",
            "--target-id",
            "t",
            "--repo",
            "o/r",
            "--head-sha",
            "a" * 40,
        ]
    )
    html = parser.parse_args(["receipt", "html", "t", "--ledger", "ledger", "--out", "receipt.html"])
    fix = parser.parse_args(
        [
            "fix",
            "finding",
            "--ledger",
            "ledger",
            "--targets",
            "targets",
            "--arms",
            "arms",
            "--out-dir",
            "out",
        ]
    )
    assert ticket.post is False and ticket.func is cli.ticket_command.ticket_command
    assert post.post is False and post.func is cli.check_command.post_command
    assert html.func is cli.receipt_command.html_command
    assert isinstance(fix.func, functools.partial) and fix.func.func is cli.fix_command.run_command


def test_new_command_secret_docs_name_the_exact_environment_and_trigger_flags() -> None:
    assert cli.__doc__ is not None
    assert "`ticket --post` reads RECEIPTS_LINEAR_API_KEY" in cli.__doc__
    assert "`check post --post` reads RECEIPTS_GITHUB_APP_ID" in cli.__doc__
    assert "RECEIPTS_GITHUB_APP_INSTALLATION_ID and RECEIPTS_GITHUB_APP_KEY_FILE" in cli.__doc__


@pytest.mark.parametrize("words", sorted(["", *COMMANDS]))
def test_help_works_for_every_command_with_no_key_network_or_docker(
    words: str, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    for name in ("MetaClient", "GlimmerClient", "DockerSandbox"):
        monkeypatch.setattr(cli, name, NotBuilt)
    with pytest.raises(SystemExit) as exit_:
        cli.main([*words.split(), "--help"])
    assert exit_.value.code == 0
    assert capsys.readouterr().out.startswith(f"usage: receipts {words}".rstrip())


def test_the_commands_that_call_no_model_run_with_no_key(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    assert cli.main(["version"]) == 0
    assert cli.main(["prices"]) == 0
    assert "List prices per 1M tokens" in capsys.readouterr().out
    assert cli.main(["spend", "arm", "--credit-confirmed"]) == 0
    assert cli.main(["spend"]) == 0
    assert "armed at" in capsys.readouterr().out
    target = tmp_path / "target.json"
    target.write_text(TARGET.model_dump_json(), encoding="utf-8")
    assert cli.main(["claims", str(target)]) == 0
    assert [claim["type"] for claim in json.loads(capsys.readouterr().out)] == ["fixed_x", "nothing_else_changed"]


def run_files(tmp_path: Path, arms: Sequence[Arm]) -> list[str]:
    run_file, arms_file = tmp_path / "run.jsonl", tmp_path / "arms.json"
    run_file.write_text(json.dumps({"target": TARGET.model_dump(mode="json"), "temperatures": "spec"}) + "\n")
    arms_file.write_text(json.dumps([arm.model_dump(mode="json") for arm in arms]))
    return ["run", "--targets", str(run_file), "--arms", str(arms_file), "--ledger-dir", str(tmp_path / "ledgers")]


def test_run_is_wired_and_reads_the_meta_key_only_once_it_starts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "DockerSandbox", NotBuilt)  # the key is checked before the sandbox is built
    assert cli.main(run_files(tmp_path, [SPARK])) == 2
    err = capsys.readouterr().err
    assert "the run could not start: MissingKeyError" in err
    assert "FALDEN_META_API_KEY" in err
    assert "not wired" not in err
    assert not (tmp_path / "ledgers").exists()  # nothing ran


def test_a_glimmer_arm_needs_the_glimmer_url_before_anything_runs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in_key(monkeypatch)
    assert cli.main(run_files(tmp_path, [SPARK, GLIMMER])) == 2
    err = capsys.readouterr().err
    assert "the run could not start: GlimmerConfigError" in err
    assert "RECEIPTS_GLIMMER_URL" in err
    assert not (tmp_path / "ledgers").exists()


# ---------------------------------------------------------------- receipts run --jobs


@pytest.mark.parametrize("jobs", [17, 64])
def test_run_refuses_more_jobs_than_the_cap_before_it_builds_anything(
    jobs: int, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in_key(monkeypatch)  # a key and good files, so only the guard stops this run
    for name in ("MetaClient", "GlimmerClient", "DockerSandbox"):
        monkeypatch.setattr(cli, name, NotBuilt)
    assert cli.main([*run_files(tmp_path, [SPARK]), "--jobs", str(jobs)]) == 2
    err = capsys.readouterr().err
    assert f"receipts run: --jobs {jobs} is refused: at most {cli.RUN_MAX_JOBS} targets run at once" in err
    assert cli.RUN_MAX_JOBS == 16
    assert "could not start" not in err  # no client and no sandbox was built
    assert not (tmp_path / "ledgers").exists()


@pytest.mark.parametrize(
    ("jobs", "said"),
    [
        ("1", "the run could not start: MissingKeyError"),
        ("2", "the run could not start: MissingKeyError"),
        ("16", "the run could not start: MissingKeyError"),
        ("0", "receipts run: --jobs is at least 1"),
    ],
)
def test_run_with_jobs_up_to_the_cap_goes_on_to_the_core_lane_s_run(
    jobs: str, said: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli, "DockerSandbox", NotBuilt)
    assert cli.main([*run_files(tmp_path, [SPARK]), "--jobs", jobs]) == 2  # no key: the core lane's run stops it
    err = capsys.readouterr().err
    assert said in err
    assert "is refused" not in err


@pytest.mark.parametrize("jobs", ["2", "16", "17"])
def test_run_help_after_a_jobs_count_needs_no_key_network_or_docker(
    jobs: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # `receipts run --jobs N --help` prints the usage and stops: the cap and the core lane's run never see it.
    for name in ("MetaClient", "GlimmerClient", "DockerSandbox"):
        monkeypatch.setattr(cli, name, NotBuilt)
    with pytest.raises(SystemExit) as exit_:
        cli.main([*run_files(tmp_path, [SPARK]), "--jobs", jobs, "--help"])
    assert exit_.value.code == 0
    out, err = capsys.readouterr()
    assert out.startswith("usage: receipts run") and "--jobs JOBS" in out
    assert "is refused" not in err
    assert not (tmp_path / "ledgers").exists()


# ---------------------------------------------------------------- the run phase's sandbox and runners


def test_the_run_phase_sandbox_is_a_new_docker_sandbox_on_every_call() -> None:
    first, second = cli.run_phase_sandbox(), cli.run_phase_sandbox()
    assert isinstance(first, DockerSandbox)
    assert isinstance(second, DockerSandbox)
    assert first is not second


def test_the_run_phase_sandbox_is_runsc_with_the_run_phase_s_flags_and_workspace(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    made: list[dict[str, object]] = []

    class Recorder:
        def __init__(self, **kwargs: object) -> None:
            made.append(kwargs)

    monkeypatch.setattr(cli, "DockerSandbox", Recorder)
    cli.run_phase_sandbox()
    # runsc, never detected; no limits or workspace path, so RUN_PHASE_LIMITS and a Tier A image's own /testbed
    assert made == [{"runsc": True}]


def test_the_run_phase_runners_are_a_fresh_mapping_on_every_call() -> None:
    first, second = cli.run_phase_runners(), cli.run_phase_runners()
    assert first is not second
    assert set(first) == set(second) == set(RunnerKind)
    for kind, runner in first.items():
        assert type(runner) is RUNNERS[kind]
        assert runner.kind is kind
        assert runner is not second[kind]
    del first[RunnerKind.PYTEST]
    assert RunnerKind.PYTEST in cli.run_phase_runners()


# ---------------------------------------------------------------- the eval lane's run phase


def eval_wiring(words: str) -> eval_cli.Wiring:
    """The eval lane's Wiring that `receipts <words>` runs on."""
    handler = dict(commands(cli.build_parser()))[words].get_default("func")
    assert isinstance(handler, functools.partial), words
    wiring = handler.keywords.get("wiring")
    assert isinstance(wiring, eval_cli.Wiring), words
    return wiring


@pytest.mark.parametrize("words", ["eval r24", "eval score"])
def test_eval_r24_and_score_run_on_the_run_phase_s_sandbox_and_runners(words: str) -> None:
    wiring = eval_wiring(words)
    assert wiring.sandbox is cli.run_phase_sandbox
    assert wiring.runners is cli.run_phase_runners


def test_eval_gets_a_new_runsc_sandbox_and_a_fresh_runner_mapping_on_every_call(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    wiring = eval_wiring("eval r24")
    first, second = wiring.sandbox(), wiring.sandbox()
    assert isinstance(first, DockerSandbox)
    assert isinstance(second, DockerSandbox)
    assert first is not second
    one, two = wiring.runners(), wiring.runners()
    assert one is not two
    assert {kind: type(runner) for kind, runner in one.items()} == RUNNERS
    for kind in RunnerKind:
        assert one[kind] is not two[kind]
    made: list[dict[str, object]] = []

    class Recorder:
        def __init__(self, **kwargs: object) -> None:
            made.append(kwargs)

    monkeypatch.setattr(cli, "DockerSandbox", Recorder)
    wiring.sandbox()
    wiring.sandbox()
    assert made == [{"runsc": True}, {"runsc": True}]  # runsc each time, never detected, never runc


# ---------------------------------------------------------------- receipts run's seams and ledgers


def test_a_run_s_seams_are_the_lanes_real_implementations(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_key(monkeypatch)
    monkeypatch.setenv("RECEIPTS_GLIMMER_URL", "http://glimmer.test:8000/v1")
    seams = cli.CoreWiring().seams([SPARK, GLIMMER])
    assert isinstance(seams.intake_client, MetaClient)
    assert seams.planner_client is seams.intake_client
    router = seams.attacker_client
    assert isinstance(router, ModelRouter)
    assert router.models == ("muse-glimmer-30b", "muse-spark-1.3")
    assert router._routes["muse-spark-1.3"] is seams.intake_client
    assert isinstance(router._routes["muse-glimmer-30b"], GlimmerClient)
    assert seams.sandbox is None and seams.runners is None  # no sandbox that the run's targets share
    assert seams.run_phase is cli.run_phase
    phase = seams.run_phase()
    assert isinstance(phase.sandbox, DockerSandbox)
    assert {kind: type(r) for kind, r in phase.runners.items()} == RUNNERS
    assert phase.release == phase.sandbox.cleanup
    check_jobs(cli.RUN_MAX_JOBS, seams)  # the core lane runs that many targets at once on these seams
    assert set(seams.attackers) == {Lens.REGRESSION, Lens.CORRECTNESS}
    # AT3 for the plan's existing-tests attack, then stage 1 and stage 2 on the regression lens for the rest.
    suite, regression_one, regression_two = seams.attackers[Lens.REGRESSION]
    assert isinstance(suite, cli.ExistingTestsOnly) and isinstance(suite.suite, RegressionAttacker)
    assert (suite.suite.wall_s, suite.suite.judge_reserve_s) == (900.0, 60.0)
    assert isinstance(regression_one, CorrectnessAttacker) and regression_one.lens is Lens.REGRESSION
    assert regression_one.run_timeout_s == 60.0
    assert isinstance(regression_two, LoopAttacker) and regression_two.lens is Lens.REGRESSION
    stage_one, stage_two = seams.attackers[Lens.CORRECTNESS]  # the attack lane's first, the surface lane's after it
    assert isinstance(stage_one, CorrectnessAttacker) and stage_one.lens is Lens.CORRECTNESS
    assert stage_one.run_timeout_s == 60.0
    assert isinstance(stage_two, LoopAttacker) and stage_two.lens is Lens.CORRECTNESS
    assert seams.case_names is expected_cases  # not passed: AT2 and AT3 repros carry Repro.cases
    assert seams.policy == RerunPolicy()
    assert seams.target_caps is TARGET_CAP_MICRO_USD
    # SK3's surrogate fixer, on the same router.
    assert isinstance(seams.surrogates, LoopSurrogateBuilder) and seams.surrogates._client is router


def test_the_surrogate_fixer_s_model_is_served_even_when_no_arm_names_it(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_key(monkeypatch)
    monkeypatch.setenv("RECEIPTS_GLIMMER_URL", "http://glimmer.test:8000/v1")
    seams = cli.CoreWiring().seams([GLIMMER])
    router = seams.attacker_client
    assert isinstance(router, ModelRouter) and SURROGATE_MODEL in router.models
    assert router._routes[SURROGATE_MODEL] is seams.intake_client
    off = cli.CoreWiring(surrogates=False).seams([GLIMMER])
    assert off.surrogates is None
    assert isinstance(off.attacker_client, ModelRouter) and off.attacker_client.models == ("muse-glimmer-30b",)


@pytest.mark.parametrize(("flag", "built"), [((), True), (("--no-surrogates",), False)])
def test_no_surrogates_reaches_the_wiring_before_the_seams_are_built(
    flag: tuple[str, ...], built: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_key(monkeypatch)
    args = cli.build_parser().parse_args([*run_files(tmp_path, [SPARK]), *flag])
    assert args.func.func is cli.at_most_max_jobs  # the jobs cap stays outermost
    choice = args.func.args[0]
    assert choice.func is cli.with_surrogate_choice
    wiring: cli.CoreWiring = choice.args[0]
    seen: list[bool] = []
    assert (
        cli.with_surrogate_choice(
            wiring, lambda a: seen.append(wiring.seams([SPARK]).surrogates is not None) or 0, args
        )
        == 0
    )
    assert seen == [built]


def test_the_refuter_is_wonder_on_the_meta_client_unless_turned_off(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_key(monkeypatch)
    seams = cli.CoreWiring().seams([SPARK])
    assert isinstance(seams.refuter, Refuter) and seams.refuter._client is seams.intake_client
    assert seams.refuter.model == "muse-wonder-1.0"
    assert cli.CoreWiring(refuter=False).seams([SPARK]).refuter is None


@pytest.mark.parametrize(("flag", "built"), [((), True), (("--no-refuter",), False)])
def test_no_refuter_reaches_the_wiring_before_the_seams_are_built(
    flag: tuple[str, ...], built: bool, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_key(monkeypatch)
    args = cli.build_parser().parse_args([*run_files(tmp_path, [SPARK]), *flag])
    assert args.func.func is cli.at_most_max_jobs  # the jobs cap stays outermost
    surrogates = args.func.args[0]
    assert surrogates.func is cli.with_surrogate_choice
    choice = surrogates.args[1]
    assert choice.func is cli.with_refuter_choice
    wiring: cli.CoreWiring = choice.args[0]
    assert wiring is surrogates.args[0]  # one wiring for both choices
    seen: list[bool] = []
    assert (
        cli.with_refuter_choice(wiring, lambda a: seen.append(wiring.seams([SPARK]).refuter is not None) or 0, args)
        == 0
    )
    assert seen == [built]


@pytest.mark.parametrize(
    ("flag", "lenses"),
    [
        ((), (Lens.REGRESSION, Lens.CORRECTNESS)),
        (("--lenses", "regression,correctness,differential"), (Lens.REGRESSION, Lens.CORRECTNESS, Lens.DIFFERENTIAL)),
    ],
)
def test_lenses_reach_the_wiring_and_only_differential_adds_an_attacker(
    flag: tuple[str, ...], lenses: tuple[Lens, ...], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_key(monkeypatch)
    args = cli.build_parser().parse_args([*run_files(tmp_path, [SPARK]), *flag])
    assert args.lenses == lenses
    refuter = args.func.args[0].args[1]
    choice = refuter.args[1]
    assert choice.func is cli.with_lens_choice
    wiring: cli.CoreWiring = choice.args[0]
    assert wiring is refuter.args[0]  # one wiring for every choice
    seen: list[object] = []
    assert cli.with_lens_choice(wiring, lambda a: seen.append(wiring.seams([SPARK])) or 0, args) == 0
    (seams,) = seen
    assert isinstance(seams, Seams) and seams.tier_a_lenses == lenses
    if Lens.DIFFERENTIAL in lenses:
        (at4,) = seams.attackers[Lens.DIFFERENTIAL]
        assert isinstance(at4, DifferentialAttacker)
    else:  # the default: Tier A planned as before, and no differential attacker for any tier
        assert set(seams.attackers) == {Lens.REGRESSION, Lens.CORRECTNESS}
        assert seams.tier_a_lenses == TIER_A_LENSES


@pytest.mark.parametrize("words", ["mutation", "differential,differential", "", "nope"])
def test_lenses_refuses_a_lens_no_attacker_takes_a_repeat_or_none(
    words: str, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args([*run_files(tmp_path, [SPARK]), "--lenses", words])
    assert "--lenses" in capsys.readouterr().err


def test_each_target_gets_its_own_runsc_sandbox_and_runners(monkeypatch: pytest.MonkeyPatch) -> None:
    first, second = cli.run_phase(), cli.run_phase()
    assert isinstance(first.sandbox, DockerSandbox) and first.sandbox is not second.sandbox
    assert first.runners is not second.runners
    for kind in RunnerKind:
        assert first.runners[kind] is not second.runners[kind]
    made: list[dict[str, object]] = []

    class Recorder:
        def __init__(self, **kwargs: object) -> None:
            made.append(kwargs)

        def cleanup(self) -> None:
            pass

    monkeypatch.setattr(cli, "DockerSandbox", Recorder)
    cli.run_phase()
    cli.run_phase()
    assert made == [{"runsc": True}, {"runsc": True}]  # runsc each time, never detected, never runc


def test_run_builds_each_target_s_sandbox_when_it_starts_and_cleans_it_up_when_it_is_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Two targets, two at once. The spend file is not armed, so every model call is refused before any request:
    # no plan is made and no sandbox is used, but each target still gets a sandbox of its own and cleans it up.
    stand_in_key(monkeypatch)
    events: list[object] = []

    class Recorder(NoWorkspace):
        def __init__(self, **kwargs: object) -> None:
            events.append(kwargs)

        def cleanup(self) -> None:
            events.append("cleanup")

    monkeypatch.setattr(cli, "DockerSandbox", Recorder)
    argv = run_files(tmp_path, [SPARK])
    two = [TARGET, TARGET.model_copy(update={"id": "swe-2"})]
    (tmp_path / "run.jsonl").write_text(
        "".join(json.dumps({"target": t.model_dump(mode="json"), "temperatures": "spec"}) + "\n" for t in two)
    )
    assert cli.main([*argv, "--jobs", "2"]) == 0
    assert "2 of 2 targets done, 0 skipped, 0 failed" in capsys.readouterr().out
    assert sorted(map(str, events)) == sorted(map(str, [{"runsc": True}, {"runsc": True}, "cleanup", "cleanup"]))


def test_a_run_s_seams_build_no_sandbox_until_a_target_starts(monkeypatch: pytest.MonkeyPatch) -> None:
    # Round 5's seams built the one sandbox every target shared; now each target builds its own when it starts.
    stand_in_key(monkeypatch)
    monkeypatch.setattr(cli, "DockerSandbox", NotBuilt)
    seams = cli.CoreWiring().seams([SPARK])
    assert seams.run_phase is cli.run_phase
    with pytest.raises(AssertionError, match="built before a run started"):
        seams.run_phase()  # the factory is the only thing that builds one


class Sandboxes:
    """Stands in for DockerSandbox in a run: records each sandbox the run builds, the thread that built it and the
    thread that cleaned it up, and how many were alive at once. The first `wave` sandboxes wait for each other in
    the constructor, so a run that builds them one target at a time breaks the barrier and fails."""

    def __init__(self, wave: int) -> None:
        self.gate = threading.Barrier(wave, timeout=20)
        self.wave = wave
        self.lock = threading.Lock()
        self.built: list[tuple[object, int, dict[str, object]]] = []
        self.cleaned: list[tuple[object, int]] = []
        self.alive = 0
        self.peak = 0

    def stand_in(self) -> type[NoWorkspace]:
        record = self

        class Recorded(NoWorkspace):
            def __init__(self, **kwargs: object) -> None:
                with record.lock:
                    record.built.append((self, threading.get_ident(), kwargs))
                    first_wave = len(record.built) <= record.wave
                    record.alive += 1
                    record.peak = max(record.peak, record.alive)
                if first_wave:
                    record.gate.wait()

            def cleanup(self) -> None:
                with record.lock:
                    record.cleaned.append((self, threading.get_ident()))
                    record.alive -= 1

        return Recorded

    def check(self, targets: int) -> None:
        """Every target built one runsc sandbox, and the thread that built each one cleaned it up, once."""
        assert len(self.built) == targets
        assert all(kwargs == {"runsc": True} for _, _, kwargs in self.built)
        assert sorted((id(s), t) for s, t, _ in self.built) == sorted((id(s), t) for s, t in self.cleaned)
        assert self.alive == 0


def test_two_jobs_run_two_targets_at_once_each_on_a_sandbox_its_own_target_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # Both targets build their sandboxes at the same moment, or the barrier breaks and the run fails; each target's
    # worker cleans up the sandbox it built, and none is left alive when the run ends.
    stand_in_key(monkeypatch)
    sandboxes = Sandboxes(wave=2)
    monkeypatch.setattr(cli, "DockerSandbox", sandboxes.stand_in())
    argv = run_files(tmp_path, [SPARK])
    two = [TARGET, TARGET.model_copy(update={"id": "swe-2"})]
    (tmp_path / "run.jsonl").write_text(
        "".join(json.dumps({"target": t.model_dump(mode="json"), "temperatures": "spec"}) + "\n" for t in two)
    )
    assert cli.main([*argv, "--jobs", "2"]) == 0
    assert "2 of 2 targets done, 0 skipped, 0 failed" in capsys.readouterr().out
    sandboxes.check(targets=2)
    assert sandboxes.peak == 2
    assert len({thread for _, thread, _ in sandboxes.built}) == 2  # two workers, one target each


# ---------------------------------------------------------------- the eval lane's runbook, through this root


def runbook_runs() -> dict[str, list[str]]:
    """Each runbook step (receipts.eval.vmplan.RUNBOOK) that runs the receipts command, by its first word after
    `receipts`, as the words after it."""
    found: dict[str, list[str]] = {}
    for step in vmplan.RUNBOOK:
        words = shlex.split(step.command)
        if words[:2] == ["uv", "run"] and "receipts" in words:
            rest = words[words.index("receipts") + 1 :]
            found[" ".join(rest[:2]) if rest[0] in ("eval", "spend") else rest[0]] = rest
    return found


def test_every_runbook_step_that_runs_receipts_reaches_its_wired_handler() -> None:
    steps = runbook_runs()
    assert sorted(steps) == ["eval r24", "eval score", "run", "spend arm"]
    parser = cli.build_parser()
    for name in ("eval r24", "eval score"):
        handler = parser.parse_args(steps[name]).func
        assert isinstance(handler, functools.partial), name
        wiring = handler.keywords["wiring"]
        assert (wiring.sandbox, wiring.runners) == (cli.run_phase_sandbox, cli.run_phase_runners), name
    run = parser.parse_args(steps["run"])
    assert isinstance(run.func, functools.partial) and run.func.func is cli.at_most_max_jobs
    assert 1 < run.jobs <= cli.RUN_MAX_JOBS  # the pilot's --jobs gets through the cap
    assert parser.parse_args(steps["spend arm"]).credit_confirmed


def test_the_runbook_s_pilot_line_runs_its_targets_that_many_at_once_each_on_a_sandbox_of_its_own(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    # The eval lane's pilot line word for word, but for its ledger dir, run from the repository root on the committed
    # pilot files: the cap lets its --jobs through, that many targets run at once (the barrier), never more (peak),
    # and each builds a runsc sandbox of its own when it starts and cleans it up when it is done. Nothing reaches a
    # model: the spend file is not armed, so every call is refused before any request.
    words = runbook_runs()["run"]
    jobs = cli.build_parser().parse_args(words).jobs
    targets = len((ROOT / words[words.index("--targets") + 1]).read_text().splitlines())
    assert targets > jobs == 16
    stand_in_key(monkeypatch)
    monkeypatch.setenv("RECEIPTS_GLIMMER_URL", "http://glimmer.test:8000/v1")
    monkeypatch.chdir(ROOT)
    tried: list[str] = []  # a refusal a client caught would hide from the exit code, so each try is recorded

    def refused(*args: object, **kwargs: object) -> None:
        tried.append(repr(args[:1]))
        raise AssertionError("a test reached for the network or a process")

    for owner, name in [(socket.socket, "connect"), (socket.socket, "connect_ex")]:
        monkeypatch.setattr(owner, name, refused)
    for module, name in [(socket, "create_connection"), (socket, "getaddrinfo"), (subprocess, "Popen")]:
        monkeypatch.setattr(module, name, refused)
    sandboxes = Sandboxes(wave=jobs)
    monkeypatch.setattr(cli, "DockerSandbox", sandboxes.stand_in())
    at = words.index("--ledger-dir") + 1
    assert cli.main([*words[:at], str(tmp_path / "pilot"), *words[at + 1 :]]) == 0
    assert f"{targets} of {targets} targets done, 0 skipped, 0 failed" in capsys.readouterr().out
    sandboxes.check(targets=targets)
    assert sandboxes.peak == jobs
    assert tried == []


def test_a_spark_only_run_builds_no_glimmer_client(monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_key(monkeypatch)
    monkeypatch.setattr(cli, "GlimmerClient", NotBuilt)
    router = cli.CoreWiring().seams([SPARK]).attacker_client
    assert isinstance(router, ModelRouter)
    assert router.models == ("muse-spark-1.3",)


def test_the_seams_read_each_endpoint_by_its_exact_name(monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(MissingKeyError, match="FALDEN_META_API_KEY"):
        cli.CoreWiring().seams([SPARK])
    stand_in_key(monkeypatch)
    with pytest.raises(GlimmerConfigError, match="RECEIPTS_GLIMMER_URL"):
        cli.CoreWiring().seams([GLIMMER])


def test_each_arm_s_ledger_is_the_surface_lane_s_jsonl_ledger(tmp_path: Path) -> None:
    path = tmp_path / "ledgers" / "spark.jsonl"
    ledger = cli.CoreWiring().ledger(path)
    assert isinstance(ledger, JsonlLedger)
    assert ledger.path == path
    assert path.is_file()
    assert ledger.entries() == ()


def test_stage_two_traces_land_beside_the_ledgers_and_a_rollup_reads_only_the_arms(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    stand_in_key(monkeypatch)
    wiring = cli.CoreWiring()
    seams = wiring.seams([SPARK])
    ledger_dir = tmp_path / "ledgers"
    wiring.ledger(ledger_dir / "spark.jsonl")
    stage_two = seams.attackers[Lens.CORRECTNESS][1]
    outcome = stage_two.attack(
        TARGET, ATTACK, arm=SPARK, client=NoCalls(), sandbox=NoWorkspace(), runner=PytestRunner()
    )
    assert [u.reason for u in outcome.unchecked] == [UncheckedReason.NO_ENV]
    lines = (ledger_dir / cli.TRACES_DIR / "spark.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    trace = AttackTrace.model_validate_json(lines[0])
    assert (trace.target_id, trace.attack_id, trace.arm) == (TARGET.id, ATTACK.id, SPARK.name)
    capsys.readouterr()
    assert cli.main(["ledger", "rollup", "--ledger", str(ledger_dir), "--json"]) == 0
    read = json.loads(capsys.readouterr().out)["ledgers"]
    assert [(ledger["arm"], ledger["path"]) for ledger in read] == [("spark", str(ledger_dir / "spark.jsonl"))]


def test_a_trace_with_no_open_ledger_or_a_bad_arm_name_is_refused(tmp_path: Path) -> None:
    wiring = cli.CoreWiring()
    trace = AttackTrace(
        target_id=TARGET.id, attack_id=ATTACK.id, arm=SPARK.name, model=SPARK.attacker_model, explore=None
    )
    with pytest.raises(RuntimeError, match="no arm ledger is open"):
        wiring.trace(trace)
    wiring.ledger(tmp_path / "spark.jsonl")
    with pytest.raises(ValueError, match="not an arm name"):
        wiring.trace(trace.model_copy(update={"arm": "../spark"}))
    assert not (tmp_path / cli.TRACES_DIR).exists()


def test_at3_takes_only_the_plan_s_existing_tests_attack() -> None:
    # Fix round 5: AT3 runs once per target. Every other regression attack goes past it, untouched, to the model
    # attackers that write new tests of unchanged behavior.
    seen: list[str] = []

    class Suite:
        lens = Lens.REGRESSION

        def attack(self, target: Target, attack: Attack, **kw: object) -> AttackOutcome:
            seen.append(attack.id)
            return AttackOutcome(attack_id=attack.id, costs=())

    target = Target(id="t", tier=Tier.A, repo="o/r", base_sha="a" * 40, patch="diff --git a/x b/x\n")
    plain = {"claim_id": "nothing_else_changed", "lens": Lens.REGRESSION, "hypothesis": "h", "mask": ContextMask.DIFF}
    suite = Attack(id="regression-1", temperature=0.0, existing_tests=True, **plain)  # type: ignore[arg-type]
    model = Attack(id="regression-2", temperature=0.8, **plain)  # type: ignore[arg-type]
    route = cli.ExistingTestsOnly(Suite())  # type: ignore[arg-type]
    kw: dict[str, object] = {"arm": SPARK, "client": None, "sandbox": None, "runner": None}
    assert route.attack(target, model, **kw) == AttackOutcome(attack_id="regression-2")  # type: ignore[arg-type]
    assert route.attack(target, suite, **kw).attack_id == "regression-1"  # type: ignore[arg-type]
    assert seen == ["regression-1"]


# ---------------------------------------------------------------- receipts fix (FX1)

FRESH = TARGET.model_copy(update={"id": "swe-1~fix1"})


def fix_args(tmp_path: Path) -> argparse.Namespace:
    argv = ["fix", "swe-1/fixed_x/correctness-1", "--ledger", str(tmp_path / "spark.jsonl")]
    argv += ["--targets", str(tmp_path / "run.jsonl"), "--arms", str(tmp_path / "arms.json")]
    return cli.build_parser().parse_args([*argv, "--out-dir", str(tmp_path / "out")])


def test_fix_is_wired_with_spark_on_the_meta_client_a_run_phase_and_the_fresh_attack(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    args = fix_args(tmp_path)
    wiring = args.func.keywords["wiring"]
    assert isinstance(wiring, FixWiring)
    assert wiring.run_phase is cli.run_phase and wiring.reattack is cli.FixReAttack
    with pytest.raises(MissingKeyError):  # the key is read only when the fixer's client is built
        wiring.client()
    stand_in_key(monkeypatch)
    client = wiring.client()
    assert isinstance(client, ModelRouter) and client.models == (FIXER_MODEL,)
    assert isinstance(client._routes[FIXER_MODEL], MetaClient)
    assert cli.FixReAttack(tmp_path / "out").out_dir == tmp_path / "out" / cli.REATTACK_DIR


def finding_for(target_id: str, third: str, status: Status, micro_usd: int) -> Finding:
    return Finding(
        id=f"{target_id}/fixed_x/{third}",
        target_id=target_id,
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        status=status,
        detail="a stand-in",
        costs=(StageCost(stage="attack", micro_usd=micro_usd),),
    )


def fresh_summary(*, failed: bool = False) -> TargetSummary:
    return TargetSummary(
        receipts_version="0",
        target_id=FRESH.id,
        tier=Tier.A,
        temperatures=TemperatureMode.SPEC,
        status="failed" if failed else "done",
        error="RuntimeError: boom" if failed else "",
        started_at="2026-10-30T00:00:00Z",
        wall_seconds=0.0,
        plan=AttackPlan(target_id=FRESH.id, attacks=()),
        costs=(StageCost(stage="intake", micro_usd=11), StageCost(stage="plan", micro_usd=13)),
    )


def test_the_fresh_attack_runs_the_pipeline_and_counts_planning_and_every_finding(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_key(monkeypatch)
    seen: list[tuple[str, tuple[Arm, ...], TemperatureMode]] = []
    old = finding_for(FRESH.id, "earlier", Status.SUSPECTED, 999)
    out = tmp_path / "out"
    JsonlLedger(out / cli.REATTACK_DIR / "earlier-patch" / "spark.jsonl").append(ledger_entry(old, datetime.now(UTC)))

    def fake_run_target(target: Target, **kwargs: Any) -> TargetSummary:
        arms = tuple(kwargs["arms"])
        seen.append((target.id, arms, kwargs["temperatures"]))
        assert isinstance(kwargs["seams"], Seams) and kwargs["seams"].run_phase is cli.run_phase
        # REG-616: the fresh attack plans with no caps, so every attack its plan holds runs.
        assert kwargs["plan_caps"] is cli.REATTACK_PLAN_CAPS == PlanCaps(max_lenses=None, max_attacks_per_lens=None)
        ledger = kwargs["ledgers"]["spark"]
        first = finding_for(target.id, "correctness-1", Status.SUSPECTED, 5)
        ledger.append(ledger_entry(first, datetime.now(UTC)))
        final = first.model_copy(
            update={"status": Status.NOT_REPRODUCED, "costs": (StageCost(stage="x", micro_usd=7),)}
        )
        ledger.append(ledger_entry(final, datetime.now(UTC)))
        return fresh_summary()

    monkeypatch.setattr(cli, "run_target", fake_run_target)
    arm = SPARK.model_copy(update={"seed": 1000})
    result = cli.FixReAttack(out)(FRESH, arm, budget_micro_usd=2_000_000)
    assert seen == [(FRESH.id, (arm,), TemperatureMode.SPEC)]  # the fixer's fresh seed reaches the pipeline
    assert [(f.id, f.status) for f in result.findings] == [(f"{FRESH.id}/fixed_x/correctness-1", Status.NOT_REPRODUCED)]
    assert result.micro_usd == 11 + 13 + 5 + 7  # planning and intake too; never the earlier fix's 999
    assert result.out_dir is not None
    assert (Path(result.out_dir) / "summaries" / f"{FRESH.id}.json").exists()


def test_a_failed_fresh_attack_raises_its_partial_total(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_key(monkeypatch)

    def fake_run_target(target: Target, **kwargs: Any) -> TargetSummary:
        kwargs["ledgers"]["spark"].append(
            ledger_entry(finding_for(target.id, "correctness-1", Status.NOT_REPRODUCED, 17), datetime.now(UTC))
        )
        kwargs["on_failure"](fresh_summary(failed=True))
        raise RuntimeError("boom")

    monkeypatch.setattr(cli, "run_target", fake_run_target)
    with pytest.raises(FixStageError) as raised:
        cli.FixReAttack(tmp_path / "out")(FRESH, SPARK, budget_micro_usd=2_000_000)
    assert sum(cost.micro_usd for cost in raised.value.costs) == 11 + 13 + 17
    assert raised.value.out_dir is not None
    assert len(list(Path(raised.value.out_dir).glob("summaries/*.json"))) == 1


def test_a_fresh_attack_that_cannot_start_spent_nothing(tmp_path: Path) -> None:
    with pytest.raises(FixStageError) as raised:  # no key: the seams are never built
        cli.FixReAttack(tmp_path / "out")(FRESH, SPARK, budget_micro_usd=2_000_000)
    assert raised.value.costs == ()
    assert isinstance(raised.value.__cause__, MissingKeyError)


@pytest.mark.parametrize("path", ["fix", "run", "defaults"])
@pytest.mark.parametrize("lens", [Lens.CORRECTNESS, Lens.REGRESSION])
def test_only_the_fix_fresh_attack_raises_stage_one_and_write_output_limits(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, path: str, lens: Lens
) -> None:
    from tests.agentloop.test_loop import FakeClient, FakeRunner, FakeSandbox, passed, response, submit

    stand_in_key(monkeypatch)
    requests: list[ModelRequest] = []

    def exercise(attackers: Sequence[Any]) -> None:
        client = FakeClient(
            response("{"),  # stage 1 retries with the same output limit
            response('{"tests": [{"path": "tests/test_limit.py", "code": "def test_limit(): pass", "why": "limit"}]}'),
            submit(),
            response("{"),  # the write retry keeps its separate limit too
            response('{"tests": [{"path": "tests/test_receipts_limit.py", "content": "def test_limit(): pass"}]}'),
        )
        attack = ATTACK.model_copy(update={"lens": lens})
        for attacker in attackers:
            outcome = attacker.attack(
                FRESH, attack, arm=SPARK, client=client, sandbox=FakeSandbox(tests=[passed()]), runner=FakeRunner()
            )
            assert not outcome.unchecked
        assert not client.answers
        requests.extend(client.requests)

    def exercise_seams(seams: Seams) -> None:
        exercise(seams.attackers[lens][-2:])

    if path == "fix":

        def fake_run_target(target: Target, **kwargs: Any) -> TargetSummary:
            exercise_seams(kwargs["seams"])
            return fresh_summary()

        monkeypatch.setattr(cli, "run_target", fake_run_target)
        cli.FixReAttack(tmp_path / "out")(FRESH, SPARK, budget_micro_usd=REATTACK_CAP_MICRO_USD)
    elif path == "run":
        exercise_seams(cli.CoreWiring().seams([SPARK]))
    else:
        exercise((CorrectnessAttacker(lens=lens), LoopAttacker(lens=lens)))

    stage_one, write = (32_000, 16_384) if path == "fix" else (16_000, 8_192)
    assert [(request.stage, request.max_tokens) for request in requests] == [
        ("at2_stage1", stage_one),
        ("at2_stage1", stage_one),
        ("attack.correctness.explore", 8_192),
        ("attack.correctness.write", write),
        ("attack.correctness.write", write),
    ]


@pytest.mark.parametrize(
    ("stage", "request_bytes", "old_tokens", "new_tokens", "old_hold", "new_hold"),
    [
        ("at2_stage1", 2_591, 16_000, 32_000, 330_364, 650_364),
        ("attack.correctness.write", 3_302, 8_192, 16_384, 177_048, 340_892),
    ],
)
def test_fix_fresh_attack_cap_fits_the_larger_last_call_hold(
    tmp_path: Path,
    stage: str,
    request_bytes: int,
    old_tokens: int,
    new_tokens: int,
    old_hold: int,
    new_hold: int,
) -> None:
    # Fixture-sized requests (round 1's probe): the write limit gains one encoded digit. Calls are sequential, so
    # only one hold is open. Reserve against the smaller of the fresh cap and what one full fixer iteration leaves.
    assert HIGHEST_PRICE.worst_case_micro_usd(request_bytes, old_tokens) == old_hold
    new_bytes = request_bytes + len(str(new_tokens)) - len(str(old_tokens))
    assert HIGHEST_PRICE.worst_case_micro_usd(new_bytes, new_tokens) == new_hold
    parent = SpendMeter(tmp_path / "spend.jsonl")
    parent.arm(credit_confirmed=True)
    budget = min(REATTACK_CAP_MICRO_USD, FIX_CAP_MICRO_USD - 250_000)
    meter = cli._FreshAttackMeter(parent, budget)
    prior = meter.reserve(3_107_000, stage="earlier", model=SPARK.attacker_model)
    meter.settle(prior, 3_107_000, estimated=False)
    assert meter.remaining() == 1_393_000
    last = meter.reserve(new_hold, stage=stage, model=SPARK.attacker_model)
    assert meter.remaining() == 1_393_000 - new_hold
    assert not meter.refused
    meter.settle(last, 0, estimated=False)
    assert parent.held_micro_usd == 0


# What intake answers for the real fresh attack below, in order, each claim quoted in the target's body.
TWO_CLAIMS = (("fixed_x", "The value must be correct."), ("tests_pass", "All tests pass."))
# A fresh attack shaped like pandas #64677's (REG-616): six claims of three masks each, 18 planned attacks.
SIX_CLAIMS = (
    *TWO_CLAIMS,
    ("adds_behavior", "It adds a value flag."),
    ("perf", "It runs much faster now."),
    ("nothing_else_changed", "Nothing else changes here."),
    ("backward_compatible", "Old callers keep working fine."),
)
REGRESSION_KINDS = frozenset({"tests_pass", "nothing_else_changed", "backward_compatible"})


def real_fix(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    tier: Tier,
    claims: Sequence[tuple[str, str]],
    attack_cost: int,
    attack_max_tokens: int,
    fixer_cost: int,
    prices: Mapping[str, int] | None = None,
) -> tuple[FixRecord, dict[str, Any], list[str], SpendMeter]:
    """`receipts fix` end to end offline: run_fix with the real FixReAttack and pipeline on a scripted Meta transport.
    Intake answers `claims`, the planner proposes every mask on the claim type's lens, and each attack is one model
    call that asks for `attack_max_tokens`, so the meter holds that worst case first, and bills `attack_cost`; every
    other call bills its stage's entry in `prices`, else 1_000. It returns the record, the fresh attack's summary, the
    stages sent and the global meter."""
    from receipts.core.pipeline import RunPhase
    from tests.agentloop.test_fixloop import Fixer, execute
    from tests.agentloop.test_fixloop import target as fix_target
    from tests.core.test_meta_client import completion
    from tests.core.test_pipeline import FakeSandbox, Runner

    parent = SpendMeter(tmp_path / "global-spend.jsonl")
    parent.arm(credit_confirmed=True)
    sent: list[str] = []
    billed = dict(prices or {})

    class Models:
        def __init__(self, *, meter: SpendMeter | None = None) -> None:
            self.request: ModelRequest | None = None
            self.inner = MetaClient(
                api_key=f"stand-in-{secrets.token_hex(8)}",
                meter=meter or parent,
                transport=httpx.MockTransport(self.reply),
                sleep=lambda _: None,
            )

        def reply(self, wire: httpx.Request) -> httpx.Response:
            assert self.request is not None
            request = self.request
            sent.append(request.stage)
            if request.stage == "intake":
                text = json.dumps(
                    {
                        "claims": [
                            {"type": kind, "source": "pr_body", "quote": quote, "text": "t"} for kind, quote in claims
                        ]
                    }
                )
            elif request.stage == "plan":
                import re

                planned = re.findall(r'<claim id="([^"]+)" type="([^"]+)" attacks="(\d)">', request.messages[1].content)
                text = json.dumps(
                    {
                        "claims": [
                            {
                                "claim_id": cid,
                                "attacks": [
                                    {
                                        "lens": "regression" if kind in REGRESSION_KINDS else "correctness",
                                        "hypothesis": "h",
                                    }
                                ]
                                * int(k),
                            }
                            for cid, kind, k in planned
                        ]
                    }
                )
            else:
                text = "no counterexample"
            amount = attack_cost if request.stage == "attack" else billed.get(request.stage, 1_000)
            return completion(text, prompt=0, output=amount // 20, model="muse-wonder-1.0")

        def complete(self, request: ModelRequest) -> ModelResponse:
            self.request = request
            return self.inner.complete(request)

    class SpendingAttacker:
        def __init__(self, *, lens: Lens = Lens.CORRECTNESS, **kwargs: Any) -> None:
            self.lens = lens

        def attack(self, target: Target, attack: Attack, *, arm: Arm, client: Any, **kwargs: Any) -> AttackOutcome:
            response = client.complete(
                ModelRequest(
                    stage="attack",
                    model=arm.attacker_model,
                    messages=(Message(role="user", content="find a counterexample"),),
                    max_tokens=attack_max_tokens,
                    temperature=0.0,
                )
            )
            assert response.cost is not None
            return AttackOutcome(attack_id=attack.id, costs=(response.cost,))

    class NoAttack(SpendingAttacker):
        def attack(self, target: Target, attack: Attack, **kwargs: Any) -> AttackOutcome:
            return AttackOutcome(attack_id=attack.id)

    monkeypatch.setattr(cli, "MetaClient", Models)
    monkeypatch.setattr(cli, "shared_meter", lambda: parent)
    monkeypatch.setattr(cli, "CorrectnessAttacker", SpendingAttacker)
    monkeypatch.setattr(cli, "RegressionAttacker", NoAttack)
    monkeypatch.setattr(cli, "LoopAttacker", NoAttack)
    monkeypatch.setattr(
        cli, "run_phase", lambda: RunPhase(sandbox=FakeSandbox(), runners={RunnerKind.PYTEST: Runner()}, release=None)
    )
    out_dir = tmp_path / "out"
    record = execute(
        tmp_path,
        target=fix_target().model_copy(update={"tier": tier, "body": " ".join(quote for _, quote in claims)}),
        arm=Arm(name="spark", attacker_model="muse-wonder-1.0", seed=7),
        fixer=Fixer(cost=fixer_cost),
        reattack=cli.FixReAttack(out_dir),
    )
    assert record.reattack_out_dir is not None
    summary = json.loads(next(Path(record.reattack_out_dir).glob("summaries/*.json")).read_text())
    return record, summary, sent, parent


@pytest.mark.parametrize(
    ("tier", "attack_cost", "fixer_cost", "status"),
    [
        (Tier.B, 600_000, 1_000, "survived_fresh_attack"),  # six $0.60 attacks fit the fresh attack's $4.50
        (Tier.C, 800_000, 1_000, "unchecked"),  # the sixth attack's $0.80 hold does not fit, so the meter refuses it
        (Tier.B, 100_000, 1_000, "survived_fresh_attack"),
        (Tier.C, 600_000, 1_500_000, "unchecked"),  # the fixer's $1.50 leaves $3.25, too little for six $0.60 attacks
    ],
)
def test_real_fresh_attack_reserves_every_call_and_stays_inside_both_caps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    tier: Tier,
    attack_cost: int,
    fixer_cost: int,
    status: str,
) -> None:
    record, summary, sent, parent = real_fix(
        tmp_path,
        monkeypatch,
        tier=tier,
        claims=TWO_CLAIMS,
        attack_cost=attack_cost,
        attack_max_tokens=attack_cost // 20,
        fixer_cost=fixer_cost,
    )
    fresh = sum(c.micro_usd for c in record.costs if c.stage == "fix-reattack")
    assert fresh == parent.spent_micro_usd <= min(REATTACK_CAP_MICRO_USD, FIX_CAP_MICRO_USD - fixer_cost)
    assert record.micro_usd == fixer_cost + fresh <= FIX_CAP_MICRO_USD
    assert parent.held_micro_usd == 0
    assert len(summary["plan"]["attacks"]) == 6
    assert record.status.value == status
    if status == "survived_fresh_attack":
        assert sent.count("attack") == 6
    else:
        assert record.reason == "budget"
        assert summary["arms"][0]["statuses"]["unchecked"] >= 1
    print(
        f"probe {tier.value} {attack_cost}: {record.status.value}/{record.reason}; "
        f"fresh={fresh}; total={record.micro_usd}"
    )


def test_the_fix_caps_fit_every_attack_of_a_fresh_attack_shaped_like_pandas_64677(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REG-616: 18 planned attacks, each billed at #64677's measured maximum of $0.1708 after a $0.48 hold (about
    # what a refuter call holds late in #64677's run; an explore call with a longer history holds more), and intake
    # and plan billed as measured. All 18 run inside the fresh attack's $4.50 and the fix survives; the whole fix
    # stays inside $4.75.
    record, summary, sent, parent = real_fix(
        tmp_path,
        monkeypatch,
        tier=Tier.B,
        claims=SIX_CLAIMS,
        attack_cost=170_800,
        attack_max_tokens=24_000,
        fixer_cost=116_000,
        prices={"intake": 9_700, "plan": 193_700},
    )
    assert len(summary["plan"]["attacks"]) == 18 and summary["plan"]["unchecked"] == []
    assert sorted(set(sent)) == ["attack", "intake", "plan"] and sent.count("attack") == 18
    assert record.status.value == "survived_fresh_attack"
    fresh = sum(c.micro_usd for c in record.costs if c.stage == "fix-reattack")
    assert fresh == parent.spent_micro_usd == 9_700 + 193_700 + 18 * 170_800
    assert record.micro_usd == 116_000 + fresh <= FIX_CAP_MICRO_USD
    assert parent.held_micro_usd == 0


def test_fresh_budget_settles_holds_and_refuses_a_retry_before_transport(tmp_path: Path) -> None:
    parent = SpendMeter(tmp_path / "spend.jsonl")
    parent.arm(credit_confirmed=True)
    request = ModelRequest(
        stage="refuter",
        model="muse-wonder-1.0",
        messages=(Message(role="user", content="judge"),),
        max_tokens=100,
        temperature=0.0,
    )
    worst = HIGHEST_PRICE.worst_case_micro_usd(len(encode_request(request)), request.max_tokens)
    meter = cli._FreshAttackMeter(parent, worst)
    sent: list[httpx.Request] = []

    def timeout(wire: httpx.Request) -> httpx.Response:
        sent.append(wire)
        raise httpx.ReadTimeout("scripted timeout", request=wire)

    client = MetaClient(
        api_key=f"stand-in-{secrets.token_hex(8)}",
        meter=meter,
        transport=httpx.MockTransport(timeout),
        sleep=lambda _: None,
    )
    with pytest.raises(SpendCapError) as refused:
        client.complete(request)
    assert refused.value.reason is UncheckedReason.BUDGET
    assert len(sent) == 1 and meter.refused
    assert meter.spent_micro_usd == parent.spent_micro_usd == worst
    assert meter.held_micro_usd == parent.held_micro_usd == 0
    assert meter.remaining() == 0


def test_fresh_budget_releases_unused_holds_and_preserves_the_global_cap(tmp_path: Path) -> None:
    parent = SpendMeter(tmp_path / "spend.jsonl", cap_micro_usd=1_000)
    parent.arm(credit_confirmed=True)
    meter = cli._FreshAttackMeter(parent, 2_000)
    hold = meter.reserve(900, stage="intake", model="muse-spark-1.3")
    meter.settle(hold, 100, estimated=False)
    assert meter.remaining() == 1_900
    second = meter.reserve(900, stage="surrogate", model="muse-spark-1.3")
    with pytest.raises(SpendCapError):
        meter.reserve(1, stage="gate", model="muse-spark-1.3")
    assert meter.refused  # Only the parent meter refused; the fresh allowance still fits.
    assert meter.held_micro_usd == 900
    meter.settle(second, 0, estimated=False)
    assert meter.remaining() == 1_900 and parent.spent_micro_usd == 100


@pytest.mark.parametrize("crosses_cap", [False, True])
def test_fixer_retry_reserves_against_the_fix_cap_before_transport(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, crosses_cap: bool
) -> None:
    from tests.agentloop.test_fixloop import Fixer, Reattack, execute
    from tests.core.test_meta_client import completion

    parent = SpendMeter(tmp_path / "parent.jsonl")
    parent.arm(credit_confirmed=True)
    sent: list[httpx.Request] = []
    clients: list[MetaClient] = []

    def reply(wire: httpx.Request) -> httpx.Response:
        sent.append(wire)
        if len(sent) == 1:
            raise httpx.ReadTimeout("scripted timeout", request=wire)
        return completion("fixed", prompt=10, output=10, model=FIXER_MODEL)

    def model(**kwargs: Any) -> MetaClient:
        client = MetaClient(
            api_key=f"stand-in-{secrets.token_hex(8)}",
            meter=kwargs.get("meter", parent),
            transport=httpx.MockTransport(reply),
            sleep=lambda _: None,
        )
        clients.append(client)
        return client

    monkeypatch.setattr(cli, "MetaClient", model)
    monkeypatch.setattr(cli, "shared_meter", lambda: parent)
    request = ModelRequest(
        stage="fix",
        model=FIXER_MODEL,
        messages=(Message(role="user", content="fix"),),
        temperature=0.2,
        max_tokens=125_000 if crosses_cap else 100,
    )
    worst = HIGHEST_PRICE.worst_case_micro_usd(len(encode_request(request)), request.max_tokens)

    class CallingFixer(Fixer):
        def fix(self, target: Target, finding: Finding, workspace: Workspace, **seams: Any) -> Any:
            response = seams["client"].complete(request)
            assert response.cost is not None
            self.cost = response.cost.micro_usd
            return super().fix(target, finding, workspace, **seams)

    attack = Reattack()
    record = execute(tmp_path, client=cli.fix_client(), fixer=CallingFixer(), reattack=attack)
    if crosses_cap:
        assert worst <= FIX_CAP_MICRO_USD < 2 * worst
        assert len(sent) == 1
        assert record.status.value == "unchecked" and record.reason == "budget"
        assert record.micro_usd == parent.spent_micro_usd == worst
        assert not attack.calls
    else:
        assert len(sent) == 2 and record.status.value == "survived_fresh_attack"
        assert record.reason is None
        assert worst < record.micro_usd == parent.spent_micro_usd < FIX_CAP_MICRO_USD
        assert attack.budgets == [min(REATTACK_CAP_MICRO_USD, FIX_CAP_MICRO_USD - record.micro_usd)]
    assert clients[0].meter.spent_micro_usd == parent.spent_micro_usd
    assert clients[0].meter.held_micro_usd == parent.held_micro_usd == 0


def test_fixer_cap_counts_settled_spend_and_every_open_hold(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_key(monkeypatch)
    parent = SpendMeter(tmp_path / "parent.jsonl")
    parent.arm(credit_confirmed=True)
    monkeypatch.setenv("RECEIPTS_SPEND_FILE", str(parent.path))
    monkeypatch.setattr(cli, "shared_meter", lambda: parent)
    client = cli.fix_client()
    assert isinstance(client, ModelRouter)
    meta = client._routes[FIXER_MODEL]
    assert isinstance(meta, MetaClient)
    meter = meta.meter
    first = meter.reserve(2_000_000, stage="fix", model=FIXER_MODEL)
    meter.settle(first, 500_000, estimated=False)
    second = meter.reserve(2_000_000, stage="fix", model=FIXER_MODEL)
    remaining = FIX_CAP_MICRO_USD - meter.spent_micro_usd - meter.held_micro_usd
    with pytest.raises(SpendCapError):
        meter.reserve(remaining + 1, stage="fix", model=FIXER_MODEL)
    boundary = meter.reserve(remaining, stage="fix", model=FIXER_MODEL)
    assert meter.spent_micro_usd + meter.held_micro_usd == FIX_CAP_MICRO_USD
    meter.settle(second, 0, estimated=False)
    meter.settle(boundary, 0, estimated=False)
    assert meter.spent_micro_usd == parent.spent_micro_usd == 500_000
    assert meter.held_micro_usd == parent.held_micro_usd == 0


@pytest.mark.parametrize("parent_only", [False, True])
def test_a_swallowed_budget_refusal_is_unchecked_in_the_ledger_and_summary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    parent_only: bool,
) -> None:
    stand_in_key(monkeypatch)
    parent = SpendMeter(tmp_path / "parent.jsonl", cap_micro_usd=100 if parent_only else 150_000_000)
    parent.arm(credit_confirmed=True)
    monkeypatch.setattr(cli, "shared_meter", lambda: parent)

    def swallow(target: Target, **kwargs: Any) -> TargetSummary:
        with suppress(SpendCapError):
            kwargs["seams"].intake_client.complete(
                ModelRequest(
                    stage="refuter",
                    model="muse-wonder-1.0",
                    temperature=0.0,
                    max_tokens=100,
                    messages=(Message(role="user", content="judge"),),
                )
            )
        return fresh_summary().model_copy(
            update={
                "costs": (),
                "arms": (
                    ArmSummary(
                        arm="spark",
                        attacker_model=SPARK.attacker_model,
                        findings=0,
                        statuses={},
                        wall_seconds=0,
                    ),
                ),
            }
        )

    monkeypatch.setattr(cli, "run_target", swallow)
    result = cli.FixReAttack(tmp_path)(FRESH, SPARK, budget_micro_usd=2_000_000 if parent_only else 100)
    assert len(result.findings) == 1
    assert result.findings[0].status is Status.UNCHECKED
    assert result.findings[0].unchecked_reason is UncheckedReason.BUDGET
    assert result.out_dir is not None
    saved = json.loads(next(Path(result.out_dir).glob("summaries/*.json")).read_text())
    assert saved["arms"][0]["findings"] == 1
    assert saved["arms"][0]["statuses"] == {"unchecked": 1}


def test_fresh_tier_cap_stops_attacks_before_the_meter_refuses(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from receipts.core.pipeline import RunPhase
    from tests.core.test_pipeline import Costs, FakeSandbox, Models, Runner

    parent = SpendMeter(tmp_path / "parent.jsonl")
    parent.arm(credit_confirmed=True)
    budgets: list[cli._FreshAttackMeter] = []
    calls: list[str] = []

    class FreePlanning(Models):
        def __init__(self, *, meter: cli._FreshAttackMeter) -> None:
            super().__init__(Costs())
            budgets.append(meter)

        def complete(self, request: ModelRequest) -> ModelResponse:
            response = super().complete(request)
            return response.model_copy(update={"cost": StageCost(stage=request.stage, micro_usd=0)})

    class ExhaustsAllowance:
        def __init__(self, *, lens: Lens = Lens.REGRESSION, **kwargs: Any) -> None:
            self.lens = lens

        def attack(self, target: Target, attack: Attack, **kwargs: Any) -> AttackOutcome:
            calls.append(attack.id)
            budget = budgets[0]
            hold = budget.reserve(budget.cap_micro_usd, stage="attack", model=SPARK.attacker_model)
            budget.settle(hold, budget.cap_micro_usd, estimated=False)
            return AttackOutcome(
                attack_id=attack.id, costs=(StageCost(stage="attack", micro_usd=budget.cap_micro_usd),)
            )

    monkeypatch.setattr(cli, "MetaClient", FreePlanning)
    monkeypatch.setattr(cli, "shared_meter", lambda: parent)
    for name in ("CorrectnessAttacker", "RegressionAttacker", "LoopAttacker"):
        monkeypatch.setattr(cli, name, ExhaustsAllowance)
    monkeypatch.setattr(
        cli, "run_phase", lambda: RunPhase(sandbox=FakeSandbox(), runners={RunnerKind.PYTEST: Runner()})
    )
    fresh = FRESH.model_copy(update={"tier": Tier.C, "body": "All tests pass."})
    result = cli.FixReAttack(tmp_path / "out")(fresh, SPARK, budget_micro_usd=500)
    assert len(calls) == 1 and not budgets[0].refused
    assert result.micro_usd == parent.spent_micro_usd == 500
    skipped = [item for item in result.findings if item.status is Status.UNCHECKED]
    assert skipped and all(item.unchecked_reason is UncheckedReason.BUDGET for item in skipped)


def test_the_fresh_attack_runs_every_attack_it_plans_past_the_caps_of_receipts_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # REG-616: a risky patch gives both Tier A claims five masks, so the plan holds 5 correctness and 5 regression
    # attacks. `receipts run` keeps 3 per lens and lists the other 4 unchecked (budget); the fresh attack runs all 10.
    from receipts.core.pipeline import RunPhase
    from tests.core.test_intake import SECURITY_PATH
    from tests.core.test_pipeline import Costs, FakeSandbox, Models, Runner

    parent = SpendMeter(tmp_path / "parent.jsonl")
    parent.arm(credit_confirmed=True)

    class FreePlanning(Models):
        def __init__(self, *, meter: cli._FreshAttackMeter) -> None:
            super().__init__(Costs())

        def complete(self, request: ModelRequest) -> ModelResponse:
            response = super().complete(request)
            return response.model_copy(update={"cost": StageCost(stage=request.stage, micro_usd=0)})

    class FindsNothing:
        def __init__(self, *, lens: Lens = Lens.REGRESSION, **kwargs: Any) -> None:
            self.lens = lens

        def attack(self, target: Target, attack: Attack, **kwargs: Any) -> AttackOutcome:
            return AttackOutcome(attack_id=attack.id)

    monkeypatch.setattr(cli, "MetaClient", FreePlanning)
    monkeypatch.setattr(cli, "shared_meter", lambda: parent)
    for name in ("CorrectnessAttacker", "RegressionAttacker", "LoopAttacker"):
        monkeypatch.setattr(cli, name, FindsNothing)
    monkeypatch.setattr(
        cli, "run_phase", lambda: RunPhase(sandbox=FakeSandbox(), runners={RunnerKind.PYTEST: Runner()})
    )
    fresh = FRESH.model_copy(update={"patch": SECURITY_PATH})
    result = cli.FixReAttack(tmp_path / "out")(fresh, SPARK, budget_micro_usd=REATTACK_CAP_MICRO_USD)
    assert result.out_dir is not None
    summary = json.loads(next(Path(result.out_dir).glob("summaries/*.json")).read_text())
    every = [*(f"correctness-{n}" for n in range(1, 6)), *(f"regression-{n}" for n in range(1, 6))]
    assert sorted(attack["id"] for attack in summary["plan"]["attacks"]) == sorted(every)
    assert summary["plan"]["unchecked"] == []
    assert [f.status for f in result.findings] == [Status.NOT_REPRODUCED] * 10
    lifted = "planned with no lens cap and no per-lens cap, not the default: 10 of 10 attacks kept"
    assert lifted in summary["planner_notes"]


def test_fresh_surrogates_stop_before_preparing_when_allowance_is_exhausted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.agentloop.test_surrogate import CLAIM, Box, Runner, target

    stand_in_key(monkeypatch)
    parent = SpendMeter(tmp_path / "parent.jsonl")
    parent.arm(credit_confirmed=True)
    budget = cli._FreshAttackMeter(parent, 100)
    hold = budget.reserve(100, stage="attack", model=SPARK.attacker_model)
    budget.settle(hold, 100, estimated=False)
    seams = cli.CoreWiring(budget=budget).seams([SPARK])
    assert seams.surrogates is not None
    box = Box()
    assert seams.surrogates.build(target(), CLAIM, sandbox=box, runner=Runner(), k=3) == ()
    assert not box.prepared and not budget.refused


def test_fresh_glimmer_reserves_zero_before_transport_and_obeys_parent_refusal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.agentloop.test_surrogate import Client

    stand_in_key(monkeypatch)
    glimmer = Client()
    monkeypatch.setattr(cli, "GlimmerClient", lambda: glimmer)
    parent = SpendMeter(tmp_path / "parent.jsonl", cap_micro_usd=1)
    parent.arm(credit_confirmed=True)
    budget = cli._FreshAttackMeter(parent, 100)
    seams = cli.CoreWiring(budget=budget).seams([GLIMMER])
    request = ModelRequest(
        stage="attack",
        model=GLIMMER.attacker_model,
        max_tokens=100,
        temperature=0.0,
        messages=(Message(role="user", content="find a counterexample"),),
    )
    seams.attacker_client.complete(request)
    assert len(glimmer.requests) == 1
    assert len(parent.path.read_text().splitlines()) == 3  # arm, zero hold, zero settlement
    assert budget.held_micro_usd == parent.held_micro_usd == 0
    spent = parent.reserve(1, stage="other", model=SPARK.attacker_model)
    parent.settle(spent, 1, estimated=False)
    with pytest.raises(SpendCapError):
        seams.attacker_client.complete(request)
    assert len(glimmer.requests) == 1 and budget.refused


@pytest.mark.parametrize("failed", [False, True])
def test_fresh_attack_keeps_model_spend_missing_from_stage_reports(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    failed: bool,
) -> None:
    from tests.core.test_meta_client import completion

    parent = SpendMeter(tmp_path / "spend.jsonl")
    parent.arm(credit_confirmed=True)

    def meta(*, meter: SpendMeter | None = None) -> MetaClient:
        return MetaClient(
            api_key=f"stand-in-{secrets.token_hex(8)}",
            meter=meter or parent,
            transport=httpx.MockTransport(lambda request: completion(prompt=0, output=50)),
        )

    def missing(target: Target, **kwargs: Any) -> TargetSummary:
        kwargs["seams"].intake_client.complete(
            ModelRequest(
                stage="attacker",
                model="muse-spark-1.3",
                max_tokens=100,
                temperature=0.0,
                messages=(Message(role="user", content="find a counterexample"),),
            )
        )
        summary = fresh_summary(failed=failed).model_copy(update={"costs": ()})
        if failed:
            kwargs["on_failure"](summary)
            raise RuntimeError("the attacker lost its cost report")
        return summary

    monkeypatch.setattr(cli, "MetaClient", meta)
    monkeypatch.setattr(cli, "shared_meter", lambda: parent)
    monkeypatch.setattr(cli, "run_target", missing)
    if failed:
        with pytest.raises(FixStageError) as raised:
            cli.FixReAttack(tmp_path)(FRESH, SPARK, budget_micro_usd=2_000_000)
        assert sum(cost.micro_usd for cost in raised.value.costs) == parent.spent_micro_usd > 0
    else:
        result = cli.FixReAttack(tmp_path)(FRESH, SPARK, budget_micro_usd=2_000_000)
        assert result.micro_usd == parent.spent_micro_usd > 0


def test_missing_glimmer_precondition_is_recorded_before_the_fixer(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from tests.agentloop.test_fixloop import Fixer, execute

    stand_in_key(monkeypatch)
    fixer = Fixer()
    record = execute(tmp_path, arm=GLIMMER, fixer=fixer, reattack=cli.FixReAttack(tmp_path / "out"))
    assert record.status.value == "unchecked" and record.reason == "runtime"
    assert "RECEIPTS_GLIMMER_URL" in record.detail
    assert not fixer.calls and record.micro_usd == 0
    cli.fix_command.print_record(record)
    assert "RECEIPTS_GLIMMER_URL" in capsys.readouterr().out


def test_two_combined_patches_keep_both_fresh_summaries(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    stand_in_key(monkeypatch)

    def fake_run(target: Target, **kwargs: Any) -> TargetSummary:
        kwargs["ledgers"]["spark"].append(
            ledger_entry(finding_for(target.id, "first", Status.NOT_REPRODUCED, 0), datetime.now(UTC))
        )
        return fresh_summary().model_copy(update={"target_id": target.id})

    monkeypatch.setattr(cli, "run_target", fake_run)
    attack = cli.FixReAttack(tmp_path / "out")
    paths = []
    for patch in ("first patch", "second patch"):
        target = FRESH.model_copy(
            update={"id": f"{FRESH.id}-{sha256(patch.encode()).hexdigest()[:12]}", "patch": patch}
        )
        result = attack(target, SPARK, budget_micro_usd=2_000_000)
        assert result.out_dir is not None
        paths.append(Path(result.out_dir))
    assert paths[0] != paths[1]
    assert all(len(list(path.glob("summaries/*.json"))) == 1 for path in paths)
    ids = [JsonlLedger(path / "spark.jsonl", read_only=True).entries()[0].finding.id for path in paths]
    assert ids[0] != ids[1]


def test_same_combined_patch_refuses_retry_without_changing_first_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    stand_in_key(monkeypatch)
    calls: list[str] = []

    def fake_run(target: Target, **kwargs: Any) -> TargetSummary:
        calls.append(target.id)
        kwargs["ledgers"]["spark"].append(
            ledger_entry(finding_for(target.id, "first", Status.NOT_REPRODUCED, 9), datetime.now(UTC))
        )
        return fresh_summary()

    monkeypatch.setattr(cli, "run_target", fake_run)
    attack = cli.FixReAttack(tmp_path / "out")
    result = attack(FRESH, SPARK, budget_micro_usd=2_000_000)
    assert result.out_dir is not None
    folder = Path(result.out_dir)
    before = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    with pytest.raises(FixStageError, match="use a fresh --out-dir") as raised:
        attack(FRESH, SPARK, budget_micro_usd=2_000_000)
    assert raised.value.costs == () and raised.value.out_dir == result.out_dir
    assert calls == [FRESH.id]
    assert before == {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}


@pytest.mark.parametrize(
    ("stage", "error_type"),
    [
        (stage, error)
        for stage in ("preflight", "pipeline")
        for error in (RuntimeError, MissingKeyError, GlimmerConfigError, SpendCapError)
    ]
    + [("summary", OSError)],
)
def test_fresh_attack_records_fixed_exception_messages(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    stage: str,
    error_type: type[Exception],
) -> None:
    from receipts.surface.text import visible
    from tests.agentloop.test_fixloop import Reattack, execute

    stand_in_key(monkeypatch)
    payload = "repository text; model output; https://endpoint.invalid/private\x1b[2J"

    def fail(*args: Any, **kwargs: Any) -> Any:
        raise error_type(payload)

    attack = cli.FixReAttack(tmp_path / "out")
    if stage == "preflight":
        monkeypatch.setattr(cli.CoreWiring, "seams", fail)
        with pytest.raises(FixStageError) as raised:
            attack.validate(SPARK)
    else:
        if stage == "pipeline":
            monkeypatch.setattr(cli, "run_target", fail)
        else:
            monkeypatch.setattr(cli, "run_target", lambda *args, **kwargs: fresh_summary())
            monkeypatch.setattr(Path, "write_text", lambda *args, **kwargs: fail())
        with pytest.raises(FixStageError) as raised:
            attack(FRESH, SPARK, budget_micro_usd=2_000_000)
    assert visible(payload) in capsys.readouterr().err
    assert payload not in str(raised.value)
    assert error_type.__name__ in str(raised.value)
    if error_type is MissingKeyError:
        assert "FALDEN_META_API_KEY" in str(raised.value)
    elif error_type is GlimmerConfigError:
        assert "RECEIPTS_GLIMMER_URL" in str(raised.value)
    elif error_type is SpendCapError:
        assert "spend meter" in str(raised.value)
    record = execute(tmp_path, reattack=Reattack(error=raised.value))
    assert "endpoint.invalid" not in record.model_dump_json()
    assert error_type.__name__ in (record.reattack_reason or "")


def test_retry_is_refused_before_fixer_spend_without_reusing_the_first_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.agentloop.test_fixloop import Client, Fixer, execute

    stand_in_key(monkeypatch)
    calls: list[str] = []

    def fake_run(target: Target, **kwargs: Any) -> TargetSummary:
        calls.append(target.id)
        kwargs["ledgers"]["spark"].append(
            ledger_entry(finding_for(target.id, "first", Status.NOT_REPRODUCED, 0), datetime.now(UTC))
        )
        return fresh_summary().model_copy(update={"target_id": target.id})

    monkeypatch.setattr(cli, "run_target", fake_run)
    first = execute(tmp_path, reattack=cli.FixReAttack(tmp_path))
    assert first.status.value == "survived_fresh_attack" and first.reattack_out_dir is not None
    folder = Path(first.reattack_out_dir)
    before = {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    fixer, client = Fixer(), Client()
    second = execute(tmp_path, fixer=fixer, client=client, reattack=cli.FixReAttack(tmp_path))
    assert second.status.value == "unchecked" and second.reason == "runtime"
    assert "use a fresh --out-dir" in second.detail
    assert not fixer.calls and not client.requests and second.micro_usd == 0
    assert second.reattack_out_dir is None and calls == [first.reattack_target_id]
    assert before == {path.relative_to(folder): path.read_bytes() for path in folder.rglob("*") if path.is_file()}
    print("probe retry: unchecked/runtime; 0 fixer calls; 0 micro-USD; no prior output directory")


@pytest.mark.parametrize("duplicate", [False, True])
@pytest.mark.parametrize("legacy_fields", [{}, {"status": "old-format", "iterations": "old-evidence"}])
def test_fix_retry_checks_only_the_finding_id_in_legacy_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, duplicate: bool, legacy_fields: dict[str, str]
) -> None:
    from tests.agentloop.test_fixloop import Client, Fixer, execute, finding

    stand_in_key(monkeypatch)

    def fake_run(target: Target, **kwargs: Any) -> TargetSummary:
        kwargs["ledgers"]["spark"].append(
            ledger_entry(finding_for(target.id, "first", Status.NOT_REPRODUCED, 0), datetime.now(UTC))
        )
        return fresh_summary().model_copy(update={"target_id": target.id})

    monkeypatch.setattr(cli, "run_target", fake_run)
    prior = legacy_fields | {"finding_id": finding().id if duplicate else "another/finding"}
    (tmp_path / "fixes.jsonl").write_text("\n" + json.dumps(prior) + "\n")
    fixer, client = Fixer(), Client()
    record = execute(tmp_path, fixer=fixer, client=client, reattack=cli.FixReAttack(tmp_path))
    if duplicate:
        assert record.status.value == "unchecked" and record.reason == "runtime"
        assert "a fix record already exists for this finding" in record.detail
        assert "use a fresh --out-dir" in record.detail
        assert not fixer.calls and not client.requests and record.micro_usd == 0
    else:
        assert record.status.value == "survived_fresh_attack" and len(fixer.calls) == 1


@pytest.mark.parametrize("line", [b'{"finding_id":', b"not json", b"null", b"[]", b"\xff"])
def test_unreadable_fix_retry_record_names_fixes_jsonl_before_spend(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, line: bytes
) -> None:
    from tests.agentloop.test_fixloop import Client, Fixer, execute

    stand_in_key(monkeypatch)
    (tmp_path / "fixes.jsonl").write_bytes(line + b"\n")
    fixer, client = Fixer(), Client()
    record = execute(tmp_path, fixer=fixer, client=client, reattack=cli.FixReAttack(tmp_path))
    assert record.status.value == "unchecked" and record.reason == "runtime"
    assert "fixes.jsonl" in record.detail
    assert not fixer.calls and not client.requests and record.micro_usd == 0


def test_two_findings_with_identical_fixed_patches_keep_both_fresh_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from tests.agentloop.test_fixloop import Box, Client, Fixer, Runner, finding, ledger, run_fix, target

    stand_in_key(monkeypatch)

    def fake_run(target: Target, **kwargs: Any) -> TargetSummary:
        kwargs["ledgers"]["spark"].append(
            ledger_entry(finding_for(target.id, "first", Status.NOT_REPRODUCED, 0), datetime.now(UTC))
        )
        return fresh_summary().model_copy(update={"target_id": target.id})

    monkeypatch.setattr(cli, "run_target", fake_run)
    findings = [finding(), finding("held")]
    records = []
    attack = cli.FixReAttack(tmp_path)
    for shown in findings:
        fixer = Fixer()
        record = run_fix(
            shown.id,
            ledger=ledger(*findings),
            target=target(),
            arm=SPARK,
            sandbox=Box(),
            runner=Runner(),
            client=Client(),
            reattack=attack,
            records_path=tmp_path / "fixes.jsonl",
            fixer=fixer,
        )
        assert record.status.value == "survived_fresh_attack" and len(fixer.calls) == 1
        records.append(record)
    assert records[0].iterations[-1].diff_sha256 == records[1].iterations[-1].diff_sha256
    assert records[0].reattack_out_dir != records[1].reattack_out_dir
    for record in records:
        assert record.reattack_out_dir is not None
        summaries = list(Path(record.reattack_out_dir).glob("summaries/*.json"))
        assert len(summaries) == 1 and json.loads(summaries[0].read_text())["target_id"] == record.reattack_target_id
