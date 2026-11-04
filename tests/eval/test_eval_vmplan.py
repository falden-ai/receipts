"""The VM choice meets the spec row (m7i.8xlarge, and m7i.16xlarge for the final, REG-563), R24 and the label runs
are planned on m7i.16xlarge with the pilot back on m7i.8xlarge (REG-568), worker counts come from the machine's vCPUs,
and the quote arithmetic is exact."""

from __future__ import annotations

import argparse
import shlex
from pathlib import Path

import pytest

import receipts.cli as receipts_cli
from receipts.eval import cli as eval_cli
from receipts.eval import vmplan
from receipts.eval.r24 import MAX_DROP_PCT, R24Report
from receipts.eval.vmplan import (
    CASES,
    CHEAPER,
    CHOSEN,
    GOLD_DROPS,
    GPU_INSTANCE,
    HOURS_PER_MONTH,
    LABEL_BLOCK_FIXED,
    LABEL_DAY_RUNS,
    LABEL_INSTANCE_TYPE,
    LABEL_RUN_MINUTES,
    OFFERS,
    RUNBOOK,
    WORKLOADS,
    Disk,
    Offer,
    SpecError,
    Step,
    Workload,
    check_offer,
    default_workers,
    harness_minutes,
    max_drops,
    offer,
    quote,
    run_minutes,
    slots_for,
    usd,
    usd_rate,
    workers_for,
)

ROOT = Path(__file__).resolve().parents[2]


def test_the_chosen_and_the_cheaper_instance_meet_the_spec() -> None:
    for name in (CHOSEN, CHEAPER):
        o = check_offer(offer(name))
        assert (o.vcpu, o.memory_gib, o.arch) == (32, 128, "x86_64")
    assert offer(CHEAPER).micro_usd_per_hour < offer(CHOSEN).micro_usd_per_hour


@pytest.mark.parametrize(
    ("change", "match"),
    [
        ({"arch": "arm64"}, "x86_64"),
        ({"vcpu": 16}, "need 32"),
        ({"memory_gib": 64}, "need 32 and 128"),
        ({"burstable": True}, "burstable"),
    ],
)
def test_plant_an_instance_off_the_spec_row_is_refused(change: dict[str, object], match: str) -> None:
    with pytest.raises(SpecError, match=match):
        check_offer(offer(CHOSEN).model_copy(update=change))


def test_the_flex_type_in_the_price_list_is_refused() -> None:
    flex = next(o for o in OFFERS if o.instance_type.startswith("m7i-flex"))
    with pytest.raises(SpecError, match="burstable"):
        check_offer(flex)


def test_disk_cost_is_exact() -> None:
    month = Disk(exists_hours=HOURS_PER_MONTH)
    assert month.micro_usd() == 60_000_000  # $40 storage + $15 IOPS + $5 throughput
    assert Disk(iops=3000, throughput_mibps=125, exists_hours=HOURS_PER_MONTH).micro_usd() == 40_000_000


def test_run_minutes_spread_targets_over_the_busy_sandboxes() -> None:
    w = Workload(name="w", date="2026-10-28", targets=12, fixed_minutes=10, fixed_basis="x")
    expected = next(c for c in CASES if c.name == "expected")
    assert run_minutes(w, expected) == 10 + 50  # 12 targets x 50 min over 12 busy sandboxes
    assert run_minutes(w, expected, slots=32) == 10 + 25
    big = w.model_copy(update={"instance_type": "m7i.16xlarge"})
    assert run_minutes(big, expected) == 10 + 25  # 64 vCPU: 32 sandboxes


def test_the_quote_adds_up_and_orders_its_cases() -> None:
    q = quote()
    assert q.instance_type == CHOSEN and q.region == "us-east-1" and q.final_instance_type == "m7i.16xlarge"
    totals = [c.total_micro_usd for c in q.cases]
    assert totals == sorted(totals)
    for c in q.cases:
        assert c.total_micro_usd == sum(line.micro_usd for line in c.lines)
        assert set(c.run_minutes) == {w.name for w in WORKLOADS}
        final = next(line for line in c.lines if "final run" in line.item)
        assert final.item.startswith("m7i.16xlarge")  # REG-563
        assert sum(1 for line in c.lines if line.item.startswith(GPU_INSTANCE)) == 2
    assert int(q.final_on_32_vcpu["final_run_minutes_64_vcpu"]) < int(q.final_on_32_vcpu["final_run_minutes_32_vcpu"])
    assert q.label_runs["label_runner_minutes"] == str(LABEL_RUN_MINUTES)


def test_unit_prices_keep_four_decimals() -> None:
    assert usd_rate(1_612_800) == "$1.6128"
    assert usd_rate(3_225_600) == "$3.2256"
    assert usd_rate(5_000) == "$0.0050"


def test_usd_rounds_up_to_the_cent() -> None:
    assert usd(1) == "$0.01"
    assert usd(1_612_800) == "$1.62"
    assert usd(90_930_000) == "$90.93"
    assert usd(1_234_567_890_000) == "$1,234,567.89"


def test_prices_are_integers_and_cited() -> None:
    q = quote()
    assert q.sources["ec2_data"].startswith("https://b0.p.awsstatic.com/pricing/")
    assert all(isinstance(o.micro_usd_per_hour, int) for o in OFFERS)
    assert Offer.model_fields["micro_usd_per_hour"].annotation is int


def test_the_final_run_is_the_full_sample_with_the_plausible_wrong_stratum() -> None:
    final = next(w for w in WORKLOADS if w.name == "final run")
    assert final.targets == 150 + 150 + 75 == 375 and final.instance_type == "m7i.16xlarge"
    assert next(w for w in WORKLOADS if w.name == "pilot").targets == 20 * 2  # both arms (REG-561)


def test_the_label_blocks_fixed_minutes_are_their_parts() -> None:
    slots = slots_for(LABEL_INSTANCE_TYPE)
    assert slots == 32  # 64 vCPU at 2 per sandbox
    assert 30 + sum(harness_minutes(n, slots=slots) for n in LABEL_DAY_RUNS.values()) + 60 == LABEL_BLOCK_FIXED
    label_runs = sum(harness_minutes(n, slots=slots) for k, n in LABEL_DAY_RUNS.items() if k.startswith("labels"))
    assert label_runs == LABEL_RUN_MINUTES
    assert harness_minutes(12, slots=16) == 4  # 12 runs x 4 sandbox-minutes over 12 busy sandboxes
    assert harness_minutes(1, slots=16) == 1  # rounded up, never down


# ---------------------------------------------------------------- REG-568


def test_reg_568_r24_and_the_labels_run_on_the_16xlarge_and_the_pilot_on_the_8xlarge() -> None:
    by = {w.name: w for w in WORKLOADS}
    assert (by["R24 and labels"].instance_type, by["R24 and labels"].targets) == ("m7i.16xlarge", 0)
    assert by["pilot"].instance_type == "m7i.8xlarge" and by["pilot"].date == by["R24 and labels"].date
    assert by["dress rehearsal"].instance_type == "m7i.8xlarge"
    assert (offer("m7i.16xlarge").micro_usd_per_hour, offer("m7i.8xlarge").micro_usd_per_hour) == (3_225_600, 1_612_800)
    q = quote()
    assert (q.label_instance_type, q.label_usd_per_hour, q.usd_per_hour) == ("m7i.16xlarge", "$3.2256", "$1.6128")
    for c in q.cases:
        day = c.pilot_day
        blocks = {b.what: b for b in day.blocks}
        r24, pilot = blocks["R24 and labels"], blocks["pilot"]
        assert (r24.instance_type, pilot.instance_type) == ("m7i.16xlarge", "m7i.8xlarge")
        assert r24.minutes == LABEL_BLOCK_FIXED == c.run_minutes["R24 and labels"]
        assert r24.micro_usd == -(-3_225_600 * r24.minutes // 60)  # $3.2256 an hour
        assert pilot.micro_usd == -(-1_612_800 * pilot.minutes // 60)  # $1.6128 an hour
        assert day.eval_vm_minutes == r24.minutes + pilot.minutes
        assert day.compute_micro_usd == sum(b.micro_usd for b in day.blocks)
        assert any(b.instance_type == GPU_INSTANCE for b in day.blocks)
    expected = next(c for c in q.cases if c.case == "expected")
    # 376 min of R24 and labels plus the pilot's 278 (round 6: --jobs 16, two waves of both arms); compute is both
    # eval VM blocks and the GPU VM's 328 min
    assert expected.pilot_day.eval_vm_hours == "10.9" and expected.pilot_day.compute_usd == "$46.08"
    assert q.label_runs["block_usd"] == usd(-(-3_225_600 * LABEL_BLOCK_FIXED // 60))


def test_reg_568_counts_diagnoses_three_harness_runs_for_every_drop_r24_allows() -> None:
    """--diagnose runs the harness gold check (a primary and two diagnostic runs) on each drop; the plan counts it at
    its most, every drop R24 reports before it stops at its limit."""
    assert LABEL_DAY_RUNS["r24_diagnose"] == 3 * max_drops(251) == 3 * 62
    at_limit = R24Report(run_id="t", instances=251, verdicts={}, dropped={str(n): "x" for n in range(62)},
                         unchecked=(), max_drop_pct=MAX_DROP_PCT)  # fmt: skip
    over = at_limit.model_copy(update={"dropped": {str(n): "x" for n in range(63)}})
    assert not at_limit.drop_pct_over_limit and over.drop_pct_over_limit  # 62 is the most R24 reports
    assert LABEL_DAY_RUNS["r24"] == LABEL_DAY_RUNS["labels_gold"] == 251


@pytest.mark.parametrize(
    ("vcpus", "memory_gib", "workers"),
    [(64, 256, 32), (32, 128, 16), (64, 64, 16), (3, None, 1), (1, None, 1), (14, 36, 7)],
)
def test_worker_counts_come_from_the_vcpus_within_the_memory(vcpus: int, memory_gib: int | None, workers: int) -> None:
    memory = None if memory_gib is None else memory_gib * 1024**3
    assert workers_for(vcpus, memory) == workers
    assert slots_for("m7i.16xlarge") == workers_for(64, 256 * 1024**3) == 32


def test_the_default_worker_count_is_this_machines(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(vmplan, "machine_vcpus", lambda: 64)
    monkeypatch.setattr(vmplan, "machine_memory_bytes", lambda: 256 * 1024**3)
    assert default_workers() == 32
    monkeypatch.setattr(vmplan, "machine_vcpus", lambda: 32)
    monkeypatch.setattr(vmplan, "machine_memory_bytes", lambda: 128 * 1024**3)
    assert default_workers() == 16


# ---------------------------------------------------------------- round 6: the pilot as `receipts run --jobs N`


def test_the_pilot_is_receipts_run_with_one_job_per_sandbox_in_waves() -> None:
    """Item 4: the core lane gives each job a sandbox of its own, so the pilot is `receipts run --jobs 16`, the
    m7i.8xlarge's sandbox count. Its 20 targets go in 2 waves (16, then 4), each job running spark and then glimmer,
    and a target run of m sandbox-minutes takes m * 100 / 75 minutes on its job's sandbox (busy 75% of the time). The
    GPU VM is up from its 60-minute boot through the whole run. Round 5's quote spread 40 target runs over 16 sandboxes
    (177 min expected, 144 on the GPU VM), which `receipts run` cannot do."""
    q = quote()
    p = q.pilot
    assert (p.instance_type, p.jobs, p.targets, p.arms, p.waves) == ("m7i.8xlarge", 16, 20, ("spark", "glimmer"), 2)
    assert p.jobs == slots_for(CHOSEN) == vmplan.PILOT_JOBS
    assert p.command == vmplan.pilot_command(16) and p.command.endswith(" --jobs 16")
    got = {
        name: (c.target_run_minutes, c.run_minutes, c.eval_vm_minutes, c.eval_vm_usd, c.gpu_vm_minutes, c.gpu_vm_usd)
        for name, c in p.cases.items()
    }
    assert got == {
        "low": (40, 160, 170, "$4.57", 220, "$12.34"),  # 30 sandbox-minutes: 2 waves x 2 arms x 40, +10 restart
        "expected": (67, 268, 278, "$7.48", 328, "$18.39"),  # 50: 2 x 2 x 67; GPU 60 + 268
        "high": (120, 480, 490, "$13.18", 540, "$30.27"),  # 90: 2 x 2 x 120
    }
    assert usd(-(-1_612_800 * 278 // 60)) == "$7.48" and usd(-(-3_363_120 * 328 // 60)) == "$18.39"
    for c in q.cases:
        blocks = {b.what: b for b in c.pilot_day.blocks}
        assert blocks["pilot"].minutes == p.cases[c.case].eval_vm_minutes == c.run_minutes["pilot"]
        assert blocks["the Glimmer arm's GPU VM"].minutes == p.cases[c.case].gpu_vm_minutes
        gpu_line = next(line for line in c.lines if line.item.endswith("the Glimmer arm on pilot day"))
        assert gpu_line.quantity == f"{p.cases[c.case].gpu_vm_minutes} min"


@pytest.mark.parametrize(("jobs", "waves"), [(16, 2), (19, 2), (20, 1), (32, 1), (1, 20)])
def test_plant_per_job_targets_run_in_waves_of_the_jobs_never_spread_over_the_sandboxes(jobs: int, waves: int) -> None:
    """A per-job workload is its targets in rounds of `jobs`, each job its target's arms in turn; a quote that spread
    the pilot's 40 target runs over the sandboxes again (round 5's model) would read 10 + 40 x 50 / 12 = 177 min."""
    pilot = next(w for w in WORKLOADS if w.name == "pilot")
    expected = next(c for c in CASES if c.name == "expected")
    assert pilot.per_job and (pilot.targets, pilot.arms) == (40, 2)
    assert vmplan.waves(pilot, jobs) == waves
    assert run_minutes(pilot, expected, slots=jobs) == 10 + waves * 2 * 67
    spread = pilot.model_copy(update={"per_job": False})
    assert run_minutes(spread, expected) == 177 != run_minutes(pilot, expected) == 278


def test_a_bigger_pilot_vm_runs_every_target_in_one_wave() -> None:
    big = quote("m7i.16xlarge").pilot
    assert (big.jobs, big.waves) == (32, 1) and big.cases["expected"].eval_vm_minutes == 10 + 2 * 67
    assert big.command.endswith(" --jobs 32")


def test_plant_a_per_job_workload_counts_whole_targets() -> None:
    with pytest.raises(ValueError, match="not whole targets of 2 arms"):
        Workload(name="w", date="2026-10-28", targets=41, fixed_minutes=0, fixed_basis="x", arms=2, per_job=True)


# ---------------------------------------------------------------- round 6: the runbook, as the commands that run

EXPECTED_RUNBOOK = [
    ("mac", "ssh -F runs/vm/ssh_config receipts-eval cat /opt/receipts/BOOT_OK"),
    (
        "mac",
        "rsync -a --delete --exclude=.git/ --exclude=.venv/ --exclude=runs/ --exclude=.receipts/ "
        "--exclude=__pycache__/ -e 'ssh -F runs/vm/ssh_config' ./ receipts-eval:/home/ubuntu/falden-receipts/",
    ),
    ("vm", "uv sync --frozen --extra eval"),
    (
        "vm",
        "uv run --frozen --offline --extra eval receipts eval r24 --instances eval/pools/swe_abs_251.txt "
        "--out runs/r24 --run-id d1 --diagnose",
    ),
    ("vm", "uv run --frozen --offline --extra eval python -m receipts.eval labels --out runs/labels --run-id d1"),
    ("mac", "rsync -a -e 'ssh -F runs/vm/ssh_config' receipts-eval:/home/ubuntu/falden-receipts/runs/ runs/"),
    (
        "mac",
        "uv run --frozen python -m receipts.eval manifest --gold-runsc runs/r24 --gold-drop eval/gold-drops.tsv "
        "--labels runs/labels --freeze",
    ),
    ("mac", "uv run --frozen python -m receipts.eval sample --plan spec"),
    ("mac", "uv run --frozen python -m receipts.eval pilot"),
    (
        "mac",
        "rsync -a --delete --exclude=.git/ --exclude=.venv/ --exclude=runs/ --exclude=.receipts/ "
        "--exclude=__pycache__/ -e 'ssh -F runs/vm/ssh_config' ./ receipts-eval:/home/ubuntu/falden-receipts/",
    ),
    ("vm", "uv sync --frozen"),
    ("vm", "uv run --frozen --offline receipts spend arm --credit-confirmed"),
    (
        "vm",
        "uv run --frozen --offline receipts run --targets eval/pilot-run.jsonl --arms eval/pilot-arms.json "
        "--ledger-dir runs/pilot --jobs 16",
    ),
    ("vm", "uv run --frozen --offline receipts eval score --ledgers runs/pilot --out runs/score"),
    ("mac", "rsync -a -e 'ssh -F runs/vm/ssh_config' receipts-eval:/home/ubuntu/falden-receipts/runs/ runs/"),
]


def test_the_runbook_pins_every_step_as_the_command_that_runs() -> None:
    """Item 2 (round 5's verify:main): the runbook's R24 line could not start, its freeze passed no --gold-drop, and
    its other steps were prose. Every step is now where it runs and the exact command, and quote.json carries them."""
    assert [(s.where, s.command) for s in RUNBOOK] == EXPECTED_RUNBOOK
    assert quote().runbook == RUNBOOK


def uv_run(command: str) -> tuple[list[str], list[str]]:
    """A `uv run` line split into uv's own flags and the command uv runs."""
    words = shlex.split(command)
    assert words[:2] == ["uv", "run"], command
    flags, rest = [], words[2:]
    while rest[0].startswith("--"):
        take = 2 if rest[0] == "--extra" else 1
        flags += rest[:take]
        rest = rest[take:]
    return flags, rest


def test_every_runbook_command_parses_with_the_cli_it_runs_and_chains_its_outputs() -> None:
    """Each `uv run` line parses with the real parser of the command it runs (receipts.cli or receipts.eval's), so a
    renamed flag fails here; each runs --frozen, and on the VM --offline too, so nothing is resolved or fetched while
    a run runs. Only the two steps that import swebench take the eval extra, and the steps that hold the Meta key
    (spend arm, run, score) run after `uv sync --frozen` removed it. Each step reads what the one before it wrote."""
    top = receipts_cli.build_parser()
    lane = eval_cli.build_parser()
    parsed: dict[str, tuple[Step, list[str], argparse.Namespace]] = {}
    for s in RUNBOOK:
        if not s.command.startswith("uv run "):
            continue
        flags, rest = uv_run(s.command)
        assert "--frozen" in flags and ("--offline" in flags) is (s.where == "vm"), s.command
        if rest[0] == "receipts":
            args = top.parse_args(rest[1:])
            name = args.command if args.command != "eval" else args.eval_command
        else:
            assert rest[:3] == ["python", "-m", "receipts.eval"], s.command
            args = lane.parse_args(rest[3:])
            name = args.eval_command
        parsed[name] = (s, flags, args)
    assert sorted(parsed) == ["labels", "manifest", "pilot", "r24", "run", "sample", "score", "spend"]
    with_extra = sorted(n for n, (_, flags, _) in parsed.items() if "--extra" in flags)
    assert with_extra == ["labels", "r24"] and all(parsed[n][1][-2:] == ["--extra", "eval"] for n in with_extra)
    r24, labels, freeze = parsed["r24"][2], parsed["labels"][2], parsed["manifest"][2]
    assert r24.diagnose and (ROOT / r24.instances).is_file() and r24.workers is None  # workers from the VM's vCPUs
    assert (freeze.gold_runsc, freeze.labels, freeze.gold_drop, freeze.freeze) == (
        r24.out,
        labels.out,
        GOLD_DROPS,
        True,
    )
    assert (ROOT / freeze.gold_drop).is_file() and not freeze.label_drop and not freeze.gold_rerun
    assert parsed["sample"][2].plan == "spec"
    spend, run, score = parsed["spend"][2], parsed["run"][2], parsed["score"][2]
    assert (spend.spend_command, spend.credit_confirmed) == ("arm", True)
    assert run.jobs == slots_for(CHOSEN) == 16 and not run.resume
    assert (ROOT / run.targets).is_file() and (ROOT / run.arms).is_file()
    assert (score.ledgers, score.provisional, score.replay_drop) == (str(run.ledger_dir), False, None)
    commands = [s.command for s in RUNBOOK]
    assert commands.index("uv sync --frozen --extra eval") < commands.index(parsed["r24"][0].command)
    assert (
        commands.index(parsed["pilot"][0].command)
        < commands.index("uv sync --frozen")
        < commands.index(parsed["spend"][0].command)
    )
    assert not any("FALDEN_META_API_KEY=" in c or "RECEIPTS_GLIMMER_URL=" in c for c in commands)


def test_the_runbook_moves_the_tree_and_the_results_but_never_a_spend_file_or_a_venv() -> None:
    push = [s for s in RUNBOOK if s.command.startswith("rsync -a --delete")]
    pull = [s for s in RUNBOOK if s.command.startswith("rsync -a -e")]
    assert len(push) == 2 and len(pull) == 2 and all(s.where == "mac" for s in (*push, *pull))
    for s in push:
        words = shlex.split(s.command)
        assert {"--exclude=runs/", "--exclude=.receipts/", "--exclude=.venv/", "--exclude=.git/"} <= set(words)
        assert words[-2:] == ["./", "receipts-eval:/home/ubuntu/falden-receipts/"]
        assert words[words.index("-e") + 1] == "ssh -F runs/vm/ssh_config"
    for s in pull:
        assert shlex.split(s.command)[-2:] == ["receipts-eval:/home/ubuntu/falden-receipts/runs/", "runs/"]


def test_the_runbook_says_when_the_founders_drops_apply() -> None:
    """The founder's three drops, each by name with a reason and never automatic: --gold-drop at the freeze (the two
    REG-571 lines are pre-approved), --label-drop there for an instance R24 kept whose label jobs stayed INFRA, and
    --replay-drop at score for a gold replay still infra after its retry."""
    freeze = next(s for s in RUNBOOK if " manifest " in s.command)
    score = next(s for s in RUNBOOK if " eval score " in s.command)
    r24 = next(s for s in RUNBOOK if " eval r24 " in s.command)
    assert "REG-571" in freeze.note and "sphinx-doc__sphinx-8265 and 8621" in freeze.note
    assert "--label-drop <instance>=<reason>" in freeze.note and "never pre-approved" in freeze.note
    assert "--gold-rerun" in freeze.note and "--gold-rerun runs/r24-rerun1" in r24.note
    assert "--replay-drop <finding id>=<reason>" in score.note and "only by the founder" in score.note
