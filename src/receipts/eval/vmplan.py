"""The eval VM (spec parameters: m7i.8xlarge on demand in us-east-1 for the pilot and the rehearsal, m7i.16xlarge for
the 11/1 final, REG-556 and REG-563; x86, 500 GB, no instance role, 2 vCPU per sandbox, stopped between runs), the
pilot's GPU VM for the Glimmer arm (g7e.2xlarge, `receipts.eval.gpuvm`), and their quote.

REG-568: R24 and the SWE-ABS label runs run on an m7i.16xlarge, the size the VM launches at; then `launch.sh resize`
returns the stopped VM to the m7i.8xlarge for the pilot (and the rehearsal); the 11/1 final is on the m7i.16xlarge
again. Worker counts are never a constant: a run uses one sandbox per 2 vCPU of the machine it runs on (the run
phase's --cpus), and no more than its memory holds at the run phase's --memory each (`default_workers`).

Nothing here calls AWS. Prices are the public on-demand list prices for US East (N. Virginia), read on 2026-10-27
from the data files that the public pricing pages load, each cited below with its publication stamp. Money is integer
micro-dollars and time is integer minutes, so the arithmetic is exact; dollars appear only when printed.

The hour estimate is a model, not a measurement: the 10/28 pilot exists to measure compute per target (spec run
gates). `quote` therefore prices three cases of sandbox-minutes per target, and the pilot's number replaces them.

The pilot runs as `receipts run --jobs N` at the VM's sandbox count (16 on the m7i.8xlarge): N targets at once, each a
job on a sandbox of its own that runs the target's two arms in turn (the core lane's per-job run phase), so its wall
time is waves of jobs, not sandbox-minutes spread over every sandbox (`run_minutes`, `PilotPlan`). The GPU VM is up
from its boot through the whole pilot, since every job calls Glimmer once its Spark arm is done.

`RUNBOOK` is every step on and around the VM as the exact command that runs, from main: where it runs, the command,
and what its exit codes and the founder's drops mean. tests/eval/test_eval_vmplan.py pins each command and parses it
with the CLI it calls.
"""

from __future__ import annotations

import math
import os
from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from receipts.contracts import RUN_PHASE_LIMITS, Frozen, docker_bytes

MICRO = 1_000_000
HOURS_PER_MONTH = 730  # AWS's own convention for monthly prices

EC2_PRICE_PAGE = "https://aws.amazon.com/ec2/pricing/on-demand/"
EC2_PRICE_DATA = (
    "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ec2-ondemand-without-sec-sel/"
    "US%20East%20%28N.%20Virginia%29/Linux/index.json"
)
EC2_PRICE_PUBLISHED = "2026-10-23T22:13:55Z"
EBS_PRICE_PAGE = "https://aws.amazon.com/ebs/pricing/"
EBS_PRICE_DATA = "https://b0.p.awsstatic.com/pricing/2.0/meteredUnitMaps/ec2/USD/current/ebs.json"
EBS_PRICE_PUBLISHED = "2026-10-23T22:13:55Z"
IPV4_PRICE_PAGE = "https://aws.amazon.com/vpc/pricing/"  # "Public IPv4 Address" tab


class Offer(Frozen):
    """One on-demand Linux price, as the public price data lists it."""

    instance_type: str
    vcpu: int
    memory_gib: int
    arch: Literal["x86_64", "arm64"]
    cpu: str
    micro_usd_per_hour: int
    rate_code: str
    burstable: bool = False
    local_nvme: bool = False


# The 32 vCPU, 128 GiB x86 general-purpose types this plan compared (not every one in the price data: m8i-flex, m5ad,
# m5d, m5n, m8a, m8id, m5dn, m6in and m6idn are left out), and the 64 vCPU size the final may need. The cheapest x86
# option here is m5a.8xlarge (older AMD cores); m6a.8xlarge is the cheapest current generation.
OFFERS: tuple[Offer, ...] = (
    Offer(
        instance_type="m7i.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="Intel Xeon (Sapphire Rapids)",
        micro_usd_per_hour=1_612_800,
        rate_code="DFY9W427PBEXH8VY.JRTCKXETXF.6YS6EN2CT7",
    ),
    Offer(
        instance_type="m6a.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="AMD EPYC (Milan)",
        micro_usd_per_hour=1_382_400,
        rate_code="5B2BCNUG3USGJJUW.JRTCKXETXF.6YS6EN2CT7",
    ),
    Offer(
        instance_type="m6i.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="Intel Xeon (Ice Lake)",
        micro_usd_per_hour=1_536_000,
        rate_code="8NPUTYG479RPDVPK.JRTCKXETXF.6YS6EN2CT7",
    ),
    Offer(
        instance_type="m8i.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="Intel Xeon (Granite Rapids)",
        micro_usd_per_hour=1_693_440,
        rate_code="C8MQZND7UXJR47MY.JRTCKXETXF.6YS6EN2CT7",
    ),
    Offer(
        instance_type="m7a.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="AMD EPYC (Genoa), one core per vCPU",
        micro_usd_per_hour=1_854_720,
        rate_code="25T5FUC5Y3JNGAFA.JRTCKXETXF.6YS6EN2CT7",
    ),
    Offer(
        instance_type="m5.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="Intel Xeon (Skylake or Cascade Lake)",
        micro_usd_per_hour=1_536_000,
        rate_code="RFGVEPVZTEFFRRQ3.JRTCKXETXF.6YS6EN2CT7",
    ),
    Offer(
        instance_type="m5a.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="AMD EPYC (Naples)",
        micro_usd_per_hour=1_376_000,
        rate_code="PSXMMQWTBA57FKDB.JRTCKXETXF.6YS6EN2CT7",
    ),
    Offer(
        instance_type="m7i-flex.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="Intel Xeon (Sapphire Rapids), flex",
        micro_usd_per_hour=1_532_160,
        rate_code="M3Q29Z86PGCBDF39.JRTCKXETXF.6YS6EN2CT7",
        burstable=True,
    ),
    Offer(
        instance_type="m6id.8xlarge",
        vcpu=32,
        memory_gib=128,
        arch="x86_64",
        cpu="Intel Xeon (Ice Lake), 1900 GB NVMe",
        micro_usd_per_hour=1_898_400,
        rate_code="WZUXR7XAU8PP9FTW.JRTCKXETXF.6YS6EN2CT7",
        local_nvme=True,
    ),
    Offer(
        instance_type="m7i.16xlarge",
        vcpu=64,
        memory_gib=256,
        arch="x86_64",
        cpu="Intel Xeon (Sapphire Rapids)",
        micro_usd_per_hour=3_225_600,
        rate_code="KZ253M3KCY8C26FG.JRTCKXETXF.6YS6EN2CT7",
    ),
)

CHOSEN = "m7i.8xlarge"  # the pilot and the rehearsal (REG-556)
LABEL_INSTANCE_TYPE = "m7i.16xlarge"  # R24 and the SWE-ABS label runs (REG-568): the size the VM launches at
FINAL_INSTANCE_TYPE = "m7i.16xlarge"  # the 11/1 final (REG-563)
CHEAPER = "m6a.8xlarge"
REGION = "us-east-1"
WHY = (
    "m7i.8xlarge is the newest mainstream Intel type at the spec's exact shape (32 vCPU, 128 GiB) and the account's "
    "home region is us-east-1; m6a.8xlarge is 14% cheaper per hour on older AMD cores, and m7i-flex is ruled out "
    "because flex instances do not sustain full CPU, which a 16-sandbox test load needs"
)


def offer(instance_type: str) -> Offer:
    for o in OFFERS:
        if o.instance_type == instance_type:
            return o
    raise KeyError(instance_type)


class SpecError(ValueError):
    """An instance that does not meet the spec's VM row."""


def check_offer(o: Offer, *, vcpu: int = 32, memory_gib: int = 128) -> Offer:
    if o.arch != "x86_64":
        raise SpecError(f"{o.instance_type}: the Epoch images are x86_64")
    if o.vcpu < vcpu or o.memory_gib < memory_gib:
        raise SpecError(f"{o.instance_type}: {o.vcpu} vCPU and {o.memory_gib} GiB, need {vcpu} and {memory_gib}")
    if o.burstable:
        raise SpecError(f"{o.instance_type}: burstable CPU cannot hold a sustained 16-sandbox load")
    return o


class Disk(Frozen):
    """gp3 root volume. Billed every hour it exists, running or stopped."""

    gib: int = 500
    iops: int = 6000  # 3,000 included
    throughput_mibps: int = 250  # 125 included
    micro_usd_per_gib_month: int = 80_000  # $0.08
    micro_usd_per_iops_month: int = 5_000  # $0.005 above 3,000
    micro_usd_per_gibps_month: int = 40_960_000  # $40.96 per GiB/s above 125 MiB/s, which is $0.04 per MiB/s
    # Created 10/28 08:00 ET, deleted 11/2 18:00 ET: room for a high-case final on m7i.16xlarge from 11/1 08:00 (about
    # 33 h; the expected case is about 22 h).
    exists_hours: int = 130

    def micro_usd(self) -> int:
        monthly = (
            self.gib * self.micro_usd_per_gib_month
            + max(0, self.iops - 3000) * self.micro_usd_per_iops_month
            + max(0, self.throughput_mibps - 125) * self.micro_usd_per_gibps_month // 1024
        )
        return -(-monthly * self.exists_hours // HOURS_PER_MONTH)


IPV4_MICRO_USD_PER_HOUR = 5_000  # $0.005, in use or idle; an auto-assigned address goes away when the VM stops


class Workload(Frozen):
    """One billed run of the VM: fixed work plus targets times sandbox-minutes, spread over the sandboxes; or, with
    `per_job`, `receipts run --jobs <the VM's sandbox count>`: one target per job on a sandbox of its own, its `arms`
    run in turn, the targets in waves of that many jobs."""

    name: str
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    instance_type: str = "m7i.8xlarge"
    targets: int = Field(ge=0)  # target runs: a target attacked by two arms counts twice
    fixed_minutes: int = Field(ge=0)
    fixed_basis: str
    baseline_sandbox_minutes: int = Field(default=0, ge=0)  # B0, B1 and B2 per target, final run only
    arms: int = Field(default=1, ge=1)  # the arms each target runs in turn
    per_job: bool = False

    @model_validator(mode="after")
    def _whole_targets(self) -> Self:
        if self.targets % self.arms:
            raise ValueError(f"{self.name}: {self.targets} target runs are not whole targets of {self.arms} arms")
        return self


class Case(Frozen):
    name: Literal["low", "expected", "high"]
    sandbox_minutes_per_target: int = Field(gt=0)


CASES: tuple[Case, ...] = (
    Case(name="low", sandbox_minutes_per_target=30),
    Case(name="expected", sandbox_minutes_per_target=50),
    Case(name="high", sandbox_minutes_per_target=90),
)

SANDBOX_VCPU = math.ceil(RUN_PHASE_LIMITS.cpus)  # the run phase's --cpus: 16 sandboxes on 32 vCPU, 32 on 64
SANDBOX_MEMORY_BYTES = docker_bytes(RUN_PHASE_LIMITS.memory)  # the run phase's --memory, also --memory-swap
PACKING_PCT = 75  # sandboxes are not all busy all the time: 3 in 4 busy on average
HARNESS_RUN_SANDBOX_MINUTES = 4  # one SWE-bench eval script (gold or patch) in one sandbox, a model like the rest


def workers_for(vcpus: int, memory_bytes: int | None = None) -> int:
    """How many sandboxes a machine runs at once: one per SANDBOX_VCPU vCPUs, and no more than its memory holds at the
    run phase's memory cap each. Never fewer than one."""
    n = vcpus // SANDBOX_VCPU
    if memory_bytes is not None:
        n = min(n, memory_bytes // SANDBOX_MEMORY_BYTES)
    return max(1, n)


def machine_vcpus() -> int:
    """The vCPUs this process may use: its CPU affinity where the platform has one (Linux), else the CPU count."""
    affinity = getattr(os, "sched_getaffinity", None)
    if affinity is not None:
        return len(affinity(0))
    return os.cpu_count() or 1


def machine_memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        return None


def default_workers() -> int:
    """The worker count for a run on this machine (REG-568: from the VM's vCPU count, never a constant)."""
    return workers_for(machine_vcpus(), machine_memory_bytes())


def slots_for(instance_type: str) -> int:
    o = offer(instance_type)
    return workers_for(o.vcpu, o.memory_gib * 1024**3)


def harness_minutes(runs: int, *, slots: int, packing_pct: int = PACKING_PCT) -> int:
    return -(-runs * HARNESS_RUN_SANDBOX_MINUTES * 100 // (slots * packing_pct))


# R24 drops at most this share before it stops (exit 3), and --diagnose runs the harness gold check on every drop: the
# primary run plus two diagnostic runs when the primary fails, so up to 3 harness runs per drop.
POOL_SIZE = 251
R24_MAX_DROP_PCT = 25
DIAGNOSE_RUNS_PER_DROP = 3


def max_drops(instances: int, pct: int = R24_MAX_DROP_PCT) -> int:
    """The most drops R24 reports without exceeding its limit (`R24Report.drop_pct_over_limit`)."""
    return instances * pct // 100


# The morning of 10/28, before any pilot target, counted from eval/manifest.json (tests/eval/test_eval_committed.py
# checks them): R24 runs every pool instance's gold in the run phase, and --diagnose runs the harness gold check on
# each drop; the label runner runs our SWE-ABS tests on every instance's gold and on every resolved record (GOOD or
# PLAUSIBLE_WRONG), and a FAIL_TO_PASS run for each unresolved record the entries published no report for (REG-564).
LABEL_DAY_RUNS: dict[str, int] = {
    "r24": POOL_SIZE,
    "r24_diagnose": DIAGNOSE_RUNS_PER_DROP * max_drops(POOL_SIZE),
    "labels_gold": POOL_SIZE,
    "labels_resolved": 1015,
    "labels_fail_to_pass": 1,
}
LABEL_DAY_BASIS: dict[str, str] = {
    "r24": "R24: gold in the run phase, one run per pool instance",
    "r24_diagnose": (
        f"R24 --diagnose: up to {DIAGNOSE_RUNS_PER_DROP} harness runs per drop, at most {max_drops(POOL_SIZE)} drops "
        f"before R24 stops at {R24_MAX_DROP_PCT}%"
    ),
    "labels_gold": "labels: SWE-ABS tests on gold",
    "labels_resolved": "labels: SWE-ABS tests on resolved records",
    "labels_fail_to_pass": "labels: FAIL_TO_PASS runs without a published report",
}
BOOT_MINUTES = 30  # boot, Docker, gVisor and the 251 image pulls
DEBUG_MINUTES = 60
RESTART_MINUTES = 10  # `launch.sh start` after the resize: boot, vm-check over SSH, the orchestrator's setup
LABEL_SLOTS = slots_for(LABEL_INSTANCE_TYPE)
LABEL_BLOCK_FIXED = (
    BOOT_MINUTES + sum(harness_minutes(n, slots=LABEL_SLOTS) for n in LABEL_DAY_RUNS.values()) + DEBUG_MINUTES
)
LABEL_RUN_MINUTES = sum(
    harness_minutes(n, slots=LABEL_SLOTS) for k, n in LABEL_DAY_RUNS.items() if k.startswith("labels")
)
PILOT_DATE = "2026-10-28"
PILOT_TARGETS = 20
PILOT_ARMS = ("spark", "glimmer")  # eval/pilot-arms.json, in the order each job runs them

WORKLOADS: tuple[Workload, ...] = (
    Workload(
        name="R24 and labels",
        date=PILOT_DATE,
        instance_type=LABEL_INSTANCE_TYPE,
        targets=0,
        fixed_minutes=LABEL_BLOCK_FIXED,
        fixed_basis=(
            f"REG-568, on {LABEL_INSTANCE_TYPE} ({LABEL_SLOTS} sandboxes): boot, Docker, gVisor and image pulls "
            f"{BOOT_MINUTES} min; "
            + "; ".join(
                f"{LABEL_DAY_BASIS[k]}, {n} run{'' if n == 1 else 's'} at {HARNESS_RUN_SANDBOX_MINUTES} "
                f"sandbox-minutes, {harness_minutes(n, slots=LABEL_SLOTS)} min"
                for k, n in LABEL_DAY_RUNS.items()
            )
            + f"; debugging {DEBUG_MINUTES} min; then stop and `launch.sh resize {CHOSEN}`"
        ),
    ),
    Workload(
        name="pilot",
        date=PILOT_DATE,
        targets=PILOT_TARGETS * len(PILOT_ARMS),
        fixed_minutes=RESTART_MINUTES,
        fixed_basis=(
            f"`launch.sh start` on {CHOSEN} after the resize, vm-check and setup {RESTART_MINUTES} min; then `receipts "
            f"run --jobs <the VM's sandbox count>`: the {PILOT_TARGETS} pilot targets that many at a time, each a job "
            f"on a sandbox of its own that runs both arms ({' then '.join(PILOT_ARMS)}) in turn, "
            f"{PILOT_TARGETS * len(PILOT_ARMS)} target runs"
        ),
        arms=len(PILOT_ARMS),
        per_job=True,
    ),
    Workload(
        name="dress rehearsal",
        date="2026-10-29",
        targets=140,
        fixed_minutes=30,
        fixed_basis=(
            "start, vm-check and result copy 30 min; 100 targets is what the spec's $60 metered spend buys, plus the "
            "40 paired targets the move rule confirms on (REG-561), as 40 more target runs"
        ),
    ),
    Workload(
        name="final run",
        date="2026-11-01",
        instance_type=FINAL_INSTANCE_TYPE,
        targets=375,
        fixed_minutes=30,
        fixed_basis="start, vm-check and result copy 30 min; 150 BAD, 150 GOOD and 75 PLAUSIBLE_WRONG",
        baseline_sandbox_minutes=33,
    ),
)
SIZED_BY_QUOTE = ("pilot", "dress rehearsal")  # the workloads `quote(instance_type)` moves to another size


def target_run_minutes(w: Workload, case: Case, *, packing_pct: int = PACKING_PCT) -> int:
    """One target run's wall minutes on a job's own sandbox: the sandbox is busy `packing_pct` of the job's time (it
    waits on the models the rest), so a run that needs m sandbox-minutes takes m * 100 / packing_pct minutes."""
    return -(-(case.sandbox_minutes_per_target + w.baseline_sandbox_minutes) * 100 // packing_pct)


def waves(w: Workload, jobs: int) -> int:
    """How many rounds of `jobs` targets a per-job workload's targets take."""
    return -(-(w.targets // w.arms) // jobs)


def run_minutes(w: Workload, case: Case, *, slots: int | None = None, packing_pct: int = PACKING_PCT) -> int:
    """Fixed minutes plus the run. A per-job workload (`receipts run --jobs N`, N the VM's sandbox count, or `slots`)
    takes its waves of jobs, each job its target's arms in turn; any other spreads its sandbox-minutes over the VM's
    sandboxes at `packing_pct` busy."""
    n = slots_for(w.instance_type) if slots is None else slots
    if w.per_job:
        return w.fixed_minutes + waves(w, n) * w.arms * target_run_minutes(w, case, packing_pct=packing_pct)
    sandbox_minutes = w.targets * (case.sandbox_minutes_per_target + w.baseline_sandbox_minutes)
    busy = n * packing_pct  # hundredths of a sandbox
    return w.fixed_minutes + -(-sandbox_minutes * 100 // busy)


class Line(Frozen):
    item: str
    quantity: str
    micro_usd: int


class DayBlock(Frozen):
    what: str
    instance_type: str
    minutes: int
    micro_usd: int


class DayPlan(Frozen):
    """One day's VM time and compute dollars, block by block (the pilot day: REG-568's two sizes, and the GPU VM)."""

    date: str
    blocks: tuple[DayBlock, ...]
    eval_vm_minutes: int
    eval_vm_hours: str
    compute_micro_usd: int

    @property
    def compute_usd(self) -> str:
        return usd(self.compute_micro_usd)


class CaseQuote(Frozen):
    case: str
    sandbox_minutes_per_target: int
    run_minutes: dict[str, int]
    lines: tuple[Line, ...]
    total_micro_usd: int
    pilot_day: DayPlan

    @property
    def total_usd(self) -> str:
        return usd(self.total_micro_usd)


class PilotCase(Frozen):
    """The pilot in one case: a target run's minutes on its job's sandbox, the run, and what both VMs cost for it."""

    target_run_minutes: int  # one target in one arm
    run_minutes: int  # the waves, each job its target's arms in turn
    eval_vm_minutes: int  # the restart after the resize, then the run
    eval_vm_usd: str
    gpu_vm_minutes: int  # its boot, then the whole run: every job calls Glimmer once its Spark arm is done
    gpu_vm_usd: str


class PilotPlan(Frozen):
    instance_type: str
    command: str  # the runbook's pilot line
    jobs: int  # the VM's sandbox count: one target per job, on a sandbox of its own
    targets: int
    arms: tuple[str, ...]  # in the order each job runs them
    waves: int
    cases: dict[str, PilotCase]


class Step(Frozen):
    """One runbook step as the exact command that runs. `mac`: the founder's checkout of main, from its root, with
    eval/data/ in place; `vm`: /home/ubuntu/falden-receipts on the eval VM, in an SSH session (runs/vm/ssh_config)."""

    where: Literal["mac", "vm"]
    command: str = Field(min_length=1)
    note: str = Field(min_length=1)


class Quote(Frozen):
    region: str
    instance_type: str
    usd_per_hour: str  # the on-demand list price, to the hundredth of a cent
    label_instance_type: str  # R24 and the label runs (REG-568)
    label_usd_per_hour: str
    final_instance_type: str
    final_usd_per_hour: str
    why: str
    cheaper_alternative: str
    sources: dict[str, str]
    assumptions: tuple[str, ...]
    cases: tuple[CaseQuote, ...]
    final_on_32_vcpu: dict[str, str]  # the same final on m7i.8xlarge: about the same dollars, twice the wall time
    label_runs: dict[str, str]  # REG-564's runs before any pilot target, and their VM hours (REG-568: on 64 vCPU)
    gpu: dict[str, str]  # the Glimmer arm's VM
    pilot: PilotPlan  # `receipts run --jobs <the pilot VM's sandbox count>`: its waves, wall time and VM dollars
    launch: tuple[str, ...]  # what the integrator does in AWS; nothing in this repository does it
    runbook: tuple[Step, ...]  # every step on and around the VM, as the exact command that runs


GPU_INSTANCE = "g7e.2xlarge"
GPU_MICRO_USD_PER_HOUR = 3_363_120  # $3.36312 (research/open-weights-2026-10-27.md, the same price data as above)
GPU_BOOT_MINUTES = 60  # boot, the vLLM image, about 65 GB of pinned weights, model load
GPU_RESTART_MINUTES = 20  # start from stopped: the weights are on disk, the server reloads them
GPU_DISK = Disk(gib=200, iops=3000, throughput_mibps=125, exists_hours=48)  # 10/28 08:00 to 10/30 08:00 ET

# The integrator's steps (eval/vm/launch.sh and gpu.sh run each AWS one with the integrator's own profile; this lane
# runs none).
LAUNCH: tuple[str, ...] = (
    "`mkdir -m 700 <a private folder outside the repository>` for the key, then `AWS_PROFILE=<integrator profile> "
    "RECEIPTS_KEY_DIR=<that folder> bash eval/vm/launch.sh launch`: it reads the founder's current public IP from two "
    "services and prints it, then prints the size it launches and the vCPUs that size needs, before anything "
    f"changes; checks the account, the Standard vCPU quota ({offer(LABEL_INSTANCE_TYPE).vcpu} or more for "
    f"{LABEL_INSTANCE_TYPE}), the pinned AMI (eval/vm/ami.json: owner, name, state, architecture, root device) and "
    "that no receipts-eval instance exists; creates the security group (inbound 22/tcp from that IP's /32 only, "
    "egress TCP 443 and 80) and checks it; creates a new ed25519 key pair and writes its private key into the folder "
    f"with mode 600; launches {LABEL_INSTANCE_TYPE} (REG-568: R24 and the label runs) from eval/vm/run-instances.json "
    "(500 GiB encrypted gp3 at 6000 IOPS and 250 MiB/s, IMDSv2 only, hop limit 1, no instance profile, termination "
    "protection, tags from eval/vm/tags.json) with eval/vm/cloud-init.yaml.gz as user data; checks the running "
    "instance; pins its SSH host keys from its console",
    "`bash eval/vm/launch.sh ssh-config`, then `ssh -F runs/vm/ssh_config receipts-eval`: SSH is the only way in "
    "(REG-562: no instance role, no Systems Manager; the boot script removes the SSM agent)",
    f"after R24 and the label runs (REG-568): copy their results off, `launch.sh stop`, `launch.sh resize {CHOSEN}` "
    f"(it prints the {offer(CHOSEN).vcpu} vCPUs it needs), then `launch.sh start` for the pilot",
    "the Glimmer arm, once the G and VT quota is granted: pin the AMI with one `aws ec2 describe-images` call, then "
    "`RECEIPTS_GPU_AMI=<ami id> bash eval/vm/gpu.sh launch` (g7e.2xlarge, reachable from the eval VM's security "
    "group only; without a pin it takes the newest image by name, and either way it refuses an image whose name is "
    "not the expected one and prints the id, name and date), wait for /opt/receipts/READY on it, `gpu.sh seal` (no "
    "egress from then on), and `gpu.sh url` for RECEIPTS_GLIMMER_URL on the eval VM; `gpu.sh stop` after the pilot; "
    "if the quota is late, eval/vm/gpu-plan.json says how to run the arm on Modal",
    "between runs: copy the run's results off the VM, then `launch.sh stop` (compute and the public IPv4 address "
    "stop billing; the volume does not); before the next run `launch.sh start`, which moves the SSH rule to the "
    "founder's current /32 if it changed, then runs /opt/receipts/vm-check.sh over SSH (boot finished, no role, "
    "IMDSv1 off, no SSM agent, runsc is gVisor and runc is not, every image on disk)",
    f"before the 11/1 final, with the VM stopped: `launch.sh resize {FINAL_INSTANCE_TYPE}` (REG-563; it checks the "
    f"Standard vCPU quota is {offer(FINAL_INSTANCE_TYPE).vcpu} or more), then `launch.sh start`",
    "after the final run: `gpu.sh terminate --yes <id>` if it still exists, then `launch.sh terminate --yes "
    "<instance-id>` (deletes the volume, the security group and the key pair in AWS; the private key file stays)",
)


VM_REPO = "/home/ubuntu/falden-receipts"
SSH_CONFIG = "runs/vm/ssh_config"  # eval/vm/launch.sh's state: written by `launch.sh launch`, `start` and `ssh-config`
GOLD_DROPS = "eval/gold-drops.tsv"  # the founder's R24 drops, one <instance_id><TAB><reason> per line
PILOT_JOBS = slots_for(CHOSEN)  # `receipts run --jobs`: the pilot VM's sandbox count
RSYNC_TO_VM = (
    "rsync -a --delete --exclude=.git/ --exclude=.venv/ --exclude=runs/ --exclude=.receipts/ --exclude=__pycache__/ "
    f"-e 'ssh -F {SSH_CONFIG}' ./ receipts-eval:{VM_REPO}/"
)
RSYNC_FROM_VM = f"rsync -a -e 'ssh -F {SSH_CONFIG}' receipts-eval:{VM_REPO}/runs/ runs/"
LONG_RUN = "Run it inside tmux or screen, so a dropped SSH connection does not stop it."


def pilot_command(jobs: int) -> str:
    """The pilot as it runs on the VM: both arms on every pilot target, `jobs` targets at once."""
    return (
        "uv run --frozen --offline receipts run --targets eval/pilot-run.jsonl --arms eval/pilot-arms.json "
        f"--ledger-dir runs/pilot --jobs {jobs}"
    )


RUNBOOK: tuple[Step, ...] = (
    Step(
        where="mac",
        command=f"ssh -F {SSH_CONFIG} receipts-eval cat /opt/receipts/BOOT_OK",
        note="The first boot finished. If it did not, read /opt/receipts/BOOT_FAILED and /opt/receipts/bootstrap.log.",
    ),
    Step(
        where="mac",
        command=RSYNC_TO_VM,
        note=(
            "Main's tree and eval/data/ to the VM. eval/data/ is not in git, and main's checkout has none: copy it "
            "from the eval lane's worktree, or rebuild it with `uv run --frozen python -m receipts.eval fetch`, "
            "`uv run --frozen --extra eval python -m receipts.eval rows` and `uv run --frozen python -m receipts.eval "
            "patches` (eval/manifest.json pins every file in it by sha256). runs/ (the VM's results, and the Mac's "
            "SSH state in runs/vm), .receipts/ (each machine's spend file) and .venv/ never move."
        ),
    ),
    Step(
        where="vm",
        command="uv sync --frozen --extra eval",
        note=(
            "The project and its eval extra from uv.lock, each file checked against its sha256, nothing resolved: "
            "pyarrow, and swebench 4.1.0 (the bytes of the wheel eval/manifest.json pins) with every package it needs."
        ),
    ),
    Step(
        where="vm",
        command=(
            "uv run --frozen --offline --extra eval receipts eval r24 --instances eval/pools/swe_abs_251.txt "
            "--out runs/r24 --run-id d1 --diagnose"
        ),
        note=(
            "R24 on the run phase's real path: the attack lane's sandbox under runsc and its runners (cli.py wires "
            "them), after the VM preflight, one worker per 2 vCPU (32 on the m7i.16xlarge). --diagnose grades each "
            "drop with swebench from the eval extra. Exit 0 done; 3 more than 25% dropped; 2 not ready, or unchecked "
            "instances: rerun the ones a rerun can settle, with --instances <a file of their ids> --out "
            "runs/r24-rerun1 --run-id d1r1, and pass --gold-rerun runs/r24-rerun1 at the freeze. "
            f"sphinx-doc__sphinx-8265 and 8621 stay unchecked whatever a run does (REG-571 drops them). {LONG_RUN}"
        ),
    ),
    Step(
        where="vm",
        command="uv run --frozen --offline --extra eval python -m receipts.eval labels --out runs/labels --run-id d1",
        note=(
            "REG-564's label runs, graded by swebench 4.1.0 from the eval extra. It skips every job with a final "
            "result, so running it again resumes; exit 2 while INFRA jobs remain. swebench's make_test_spec fetches "
            "some repositories' requirement files from raw.githubusercontent.com, which the VM's 443 egress allows, "
            f"and so does --diagnose. {LONG_RUN}"
        ),
    ),
    Step(
        where="mac",
        command=RSYNC_FROM_VM,
        note="R24's and the label runs' results to the Mac: runs/r24, runs/labels and any rerun.",
    ),
    Step(
        where="mac",
        command=(
            f"uv run --frozen python -m receipts.eval manifest --gold-runsc runs/r24 --gold-drop {GOLD_DROPS} "
            "--labels runs/labels --freeze"
        ),
        note=(
            f"The freeze. {GOLD_DROPS} holds the founder's two pre-approved drops, sphinx-doc__sphinx-8265 and 8621 "
            "(REG-571: no FAIL_TO_PASS id of either names a case). Add --gold-rerun runs/<rerun> for each R24 rerun; "
            f"any other instance still unchecked needs a founder line in {GOLD_DROPS}, by name with a reason. "
            "--label-drop <instance>=<reason> is the founder's too, never pre-approved: only for an instance R24 kept "
            "whose unfinished label jobs were INFRA on every run, and it needs --labels and --gold-runsc. Exit 2 while "
            "anything blocks the freeze."
        ),
    ),
    Step(
        where="mac",
        command="uv run --frozen python -m receipts.eval sample --plan spec",
        note=(
            "The seeded 150+150+75 draw and its 40/60 split from the frozen manifest; exit 3 when the candidates "
            "cannot fill it: then --plan core (100+100+50)."
        ),
    ),
    Step(
        where="mac",
        command="uv run --frozen python -m receipts.eval pilot",
        note=(
            "eval/pilot.json, eval/pilot-run.jsonl and eval/pilot-arms.json from the frozen sample. The integrator "
            f"commits them with eval/manifest.json and eval/sample.json, then `launch.sh stop`, `launch.sh resize "
            f"{CHOSEN}` and `launch.sh start` (REG-568), and starts the GPU VM (`gpu.sh launch`, `seal`, `url`)."
        ),
    ),
    Step(
        where="mac",
        command=RSYNC_TO_VM,
        note="The frozen manifest, sample and pilot files to the VM.",
    ),
    Step(
        where="vm",
        command="uv sync --frozen",
        note=(
            "An exact sync to the project's own dependencies: it removes the eval extra, so the process that holds "
            "the Meta key runs without the harness's packages."
        ),
    ),
    Step(
        where="vm",
        command="uv run --frozen --offline receipts spend arm --credit-confirmed",
        note=(
            "The founder, once the Meta credit is confirmed as applied: arms this VM's .receipts/spend.jsonl at the "
            "$150 cap. Every model call is refused until then."
        ),
    ),
    Step(
        where="vm",
        command=pilot_command(PILOT_JOBS),
        note=(
            "The pilot. In the same shell first: `read -rs FALDEN_META_API_KEY && export FALDEN_META_API_KEY` (the "
            "founder pastes the key; it is never written to the VM's disk or shell history, and never reaches a "
            "sandbox) and `export RECEIPTS_GLIMMER_URL=<what gpu.sh url printed>`. --jobs is the VM's sandbox count: "
            "one target per job, on a sandbox of its own, running spark and then glimmer (the core lane's per-job "
            f"run phase). After a stop, run it again with --resume. {LONG_RUN}"
        ),
    ),
    Step(
        where="vm",
        command="uv run --frozen --offline receipts eval score --ledgers runs/pilot --out runs/score",
        note=(
            "Every blocker's repro replayed on gold in the run phase, then REG-561's rule. Exit 2 while a replay is "
            "unscored infra: run it again with the same --out (settled replays are cached). A replay still infra "
            "after that may be dropped only by the founder, by name: --replay-drop <finding id>=<reason>, which "
            "counts that finding on neither side."
        ),
    ),
    Step(
        where="mac",
        command=RSYNC_FROM_VM,
        note="The pilot's ledgers, summaries, traces and score to the Mac; then `gpu.sh stop` and `launch.sh stop`.",
    ),
)


def usd_rate(micro: int) -> str:
    """A unit price to four decimals, as the price list shows it: 1_612_800 -> $1.6128."""
    return f"${micro // MICRO}.{micro % MICRO // 100:04d}"


def usd(micro: int) -> str:
    cents = -(-micro // 10_000)
    return f"${cents // 100:,}.{cents % 100:02d}"


GLIMMER_REHEARSAL_TARGETS = 40  # REG-561's confirmation on 40 paired targets


def glimmer_block_minutes(targets: int, case: Case, *, slots: int | None = None, packing_pct: int = PACKING_PCT) -> int:
    """The GPU VM is up while the Glimmer arm's targets run as one block on the eval VM (then it is stopped)."""
    busy = (slots_for(CHOSEN) if slots is None else slots) * packing_pct
    return -(-targets * case.sandbox_minutes_per_target * 100 // busy)


def pilot_gpu_minutes(pilot: Workload, case: Case) -> int:
    """The GPU VM on pilot day: its boot, then the pilot's whole run, since each job runs its target's Glimmer arm
    right after its Spark arm, so Glimmer is called from the first wave to the last. The eval VM's restart after the
    resize happens during the GPU VM's boot."""
    return GPU_BOOT_MINUTES + run_minutes(pilot, case) - pilot.fixed_minutes


def _compute(instance_type: str, minutes: int) -> int:
    return -(-check_offer(offer(instance_type)).micro_usd_per_hour * minutes // 60)


def quote_case(case: Case, workloads: Sequence[Workload], disk: Disk) -> CaseQuote:
    minutes = {w.name: run_minutes(w, case) for w in workloads}
    total_minutes = sum(minutes.values())
    lines = []
    for w, m in zip(workloads, minutes.values(), strict=True):
        lines.append(
            Line(
                item=f"{w.instance_type} on-demand, {w.name} ({w.date})",
                quantity=f"{m} min",
                micro_usd=_compute(w.instance_type, m),
            )
        )
    lines.append(
        Line(
            item=f"gp3 {disk.gib} GiB, {disk.iops} IOPS, {disk.throughput_mibps} MiB/s",
            quantity=f"{disk.exists_hours} h",
            micro_usd=disk.micro_usd(),
        )
    )
    gpu_pilot = pilot_gpu_minutes(next(w for w in workloads if w.name == "pilot"), case)
    gpu_rehearsal = GPU_RESTART_MINUTES + glimmer_block_minutes(GLIMMER_REHEARSAL_TARGETS, case)
    gpu_pilot_usd = -(-GPU_MICRO_USD_PER_HOUR * gpu_pilot // 60)
    lines += [
        Line(
            item=f"{GPU_INSTANCE} on-demand, the Glimmer arm on pilot day",
            quantity=f"{gpu_pilot} min",
            micro_usd=gpu_pilot_usd,
        ),
        Line(
            item=f"{GPU_INSTANCE} on-demand, the 40 paired targets in the rehearsal (only if the move rule is met)",
            quantity=f"{gpu_rehearsal} min",
            micro_usd=-(-GPU_MICRO_USD_PER_HOUR * gpu_rehearsal // 60),
        ),
        Line(
            item=f"gp3 {GPU_DISK.gib} GiB for the GPU VM",
            quantity=f"{GPU_DISK.exists_hours} h",
            micro_usd=GPU_DISK.micro_usd(),
        ),
    ]
    ipv4_minutes = total_minutes + gpu_pilot + gpu_rehearsal
    lines.append(
        Line(
            item="public IPv4 addresses while running (eval VM and GPU VM)",
            quantity=f"{ipv4_minutes} min",
            micro_usd=-(-IPV4_MICRO_USD_PER_HOUR * ipv4_minutes // 60),
        )
    )
    lines.append(Line(item="data in (image and weight pulls) and data out under 100 GB", quantity="0", micro_usd=0))
    day = [w for w in workloads if w.date == PILOT_DATE]
    blocks = [DayBlock(what=w.name, instance_type=w.instance_type, minutes=minutes[w.name],
                       micro_usd=_compute(w.instance_type, minutes[w.name])) for w in day]  # fmt: skip
    eval_minutes = sum(b.minutes for b in blocks)
    blocks.append(
        DayBlock(
            what="the Glimmer arm's GPU VM", instance_type=GPU_INSTANCE, minutes=gpu_pilot, micro_usd=gpu_pilot_usd
        )
    )
    return CaseQuote(
        case=case.name,
        sandbox_minutes_per_target=case.sandbox_minutes_per_target,
        run_minutes=minutes,
        lines=tuple(lines),
        total_micro_usd=sum(line.micro_usd for line in lines),
        pilot_day=DayPlan(
            date=PILOT_DATE,
            blocks=tuple(blocks),
            eval_vm_minutes=eval_minutes,
            eval_vm_hours=f"{eval_minutes / 60:.1f}",
            compute_micro_usd=sum(b.micro_usd for b in blocks),
        ),
    )


def pilot_plan(pilot: Workload, cases: Sequence[Case]) -> PilotPlan:
    """`receipts run --jobs N` with N the pilot VM's sandbox count, case by case: its waves, a target run's minutes,
    and the minutes and dollars of the eval VM (the restart after the resize, then the run) and of the GPU VM."""
    jobs = slots_for(pilot.instance_type)
    by_case = {}
    for c in cases:
        eval_minutes = run_minutes(pilot, c)
        gpu_minutes = pilot_gpu_minutes(pilot, c)
        by_case[c.name] = PilotCase(
            target_run_minutes=target_run_minutes(pilot, c),
            run_minutes=eval_minutes - pilot.fixed_minutes,
            eval_vm_minutes=eval_minutes,
            eval_vm_usd=usd(_compute(pilot.instance_type, eval_minutes)),
            gpu_vm_minutes=gpu_minutes,
            gpu_vm_usd=usd(-(-GPU_MICRO_USD_PER_HOUR * gpu_minutes // 60)),
        )
    return PilotPlan(
        instance_type=pilot.instance_type,
        command=pilot_command(jobs),
        jobs=jobs,
        targets=pilot.targets // pilot.arms,
        arms=PILOT_ARMS,
        waves=waves(pilot, jobs),
        cases=by_case,
    )


def quote(
    instance_type: str = CHOSEN,
    *,
    workloads: Sequence[Workload] = WORKLOADS,
    cases: Sequence[Case] = CASES,
    disk: Disk | None = None,
) -> Quote:
    """The quote with the pilot and the rehearsal on `instance_type`, R24 and the labels on LABEL_INSTANCE_TYPE
    (REG-568), and the final as WORKLOADS has it."""
    o = check_offer(offer(instance_type))
    d = disk or Disk()
    ws = [w.model_copy(update={"instance_type": o.instance_type}) if w.name in SIZED_BY_QUOTE else w for w in workloads]
    case_quotes = tuple(quote_case(c, ws, d) for c in cases)
    expected = next(c for c in cases if c.name == "expected")
    final = next(w for w in ws if w.name == "final run")
    final_offer = check_offer(offer(final.instance_type))
    label_offer = check_offer(offer(LABEL_INSTANCE_TYPE))
    on_32 = final.model_copy(update={"instance_type": CHOSEN})
    on_32_minutes = run_minutes(on_32, expected)
    label_slots = LABEL_SLOTS
    pilot = pilot_plan(next(w for w in ws if w.name == "pilot"), cases)
    return Quote(
        region=REGION,
        instance_type=o.instance_type,
        usd_per_hour=usd_rate(o.micro_usd_per_hour),
        label_instance_type=label_offer.instance_type,
        label_usd_per_hour=usd_rate(label_offer.micro_usd_per_hour),
        final_instance_type=final_offer.instance_type,
        final_usd_per_hour=usd_rate(final_offer.micro_usd_per_hour),
        why=WHY,
        cheaper_alternative=CHEAPER,
        sources={
            "ec2_page": EC2_PRICE_PAGE,
            "ec2_data": EC2_PRICE_DATA,
            "ec2_published": EC2_PRICE_PUBLISHED,
            "ebs_page": EBS_PRICE_PAGE,
            "ebs_data": EBS_PRICE_DATA,
            "ebs_published": EBS_PRICE_PUBLISHED,
            "ipv4_page": IPV4_PRICE_PAGE,
        },
        assumptions=(
            "on-demand Linux is billed per second with a 60 second minimum, from launch until stop (EC2 page)",
            "the VMs are stopped between runs, so compute and the public IPv4 addresses bill only while they run",
            "the gp3 volumes bill for every hour they exist, stopped or not",
            f"{SANDBOX_VCPU} vCPU per sandbox ({slots_for(CHOSEN)} on 32 vCPU, {label_slots} on 64), {PACKING_PCT}% "
            "busy on average; the runs take their worker count from the VM's vCPUs the same way",
            "data transfer in is $0.00 per GB and the first 100 GB out each month are free (EC2 page)",
            "sandbox-minutes per target are a model; the pilot measures the real number",
            "the final run's baselines (B0, B1, B2 through the SK1 gates) add 33 sandbox-minutes per target",
            f"one SWE-bench eval script takes {HARNESS_RUN_SANDBOX_MINUTES} sandbox-minutes on average (a model)",
            f"R24 --diagnose is counted at its most: {DIAGNOSE_RUNS_PER_DROP} harness runs for each of "
            f"{max_drops(POOL_SIZE)} drops, the most R24 reports before it stops at {R24_MAX_DROP_PCT}%",
            f"R24 and the label runs on {LABEL_INSTANCE_TYPE}, then a resize to {CHOSEN} for the pilot (REG-568); "
            "the resize happens while the VM is stopped, so it bills nothing",
            f"the pilot is `receipts run --jobs {pilot.jobs}`, the pilot VM's sandbox count: {pilot.targets} targets "
            f"in {pilot.waves} waves, each a job on a sandbox of its own that runs {' then '.join(pilot.arms)} in "
            f"turn, its sandbox {PACKING_PCT}% busy, so one target run of m sandbox-minutes takes m * 100 / "
            f"{PACKING_PCT} minutes",
            f"the GPU VM is up from its boot ({GPU_BOOT_MINUTES} min) through the whole pilot, since every job calls "
            f"Glimmer after its Spark arm; in the rehearsal it is up for the {GLIMMER_REHEARSAL_TARGETS} paired "
            f"targets' block plus {GPU_RESTART_MINUTES} min from stopped",
            "the rehearsal and the final keep the sandbox-minutes model until the pilot measures a target's time",
        ),
        cases=case_quotes,
        final_on_32_vcpu={
            "instance_type": CHOSEN,
            "final_run_minutes_32_vcpu": str(on_32_minutes),
            "final_run_minutes_64_vcpu": str(run_minutes(final, expected)),
        },
        label_runs={
            "instance_type": LABEL_INSTANCE_TYPE,
            "sandboxes": str(label_slots),
            "runs": str(sum(LABEL_DAY_RUNS.values())),
            "breakdown": "; ".join(f"{LABEL_DAY_BASIS[k]}: {v}" for k, v in LABEL_DAY_RUNS.items()),
            "label_runner_minutes": str(LABEL_RUN_MINUTES),
            "label_runner_vm_hours": f"{LABEL_RUN_MINUTES / 60:.1f}",
            "block_minutes": str(LABEL_BLOCK_FIXED),
            "block_vm_hours": f"{LABEL_BLOCK_FIXED / 60:.1f}",
            "block_usd": usd(_compute(LABEL_INSTANCE_TYPE, LABEL_BLOCK_FIXED)),
        },
        gpu={
            "instance_type": GPU_INSTANCE,
            "usd_per_hour": usd_rate(GPU_MICRO_USD_PER_HOUR),
            "boot_minutes": str(GPU_BOOT_MINUTES),
            "plan": "eval/vm/gpu-plan.json",
        },
        pilot=pilot,
        launch=LAUNCH,
        runbook=RUNBOOK,
    )
