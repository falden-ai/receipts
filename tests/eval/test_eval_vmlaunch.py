"""The VM launch inputs and eval/vm/launch.sh and gpu.sh, run here against fake `aws`, `curl` and `ssh` commands that
record every call (nothing reaches AWS). REG-562: SSH only, 22/tcp from the founder's current /32 read at launch and
printed first, a new key pair whose private key lands in a mode-700 folder with mode 600, no instance role, no
Systems Manager. REG-568: the VM launches as an m7i.16xlarge for R24 and the label runs, then resizes to m7i.8xlarge
for the pilot; REG-563: back to m7i.16xlarge for the final. The Glimmer VM is reachable from the eval VM's group only,
launches from a pinned (or name-checked) AMI and has no egress once sealed."""

from __future__ import annotations

import json
import os
import re
import stat
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from receipts.eval.gpuvm import GPU_AMI_NAME_PREFIX, GPU_INSTANCE_TYPE, VLLM_PORT
from receipts.eval.vmlaunch import AMI, EGRESS_PORTS, TAGS, egress_rules, expected_identity, files, run_instances_input

ROOT = Path(__file__).resolve().parents[2]
LAUNCH = ROOT / "eval" / "vm" / "launch.sh"
GPU = ROOT / "eval" / "vm" / "gpu.sh"
INSTANCE = "i-0123456789abcdef0"
GPU_INSTANCE = "i-0fedcba9876543210"
GROUP = "sg-0abc"
GPU_GROUP = "sg-0gpu"
MY_IP = "73.12.34.56"  # a public address: the documentation nets (203.0.113/24 and the rest) are refused
KEY_MATERIAL = "-----BEGIN TEST" + " PRIVATE KEY-----\nnot-a-key\n-----END TEST" + " PRIVATE KEY-----"
HOST_KEY = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAITestHostKeyOnly"
CONSOLE = (
    "cloud-init: boot\n-----BEGIN SSH HOST KEY KEYS-----\n"
    f"{HOST_KEY} root@ip-10-0-0-5\necdsa-sha2-nistp256 AAAAE2VjZHNhTestOnly root@ip-10-0-0-5\n"
    "-----END SSH HOST KEY KEYS-----\nlogin:\n"
)

FAKE_AWS = r"""
import json, os, sys
argv = sys.argv[1:]
with open(os.environ["FAKE_LOG"], "a") as log:
    log.write(json.dumps(["aws", *argv]) + "\n")
args = [a for i, a in enumerate(argv) if a != "--region" and (i == 0 or argv[i - 1] != "--region")]
key = f"{args[0]} {args[1]}"
def has(text):
    return any(text in a for a in args)
def value(flag):
    return args[args.index(flag) + 1] if flag in args else ""
if key == "ec2 describe-instances":
    if has("--filters"):
        key += " live"
    elif has("MetadataOptions"):
        key += " verify"
    elif has("LaunchTime"):
        key += " status"
    elif has("PublicIpAddress"):
        key += " public-ip"
    elif has("PrivateIpAddress"):
        key += " private-ip"
    elif has("State.Name"):
        key += " state"
    key += " " + value("--instance-ids")
if key == "ec2 describe-security-groups":
    if has("--filters"):
        key += " lookup"
    else:
        key += (" egress " if has("IpPermissionsEgress") else " ingress ") + value("--group-ids")
if key == "service-quotas get-service-quota":
    key += " " + value("--quota-code")
if key == "ec2 describe-images":
    key += " " + (value("--image-ids") or "by-name")
scenario = json.load(open(os.environ["FAKE_SCENARIO"]))
answer = scenario.get(key, {"out": ""})
if isinstance(answer, list):
    counts_path = os.environ["FAKE_SCENARIO"] + ".counts"
    counts = json.load(open(counts_path)) if os.path.exists(counts_path) else {}
    n = counts.get(key, 0)
    counts[key] = n + 1
    json.dump(counts, open(counts_path, "w"))
    answer = answer[min(n, len(answer) - 1)]
if key == "ec2 create-key-pair" and answer.get("out") == "KEY":
    answer = {"out": os.environ["FAKE_KEY"]}
if "{KEY}" in answer.get("out", ""):
    name = open(os.path.join(os.environ["RECEIPTS_EVAL_STATE"], "key-name")).read().strip()
    answer = dict(answer, out=answer["out"].replace("{KEY}", name))
if answer.get("out"):
    print(answer["out"])
sys.stderr.write(answer.get("err", ""))
sys.exit(answer.get("code", 0))
"""

FAKE_CURL = """#!/usr/bin/env bash
echo "[\\"curl\\", \\"${@: -1}\\"]" >> "$FAKE_LOG"
case "${@: -1}" in
  https://checkip.amazonaws.com) [ -n "$IP_A" ] || exit 7; echo "$IP_A" ;;
  https://api.ipify.org) [ -n "$IP_B" ] || exit 7; printf '%s' "$IP_B" ;;
  *) exit 99 ;;
esac
"""

FAKE_SSH = """#!/usr/bin/env bash
printf '%s\\n' "$*" | python3 -c 'import json,sys; print(json.dumps(["ssh", sys.stdin.read().strip()]))' >> "$FAKE_LOG"
echo "${SSH_OUT:-vm ok: gVisor, 251 images, no instance role, no SSM agent}"
exit "${SSH_CODE:-0}"
"""


def eval_ingress(ip: str = MY_IP) -> dict[str, str]:
    return {"out": f"tcp\t22\t22\t{ip}/32\t0\t0\t0"}


def eval_egress(gpu: bool = False) -> dict[str, str]:
    lines = ["tcp\t443\t443\t0.0.0.0/0\t", "tcp\t80\t80\t0.0.0.0/0\t"]
    if gpu:
        lines += [f"tcp\t8000\t8000\t\t{GPU_GROUP}", f"tcp\t22\t22\t\t{GPU_GROUP}"]
    return {"out": "\n".join(lines)}


def happy() -> dict[str, Any]:
    return {
        "sts get-caller-identity": {"out": "123456789012"},
        "service-quotas get-service-quota L-1216C47A": {"out": "64.0"},
        "service-quotas get-service-quota L-DB2E81BA": {"out": "16.0"},
        f"ec2 describe-images {AMI.image_id}": {"out": expected_identity()},
        "ec2 describe-instances live ": {"out": ""},
        f"ec2 describe-instances verify {INSTANCE}": {"out": "PLACEHOLDER"},
        f"ec2 describe-instances status {INSTANCE}": {"out": f"{INSTANCE}\trunning\tm7i.8xlarge\t198.51.100.4\tT"},
        f"ec2 describe-instances public-ip {INSTANCE}": {"out": "198.51.100.4"},
        f"ec2 describe-instances state {INSTANCE}": {"out": "stopped"},
        f"ec2 describe-instances private-ip {GPU_INSTANCE}": {"out": "10.0.0.9"},
        "ec2 describe-vpcs": {"out": "vpc-0123"},
        "ec2 describe-security-groups lookup": {"out": "None"},
        f"ec2 describe-security-groups ingress {GROUP}": eval_ingress(),
        f"ec2 describe-security-groups egress {GROUP}": eval_egress(),
        f"ec2 describe-security-groups ingress {GPU_GROUP}": {
            "out": f"tcp\t8000\t8000\t\t0\t0\t{GROUP}\ntcp\t22\t22\t\t0\t0\t{GROUP}"
        },
        f"ec2 describe-security-groups egress {GPU_GROUP}": {"out": "tcp\t443\t443\t0.0.0.0/0"},
        "ec2 create-security-group": {"out": GROUP},
        "ec2 describe-key-pairs": {"code": 254, "err": "InvalidKeyPair.NotFound"},
        "ec2 create-key-pair": {"out": "KEY"},
        "ec2 run-instances": {"out": INSTANCE},
        "ec2 get-console-output": {"out": CONSOLE},
    }


def verify_answer(key_name: str, kind: str = "m7i.8xlarge", profile: str = "None") -> dict[str, str]:
    return {"out": f"{kind}\t{AMI.image_id}\trequired\t1\t{profile}\t{key_name}"}


def env_for(tmp_path: Path, scenario: dict[str, Any], *, profile: bool = True, ips: tuple[str, str] = (MY_IP, MY_IP),
            keys: Path | None = None, ssh_out: str | None = None, ssh_code: int = 0,
            extra: dict[str, str] | None = None) -> dict[str, str]:  # fmt: skip
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    (tmp_path / "fake_aws.py").write_text(FAKE_AWS)
    aws = bin_dir / "aws"
    aws.write_text(f'#!/usr/bin/env bash\nexec "{sys.executable}" -I "{tmp_path / "fake_aws.py"}" "$@"\n')
    for name, body in (("curl", FAKE_CURL), ("ssh", FAKE_SSH)):
        (bin_dir / name).write_text(body)
    for f in bin_dir.iterdir():
        f.chmod(f.stat().st_mode | stat.S_IEXEC)
    (tmp_path / "scenario.json").write_text(json.dumps(scenario))
    (tmp_path / "scenario.json.counts").unlink(missing_ok=True)
    log = tmp_path / "calls.jsonl"
    if not log.exists():
        log.write_text("")
    if keys is None:
        keys = tmp_path / "keys"
        keys.mkdir(exist_ok=True)
        keys.chmod(0o700)
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "HOME": str(tmp_path),
        "FAKE_SCENARIO": str(tmp_path / "scenario.json"),
        "FAKE_LOG": str(log),
        "FAKE_KEY": KEY_MATERIAL,
        "RECEIPTS_EVAL_STATE": str(tmp_path / "state"),
        "RECEIPTS_EVAL_POLL_S": "0",
        "RECEIPTS_KEY_DIR": str(keys),
        "IP_A": ips[0],
        "IP_B": ips[1],
        "SSH_CODE": str(ssh_code),
    }
    if ssh_out is not None:
        env["SSH_OUT"] = ssh_out
    if profile:
        env["AWS_PROFILE"] = "integrator-test"
    return env | (extra or {})


def run(script: Path, tmp_path: Path, command: list[str], scenario: dict[str, Any], **kw: Any) -> Any:
    env = env_for(tmp_path, scenario, **kw)
    log = Path(env["FAKE_LOG"])
    log.write_text("")
    p = subprocess.run(
        ["bash", str(script), *command], env=env, capture_output=True, text=True, timeout=120, check=False
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    return p, calls


def ops(calls: list[list[str]]) -> list[str]:
    return [" ".join(c[3:5]) if c[0] == "aws" else c[0] for c in calls]


def launch_scenario() -> dict[str, Any]:
    """The happy path; verify answers with the key pair launch.sh just created, whatever its timestamped name, at the
    launch size (REG-568)."""
    s = happy()
    s[f"ec2 describe-instances verify {INSTANCE}"] = verify_answer("{KEY}", "m7i.16xlarge")
    return s


def test_launch_reads_the_ip_first_and_launches_with_ssh_only_a_new_key_and_no_role(tmp_path: Path) -> None:
    p, calls = run(LAUNCH, tmp_path, ["launch"], launch_scenario())
    assert p.returncode == 0, p.stderr
    state = tmp_path / "state"
    key_name = (state / "key-name").read_text().strip()
    order = ops(calls)
    assert order[:2] == ["curl", "curl"]  # the founder's IP is read before any AWS call
    assert f"SSH will be open to {MY_IP}/32 only" in p.stdout.splitlines()[0]
    assert all(c[1:3] == ["--region", "us-east-1"] for c in calls if c[0] == "aws")
    ingress = next(c for c in calls if c[3:5] == ["ec2", "authorize-security-group-ingress"])
    rule = json.loads(ingress[ingress.index("--ip-permissions") + 1])
    assert rule == [
        {"IpProtocol": "tcp", "FromPort": 22, "ToPort": 22,
         "IpRanges": [{"CidrIp": f"{MY_IP}/32", "Description": "founder SSH, REG-562"}]}
    ]  # fmt: skip
    assert order.index("ec2 authorize-security-group-ingress") < order.index("ec2 run-instances")
    launch = next(c for c in calls if c[3:5] == ["ec2", "run-instances"])
    assert launch[launch.index("--key-name") + 1] == key_name and key_name.startswith("receipts-eval-")
    assert launch[launch.index("--security-group-ids") + 1] == GROUP
    assert f"fileb://{ROOT}/eval/vm/cloud-init.yaml.gz" in launch
    flat = " ".join(a for c in calls for a in c)
    assert "iam-instance-profile" not in flat.lower() and "IamInstanceProfile=" not in flat
    assert not any(c[3] == "ssm" for c in calls if c[0] == "aws")  # no Systems Manager call at all
    create = next(c for c in calls if c[3:5] == ["ec2", "create-key-pair"])
    assert "ed25519" in create
    key = tmp_path / "keys" / f"{key_name}.pem"
    assert key.read_text().strip() == KEY_MATERIAL and stat.S_IMODE(key.stat().st_mode) == 0o600
    assert "not-a-key" not in p.stdout + p.stderr  # the private key is never printed
    assert (state / "known_hosts").read_text().startswith(f"receipts-eval {HOST_KEY}\n")
    cfg = (state / "ssh_config").read_text()
    assert "HostKeyAlias receipts-eval" in cfg and "StrictHostKeyChecking yes" in cfg and str(key) in cfg
    assert "HostName 198.51.100.4" in cfg
    # REG-568: it says what it launches and the vCPUs that needs before any AWS call, and launches exactly that
    lines = p.stdout.splitlines()
    assert (
        lines[1].startswith("launch.sh: launching m7i.16xlarge, which needs 64 vCPUs")
        and "resize m7i.8xlarge" in lines[1]
    )
    assert lines.index(lines[1]) < next(n for n, line in enumerate(lines) if line.startswith("launch.sh: account "))
    assert (state / "instance-type").read_text().strip() == "m7i.16xlarge"
    assert json.loads(files()["run-instances.json"])["InstanceType"] == "m7i.16xlarge"


@pytest.mark.parametrize(
    ("plant", "env", "match"),
    [
        ({}, {"ips": ("73.12.34.56", "73.12.34.57")}, "the IP services disagree"),
        ({}, {"ips": ("10.1.2.3", "10.1.2.3")}, "not a public IPv4 address"),
        ({}, {"ips": ("192.168.1.10", "192.168.1.10")}, "not a public IPv4 address"),
        ({}, {"ips": ("", "")}, "could not read the public IP"),
        ({"service-quotas get-service-quota L-1216C47A": {"out": "16.0"}}, {}, "quota is 16.0"),
        ({"service-quotas get-service-quota L-1216C47A": {"out": "32.0"}}, {}, "below the 64 that m7i.16xlarge needs"),
        (
            {f"ec2 describe-images {AMI.image_id}": {"out": expected_identity().replace("20261004", "20261023")}},
            {},
            "not the pinned image",
        ),
        ({"ec2 describe-instances live ": {"out": "i-0existing"}}, {}, "already exists"),
        ({"ec2 describe-vpcs": {"out": "None"}}, {}, "no default VPC"),
        ({"ec2 describe-security-groups lookup": {"out": "sg-0old"}}, {}, "already exists"),
        (
            {f"ec2 describe-security-groups ingress {GROUP}": {"out": "tcp\t22\t22\t0.0.0.0/0\t0\t0\t0"}},
            {},
            "want only 22/tcp",
        ),
        (
            {f"ec2 describe-security-groups ingress {GROUP}": {"out": f"tcp\t22\t22\t{MY_IP}/32\t1\t0\t0"}},
            {},
            "want only 22/tcp",
        ),
        ({f"ec2 describe-security-groups egress {GROUP}": {"out": "-1\tNone\tNone\t0.0.0.0/0\t"}}, {}, "allows egress"),
        ({"ec2 describe-key-pairs": {"out": "receipts-eval-x"}}, {}, "already exists in AWS"),
        ({"ec2 create-key-pair": {"out": "no key here"}}, {}, "returned no private key"),
    ],
)
def test_plant_a_launch_that_would_break_a_rule_stops_before_any_instance_exists(
    tmp_path: Path, plant: dict[str, Any], env: dict[str, Any], match: str
) -> None:
    p, calls = run(LAUNCH, tmp_path, ["launch"], launch_scenario() | plant, **env)
    assert p.returncode != 0 and match in p.stderr, p.stderr
    assert "ec2 run-instances" not in ops(calls)


@pytest.mark.parametrize("bad", ["unset", "missing", "open", "inside"])
def test_plant_a_key_folder_that_is_not_private_stops_before_any_aws_call(tmp_path: Path, bad: str) -> None:
    script = LAUNCH
    keys = tmp_path / "keys"
    if bad == "inside":  # a copy of the scripts in a scratch repository, with the key folder inside it
        repo_vm = tmp_path / "repo" / "eval" / "vm"
        repo_vm.mkdir(parents=True)
        for name in ("launch.sh", "launch.env"):
            (repo_vm / name).write_text((ROOT / "eval" / "vm" / name).read_text())
        (repo_vm / "cloud-init.yaml.gz").write_bytes(b"user data")
        script = repo_vm / "launch.sh"
        keys = tmp_path / "repo" / "keys"
    elif bad == "missing":
        keys = tmp_path / "nope"
    if bad in ("open", "inside"):
        keys.mkdir()
        keys.chmod(0o755 if bad == "open" else 0o700)
    env = env_for(tmp_path, launch_scenario(), keys=keys if bad != "unset" else None)
    if bad == "unset":
        env.pop("RECEIPTS_KEY_DIR")
    log = Path(env["FAKE_LOG"])
    log.write_text("")
    p = subprocess.run(
        ["bash", str(script), "launch"], env=env, capture_output=True, text=True, timeout=60, check=False
    )
    calls = [json.loads(line) for line in log.read_text().splitlines()]
    why = {"unset": "set RECEIPTS_KEY_DIR", "missing": "is not a folder", "open": "has mode 755",
           "inside": "inside the repository"}[bad]  # fmt: skip
    assert p.returncode != 0 and why in p.stderr, p.stderr
    assert not any(c[0] == "aws" for c in calls)


def test_plant_no_profile_means_no_aws_call_at_all(tmp_path: Path) -> None:
    p, calls = run(LAUNCH, tmp_path, ["launch"], launch_scenario(), profile=False)
    assert p.returncode != 0 and "AWS_PROFILE" in p.stderr and calls == []


def seeded_state(tmp_path: Path, *, gpu: bool = False) -> Path:
    state = tmp_path / "state"
    state.mkdir(exist_ok=True)
    values = {
        "instance-id": INSTANCE,
        "security-group-id": GROUP,
        "key-name": "receipts-eval-20261028T120000Z",
        "key-file": str(tmp_path / "keys" / "receipts-eval-20261028T120000Z.pem"),
        "ssh-from": MY_IP,
        "instance-type": "m7i.8xlarge",
        "known_hosts": f"receipts-eval {HOST_KEY}",
    }
    if gpu:
        values |= {
            "gpu-instance-id": GPU_INSTANCE,
            "gpu-security-group-id": GPU_GROUP,
            "gpu-private-ip": "10.0.0.9",
            "gpu-image-id": "ami-0aaaaaaaaaaaaaaaa",
        }
    for k, v in values.items():
        (state / k).write_text(v + "\n")
    return state


def started(key: str = "receipts-eval-20261028T120000Z", kind: str = "m7i.8xlarge") -> dict[str, Any]:
    s = happy()
    s[f"ec2 describe-instances verify {INSTANCE}"] = verify_answer(key, kind)
    return s


def test_plant_an_instance_that_came_up_with_a_role_is_stopped_and_reported(tmp_path: Path) -> None:
    seeded_state(tmp_path)
    s = started()
    s[f"ec2 describe-instances verify {INSTANCE}"] = verify_answer(
        "receipts-eval-20261028T120000Z", profile="arn:aws:iam::1:instance-profile/x"
    )
    p, calls = run(LAUNCH, tmp_path, ["start"], s)
    assert p.returncode != 0 and "is not what was asked for" in p.stderr
    assert "ec2 stop-instances" in ops(calls)


def test_stop_then_start_moves_ssh_to_the_current_ip_and_runs_vm_check_over_ssh(tmp_path: Path) -> None:
    state = seeded_state(tmp_path)
    p, calls = run(LAUNCH, tmp_path, ["stop"], started())
    assert p.returncode == 0 and ops(calls) == ["ec2 stop-instances", "ec2 wait"]
    assert "volume bills until terminate" in p.stdout
    new_ip = "73.12.34.200"
    s = started() | {f"ec2 describe-security-groups ingress {GROUP}": eval_ingress(new_ip)}
    p, calls = run(LAUNCH, tmp_path, ["start"], s, ips=(new_ip, new_ip))
    assert p.returncode == 0, p.stderr
    order = ops(calls)
    assert order[:2] == ["curl", "curl"]
    assert order.index("ec2 authorize-security-group-ingress") < order.index("ec2 revoke-security-group-ingress")
    assert order.index("ec2 revoke-security-group-ingress") < order.index("ec2 start-instances")
    revoked = next(c for c in calls if c[3:5] == ["ec2", "revoke-security-group-ingress"])
    assert f"{MY_IP}/32" in revoked[-1] and (state / "ssh-from").read_text().strip() == new_ip
    ssh = next(c for c in calls if c[0] == "ssh")
    assert "receipts-eval sudo bash /opt/receipts/vm-check.sh" in ssh[1] and "vm ok" in p.stdout


def test_plant_a_failed_vm_check_after_start_is_an_error(tmp_path: Path) -> None:
    seeded_state(tmp_path)
    p, _ = run(LAUNCH, tmp_path, ["start"], started(), ssh_code=1, ssh_out="runsc did not run gVisor")
    assert p.returncode != 0 and "vm-check failed" in p.stderr
    p, _ = run(LAUNCH, tmp_path, ["start"], started(), ssh_out="something else")
    assert p.returncode != 0 and "vm-check on" in p.stderr


def test_resize_takes_only_the_two_sizes_and_only_a_stopped_vm(tmp_path: Path) -> None:
    state = seeded_state(tmp_path)
    p, calls = run(LAUNCH, tmp_path, ["resize", "m7i.16xlarge"], started())
    assert p.returncode == 0, p.stderr
    modify = next(c for c in calls if c[3:5] == ["ec2", "modify-instance-attribute"])
    assert '{"Value": "m7i.16xlarge"}' in modify and (state / "instance-type").read_text().strip() == "m7i.16xlarge"
    assert "resizing to m7i.16xlarge, which needs 64 vCPUs" in p.stdout
    # REG-568: after R24 and the labels, back to the pilot's size, which needs only 32
    s = started() | {"service-quotas get-service-quota L-1216C47A": {"out": "32.0"}}
    p, calls = run(LAUNCH, tmp_path, ["resize", "m7i.8xlarge"], s)
    assert p.returncode == 0, p.stderr
    modify = next(c for c in calls if c[3:5] == ["ec2", "modify-instance-attribute"])
    assert '{"Value": "m7i.8xlarge"}' in modify and (state / "instance-type").read_text().strip() == "m7i.8xlarge"
    assert "resizing to m7i.8xlarge, which needs 32 vCPUs" in p.stdout
    for args, scenario, match in (
        (["resize", "m7i.48xlarge"], started(), "not 'm7i.48xlarge'"),
        (["resize", "m7i.16xlarge"], started() | {f"ec2 describe-instances state {INSTANCE}": {"out": "running"}},
         "stop it first"),
        (["resize", "m7i.16xlarge"], started() | {"service-quotas get-service-quota L-1216C47A": {"out": "32.0"}},
         "needs 64"),
    ):  # fmt: skip
        p, calls = run(LAUNCH, tmp_path, args, scenario)
        assert p.returncode != 0 and match in p.stderr, p.stderr
        assert "ec2 modify-instance-attribute" not in ops(calls)


def test_terminate_needs_the_id_typed_out_and_the_glimmer_vm_gone(tmp_path: Path) -> None:
    seeded_state(tmp_path, gpu=True)
    p, calls = run(LAUNCH, tmp_path, ["terminate", "--yes", INSTANCE], started())
    assert p.returncode != 0 and "gpu.sh terminate" in p.stderr and calls == []
    for f in ("gpu-instance-id", "gpu-security-group-id", "gpu-private-ip", "gpu-image-id"):
        (tmp_path / "state" / f).unlink()
    for args in (["terminate"], ["terminate", "--yes"], ["terminate", "--yes", "i-0other"]):
        p, calls = run(LAUNCH, tmp_path, args, started())
        assert p.returncode != 0 and calls == []
    p, calls = run(LAUNCH, tmp_path, ["terminate", "--yes", INSTANCE], started())
    assert p.returncode == 0, p.stderr
    assert ops(calls) == [
        "ec2 modify-instance-attribute",
        "ec2 terminate-instances",
        "ec2 wait",
        "ec2 delete-security-group",
        "ec2 delete-key-pair",
    ]
    assert "is still on disk" in p.stdout


def test_plant_no_host_keys_in_the_console_never_falls_back_to_trust_on_first_use(tmp_path: Path) -> None:
    seeded_state(tmp_path)
    s = started() | {"ec2 get-console-output": {"out": "booting\n"}}
    p, _ = run(LAUNCH, tmp_path, ["host-keys"], s)
    assert p.returncode != 0 and "no host keys" in p.stderr


# ---------------------------------------------------------------- the Glimmer VM


GPU_AMI_NAME = f"{GPU_AMI_NAME_PREFIX} 20261020"
PINNED_AMI = "ami-0bbbbbbbbbbbbbbbb"


def ami_answer(ami: str = "ami-0aaaaaaaaaaaaaaaa", name: str = GPU_AMI_NAME, state: str = "available",
               arch: str = "x86_64") -> dict[str, str]:  # fmt: skip
    return {"out": f"{ami}\t2026-10-20T00:00:00.000Z\t898082745236\t{name}\t{state}\t{arch}"}


def gpu_scenario() -> dict[str, Any]:
    s = started()
    s["ec2 describe-images by-name"] = ami_answer()
    s["ec2 create-security-group"] = {"out": GPU_GROUP}
    s["ec2 run-instances"] = {"out": GPU_INSTANCE}
    s[f"ec2 describe-instances verify {GPU_INSTANCE}"] = {
        "out": f"{GPU_INSTANCE_TYPE}\tami-0aaaaaaaaaaaaaaaa\trequired\t1\tNone\treceipts-eval-20261028T120000Z"
    }
    return s


def test_gpu_launch_is_reachable_from_the_eval_vm_only(tmp_path: Path) -> None:
    state = seeded_state(tmp_path)
    p, calls = run(GPU, tmp_path, ["launch"], gpu_scenario())
    assert p.returncode == 0, p.stderr
    order = ops(calls)
    images = next(c for c in calls if c[3:5] == ["ec2", "describe-images"])
    assert "amazon" in images and "--image-ids" not in images  # not pinned: the newest by its exact name
    assert f"Name=name,Values={GPU_AMI_NAME_PREFIX} ????????" in images
    assert (
        f"image ami-0aaaaaaaaaaaaaaaa (not pinned: the newest by name): {GPU_AMI_NAME}, created 2026-10-20" in p.stdout
    )
    ingress = [c for c in calls if c[3:5] == ["ec2", "authorize-security-group-ingress"]]
    assert len(ingress) == 2
    for c in ingress:
        assert c[c.index("--group-id") + 1] == GPU_GROUP
        perm = json.loads(c[c.index("--ip-permissions") + 1])[0]
        assert "IpRanges" not in perm and perm["UserIdGroupPairs"][0]["GroupId"] == GROUP  # no public port
    egress = [c for c in calls if c[3:5] == ["ec2", "authorize-security-group-egress"]]
    assert {c[c.index("--group-id") + 1] for c in egress} == {GPU_GROUP, GROUP}
    launch = next(c for c in calls if c[3:5] == ["ec2", "run-instances"])
    assert launch[launch.index("--image-id") + 1] == "ami-0aaaaaaaaaaaaaaaa"
    assert launch[launch.index("--key-name") + 1] == "receipts-eval-20261028T120000Z"
    assert f"fileb://{ROOT}/eval/vm/gpu-cloud-init.yaml" in launch
    assert order.index("ec2 authorize-security-group-ingress") < order.index("ec2 run-instances")
    assert (state / "gpu-private-ip").read_text().strip() == "10.0.0.9"
    assert "receipts-glimmer ssh-ed25519" in (state / "known_hosts").read_text()
    assert not any(c[3] == "ssm" for c in calls if c[0] == "aws")
    p, _ = run(GPU, tmp_path, ["url"], gpu_scenario())
    assert p.stdout.strip() == f"RECEIPTS_GLIMMER_URL=http://10.0.0.9:{VLLM_PORT}/v1"


@pytest.mark.parametrize(
    ("plant", "match"),
    [
        ({"service-quotas get-service-quota L-DB2E81BA": {"out": "0.0"}}, "Modal"),
        ({"ec2 describe-images by-name": {"out": "None\tNone\tNone\tNone\tNone\tNone"}}, "no available image"),
        ({"ec2 describe-images by-name": ami_answer(name=f"{GPU_AMI_NAME_PREFIX} 20261020 custom")}, "is named"),
        (
            {
                "ec2 describe-images by-name": ami_answer(
                    name="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261020"
                )
            },
            "is named",
        ),
        (
            {f"ec2 describe-security-groups ingress {GPU_GROUP}": {"out": "tcp\t8000\t8000\t0.0.0.0/0\t0\t0\t"}},
            "want 22/tcp and 8000/tcp from",
        ),
        ({f"ec2 describe-security-groups egress {GPU_GROUP}": {"out": "-1\tNone\tNone\t0.0.0.0/0"}}, "allows egress"),
    ],
)
def test_plant_a_gpu_launch_that_would_break_a_rule_stops_before_the_instance(
    tmp_path: Path, plant: dict[str, Any], match: str
) -> None:
    seeded_state(tmp_path)
    p, calls = run(GPU, tmp_path, ["launch"], gpu_scenario() | plant)
    assert p.returncode != 0 and match in p.stderr, p.stderr
    assert "ec2 run-instances" not in ops(calls)


def test_plant_a_gpu_launch_without_the_eval_vm_stops_first(tmp_path: Path) -> None:
    p, calls = run(GPU, tmp_path, ["launch"], gpu_scenario())
    assert p.returncode != 0 and "launch the eval VM first" in p.stderr and calls == []


def test_seal_needs_a_ready_server_and_leaves_no_egress(tmp_path: Path) -> None:
    seeded_state(tmp_path, gpu=True)
    sealed = gpu_scenario() | {f"ec2 describe-security-groups egress {GPU_GROUP}": {"out": ""}}
    p, calls = run(GPU, tmp_path, ["seal"], sealed, ssh_code=1)
    assert p.returncode != 0 and "READY" in p.stderr and "ec2 revoke-security-group-egress" not in ops(calls)
    p, calls = run(GPU, tmp_path, ["seal"], sealed)
    assert p.returncode == 0, p.stderr
    revoke = next(c for c in calls if c[3:5] == ["ec2", "revoke-security-group-egress"])
    assert revoke[revoke.index("--group-id") + 1] == GPU_GROUP and "sealed" in p.stdout
    p, _ = run(GPU, tmp_path, ["seal"], gpu_scenario())  # egress still there after the revoke: refused
    assert p.returncode != 0 and "allows egress" in p.stderr


def test_gpu_terminate_takes_the_eval_vms_rules_with_it(tmp_path: Path) -> None:
    state = seeded_state(tmp_path, gpu=True)
    p, calls = run(GPU, tmp_path, ["terminate", "--yes", "i-0other"], gpu_scenario())
    assert p.returncode != 0 and calls == []
    p, calls = run(GPU, tmp_path, ["terminate", "--yes", GPU_INSTANCE], gpu_scenario())
    assert p.returncode == 0, p.stderr
    revokes = [c for c in calls if c[3:5] == ["ec2", "revoke-security-group-egress"]]
    assert len(revokes) == 2 and all(c[c.index("--group-id") + 1] == GROUP for c in revokes)
    assert "ec2 delete-security-group" in ops(calls) and not (state / "gpu-instance-id").exists()
    assert "receipts-glimmer" not in (state / "known_hosts").read_text()


def test_the_eval_vms_group_check_accepts_the_glimmer_rules_only_while_the_glimmer_vm_exists(tmp_path: Path) -> None:
    seeded_state(tmp_path, gpu=True)
    s = started() | {f"ec2 describe-security-groups egress {GROUP}": eval_egress(gpu=True)}
    p, _ = run(LAUNCH, tmp_path, ["start"], s)
    assert p.returncode == 0, p.stderr
    for f in ("gpu-instance-id", "gpu-security-group-id", "gpu-private-ip", "gpu-image-id"):
        (tmp_path / "state" / f).unlink()
    p, _ = run(LAUNCH, tmp_path, ["start"], s)
    assert p.returncode != 0 and "allows egress" in p.stderr


# ---------------------------------------------------------------- every non-public address is refused (LS8, LS9)

REFUSED = (
    "0.1.2.3", "10.1.2.3", "100.64.0.1", "100.127.255.254", "127.0.0.1", "169.254.169.254", "172.16.0.1",
    "172.31.255.1", "192.0.0.8", "192.0.2.7", "192.88.99.1", "192.168.1.10", "198.18.0.1", "198.19.255.1",
    "198.51.100.7", "203.0.113.7", "224.0.0.251", "239.1.2.3", "240.0.0.1", "255.255.255.255", "010.1.2.3",
    "1.2.3.256", "1.2.3", "::1", "1.2.3.4.5", "a.b.c.d",
)  # fmt: skip
ACCEPTED = ("1.1.1.1", "100.63.255.1", "100.128.0.1", "172.15.0.1", "172.32.0.1", "192.0.1.1", "192.169.0.1",
            "198.17.0.1", "198.20.0.1", "198.51.99.1", "203.0.112.1", "223.255.255.1")  # fmt: skip


@pytest.mark.parametrize("ip", REFUSED)
def test_plant_a_non_public_address_is_never_the_founders_32(tmp_path: Path, ip: str) -> None:
    """LS8 and LS9: the shared 100.64/10 block and multicast were accepted as the founder's /32. Every block of IANA's
    special-purpose registry is refused, before any AWS call."""
    p, calls = run(LAUNCH, tmp_path, ["launch"], launch_scenario(), ips=(ip, ip))
    assert p.returncode != 0 and "not a public IPv4 address" in p.stderr, (ip, p.stderr)
    assert not any(c[0] == "aws" for c in calls)


@pytest.mark.parametrize("ip", ACCEPTED)
def test_a_public_address_next_to_a_refused_block_is_accepted(tmp_path: Path, ip: str) -> None:
    s = launch_scenario() | {f"ec2 describe-security-groups ingress {GROUP}": eval_ingress(ip)}
    p, calls = run(LAUNCH, tmp_path, ["launch"], s, ips=(ip, ip))
    assert p.returncode == 0, (ip, p.stderr)
    ingress = next(c for c in calls if c[3:5] == ["ec2", "authorize-security-group-ingress"])
    assert f'"CidrIp":"{ip}/32"' in ingress[ingress.index("--ip-permissions") + 1]


# ---------------------------------------------------------------- the Glimmer VM's AMI, pinned (the integrator's call)


@pytest.mark.parametrize("how", ["flag", "environment"])
def test_gpu_launch_takes_the_pinned_ami_and_prints_its_id_name_and_date(tmp_path: Path, how: str) -> None:
    state = seeded_state(tmp_path)
    s = gpu_scenario() | {
        f"ec2 describe-images {PINNED_AMI}": ami_answer(PINNED_AMI),
        f"ec2 describe-instances verify {GPU_INSTANCE}": {
            "out": f"{GPU_INSTANCE_TYPE}\t{PINNED_AMI}\trequired\t1\tNone\treceipts-eval-20261028T120000Z"
        },
    }
    args, extra = (
        (["launch", "--ami", PINNED_AMI], None) if how == "flag" else (["launch"], {"RECEIPTS_GPU_AMI": PINNED_AMI})
    )
    p, calls = run(GPU, tmp_path, args, s, extra=extra)
    assert p.returncode == 0, p.stderr
    [images] = [c for c in calls if c[3:5] == ["ec2", "describe-images"]]
    assert images[images.index("--image-ids") + 1] == PINNED_AMI and "--filters" not in images
    assert f"image {PINNED_AMI} (pinned): {GPU_AMI_NAME}, created 2026-10-20T00:00:00.000Z" in p.stdout
    launch = next(c for c in calls if c[3:5] == ["ec2", "run-instances"])
    assert launch[launch.index("--image-id") + 1] == PINNED_AMI
    assert (state / "gpu-image-id").read_text().strip() == PINNED_AMI


@pytest.mark.parametrize(
    ("answer", "match"),
    [
        (
            ami_answer(PINNED_AMI, name="Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 22.04) 20261020"),
            "is named",
        ),
        (ami_answer(PINNED_AMI, name=f"{GPU_AMI_NAME_PREFIX} 20261020-patched"), "is named"),
        (ami_answer(PINNED_AMI, name="Someone's GPU image"), "is named"),
        (ami_answer("ami-0ccccccccccccccccc"[:21]), "for the pinned"),
        (ami_answer(PINNED_AMI, state="deprecated"), "want available on x86_64"),
        (ami_answer(PINNED_AMI, arch="arm64"), "want available on x86_64"),
        ({"out": "None\tNone\tNone\tNone\tNone\tNone"}, "no available image ami-0bbbbbbbbbbbbbbbb"),
    ],
)
def test_plant_a_pinned_ami_that_is_not_the_expected_image_is_refused(
    tmp_path: Path, answer: dict[str, str], match: str
) -> None:
    """G4: any AMI name passed. A pinned id must come back as itself, Amazon's, available, x86_64 and named exactly
    '<prefix> <YYYYMMDD>'; anything else stops the launch before an instance exists."""
    seeded_state(tmp_path)
    s = gpu_scenario() | {f"ec2 describe-images {PINNED_AMI}": answer}
    p, calls = run(GPU, tmp_path, ["launch", "--ami", PINNED_AMI], s)
    assert p.returncode != 0 and match in p.stderr, p.stderr
    assert "ec2 run-instances" not in ops(calls) and "ec2 create-security-group" not in ops(calls)


@pytest.mark.parametrize(
    "args", [["launch", "--ami", "ami-xyz"], ["launch", "--ami"], ["launch", "--image", "ami-0bbbbbbbbbbbbbbbb"]]
)
def test_plant_a_malformed_ami_pin_is_refused_before_any_image_lookup(tmp_path: Path, args: list[str]) -> None:
    seeded_state(tmp_path)
    p, calls = run(GPU, tmp_path, args, gpu_scenario())
    assert p.returncode != 0 and ("not an AMI id" in p.stderr or "usage" in p.stderr), p.stderr
    assert "ec2 describe-images" not in ops(calls) and "ec2 run-instances" not in ops(calls)


def test_gpu_start_verifies_the_instance_before_it_waits_for_the_server(tmp_path: Path) -> None:
    """G6: `gpu.sh start` never verified the instance in a test. A VM that comes back as anything but what was launched
    (type, image, IMDSv2, no role, the key pair) is stopped and reported, and nothing reaches it."""
    seeded_state(tmp_path, gpu=True)
    served = '{"object":"list","data":[{"id":"muse-glimmer-30b"}]}'
    p, calls = run(GPU, tmp_path, ["start"], gpu_scenario(), ssh_out=served)
    assert p.returncode == 0 and "vLLM serves muse-glimmer-30b" in p.stdout, p.stderr
    order = ops(calls)
    assert order.index("ec2 describe-instances") < order.index("ssh")  # verified before anything talks to it
    for wrong in (
        "g7e.4xlarge\tami-0aaaaaaaaaaaaaaaa\trequired\t1\tNone\treceipts-eval-20261028T120000Z",
        f"{GPU_INSTANCE_TYPE}\tami-0dddddddddddddddd\trequired\t1\tNone\treceipts-eval-20261028T120000Z",
        f"{GPU_INSTANCE_TYPE}\tami-0aaaaaaaaaaaaaaaa\toptional\t1\tNone\treceipts-eval-20261028T120000Z",
        f"{GPU_INSTANCE_TYPE}\tami-0aaaaaaaaaaaaaaaa\trequired\t1\tarn:aws:iam::1:instance-profile/x\treceipts-eval-20261028T120000Z",
    ):
        s = gpu_scenario() | {f"ec2 describe-instances verify {GPU_INSTANCE}": {"out": wrong}}
        p, calls = run(GPU, tmp_path, ["start"], s, ssh_out=served)
        assert p.returncode != 0 and "is not what was asked for" in p.stderr, (wrong, p.stderr)
        assert "ec2 stop-instances" in ops(calls) and "ssh" not in ops(calls)


# ---------------------------------------------------------------- the JSON inputs


def test_run_instances_carries_every_rule_and_leaves_out_what_must_not_be_there() -> None:
    r = run_instances_input()
    assert {"IamInstanceProfile", "UserData", "SecurityGroupIds", "NetworkInterfaces", "KeyName"}.isdisjoint(r)
    assert (r["ImageId"], r["InstanceType"]) == (AMI.image_id, "m7i.16xlarge")  # REG-568: R24 and the labels first
    assert r["MetadataOptions"]["HttpTokens"] == "required" and r["MetadataOptions"]["HttpPutResponseHopLimit"] == 1
    ebs = r["BlockDeviceMappings"][0]["Ebs"]
    assert r["BlockDeviceMappings"][0]["DeviceName"] == AMI.root_device_name
    assert (ebs["VolumeSize"], ebs["VolumeType"], ebs["Iops"], ebs["Throughput"]) == (500, "gp3", 6000, 250)
    assert ebs["Encrypted"] and ebs["DeleteOnTermination"]
    assert r["DisableApiTermination"] is True and r["InstanceInitiatedShutdownBehavior"] == "stop"
    for spec in r["TagSpecifications"]:
        assert {(t["Key"], t["Value"]) for t in spec["Tags"]} == set(TAGS)
    gpu = json.loads(files()["gpu-run-instances.json"])
    assert {"IamInstanceProfile", "UserData", "SecurityGroupIds", "KeyName", "ImageId"}.isdisjoint(gpu)
    assert gpu["InstanceType"] == "g7e.2xlarge" and gpu["MetadataOptions"]["HttpTokens"] == "required"


def test_egress_is_tcp_443_and_80_only_and_the_tags_carry_nothing_private() -> None:
    assert [(e["IpProtocol"], e["FromPort"], e["ToPort"]) for e in egress_rules()] == [
        ("tcp", p, p) for p in EGRESS_PORTS
    ]
    assert EGRESS_PORTS == (443, 80)
    assert [(e["FromPort"], e["ToPort"]) for e in json.loads(files()["gpu-egress.json"])] == [(443, 443)]
    for key, value in TAGS:
        assert "@" not in value and "FAL-" not in value and "key" not in key.lower()


def test_the_ami_pin_is_canonicals_and_launch_env_reads_back_in_bash(tmp_path: Path) -> None:
    assert AMI.owner_id == "099720109477" and AMI.serial in AMI.name and AMI.region == "us-east-1"
    env = tmp_path / "launch.env"
    env.write_text(files()["launch.env"])
    out = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{env}"; echo "$AMI_ID|$AMI_IDENTITY|$LAUNCH_INSTANCE_TYPE $LAUNCH_VCPU|$PILOT_INSTANCE_TYPE '
            '$PILOT_VCPU|$FINAL_INSTANCE_TYPE $FINAL_VCPU|$IP_SOURCES"',
        ],
        capture_output=True,
        text=True,
        check=True,
        timeout=30,
    ).stdout.strip()
    assert out == (
        f"{AMI.image_id}|{expected_identity().replace(chr(9), '|')}|m7i.16xlarge 64|m7i.8xlarge 32|m7i.16xlarge 64|"
        "https://checkip.amazonaws.com https://api.ipify.org"
    )


# AWS's rule for a security group's description and each rule's description, under 256 characters from this set.
# CreateSecurityGroup refused the apostrophe in "founder's" on 2026-10-28, before anything was created.
AWS_DESCRIPTION = re.compile(r"[a-zA-Z0-9. _\-:/()#,@\[\]+=&;{}!$*]{0,255}")


def test_every_security_group_and_rule_description_is_one_aws_accepts() -> None:
    found: list[str] = []
    for script in (LAUNCH, GPU):
        text = script.read_text()
        found += re.findall(r'--description "([^"]*)"', text)
        found += re.findall(r'"Description":"([^"%]*)"', text)
        found += re.findall(r'group_pair \S+ \S+ "([^"]*)"', text)
    for name in ("egress.json", "gpu-egress.json"):
        found += re.findall(r'"Description": "([^"]*)"', (ROOT / "eval" / "vm" / name).read_text())
    assert len(found) == 10
    assert [d for d in found if not AWS_DESCRIPTION.fullmatch(d)] == []
