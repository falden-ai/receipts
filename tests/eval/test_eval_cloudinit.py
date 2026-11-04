"""The user data pulls by digest only, keeps every guard whole, proves gVisor by its kernel banner, removes the SSM
agent, carries nothing secret-shaped and fits EC2's limit. The metadata guard, the SSM guard and vm-check.sh run here
against fake curl, snap, pgrep and docker commands, with one plant per check. The GPU VM's user data too."""

from __future__ import annotations

import gzip
import hashlib
import json
import os
import stat
import subprocess
from pathlib import Path

import pytest

from receipts.contracts import RUN_PHASE_LIMITS
from receipts.eval.cloudinit import (
    GPU_GUARDS,
    USER_DATA_LIMIT,
    UserDataError,
    check,
    check_gpu,
    guards,
    image_lines,
    render,
    render_gpu_text,
    render_text,
    scripts,
)
from receipts.eval.gpuvm import DRAFTER, MODEL, SERVED_NAME, VLLM_IMAGE, plan, serve_args
from receipts.eval.images import ImageRef


def image(n: int) -> ImageRef:
    iid = f"django__django-{10000 + n}"
    return ImageRef(
        instance_id=iid,
        image=f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}",
        digest="sha256:" + hashlib.sha256(str(n).encode()).hexdigest(),
        media_type="application/vnd.docker.distribution.manifest.v2+json",
    )


IMAGES = [image(n) for n in range(3)]


def test_a_render_passes_its_own_check_and_lists_every_digest() -> None:
    text = render_text(IMAGES)
    check(text)
    for i in IMAGES:
        assert f"{i.instance_id} {i.digest}" in text
    data, gz = render(IMAGES)
    assert not gz and data == text.encode()


def test_the_pool_sized_list_ships_gzipped_under_the_limit_and_unpacks_to_the_same_text() -> None:
    many = [image(n) for n in range(251)]
    data, gz = render(many)
    assert gz and len(data) <= USER_DATA_LIMIT
    assert gzip.decompress(data).decode() == render_text(many)


def test_plant_a_list_too_large_even_gzipped_is_refused() -> None:
    with pytest.raises(UserDataError, match="over EC2's"):
        render([image(n) for n in range(400)])


@pytest.mark.parametrize(
    "secret",
    [
        "export FALDEN_META_API_KEY=whatever",
        "AKIA" + "ABCDEFGHIJKLMNOP",
        "aws_secret_access_key = x",
        "-----BEGIN " + "OPENSSH PRIVATE KEY-----",
        "gh" + "p_" + "a" * 36,
        "hf_" + "b" * 34,
    ],
)
def test_plant_secret_shaped_text_is_refused(secret: str) -> None:
    text = render_text(IMAGES).replace("runcmd:\n", f"# {secret}\nruncmd:\n")
    with pytest.raises(UserDataError, match="secret-shaped"):
        check(text)


def test_plant_a_pull_by_tag_is_refused() -> None:
    for planted in ("docker pull ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1:latest", "x:latest"):
        text = render_text(IMAGES).replace("runcmd:\n", f"# {planted}\nruncmd:\n")
        with pytest.raises(UserDataError, match="by tag"):
            check(text)
    text = render_text(IMAGES).replace("runcmd:\n", "# ghcr.io/epoch-research/x:v1\nruncmd:\n")
    with pytest.raises(UserDataError, match="by tag"):
        check(text)


def test_plant_an_image_line_without_a_digest_is_refused() -> None:
    text = render_text(IMAGES)
    first = f"{IMAGES[0].instance_id} {IMAGES[0].digest}"
    with pytest.raises(UserDataError, match="is not '<instance_id> sha256"):
        check(text.replace(first, f"{IMAGES[0].instance_id} latest"))


@pytest.mark.parametrize("guard", guards())
def test_plant_a_script_that_lost_a_guard_is_refused(guard: str) -> None:
    text = render_text(IMAGES)
    assert guard in text
    with pytest.raises(UserDataError, match="lost or weakened a guard"):
        check(text.replace(guard, "true"))


@pytest.mark.parametrize("guard", guards())
def test_plant_a_guard_whose_failure_was_neutered_is_refused(guard: str) -> None:
    """The reviewer's survivor: `|| fail` turned into `|| echo` kept every substring the old check looked for."""
    weakened = guard.replace("|| fail", "|| echo").replace("then fail", "then echo").replace("exit 1;", "true;")
    assert weakened != guard
    with pytest.raises(UserDataError, match="lost or weakened a guard"):
        check(render_text(IMAGES).replace(guard, weakened))


def test_the_probe_runs_dmesg_with_the_run_phase_limits_and_never_trusts_dockers_runtime_field() -> None:
    probe = scripts()["probe.sh"]
    lim = RUN_PHASE_LIMITS
    flags = (
        f"--network={lim.network}",
        f"--cap-drop={lim.cap_drop}",
        f"--pids-limit={lim.pids_limit}",
        f"--memory={lim.memory}",
        f"--shm-size={lim.shm_size}",
        "--security-opt=no-new-privileges",
    )
    for flag in flags:
        assert flag in probe
    assert probe.rstrip().endswith('"$1" dmesg\n}')
    text = render_text(IMAGES)
    assert "HostConfig.Runtime}}' " not in text and "Starting gVisor" in text
    assert "usermod -aG docker ubuntu" in text  # the runsc check runs as ubuntu
    assert "snap remove --purge amazon-ssm-agent" in text  # REG-562: no Systems Manager


def test_duplicate_or_empty_image_lists_are_refused() -> None:
    with pytest.raises(UserDataError, match="twice"):
        image_lines([IMAGES[0], IMAGES[0]])
    with pytest.raises(UserDataError, match="no images"):
        image_lines([])


# ---------------------------------------------------------------- the metadata guard, run against a fake curl

FAKE_CURL = """#!/usr/bin/env bash
# Answers like the instance metadata service would: the codes come from the environment.
url="${@: -1}"
case "$url" in
  */api/token) [ "$TOKEN_OK" = 1 ] && { echo "tok"; exit 0; } || { echo "curl: (7) refused" >&2; exit 7; } ;;
  */iam/security-credentials/) echo -n "$ROLE_CODE" ;;
  */meta-data/) echo -n "$V1_CODE" ;;
  *) echo "unexpected $url" >&2; exit 99 ;;
esac
"""


def run_imds_check(tmp_path: Path, *, token_ok: bool = True, role: str = "404", v1: str = "401") -> int:
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    curl = bin_dir / "curl"
    curl.write_text(FAKE_CURL)
    curl.chmod(curl.stat().st_mode | stat.S_IEXEC)
    script = tmp_path / "check-imds.sh"
    script.write_text(scripts()["check-imds.sh"])
    env = {
        "PATH": f"{bin_dir}{os.pathsep}/usr/bin:/bin",
        "TOKEN_OK": "1" if token_ok else "0",
        "ROLE_CODE": role,
        "V1_CODE": v1,
    }
    return subprocess.run(["bash", str(script)], env=env, capture_output=True, check=False, timeout=30).returncode


def test_the_metadata_guard_passes_only_with_no_role_and_imdsv1_off(tmp_path: Path) -> None:
    assert run_imds_check(tmp_path) == 0


@pytest.mark.parametrize(
    "plant",
    [
        {"role": "200"},  # an instance profile is attached
        {"v1": "200"},  # IMDSv1 answers
        {"token_ok": False},  # no IMDSv2 token at all
    ],
)
def test_plant_a_vm_with_a_role_or_imdsv1_fails_the_metadata_guard(tmp_path: Path, plant: dict[str, object]) -> None:
    assert run_imds_check(tmp_path, **plant) != 0  # type: ignore[arg-type]


# ---------------------------------------------------------------- the SSM guard and vm-check.sh, against fakes

FAKE_SNAP = """#!/usr/bin/env bash
[ "$1 $2" = "list amazon-ssm-agent" ] && [ "$SSM_SNAP" = 1 ] && exit 0
exit 1
"""
FAKE_PGREP = """#!/usr/bin/env bash
[ "$SSM_PROC" = 1 ] && exit 0
exit 1
"""
FAKE_DOCKER = """#!/usr/bin/env bash
if [ "$1" = run ]; then
  for a in "$@"; do case "$a" in --runtime=*) rt="${a#--runtime=}" ;; esac; done
  want=0
  [ "$rt" = runsc ] && want="$RUNSC_GVISOR"
  [ "$rt" = runc ] && want="$RUNC_GVISOR"
  if [ "$want" = 1 ]; then echo "[    0.000000] Starting gVisor..."; exit 0; fi
  if [ "$want" = host ]; then echo "[    0.000000] Linux version 6.8.0-1015-aws"; exit 0; fi
  echo "dmesg: read kernel buffer failed: Operation not permitted" >&2; exit 1
fi
if [ "$1 $2" = "image inspect" ]; then
  case "$3" in *"$MISSING_IMAGE"*) [ -n "$MISSING_IMAGE" ] && exit 1 ;; esac
  exit 0
fi
exit 99
"""


def vm_root(tmp_path: Path) -> Path:
    root = tmp_path / "receipts"
    root.mkdir()
    for name, body in scripts().items():
        (root / name).write_text(body)
    (root / "images.txt").write_text(image_lines(IMAGES))
    (root / "BOOT_OK").write_text("ok\n")
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, body in (("curl", FAKE_CURL), ("snap", FAKE_SNAP), ("pgrep", FAKE_PGREP), ("docker", FAKE_DOCKER)):
        (bin_dir / name).write_text(body)
        (bin_dir / name).chmod(0o755)
    return root


def run_vm_check(tmp_path: Path, **plant: str) -> subprocess.CompletedProcess[str]:
    root = vm_root(tmp_path)
    if plant.pop("no_boot_ok", ""):
        (root / "BOOT_OK").unlink()
    env = {
        "PATH": f"{tmp_path / 'bin'}{os.pathsep}/usr/bin:/bin",
        "RECEIPTS_VM_ROOT": str(root),
        "TOKEN_OK": "1",
        "ROLE_CODE": "404",
        "V1_CODE": "401",
        "SSM_SNAP": "0",
        "SSM_PROC": "0",
        "RUNSC_GVISOR": "1",
        "RUNC_GVISOR": "0",
        "MISSING_IMAGE": "",
    } | plant
    return subprocess.run(
        ["bash", str(root / "vm-check.sh")], env=env, capture_output=True, text=True, check=False, timeout=60
    )


def test_vm_check_passes_on_a_ready_vm(tmp_path: Path) -> None:
    p = run_vm_check(tmp_path)
    assert p.returncode == 0, p.stdout + p.stderr
    assert p.stdout.strip().endswith("vm ok: gVisor, 3 images, no instance role, no SSM agent")


@pytest.mark.parametrize(
    ("plant", "says"),
    [
        ({"no_boot_ok": "1"}, "bootstrap never finished"),
        ({"ROLE_CODE": "200"}, "instance metadata check failed"),
        ({"V1_CODE": "200"}, "instance metadata check failed"),
        ({"SSM_SNAP": "1"}, "Systems Manager agent"),
        ({"SSM_PROC": "1"}, "Systems Manager agent"),
        ({"RUNSC_GVISOR": "0"}, "runsc did not run gVisor"),
        ({"RUNSC_GVISOR": "host"}, "runsc did not run gVisor"),  # dmesg ran and exited 0, on the host kernel
        ({"RUNC_GVISOR": "1"}, "the runc probe ran on gVisor"),
        ({"MISSING_IMAGE": "django__django-10001"}, "1 pinned images are missing"),
    ],
)
def test_plant_each_vm_check_fails_on_its_broken_input(tmp_path: Path, plant: dict[str, str], says: str) -> None:
    """The re-review's blocker: none of vm-check.sh's checks had a plant."""
    p = run_vm_check(tmp_path, **plant)
    assert p.returncode != 0 and says in p.stdout, p.stdout + p.stderr
    assert "vm ok" not in p.stdout


@pytest.mark.parametrize(("snap", "proc", "code"), [("0", "0", 0), ("1", "0", 1), ("0", "1", 1)])
def test_the_ssm_guard_passes_only_with_no_agent_installed_or_running(
    tmp_path: Path, snap: str, proc: str, code: int
) -> None:
    root = vm_root(tmp_path)
    env = {"PATH": f"{tmp_path / 'bin'}{os.pathsep}/usr/bin:/bin", "SSM_SNAP": snap, "SSM_PROC": proc}
    p = subprocess.run(["bash", str(root / "check-no-ssm.sh")], env=env, capture_output=True, text=True, timeout=30)
    assert (p.returncode == 0) is (code == 0)


# ---------------------------------------------------------------- the GPU VM's user data


def test_the_gpu_user_data_serves_glimmer_offline_from_pinned_weights() -> None:
    text = render_gpu_text()
    check_gpu(text)
    assert VLLM_IMAGE.ref in text and ":v0.28.0" not in text
    assert MODEL.revision in text and DRAFTER.revision in text
    flags = (
        "--tool-call-parser muse_glimmer",
        "--enable-auto-tool-choice",
        "--dtype bfloat16",
        f"--served-model-name {SERVED_NAME}",
        "--max-num-seqs 32",
        "--speculative-config",
        '"method": "dflash"',
        '"num_speculative_tokens": 15',
    )
    for flag in flags:
        assert flag in text, flag
    assert "HF_HUB_OFFLINE=1" in text and "snap remove --purge amazon-ssm-agent" in text
    assert len(text.encode()) <= USER_DATA_LIMIT


def test_glimmer_runs_without_a_reasoning_parser_and_keeps_the_drafter_and_tool_parser() -> None:
    """The 2026-10-29 VM tests: with --reasoning-parser muse_glimmer, vLLM 0.28.0 never applied the json_schema; without
    it the schema held. The drafter and the tool parser stay."""
    args = serve_args()
    assert not any(a.startswith("--reasoning-parser") for a in args)
    assert "muse_glimmer" in args and args[args.index("--tool-call-parser") + 1] == "muse_glimmer"
    speculative = json.loads(args[args.index("--speculative-config") + 1])
    assert speculative == {"method": "dflash", "model": DRAFTER.local_dir, "num_speculative_tokens": 15}
    assert "--reasoning-parser" not in render_gpu_text()
    assert plan()["serve_args"] == args


@pytest.mark.parametrize("planted", ["--reasoning-parser muse_glimmer", "--reasoning-parser=muse_glimmer"])
def test_plant_gpu_user_data_with_a_reasoning_parser_is_refused(planted: str) -> None:
    text = render_gpu_text().replace("--tool-call-parser muse_glimmer", f"--tool-call-parser muse_glimmer {planted}")
    assert planted in text
    with pytest.raises(UserDataError, match="reasoning parser"):
        check_gpu(text)


@pytest.mark.parametrize("guard", GPU_GUARDS)
def test_plant_gpu_user_data_that_lost_a_guard_is_refused(guard: str) -> None:
    text = render_gpu_text()
    assert guard.strip() in text
    with pytest.raises(UserDataError, match="lost or weakened a guard"):
        check_gpu(text.replace(guard.strip(), "true"))


@pytest.mark.parametrize(
    ("planted", "match"),
    [
        ("docker pull vllm/vllm-openai:v0.28.0", "by tag"),
        ("export HF_TOKEN=x  # hf_token", "no token"),
        ("vllm serve --api-key abc", "no token"),
        ("hf_" + "c" * 34, "secret-shaped"),
    ],
)
def test_plant_gpu_user_data_with_a_tag_or_a_token_is_refused(planted: str, match: str) -> None:
    text = render_gpu_text().replace("runcmd:\n", f"# {planted}\nruncmd:\n")
    with pytest.raises(UserDataError, match=match):
        check_gpu(text)
