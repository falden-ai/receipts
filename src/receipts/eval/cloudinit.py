"""The eval VM's cloud-init user data: Docker, gVisor's runsc registered as a Docker runtime, and the Epoch images
pulled by digest. Nothing runs on AWS from here; this only writes the file the integrator launches with.

The boot script refuses to go on if the VM has an instance role or answers IMDSv1 (spec parameters: no instance role),
if the Systems Manager agent cannot be removed (REG-562: SSH only, no Systems Manager), or if runsc is registered but
a probe container does not really run on gVisor. What is pinned: the gVisor and uv tarballs by sha512 and sha256, and
every image by digest. What is not: the apt packages (Ubuntu's signed archive, at whatever version the mirror serves)
and swebench 4.1.0's own dependencies from PyPI (pinned by version only).

EC2 user data is readable from instance metadata by any process on the VM that can reach it, so `check` refuses
anything that looks like a secret. EC2 caps user data at 16 KB before base64; when the plain file is over that,
`render` returns the gzip form, which cloud-init unpacks.

Five scripts land in /opt/receipts: bootstrap.sh (first boot only), probe.sh (the kernel probe both others use),
check-imds.sh (the metadata guard), check-no-ssm.sh (no Systems Manager agent installed or running) and vm-check.sh
(run over SSH after every start: the boot finished, the metadata guard holds, no SSM agent, runsc is still gVisor and
runc is not, every image is still on disk). The tests run check-imds.sh, check-no-ssm.sh and vm-check.sh against fake
curl, snap, pgrep and docker commands, one plant per check.
"""

from __future__ import annotations

import gzip
import re
from collections.abc import Sequence

from receipts.eval.gpuvm import VLLM_IMAGE, boot_script
from receipts.eval.images import ImageRef
from receipts.eval.runsc_check import PRIMARY, limit_flags

USER_DATA_LIMIT = 16 * 1024  # bytes, raw, before base64 (EC2 user guide, "Run commands ... with user data input")

GVISOR_RELEASE = "20261005.0"  # published 2026-10-08
GVISOR_SHA512 = (
    "27fed731b432d6a39d908956623fd273979fda66b3ccf4b9c12413ea417c516e"
    "18a9dc3cced403fb28efe495d64dce4251f51cf8c4d9f4a72f9833bd96b6bb35"
)
UV_VERSION = "0.9.13"
UV_SHA256 = "c45a44144bf23a2182e143227b4ad0bbe41a2bb7161a637c02e968906af53fd1"
SWEBENCH_VERSION = "4.1.0"  # the harness SWE-ABS's own evaluator builds on
PULL_WORKERS = 8

_LINE = re.compile(r"^[A-Za-z0-9_.-]+__[A-Za-z0-9_.-]+-\d+ sha256:[0-9a-f]{64}$")
_TAGGED_REF = re.compile(r"(?:ghcr|docker)\.io/[^\s\"'@$]+:[A-Za-z0-9._-]+")
# What the scripts must keep doing, as whole lines: `check` refuses a render that dropped or weakened any of them (a
# guard whose `|| fail` became `|| echo` no longer matches).
GUARDS = (
    'bash "$R/check-imds.sh" || fail "the instance metadata check failed"',
    '[ "$role" = "404" ] || { echo "an instance role is attached (iam/security-credentials answered $role)"; exit 1; }',
    '[ "$v1" = "401" ] || { echo "IMDSv1 answered $v1; launch with HttpTokens=required"; exit 1; }',
    'snap remove --purge amazon-ssm-agent || fail "could not remove the SSM agent snap"',
    'bash "$R/check-no-ssm.sh" || fail "a Systems Manager agent is still on this VM"',
    'if snap list amazon-ssm-agent > /dev/null 2>&1; then echo "the amazon-ssm-agent snap is installed"; exit 1; fi',
    'if pgrep -f amazon-ssm-agent > /dev/null 2>&1; then echo "an SSM agent process is running"; exit 1; fi',
    'echo "@GVISOR_SHA512@  gvisor.tar.zstd" | sha512sum -c - || fail "gVisor tarball sha512 mismatch"',
    '/usr/local/bin/runsc install || fail "runsc install failed"',
    "docker info --format '{{json .Runtimes}}' | grep -q '\"runsc\"' || fail \"runsc is not registered with Docker\"",
    '[ "$got" = "$want" ] || fail "pulled $got of $want images; see $R/pull.log"',
    'probe "$P" runsc > "$R/runsc-dmesg.txt" 2>&1 || fail "the probe would not run under runsc"',
    'grep -q \'Starting gVisor\\.\\.\\.\' "$R/runsc-dmesg.txt" || fail "runsc ran the probe, but not on gVisor"',
    'if probe "$P" runc 2>&1 | grep -q \'Starting gVisor\'; then fail "runc ran the probe on gVisor"; fi',
    'echo "@UV_SHA256@  uv.tar.gz" | sha256sum -c - || fail "uv tarball sha256 mismatch"',
    'usermod -aG docker ubuntu || fail "could not add ubuntu to the docker group"',
    # vm-check.sh, run after every start
    '[ -f "$R/BOOT_OK" ] || { echo "the first-boot bootstrap never finished; read $R/bootstrap.log"; exit 1; }',
    'bash "$R/check-imds.sh" || { echo "the instance metadata check failed"; exit 1; }',
    'bash "$R/check-no-ssm.sh" || { echo "a Systems Manager agent is on this VM"; exit 1; }',
    'probe "$P" runsc 2>&1 | grep -q \'Starting gVisor\\.\\.\\.\' || { echo "runsc did not run gVisor"; exit 1; }',
    'if probe "$P" runc 2>&1 | grep -q \'Starting gVisor\'; then echo "the runc probe ran on gVisor"; exit 1; fi',
    '[ "$missing" = 0 ] || { echo "$missing pinned images are missing"; exit 1; }',
)
_SECRETS = (
    re.compile(r"FALDEN_META_API_KEY"),
    re.compile(r"AKIA[0-9A-Z]{16}"),
    re.compile(r"ASIA[0-9A-Z]{16}"),
    re.compile(r"(?i)aws_secret_access_key"),
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----"),
    re.compile(r"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(r"\bhf_[A-Za-z0-9]{20,}"),
    re.compile(r"(?i)\bsk-[A-Za-z0-9_-]{20,}"),
)

BOOTSTRAP = r"""#!/usr/bin/env bash
# Receipts eval VM bootstrap (lane eval, EV1), first boot only. Generated by `receipts eval cloud-init`.
set -euo pipefail
R=/opt/receipts
exec > >(tee -a "$R/bootstrap.log") 2>&1
fail() { echo "BOOTSTRAP FAILED: $*" | tee "$R/BOOT_FAILED"; exit 1; }
rm -f "$R/BOOT_OK" "$R/BOOT_FAILED"

# 1. The spec forbids an instance role, and IMDSv1 would let any process read this user data unauthenticated.
bash "$R/check-imds.sh" || fail "the instance metadata check failed"
usermod -aG docker ubuntu || fail "could not add ubuntu to the docker group"
# No Systems Manager (REG-562): the Ubuntu image ships the SSM agent as a snap. SSH is the only way in.
if snap list amazon-ssm-agent > /dev/null 2>&1; then
  snap remove --purge amazon-ssm-agent || fail "could not remove the SSM agent snap"
fi
bash "$R/check-no-ssm.sh" || fail "a Systems Manager agent is still on this VM"

# 2. gVisor, pinned by release and sha512, registered with Docker as the runsc runtime.
cd /tmp
curl -fsSL -o gvisor.tar.zstd "https://storage.googleapis.com/gvisor/releases/release/@GVISOR_RELEASE@/x86_64/gvisor.tar.zstd"
echo "@GVISOR_SHA512@  gvisor.tar.zstd" | sha512sum -c - || fail "gVisor tarball sha512 mismatch"
tar --zstd -xf gvisor.tar.zstd -C /usr/local/bin
/usr/local/bin/runsc install || fail "runsc install failed"
systemctl restart docker
docker info --format '{{json .Runtimes}}' | grep -q '"runsc"' || fail "runsc is not registered with Docker"

# 3. The Epoch SWE-bench images, by digest only, @PULL_WORKERS@ at a time. A pull by digest is checked by Docker itself.
pull() {
  ref="ghcr.io/epoch-research/swe-bench.eval.x86_64.$1@$2"
  if docker pull -q "$ref" >/dev/null; then echo "ok $1"; else echo "FAILED $1 $ref"; fi
}
export -f pull
: > "$R/pull.log"
xargs -a "$R/images.txt" -P @PULL_WORKERS@ -L 1 bash -c 'pull "$0" "$1"' >> "$R/pull.log"
want=$(grep -c . "$R/images.txt")
got=$(grep -c '^ok ' "$R/pull.log" || true)
[ "$got" = "$want" ] || fail "pulled $got of $want images; see $R/pull.log"

# 4. gVisor really runs. Under runsc the kernel log starts "Starting gVisor..."; under runc with every capability
#    dropped it cannot be read at all. (Docker's HostConfig.Runtime only echoes the flag it was given.)
source "$R/probe.sh"
read -r first digest < "$R/images.txt"
P="ghcr.io/epoch-research/swe-bench.eval.x86_64.$first@$digest"
probe "$P" runsc > "$R/runsc-dmesg.txt" 2>&1 || fail "the probe would not run under runsc"
grep -q 'Starting gVisor\.\.\.' "$R/runsc-dmesg.txt" || fail "runsc ran the probe, but not on gVisor"
if probe "$P" runc 2>&1 | grep -q 'Starting gVisor'; then fail "runc ran the probe on gVisor"; fi

# 5. uv, pinned by sha256, and the SWE-bench harness the runsc check grades with.
curl -fsSL -o uv.tar.gz "https://github.com/astral-sh/uv/releases/download/@UV_VERSION@/uv-x86_64-unknown-linux-gnu.tar.gz"
echo "@UV_SHA256@  uv.tar.gz" | sha256sum -c - || fail "uv tarball sha256 mismatch"
tar -xzf uv.tar.gz --strip-components=1 -C /usr/local/bin
uv venv "$R/venv" --python 3.12
uv pip install --python "$R/venv/bin/python" "swebench==@SWEBENCH_VERSION@"

{
  docker --version
  /usr/local/bin/runsc --version | head -1
  uv --version
  "$R/venv/bin/python" -c 'import importlib.metadata as m; print("swebench", m.version("swebench"))'
} > "$R/versions.txt"
echo "ok $(date -u +%FT%TZ) images=$got" > "$R/BOOT_OK"
"""

# The run phase's flags (contracts.RUN_PHASE_LIMITS), so the boot probe runs what the gold check runs.
PROBE = r"""# Sourced by bootstrap.sh and vm-check.sh: probe <image ref> <runtime> runs dmesg in a throwaway container.
probe() {
  docker run --rm --pull=never --runtime="$2" @LIMIT_FLAGS@ "$1" dmesg
}
"""

CHECK_IMDS = r"""#!/usr/bin/env bash
# Exits 0 only when this VM has no instance role and IMDSv1 is off. Generated by `receipts eval cloud-init`.
set -uo pipefail
M=http://169.254.169.254/latest
TOKEN=$(curl -sS -X PUT -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' "$M/api/token") \
  || { echo "no IMDSv2 token"; exit 1; }
role=$(curl -s -o /dev/null -w '%{http_code}' -H "X-aws-ec2-metadata-token: $TOKEN" \
  "$M/meta-data/iam/security-credentials/")
[ "$role" = "404" ] || { echo "an instance role is attached (iam/security-credentials answered $role)"; exit 1; }
v1=$(curl -s -o /dev/null -w '%{http_code}' "$M/meta-data/")
[ "$v1" = "401" ] || { echo "IMDSv1 answered $v1; launch with HttpTokens=required"; exit 1; }
echo "imds ok: no instance role, IMDSv1 off"
"""

CHECK_NO_SSM = r"""#!/usr/bin/env bash
# Exits 0 only when no Systems Manager agent is installed or running (REG-562). Generated by `receipts eval cloud-init`.
set -uo pipefail
if snap list amazon-ssm-agent > /dev/null 2>&1; then echo "the amazon-ssm-agent snap is installed"; exit 1; fi
if pgrep -f amazon-ssm-agent > /dev/null 2>&1; then echo "an SSM agent process is running"; exit 1; fi
echo "no SSM agent"
"""

VM_CHECK = r"""#!/usr/bin/env bash
# Run over SSH after every start (eval/vm/launch.sh start does): the boot finished, the metadata guard holds, no SSM
# agent, runsc is still gVisor and runc is not, and every pinned image is still on disk. Exits non-zero on the first
# problem. RECEIPTS_VM_ROOT exists for the tests; on the VM it is /opt/receipts.
set -uo pipefail
R="${RECEIPTS_VM_ROOT:-/opt/receipts}"
[ -f "$R/BOOT_OK" ] || { echo "the first-boot bootstrap never finished; read $R/bootstrap.log"; exit 1; }
bash "$R/check-imds.sh" || { echo "the instance metadata check failed"; exit 1; }
bash "$R/check-no-ssm.sh" || { echo "a Systems Manager agent is on this VM"; exit 1; }
source "$R/probe.sh"
read -r first digest < "$R/images.txt"
P="ghcr.io/epoch-research/swe-bench.eval.x86_64.$first@$digest"
probe "$P" runsc 2>&1 | grep -q 'Starting gVisor\.\.\.' || { echo "runsc did not run gVisor"; exit 1; }
if probe "$P" runc 2>&1 | grep -q 'Starting gVisor'; then echo "the runc probe ran on gVisor"; exit 1; fi
missing=0
while read -r id d; do
  docker image inspect "ghcr.io/epoch-research/swe-bench.eval.x86_64.$id@$d" > /dev/null 2>&1 \
    || missing=$((missing + 1))
done < "$R/images.txt"
[ "$missing" = 0 ] || { echo "$missing pinned images are missing"; exit 1; }
echo "vm ok: gVisor, $(grep -c . "$R/images.txt") images, no instance role, no SSM agent"
"""


class UserDataError(ValueError):
    """User data that must not be launched."""


def _indent(text: str, spaces: int) -> str:
    pad = " " * spaces
    return "".join(f"{pad}{line}\n" if line else "\n" for line in text.splitlines())


def image_lines(images: Sequence[ImageRef]) -> str:
    if not images:
        raise UserDataError("no images to pull")
    ids = [i.instance_id for i in images]
    if len(set(ids)) != len(ids):
        raise UserDataError("an instance appears twice in the image list")
    return "".join(f"{i.instance_id} {i.digest}\n" for i in sorted(images, key=lambda i: i.instance_id))


def _fill(text: str) -> str:
    for key, value in (
        ("@GVISOR_RELEASE@", GVISOR_RELEASE),
        ("@GVISOR_SHA512@", GVISOR_SHA512),
        ("@UV_VERSION@", UV_VERSION),
        ("@UV_SHA256@", UV_SHA256),
        ("@SWEBENCH_VERSION@", SWEBENCH_VERSION),
        ("@PULL_WORKERS@", str(PULL_WORKERS)),
        ("@LIMIT_FLAGS@", " ".join(f for f in limit_flags(PRIMARY) if not f.startswith("--runtime="))),
    ):
        text = text.replace(key, value)
    return text


def bootstrap_script() -> str:
    return _fill(BOOTSTRAP)


def scripts() -> dict[str, str]:
    """Every file the user data writes under /opt/receipts, other than the image list."""
    return {
        "bootstrap.sh": _fill(BOOTSTRAP),
        "probe.sh": _fill(PROBE),
        "check-imds.sh": _fill(CHECK_IMDS),
        "check-no-ssm.sh": _fill(CHECK_NO_SSM),
        "vm-check.sh": _fill(VM_CHECK),
    }


def guards() -> tuple[str, ...]:
    return tuple(_fill(g) for g in GUARDS)


def render_text(images: Sequence[ImageRef]) -> str:
    header = (
        "#cloud-config\n"
        "# Receipts eval VM (lane eval, EV1). Generated by `receipts eval cloud-init`; do not edit by hand.\n"
        "# Any process on the VM can read this user data through instance metadata: nothing secret belongs here.\n"
    )
    body = (
        "package_update: true\n"
        "packages:\n"
        "  - docker.io\n"
        "  - zstd\n"
        "  - git\n"
        "  - rsync\n"
        "write_files:\n"
        "  - path: /opt/receipts/images.txt\n"
        '    permissions: "0644"\n'
        "    content: |\n"
        f"{_indent(image_lines(images), 6)}"
        + "".join(
            f"  - path: /opt/receipts/{name}\n"
            f'    permissions: "{"0644" if name == "probe.sh" else "0755"}"\n'
            "    content: |\n"
            f"{_indent(body, 6)}"
            for name, body in scripts().items()
        )
        + "runcmd:\n"
        "  - [bash, /opt/receipts/bootstrap.sh]\n"
    )
    text = header + body
    check(text)
    return text


def check(text: str) -> None:
    """Refuse user data that pulls by tag, lacks a guard, or carries anything secret-shaped."""
    if not text.startswith("#cloud-config\n"):
        raise UserDataError("user data must start with #cloud-config")
    for pattern in _SECRETS:
        if pattern.search(text):
            raise UserDataError(f"secret-shaped text in user data ({pattern.pattern}); user data is not private")
    if ":latest" in text or _TAGGED_REF.search(text):
        raise UserDataError("an image is referenced by tag; pull by digest only")
    lines = _section(text, "/opt/receipts/images.txt")
    if not lines:
        raise UserDataError("no image list")
    for line in lines:
        if not _LINE.match(line):
            raise UserDataError(f"image line {line!r} is not '<instance_id> sha256:<64 hex>'")
    present = {line.strip() for line in text.splitlines()}
    for needle in guards():
        if needle not in present:
            raise UserDataError(f"the scripts lost or weakened a guard: {needle!r}")


def _section(text: str, path: str) -> list[str]:
    """The content lines of one write_files entry."""
    lines = text.splitlines()
    try:
        start = lines.index(f"  - path: {path}")
    except ValueError:
        return []
    out = []
    for line in lines[start + 1 :]:
        if line.startswith("  - path:") or not line.startswith("  "):
            break
        if line.startswith("      "):
            out.append(line[6:])
    return out


def render(images: Sequence[ImageRef]) -> tuple[bytes, bool]:
    """The bytes to pass as user data, and whether they are gzip. Refuses anything EC2 would reject."""
    text = render_text(images).encode("utf-8")
    if len(text) <= USER_DATA_LIMIT:
        return text, False
    packed = gzip.compress(text, compresslevel=9, mtime=0)
    if len(packed) > USER_DATA_LIMIT:
        raise UserDataError(f"user data is {len(packed)} bytes gzipped, over EC2's {USER_DATA_LIMIT}")
    return packed, True


# The GPU VM's user data (receipts.eval.gpuvm): the same metadata and SSM guards, then the Glimmer server.
GPU_GUARDS = (
    'bash "$R/check-imds.sh" || fail "the instance metadata check failed"',
    'bash "$R/check-no-ssm.sh" || fail "a Systems Manager agent is still on this VM"',
    "nvidia-smi -L | grep -q 'RTX PRO 6000' || fail \"no RTX PRO 6000 GPU on this VM\"",
    f'docker pull -q {VLLM_IMAGE.ref} > /dev/null || fail "could not pull the pinned vLLM image"',
    '|| fail "a container cannot reach the GPU (NVIDIA container toolkit)"',
    "-v /opt/models:/models:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \\",
    '[ "$role" = "404" ] || { echo "an instance role is attached (iam/security-credentials answered $role)"; exit 1; }',
    '[ "$v1" = "401" ] || { echo "IMDSv1 answered $v1; launch with HttpTokens=required"; exit 1; }',
)


def gpu_scripts() -> dict[str, str]:
    return {
        "check-imds.sh": _fill(CHECK_IMDS),
        "check-no-ssm.sh": _fill(CHECK_NO_SSM),
        "gpu-bootstrap.sh": boot_script(),
    }


def render_gpu_text() -> str:
    header = (
        "#cloud-config\n"
        "# Receipts GPU VM (the Glimmer arm). Generated by `receipts eval vm-files`; do not edit by hand.\n"
        "# Any process on the VM can read this user data through instance metadata: nothing secret belongs here.\n"
    )
    body = (
        "write_files:\n"
        + "".join(
            f'  - path: /opt/receipts/{name}\n    permissions: "0755"\n    content: |\n{_indent(text, 6)}'
            for name, text in gpu_scripts().items()
        )
        + "runcmd:\n"
        "  - [bash, /opt/receipts/gpu-bootstrap.sh]\n"
    )
    text = header + body
    check_gpu(text)
    return text


def check_gpu(text: str) -> None:
    """Refuse GPU user data that pulls by tag, lacks a guard, carries anything secret-shaped, or starts vLLM with a
    reasoning parser, which on the pinned image turns off response_format json_schema (receipts.eval.gpuvm)."""
    if not text.startswith("#cloud-config\n"):
        raise UserDataError("user data must start with #cloud-config")
    for pattern in _SECRETS:
        if pattern.search(text):
            raise UserDataError(f"secret-shaped text in user data ({pattern.pattern}); user data is not private")
    if ":latest" in text or _TAGGED_REF.search(text) or "vllm-openai:" in text:
        raise UserDataError("an image is referenced by tag; pull by digest only")
    if re.search(r"(?i)\bhf_token\b|huggingface-cli login|--api-key", text):
        raise UserDataError("the GPU VM takes no token of any kind")
    if "--reasoning-parser" in text:
        raise UserDataError("vLLM starts with a reasoning parser, which drops the json_schema on the pinned image")
    present = {line.strip() for line in text.splitlines()}
    for needle in GPU_GUARDS:
        if needle.strip() not in present:
            raise UserDataError(f"the GPU scripts lost or weakened a guard: {needle!r}")
    if len(text.encode("utf-8")) > USER_DATA_LIMIT:
        raise UserDataError("the GPU user data is over EC2's limit")
