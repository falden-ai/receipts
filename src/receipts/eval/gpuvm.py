"""The GPU VM for the pilot's Glimmer arm (REG-561): Muse Glimmer 30B on Falden's own vLLM, reachable only from the
eval VM. Nothing here calls AWS; this renders the inputs eval/vm/gpu.sh launches with, and the integrator runs it.

The machine: g7e.2xlarge on demand in us-east-1 (1x RTX PRO 6000, 96 GB; spec parameters, $3.36312/h), the newest
AWS Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04), which ships the NVIDIA driver, Docker and the NVIDIA
container toolkit. The integrator pins its AMI id with one describe-images call before launch and passes it to gpu.sh
(`--ami <id>` or RECEIPTS_GPU_AMI); without a pin gpu.sh takes the newest of Amazon's images by that name. Either way
gpu.sh refuses an image that is not Amazon's, available and x86_64, or whose name is not exactly the prefix and an
8-digit date (`GPU_AMI_NAME_REGEX`), and prints its id, name and date before anything starts. No instance profile,
IMDSv2 only, the SSM agent
removed at boot. The security group admits 8000/tcp (vLLM) and 22/tcp (SSH through the eval VM) from the eval VM's
security group only: no public port. Egress is TCP 443 only, for the image and the weights, and `gpu.sh seal`
removes it once the server is up (spec: no egress after the weights are pulled). Stopped after the pilot.

The server follows the vLLM recipe for Muse Glimmer 30B (https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B, read
2026-10-27): image vllm/vllm-openai:v0.28.0, pinned here by its linux/amd64 digest; BF16 weights; the muse_glimmer tool
call parser; `--generation-config auto`; one GPU, 131,072 context; and the DFlash drafter
(meta-models/Muse-Glimmer-30B-assistant) with `--max-num-seqs 32` and 15 speculative tokens, as the recipe's
speculative decoding section gives them. The served name is muse-glimmer-30b, the id contracts allows.

One departure from the recipe: no `--reasoning-parser`. The recipe runs the muse_glimmer reasoning parser beside the
tool parser, but on this image with it on, vLLM logs "Auto-initialization of reasoning token IDs failed" and never
applies response_format json_schema (the GPU VM tests of 2026-10-29: of 31 stage-1 answers in the sweep, 3 parsed as
JSON). With it off the schema is enforced and the drafter still gives about 300 tokens a second per request; dropping
the drafter instead cut that 3x and did not help. What it costs: vLLM no longer splits reasoning_content from content
or counts reasoning tokens. Neither was in use: no client here reads reasoning_content, and with the parser on every
Glimmer answer in those tests already reported reasoning_tokens 0, as its token ids never initialized. So a Glimmer
answer's reasoning_tokens stays 0, and everything it generates counts in output_tokens.

Both checkpoints are ungated Apache-2.0 repositories, so no Hugging Face token goes anywhere; they are downloaded at
their pinned revisions into local folders, and the server runs offline from those.

If the G and VT quota is not granted in time (AWS case 179312451500652), `MODAL_NOTE` says how to run the arm on Modal
instead; no code here does that.
"""

from __future__ import annotations

import json
import shlex
from typing import Any, Literal

from pydantic import Field

from receipts.contracts import IMAGE_REF, Frozen
from receipts.eval.hashing import pretty_json
from receipts.eval.vmplan import REGION

GPU_INSTANCE_TYPE = "g7e.2xlarge"
GPU_VCPU = 8  # counted against the Running On-Demand G and VT instances quota (L-DB2E81BA)
GPU_PRICE_MICRO_USD_PER_HOUR = 3_363_120  # $3.36312, AWS's public price data published 2026-10-23T22:13:55Z
GPU_NAME = "receipts-glimmer"
GPU_SECURITY_GROUP = "receipts-glimmer-from-eval"
GPU_AMI_NAME_PREFIX = "Deep Learning Base OSS Nvidia Driver GPU AMI (Ubuntu 24.04)"
# AWS names each release '<prefix> <YYYYMMDD>' (the DLAMI release notes, e.g. '... (Ubuntu 24.04) 20260225'); its own
# describe-images example filters on the prefix and eight '?' for the date. bash and Python read the same ERE.
GPU_AMI_NAME_REGEX = r"^Deep Learning Base OSS Nvidia Driver GPU AMI \(Ubuntu 24\.04\) [0-9]{8}$"
GPU_AMI_OWNER = "amazon"
GPU_DISK_GIB = 200  # BF16 weights 59.6 GB, the drafter 5.1 GB, the vLLM image, the AMI's own tools
VLLM_PORT = 8000
SERVED_NAME = "muse-glimmer-30b"


class Checkpoint(Frozen):
    repo: str
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    local_dir: str
    gated: Literal[False] = False
    license: Literal["apache-2.0"] = "apache-2.0"
    read_at: str


# https://huggingface.co/api/models/<repo>, read 2026-10-27: "gated": false, "license": "apache-2.0".
MODEL = Checkpoint(
    repo="meta-models/Muse-Glimmer-30B",
    revision="cec94fcfd6512f2cfe708138c32533b346ea4f69",
    local_dir="/models/glimmer",
    read_at="2026-10-27",
)
DRAFTER = Checkpoint(
    repo="meta-models/Muse-Glimmer-30B-assistant",
    revision="c676cce91658f9568aaf264a416211e2782735b7",
    local_dir="/models/glimmer-assistant",
    read_at="2026-10-27",
)


class ImagePin(Frozen):
    ref: str = Field(pattern=IMAGE_REF)
    tag: str
    index_digest: str = Field(pattern=r"^sha256:[0-9a-f]{64}$")
    platform: Literal["linux/amd64"] = "linux/amd64"
    read_at: str


# Docker Hub, vllm/vllm-openai:v0.28.0, read 2026-10-27: the tag's manifest list sha256:61fc8a89... names
# sha256:2286e853... for linux/amd64. Pulling by the platform digest pins the exact bytes the GPU VM runs.
VLLM_IMAGE = ImagePin(
    ref="docker.io/vllm/vllm-openai@sha256:2286e8533ca8b6bc777594bae30524f1426ba46ca21797524e06df6a94b06635",
    tag="v0.28.0",
    index_digest="sha256:61fc8a896b0a4fbbbdc063bc4b0dbc25ce98e02b5050c24aeb7830ac02039b14",
    read_at="2026-10-27",
)


def serve_args() -> list[str]:
    """`vllm serve` arguments (the image's entrypoint), from the recipe, with our served name and local weights."""
    speculative = {"method": "dflash", "model": DRAFTER.local_dir, "num_speculative_tokens": 15}
    return [
        "--model",
        MODEL.local_dir,
        "--served-model-name",
        SERVED_NAME,
        "--dtype",
        "bfloat16",
        "--tensor-parallel-size",
        "1",
        "--max-model-len",
        "131072",
        "--gpu-memory-utilization",
        "0.92",
        "--max-num-seqs",
        "32",
        "--enable-auto-tool-choice",
        "--tool-call-parser",
        "muse_glimmer",  # no --reasoning-parser: with it, vLLM 0.28.0 drops the json_schema (module docstring)
        "--generation-config",
        "auto",
        "--speculative-config",
        json.dumps(speculative, sort_keys=True),
        "--host",
        "0.0.0.0",  # inside the container; the security group admits only the eval VM
        "--port",
        str(VLLM_PORT),
    ]


MODAL_NOTE = (
    "If the G and VT quota (AWS case 179312451500652) is not granted before the pilot, run the Glimmer arm on Modal "
    "instead: Modal lists Muse Glimmer 30B BF16 as a dedicated endpoint on an RTX PRO 6000 at $0.000842 per second "
    "($3.03/h; research/open-weights-2026-10-27.md, unverified whether the event credit covers it). Deploy the same "
    "image and arguments (gpu-plan.json: serve_args, the two pinned revisions), give the endpoint Modal's own auth "
    "token, and point RECEIPTS_GLIMMER_URL at it from the eval VM. What changes: the endpoint is on the internet "
    "behind a token instead of inside the eval VM's security group, and the token is a secret the orchestrator holds "
    "and no sandbox sees. The arm, its seed, step limit and the move rule stay as pilot.json fixes them."
)


def boot_script() -> str:
    """The GPU VM's first-boot script: guards, the pinned image, the pinned weights, the offline server, READY."""
    image = shlex.quote(VLLM_IMAGE.ref)
    downloads = "\n".join(
        "docker run --rm --entrypoint python3 -v /opt/models:/models -e HF_HUB_DISABLE_TELEMETRY=1 "
        f"{image} -c "
        + shlex.quote(
            "from huggingface_hub import snapshot_download; "
            f'snapshot_download(repo_id="{c.repo}", revision="{c.revision}", local_dir="{c.local_dir}")'
        )
        + f' || fail "could not download {c.repo} at {c.revision[:12]}"'
        for c in (MODEL, DRAFTER)
    )
    serve = " ".join(shlex.quote(a) for a in serve_args())
    return f"""#!/usr/bin/env bash
# Receipts GPU VM bootstrap (lane eval, the Glimmer arm), first boot only. Generated by `receipts eval vm-files`.
set -euo pipefail
R=/opt/receipts
exec > >(tee -a "$R/bootstrap.log") 2>&1
fail() {{ echo "BOOTSTRAP FAILED: $*" | tee "$R/BOOT_FAILED"; exit 1; }}
rm -f "$R/READY" "$R/BOOT_FAILED"
bash "$R/check-imds.sh" || fail "the instance metadata check failed"
if snap list amazon-ssm-agent > /dev/null 2>&1; then
  snap remove --purge amazon-ssm-agent || fail "could not remove the SSM agent snap"
fi
if systemctl list-unit-files amazon-ssm-agent.service > /dev/null 2>&1; then
  systemctl disable --now amazon-ssm-agent.service || true
  apt-get purge -y amazon-ssm-agent || true
fi
bash "$R/check-no-ssm.sh" || fail "a Systems Manager agent is still on this VM"
nvidia-smi -L | grep -q 'RTX PRO 6000' || fail "no RTX PRO 6000 GPU on this VM"
docker pull -q {image} > /dev/null || fail "could not pull the pinned vLLM image"
docker run --rm --gpus all --entrypoint nvidia-smi {image} -L | grep -q 'RTX PRO 6000' \\
  || fail "a container cannot reach the GPU (NVIDIA container toolkit)"
mkdir -p /opt/models
{downloads}
docker rm -f glimmer > /dev/null 2>&1 || true
docker run -d --name glimmer --restart unless-stopped --gpus all --ipc=host -p {VLLM_PORT}:{VLLM_PORT} \\
  -v /opt/models:/models:ro -e HF_HUB_OFFLINE=1 -e TRANSFORMERS_OFFLINE=1 -e HF_HUB_DISABLE_TELEMETRY=1 \\
  {image} {serve} > /dev/null || fail "could not start vLLM"
for _ in $(seq 1 160); do
  if curl -fsS "http://127.0.0.1:{VLLM_PORT}/v1/models" 2> /dev/null | grep -q '"{SERVED_NAME}"'; then
    echo "ok $(date -u +%FT%TZ)" > "$R/READY"
    exit 0
  fi
  sleep 15
done
fail "vLLM never served {SERVED_NAME}; read docker logs glimmer"
"""


def run_instances_input() -> dict[str, Any]:
    """`aws ec2 run-instances --cli-input-json` for the GPU VM. Left out on purpose: ImageId (resolved at launch),
    KeyName (the eval VM's new key pair), IamInstanceProfile (there is none), UserData and SecurityGroupIds."""
    tags = [
        {"Key": "Name", "Value": GPU_NAME},
        {"Key": "Project", "Value": "falden-receipts"},
        {"Key": "Lane", "Value": "eval"},
        {"Key": "Purpose", "Value": "glimmer-arm"},
        {"Key": "StopAfterPilot", "Value": "true"},
        {"Key": "DeleteAfter", "Value": "2026-11-04"},
    ]
    return {
        "InstanceType": GPU_INSTANCE_TYPE,
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [
            {
                "DeviceName": "/dev/sda1",
                "Ebs": {
                    "VolumeSize": GPU_DISK_GIB,
                    "VolumeType": "gp3",
                    "DeleteOnTermination": True,
                    "Encrypted": True,
                },
            }
        ],
        "MetadataOptions": {
            "HttpEndpoint": "enabled",
            "HttpTokens": "required",
            "HttpPutResponseHopLimit": 1,
            "InstanceMetadataTags": "disabled",
        },
        "InstanceInitiatedShutdownBehavior": "stop",
        "DisableApiTermination": True,
        "Monitoring": {"Enabled": False},
        "TagSpecifications": [{"ResourceType": kind, "Tags": tags} for kind in ("instance", "volume")],
    }


def egress_rules() -> list[dict[str, Any]]:
    """TCP 443 only, for the image and the weights; `gpu.sh seal` revokes it once vLLM is up."""
    return [
        {
            "IpProtocol": "tcp",
            "FromPort": 443,
            "ToPort": 443,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "receipts glimmer: image and weights, then sealed"}],
        }
    ]


def plan() -> dict[str, Any]:
    return {
        "region": REGION,
        "instance_type": GPU_INSTANCE_TYPE,
        "vcpu_quota_needed": GPU_VCPU,
        "usd_per_hour": "$3.36312",
        "ami": {
            "owner": GPU_AMI_OWNER,
            "name_prefix": GPU_AMI_NAME_PREFIX,
            "name_regex": GPU_AMI_NAME_REGEX,
            "pinned": "by the integrator: one describe-images call, then gpu.sh launch --ami <id> or RECEIPTS_GPU_AMI",
            "fallback": "without a pin, the newest available image whose name matches, resolved at launch",
        },
        "vllm_image": VLLM_IMAGE.model_dump(mode="json"),
        "model": MODEL.model_dump(mode="json"),
        "drafter": DRAFTER.model_dump(mode="json"),
        "served_name": SERVED_NAME,
        "port": VLLM_PORT,
        "serve_args": serve_args(),
        "recipe": "https://recipes.vllm.ai/meta-models/Muse-Glimmer-30B (read 2026-10-27)",
        "recipe_departure": "no --reasoning-parser: with it, vLLM 0.28.0 never applies response_format json_schema",
        "reachable_from": "the eval VM's security group only, on 8000/tcp (vLLM) and 22/tcp (SSH through the eval VM)",
        "egress": "TCP 443 until the server is up, then none (gpu.sh seal)",
        "client": f"RECEIPTS_GLIMMER_URL=http://<private ip>:{VLLM_PORT}/v1 on the eval VM (gpu.sh url prints it)",
        "stop": "gpu.sh stop after the pilot; terminate after the final if the arm moves, or after the pilot if not",
        "if_quota_slips": MODAL_NOTE,
    }


def files() -> dict[str, str]:
    """eval/vm/ files gpu.sh reads, by name."""
    return {
        "gpu-plan.json": pretty_json(plan()),
        "gpu-run-instances.json": pretty_json(run_instances_input()),
        "gpu-egress.json": pretty_json(egress_rules()),
        "gpu.env": "# Sourced by eval/vm/gpu.sh. Generated by `python -m receipts.eval vm-files`.\n"
        + "".join(
            f"{k}={shlex.quote(v)}\n"
            for k, v in (
                ("REGION", REGION),
                ("GPU_INSTANCE_TYPE", GPU_INSTANCE_TYPE),
                ("GPU_VCPU", str(GPU_VCPU)),
                ("GPU_DISK_GIB", str(GPU_DISK_GIB)),
                ("GPU_NAME", GPU_NAME),
                ("GPU_SECURITY_GROUP", GPU_SECURITY_GROUP),
                ("GPU_AMI_OWNER", GPU_AMI_OWNER),
                ("GPU_AMI_NAME_PREFIX", GPU_AMI_NAME_PREFIX),
                ("GPU_AMI_NAME_REGEX", GPU_AMI_NAME_REGEX),
                ("VLLM_PORT", str(VLLM_PORT)),
                ("SERVED_NAME", SERVED_NAME),
            )
        ),
    }
