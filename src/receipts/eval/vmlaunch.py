"""The eval VM's exact launch inputs, for the integrator to run. Nothing here calls AWS: `files` renders the inputs
under eval/vm/, and eval/vm/launch.sh runs them with the integrator's own AWS profile.

The founder's decisions: m7i.8xlarge on demand in us-east-1 for the pilot and the rehearsal, m7i.16xlarge for the 11/1
final (REG-556, REG-563), and m7i.16xlarge for R24 and the label runs, which is the size the VM launches at, then
`launch.sh resize m7i.8xlarge` before the pilot (REG-568; a resize keeps the volume and the pulled images, and
launch.sh prints the size and the vCPUs it needs before it launches or resizes); stopped between runs; and
access by SSH only (REG-562): 22/tcp from the founder's current public IP as one /32, read when the VM launches or
starts and printed before anything changes, a new key pair whose private key lands in a private folder
($RECEIPTS_KEY_DIR, mode 700) with mode 600, no instance role and no Systems Manager.

The machine: Canonical's Ubuntu 24.04 LTS amd64 image, pinned by AMI id (an AMI id never changes what it points to)
and checked by owner, name, state, architecture and root device before launch; no instance profile; IMDSv2 only, hop
limit 1; a 500 GiB encrypted gp3 root volume; termination protection on; a security group whose one inbound rule is
the founder's /32 on 22/tcp and whose egress is TCP 443 and 80 only. The boot script removes the image's SSM agent.
SSH pins the host key the VM prints to its console at first boot (no trust on first use).
"""

from __future__ import annotations

import shlex
from typing import Any

from pydantic import Field

from receipts.contracts import Frozen
from receipts.eval import gpuvm
from receipts.eval.cloudinit import render_gpu_text
from receipts.eval.hashing import pretty_json
from receipts.eval.vmplan import CHOSEN, FINAL_INSTANCE_TYPE, LABEL_INSTANCE_TYPE, REGION, Disk, offer


class AmiPin(Frozen):
    """One Ubuntu image, as Canonical's AWS catalog lists it."""

    region: str
    image_id: str = Field(pattern=r"^ami-[0-9a-f]{17}$")
    name: str
    owner_id: str = Field(pattern=r"^\d{12}$")
    serial: str = Field(pattern=r"^\d{8}(\.\d+)?$")
    architecture: str
    root_device_name: str
    root_device_type: str
    virtualization: str
    catalog_url: str
    catalog_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    catalog_updated: str
    read_at: str


# Canonical's released AWS catalog (simplestreams), read 2026-10-27: product com.ubuntu.cloud:server:24.04:amd64,
# version 20261004, item usee1hg (us-east-1, hvm, root_store ssd-gp3), pubname ubuntu-noble-24.04-amd64-server-20261004.
# The catalog's later serials (20261017, 20261023) were published to a few other Regions only.
AMI = AmiPin(
    region="us-east-1",
    image_id="ami-0fa5967347d08d2df",
    name="ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-20261004",
    owner_id="099720109477",  # Canonical
    serial="20261004",
    architecture="x86_64",
    root_device_name="/dev/sda1",
    root_device_type="ebs",
    virtualization="hvm",
    catalog_url="https://cloud-images.ubuntu.com/releases/streams/v1/com.ubuntu.cloud:released:aws.json",
    catalog_sha256="2429afdb1f7d243412804bcab89de7112460d0bc2d246adfeb409a61113aab8b",
    catalog_updated="Fri, 23 Oct 2026 21:16:12 +0000",
    read_at="2026-10-27",
)

NAME = "receipts-eval"
SECURITY_GROUP = "receipts-eval-ssh"
KEY_PREFIX = "receipts-eval"  # launch.sh names each new key pair <prefix>-<UTC timestamp>
LAUNCH_INSTANCE_TYPE = LABEL_INSTANCE_TYPE  # REG-568: R24 and the label runs first
PILOT_INSTANCE_TYPE = CHOSEN  # REG-556: `launch.sh resize` before the pilot
SIZES: dict[str, int] = {t: offer(t).vcpu for t in (LAUNCH_INSTANCE_TYPE, PILOT_INSTANCE_TYPE, FINAL_INSTANCE_TYPE)}
SSH_PORT = 22
# The founder's public IP is read from two independent services, which must agree.
IP_SOURCES = ("https://checkip.amazonaws.com", "https://api.ipify.org")
TAGS: tuple[tuple[str, str], ...] = (
    ("Name", NAME),
    ("Project", "falden-receipts"),
    ("Lane", "eval"),
    ("Purpose", "tier-a-eval"),
    ("StopBetweenRuns", "true"),
    ("DeleteAfter", "2026-11-04"),
)
EGRESS_PORTS = (443, 80)  # 443: Docker registries, GitHub, PyPI, gVisor, Systems Manager; 80: Ubuntu's apt mirrors


def tag_list() -> list[dict[str, str]]:
    return [{"Key": k, "Value": v} for k, v in TAGS]


def run_instances_input(disk: Disk | None = None) -> dict[str, Any]:
    """`aws ec2 run-instances --cli-input-json`, at the launch size (REG-568). Left out on purpose: IamInstanceProfile
    (there is none), KeyName (the new key pair launch.sh creates), UserData (fileb://eval/vm/cloud-init.yaml.gz) and
    SecurityGroupIds (the group launch.sh creates)."""
    d = disk or Disk()
    return {
        "ImageId": AMI.image_id,
        "InstanceType": LAUNCH_INSTANCE_TYPE,
        "MinCount": 1,
        "MaxCount": 1,
        "BlockDeviceMappings": [
            {
                "DeviceName": AMI.root_device_name,
                "Ebs": {
                    "VolumeSize": d.gib,
                    "VolumeType": "gp3",
                    "Iops": d.iops,
                    "Throughput": d.throughput_mibps,
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
        "TagSpecifications": [
            {"ResourceType": kind, "Tags": tag_list()} for kind in ("instance", "volume", "network-interface")
        ],
    }


def egress_rules() -> list[dict[str, Any]]:
    """`aws ec2 authorize-security-group-egress --ip-permissions`: TCP 443 and 80 to anywhere, nothing else."""
    return [
        {
            "IpProtocol": "tcp",
            "FromPort": port,
            "ToPort": port,
            "IpRanges": [{"CidrIp": "0.0.0.0/0", "Description": "receipts eval VM egress"}],
        }
        for port in EGRESS_PORTS
    ]


def expected_identity() -> str:
    """What launch.sh requires `describe-images` to print for the pinned AMI, tab-separated."""
    return "\t".join((AMI.name, "available", AMI.architecture, AMI.root_device_name, AMI.root_device_type))


def files() -> dict[str, str]:
    """eval/vm/ files launch.sh and gpu.sh read, by name."""
    return (
        gpuvm.files()
        | {"gpu-cloud-init.yaml": render_gpu_text()}
        | {
            "ami.json": pretty_json(AMI.model_dump(mode="json")),
            "run-instances.json": pretty_json(run_instances_input()),
            "egress.json": pretty_json(egress_rules()),
            "tags.json": pretty_json(tag_list()),
            "launch.env": "# Sourced by eval/vm/launch.sh. Generated by `python -m receipts.eval vm-files`.\n"
            + "".join(
                f"{k}={shlex.quote(v)}\n"
                for k, v in (
                    ("REGION", REGION),
                    ("LAUNCH_INSTANCE_TYPE", LAUNCH_INSTANCE_TYPE),
                    ("LAUNCH_VCPU", str(SIZES[LAUNCH_INSTANCE_TYPE])),
                    ("PILOT_INSTANCE_TYPE", PILOT_INSTANCE_TYPE),
                    ("PILOT_VCPU", str(SIZES[PILOT_INSTANCE_TYPE])),
                    ("FINAL_INSTANCE_TYPE", FINAL_INSTANCE_TYPE),
                    ("FINAL_VCPU", str(SIZES[FINAL_INSTANCE_TYPE])),
                    ("AMI_ID", AMI.image_id),
                    ("AMI_OWNER", AMI.owner_id),
                    ("AMI_IDENTITY", expected_identity().replace("\t", "|")),
                    ("NAME", NAME),
                    ("SECURITY_GROUP", SECURITY_GROUP),
                    ("KEY_PREFIX", KEY_PREFIX),
                    ("SSH_PORT", str(SSH_PORT)),
                    ("IP_SOURCES", " ".join(IP_SOURCES)),
                )
            ),
        }
    )
