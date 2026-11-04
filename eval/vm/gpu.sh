#!/usr/bin/env bash
# The Glimmer arm's GPU VM (REG-561), for the integrator: launch, status, url, seal, stop, start, terminate.
#
# Runs with the caller's own AWS_PROFILE in us-east-1, after the eval VM exists (launch.sh), and shares its state
# folder. The eval lane wrote this file and never runs it. The VM: g7e.2xlarge from an AWS Deep Learning Base OSS
# Nvidia Driver GPU AMI (Ubuntu 24.04): the AMI id the integrator pinned with one describe-images call (`launch --ami
# <id>` or RECEIPTS_GPU_AMI), else the newest of Amazon's images by that name. Either way the image must be Amazon's,
# available, x86_64 and named exactly '<prefix> <YYYYMMDD>' (GPU_AMI_NAME_REGEX), and its id, name and date are
# printed before launch; the eval VM's key pair; no instance role; IMDSv2 only; the SSM agent removed at boot. Its
# security group
# admits 8000/tcp (vLLM) and 22/tcp (SSH, through the eval VM) from the eval VM's security group only: no public port.
# The eval VM's group gets the matching egress. Egress TCP 443 lets it pull the pinned vLLM image and the pinned
# weights; `seal` removes it once vLLM serves muse-glimmer-30b, so the GPU VM has no egress during the pilot. Stop it
# after the pilot (eval/vm/gpu-plan.json says what moves it to the final). Inputs: gpu.env, gpu-run-instances.json,
# gpu-egress.json and gpu-cloud-init.yaml, made by `python -m receipts.eval vm-files`.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VM="$ROOT/eval/vm"
STATE="${RECEIPTS_EVAL_STATE:-$ROOT/runs/vm}"
POLL_S="${RECEIPTS_EVAL_POLL_S:-15}"
# shellcheck source=/dev/null
source "$VM/gpu.env"

die() { echo "gpu.sh: $*" >&2; exit 1; }
say() { echo "gpu.sh: $*"; }
a() { aws --region "$REGION" "$@"; }
need_profile() { [ -n "${AWS_PROFILE:-}" ] || die "set AWS_PROFILE to the integrator's Falden profile first"; }
state() { [ -f "$STATE/$1" ] || die "no $1 is recorded in $STATE"; cat "$STATE/$1"; }

group_pair() {
  # One TCP port to or from one security group; a description only when authorizing (a revoke matches without it).
  if [ -n "${3:-}" ]; then
    printf '[{"IpProtocol":"tcp","FromPort":%s,"ToPort":%s,"UserIdGroupPairs":[{"GroupId":"%s","Description":"%s"}]}]' \
      "$1" "$1" "$2" "$3"
  else
    printf '[{"IpProtocol":"tcp","FromPort":%s,"ToPort":%s,"UserIdGroupPairs":[{"GroupId":"%s"}]}]' "$1" "$1" "$2"
  fi
}

resolve_ami() {
  # The image to launch: the pinned id if there is one, else the newest available image by its exact name. Either way
  # it must be the pinned one (when pinned), Amazon's, available, x86_64 and named exactly '<prefix> <YYYYMMDD>'.
  local pinned="$1" found how wanted ami_re='^ami-([0-9a-f]{8}|[0-9a-f]{17})$'
  if [ -n "$pinned" ]; then
    [[ $pinned =~ $ami_re ]] || die "'$pinned' is not an AMI id"
    found=$(a ec2 describe-images --image-ids "$pinned" --owners "$GPU_AMI_OWNER" \
      --query 'Images[0].[ImageId,CreationDate,OwnerId,Name,State,Architecture]' --output text)
    how="pinned"
  else
    found=$(a ec2 describe-images --owners "$GPU_AMI_OWNER" \
      --filters "Name=name,Values=$GPU_AMI_NAME_PREFIX ????????" Name=state,Values=available Name=architecture,Values=x86_64 \
      --query 'sort_by(Images,&CreationDate)[-1].[ImageId,CreationDate,OwnerId,Name,State,Architecture]' --output text)
    how="not pinned: the newest by name"
  fi
  IFS=$'\t' read -r ami_id ami_created ami_owner ami_name ami_state ami_arch <<< "$found"
  wanted="$pinned"
  [ -n "$wanted" ] || wanted="named '$GPU_AMI_NAME_PREFIX <YYYYMMDD>'"
  case "$ami_id" in ami-*) ;; *) die "no available image $wanted from $GPU_AMI_OWNER" ;; esac
  [ -z "$pinned" ] || [ "$ami_id" = "$pinned" ] || die "describe-images answered $ami_id for the pinned $pinned"
  [[ $ami_name =~ $GPU_AMI_NAME_REGEX ]] || die "$ami_id is named '$ami_name', not '$GPU_AMI_NAME_PREFIX <YYYYMMDD>'"
  [ "$ami_state" = "available" ] && [ "$ami_arch" = "x86_64" ] \
    || die "$ami_id is $ami_state on $ami_arch; want available on x86_64"
  say "image $ami_id ($how): $ami_name, created $ami_created, owner $ami_owner"
}

preflight() {
  [ -f "$STATE/instance-id" ] && [ -f "$STATE/security-group-id" ] || die "launch the eval VM first (launch.sh launch)"
  [ ! -f "$STATE/gpu-instance-id" ] || die "a Glimmer VM is already recorded ($(cat "$STATE/gpu-instance-id")); use start, stop or terminate"
  quota=$(a service-quotas get-service-quota --service-code ec2 --quota-code L-DB2E81BA --query Quota.Value --output text)
  [ "${quota%.*}" -ge "$GPU_VCPU" ] \
    || die "the On-Demand G and VT vCPU quota is $quota; $GPU_INSTANCE_TYPE needs $GPU_VCPU (AWS case 179312451500652; eval/vm/gpu-plan.json says how to run the arm on Modal instead)"
  resolve_ami "$1"
  vpc=$(a ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  case "$vpc" in vpc-*) ;; *) die "no default VPC in $REGION" ;; esac
}

check_gpu_sg() {
  # Inbound: 22/tcp and 8000/tcp from the eval VM's group and nothing else. Egress: TCP 443 until sealed, then none.
  local gsg="$1" esg="$2" want_egress="$3" ingress egress
  ingress=$(a ec2 describe-security-groups --group-ids "$gsg" --query \
    'SecurityGroups[0].IpPermissions[].[IpProtocol,FromPort,ToPort,join(`,`,IpRanges[].CidrIp),length(Ipv6Ranges),length(PrefixListIds),join(`,`,UserIdGroupPairs[].GroupId)]' \
    --output text | LC_ALL=C sort | tr '\t\n' ',;')
  [ "$ingress" = "tcp,22,22,,0,0,$esg;tcp,$VLLM_PORT,$VLLM_PORT,,0,0,$esg;" ] \
    || die "security group $gsg allows inbound '$ingress'; want 22/tcp and $VLLM_PORT/tcp from $esg only"
  egress=$(a ec2 describe-security-groups --group-ids "$gsg" --query \
    'SecurityGroups[0].IpPermissionsEgress[].[IpProtocol,FromPort,ToPort,join(`,`,IpRanges[].CidrIp)]' \
    --output text | LC_ALL=C sort | tr '\t\n' ',;')
  [ "$egress" = "$want_egress" ] || die "security group $gsg allows egress '$egress', want '$want_egress'"
  say "security group $gsg: inbound from $esg only; egress '${want_egress:-none}'"
}

security_group() {
  esg=$(state security-group-id)
  gsg=$(a ec2 create-security-group --group-name "$GPU_SECURITY_GROUP" --vpc-id "$vpc" \
    --description "Receipts Glimmer VM: vLLM and SSH from the eval VM group only" \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$GPU_SECURITY_GROUP},{Key=Project,Value=falden-receipts}]" \
    --query GroupId --output text)
  echo "$gsg" > "$STATE/gpu-security-group-id"
  a ec2 revoke-security-group-egress --group-id "$gsg" \
    --ip-permissions '[{"IpProtocol":"-1","IpRanges":[{"CidrIp":"0.0.0.0/0"}]}]' > /dev/null
  a ec2 authorize-security-group-egress --group-id "$gsg" --ip-permissions "file://$VM/gpu-egress.json" > /dev/null
  a ec2 authorize-security-group-ingress --group-id "$gsg" --ip-permissions "$(group_pair "$VLLM_PORT" "$esg" "vLLM from the eval VM")" > /dev/null
  a ec2 authorize-security-group-ingress --group-id "$gsg" --ip-permissions "$(group_pair 22 "$esg" "SSH through the eval VM")" > /dev/null
  a ec2 authorize-security-group-egress --group-id "$esg" --ip-permissions "$(group_pair "$VLLM_PORT" "$gsg" "vLLM on the Glimmer VM")" > /dev/null
  a ec2 authorize-security-group-egress --group-id "$esg" --ip-permissions "$(group_pair 22 "$gsg" "SSH to the Glimmer VM")" > /dev/null
  check_gpu_sg "$gsg" "$esg" "tcp,443,443,0.0.0.0/0;"
}

verify_instance() {
  got=$(a ec2 describe-instances --instance-ids "$1" --query \
    'Reservations[0].Instances[0].[InstanceType,ImageId,MetadataOptions.HttpTokens,MetadataOptions.HttpPutResponseHopLimit,IamInstanceProfile.Arn,KeyName]' \
    --output text | tr '\t' '|')
  want="$GPU_INSTANCE_TYPE|$(state gpu-image-id)|required|1|None|$(state key-name)"
  if [ "$got" != "$want" ]; then
    a ec2 stop-instances --instance-ids "$1" > /dev/null || true
    die "instance $1 is not what was asked for (got '$got', want '$want'); it has been stopped"
  fi
}

host_keys() {
  local id="$1" out keys
  for _ in $(seq 1 80); do
    out=$(a ec2 get-console-output --instance-id "$id" --latest --query Output --output text || true)
    keys=$(printf '%s\n' "$out" | sed -n '/-----BEGIN SSH HOST KEY KEYS-----/,/-----END SSH HOST KEY KEYS-----/p' \
      | grep -E '^(ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa) ' || true)
    if [ -n "$keys" ]; then
      grep -v '^receipts-glimmer ' "$STATE/known_hosts" > "$STATE/known_hosts.part" 2> /dev/null || true
      printf '%s\n' "$keys" | awk '{print "receipts-glimmer " $1 " " $2}' >> "$STATE/known_hosts.part"
      mv "$STATE/known_hosts.part" "$STATE/known_hosts"
      say "pinned the Glimmer VM's host keys from its console"
      return 0
    fi
    sleep "$POLL_S"
  done
  die "no host keys in the console output of $id yet; rerun 'gpu.sh launch' steps later"
}

launch() {
  need_profile
  local pinned="${RECEIPTS_GPU_AMI:-}"
  if [ "$#" -gt 0 ]; then
    [ "$1" = "--ami" ] && [ "$#" -eq 2 ] || die "usage: gpu.sh launch [--ami <ami id>]"
    pinned="$2"
  fi
  [ -f "$VM/gpu-cloud-init.yaml" ] || die "no GPU user data; run 'python -m receipts.eval vm-files' first"
  preflight "$pinned"
  echo "$ami_id" > "$STATE/gpu-image-id"
  security_group
  id=$(a ec2 run-instances --cli-input-json "file://$VM/gpu-run-instances.json" --image-id "$ami_id" \
    --key-name "$(state key-name)" --security-group-ids "$gsg" --user-data "fileb://$VM/gpu-cloud-init.yaml" \
    --query 'Instances[0].InstanceId' --output text)
  case "$id" in i-*) ;; *) die "run-instances returned '$id'" ;; esac
  echo "$id" > "$STATE/gpu-instance-id"
  say "launched $id; waiting for it to run"
  a ec2 wait instance-running --instance-ids "$id"
  verify_instance "$id"
  ip=$(a ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].PrivateIpAddress' --output text)
  case "$ip" in None | "") die "$id has no private IP address" ;; esac
  echo "$ip" > "$STATE/gpu-private-ip"
  host_keys "$id"
  say "next: wait for /opt/receipts/READY on the Glimmer VM (launch.sh ssh-config, then ssh receipts-glimmer), then 'gpu.sh seal'"
}

status() {
  need_profile
  a ec2 describe-instances --instance-ids "$(state gpu-instance-id)" \
    --query 'Reservations[0].Instances[0].[InstanceId,State.Name,InstanceType,PrivateIpAddress]' --output text
}

url() { echo "RECEIPTS_GLIMMER_URL=http://$(state gpu-private-ip):$VLLM_PORT/v1"; }

seal() {
  need_profile
  gsg=$(state gpu-security-group-id)
  esg=$(state security-group-id)
  ssh -F "$STATE/ssh_config" receipts-glimmer test -f /opt/receipts/READY \
    || die "vLLM is not serving $SERVED_NAME yet (no /opt/receipts/READY); seal after it is"
  a ec2 revoke-security-group-egress --group-id "$gsg" --ip-permissions "file://$VM/gpu-egress.json" > /dev/null
  check_gpu_sg "$gsg" "$esg" ""
  say "sealed: the Glimmer VM has no egress"
}

stop() {
  need_profile
  id=$(state gpu-instance-id)
  a ec2 stop-instances --instance-ids "$id" > /dev/null
  a ec2 wait instance-stopped --instance-ids "$id"
  say "$id stopped. GPU billing stops; its $GPU_DISK_GIB GiB volume bills until terminate."
}

start() {
  need_profile
  id=$(state gpu-instance-id)
  a ec2 start-instances --instance-ids "$id" > /dev/null
  a ec2 wait instance-running --instance-ids "$id"
  verify_instance "$id"
  ip=$(state gpu-private-ip)
  for _ in $(seq 1 80); do
    if ssh -F "$STATE/ssh_config" receipts-eval curl -fsS "http://$ip:$VLLM_PORT/v1/models" 2> /dev/null \
      | grep -q "\"$SERVED_NAME\""; then
      say "vLLM serves $SERVED_NAME at http://$ip:$VLLM_PORT/v1 (from the eval VM)"
      return 0
    fi
    sleep "$POLL_S"
  done
  die "vLLM on $id did not come back; ssh receipts-glimmer and read docker logs glimmer"
}

terminate() {
  need_profile
  id=$(state gpu-instance-id)
  [ "${1:-}" = "--yes" ] && [ "${2:-}" = "$id" ] \
    || die "terminate deletes the Glimmer VM and its volume for good; to go ahead, run: gpu.sh terminate --yes $id"
  gsg=$(state gpu-security-group-id)
  esg=$(state security-group-id)
  a ec2 modify-instance-attribute --instance-id "$id" --no-disable-api-termination
  a ec2 terminate-instances --instance-ids "$id" > /dev/null
  a ec2 wait instance-terminated --instance-ids "$id"
  a ec2 revoke-security-group-egress --group-id "$esg" --ip-permissions "$(group_pair "$VLLM_PORT" "$gsg")" > /dev/null
  a ec2 revoke-security-group-egress --group-id "$esg" --ip-permissions "$(group_pair 22 "$gsg")" > /dev/null
  a ec2 delete-security-group --group-id "$gsg"
  grep -v '^receipts-glimmer ' "$STATE/known_hosts" > "$STATE/known_hosts.part" 2> /dev/null || true
  mv "$STATE/known_hosts.part" "$STATE/known_hosts"
  rm -f "$STATE/gpu-instance-id" "$STATE/gpu-security-group-id" "$STATE/gpu-private-ip" "$STATE/gpu-image-id"
  say "$id terminated, with its volume; the eval VM's group no longer reaches it"
}

case "${1:-}" in
  launch)
    shift
    launch "$@"
    ;;
  status) status ;;
  url) url ;;
  seal) seal ;;
  stop) stop ;;
  start) start ;;
  terminate)
    shift
    terminate "$@"
    ;;
  *)
    echo "usage: gpu.sh launch [--ami <ami id>] | status | url | seal | stop | start | terminate --yes <id>" >&2
    exit 2
    ;;
esac
