#!/usr/bin/env bash
# The Receipts eval VM, for the integrator: launch, status, ssh-config, stop, start, resize, host-keys, terminate.
#
# Every AWS call here runs with the caller's own profile (AWS_PROFILE) in us-east-1; the eval lane wrote this file and
# never runs it. Inputs, all under eval/vm/: launch.env, run-instances.json, egress.json and cloud-init.yaml.gz (made
# by `python -m receipts.eval vm-files` and `cloud-init`). State: runs/vm/ (not in git).
#
# Access is SSH only (REG-562): 22/tcp from the founder's current public IP as one /32, read from two independent
# services when the VM launches or starts and printed before anything changes; a new key pair whose private key is
# written to $RECEIPTS_KEY_DIR (a folder of yours with mode 700) with mode 600 and never printed; no instance role; no
# Systems Manager (the boot script removes the agent). SSH pins the host keys the VM prints to its console at first
# boot, so nothing is trusted on first use. Sizes: the VM launches as an m7i.16xlarge for R24 and the label runs
# (REG-568), then `resize m7i.8xlarge` for the pilot and the rehearsal (REG-556), and `resize m7i.16xlarge` again for
# the 11/1 final (REG-563); a resize needs the VM stopped and keeps its volume and pulled images, and launch and resize
# print the size and the vCPUs it needs first. Between runs: copy the results off, `stop`; before the next run,
# `start`, which runs /opt/receipts/vm-check.sh over SSH.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
VM="$ROOT/eval/vm"
STATE="${RECEIPTS_EVAL_STATE:-$ROOT/runs/vm}"
POLL_S="${RECEIPTS_EVAL_POLL_S:-15}"
# shellcheck source=/dev/null
source "$VM/launch.env"

die() { echo "launch.sh: $*" >&2; exit 1; }
say() { echo "launch.sh: $*"; }
a() { aws --region "$REGION" "$@"; }
need_profile() { [ -n "${AWS_PROFILE:-}" ] || die "set AWS_PROFILE to the integrator's Falden profile first"; }
state() { [ -f "$STATE/$1" ] || die "no $1 is recorded in $STATE; run 'launch.sh launch' first"; cat "$STATE/$1"; }
perm() { stat -c %a "$1" 2> /dev/null || stat -f %Lp "$1"; }
owner() { stat -c %u "$1" 2> /dev/null || stat -f %u "$1"; }

key_dir() {
  local d="${RECEIPTS_KEY_DIR:-}"
  [ -n "$d" ] || die "set RECEIPTS_KEY_DIR to a private folder of yours (mode 700) for the VM's private key"
  [ -d "$d" ] || die "RECEIPTS_KEY_DIR $d is not a folder"
  [ "$(perm "$d")" = "700" ] || die "RECEIPTS_KEY_DIR $d has mode $(perm "$d"); it must be 700"
  [ "$(owner "$d")" = "$(id -u)" ] || die "RECEIPTS_KEY_DIR $d is not yours"
  case "$(cd "$d" && pwd -P)/" in
    "$(cd "$ROOT" && pwd -P)/"*) die "RECEIPTS_KEY_DIR is inside the repository; keep the key out of it" ;;
  esac
  (cd "$d" && pwd -P)
}

public_ipv4() {
  # A globally reachable IPv4 address: four decimal octets of 0-255 with no leading zero, outside every block of
  # IANA's special-purpose registry: this network, private, shared (CGNAT), loopback, link-local, IETF protocol
  # assignments, the three documentation nets, the 6to4 relay, benchmarking, multicast, reserved and broadcast.
  local ip="$1" o octet='(0|[1-9][0-9]{0,2})'
  local dotted="^$octet\\.$octet\\.$octet\\.$octet\$"
  [[ $ip =~ $dotted ]] || return 1
  local o1=${BASH_REMATCH[1]} o2=${BASH_REMATCH[2]} o3=${BASH_REMATCH[3]} o4=${BASH_REMATCH[4]}
  for o in "$o1" "$o2" "$o3" "$o4"; do [ "$o" -le 255 ] || return 1; done
  case "$o1" in 0 | 10 | 127) return 1 ;; esac                               # this network, private, loopback
  [ "$o1" -ge 224 ] && return 1                                               # multicast, reserved, broadcast
  [ "$o1" -eq 100 ] && [ "$o2" -ge 64 ] && [ "$o2" -le 127 ] && return 1      # 100.64/10 shared
  [ "$o1" -eq 169 ] && [ "$o2" -eq 254 ] && return 1                          # 169.254/16 link-local
  [ "$o1" -eq 172 ] && [ "$o2" -ge 16 ] && [ "$o2" -le 31 ] && return 1       # 172.16/12 private
  [ "$o1" -eq 192 ] && [ "$o2" -eq 168 ] && return 1                          # 192.168/16 private
  [ "$o1" -eq 192 ] && [ "$o2" -eq 0 ] && [ "$o3" -eq 0 ] && return 1         # 192.0.0/24 IETF assignments
  [ "$o1" -eq 192 ] && [ "$o2" -eq 0 ] && [ "$o3" -eq 2 ] && return 1         # 192.0.2/24 documentation
  [ "$o1" -eq 192 ] && [ "$o2" -eq 88 ] && [ "$o3" -eq 99 ] && return 1       # 192.88.99/24 6to4 relay
  [ "$o1" -eq 198 ] && [ "$o2" -ge 18 ] && [ "$o2" -le 19 ] && return 1       # 198.18/15 benchmarking
  [ "$o1" -eq 198 ] && [ "$o2" -eq 51 ] && [ "$o3" -eq 100 ] && return 1      # 198.51.100/24 documentation
  [ "$o1" -eq 203 ] && [ "$o2" -eq 0 ] && [ "$o3" -eq 113 ] && return 1       # 203.0.113/24 documentation
  return 0
}

my_ip() {
  # The founder's current public IPv4, from two independent services that must agree.
  local seen="" ip src
  for src in $IP_SOURCES; do
    ip=$(curl -fsS --max-time 10 "$src" | tr -d '[:space:]') || die "could not read the public IP from $src"
    public_ipv4 "$ip" || die "$src answered '$ip', which is not a public IPv4 address"
    [ -z "$seen" ] || [ "$seen" = "$ip" ] || die "the IP services disagree ($seen, $ip); retry from one network"
    seen="$ip"
  done
  echo "$seen"
}

preflight() {
  account=$(a sts get-caller-identity --query Account --output text)
  say "account $account, region $REGION, profile $AWS_PROFILE"
  quota=$(a service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query Quota.Value --output text)
  [ "${quota%.*}" -ge "$LAUNCH_VCPU" ] \
    || die "the On-Demand Standard vCPU quota is $quota, below the $LAUNCH_VCPU that $LAUNCH_INSTANCE_TYPE needs"
  got=$(a ec2 describe-images --image-ids "$AMI_ID" --owners "$AMI_OWNER" \
    --query 'Images[0].[Name,State,Architecture,RootDeviceName,RootDeviceType]' --output text | tr '\t' '|')
  [ "$got" = "$AMI_IDENTITY" ] || die "$AMI_ID from owner $AMI_OWNER is not the pinned image: got '$got', want '$AMI_IDENTITY'"
  live=$(a ec2 describe-instances --filters "Name=tag:Name,Values=$NAME" \
    "Name=instance-state-name,Values=pending,running,stopping,stopped" \
    --query 'Reservations[].Instances[].InstanceId' --output text)
  [ -z "$live" ] || die "an instance named $NAME already exists ($live); use start, stop or terminate"
  vpc=$(a ec2 describe-vpcs --filters Name=isDefault,Values=true --query 'Vpcs[0].VpcId' --output text)
  case "$vpc" in vpc-*) ;; *) die "no default VPC in $REGION" ;; esac
}

ssh_rule() {
  printf '[{"IpProtocol":"tcp","FromPort":%s,"ToPort":%s,"IpRanges":[{"CidrIp":"%s/32","Description":"founder SSH, REG-562"}]}]' \
    "$SSH_PORT" "$SSH_PORT" "$1"
}

check_sg() {
  # Exactly one inbound rule, 22/tcp from the founder's /32; egress TCP 443 and 80 to anywhere, plus 8000 and 22 to
  # the Glimmer VM's group when gpu.sh has made one.
  local sg="$1" ip="$2" ingress want_egress egress gsg
  ingress=$(a ec2 describe-security-groups --group-ids "$sg" --query \
    'SecurityGroups[0].IpPermissions[].[IpProtocol,FromPort,ToPort,join(`,`,IpRanges[].CidrIp),length(Ipv6Ranges),length(PrefixListIds),length(UserIdGroupPairs)]' \
    --output text | tr '\t\n' ',;')
  [ "$ingress" = "tcp,$SSH_PORT,$SSH_PORT,$ip/32,0,0,0;" ] \
    || die "security group $sg allows inbound '$ingress'; want only $SSH_PORT/tcp from $ip/32"
  want_egress="tcp,443,443,0.0.0.0/0,;tcp,80,80,0.0.0.0/0,;"
  if [ -f "$STATE/gpu-security-group-id" ]; then
    gsg=$(cat "$STATE/gpu-security-group-id")
    want_egress="tcp,22,22,,$gsg;tcp,443,443,0.0.0.0/0,;tcp,80,80,0.0.0.0/0,;tcp,8000,8000,,$gsg;"
  fi
  egress=$(a ec2 describe-security-groups --group-ids "$sg" --query \
    'SecurityGroups[0].IpPermissionsEgress[].[IpProtocol,FromPort,ToPort,join(`,`,IpRanges[].CidrIp),join(`,`,UserIdGroupPairs[].GroupId)]' \
    --output text | LC_ALL=C sort | tr '\t\n' ',;')
  [ "$egress" = "$want_egress" ] || die "security group $sg allows egress '$egress', want '$want_egress'"
  say "security group $sg: inbound $SSH_PORT/tcp from $ip/32 only; egress as planned"
}

security_group() {
  local ip="$1"
  sg=$(a ec2 describe-security-groups --filters "Name=group-name,Values=$SECURITY_GROUP" "Name=vpc-id,Values=$vpc" \
    --query 'SecurityGroups[0].GroupId' --output text)
  [ -z "$sg" ] || [ "$sg" = "None" ] || die "security group $SECURITY_GROUP already exists ($sg); terminate cleans it up"
  sg=$(a ec2 create-security-group --group-name "$SECURITY_GROUP" --vpc-id "$vpc" \
    --description "Receipts eval VM: SSH from the founder IP only, egress TCP 443 and 80" \
    --tag-specifications "ResourceType=security-group,Tags=[{Key=Name,Value=$SECURITY_GROUP},{Key=Project,Value=falden-receipts}]" \
    --query GroupId --output text)
  mkdir -p "$STATE"
  echo "$sg" > "$STATE/security-group-id"
  a ec2 revoke-security-group-egress --group-id "$sg" \
    --ip-permissions '[{"IpProtocol":"-1","IpRanges":[{"CidrIp":"0.0.0.0/0"}]}]' > /dev/null
  a ec2 authorize-security-group-egress --group-id "$sg" --ip-permissions "file://$VM/egress.json" > /dev/null
  a ec2 authorize-security-group-ingress --group-id "$sg" --ip-permissions "$(ssh_rule "$ip")" > /dev/null
  echo "$ip" > "$STATE/ssh-from"
  check_sg "$sg" "$ip"
}

new_key_pair() {
  local kd="$1" name file
  name="$KEY_PREFIX-$(date -u +%Y%m%dT%H%M%SZ)"
  file="$kd/$name.pem"
  [ ! -e "$file" ] || die "$file already exists; a new key pair never overwrites a key"
  if a ec2 describe-key-pairs --key-names "$name" > /dev/null 2>&1; then die "key pair $name already exists in AWS"; fi
  (
    umask 077
    a ec2 create-key-pair --key-name "$name" --key-type ed25519 --key-format pem \
      --query KeyMaterial --output text > "$file.part"
  ) || { rm -f "$file.part"; die "create-key-pair failed"; }
  grep -q 'PRIVATE KEY' "$file.part" || { rm -f "$file.part"; die "create-key-pair returned no private key"; }
  chmod 600 "$file.part"
  mv "$file.part" "$file"
  [ "$(perm "$file")" = "600" ] || die "$file has mode $(perm "$file"); it must be 600"
  echo "$name" > "$STATE/key-name"
  echo "$file" > "$STATE/key-file"
  say "new key pair $name; its private key is $file (mode 600), and nothing printed it"
}

verify_instance() {
  local want_type
  want_type=$(cat "$STATE/instance-type" 2> /dev/null || echo "$LAUNCH_INSTANCE_TYPE")
  got=$(a ec2 describe-instances --instance-ids "$1" --query \
    'Reservations[0].Instances[0].[InstanceType,ImageId,MetadataOptions.HttpTokens,MetadataOptions.HttpPutResponseHopLimit,IamInstanceProfile.Arn,KeyName]' \
    --output text | tr '\t' '|')
  want="$want_type|$AMI_ID|required|1|None|$(state key-name)"
  if [ "$got" != "$want" ]; then
    a ec2 stop-instances --instance-ids "$1" > /dev/null || true
    die "instance $1 is not what was asked for (got '$got', want '$want'); it has been stopped"
  fi
}

host_keys() {
  # The VM's SSH host keys, as cloud-init prints them to the console at first boot; written as known_hosts for the
  # alias receipts-eval, so a changed key or a changed address never connects silently.
  local id="$1" out keys
  for _ in $(seq 1 80); do
    out=$(a ec2 get-console-output --instance-id "$id" --latest --query Output --output text || true)
    keys=$(printf '%s\n' "$out" | sed -n '/-----BEGIN SSH HOST KEY KEYS-----/,/-----END SSH HOST KEY KEYS-----/p' \
      | grep -E '^(ssh-ed25519|ecdsa-sha2-nistp256|ssh-rsa) ' || true)
    if [ -n "$keys" ]; then
      printf '%s\n' "$keys" | awk '{print "receipts-eval " $1 " " $2}' > "$STATE/known_hosts"
      say "pinned $(grep -c . "$STATE/known_hosts") host keys from the console of $id"
      return 0
    fi
    sleep "$POLL_S"
  done
  die "no host keys in the console output of $id yet; run 'launch.sh host-keys' once the VM has booted"
}

write_ssh_config() {
  local id ip
  id=$(state instance-id)
  ip=$(a ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].PublicIpAddress' --output text)
  case "$ip" in None | "") die "$id has no public IP address (is it running?)" ;; esac
  cat > "$STATE/ssh_config" << EOF
Host receipts-eval
  HostName $ip
  User ubuntu
  IdentityFile $(state key-file)
  IdentitiesOnly yes
  HostKeyAlias receipts-eval
  UserKnownHostsFile $STATE/known_hosts
  StrictHostKeyChecking yes
EOF
  if [ -f "$STATE/gpu-private-ip" ]; then
    cat >> "$STATE/ssh_config" << EOF

Host receipts-glimmer
  HostName $(cat "$STATE/gpu-private-ip")
  User ubuntu
  ProxyJump receipts-eval
  IdentityFile $(state key-file)
  IdentitiesOnly yes
  HostKeyAlias receipts-glimmer
  UserKnownHostsFile $STATE/known_hosts
  StrictHostKeyChecking yes
EOF
  fi
}

launch() {
  need_profile
  [ -f "$VM/cloud-init.yaml.gz" ] || die "no user data; run 'python -m receipts.eval cloud-init' first"
  kd=$(key_dir)
  ip=$(my_ip)
  say "SSH will be open to $ip/32 only ($SSH_PORT/tcp), read just now from: $IP_SOURCES"
  say "launching $LAUNCH_INSTANCE_TYPE, which needs $LAUNCH_VCPU vCPUs of the On-Demand Standard quota (REG-568: R24 and the label runs; 'launch.sh resize $PILOT_INSTANCE_TYPE' before the pilot)"
  preflight
  mkdir -p "$STATE"
  security_group "$ip"
  new_key_pair "$kd"
  id=$(a ec2 run-instances --cli-input-json "file://$VM/run-instances.json" --key-name "$(state key-name)" \
    --security-group-ids "$sg" --user-data "fileb://$VM/cloud-init.yaml.gz" --query 'Instances[0].InstanceId' --output text)
  case "$id" in i-*) ;; *) die "run-instances returned '$id'" ;; esac
  echo "$id" > "$STATE/instance-id"
  echo "$LAUNCH_INSTANCE_TYPE" > "$STATE/instance-type"
  say "launched $id; waiting for it to run"
  a ec2 wait instance-running --instance-ids "$id"
  verify_instance "$id"
  host_keys "$id"
  write_ssh_config
  say "next: wait for /opt/receipts/BOOT_OK: ssh -F $STATE/ssh_config receipts-eval ls /opt/receipts"
}

status() {
  need_profile
  a ec2 describe-instances --instance-ids "$(state instance-id)" \
    --query 'Reservations[0].Instances[0].[InstanceId,State.Name,InstanceType,PublicIpAddress,LaunchTime]' --output text
}

ssh_config() {
  need_profile
  write_ssh_config
  cat "$STATE/ssh_config"
}

update_ssh_rule() {
  # The founder's IP may have changed since the last start: move the one inbound rule to the current /32.
  local sg ip old
  sg=$(state security-group-id)
  ip="$1"
  old=$(state ssh-from)
  if [ "$old" != "$ip" ]; then
    a ec2 authorize-security-group-ingress --group-id "$sg" --ip-permissions "$(ssh_rule "$ip")" > /dev/null
    a ec2 revoke-security-group-ingress --group-id "$sg" --ip-permissions "$(ssh_rule "$old")" > /dev/null
    echo "$ip" > "$STATE/ssh-from"
    say "moved SSH from $old/32 to $ip/32"
  fi
  check_sg "$sg" "$ip"
}

stop() {
  need_profile
  id=$(state instance-id)
  a ec2 stop-instances --instance-ids "$id" > /dev/null
  a ec2 wait instance-stopped --instance-ids "$id"
  say "$id stopped. Compute and its public IPv4 address stop billing; the 500 GiB volume bills until terminate."
}

start() {
  need_profile
  id=$(state instance-id)
  ip=$(my_ip)
  say "SSH will be open to $ip/32 only ($SSH_PORT/tcp), read just now from: $IP_SOURCES"
  update_ssh_rule "$ip"
  a ec2 start-instances --instance-ids "$id" > /dev/null
  a ec2 wait instance-running --instance-ids "$id"
  verify_instance "$id"
  write_ssh_config
  out=$(ssh -F "$STATE/ssh_config" receipts-eval sudo bash /opt/receipts/vm-check.sh) \
    || die "vm-check failed on $id: $out"
  case "$out" in
    *"vm ok"*) say "vm-check: $out" ;;
    *) die "vm-check on $id said: $out" ;;
  esac
}

resize() {
  need_profile
  local to="${1:-}" vcpu state_name
  case "$to" in
    "$PILOT_INSTANCE_TYPE") vcpu="$PILOT_VCPU" ;;
    "$FINAL_INSTANCE_TYPE") vcpu="$FINAL_VCPU" ;;
    "$LAUNCH_INSTANCE_TYPE") vcpu="$LAUNCH_VCPU" ;;
    *) die "resize to $PILOT_INSTANCE_TYPE (the pilot, the rehearsal) or $FINAL_INSTANCE_TYPE (R24 and the labels, the 11/1 final), not '$to'" ;;
  esac
  say "resizing to $to, which needs $vcpu vCPUs of the On-Demand Standard quota"
  id=$(state instance-id)
  state_name=$(a ec2 describe-instances --instance-ids "$id" --query 'Reservations[0].Instances[0].State.Name' --output text)
  [ "$state_name" = "stopped" ] || die "$id is $state_name; stop it first"
  quota=$(a service-quotas get-service-quota --service-code ec2 --quota-code L-1216C47A --query Quota.Value --output text)
  [ "${quota%.*}" -ge "$vcpu" ] || die "the On-Demand Standard vCPU quota is $quota; $to needs $vcpu"
  a ec2 modify-instance-attribute --instance-id "$id" --instance-type "{\"Value\": \"$to\"}"
  echo "$to" > "$STATE/instance-type"
  say "$id is now $to; 'launch.sh start' before the run"
}

terminate() {
  need_profile
  id=$(state instance-id)
  [ "${1:-}" = "--yes" ] && [ "${2:-}" = "$id" ] \
    || die "terminate deletes the VM and its volume for good; to go ahead, run: launch.sh terminate --yes $id"
  [ ! -f "$STATE/gpu-instance-id" ] || die "the Glimmer VM still exists; run 'gpu.sh terminate' first"
  a ec2 modify-instance-attribute --instance-id "$id" --no-disable-api-termination
  a ec2 terminate-instances --instance-ids "$id" > /dev/null
  a ec2 wait instance-terminated --instance-ids "$id"
  a ec2 delete-security-group --group-id "$(state security-group-id)"
  a ec2 delete-key-pair --key-name "$(state key-name)"
  say "$id terminated, with its volume; key pair $(state key-name) deleted in AWS; $(state key-file) is still on disk"
  rm -f "$STATE/instance-id" "$STATE/security-group-id" "$STATE/key-name" "$STATE/key-file" "$STATE/ssh-from" \
    "$STATE/instance-type" "$STATE/known_hosts" "$STATE/ssh_config"
}

case "${1:-}" in
  launch) launch ;;
  status) status ;;
  ssh-config) ssh_config ;;
  host-keys)
    need_profile
    host_keys "$(state instance-id)"
    ;;
  stop) stop ;;
  start) start ;;
  resize)
    shift
    resize "$@"
    ;;
  terminate)
    shift
    terminate "$@"
    ;;
  *)
    echo "usage: launch.sh launch | status | ssh-config | host-keys | stop | start | resize <type> | terminate --yes <id>" >&2
    exit 2
    ;;
esac
