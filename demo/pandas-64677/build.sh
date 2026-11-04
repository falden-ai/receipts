#!/usr/bin/env bash
# Run on the eval VM under the founder's shared heavy lock. The loopback registry must already be running.
set -euo pipefail
DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RECEIPTS_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
PANDAS_CLONE="${PANDAS_CLONE:?set PANDAS_CLONE to a local pandas clone that contains the pinned base}"
DEMO_TAG=127.0.0.1:5000/receipts-demo/pandas-64677:build
DEMO_BUILD_DIR="$(mktemp -d "$DEMO_DIR/.build-XXXXXXXX")"
trap 'rm -rf "$DEMO_BUILD_DIR"' EXIT
cd "$RECEIPTS_ROOT"

command -v docker >/dev/null
docker info >/dev/null
# Validate the pinned diffs before any build or push.
UV_OFFLINE=1 uv run --frozen python - "$DEMO_DIR" <<'PY'
import sys
from pathlib import Path
from receipts.eval.demo import load_pins
load_pins(Path(sys.argv[1]))
PY

BASE_SHA=82315e7fb597d53f797f62d2a83f9ee9dcad5042
WHATSNEW=doc/source/whatsnew/v3.1.0.rst
# .gitattributes export-ignores doc/. Restore exactly this base file, with its terminating newline.
git -C "$PANDAS_CLONE" archive --format=tar \
    --add-virtual-file="$WHATSNEW:$(git -C "$PANDAS_CLONE" show "$BASE_SHA:$WHATSNEW")"$'\n' \
    "$BASE_SHA" > "$DEMO_BUILD_DIR/pandas-82315e7.with-whatsnew.tar"
cp "$DEMO_DIR/Dockerfile" "$DEMO_DIR/requirements.lock" "$DEMO_BUILD_DIR/"
UV_OFFLINE=1 uv run --frozen python - "$DEMO_DIR" "$DEMO_BUILD_DIR" <<'PY'
import hashlib
import json
import sys
from pathlib import Path
pins_dir, context = map(Path, sys.argv[1:])
pins = json.loads((pins_dir / "pins.json").read_text())
if (context / "Dockerfile").read_text().splitlines()[0] != "FROM " + pins["base_image"]["ref"]:
    raise SystemExit("Dockerfile FROM disagrees with pins.json")
for name, expected in (
    ("pandas-82315e7.with-whatsnew.tar", pins["source_tarball"]["with_whatsnew"]["sha256"]),
    ("requirements.lock", pins["requirements_lock"]["sha256"]),
):
    actual = hashlib.sha256((context / name).read_bytes()).hexdigest()
    if actual != expected:
        raise SystemExit(f"{name}: {actual} != pinned {expected}")
    print(f"{name}: sha256 {actual}")
PY

docker build --platform linux/amd64 --tag "$DEMO_TAG" "$DEMO_BUILD_DIR"
# Test only a local image. The test never pulls and the container has no network.
RECEIPTS_RUN_DOCKER=1 RECEIPTS_DEMO_IMAGE="$DEMO_TAG" \
RECEIPTS_DEMO_EVIDENCE="$DEMO_BUILD_DIR/smoke.json" UV_OFFLINE=1 \
    uv run --frozen pytest tests/eval/test_eval_demo.py -k docker \
        -o 'markers=docker: requires an opted-in local Docker daemon and the shared heavy lock'
docker push "$DEMO_TAG"
docker inspect --format '{{index .RepoDigests 0}}' "$DEMO_TAG" > "$DEMO_BUILD_DIR/image-ref.txt"

# Publish the measured pins atomically only after every smoke check and the loopback push pass.
UV_OFFLINE=1 uv run --frozen python - "$DEMO_DIR" "$DEMO_BUILD_DIR" <<'PY'
import datetime as dt
import json
import re
import sys
from pathlib import Path
from receipts.contracts import Target
from receipts.eval.hashing import write_text_atomic

pins_dir, context = map(Path, sys.argv[1:])
path = pins_dir / "pins.json"
pins = json.loads(path.read_text())
evidence = json.loads((context / "smoke.json").read_text())
ref = (context / "image-ref.txt").read_text().strip()
if not re.fullmatch(r"127\.0\.0\.1:5000/receipts-demo/pandas-64677@sha256:[0-9a-f]{64}", ref):
    raise SystemExit(f"unexpected VM registry reference: {ref!r}")
Target.model_validate({**json.loads((pins_dir / "target.template.json").read_text()), "image": ref})
date = dt.datetime.now(dt.UTC).date().isoformat()
pins["image"].update(ref=ref, date_read=date)
pins["apt"].update(versions=evidence["apt_versions"], date_read=date)
history = pins.setdefault("builds", [])
evidence.update(
    date_read=date, image_ref=ref,
    source_tarball_sha256=pins["source_tarball"]["with_whatsnew"]["sha256"],
    requirements_lock_sha256=pins["requirements_lock"]["sha256"],
    base_image_ref=pins["base_image"]["ref"],
)
for key in ("suite_counts", "apt_versions", "source_tarball_sha256", "requirements_lock_sha256", "base_image_ref"):
    if history and history[-1][key] != evidence[key]:
        raise SystemExit(f"{key} changed from the preceding build; pins were not updated")
history.append(evidence)
write_text_atomic(path, json.dumps(pins, indent=2) + "\n")
print(f"pinned {ref}")
print(json.dumps(evidence, indent=2))
PY
