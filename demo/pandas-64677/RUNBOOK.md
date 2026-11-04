# pandas #64677 regression demo

Run on the eval VM with gVisor `runsc` installed. Tier B refuses `runc`. Hold the founder's shared heavy lock while
building or running the Docker smoke test.

The required catch fails on the PR patch in five of five runs and passes on base in five of five runs. A separate
fixcheck then replays that same repro on base plus PR plus the later fix. Passing on the fix does not establish a
regression catch. The October 31 run found 9 blockers of that kind, so the Tier B fallback set on October 30 was
not needed.

## Inputs and evidence

All input provenance and read dates are in `pins.json`. The base is the merge's sole parent,
`82315e7fb597d53f797f62d2a83f9ee9dcad5042`; the PR API's older base field is deliberately unused. The replay's
`patch_sha` is merge `d27e2f320150e3b064f3256a8bada7478e89da53`.

`patch.diff` is the byte-for-byte git diff from base to that merge. `fix.diff` is one combined diff against base:
base plus PR plus the independent fix from merge `427d6bd2c76219f1896848f243936e919f67972b`, parent
`5c4177adad326483cadae6b77422bef66a98545c`. Plain application to the PR merge failed on context. `git apply --3way`
applied cleanly, with no conflicts and no hand merge. Every added and removed line in the fix's `groupby.py` hunks
is preserved; one context line differs. The combined diff is the later fix adapted onto the PR, not the complete
tree at the later merge.

The founder-approved archive is `git archive` at base plus exactly the base version of
`doc/source/whatsnew/v3.1.0.rst`. pandas export-ignores `doc/`, but both diffs touch that file. The resulting tarball
SHA-256 is `f10fd12b16d923815aa05b0fd4f586ae67179bb00a3034980283263fe9c3fd07`. The exact archive command lives in
`build.sh` and `pins.json`; it preserves the file's final newline. The plain archive hash is retained for provenance
and is not used. No other ignored files are restored; groupby has no non-Python test-data files at base.

Python 3.12 matches pandas CI at base. The hashed lock follows the base `pyproject.toml` ranges, resolved with
`--exclude-newer 2026-03-19`, because pandas treats warnings as errors. It includes build requirements, test
requirements and their dependencies. Keep its prepared bytes. The image uses a non-editable install only to build
extensions, copies every wheel extension and missing generated file into `/workspace`, and uninstalls pandas.
The copy script is written out with `printf`, never a heredoc: the eval VM's docker has no buildx, and its legacy
builder runs a `RUN` heredoc with an empty body, so the copy never ran there and the image had no extension
(FAL-643). The build then imports pandas from `/workspace` alone and fails if it cannot.
The wheel's `pandas/pyproject.toml`, the root config pandas installs for `pandas.test()`, is not copied. In
`/workspace/pandas` it became pytest's rootdir, so every case id read `tests/...` rather than `pandas/tests/...` and
every SK1 replay reported its expected case never ran (FAL-643). The build now fails unless pytest, run from
`/workspace`, collects ids under `pandas/`, and the docker smoke test checks every groupby case's classname and file.
There is no editable rebuild hook. This base has no export-substituted `_version.py`; the generated
`_version_meson.py` is copied from the wheel. A `/src` build without Git metadata may report `0+unknown`; the base
commit and source archive pins identify the code.

Only the base source, Dockerfile and hashed lock enter the Docker build context. Neither diff, target text nor fix
metadata enters the image. The target template contains only the merged PR's title, body and patch, with no issue
text, later fix or future commit messages. `load_pins` checks both diff hashes before producing a target.

Build-phase network access is limited to:

- Docker Hub for the pinned `python:3.12-slim` base and pinned `registry:2` image.
- Debian apt mirrors for `build-essential`, `git` and their dependencies.
- PyPI for the hash-checked dependencies. Installing `/src` uses `--no-build-isolation --no-deps`.

Source comes from a local clone of public pandas, never a git host inside the container. Building may require
network; smoke tests and Receipts run containers use `--network=none`. The only image push is to the VM's loopback
registry, never a public registry.

## Eval VM commands, in order

Run from the Receipts repository root. Use the already-synced virtualenv. Set `PANDAS_CLONE` to a local pandas
clone containing the pinned base. On October 31 the VM used its own clone of public pandas, and `build.sh` checked
the source archive's hash against `pins.json`. Do not fetch during the run.

```bash
export PANDAS_CLONE=/path/to/pandas
mkdir -p runs/demo
REGISTRY_IMAGE="$(UV_OFFLINE=1 uv run --frozen python -c 'import json; print(json.load(open("demo/pandas-64677/pins.json"))["registry_image"]["ref"])')"
docker pull "$REGISTRY_IMAGE"
docker run -d --name receipts-demo-registry --restart unless-stopped \
  -p 127.0.0.1:5000:5000 "$REGISTRY_IMAGE"
```

Build on this VM, under the held shared heavy lock. `build.sh` checks the source and lock hashes, builds, runs the
offline smoke test, pushes to loopback, reads the VM RepoDigest and atomically records measurements in `pins.json`.
Run it twice. The second run refuses changes in the source hash, lock hash, base image pin, installed apt versions
or groupby suite counts. Each push is pinned from the VM's own registry.

```bash
bash demo/pandas-64677/build.sh
bash demo/pandas-64677/build.sh
```

Alternatively, when a smoke-tested image was built on the PC, save it there and load it on this VM. This replaces
the two VM builds only when their recorded evidence accompanies the image. Run the VM smoke test again, push and
read the digest here; a push may recompress layers, so a digest read on the PC is not the VM pin.

```bash
# On the PC, after two successful builds:
docker save -o pandas-64677.image.tar 127.0.0.1:5000/receipts-demo/pandas-64677:build
# Transfer that file using the founder's existing process, then on the eval VM:
docker load -i pandas-64677.image.tar
RECEIPTS_RUN_DOCKER=1 RECEIPTS_DEMO_IMAGE=127.0.0.1:5000/receipts-demo/pandas-64677:build \
  UV_OFFLINE=1 uv run --frozen pytest tests/eval/test_eval_demo.py -k docker \
  -o 'markers=docker: requires an opted-in local Docker daemon and the shared heavy lock'
docker push 127.0.0.1:5000/receipts-demo/pandas-64677:build
```

After either path, pin the reference read on this VM. No local image ID can be used as a Target image: a repository
name before `@sha256:` is required.

```bash
DEMO_IMAGE="$(docker inspect --format '{{index .RepoDigests 0}}' 127.0.0.1:5000/receipts-demo/pandas-64677:build)"
export DEMO_IMAGE
UV_OFFLINE=1 uv run --frozen python - <<'PY'
import datetime as dt
import json
import os
import re
from pathlib import Path
from receipts.eval.hashing import write_text_atomic
ref = os.environ["DEMO_IMAGE"]
if not re.fullmatch(r"127\.0\.0\.1:5000/receipts-demo/pandas-64677@sha256:[0-9a-f]{64}", ref):
    raise SystemExit("unexpected loopback registry reference")
p = Path("demo/pandas-64677/pins.json")
pins = json.loads(p.read_text())
pins["image"].update(ref=ref, date_read=dt.datetime.now(dt.UTC).date().isoformat())
write_text_atomic(p, json.dumps(pins, indent=2) + "\n")
PY
UV_OFFLINE=1 uv run --frozen receipts eval demo target \
  --pins demo/pandas-64677 --image "$DEMO_IMAGE" --out runs/demo/pandas-64677.jsonl
```

Screen hygiene: have `FALDEN_META_API_KEY` set in the caller's environment before recording. Never echo it, use
`set -x`, dump the environment or put its value in a command, fixture or log. The run's model client reads only that
exact key. The sandbox receives no model credentials. The run reaches the configured model API on the host;
repository code and repros remain offline inside gVisor.

A Tier B target with a pinned image and no checkout copies `/workspace` from that image. Tier A image resolvers
work as before, and no fallback checkout callable is needed. This has been on main since October 30, and the
October 31 run used it.

```bash
UV_OFFLINE=1 uv run --frozen receipts run \
  --targets runs/demo/pandas-64677.jsonl --arms demo/arms.json --ledger-dir runs/demo
UV_OFFLINE=1 uv run --frozen receipts ledger rollup --ledger runs/demo/spark.jsonl
# Set this to the regression-catch blocker id in the ledger, not an unchecked or flaky finding.
FINDING_ID='<finding id>'
UV_OFFLINE=1 uv run --frozen receipts replay "$FINDING_ID" --ledger runs/demo/spark.jsonl
UV_OFFLINE=1 uv run --frozen receipts eval demo fixcheck \
  --target runs/demo/pandas-64677.jsonl --ledger runs/demo/spark.jsonl --finding "$FINDING_ID" \
  --fix demo/pandas-64677/fix.diff --out runs/demo/fixcheck.json
```

Fixcheck opens the ledger read-only through the Ledger interface, selects the latest state of that finding,
refuses a non-blocker or mismatched replay pins, prepares `Side.GOLD` with the combined fix diff and runs the repro
five times with fresh containers, seeds 0 through 4 and its expected cases. Each replay has a 900 second timeout.
Exit codes: 0 for five PASS outcomes; 1 for any other outcome; 2 for bad input. It writes all five results, the fix
diff hash and the prepared image digest. It does not rewrite the blocker or replace its original oracle.

## Docker evidence from the October 31 VM builds

Both builds ran on the eval VM on October 31, and the Docker smoke test then passed; `pins.json.builds` holds their
measurements. The smoke test is `test_docker_pandas_base_import_suite_and_both_diffs` in
`tests/eval/test_eval_demo.py`; it skips unless `RECEIPTS_RUN_DOCKER=1` and a daemon is reachable. It never pulls.
It checks image environment names and values with `refused_env_names`, copies the baked tree, checks both diffs
with host `git apply --check`, and seeds a fresh owned tmpfs before importing and testing as UID/GID 65534 under
the sandbox's limits, read-only root and disabled network.

Before a fix run, smoke-check git in the built image:

```sh
docker run --rm --pull=never --network=none --read-only --user=65534:65534 "$DEMO_IMAGE" git --version
```

Each successful build appends measured `workspace_bytes`, `wall_seconds`, `suite_counts`, `apt_versions`, import
path, version, source hash and VM image reference to `pins.json.builds`. Copy the two observed results below before
the video. Workspace size must stay below 700,000,000 bytes. If groupby wall time exceeds 600 seconds, tell the
integrator to decide the marker selection or accept AT3 unchecked on its 900 second timeout.

| Measurement | Build 1 | Build 2 |
| --- | --- | --- |
| Source tarball SHA-256 | `f10fd12b16d923815aa05b0fd4f586ae67179bb00a3034980283263fe9c3fd07` | Same as build 1 |
| `du -sb /workspace` | 51,219,193 | 51,219,193 |
| Groupby test count, skips, failures, errors | 23,215 tests, 898 skipped, 0 failures, 0 errors | Same as build 1 |
| Groupby wall seconds at 2 CPUs | 57.2 | 57.8 |
| VM loopback image digest | `127.0.0.1:5000/receipts-demo/pandas-64677@sha256:d0c43cd34df4d073c257d906ce0c3bc36fa99adbe100df41332ff2915f90252a` | Same as build 1 |

The integrator runs `CLAUDE_SKIP_VERIFY=1 .claude/verify.sh` unsandboxed after the lane report. This lane runs its
own frozen offline lint, format, strict typecheck and unit tests, with no model or network.
