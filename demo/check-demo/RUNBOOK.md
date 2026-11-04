# Receipts GitHub check demo

Falden-owned demo repository. This regression was planted for the demonstration.
Keep that sentence visible in the PR and on screen. Do not describe this as a bug
found in the wild. Copy this directory into a separate public `receipts-demo`
repository under a Falden-owned account or organization. It contains only the
public demo package, tests, image recipe, and setup instructions.

## Set up the repository and App

1. Create the public `OWNER/receipts-demo` repository from this directory. The
   starting package is correct: the bulk discount applies at `quantity >= 5`.
2. Create the GitHub App **Falden Receipts** under that owner. Turn webhooks off.
   Grant **Checks: read and write** and **Metadata: read** only. Make the App
   installable only on that account and install it on `receipts-demo` only.
   There is no receiver, workflow, or `pull_request_target` trigger.
3. Generate the App's RSA private key. On the PC save it outside every repository
   at `~/.config/falden-receipts/github-app.pem`, mode 600. Do not put it in
   an env file, the image, the eval VM, a run file, or a ledger. Record the App
   id and installation id. Posting uses system OpenSSL, with no Python signing
   dependency.
4. In the posting shell on the PC, set only these App configuration variables:

   ```sh
   export RECEIPTS_GITHUB_APP_ID=APP_ID
   export RECEIPTS_GITHUB_APP_INSTALLATION_ID=INSTALLATION_ID
   export RECEIPTS_GITHUB_APP_KEY_FILE="$HOME/.config/falden-receipts/github-app.pem"
   chmod 600 "$RECEIPTS_GITHUB_APP_KEY_FILE"
   ```

The poster mints a short-lived token requesting only `receipts-demo` and
`checks: write`. It refuses a token that does not include the requested full
repository name. No App credential belongs in the untrusted-code run phase.

## Build the base image and plant the disclosed regression

Build from the correct base tree before opening the demo PR. Image building
requires Docker and package-download access; Receipts tests do neither.
The Dockerfile installs git and pytest and copies only the package and tests to
`/workspace`. The build context allowlist excludes credentials and Git history.

```sh
docker build -t ghcr.io/OWNER/receipts-demo:base .
docker run --rm --network none --user 65534:65534 ghcr.io/OWNER/receipts-demo:base
docker run --rm --network none --user 65534:65534 ghcr.io/OWNER/receipts-demo:base git --version
docker push ghcr.io/OWNER/receipts-demo:base
docker image inspect --format '{{index .RepoDigests 0}}' ghcr.io/OWNER/receipts-demo:base
```

Record the base commit and the resulting repository digest, which must be
`name@sha256:<64 hex characters>`. A local `sha256:...` image id is not a valid
target image reference. The digest pins the exact built image; the Dockerfile's
Python base tag is resolved at build time.

On 11/1 the image was built and both no-network checks ran on the eval VM, which
was given no registry credential. The VM saved the image with `docker save`, and
the PC, which had no Docker daemon running, pushed it with `crane` instead of
`docker push`. A newly published package is private. After an organization owner
allowed public packages, the package was made public, so the VM pulls it with
no credential. Check the image's history and files for secrets before that step:
GitHub does not let a public package go private again. The pinned image is
`ghcr.io/falden-ai/receipts-demo@sha256:c9825352a56bc24942994fbf5432e3a56dc2d1bd5904c9b6c2a91d9670a2a86e`,
built from base `70317f960c2e9206a08d8d08a0ff02fe565e622a`.

Have a coding agent open a PR from a branch of this same repository. Deliberately
change `quantity >= 5` to `quantity > 5`, leaving the tests intact. Use this PR
body and show it in the video:

> Falden-owned demo repository. This regression was planted for the demonstration.
> The discount must apply at five items or more.
>
> Below is how a coding agent might describe this change. Its claims are false on purpose, and Receipts checks them:
>
> > Simplifies the bulk discount check in `total_cents`. No behavior change: orders of five or more items still get the 10 percent discount. All existing tests pass.

On 11/1 a run on the two disclosure lines alone extracted no claims and had no
findings; it was not posted. The quoted summary gives Receipts claims to test.

The base tests pass; this patch fails the quantity-five test. This is a planted
regression with a base oracle, not a claim about a third-party repository.

## Prepare the LIVE target and run Receipts

`target.template.json` is a RunItem template, not an executable target yet.
Replace `OWNER`, both SHA placeholders, the image digest, the diff placeholder and
the body with the exact PR values, since the template's body holds only the
disclosure. Keep `tier` as `live` and `temperatures` as `spec`.
Serialize the result as one JSON object on one line in `run.jsonl`. The patch is
the actual unified diff from that base to that head, including hunk headers.

A LIVE target with an image and no checkout gets its base tree from the image's
`/workspace`. Each side gets its own copy of that tree, and the patch is applied
only to the patch side's copy. Without `runsc` a LIVE target is refused rather
than run under `runc`; do not change the tier to get around that.

On the eval VM, pull the digest-pinned image in the trusted preparation phase.
Confirm `runsc` is installed; LIVE runs require gVisor. Use the existing approved
arms file for the demo run:

```sh
UV_OFFLINE=1 uv run --frozen receipts run --targets run.jsonl --arms demo/arms.json --ledger-dir "runs/check-demo-$(date -u +%Y%m%dT%H%M%SZ)/ledgers" --jobs 1
```

The run phase has no network. Keep repository code and its sandbox separate from
the model credential. Confirm the ledger records five failing patch runs and
five passing base runs before expecting a regression blocker. If no blocker is
confirmed, keep the honest green or neutral check; do not alter the evidence.

Copy `run.jsonl` and the selected arm's ledger from the VM to the PC. The App key
stays on the PC. Review both files before posting because the check includes the
quote, repro, replay key, and captured test output in the public repository.

## Preview, post, and record the frame

`receipts check post` is part of the main CLI.

```sh
UV_OFFLINE=1 uv run --frozen receipts check post --ledger LEDGER.jsonl --targets run.jsonl --target-id receipts-demo-pr-1 --repo OWNER/receipts-demo --head-sha HEAD_SHA
UV_OFFLINE=1 uv run --frozen receipts check post --ledger LEDGER.jsonl --targets run.jsonl --target-id receipts-demo-pr-1 --repo OWNER/receipts-demo --head-sha HEAD_SHA --post --json
UV_OFFLINE=1 uv run --frozen receipts check post --ledger LEDGER.jsonl --targets run.jsonl --target-id receipts-demo-pr-1 --repo OWNER/receipts-demo --head-sha HEAD_SHA --post --json
```

The first call only prints the payload. It reads no environment variable and
makes no API call. Every post creates a new check run, so the posted calls must
return different ids and URLs. GitHub shows the newest **Falden Receipts** check
for the App on that head. Confirm it has the current conclusion and annotations.

GitHub appends annotations when a check run is updated; it cannot clear old
annotations. A fresh run prevents repeated or refuted blockers from leaving
duplicate or stale annotations on the newest check. Older runs remain in the
history. Only additional batches of more than 50 annotations update the run
created by that same post.

If a confirmed blocker has no added non-test line to annotate, the check still
posts and concludes failure. Its summary names the finding id and explains why
no line can be annotated. Only that blocker's line annotation is omitted. This
includes deletion-only and test-only patches; it never prevents the check from
posting.

The check fails only on a confirmed blocker. Without a blocker, suspected,
flaky, possibly toxic, or s_sat_inconclusive findings make it neutral, including
when mixed with clean results. It is neutral with "Nothing was checked" when
all findings are unchecked or there are none. Otherwise it succeeds. There are
no Re-verify, File ticket, or Accept risk buttons. Those actions require a
webhook receiver that this slice does not provide.

Capture the check and annotation with the planted-demo disclosure on screen.
The live run on 11/1 (00:28Z on 11/2) and its two posts recorded these values:

| Evidence | Value |
| --- | --- |
| PR URL and head SHA | https://github.com/falden-ai/receipts-demo/pull/1, head `1968d37e4a905d635b65f0e8c7ceca9e034107e2` |
| Base image repository digest | `ghcr.io/falden-ai/receipts-demo@sha256:c9825352a56bc24942994fbf5432e3a56dc2d1bd5904c9b6c2a91d9670a2a86e` |
| Selected ledger path | `runs/check-demo-20261102T002817Z/ledgers/spark.jsonl` (Spark arm: 11 findings, 8 blockers) |
| First check-run id and URL | 125031904405, https://github.com/falden-ai/receipts-demo/runs/125031904405 |
| Second check-run id and URL | 125031919634, https://github.com/falden-ai/receipts-demo/runs/125031919634 |
| Number of this App's check runs on the SHA | Two, both failure: "8 blockers; 0 suspected, 3 unchecked" |
| Newest check's annotations | Eight, one per blocker, all at `checkout_demo/__init__.py:9` |
| Screenshot path | Pending capture |

Exit codes: 0 for preview or successful post, 1 for invalid input or a GitHub
error, 2 for missing or unsafe App configuration, 3 when the installation token
does not include `--repo`. A key with mode 644 is refused before any API call.
