# Falden Receipts

**Your agent says it's fixed. Receipts tries to break it.**

**Demo video (under 3 minutes):** [youtu.be/IVWiuEFBw6w](https://youtu.be/IVWiuEFBw6w)

**One real catch.** On pandas #64677, a performance pull request that pandas later followed with a fix of its own, Receipts found 9 blockers: tests that failed 5 of 5 on the PR and passed 5 of 5 on the code before it. One of them also passes 5 of 5 on pandas' own later fix. Receipts' fix loop then made six attempts. The fifth passed the repro it was shown and the one held back, and a fresh attack on it caught a performance regression in the fix itself, so Receipts does not call it fixed. One case, not part of the measured results: see [The pandas demo](#the-pandas-demo).

## Try it in 2 minutes

No key, no Docker, no model call: `receipts demo` replays the recorded run on pandas #64677 from its records in `demo/pandas-64677/evidence/` and writes its receipts as HTML. Nothing is attacked or rerun. It needs git, [uv](https://docs.astral.sh/uv/) and Python 3.12 or later (uv installs it if you have none); only `uv sync` uses the network. We tested it on Linux x86_64. It does not run on Windows itself, since the ledger locks files with `fcntl`; use WSL there.

```sh
git clone https://github.com/falden-ai/receipts
cd receipts
uv sync --frozen
uv run receipts demo
```

Open the two pages it writes in a browser:

- `runs/demo/pandas-64677.html`, the Oct 31 run on the PR: 9 blockers, each with the claim it quotes, its repro test and command, and its 5 runs on the PR that fail and 5 on the code before it that pass, with each run's output; who checked what, at what cost, per stage; and the other 13 findings, refuted, with no repro, or unchecked over the plan's caps.
- `runs/demo/pandas-64677-fix-5.html`, the fresh attack on fix attempt 5: its one blocker, correctness-9.1, the performance regression in the fix itself, which fails 5 of 5 on the fix and passes 5 of 5 on base.

Intake and planning costs are on the run's target summary, which is not in this repository, so the pages leave them out. What each record backs is in `demo/pandas-64677/evidence/README.md`.

## Run it on your PR

There is no hosted version. Send a link to a pull request to alexander@falden.ai: we run Receipts on it ourselves, from our own credits, and send you back its HTML receipt within 2 business days.

What we need:

- **A public repository.** While on Meta's free credits, content sent to the API may be used for product improvement, so we run it only on public code.
- **Python.** Receipts attacks Python only.
- **Tests that run in a container.** We build an image of the PR's repository beforehand; the run phase has no network, so the PR's tests must run from that image without fetching anything.

What the receipt can and cannot tell you:

- A blocker is a test that fails 5 of 5 on the PR and passes 5 of 5 on the code before it. It is a counterexample a valid fix should pass, not a proof the PR is wrong.
- Alarms on live PRs need a human check before anyone acts on them. On the measured eval's test split, blockers flagged 36.7% of good patches (33 of 90), and a live PR is outside those measured numbers.
- The models may have seen public fixes in training. Attacks past the plan's caps are listed as unchecked, not run. See [Limits](#limits) for the rest.

## The problem

Coding agents open pull requests that say "fixed" and "all tests pass". The tests they cite can be ones the agent wrote or picked, so passing them says little about the claim. A wrong fix that passes its own tests merges, and the bug report arrives weeks later.

## How it works

Receipts reads what the agent claimed, attacks the claims its budget covers with runnable tests in a sandbox, and makes a finding a blocker only if it survives a skeptic. No runnable repro, no blocker.

1. **Claims.** Intake turns the PR into typed claims (`fixed_x`, `nothing_else_changed`, `backward_compatible` and others), each quoting the sentence it came from.
2. **Plan.** The planner picks attacks per claim within a budget. What the budget skips is listed as unchecked.
3. **Attackers.** Once per target, the regression attacker reruns the existing tests over the changed files on the code before and after the change; every other regression attack has a model write new tests of behavior the change should not alter. The correctness attacker asks for up to 3 candidate repro tests in one call. If none fails, an agent loop explores the code and writes more. A differential attacker runs model-written usage examples on base and patch, and a divergence becomes a repro only if a judge call finds it unintended and quotes the claim, an existing test or an unchanged docstring that it contradicts; any other divergence that repeats in all three runs is recorded as suspected. It is opt-in, behind `--lenses regression,correctness,differential`, not in the measured config, and attacks SWE-bench targets only.
4. **Skeptic code gates.** Only an assertion failure, or a crash in changed code, counts. Fake tests (mocking the changed function, call-count asserts) are thrown out. A blocker fails on the patch 5 times out of 5.
5. **Surrogate fixes.** For a `fixed_x` or `adds_behavior` claim, an agent that never sees the patch or the repro writes up to 3 fixes of its own. The repro must pass on at least one plausible surrogate: one whose own new test fails on base and passes on it, and which still passes the existing tests, for the files it changes, that pass on base.
6. **Refuter.** A different model tries to knock each blocker down. It can only downgrade.
7. **Ledger.** Each finding gets a status (`blocker`, `suspected`, `flaky`, `refuted`, `unchecked` and others). A blocker always carries its runs and a record of how it ran (base SHA, patch SHA, harness hash, seed, image digest); an attack over the plan's caps, or one that produced no repro, has its status but no record of how it ran. Each cost is recorded once, under its stage; intake and planning costs go on the target's summary, not the ledger.
8. **Receipt.** `receipts receipt html` writes an HTML page for one target: who checked what, at what cost, and what went unchecked.
9. **Ticket.** `receipts ticket` drafts one blocker as a Linear ticket with its repro, and `--post` files it. Only a blocker can become a ticket, and only when an operator files it.
10. **Check.** `receipts check post` turns a target's ledger into a GitHub check, posted through a GitHub App, that fails only on a confirmed blocker. Suspected or inconclusive findings leave it neutral. An operator posts it; Receipts does not listen for pull-request events.
11. **Fix loop.** `receipts fix` hands one confirmed blocker to `muse-spark-1.3`, which writes a fix from outside the sandbox; the sandbox only runs its commands, with no network. The fix gets up to 3 tries to pass the shown repro. A second repro for the same claim, one that fails on the patch, is held back, and the patch with the fix is attacked again, once, with new attacker calls. Only "survived fresh attack" counts as fixed, and a claim with no second failing repro cannot reach it. No fix on the pandas demo has reached it (see [The pandas demo](#the-pandas-demo)).

A **regression catch** fails on the patch and passes on the code before it. A **fix-validity catch** fails on the patch and passes on an independent plausible fix. Either is a counterexample a valid fix should pass, not a proof the patch is wrong. In the final's test split, 43% of the Spark blockers replayed on the reference fix (125 of 291) did not pass there either.

### Which Meta model does what

| Stage | Model |
|---|---|
| Intake (typed claims) | `muse-spark-1.3` (SWE-bench targets get two template claims instead, with no model call) |
| Planner | `muse-wonder-1.0` (falls back to `muse-spark-1.3`) |
| Attackers | `muse-spark-1.3` in the measured config. A second arm ran open-weights `muse-glimmer-30b` on a GPU we rent on Modal, served with vLLM: it caught 15 of 135, against Spark's 56. |
| Surrogate fixes | `muse-spark-1.3` |
| Refuter | `muse-wonder-1.0`, a different model from the Spark and Glimmer attackers |
| Fix loop | `muse-spark-1.3`, called from outside the sandbox |

## Built during the event

Falden, Inc. built everything in this repository during the hackathon, Oct 26 to Nov 4, 2026, apart from the third-party material listed below. That includes the code and prompts in `src/receipts/`, the tests, the eval configuration with its sample and split in `eval/`, the final's results in `eval/final/`, the pandas demo target in `demo/pandas-64677/` and the GitHub check demo in `demo/check-demo/`.

We built it in a private repository; this public one is a fresh copy of its final tree, with no earlier history. Commit ids in this README (40abf69 for the final run, d513d4c for the pandas demo's fix attempt 1, 4650268 for its attempt 5, aebaa90 for the last change to `src/`, 18b932e for the public-copy corrections) name commits in the private repository.

Used as published, not built by us: Meta's Muse models, called through Meta's API, and the open weights of `muse-glimmer-30b`, served with vLLM on Modal; gVisor, Docker and the Python packages pinned in `uv.lock`; SWE-bench Verified, the agent patches from the SWE-bench experiments and Epoch's SWE-bench images; the upstream projects' code in those patches and in the test fixtures under `tests/attack/fixtures/`; SWE-ABS's strengthened tests; code adapted from SWE-ABS and the SWE-bench harness in `src/receipts/eval/labelrun.py`, `src/receipts/eval/specs.py`, `src/receipts/eval/runsc_check.py`, `src/receipts/attack/runners.py`, `src/receipts/attack/sandbox.py` and `src/receipts/contracts.py`, and the harness's install and test commands in `eval/test-specs.json` (MIT; see NOTICE); the pandas code and pull-request text in `demo/pandas-64677/`; and Testora's five questions, which the differential attacker's prompt quotes verbatim. The research the design draws on, Testora included, is credited under [Prior work](#prior-work-and-what-is-new-here).

## Results

We ran Receipts' frozen config, with `muse-spark-1.3` as the attacker, on SWE-bench Verified agent patches. The sample and its split were frozen Oct 29; the run went from Oct 31 to Nov 1 at commit 40abf69. The spend cap cut 16 targets short and kept 3 from starting; all 19 then ran under a second, $50 cap.

- **Pool.** SWE-ABS's 251 strengthened instances (arXiv:2603.00520v1; dataset revision 79fc07a8, repository 26b4c736). The paper found overfitting tests on 53 of the 500 Verified instances, 45 of them among these 251, and says it corrected them. It names one, which is not in our pool, and we found no list of the rest, so we could not check the corrections and some plausible-wrong labels may still be wrong.
- **Labels and oracle.** Labels come from SWE-bench's own tests and SWE-ABS's stronger tests. A catch must fail on the agent's patch and pass on the reference fix. No model saw a gold patch: gold was used to label, to check the harness and to score, and 12 pool instances were dropped before the freeze, leaving 239: 8 whose gold the harness could not check (`eval/gold-drops.tsv`) and 4 whose gold did not pass under gVisor (`gold_runsc` in `eval/manifest.json`).
- **Claims.** On SWE-bench targets each patch gets two template claims (the issue is fixed; nothing else changed). Extraction of the agent's own claims ran only on the two demo pull requests (pandas #64677 and receipts-demo PR 1), not in the measured eval.
- **Isolation.** Git history stripped, and no network in the run phase.

Scored on the pre-registered test split, 60% of the sample: 225 targets, 135 bad or plausible-wrong and 90 good. Rates carry Wilson 95% intervals; MCC's interval is a bootstrap over targets (10,000 draws). The last column counts repros, several to a target, and its Wilson interval treats them as independent, so it is likely too narrow.

| Spark arm, test split | WDR: bad or plausible-wrong patches caught | Bad or plausible-wrong patches flagged | False alarms: good patches flagged | CAR: good patches accepted | MCC, any flag as the alarm | Repros that do not pass on the reference fix |
|---|---|---|---|---|---|---|
| **Blockers only** (what the check fails on) | 41.5% [33.5%, 49.9%] (56/135) | 59.3% [50.8%, 67.2%] (80/135) | 36.7% [27.4%, 47.0%] (33/90) | 63.3% [53.0%, 72.6%] (57/90) | 0.221 [0.094, 0.347] | 43.0% [37.4%, 48.7%] (125/291) |
| Plus fix-validity catches judged against the reference fix (not available on a live PR) | 55.6% [47.1%, 63.7%] (75/135) | 68.1% [59.9%, 75.4%] (92/135) | 38.9% [29.5%, 49.2%] (35/90) | 61.1% [50.8%, 70.5%] (55/90) | 0.289 [0.162, 0.413] | 59.4% [55.4%, 63.2%] (360/606) |

WDR counts a bad patch as caught only when a repro also passes on the reference fix: a blocker's for blockers only, and a blocker's or a fix-validity catch's with fix-validity catches. The MCC takes any flag as the alarm instead: for blockers only, the 80 of 135 bad patches flagged and the 33 of 90 good ones; with fix-validity catches, 92 and 35, a fix-validity catch counting as a flag. Counting a bad patch only when it is caught, while every flag on a good patch still counts as a false alarm, the MCC is 0.05 for blockers only and 0.16 with fix-validity catches.

- **Calibration split.** On the other 40% (150 targets), blockers only: 35.6% [26.4%, 45.8%] (32/90) caught, false alarms on 40.0% [28.6%, 52.6%] (24/60), and 46.7% (42/90) of bad patches flagged; MCC 0.066 [-0.095, 0.224] with any flag as the alarm, -0.05 counting a bad patch only when it is caught. Its report is in `eval/final/` too.
- **PPV depends on how many PRs are bad to begin with.** The report's blockers-only PPV takes the test split's WDR (56 of 135 bad patches caught) as the sensitivity and CAR (57 of 90 good patches accepted) as the specificity: 22.0% if 20% of PRs are bad, 53.1% if half are, 3.0% if 2.66% are. It leaves out bad patches flagged without a catch; counting any flag on a bad patch (80 of 135), the share of flagged PRs that are bad is 28.8%, 61.8% and 4.2%. That 2.66% is the share of agent-authored PRs that contain at least one reverting commit written by the agent: 893 of 33,580 in the AIDev dataset (Oukhay et al., "When AI Code Doesn't Stick: An Empirical Study on Reverted Changes Introduced by AI Coding Agents", MSR 2026, doi:10.1145/3793302.3793587). A reverting commit is not the same thing as a bad PR, so this case only illustrates a low base rate.
- **Cost per PR**, Spark arm, test split: $0.90 at the median and $1.45 at p95, for attacks, surrogate fixes and the refuter. Planning, one plan per target that both arms share, is not in these figures: it cost $0.19 per target at the median and $0.31 at p95. With it, a Spark run costs $1.10 per PR at the median and $1.71 at p95 (each target's own total: its Spark cost in `score.json` plus its planning cost in the final's target summaries in `eval/final/targets/`).
- **Glimmer.** The second arm caught 15 of 135 (11.1% [6.9%, 17.5%]), against Spark's 56, with false alarms on 11 of 90. It did not meet the pre-registered rule for moving the attackers to Glimmer. Its GPU time is on Modal's bill, not on Receipts' spend meter: $32.63 for one GPU, about 10.8 GPU-hours. That is the Glimmer app's whole bill for the hours the final ran: all 375 targets on both splits, the top-up included.
- **Disputed labels.** 13 of the 33 false alarms are on good patches where a blocker's repro passes on the reference fix, so the patch may have a defect its label missed. They are reported as "GOOD label disputed" and still count as false alarms; no label was changed.
- **Evidence.** The final's `score.json`, `report.json` and `report.md` for each split, and its target summaries (one per target, with that target's planning cost), are in `eval/final/` (`test/` and `calibration/` for the splits, `targets/` for the 375 summaries), with their sha256 in `eval/final/SHA256SUMS`. `receipts eval report` rebuilds `report.md` and `report.json` from `score.json` byte for byte (see [Quickstart](#quickstart)). The run's ledgers are not in this repository, so `score.json` itself cannot be rebuilt from here. Commit 40abf69 is in our private development repository (see [Built during the event](#built-during-the-event)). After it come six commits up to aebaa90, not counting merges, and none of them touches the scoring code in `src/receipts/eval/`. Three change `receipts fix`: its fresh attack runs with no plan caps and larger output limits, the fix's spend cap rises from $3.00 to $4.75 and its fresh attack's from $2.00 to $4.50, and the fixer's own calls get a hard cap. The new parameters default to the old values, so `receipts run` keeps the final's plan caps and output limits. One adds an identity check on the Meta client's spend file. One changes how the agent loop reads a line inside a code block that names a fence. One adds the demo image's pins. The commits after aebaa90 change nothing in `src/`; one of them, 18b932e, corrects the two demo runbooks, the pandas build script's default clone path and one rule in `AGENTS.md`.

### The pandas demo

One real pull request, pandas #64677 ("PERF: faster GroupBy.any/all for boolean-dtype columns"), which pandas later followed with a fix of its own. It is one case, not part of the numbers above. Its records, each attempt's included, are in `demo/pandas-64677/evidence/`, with a README of what each file backs.

- **The catch.** A run on Oct 31 found 9 blockers on the PR. Each failed 5 of 5 on the PR and passed 5 of 5 on the code before it, in fresh containers. One of them, regression-2.1, also passes 5 of 5 on pandas' own later fix; the other 8 were not checked against it.
- **The fix loop.** Six fix attempts in all, and none reached "survived fresh attack": two produced no usable fix, and three passed the shown and held-out repros but their fresh attacks left findings unchecked. The fifth fix, from `muse-spark-1.3`, passed the shown and held-out repros 5 of 5 each in fresh containers. Its fresh attack then caught a regression in the fix itself: on 50,000 groups, all but at most 200 of them empty, `GroupBy.any()` took 2.9 to 3.8 ms against `max()`'s 1.0 to 1.1 ms, against the PR's own claim of a faster `GroupBy.any/all`. That repro fails 5 of 5 on the fix and passes 5 of 5 on base (an independent replay also failed 5 of 5), so the attempt stands at "passes the shown repros", not fixed. The fix record words this as "fresh-attack repro ... not independently verified": `src/receipts/agentloop/fixloop.py` sets that reason when the repro does not pass on the fix in the independent replay, so here it means the fix failed the repro, not that the failure went unconfirmed. The fifth attempt cost $2.62. The sixth ended unchecked: no blocker in 21 findings, and one finding left unchecked by the cap of 5 candidates judged per claim.
- **Provenance.** The Oct 31 run used our eval VM's image. Its own records name no Receipts commit; fix attempt 1, 11 minutes after the run ended, pinned d513d4c. The image's base, Python packages and pandas commit are pinned, but its Debian build packages are not, so a rebuild elsewhere will likely get a new digest.
- **What the models could know.** pandas' fix is public, so the models may have seen it. Receipts read only the PR: no issue, no later commits.

### The GitHub check demo

falden-ai/receipts-demo is a Falden-owned demo repository. Its PR 1 (https://github.com/falden-ai/receipts-demo/pull/1) changes `quantity >= 5` to `quantity > 5` on purpose, and its description opens with that disclosure: "Falden-owned demo repository. This regression was planted for the demonstration. The discount must apply at five items or more." Below that it quotes how a coding agent might describe the change, with claims that are false on purpose: "No behavior change: orders of five or more items still get the 10 percent discount. All existing tests pass." Receipts ran on it, and an operator posted the result as the "Falden Receipts" check (https://github.com/falden-ai/receipts-demo/runs/125031919634): failure, with 8 blockers. Each blocker is a test that fails 5 of 5 on the PR and passes 5 of 5 on the base commit. All 8 annotations sit on the changed line, `checkout_demo/__init__.py:9`: the tracebacks did not locate a line, so the check put each on the line the PR adds. Three more attacks were over the plan's cap of 3 per lens and are listed as unchecked.

## How the design differs

No baseline arm ran. The Receipts column holds the table's only measured number; the other two columns describe designs.

| | Existing tests in CI | One model call asked "is this correct?" (must return a repro) | Receipts |
|---|---|---|---|
| Executes code | Yes, on the patch | Its one repro | Yes, sandboxed, on base and patch |
| A skeptic tries to refute a finding before it counts | No | Code gates only | Code gates, surrogate fixes and a different-model refuter |
| Published false-alarm rate with n and CI | No | No | 36.7% [27.4%, 47.0%] (33/90), Spark arm, test split |

## Install

Needs Linux on x86_64, Python 3.12 or later, [uv](https://docs.astral.sh/uv/), git, `jq` for the examples below, Docker with gVisor (`runsc`) registered as a runtime, and `openssl` for `check post --post`. A host without `runsc` fails every attack, after intake and planning, rather than falling back to another runtime. `.claude/verify.sh` (ruff, ruff format, mypy and pytest) passes on uv's CPython 3.12.16 and 3.14.10, and skips its 25 Docker-backed tests unless `RECEIPTS_RUN_DOCKER=1` is set on a host with Docker. On Ubuntu's own Python 3.12, 3 fix-loop tests fail because its sitecustomize.py loads before the tests' own; `uv sync --frozen --python 3.14.10` avoids that.

```sh
git clone https://github.com/falden-ai/receipts
cd receipts
uv sync --frozen
uv run receipts --help
```

## Quickstart

No key, no model call. `prices` lists the prices the spend meter uses, and `claims` prints the template claims for that SWE-bench target:

```sh
uv run receipts prices
mkdir -p runs
head -n 1 eval/pilot-run.jsonl | jq -c .target > runs/target.json
uv run receipts claims runs/target.json
```

### One target, Spark only

About $1 per target: the final's median was $0.90 before the planning call and $1.10 with it. Model calls are refused until the spend meter is armed. Receipts never pulls images, so pull the target's image by its digest beforehand. Below, `spend arm` sets a $5 cap (the default is $150): arm it only once your Meta credit is applied. `spend` shows what is armed and spent, and `replay` prints a finding's run record and repro without rerunning them. Replace each `<...>` here and below before running it; flags in square brackets are optional.

```sh
head -n 1 eval/pilot-run.jsonl > runs/one.jsonl
docker pull "$(jq -r .target.image runs/one.jsonl)"
read -rs FALDEN_META_API_KEY && export FALDEN_META_API_KEY
uv run receipts spend arm --credit-confirmed --cap-micro-usd 5000000
uv run receipts spend
uv run receipts run --targets runs/one.jsonl --arms demo/arms.json --ledger-dir runs/one
uv run receipts ledger rollup --ledger runs/one --arms demo/arms.json --by-target
uv run receipts replay <finding-id> --ledger runs/one/spark.jsonl
```

### Rebuild the final's report

No key, no model call. Check the files with `sha256sum -c eval/final/SHA256SUMS`, then rebuild the test split's report and compare:

```sh
uv run receipts eval report --score eval/final/test/score.json --out runs/rebuilt-test
cmp runs/rebuilt-test/report.md eval/final/test/report.md
cmp runs/rebuilt-test/report.json eval/final/test/report.json
```

### Receipt, ticket, check and fix

Without `--post`, `ticket` prints a draft and `check post` a preview, and neither reads a key. With it, `ticket` reads `RECEIPTS_LINEAR_API_KEY`, and `check post` reads `RECEIPTS_GITHUB_APP_ID`, `RECEIPTS_GITHUB_APP_INSTALLATION_ID` and `RECEIPTS_GITHUB_APP_KEY_FILE` (the path of the App's private key). `check post` needs a pull-request target, whose patch SHA is the PR's head commit: `--head-sha` must equal it and `--repo` must name the target's repository. A SWE-bench target such as `runs/one.jsonl` has no head commit to post to, so even the preview refuses it. In the example, `runs/pr.jsonl` is `demo/check-demo/target.template.json` filled in for your PR and written on one line (`jq -c .`), and `runs/pr/` is its ledger directory from `receipts run`. Build its image and set up the GitHub App as in `demo/check-demo/RUNBOOK.md`.

```sh
uv run receipts receipt html <target-id> --ledger runs/one/spark.jsonl --out runs/receipt.html
uv run receipts ticket <finding-id> --ledger runs/one/spark.jsonl --team-id <uuid> --receipt-base <https-url> --expect-org <workspace-key> [--post]
uv run receipts check post --ledger runs/pr/spark.jsonl --targets runs/pr.jsonl --target-id <pr-target-id> --repo <owner/name> --head-sha <pr-head-sha> [--post]
uv run receipts fix <finding-id> --ledger runs/one/spark.jsonl --targets runs/one.jsonl --arms demo/arms.json --out-dir runs/fix [--max-iterations {1,2,3}]
```

## Security posture

- **Attack sandbox:** gVisor `runsc`; all capabilities dropped, uid 65534, read-only root, capped pids, memory, CPU and disk.
- **Network.** The run phase, where agent code and repros run, has no network. Images are pulled or built beforehand, with network access (on our eval VM, HTTP and HTTPS to any host): SWE-bench images by digest, and the pandas demo image from a digest-pinned base, hash-checked Python packages and a pinned pandas commit, plus Debian build packages that are not version-pinned; and the check demo's image from the `python:3.12-slim` tag, with pytest pinned by version only and Debian's git and pytest's own dependencies not pinned.
- **Credentials stay out of the sandbox.** In the `receipts` command they come only from named environment variables: the Meta key (`FALDEN_META_API_KEY`), a Modal key and secret for a Modal-hosted Glimmer (`MODAL_PROXY_TOKEN_ID`, `MODAL_PROXY_TOKEN_SECRET`), and for posting, a Linear key (`RECEIPTS_LINEAR_API_KEY`) or a GitHub App key file (`RECEIPTS_GITHUB_APP_KEY_FILE`). Model calls come from the orchestrator outside the sandbox, and none of these values enters it. The eval VM scripts in `eval/vm/` use the operator's own AWS profile.
- **Git history is stripped** from the workspace, so no later commit is in it. The models may still have seen a public fix.
- **Model allowlist.** Any `-contributor` model id is refused before a request. That suffix selects Meta's Contributor mode, in which content may be used to train future Meta models.
- **Spend cap.** Meta API calls stop at the cap. A response that runs past its token bounds is metered in full and can carry spend past it; every later call is then refused. Glimmer calls are not metered: the code bounds them by step and time limits, and their GPU time is on Modal's bill.

## Limits

- Python only. Measured on SWE-bench Verified only; Tier B (merged agent PRs with a later public fix) is the pandas demo alone, not measured.
- Alarms on live PRs need a human check before anyone acts on them.
- The models may have seen public fixes in training: the attackers and the surrogate fixer alike, on the SWE-bench eval and on the pandas demo.
- By default only the regression and correctness lenses run (the differential lens is opt-in and not in the measured config). A lens the run does not use is attacked under the claim's default lens, or, when `--lenses` leaves that one out too, under the earliest lens it names; attacks past the plan's caps are listed as unchecked.
- "Passes the shown repros" is not "fixed", and a fix that games the test process can still pass a replay.
- Resistance to prompt injection was not measured. Repository text reaches the models as labeled data.
- Receipts does not check repository visibility. While on Meta's free credits, content sent to the API may be used for product improvement (Meta's billing page, read 2026-10-27), so we run it only on public code.

## Prior work and what is new here

Receipts is an independent project by Falden, Inc. It is not affiliated with or endorsed by Meta or by the authors of the papers below.

| Receipts part | Builds on |
|---|---|
| Regression catch: fails on the patch, passes on base | Meta's catching tests, which "fail on the proposed changes (a code revision) but pass on the parent revision": Becker et al., "Just-in-Time Catching Test Generation at Meta", FSE Companion 2026, arXiv:2601.22832, doi:10.1145/3803437.3805202; Harman, O'Hearn and Sengupta, "Harden and Catch for Just-in-Time Assured LLM-Based Software Testing: Open Research Challenges", FSE Companion 2025, arXiv:2504.16472, doi:10.1145/3696630.3734199. The regression oracle itself is older: BERT (behavioral regression testing, ICST 2010) and SWE-bench PASS_TO_PASS. |
| Correctness attacker | JiT's dodgy-diff workflow (arXiv:2601.22832, section 3.3) |
| Skeptic fake-test rules | JiT's RubFake assessor (arXiv:2601.22832, section 6) |
| Skeptic model refutation | JiT's LLM-as-judge assessors (arXiv:2601.22832, section 6.1) |
| Five-run flakiness rule | Alshahwan et al., "Automated Unit Test Improvement using Large Language Models at Meta" (TestGen-LLM), FSE Companion 2024, arXiv:2402.09171, doi:10.1145/3663529.3663839 |
| Differential attacker (opt-in, not in the measured config) | Testora (Pradel, ICSE 2026, arXiv:2503.18597) |
| Surrogate fixes and context masks | e-Otter++ (arXiv:2508.06365) and Otter++ (arXiv:2502.05368) |
| Eval pool and labels | SWE-bench Verified's FAIL_TO_PASS tests, and SWE-ABS's strengthened instances (Yu et al., arXiv:2603.00520) |
| Attacker loop design (no code copied) | mini-swe-agent, cited as its authors ask: SWE-agent (Yang et al., NeurIPS 2024, arXiv:2405.15793) |

The arXiv versions of the Meta papers above are licensed CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/). Quotations are marked. The differential attacker's prompt quotes Testora's five questions verbatim (Pradel, ICSE 2026, arXiv:2503.18597v3); all other code and prompts are original, apart from the third-party files listed in NOTICE.

### What is new here

The combination: attacks driven by the agent's own typed claims, with the quoted sentence as the oracle; surrogate fixes as a gate on fix-validity blockers; code gates and fresh-container reruns ahead of a downgrade-only refutation by a different Muse model; a ledger that records every finding's status and how each blocker ran; and on top of it, Linear tickets, a GitHub check, an HTML receipt and a fix loop that holds one repro back and attacks the fix again with new attacker calls. An agent-authored PR may have no one who can say whether a change was intended; these gates stand in for that question.

### Related open-source work

- jittest (Apache-2.0): a differential test-execution gate for agent-authored pull requests, with signed receipts. https://github.com/Kartik24Hulmukh/jittest
- JitTest (.NET): LLM-driven mutation testing inspired by Meta's catching-test research. https://www.nuget.org/packages/JitTest/
- jitgen (MIT): a proof-of-concept GitHub Action that generates JiT-style catching tests for Go. https://github.com/lukaswoellhaf/jitgen

How Receipts differs from these three: the pull request's own claims are the oracle, and a failing test becomes a blocker only if the skeptic, including surrogate fixes written without seeing the patch, cannot knock it down. Receipts does not sign its records.

## License

Apache License 2.0. See [LICENSE](LICENSE) and [NOTICE](NOTICE). Material from pandas, SWE-bench Verified, the SWE-bench experiments, the SWE-bench harness and SWE-ABS, and the upstream projects' code in it, keeps its own terms; NOTICE lists each file.

## Contributors

Built by Falden, Inc.

---

Builds on Meta research: arXiv 2601.22832 and 2402.09171. Independent project, not endorsed by Meta.
