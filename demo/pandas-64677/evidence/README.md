# Evidence for the pandas #64677 demo

These are unedited, byte for byte copies of the records behind "The pandas demo" in the top-level README: the Oct 31 run on pandas #64677 and the six fix attempts after it, numbered `fix-1` to `fix-6` in the order they ran. Folder names keep the source folders' time stamps, and paths inside a fix folder follow the record's own `reattack_out_dir` (`runs/demo-fix-<stamp>/` there is `fix-<n>-<stamp>/` here). None of the six records has the status `survived_fresh_attack`, the only one that counts as fixed. Model traces, most logs, the per-target summaries and three later fixcheck reruns with the same result are left out; the demo's inputs (`pins.json`, `patch.diff`, `fix.diff`) are one folder up. From the repository root, `sha256sum -c demo/pandas-64677/evidence/SHA256SUMS` checks every file.

## run-2026-10-31T134142Z

`spark.jsonl` is the run's ledger, one JSON line per finding: 22 findings, 9 of them blockers (correctness-2.1, 2.2, 3.1 and 3.2; regression-2.1, 2.2, 2.3, 3.1 and 3.2). Each blocker carries 5 runs on the PR that fail on an assertion and 5 on the code before it that pass, all in fresh containers from image `sha256:d0c43cd3...`, plus a `replay` record of how it ran (base SHA, patch SHA, harness hash, seed, image digest). The attacks over the plan's caps and the one with no repro have a status and no such record. It backs the 9 blockers and their 5 of 5 counts, and the Ledger step of "How it works".

`pandas-64677.jsonl` is the target that run read: the PR's title, body and diff (the same bytes as `demo/pandas-64677/patch.diff`), base 82315e7, PR merge d27e2f3, and the image `127.0.0.1:5000/receipts-demo/pandas-64677@sha256:d0c43cd34df4d073c257d906ce0c3bc36fa99adbe100df41332ff2915f90252a`, a digest in the eval VM's own loopback registry that `pins.json` records for both VM builds. Its issue, commit message and session summary fields are empty. It backs "our eval VM's image" in the provenance line and "Receipts read only the PR: no issue, no later commits."

Every 5-run replay in these records uses seeds 0 to 4, the same as the Oct 31 run's. A fresh attack's freshness is in the attacker arm's seeds, which the fix loop sets for it; the records do not print them.

## fix-1-20261031T141600Z

`fixcheck.json` runs regression-2.1's repro 5 times on the PR with pandas' later fix applied (side `gold` in the file). Its `fix_diff_sha256`, ccc08a43..., is the sha256 of `demo/pandas-64677/fix.diff`, which `pins.json` records as pandas' own fix from its PR 65101. All 5 runs pass, each in a fresh container from the run's image. It backs "regression-2.1 also passes 5 of 5 on pandas' own later fix"; no other blocker has a check like it.

`fix/fixes.jsonl` is the first fix attempt's record: regression-2.1 went to `muse-spark-1.3`, with regression-2.2 held back. The model gave 3 answers in a row that could not be used, so no patch ran and no fresh attack followed, and the status is `unchecked` with the reason `runtime`. It is the first of the "six fix attempts in all".

`fixrun-20261031T141559Z.log` is the operator's log of that attempt. It is here because it names the Receipts commit: its first line, at 14:16:00Z on Oct 31, pins d513d4cc4a895b50eaf07b8da14ef8c17775cd41, 11 minutes after the run's last ledger line at 14:04:44Z. The run's own files name no commit.

## fix-2-20261031T143137Z

`fix/fixes.jsonl` is the second attempt's record, with the same blocker, the same held-back repro and the same outcome as the first: 3 unusable answers, no patch run, no fresh attack, `unchecked` with the reason `runtime`.

## fix-3-20261031T144523Z

`fix/fixes.jsonl` is the third attempt's record. Its fix passed the shown repro and the held-back one 5 of 5 each in fresh containers, and its fresh attack ran but left findings unchecked over the caps, so the attempt ended `unchecked` with the reason `budget`.

`fix/reattack/fdd9c76a2d59-c65523604236/spark.jsonl` is that fresh attack's ledger: 18 findings and no blocker. Twelve are unchecked, over the plan's cap of 3 attacks per lens, 5 produced no repro and one was refuted because it fails on base too. With the record above, it shows this attempt did not survive a fresh attack.

## fix-4-20261101T144850Z

`fix/fixes.jsonl` is the fourth attempt's record. The fix passed the shown and held-back repros 5 of 5 each in fresh containers, and the status is `passes_shown_repros`, with the reason "fresh attack contains unchecked findings".

`fix/reattack/fdd9c76a2d59-92d982fcba87/spark.jsonl` is that fresh attack's ledger: 20 findings and no blocker. Two are `suspected`: they fail on the fix and pass on base, but the refuter, `muse-wonder-1.0`, found them unsupported by the quoted claim. Two more are unchecked after a truncated model answer, 2 were refuted for failing on base too, and 14 produced no repro.

## fix-5-20261101T172423Z

`fix/fixes.jsonl` is the fifth attempt's record. On its first try `muse-spark-1.3` changed only `pandas/core/groupby/groupby.py`, and the fix passed the shown repro (regression-2.1) and the held-back one (regression-2.2) 5 of 5 each in fresh containers. The independent replay of its fresh attack's blocker, correctness-9.1, under `reattack_replay_runs`, failed 5 of 5 too, with `any()` at 2.9 to 3.8 ms against `max()` at 1.0 to 1.1 ms. So the status is `passes_shown_repros`, `reattack_reason` reads "fresh-attack repro ... correctness-9.1 not independently verified" (in this code path, the fix failed that repro in the independent replay), and `micro_usd` is 2616386, the $2.62 in the README.

`fixrun-pc-20261101T172423Z.log` is the operator's log of this attempt. Its first line, at 17:24:23Z on Nov 1, pins the Receipts commit 46502689fb978189c503e686b5f453fd2643f423.

`fix/reattack/fdd9c76a2d59-a0027791a01e/spark.jsonl` is that fresh attack's ledger: 19 findings, one blocker. The blocker, correctness-9.1, groups 1,000 rows of 5 boolean columns by a categorical with 50,000 categories, at most 200 of them used, with `observed=False`, and asserts that `GroupBy.any()` takes at most 1.5 times as long as `max()`. It backs the catch: it fails 5 of 5 on the fix (`any()` 3.0 to 3.2 ms, `max()` 1.0 ms) and passes 5 of 5 on base, in fresh containers, against the PR's own claim "faster GroupBy.any/all for boolean-dtype columns". The other 18 were refuted (3) or produced no repro (15).

## fix-6-20261101T225702Z

`fix/fixes.jsonl` is the sixth attempt's record. The fix passed the shown and held-back repros 5 of 5 each in fresh containers, and the status is `unchecked` with the reason `budget`, because its fresh attack left a finding unchecked.

`fix/reattack/fdd9c76a2d59-c910969b720c/spark.jsonl` is that fresh attack's ledger: 21 findings and no blocker. One, correctness-6.3, is unchecked, "over the top-n of 5 candidates judged per claim"; 8 were refuted for failing on base too and 12 produced no repro. It backs "The sixth ended unchecked: no blocker in 21 findings, and one finding left unchecked by the cap of 5 candidates per claim."
