# Spark fix loop

The orchestrator calls Muse Spark outside the sandbox. Repository execution has
no network and no secrets. Muse Code, its CLI, tokens, and sandbox egress are
outside this slice.

`FixWiring` is wired in the CLI. With a demo blocker on the eval VM:

```sh
receipts fix <finding-id> --ledger runs/demo/spark.jsonl \
  --targets runs/demo/pandas-64677.jsonl --arms demo/arms.json \
  --out-dir runs/demo/fix
```

The command appends a `FixRecord` to `runs/demo/fix/fixes.jsonl`, under `--out-dir`.
An existing record for the same finding in the same `--out-dir` refuses a rerun
before any fixer or model spend. It appends an `unchecked/runtime` record with
zero cost and leaves the prior fresh-attack outputs intact. One unreadable or
non-object line in `fixes.jsonl` refuses every finding in that `--out-dir`.
If a process is killed between creating the fresh-attack directory and appending
its record, a retry that produces the same combined patch triggers
`FileExistsError` on the orphan directory only after the fixer has spent.
Use a fresh `--out-dir` to try again.
A fix spends at most $4.75. Its fresh attack spends at most the smaller of $4.50
and what the fix has left. Unlike `receipts run`, which keeps at most 3 attacks
per lens and 4 lenses, the fresh attack runs every attack its plan holds
(REG-616). A plan too big for that budget still tries every attack: the spend
meter refuses each request that no longer fits before it is sent, and the fix
then reads `unchecked: budget`.
Only the fix's fresh attack uses 32,000 output tokens for `at2_stage1` and 16,384
for `attack.correctness.write`, on both model attack lenses. Exploration stays
at 8,192. Other runs keep the 16,000 stage-1 and 8,192 write limits.
At $3.107 settled, the $4.50 fresh-attack cap leaves $1.393 for the last call.
Calls run sequentially. #64677's larger stage-1 hold, about $0.654 to $0.679
($0.692608 at most), leaves about $0.714; its write hold, about $0.35 to $0.375,
leaves about $1.018. Each cap's $0.75 increase covers
the largest added hold, $0.32, at unchanged preceding spend. Longer answers can
still spend more, and every call remains subject to the meter.
Each fresh attack writes its ledgers, traces and summary to
`runs/demo/fix/reattack/<finding sha256 prefix>-<combined patch sha256 prefix>/`,
using the first 12 hex characters of each hash.
The record and console name that directory. Only
`survived_fresh_attack` may be called fixed. `passes_shown_repros` means exactly
that. Coverage remains `unchecked (not measured)`.

## Verification boundary

The AST scan checks the fixer's added Python lines against full workspace source.
It refuses new import-time executable statements except definitions, imports,
and assignments of literals or names. It also refuses process exit, environment,
output-stream, runner, introspection, and dynamic namespace references specified
by the guard. Reference and statement checks apply only to added lines. Alias
resolution uses the full source and rejects ambiguous process or runner aliases,
even if those bindings predate the fix. The scan never imports the repository.

Shown, held-out, and returned fresh-attack repros get five isolated direct-call
checks per case. Their pytest runs also get a deliberately failing canary as
supporting evidence on pytest runners. Django and Sympy report
`canary_scope: none: not a pytest runner`. The `ReAttack(Target, Arm, budget_micro_usd=...)` seam exposes no runner or sandbox
hook, so the integrator's internal re-attack runs do not get this canary. Each
pytest record states `canary_scope: lane replays only; internal ReAttack runs are not
covered`; the command prints this limit. Findings without a repro cannot receive
a lane replay.

A cheat inside the very function the repro calls is indistinguishable from a
change to the asserted behavior by an in-process check. The held-out repro and
fresh attack exist to catch that residual. Static restrictions and passing runs
do not prove that no such cheat exists, especially in the opaque internal
re-attack runs. The direct runner refuses PATH values with empty or relative
entries. It starts with `python -I -S`, consumes the nonce, replaces stdin with
`/dev/null`, closes descriptors from 3 up to the smaller of `SC_OPEN_MAX` and
`2**31 - 1` (exclusive), and closes every descriptor above 2 listed in
`/proc/self/fd` or `/dev/fd`. This happens before `site.main()` loads site-packages,
editable-install `.pth` files, and startup hooks. Child interpreters drop `-S` so
spawn and forkserver children also load those hooks. Before startup hooks run,
the runner captures `os.fstat`, `os.write`, `os._exit`, and fd 1's device and inode.
A changed fd 1 or replaced `sys.stdout` refuses certification without writing the
nonce. Otherwise the captured `os.write` emits the verdict and the captured
`os._exit` exits immediately, skipping atexit handlers and thread joins. The tar
fallback also starts with `-I -S`; this change reaches the default run.

Code in the replay's own process, including startup hooks, can still interfere
through frames, gc, ctypes, memory, a thread racing the runner, or patches to
runner internals. On any replay containing the agent's patch, this can certify a
pass. On the agent-patch replay, it can falsely mark a combined failure as present
before the fix and exempt it. The direct runner is not a security boundary.

## Remaining integration and demo work

The integrator owns command registration and the re-attack adapter. A held-out
repro is required to earn `survived_fresh_attack`. No live model, Docker, network,
or demo run is part of this lane's local verification.
