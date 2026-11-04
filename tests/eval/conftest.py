"""Shared fixtures for the eval lane's tests."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence

import pytest

from receipts.contracts import (
    CaseResult,
    FailureType,
    Outcome,
    PatchMethod,
    RunnerKind,
    RunResult,
    Side,
    Target,
    Workspace,
)
from receipts.eval.labels import Label, Need
from receipts.eval.manifest import EntryCounts, EntryRecord, Manifest, Record, pool_records
from receipts.eval.pool import Pool

MODELS = ["m-a", "m-b", "m-c", "m-d"]
MANIFEST_REPOS = ["django__django", "sympy__sympy", "astropy__astropy", "pytest-dev__pytest", "psf__requests"]
LABELS = (Label.BAD, Label.GOOD, Label.PLAUSIBLE_WRONG)

SyntheticManifest = Callable[..., tuple[Manifest, str]]


def repo(iid: str) -> str:
    owner, rest = iid.split("__")
    return f"{owner}/{rest.rsplit('-', 1)[0]}"


def manifest_pool(per_repo: int = 40) -> list[str]:
    return [f"{r}-{2000 + n}" for r in MANIFEST_REPOS for n in range(per_repo)]


def entry_record(model: str) -> EntryRecord:
    zero = dict.fromkeys(
        (
            "instances", "resolved", "unresolved", "no_generation", "pool_instances", "pool_resolved",
            "pool_unresolved", "patches_pulled", "reports_pulled", "empty_patches",
        ),
        0,
    )  # fmt: skip
    return EntryRecord(
        key=f"entry-{model}",
        folder="f",
        agent="a",
        agent_version="2.0.0",
        bash_only=True,
        model=model,
        model_id=model,
        model_org="o",
        open_weights=False,
        date="2026-02-17",
        published_resolved_pct="70",
        resolved_source="r",
        metadata_source="m",
        patch_store="s3-logs",
        patch_location="s3://swe-bench-submissions/x/logs",
        logs_note="",
        counts=EntryCounts(pool_swe_abs_published={}, pool_provisional={}, pool_final={}, **zero),
    )


def make_manifest(
    *, frozen: bool, gold: dict[str, bool | None] | None = None, ids: list[str] | None = None
) -> tuple[Manifest, str]:
    """Every model has a record on every instance, labels rotating so each stratum is well supplied. A frozen one has
    final labels and every instance checked under runsc unless `gold` says otherwise. Returns it with its pool list."""
    pool_ids = ids or manifest_pool()
    verdicts = dict.fromkeys(pool_ids, True) | (gold or {})
    recs = []
    for n, iid in enumerate(pool_ids):
        for m, model in enumerate(MODELS):
            label = LABELS[(n + m) % 3]
            g = verdicts[iid] if frozen or gold else None
            recs.append(
                Record(
                    entry=f"entry-{model}",
                    instance_id=iid,
                    repo=repo(iid),
                    pools=("swe_abs_251",),
                    resolved=label is not Label.BAD,
                    no_generation=False,
                    swe_abs_published=None,
                    provisional=label,
                    assumed=(),
                    final=label if frozen else Label.PENDING,
                    needs=() if frozen and g is not None else (Need.GOLD_RUNSC,),
                    gold_runsc=g,
                )
            )
    pool_text = "".join(f"{i}\n" for i in sorted(pool_ids))
    pool = Pool(name="swe_abs_251", instance_ids=tuple(sorted(pool_ids)), per_repo={}, rule="synthetic", checks={})
    manifest = Manifest(
        status="frozen" if frozen else "draft",
        created_at="2026-10-28T00:00:00Z",
        pool_choice="swe_abs_251",
        blocked_on=(),
        sources=(),
        pools=pool_records(pool),
        entries=tuple(entry_record(model) for model in MODELS),
        records=tuple(recs),
    )
    return manifest, pool_text


@pytest.fixture
def synthetic_manifest() -> SyntheticManifest:
    return make_manifest


# ---------------------------------------------------------------- the run phase, faked (R24, scoring)

BANNER_DMESG = "[    0.000000] Starting gVisor...\n[    0.4] Ready!\n"


def run_result(outcome: Outcome, side: Side = Side.GOLD, **kw: object) -> RunResult:
    failure = FailureType.ASSERTION if outcome is Outcome.FAIL else None
    defaults: dict[str, object] = {"failure_type": failure, "exit_code": 0 if outcome is Outcome.PASS else 1}
    return RunResult.model_validate(
        {"side": side, "outcome": outcome, "fresh_container": True, "duration_s": 2.0, **defaults, **kw}
    )


class FakeRunner:
    def __init__(self, kind: RunnerKind) -> None:
        self.kind = kind

    def command(self, target: Target, tests: Sequence[str], *, report_path: str) -> list[str]:
        return ["run", *tests]

    def cases(self, output: str, report: str | None, changed_files: Sequence[str]) -> tuple[CaseResult, ...]:
        return ()


Answer = Outcome | RunResult | Exception


class FakeRunPhase:
    """contracts.Sandbox, faked: the probe shows gVisor; run_test answers `outcomes[target id]` (PASS by default): one
    answer, or a list of them in order, the last repeating; an Exception is raised."""

    def __init__(self, outcomes: Mapping[str, Answer | Sequence[Answer]] | None = None) -> None:
        self.outcomes: dict[str, list[Answer]] = {
            k: list(v) if isinstance(v, list | tuple) else [v] for k, v in (outcomes or {}).items()
        }
        self.prepared: list[tuple[str, Side, str | None]] = []
        self.ran: list[dict[str, object]] = []
        self.closed = 0

    def prepare(self, target: Target, side: Side, *, patch: str | None = None) -> Workspace:
        self.prepared.append((target.id, side, patch))
        method = None if patch is None else PatchMethod.GIT_APPLY  # contract v3: a patched workspace names its method
        return Workspace(
            id=target.id, side=side, root="/testbed", image_digest="sha256:" + "e" * 64, patch_method=method
        )

    def run(self, workspace: Workspace, command: Sequence[str], **kw: object) -> RunResult:
        return run_result(Outcome.PASS, stdout_tail=BANNER_DMESG)

    def run_test(self, workspace: Workspace, tests: Sequence[str], **kw: object) -> RunResult:
        self.ran.append({"target": workspace.id, "tests": tuple(tests), **kw})
        answers = self.outcomes.get(workspace.id, [Outcome.PASS])
        answer = answers.pop(0) if len(answers) > 1 else answers[0]
        if isinstance(answer, Exception):
            raise answer
        return answer if isinstance(answer, RunResult) else run_result(answer)

    def close(self, workspace: Workspace) -> None:
        self.closed += 1


def fake_runners() -> dict[RunnerKind, FakeRunner]:
    return {k: FakeRunner(k) for k in RunnerKind}
