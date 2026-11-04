"""The five agent entries whose patches Tier A verifies: the spec's 3 mini-swe-agent bash-only v2.x entries plus 1
OpenHands entry (section 3), and a fourth mini entry the integrator added on 2026-10-27 (REG-565) because four entries
could not supply 150 BAD. The parsers for their published per-instance results live here too.

All five sit in SWE-bench/experiments at commit 40f164d5 (main on 2026-09-03), under evaluation/verified/. Each has a
per-instance resolved list in the repository, SWE-ABS's own per-patch verdicts (2026-10-26), and per-instance
evaluation logs in the public bucket s3://swe-bench-submissions: <logs>/<instance_id>/patch.diff (the patch the harness
graded) and report.json (its per-test FAIL_TO_PASS and PASS_TO_PASS results). The three bash-only entries whose
metadata.yaml says `logs: null` (glm-5-high, minimax-2-5-high, gemini-3-flash-high) still have that folder in the
bucket; listings of 2026-10-27 found it, and `logs_note` says so. Every report's resolved flag is checked against the
repository's resolved list, and its test lists against the dataset's, before any label uses it.

Why these five:
- Six bash-only mini-swe-agent v2.x entries have both a resolved list in the repository and SWE-ABS verdicts:
  claude-4-5-opus-high, claude-4-6-opus, gemini-3-flash-high, glm-5-high, gpt-5-2-high and minimax-2-5-high.
  BAD supply is the binding constraint at 150 BAD, so the pick follows the unresolved count on the 251 pool:
  glm-5-high 56, gpt-5-2-high 48, claude-4-6-opus 47, minimax-2-5-high 46, the other two 42 each. gpt-5-2-high is
  passed over so that the entries come from different model makers (the OpenHands entry already runs GPT-5).
  (These counts are of unresolved records. The per-test reports later showed that 12 of the first four entries'
  unresolved records are empty patches and 37 fail only PASS_TO_PASS, which left 148 BAD on the pool, short of 150.)
- gemini-3-flash-high is the fifth (REG-565): it adds BAD records from a fifth model maker, Google.
- 20250807_openhands_gpt5 is the only entry by OpenHands itself that SWE-ABS re-evaluated, and it carries the
  SWE-bench team's checked mark.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Sequence
from enum import StrEnum

from pydantic import Field

from receipts.contracts import Frozen
from receipts.eval.pool import INSTANCE_ID
from receipts.eval.sources import EXPERIMENTS_COMMIT


class Agent(StrEnum):
    MINI = "mini-SWE-agent"
    OPENHANDS = "OpenHands"


class PatchStore(StrEnum):
    S3_LOGS = "s3-logs"  # <assets.logs>/<instance_id>/patch.diff and report.json
    S3_TRAJS = "s3-trajs"  # <assets.trajs>/<instance_id>/<instance_id>.traj.json, info.submission
    GITHUB_REPO = "github-repo"  # all_preds.jsonl and logs/ in the submitter's own repository


class Entry(Frozen):
    key: str = Field(min_length=1)  # the folder name under evaluation/verified/
    agent: Agent
    agent_version: str = Field(min_length=1)
    bash_only: bool
    model: str = Field(min_length=1)  # tags.model_display plus reasoning effort, as the leaderboard shows it
    model_id: str = Field(min_length=1)  # tags.model[0]
    model_org: str = Field(min_length=1)
    open_weights: bool
    date: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")  # the folder's date prefix
    resolved_file: str  # per_instance_details.json or results/results.json, relative to the folder
    published_resolved_pct: str  # info.resolved in metadata.yaml (mini), or resolved/500 from results.json (OpenHands)
    patch_store: PatchStore
    patch_location: str = Field(min_length=1)
    swe_abs_agent_id: str  # the agent id in SWE-ABS's verdict file
    logs_note: str = ""  # where patch_location came from, when it is not metadata.yaml's assets.logs

    @property
    def folder(self) -> str:
        return f"evaluation/verified/{self.key}"

    @property
    def resolved_source(self) -> str:
        return f"experiments:{self.key}:{self.resolved_file}"

    @property
    def metadata_source(self) -> str:
        return f"experiments:{self.key}:metadata.yaml"


_UNLISTED_LOGS = (
    "metadata.yaml lists assets.logs as null; a listing of the bucket on 2026-10-27 shows this folder with "
    "<instance_id>/patch.diff, report.json, test_output.txt and job-output.json for every instance"
)

SELECTED: tuple[Entry, ...] = (
    Entry(
        key="20260217_mini-v2.0.0_glm-5-high",
        agent=Agent.MINI,
        agent_version="2.0.0",
        bash_only=True,
        model="GLM 5 (high reasoning)",
        model_id="glm-5",
        model_org="Z-AI",
        open_weights=True,
        date="2026-02-17",
        resolved_file="per_instance_details.json",
        published_resolved_pct="72.8",
        patch_store=PatchStore.S3_LOGS,
        patch_location="s3://swe-bench-submissions/bash-only/20260217_mini-v2.0.0_glm-5-high/logs",
        swe_abs_agent_id="20260217_mini-v2.0.0_glm-5-high",
        logs_note=_UNLISTED_LOGS,
    ),
    Entry(
        key="20260217_mini-v2.0.0_claude-4-6-opus",
        agent=Agent.MINI,
        agent_version="2.0.0",
        bash_only=True,
        model="Claude 4.6 Opus",
        model_id="claude-opus-4-6",
        model_org="Anthropic",
        open_weights=False,
        date="2026-02-17",
        resolved_file="per_instance_details.json",
        published_resolved_pct="75.6",
        patch_store=PatchStore.S3_LOGS,
        patch_location="s3://swe-bench-submissions/bash-only/20260217_mini-v2.0.0_claude-4-6-opus/logs",
        swe_abs_agent_id="20260217_mini-v2.0.0_claude-4-6-opus",
    ),
    Entry(
        key="20260217_mini-v2.0.0_minimax-2-5-high",
        agent=Agent.MINI,
        agent_version="2.0.0",
        bash_only=True,
        model="MiniMax M2.5 (high reasoning)",
        model_id="minimax-m2.5",
        model_org="Minimax",
        open_weights=True,
        date="2026-02-17",
        resolved_file="per_instance_details.json",
        published_resolved_pct="75.8",
        patch_store=PatchStore.S3_LOGS,
        patch_location="s3://swe-bench-submissions/bash-only/20260217_mini-v2.0.0_minimax-2-5-high/logs",
        swe_abs_agent_id="20260217_mini-v2.0.0_minimax-2-5-high",
        logs_note=_UNLISTED_LOGS,
    ),
    Entry(
        key="20260217_mini-v2.0.0_gemini-3-flash-high",
        agent=Agent.MINI,
        agent_version="2.0.0",
        bash_only=True,
        model="Gemini 3 Flash (high reasoning)",
        model_id="gemini-3-flash-preview",
        model_org="Google DeepMind",
        open_weights=False,
        date="2026-02-17",
        resolved_file="per_instance_details.json",
        published_resolved_pct="75.8",
        patch_store=PatchStore.S3_LOGS,
        patch_location="s3://swe-bench-submissions/bash-only/20260217_mini-v2.0.0_gemini-3-flash-high/logs",
        swe_abs_agent_id="20260217_mini-v2.0.0_gemini-3-flash-high",
        logs_note=_UNLISTED_LOGS,
    ),
    Entry(
        key="20250807_openhands_gpt5",
        agent=Agent.OPENHANDS,
        agent_version="unversioned",
        bash_only=False,
        model="GPT-5 (2025-08-07)",
        model_id="openai/gpt-5-2025-08-07",
        model_org="OpenAI",
        open_weights=False,
        date="2025-08-07",
        resolved_file="results/results.json",
        published_resolved_pct="71.8",
        patch_store=PatchStore.S3_LOGS,
        patch_location="s3://swe-bench-submissions/verified/20250807_openhands_gpt5/logs",
        swe_abs_agent_id="20250807_openhands_gpt5",
    ),
)


class SelectionError(ValueError):
    """An entry set that does not match the spec's 3 + 1 plus the fifth entry (REG-565)."""


# The spec's 3 mini-swe-agent entries plus 1 OpenHands entry, and the integrator's fifth mini entry (REG-565).
MINI_ENTRIES = 4
OPENHANDS_ENTRIES = 1


def check_selection(entries: Sequence[Entry]) -> None:
    """4 mini-swe-agent bash-only v2.x entries plus 1 OpenHands entry, each a different model from its own maker."""
    minis = [e for e in entries if e.agent is Agent.MINI]
    openhands = [e for e in entries if e.agent is Agent.OPENHANDS]
    total = MINI_ENTRIES + OPENHANDS_ENTRIES
    if len(entries) != total or len(minis) != MINI_ENTRIES or len(openhands) != OPENHANDS_ENTRIES:
        raise SelectionError(
            f"need {MINI_ENTRIES} mini-swe-agent entries and {OPENHANDS_ENTRIES} OpenHands entry, "
            f"got {[e.key for e in entries]}"
        )
    for e in minis:
        if not e.bash_only:
            raise SelectionError(f"{e.key}: the mini-swe-agent entries are bash-only runs")
        if e.agent_version.split(".")[0] != "2":
            raise SelectionError(f"{e.key}: mini-swe-agent {e.agent_version} is not v2.x")
    if len({e.key for e in entries}) != total:
        raise SelectionError("duplicate entry")
    if len({e.model_id for e in entries}) != total:
        raise SelectionError("each entry must use a different model, or the sampler's model cap means nothing")
    if len({e.model_org for e in entries}) != total:
        raise SelectionError("each entry comes from a different model maker")


class ResolvedList(Frozen):
    """What an entry's published results say, per instance."""

    entry: str
    resolved: frozenset[str]
    no_generation: frozenset[str]  # the agent submitted no patch
    reported: frozenset[str]  # instances the file covers
    no_logs: frozenset[str] = frozenset()  # results.json: evaluated with no logs, so neither resolved nor unresolved

    @property
    def unresolved(self) -> frozenset[str]:
        return self.reported - self.resolved - self.no_logs

    def resolved_status(self, instance_id: str) -> bool | None:
        """True or False from the published results; None when they cannot say (no logs)."""
        if instance_id in self.no_logs:
            return None
        return instance_id in self.resolved


def _ids(values: Iterable[object], what: str) -> frozenset[str]:
    out = set()
    for v in values:
        if not isinstance(v, str):
            raise ValueError(f"{what}: {v!r} is not an instance id")
        out.add(v)
    return frozenset(out)


def parse_resolved(entry: Entry, raw: bytes | str, universe: Iterable[str]) -> ResolvedList:
    """Read per_instance_details.json (mini) or results/results.json (OpenHands) into one shape."""
    pattern = re.compile(INSTANCE_ID)
    all_ids = frozenset(universe)
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError(f"{entry.key}: results must be a JSON object")
    if entry.resolved_file == "per_instance_details.json":
        resolved = set()
        for iid, detail in data.items():
            if not isinstance(detail, dict) or not isinstance(detail.get("resolved"), bool):
                raise ValueError(f"{entry.key}: {iid} has no boolean 'resolved'")
            if detail["resolved"]:
                resolved.add(iid)
        reported = _ids(data, entry.key)
        out = ResolvedList(entry=entry.key, resolved=frozenset(resolved), no_generation=frozenset(), reported=reported)
    elif entry.resolved_file == "results/results.json":
        if not isinstance(data.get("resolved"), list):
            raise ValueError(f"{entry.key}: results.json has no 'resolved' list")
        resolved_ids = _ids(data["resolved"], entry.key)
        no_gen = _ids(data.get("no_generation", []), entry.key)
        no_logs = _ids(data.get("no_logs", []), entry.key)
        out = ResolvedList(
            entry=entry.key, resolved=resolved_ids, no_generation=no_gen, reported=all_ids, no_logs=no_logs
        )
    else:
        raise ValueError(f"{entry.key}: unknown results file {entry.resolved_file}")
    bad = [i for i in out.reported | out.resolved | out.no_generation | out.no_logs if not pattern.match(i)]
    if bad:
        raise ValueError(f"{entry.key}: not instance ids: {sorted(bad)[:5]}")
    stray = (out.reported | out.resolved | out.no_generation | out.no_logs) - all_ids
    if stray:
        raise ValueError(f"{entry.key}: instances outside SWE-bench Verified: {sorted(stray)[:5]}")
    if out.reported != all_ids:
        raise ValueError(f"{entry.key}: covers {len(out.reported)} of {len(all_ids)} instances")
    if out.resolved & out.no_generation:
        raise ValueError(f"{entry.key}: an instance with no patch cannot be resolved")
    if out.resolved & out.no_logs:
        raise ValueError(f"{entry.key}: an instance with no logs cannot be resolved")
    return out


def experiments_url(entry: Entry, file: str) -> str:
    return f"https://github.com/SWE-bench/experiments/blob/{EXPERIMENTS_COMMIT}/{entry.folder}/{file}"
