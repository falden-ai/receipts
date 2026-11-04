"""One target's receipt as inert, self-contained HTML. No core-lane imports or external assets."""

from __future__ import annotations

import html
import math
import shlex
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from receipts.contracts import Claim, Finding, LedgerEntry, StageCost, Status, Tier, Unchecked
from receipts.surface.cli_commands import OUTSIDE_THE_LEDGERS
from receipts.surface.ledger import JsonlLedger, LatencyBasis, Percentiles, Rollup, StageRollup, usd
from receipts.surface.markdown import strip_links
from receipts.surface.text import visible
from receipts.surface.tickets import (
    CASES_LIMIT,
    COMMAND_LIMIT,
    FILE_LIMIT,
    FILES_LIMIT,
    KIND_PHRASES,
    OUTPUT_LIMIT,
    PATH_LIMIT,
    QUOTE_LIMIT,
)

CSP = "default-src 'none'; style-src 'unsafe-inline'; img-src 'none'; base-uri 'none'; form-action 'none'"

# All attributes and CSS are constants. Ledger and summary values appear only in escaped text nodes.
CSS = """\
:root { color-scheme: light dark; --paper: #EDECEB; --ink: #191919;
  --accent: #2200FF; --rule: #b6b5b4; }
@media (prefers-color-scheme: dark) {
  :root { --paper: #191919; --ink: #EDECEB; --accent: #9B8FFF; --rule: #676665; }
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--paper); color: var(--ink);
  font: 16px/1.55 system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; }
main { max-width: 1120px; margin: auto; padding: 32px clamp(16px, 4vw, 48px) 64px; }
header, section { border-bottom: 1px solid var(--rule); padding-bottom: 28px; margin-bottom: 28px; }
h1, h2, h3, h4 { line-height: 1.15; font-family: Georgia, "Times New Roman", serif; }
h1 { font-size: clamp(36px, 6vw, 56px); margin: 8px 0 24px; font-weight: 400; }
h2 { font-size: 30px; margin: 0 0 20px; font-weight: 400; }
h3 { font-size: 24px; margin: 0 0 12px; }
h4 { font-size: 19px; margin: 20px 0 8px; }
p { margin: 8px 0 16px; }
code { font: .875em/1.5 ui-monospace, SFMono-Regular, Consolas, monospace; overflow-wrap: anywhere; }
pre { max-width: 100%; overflow: auto; padding: 12px; border: 1px solid var(--rule); border-radius: 2px;
  white-space: pre; tab-size: 4; }
pre code { font-size: 13px; overflow-wrap: normal; }
dl { display: grid; grid-template-columns: 120px minmax(0, 1fr); gap: 8px 16px; margin: 0; }
dt { font-size: 13px; font-weight: 600; }
dd { margin: 0; min-width: 0; overflow-wrap: anywhere; }
.eyebrow { color: var(--accent); font-size: 13px; font-weight: 650; letter-spacing: .08em; text-transform: uppercase; }
.verdict { color: var(--accent); font: 400 32px/1.2 Georgia, "Times New Roman", serif; }
.table-scroll { max-width: 100%; overflow: auto; }
table { border-collapse: collapse; width: 100%; font-size: 13px; text-align: left; }
caption { text-align: left; margin-bottom: 8px; font-weight: 600; }
th, td { padding: 10px 12px 10px 0; border-bottom: 1px solid var(--rule); vertical-align: top; }
th { font-weight: 600; white-space: nowrap; }
td { min-width: 64px; }
td code { white-space: nowrap; }
article { min-width: 0; margin: 24px 0 32px; }
li { margin-bottom: 16px; overflow-wrap: anywhere; }
li pre { margin: 8px 0; }
strong { font-weight: 650; }
@media (max-width: 480px) {
  main { padding-top: 24px; }
  dl { grid-template-columns: 1fr; gap: 2px; }
  dd { margin-bottom: 10px; }
}
"""


class SummaryPlan(BaseModel):
    """Only the planned-out work crosses this lane's summary boundary."""

    model_config = ConfigDict(extra="ignore", frozen=True)
    unchecked: tuple[Unchecked, ...] = ()


class ReceiptSummary(BaseModel):
    """The core summary's public fields used here; unrelated fields are deliberately ignored."""

    model_config = ConfigDict(extra="ignore", frozen=True, allow_inf_nan=False)
    target_id: str = Field(min_length=1)
    tier: Tier
    claims: tuple[Claim, ...] = ()
    plan: SummaryPlan
    costs: tuple[StageCost, ...] = ()
    planner_model: str | None = None
    planner_fallback: bool = False
    refuter_model: str | None = None
    wall_seconds: float = Field(ge=0)


@dataclass(frozen=True)
class ReceiptView:
    """Latest finding states alongside rollups of every incremental cost."""

    target_id: str
    entries: tuple[LedgerEntry, ...]
    rollup: Rollup
    summary: ReceiptSummary | None
    run_percentiles: Percentiles | None
    base_shas: tuple[str, ...]
    patch_shas: tuple[str, ...]
    versions: tuple[str, ...]
    latest_at: str


class UnknownTargetError(ValueError):
    """There are no entries for this target."""


def read_summary(path: Path) -> ReceiptSummary:
    return ReceiptSummary.model_validate_json(path.read_bytes())


def build_view(
    target_id: str,
    ledger: JsonlLedger,
    *,
    summary: ReceiptSummary | None = None,
    run_ledger: JsonlLedger | None = None,
) -> ReceiptView:
    entries = ledger.entries(target_id)
    if not entries:
        raise UnknownTargetError(f"no entries for target {target_id}")
    if summary is not None and summary.target_id != target_id:
        raise ValueError("the summary target_id does not match the requested target")
    latest = {entry.finding.id: entry for entry in entries}
    keys = [entry.finding.replay for entry in entries if entry.finding.replay is not None]
    percentiles = run_ledger.rollup().percentiles if run_ledger is not None else None
    if percentiles is not None and percentiles.targets == 0:
        raise ValueError("the run ledger has no targets")
    return ReceiptView(
        target_id=target_id,
        entries=tuple(latest[key] for key in sorted(latest)),
        rollup=ledger.rollup(target_id),
        summary=summary,
        run_percentiles=percentiles,
        base_shas=tuple(sorted({key.base_sha for key in keys})),
        patch_shas=tuple(sorted({key.patch_sha for key in keys})),
        versions=tuple(sorted({entry.receipts_version for entry in entries})),
        latest_at=entries[-1].at,
    )


def _shown(text: str) -> str:
    return strip_links(visible(text, keep="\t\n"))


def _text(text: str) -> str:
    return html.escape(_shown(text), quote=True)


def _inline(text: str) -> str:
    return f"<code>{_text(text)}</code>"


def _cut(shown: str, limit: int | None, finding_id: str | None) -> str:
    if limit is None or len(shown) <= limit:
        return shown
    where = f"receipts replay {_shown(finding_id)}" if finding_id is not None else "the rest is in the summary"
    return shown[:limit] + f"\n... ({len(shown) - limit:,} more characters cut; {where})"


def _pre(text: str, finding_id: str | None, limit: int | None = QUOTE_LIMIT) -> str:
    return "<pre><code>" + html.escape(_cut(_shown(text), limit, finding_id), quote=True) + "</code></pre>"


def _table(headers: tuple[str, ...], rows: Sequence[tuple[str, ...]], caption: str) -> str:
    # Cells are already rendered by _text, _inline or _pre. Headers and captions are fixed renderer text.
    return (
        '<div class="table-scroll" tabindex="0">\n<table>\n'
        f"<caption>{caption}</caption>\n<thead><tr>"
        + "".join(f'<th scope="col">{header}</th>' for header in headers)
        + "</tr></thead>\n<tbody>\n"
        + "\n".join("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>" for row in rows)
        + "\n</tbody></table>\n</div>"
    )


def _stages(view: ReceiptView) -> tuple[StageRollup, ...]:
    costs = list(view.summary.costs) if view.summary is not None else []
    groups: dict[tuple[str, str | None], list[StageCost | StageRollup]] = {}
    stage_costs: list[StageRollup | StageCost] = [*view.rollup.stages, *costs]
    for cost in stage_costs:
        groups.setdefault((cost.stage, cost.model), []).append(cost)
    return tuple(
        StageRollup(
            stage=stage,
            model=model,
            records=sum(c.records if isinstance(c, StageRollup) else 1 for c in group),
            input_tokens=sum(c.input_tokens for c in group),
            cached_tokens=sum(c.cached_tokens for c in group),
            output_tokens=sum(c.output_tokens for c in group),
            reasoning_tokens=sum(c.reasoning_tokens for c in group),
            micro_usd=sum(c.micro_usd for c in group),
            seconds=math.fsum(c.seconds for c in group),
        )
        for (stage, model), group in sorted(groups.items(), key=lambda pair: (pair[0][0], pair[0][1] or ""))
    )


def _verdict(view: ReceiptView) -> str:
    counts = view.rollup.totals.statuses
    if counts[Status.BLOCKER]:
        count = counts[Status.BLOCKER]
        return f"{count} blocker" + ("s" if count != 1 else "")
    if counts[Status.SUSPECTED]:
        return "suspected only"
    if counts[Status.UNCHECKED] == view.rollup.totals.findings:
        return "nothing checked"
    return "nothing caught"


def _files(finding: Finding) -> str:
    assert finding.repro is not None
    parts: list[str] = []
    remaining = FILES_LIMIT
    files = sorted(finding.repro.files.items())
    for index, (path, content) in enumerate(files):
        name = _cut(_shown(path), PATH_LIMIT, finding.id)
        heading = f"File: {name}\n"
        if remaining <= len(heading):
            parts.append("<p>" + _text(f"{len(files) - index} files cut; receipts replay {finding.id}") + "</p>")
            break
        shown = _shown(content)
        limit = min(FILE_LIMIT, remaining - len(heading))
        parts.append(
            "<pre><code>" + html.escape(heading + _cut(shown, limit, finding.id), quote=True) + "</code></pre>"
        )
        remaining -= len(heading) + min(len(shown), limit)
    return "\n".join(parts)


def _blocker(finding: Finding, summary: ReceiptSummary | None) -> str:
    assert finding.repro is not None and finding.kind is not None and finding.replay is not None
    repro, key = finding.repro, finding.replay
    claim = next((c for c in summary.claims if c.id == finding.claim_id), None) if summary is not None else None
    parts = [
        "<article>",
        f"<h3>{_text(KIND_PHRASES[finding.kind])}</h3>",
        f"<p>{_inline(finding.id)}</p>",
        "<h4>Quoted claim</h4>",
        _pre(finding.quote or (claim.span.text if claim is not None else repro.quote), finding.id),
    ]
    if finding.detail:
        parts += ["<h4>Detail</h4>", _pre(finding.detail, finding.id)]
    if finding.refutation:
        parts += [
            "<h4>A different model's refutation, which did not hold</h4>",
            _pre(finding.refutation, finding.id),
        ]
    parts += [
        "<h4>Repro files</h4>",
        _files(finding),
        "<h4>Command</h4>",
        _pre(shlex.join(repro.command), finding.id, COMMAND_LIMIT),
    ]
    if repro.cases:
        parts += ["<h4>Cases</h4>", _pre("\n".join(repro.cases), finding.id, CASES_LIMIT)]
    rows = [
        (
            _text(run.side.value),
            _text(run.outcome.value),
            _text(run.failure_type.value if run.failure_type is not None else "none"),
            "yes" if run.fresh_container else "no",
            _text(str(run.seed) if run.seed is not None else "not recorded"),
        )
        for run in finding.runs
    ]
    parts.append(_table(("Side", "Outcome", "Failure type", "Fresh", "Seed"), rows, "Runs"))
    for number, run in enumerate(finding.runs, start=1):
        if run.stdout_tail or run.stderr_tail or run.problem:
            parts.append(f"<h4>Run {number} output</h4>")
        for label, text in (("stdout", run.stdout_tail), ("stderr", run.stderr_tail), ("problem", run.problem)):
            if text:
                parts += [f"<p>{label}</p>", _pre(text, finding.id, OUTPUT_LIMIT)]
    parts += [
        "<h4>Replay key</h4>",
        _pre(
            f"base_sha: {key.base_sha}\npatch_sha: {key.patch_sha}\nharness_hash: {key.harness_hash}\n"
            f"seed: {key.seed}\nimage_digest: {key.image_digest}",
            finding.id,
            limit=None,
        ),
        "</article>",
    ]
    return "\n".join(parts)


def _basis(basis: LatencyBasis) -> str:
    return {
        LatencyBasis.WALL: "latency (wall)",
        LatencyBasis.SUMMED_STAGE_SECONDS: "summed stage seconds (summed_stage_seconds)",
        LatencyBasis.MIXED: "mixed wall and summed stage seconds (mixed)",
    }[basis]


def _costs(view: ReceiptView) -> str:
    target = view.rollup.targets[0]
    summary_cost = sum(c.micro_usd for c in view.summary.costs) if view.summary is not None else 0
    scope = "including summary costs" if view.summary is not None else "ledger entries only"
    parts = [
        f"<p>Target cost: {_text(usd(target.micro_usd + summary_cost))} ({scope}).</p>",
        f"<p>Target {_text(_basis(target.latency_basis))}: {_text(f'{target.latency_seconds:,.3f}')} s.</p>",
    ]
    if view.summary is not None:
        parts.append(f"<p>Summary wall time: {_text(f'{view.summary.wall_seconds:,.3f}')} s (all arms).</p>")
    p = view.run_percentiles
    if p is None or p.targets <= 1:
        parts.append("<p>p50/p95 need more than one target (n=1)</p>")
    else:
        assert p.p50_micro_usd is not None and p.p95_micro_usd is not None
        assert p.p50_seconds is not None and p.p95_seconds is not None and p.latency_basis is not None
        parts += [
            f"<p>Run cost: p50 {_text(usd(p.p50_micro_usd))}; p95 {_text(usd(p.p95_micro_usd))} (n={p.targets}). "
            "Ledger entries only; summary costs are not included.</p>",
            f"<p>Run {_text(_basis(p.latency_basis))}: p50 {_text(f'{p.p50_seconds:,.3f}')} s; "
            f"p95 {_text(f'{p.p95_seconds:,.3f}')} s (n={p.targets}).</p>",
        ]
    return "\n".join(parts)


def render_receipt(view: ReceiptView) -> str:
    """Render deterministically. Every value from either input crosses the same text-only boundary."""
    summary = view.summary
    parts = [
        "<!doctype html>",
        '<html lang="en">',
        "<head>",
        f'<meta http-equiv="Content-Security-Policy" content="{html.escape(CSP, quote=True)}">',
        '<meta charset="utf-8">',
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        "<title>Falden Receipts</title>",
        f"<style>\n{CSS}</style>",
        "</head>",
        "<body><main>",
        "<header>",
        '<p class="eyebrow">One target, every finding</p>',
        "<h1>Falden Receipts</h1>",
        "<dl>",
    ]
    for label, value in (
        ("Target", view.target_id),
        ("Repo", "Not recorded"),
        ("Tier", summary.tier.value if summary is not None else "Not recorded"),
        ("Base SHA", "\n".join(view.base_shas) or "Not recorded"),
        ("Patch SHA", "\n".join(view.patch_shas) or "Not recorded"),
        ("Receipts version", ", ".join(view.versions)),
        ("Latest entry", view.latest_at),
    ):
        parts.append(f"<dt>{label}</dt><dd>{_inline(value)}</dd>")
    parts += [
        "</dl></header>",
        '<section aria-label="Verdict">',
        f'<p class="verdict">{_text(_verdict(view))}</p>',
        "<p>" + _text("; ".join(f"{s.value}: {view.rollup.totals.statuses[s]}" for s in Status)) + "</p>",
        "</section>",
        "<section><h2>Who checked what</h2>",
    ]
    rows = [
        (
            _text(c.stage),
            _inline(c.model if c.model is not None else "code"),
            _text(str(c.input_tokens)),
            _text(str(c.cached_tokens)),
            _text(str(c.output_tokens)),
            _text(str(c.reasoning_tokens)),
            _text(usd(c.micro_usd)),
            _text(f"{c.seconds:,.3f}"),
        )
        for c in _stages(view)
    ]
    if rows:
        parts.append(
            _table(("Stage", "Model", "Input", "Cached", "Output", "Reasoning", "Dollars", "Seconds"), rows, "Stages")
        )
    else:
        parts.append("<p>No stage costs were recorded.</p>")
    parts.append("<p>Reasoning tokens are included in output tokens.</p>")
    if summary is None:
        parts += [
            "<p>Intake and planning costs are not shown. No summary was supplied.</p>",
            f"<p>{_text(OUTSIDE_THE_LEDGERS)}</p>",
        ]
    else:
        parts.append(
            f"<p>Planner model: {_inline(summary.planner_model or 'not recorded')}. "
            f"Planner fell back: {'yes' if summary.planner_fallback else 'no'}. "
            f"Refuter model: {_inline(summary.refuter_model or 'none recorded')}.</p>"
        )
    parts += ["</section>", "<section><h2>Blockers</h2>"]
    blockers = [e.finding for e in view.entries if e.finding.status is Status.BLOCKER]
    parts += [_blocker(finding, summary) for finding in blockers] or ["<p>No blockers.</p>"]
    parts += ["</section>", "<section><h2>Suspected, flaky, refuted and the rest</h2>"]
    others = [e.finding for e in view.entries if e.finding.status not in (Status.BLOCKER, Status.UNCHECKED)]
    if others:
        parts.append("<ul>")
        for finding in others:
            parts.append(
                f"<li>{_text(finding.status.value)}; {_text(finding.lens.value)}; {_inline(finding.id)}"
                + _pre(finding.detail or "No detail recorded.", finding.id)
                + (
                    "<p>A different model's refutation</p>" + _pre(finding.refutation, finding.id)
                    if finding.refutation
                    else ""
                )
                + "</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>No other findings.</p>")
    parts += ["</section>", "<section><h2>Not checked</h2>"]
    unchecked = [e.finding for e in view.entries if e.finding.status is Status.UNCHECKED]
    planned = summary.plan.unchecked if summary is not None else ()
    if unchecked or planned:
        parts.append("<ul>")
        for finding in unchecked:
            assert finding.unchecked_reason is not None
            parts.append(
                f"<li>{_text(finding.lens.value)}; {_text(finding.unchecked_reason.value)}; {_inline(finding.id)}"
                + _pre(finding.detail or "No detail recorded.", finding.id)
                + "</li>"
            )
        for item in planned:
            parts.append(
                f"<li>Planned out: {_inline(item.what)}; {_text(item.reason.value)}"
                + _pre(item.detail or "No detail recorded.", None)
                + "</li>"
            )
        parts.append("</ul>")
    else:
        parts.append("<p>Nothing was left unchecked.</p>")
    parts += [
        "</section>",
        "<section><h2>Cost and latency</h2>",
        _costs(view),
        "</section>",
        "</main></body></html>",
    ]
    return "\n".join(parts) + "\n"
