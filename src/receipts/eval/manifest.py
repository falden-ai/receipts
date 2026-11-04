"""eval/manifest.json: every input of Tier A, pinned by hash, and one record per (entry, instance) on the pool.

The manifest is built only from files whose bytes are pinned: the upstream sources in `receipts.eval.sources`, the
rows JSONL derived from the two pinned datasets, each entry's pulled patches and reports (indexed by sha256), R24's
results (the run-phase gold check, `receipts.eval.r24`, with any reruns and the founder's drops) and the label
runner's results (`receipts.eval.labelrun`). The same inputs always give the same records and the same
`content_sha256` (which leaves out only `created_at`), and the sample's seed is that hash.

It stays a draft while anything a frozen sample needs is missing: the pool choice, a patch, the R24 verdict on an
instance's gold, or a run that decides a label. `freeze` refuses until nothing is, so a frozen sample is drawn only
from final labels and from instances whose gold passed in the run phase under runsc (spec section 3, R24). An
instance R24 drops needs nothing more: it is never sampled. Infrastructure that fails the same way on every run has
a way out only through the founder: `--gold-drop` for R24, `--label-drop` for a label run, each recorded with its
reason (the label drops in `labels.drops`); nothing is dropped on its own.
"""

from __future__ import annotations

import json
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Literal

from pydantic import Field

from receipts.contracts import Frozen
from receipts.eval.entries import Entry, ResolvedList, parse_resolved
from receipts.eval.hashing import SHA256, canonical_json, sha256_bytes, sha256_file, sha256_text
from receipts.eval.labelrun import LabelEvidence, LandedLabelDrop, RecordLabelEvidence
from receipts.eval.labels import Evidence, Label, Need, final_label, needs, provisional_label
from receipts.eval.pool import Pool, Verdict, load_agent_keys, load_verdicts, repo_of, universe, verdict_index
from receipts.eval.pool import derive_251 as _derive_251
from receipts.eval.r24 import R24Merge
from receipts.eval.reports import Report, parse_report, report_evidence
from receipts.eval.rows import ABS_ROWS_PATH, ROWS_PATH, Row, check_abs_row, load_abs
from receipts.eval.rows import load as load_rows_jsonl
from receipts.eval.sources import SOURCES, Source, source, verify_file

SCHEMA: Literal["receipts.eval.manifest/2"] = "receipts.eval.manifest/2"
PoolName = Literal["swe_abs_251", "swe_abs_206"]

# The founder's pool choice, 2026-10-27 (spec section 3): the 251 corrected SWE-ABS instances (arXiv v1).
FOUNDER_POOL_CHOICE: PoolName = "swe_abs_251"


class SourceRecord(Frozen):
    name: str
    url: str
    revision: str
    path: str
    sha256: str = Field(pattern=SHA256)
    size: int


class DerivedRecord(Frozen):
    """A file this lane derived from a pinned source, pinned in turn."""

    name: str
    derived_from: str
    path: str  # relative to eval/data/
    sha256: str = Field(pattern=SHA256)
    size: int
    how: str


class R24InputRecord(Frozen):
    role: Literal["base", "rerun", "drop"]
    sha256: str = Field(pattern=SHA256)
    run_ids: tuple[str, ...]
    lines: int


class GoldRunscRecord(Frozen):
    """Where the per-instance R24 verdicts came from: the run-phase check's results.jsonl, then each rerun (which may
    replace only an INFRA verdict), then the founder's drops (only of instances still unchecked)."""

    sha256: str = Field(pattern=SHA256)  # of the merged verdicts and their reasons
    inputs: tuple[R24InputRecord, ...]
    instances: int
    kept: int
    dropped: int
    unchecked: int  # INFRA: rerun before reading anything into them
    founder_drops: int = 0


class LabelRunRecord(Frozen):
    """Where our label runs came from (labelrun's results.jsonl), and how they compare with SWE-ABS's published
    verdicts on the same resolved patches."""

    sha256: str = Field(pattern=SHA256)  # of the results file's text
    run_ids: tuple[str, ...]
    final_results: int
    gold_oracles: int  # instances whose gold passes the SWE-ABS tests in our run
    no_oracle: int  # instances whose gold does not
    agree_with_published: int  # resolved records where our verdict equals SWE-ABS's published one
    disagree_with_published: int
    rule_disagreements: int  # records where our rule and SWE-ABS's own rule read the same run differently
    drops: tuple[LandedLabelDrop, ...] = ()  # the founder's label drops, each with its reason, jobs and records


class PoolRecord(Frozen):
    name: PoolName
    available: bool
    size: int | None
    per_repo: dict[str, int] | None
    rule: str
    checks: dict[str, str]
    list_file: str | None  # under eval/pools/, committed
    list_sha256: str | None
    why_unavailable: str = ""


class EntryCounts(Frozen):
    instances: int
    resolved: int
    unresolved: int
    no_generation: int
    pool_instances: int
    pool_resolved: int
    pool_unresolved: int
    pool_swe_abs_published: dict[str, int]  # passed / failed / error / not_run, on the pool
    pool_provisional: dict[str, int]  # provisional label counts, on the pool
    pool_final: dict[str, int]  # final label counts, on the pool
    patches_pulled: int
    reports_pulled: int
    empty_patches: int


class EntryRecord(Frozen):
    key: str
    folder: str
    agent: str
    agent_version: str
    bash_only: bool
    model: str
    model_id: str
    model_org: str
    open_weights: bool
    date: str
    published_resolved_pct: str
    resolved_source: str
    metadata_source: str
    patch_store: str
    patch_location: str
    logs_note: str
    counts: EntryCounts


class Record(Frozen):
    entry: str
    instance_id: str
    repo: str
    pools: tuple[PoolName, ...]
    resolved: bool | None  # the published results; None when they cannot say (no logs)
    no_generation: bool
    swe_abs_published: Literal["passed", "failed", "not_run", "error"] | None
    provisional: Label
    assumed: tuple[str, ...]
    final: Label
    needs: tuple[Need, ...]
    gold_runsc: bool | None = None  # R24: the instance's gold passed FAIL_TO_PASS and PASS_TO_PASS in the run phase
    patch_sha256: str | None = Field(default=None, pattern=SHA256)
    patch_bytes: int | None = None
    empty_patch: bool | None = None
    report_sha256: str | None = Field(default=None, pattern=SHA256)
    fail_to_pass_passed: bool | None = None  # from the published per-test report, or our FAIL_TO_PASS run
    pass_to_pass_passed: bool | None = None
    f2p_source: Literal["report", "run"] | None = None
    swe_abs_passed: bool | None = None  # our run of the SWE-ABS tests on this patch, our rule
    swe_abs_rule: bool | None = None  # SWE-ABS's own rule, read from the same run
    gold_swe_abs: bool | None = None  # the instance's gold passes the SWE-ABS tests in our run
    patch_applies: bool | None = None  # False: our harness run could not apply the patch


class Manifest(Frozen):
    format: Literal["receipts.eval.manifest/2"] = SCHEMA
    status: Literal["draft", "frozen"]
    created_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    pool_choice: PoolName | None
    blocked_on: tuple[str, ...]
    sources: tuple[SourceRecord, ...]
    derived: tuple[DerivedRecord, ...] = ()
    gold_runsc: GoldRunscRecord | None = None
    labels: LabelRunRecord | None = None
    pools: tuple[PoolRecord, ...]
    entries: tuple[EntryRecord, ...]
    records: tuple[Record, ...]

    def derived_file(self, name: str) -> DerivedRecord:
        for d in self.derived:
            if d.name == name:
                return d
        raise KeyError(name)


class PatchIndexEntry(Frozen):
    instance_id: str
    sha256: str = Field(pattern=SHA256)
    size: int = Field(ge=0)
    path: str  # relative to eval/data/
    report_sha256: str | None = Field(default=None, pattern=SHA256)
    report_size: int | None = Field(default=None, ge=0)
    report_path: str | None = None

    @property
    def has_report(self) -> bool:
        return self.report_path is not None


class PatchIndex(Frozen):
    entry: str
    store: str
    patches: tuple[PatchIndexEntry, ...]


def patch_index_path(data_dir: Path, entry: str) -> Path:
    return data_dir / "patches" / entry / "index.json"


def _check_bytes(file: Path, sha256: str, size: int, what: str) -> bytes:
    if not file.is_file():
        raise ValueError(f"{file}: {what} is missing")
    data = file.read_bytes()
    if len(data) != size or sha256_bytes(data) != sha256:
        raise ValueError(f"{file}: {what} bytes do not match its index entry")
    return data


def load_patch_index(data_dir: Path, entry: str) -> dict[str, PatchIndexEntry]:
    path = patch_index_path(data_dir, entry)
    if not path.is_file():
        return {}
    index = PatchIndex.model_validate_json(path.read_text())
    if index.entry != entry:
        raise ValueError(f"{path}: index is for {index.entry}, not {entry}")
    out = {}
    for p in index.patches:
        _check_bytes(data_dir / p.path, p.sha256, p.size, "patch")
        if p.report_path is not None:
            if p.report_sha256 is None or p.report_size is None:
                raise ValueError(f"{path}: {p.instance_id} names a report without its hash")
            _check_bytes(data_dir / p.report_path, p.report_sha256, p.report_size, "report")
        out[p.instance_id] = p
    return out


def load_reports(data_dir: Path, index: Mapping[str, PatchIndexEntry]) -> dict[str, Report]:
    return {
        iid: parse_report((data_dir / p.report_path).read_bytes(), iid)
        for iid, p in index.items()
        if p.report_path is not None
    }


def patch_text(data_dir: Path, p: PatchIndexEntry) -> str:
    return _check_bytes(data_dir / p.path, p.sha256, p.size, "patch").decode("utf-8")


def _source_record(s: Source) -> SourceRecord:
    return SourceRecord(name=s.name, url=s.url, revision=s.revision, path=s.path, sha256=s.sha256, size=s.size)


def pool_list_text(pool: Pool) -> str:
    return "".join(f"{i}\n" for i in pool.instance_ids)


def pool_records(pool_251: Pool, list_dir: str = "eval/pools") -> tuple[PoolRecord, ...]:
    text = pool_list_text(pool_251)
    return (
        PoolRecord(
            name="swe_abs_251",
            available=True,
            size=pool_251.size,
            per_repo=pool_251.per_repo,
            rule=pool_251.rule,
            checks=pool_251.checks,
            list_file=f"{list_dir}/swe_abs_251.txt",
            list_sha256=sha256_text(text),
        ),
        PoolRecord(
            name="swe_abs_206",
            available=False,
            size=None,
            per_repo=None,
            rule=(
                "swe_abs_251 minus the 53 instances with overfitting tests (arXiv 2603.00520v1 section 4.4.1, Table 2 "
                "SWE-ABS*)"
            ),
            checks={},
            list_file=None,
            list_sha256=None,
            why_unavailable=(
                "the 53 overfitting instances are not named in the paper (arXiv 2603.00520v1), the repository "
                "(OpenAgentEval/SWE-ABS at 26b4c736) or the dataset (OpenAgentLab/SWE-Bench_Verified_ABS, one content "
                "version since 2026-03-01); 251 - 206 = 45 of them are strengthened, and no published file says which"
            ),
        ),
    )


def _gold(gold_runsc: Mapping[str, bool | None] | None, iid: str) -> bool | None:
    return None if gold_runsc is None else gold_runsc.get(iid)


def build_records(
    entry: Entry,
    resolved: ResolvedList,
    pool_ids: Sequence[str],
    verdicts: Mapping[tuple[str, str], Verdict],
    patches: Mapping[str, PatchIndexEntry],
    reports: Mapping[str, Report],
    rows: Mapping[str, Row],
    gold_runsc: Mapping[str, bool | None] | None = None,
    labels: LabelEvidence | None = None,
) -> list[Record]:
    out = []
    for iid in pool_ids:
        if iid not in rows:
            raise ValueError(f"{iid}: no SWE-bench Verified row")
        v = verdicts.get((entry.swe_abs_agent_id, iid))
        is_resolved = resolved.resolved_status(iid)
        if (
            v is not None
            and v.original_tests_passed is not None
            and is_resolved is not None
            and v.original_tests_passed != is_resolved
        ):
            raise ValueError(f"{entry.key} {iid}: SWE-ABS's original verdict disagrees with the published result")
        published = None if v is None or v.re_evaluation_status in ("not_run", "error") else v.swe_abs_tests_passed
        p = patches.get(iid)
        report = reports.get(iid)
        if report is not None and p is None:
            raise ValueError(f"{entry.key} {iid}: a report without its patch")
        read = (
            None
            if report is None
            else report_evidence(
                report, iid, rows[iid].FAIL_TO_PASS, rows[iid].PASS_TO_PASS, published_resolved=is_resolved
            )
        )
        ours = (
            RecordLabelEvidence() if labels is None else labels.records.get(f"{entry.key}~{iid}", RecordLabelEvidence())
        )
        dropped = labels is not None and f"{entry.key}~{iid}" in labels.dropped
        f2p, p2p = (None, None) if read is None else (read.fail_to_pass_passed, read.pass_to_pass_passed)
        source: Literal["report", "run"] | None = "report" if f2p is not None else None
        if ours.fail_to_pass_passed is not None:
            if f2p is not None and (f2p, p2p) != (ours.fail_to_pass_passed, ours.pass_to_pass_passed):
                raise ValueError(f"{entry.key} {iid}: our FAIL_TO_PASS run disagrees with the published report")
            f2p, p2p, source = ours.fail_to_pass_passed, ours.pass_to_pass_passed, "run"
        empty = True if iid in resolved.no_generation else (None if p is None else _is_empty(p))
        evidence = Evidence(
            resolved=is_resolved,
            fail_to_pass_passed=f2p,
            pass_to_pass_passed=p2p,
            empty_patch=empty,
            swe_abs_published=published,
            swe_abs_passed=ours.swe_abs_passed,
            gold_passes_swe_abs=None if labels is None else labels.gold.get(iid),
            patch_applies=ours.patch_applies,
            patch_pulled=p is not None,
            gold_passes_runsc=_gold(gold_runsc, iid),
            label_dropped=dropped,
        )
        prov = provisional_label(evidence)
        out.append(
            Record(
                entry=entry.key,
                instance_id=iid,
                repo=repo_of(iid),
                pools=("swe_abs_251",),
                resolved=is_resolved,
                no_generation=iid in resolved.no_generation,
                swe_abs_published=None if v is None else v.re_evaluation_status,
                provisional=prov.label,
                assumed=prov.assumed,
                final=final_label(evidence),
                needs=needs(evidence),
                gold_runsc=evidence.gold_passes_runsc,
                patch_sha256=None if p is None else p.sha256,
                patch_bytes=None if p is None else p.size,
                empty_patch=empty,
                report_sha256=None if p is None else p.report_sha256,
                fail_to_pass_passed=evidence.fail_to_pass_passed,
                pass_to_pass_passed=evidence.pass_to_pass_passed,
                f2p_source=source,
                swe_abs_passed=evidence.swe_abs_passed,
                swe_abs_rule=ours.swe_abs_rule,
                gold_swe_abs=evidence.gold_passes_swe_abs,
                patch_applies=evidence.patch_applies,
            )
        )
    return out


def _is_empty(p: PatchIndexEntry) -> bool:
    """An empty patch.diff is an agent that submitted nothing. Whitespace-only files are 1 or 2 bytes."""
    return p.size <= 2 and p.sha256 in _EMPTY_PATCH_HASHES


_EMPTY_PATCH_HASHES = frozenset(sha256_bytes(b) for b in (b"", b"\n", b" ", b"\r\n", b"\n\n"))


def _counts(entry: Entry, resolved: ResolvedList, records: Sequence[Record]) -> EntryCounts:
    return EntryCounts(
        instances=len(resolved.reported),
        resolved=len(resolved.resolved),
        unresolved=len(resolved.unresolved),
        no_generation=len(resolved.no_generation),
        pool_instances=len(records),
        pool_resolved=sum(1 for r in records if r.resolved is True),
        pool_unresolved=sum(1 for r in records if r.resolved is False),
        pool_swe_abs_published=dict(sorted(Counter(r.swe_abs_published or "none" for r in records).items())),
        pool_provisional=dict(sorted(Counter(r.provisional.value for r in records).items())),
        pool_final=dict(sorted(Counter(r.final.value for r in records).items())),
        patches_pulled=sum(1 for r in records if r.patch_sha256 is not None),
        reports_pulled=sum(1 for r in records if r.report_sha256 is not None),
        empty_patches=sum(1 for r in records if r.empty_patch is True),
    )


def entry_record(entry: Entry, counts: EntryCounts) -> EntryRecord:
    return EntryRecord(
        key=entry.key,
        folder=entry.folder,
        agent=entry.agent.value,
        agent_version=entry.agent_version,
        bash_only=entry.bash_only,
        model=entry.model,
        model_id=entry.model_id,
        model_org=entry.model_org,
        open_weights=entry.open_weights,
        date=entry.date,
        published_resolved_pct=entry.published_resolved_pct,
        resolved_source=entry.resolved_source,
        metadata_source=entry.metadata_source,
        patch_store=entry.patch_store.value,
        patch_location=entry.patch_location,
        logs_note=entry.logs_note,
        counts=counts,
    )


def blockers(records: Iterable[Record], pool_choice: PoolName | None) -> tuple[str, ...]:
    """Everything that stands between this manifest and a frozen sample, in the order it gets done. Records on an
    instance R24 dropped stand in no one's way."""
    recs = list(records)
    out = []
    if pool_choice is None:
        out.append("pool choice: swe_abs_251 or swe_abs_206 (the founder's, before the pilot)")
    missing = Counter(r.entry for r in recs if Need.PATCH in r.needs)
    for entry, n in sorted(missing.items()):
        out.append(f"patches: {n} not pulled for {entry}")
    unchecked = sorted({r.instance_id for r in recs if Need.GOLD_RUNSC in r.needs})
    if unchecked:
        out.append(
            f"gold runsc: {len(unchecked)} instances not yet checked in the run phase under runsc "
            "(receipts eval r24, then manifest --gold-runsc)"
        )
    swe_abs = sum(1 for r in recs if Need.SWE_ABS_RUN in r.needs)
    if swe_abs:
        out.append(f"labels: {swe_abs} resolved patches need our run of the SWE-ABS tests (GOOD or PLAUSIBLE_WRONG)")
    gold_abs = len({r.instance_id for r in recs if Need.GOLD_SWE_ABS in r.needs})
    if gold_abs:
        out.append(f"labels: {gold_abs} instances need our run of the SWE-ABS tests on gold")
    f2p = sum(1 for r in recs if Need.FAIL_TO_PASS_DETAIL in r.needs)
    if f2p:
        out.append(f"labels: {f2p} unresolved patches need a FAIL_TO_PASS run (no published per-test report)")
    pending = sum(1 for r in recs if r.final is Label.PENDING and not r.needs and r.gold_runsc is not False)
    if pending:
        out.append(f"labels: {pending} records are PENDING with nothing left to run")
    return tuple(out)


def derive_251(data_dir: Path) -> Pool:
    verdicts = load_verdicts(verify_file(source("swe-abs:verdicts"), data_dir).read_bytes())
    agents = load_agent_keys(verify_file(source("swe-abs:key-map-top30"), data_dir).read_bytes())
    return _derive_251(verdicts, agents)


def _derived(data_dir: Path, name: str, derived_from: str, path: str, how: str) -> DerivedRecord:
    file = data_dir / path
    if not file.is_file():
        raise ValueError(f"{file} is missing; run `uv run --frozen --extra eval python -m receipts.eval rows`")
    return DerivedRecord(
        name=name, derived_from=derived_from, path=path, sha256=sha256_file(file), size=file.stat().st_size, how=how
    )


def rows_record(data_dir: Path) -> DerivedRecord:
    return _derived(
        data_dir,
        "swe-bench-verified:rows",
        "swe-bench-verified:dataset",
        ROWS_PATH,
        "receipts.eval.rows.convert: one canonical JSON line per row, sorted by instance id, test lists decoded",
    )


def abs_rows_record(data_dir: Path) -> DerivedRecord:
    return _derived(
        data_dir,
        "swe-abs:rows",
        "swe-abs:dataset",
        ABS_ROWS_PATH,
        "receipts.eval.rows.convert_abs: one canonical JSON line per row, sorted by instance id, test lists decoded",
    )


def label_run_record(
    text: str,
    labels: LabelEvidence,
    run_ids: Iterable[str],
    final: int,
    drops: Sequence[LandedLabelDrop] = (),
) -> LabelRunRecord:
    """Provenance for merged label results; the agreement counts are filled from the built records."""
    return LabelRunRecord(
        sha256=sha256_text(text),
        run_ids=tuple(sorted(set(run_ids))),
        final_results=final,
        gold_oracles=sum(1 for ok in labels.gold.values() if ok),
        no_oracle=sum(1 for ok in labels.gold.values() if not ok),
        agree_with_published=0,
        disagree_with_published=0,
        rule_disagreements=0,
        drops=tuple(drops),
    )


def _agreement(record: LabelRunRecord, records: Sequence[Record]) -> LabelRunRecord:
    ran = [r for r in records if r.swe_abs_passed is not None]
    published = [r for r in ran if r.swe_abs_published in ("passed", "failed")]
    agree = sum(1 for r in published if r.swe_abs_passed == (r.swe_abs_published == "passed"))
    return record.model_copy(
        update={
            "agree_with_published": agree,
            "disagree_with_published": len(published) - agree,
            "rule_disagreements": sum(
                1 for r in ran if r.swe_abs_rule is not None and r.swe_abs_rule != r.swe_abs_passed
            ),
        }
    )


def gold_record(merge: R24Merge) -> GoldRunscRecord:
    v = merge.verdicts
    return GoldRunscRecord(
        sha256=merge.sha256,
        inputs=tuple(
            R24InputRecord(role=i.role, sha256=i.sha256, run_ids=i.run_ids, lines=i.lines) for i in merge.inputs
        ),
        instances=len(v),
        kept=sum(1 for x in v.values() if x is True),
        dropped=sum(1 for x in v.values() if x is False),
        unchecked=sum(1 for x in v.values() if x is None),
        founder_drops=sum(1 for why in merge.reasons.values() if why.startswith("founder: ")),
    )


def check_label_keys(labels: LabelEvidence, known: set[str]) -> None:
    """Label results and the founder's label drops may name only records the manifest has."""
    for what, keys in (("label drops", set(labels.dropped)), ("label results", set(labels.records))):
        if keys - known:
            raise ValueError(f"{what} for records the manifest does not have: {sorted(keys - known)[:3]}")


def build_manifest(
    data_dir: Path,
    entries: Sequence[Entry],
    created_at: str,
    *,
    pool_choice: PoolName | None = FOUNDER_POOL_CHOICE,
    gold: R24Merge | None = None,
    labels: LabelEvidence | None = None,
    label_record: LabelRunRecord | None = None,
) -> tuple[Manifest, Pool]:
    if (labels is None) != (label_record is None):
        raise ValueError("label evidence and its provenance come together")
    for s in SOURCES:
        verify_file(s, data_dir)
    verdict_file = load_verdicts((data_dir / source("swe-abs:verdicts").path).read_bytes())
    agents = load_agent_keys((data_dir / source("swe-abs:key-map-top30").path).read_bytes())
    pool = _derive_251(verdict_file, agents)
    all_ids = universe(verdict_file)
    index = verdict_index(verdict_file)
    rows_pin, abs_pin = rows_record(data_dir), abs_rows_record(data_dir)
    rows = load_rows_jsonl(data_dir / rows_pin.path, expect_sha256=rows_pin.sha256)
    abs_rows = load_abs(data_dir / abs_pin.path, expect_sha256=abs_pin.sha256)
    if set(rows) != set(all_ids) or set(abs_rows) != set(all_ids):
        raise ValueError("the rows JSONL files do not cover exactly the instances SWE-ABS re-evaluated")
    for iid in pool.instance_ids:
        check_abs_row(abs_rows[iid], rows[iid])
    gold_runsc = None if gold is None else gold.verdicts
    if gold_runsc is not None and set(gold_runsc) != set(pool.instance_ids):
        raise ValueError("the R24 verdicts do not cover exactly the pool")
    if labels is not None:
        stray = set(labels.gold) - set(pool.instance_ids)
        if stray:
            raise ValueError(f"label results for instances off the pool: {sorted(stray)[:3]}")
    records: list[Record] = []
    entry_records = []
    for entry in entries:
        raw = (data_dir / source(entry.resolved_source).path).read_bytes()
        resolved = parse_resolved(entry, raw, all_ids)
        patches = load_patch_index(data_dir, entry.key)
        mine = build_records(
            entry,
            resolved,
            pool.instance_ids,
            index,
            patches,
            load_reports(data_dir, patches),
            rows,
            gold_runsc,
            labels,
        )
        records.extend(mine)
        entry_records.append(entry_record(entry, _counts(entry, resolved, mine)))
    if labels is not None:
        check_label_keys(labels, {f"{r.entry}~{r.instance_id}" for r in records})
    manifest = Manifest(
        status="draft",
        created_at=created_at,
        pool_choice=pool_choice,
        blocked_on=blockers(records, pool_choice),
        sources=tuple(_source_record(s) for s in SOURCES),
        derived=(rows_pin, abs_pin),
        gold_runsc=None if gold is None else gold_record(gold),
        labels=None if label_record is None else _agreement(label_record, records),
        pools=pool_records(pool),
        entries=tuple(entry_records),
        records=tuple(records),
    )
    return manifest, pool


class NotFreezableError(ValueError):
    """A manifest that still has blockers."""


def freeze(manifest: Manifest) -> Manifest:
    """Mark a manifest frozen; refuses while the pool choice, a patch, an R24 verdict or a label run is missing."""
    if manifest.pool_choice is None:
        raise NotFreezableError("no pool choice")
    chosen = next(p for p in manifest.pools if p.name == manifest.pool_choice)
    if not chosen.available:
        raise NotFreezableError(f"{chosen.name} is not available: {chosen.why_unavailable}")
    left = blockers(manifest.records, manifest.pool_choice)
    if left:
        raise NotFreezableError("; ".join(left))
    if manifest.gold_runsc is None:
        raise NotFreezableError("no R24 results are recorded")
    if manifest.labels is None and any(r.resolved for r in manifest.records):
        raise NotFreezableError("no label-run results are recorded")
    return manifest.model_copy(update={"status": "frozen", "blocked_on": ()})


def dump(manifest: Manifest) -> str:
    """Sorted, indented JSON with one compact line per record, so a diff shows exactly which records moved."""
    data = manifest.model_dump(mode="json")
    records = data.pop("records")
    head = json.dumps(data, sort_keys=True, indent=2, ensure_ascii=False)
    body = ",\n".join(f"    {canonical_json(r)}" for r in records)
    return head[: head.rindex("}")].rstrip() + ',\n  "records": [\n' + body + "\n  ]\n}\n"


def load(text: str) -> Manifest:
    return Manifest.model_validate_json(text)


def content_sha256(manifest: Manifest) -> str:
    """The hash a sample binds to: everything in the manifest except when it was written."""
    return sha256_text(canonical_json(manifest.model_dump(mode="json", exclude={"created_at"})))


def label_counts(records: Iterable[Record], *, provisional: bool = True) -> dict[str, int]:
    return dict(sorted(Counter((r.provisional if provisional else r.final).value for r in records).items()))
