"""`receipts eval ...` (or `python -m receipts.eval ...`): the lane's commands. Every handler returns its exit code.

Run from the repository root. Downloads land in eval/data/ (not committed); everything else is written under eval/
and committed. No command here talks to a model or uses AWS credentials. `fetch`, `images` and `patches` read public
files over HTTPS, the last from the public bucket s3://swe-bench-submissions with anonymous GETs (approved by the
founder, 2026-10-27). `rows` needs pyarrow and `runsc-check`, `labels` and `r24 --diagnose` need swebench 4.1.0, both
from the project's eval extra (`uv sync --frozen --extra eval`; uv.lock pins every file). `runsc-check`, `labels` and
`r24` are for the eval VM only; `r24` and `score` run the run phase's sandbox, which the integrator wires in
(`Wiring`), because the eval lane codes against contracts.Sandbox and contracts.SuiteRunner and never imports another
lane's code. eval/vm/quote.json's runbook gives each VM step as the command that runs.

Worker counts (`runsc-check`, `labels`, `r24`) default to one sandbox per 2 vCPU of the machine they run on, within its
memory (REG-568: 32 on the m7i.16xlarge that runs R24 and the labels, 16 on the pilot's m7i.8xlarge); --workers sets
one. `r24` proves the VM first (runsc is gVisor, runc is not) with the harness check's preflight.

Exit codes: 0 done; 2 refused or not ready (a check failed, the VM is not ready, results still unchecked, a gold
replay still unscored); 3 a result over its limit (too many drops) or a sample the candidates cannot fill.

Infrastructure that fails the same way on every run is settled only by the founder, by name and with a reason, and
never automatically: `manifest --gold-drop` for an R24 instance still unchecked, `manifest --label-drop` for an
instance whose label jobs were INFRA on every run, and `score --replay-drop` for a gold replay still unscored.
"""

from __future__ import annotations

import argparse
import datetime as dt
import functools
import json
import sys
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

from receipts.contracts import RunnerKind, Sandbox, SuiteRunner
from receipts.eval import entries as entries_mod
from receipts.eval.hashing import pretty_json, write_text_atomic

if TYPE_CHECKING:
    from receipts.eval import score as score_types
    from receipts.eval.labelrun import PlanInputs
    from receipts.eval.manifest import Manifest
    from receipts.eval.rows import Row
    from receipts.eval.sampler import Sample, SamplePlan


@dataclass(frozen=True)
class Wiring:
    """The run phase, from the integrator's composition root (src/receipts/cli.py): the attack lane's sandbox under
    runsc with the run phase's flags, and its runner for each RunnerKind."""

    sandbox: Callable[[], Sandbox]
    runners: Callable[[], Mapping[RunnerKind, SuiteRunner]]


def utc_now() -> str:
    return dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def repo_root() -> Path:
    here = Path.cwd().resolve()
    for p in (here, *here.parents):
        if (p / "pyproject.toml").is_file() and (p / "src" / "receipts").is_dir():
            return p
    raise SystemExit("run from inside the falden-receipts repository")


def _paths(args: argparse.Namespace) -> tuple[Path, Path]:
    root = Path(args.root).resolve() if args.root else repo_root()
    return root / "eval", root / "eval" / "data"


def _fail(message: str, code: int = 2) -> int:
    print(f"receipts eval: {message}", file=sys.stderr)
    return code


def _workers(args: argparse.Namespace) -> int:
    """--workers, or one sandbox per 2 vCPU of this machine within its memory (REG-568: never a constant)."""
    from receipts.eval.vmplan import default_workers, machine_vcpus

    n = int(args.workers) if args.workers is not None else default_workers()
    if n < 1:
        raise SystemExit("receipts eval: --workers is at least 1")
    why = "--workers" if args.workers is not None else f"{machine_vcpus()} vCPUs here, 2 per sandbox"
    print(f"workers: {n} ({why})", file=sys.stderr)
    return n


def cmd_fetch(args: argparse.Namespace) -> int:
    from receipts.eval.sources import SOURCES, fetch, make_client

    _, data = _paths(args)
    with make_client() as client:
        for s in SOURCES:
            path = fetch(s, data, client)
            print(f"ok {s.sha256[:12]} {s.size:>9} {path.relative_to(data.parent.parent)}")
    return 0


def cmd_rows(args: argparse.Namespace) -> int:
    from receipts.eval.rows import ABS_ROWS_PATH, ROWS_PATH, convert, convert_abs
    from receipts.eval.sources import source, verify_file

    _, data = _paths(args)
    try:
        n = convert(verify_file(source("swe-bench-verified:dataset"), data), data / ROWS_PATH)
        m = convert_abs(verify_file(source("swe-abs:dataset"), data), data / ABS_ROWS_PATH)
    except ModuleNotFoundError:
        return _fail("rows needs pyarrow: uv run --frozen --extra eval python -m receipts.eval rows")
    print(f"wrote {n} rows to {data / ROWS_PATH} and {m} to {data / ABS_ROWS_PATH}")
    return 0


def cmd_pool(args: argparse.Namespace) -> int:
    from receipts.eval.manifest import derive_251, pool_list_text, pool_records

    ev, data = _paths(args)
    pool = derive_251(data)
    write_text_atomic(ev / "pools" / "swe_abs_251.txt", pool_list_text(pool))
    records = pool_records(pool)
    write_text_atomic(ev / "pools" / "pools.json", pretty_json([r.model_dump(mode="json") for r in records]))
    print(f"swe_abs_251: {pool.size} instances; checks: {json.dumps(pool.checks)}")
    print(f"swe_abs_206: not derivable: {records[1].why_unavailable}")
    return 0


def _rows(manifest: Manifest, data: Path) -> dict[str, Row]:
    from receipts.eval.rows import load as load_rows

    pin = manifest.derived_file("swe-bench-verified:rows")
    return load_rows(data / pin.path, expect_sha256=pin.sha256)


def cmd_test_specs(args: argparse.Namespace) -> int:
    from receipts.eval.rows import ROWS_PATH, load_unpinned
    from receipts.eval.sources import source, verify_file
    from receipts.eval.specs import SPECS_FILE, derive, dump

    ev, data = _paths(args)
    wheel = source("swebench:wheel")
    rows, _ = load_unpinned(data / ROWS_PATH)
    pool_ids = (ev / "pools" / "swe_abs_251.txt").read_text().split()
    instances = {i: (rows[i].repo, rows[i].version) for i in pool_ids}
    table = derive(verify_file(wheel, data), wheel.sha256, set(instances.values()), instances=instances)
    write_text_atomic(ev / SPECS_FILE, dump(table))
    print(
        f"wrote {ev / SPECS_FILE}: {len(table.specs)} repository versions from swebench {table.swebench_version}, "
        f"and the version of each of its {len(table.instances)} pool instances"
    )
    return 0


def plan_inputs(manifest: Manifest, rows: Mapping[str, Row], *, with_drops: bool = True) -> PlanInputs:
    """The label runner's plan from a manifest: gold on every pool instance, the SWE-ABS tests on every resolved
    pulled non-empty patch, FAIL_TO_PASS on every unresolved one whose published report does not settle it."""
    from receipts.eval.hashing import sha256_text
    from receipts.eval.labelrun import PlanInputs

    pool = sorted({r.instance_id for r in manifest.records})
    usable = [r for r in manifest.records if r.patch_sha256 is not None and r.empty_patch is not True]
    return PlanInputs(
        pool=tuple(pool),
        dropped=frozenset(r.instance_id for r in manifest.records if r.gold_runsc is False and with_drops),
        resolved=tuple((r.entry, r.instance_id, r.patch_sha256 or "") for r in usable if r.resolved is True),
        no_report=tuple(
            (r.entry, r.instance_id, r.patch_sha256 or "")
            for r in usable
            if r.resolved is False and r.f2p_source != "report"  # no report, or one that could not say
        ),
        gold_sha256={i: sha256_text(rows[i].patch) for i in pool},
    )


def cmd_manifest(args: argparse.Namespace) -> int:
    from receipts.eval import labelrun, r24
    from receipts.eval.manifest import (
        NotFreezableError,
        build_manifest,
        dump,
        freeze,
        label_counts,
        label_run_record,
    )

    ev, data = _paths(args)
    entries_mod.check_selection(entries_mod.SELECTED)
    pool_ids = (ev / "pools" / "swe_abs_251.txt").read_text().split()
    gold = None
    if (args.gold_rerun or args.gold_drop) and not args.gold_runsc:
        return _fail("--gold-rerun and --gold-drop add to an R24 result: give --gold-runsc too")
    if args.label_drop and not (args.labels and args.gold_runsc):
        return _fail(
            "--label-drop needs --labels and --gold-runsc: a drop lands only on an instance R24 kept whose label jobs "
            "were INFRA on every run"
        )
    if args.gold_runsc:
        from receipts.eval.images import ImageSet

        try:
            pins = ImageSet.model_validate_json((ev / "images.json").read_text()).by_instance
            base = (Path(args.gold_runsc) / "results.jsonl").read_text()
            reruns = [(Path(d) / "results.jsonl").read_text() for d in args.gold_rerun or ()]
            drops = r24.parse_drops(Path(args.gold_drop).read_text()) if args.gold_drop else []
            images = {iid: ref.ref for iid, ref in pins.items()}
            gold = r24.merge(base, pool_ids, images, reruns=reruns, drops=drops)
        except (r24.R24ResultsError, OSError, ValueError) as e:
            return _fail(f"--gold-runsc refused: {e}")
    labels = record = None
    if args.labels:
        draft, _ = build_manifest(data, entries_mod.SELECTED, utc_now(), pool_choice=args.pool, gold=gold)
        rows = _rows(draft, data)
        planned = labelrun.plan(plan_inputs(draft, rows, with_drops=False))
        jobs = {j.id: j for j in planned}
        try:
            text = labelrun.results_path(Path(args.labels)).read_text()
            final = labelrun.read_results(text.splitlines(), jobs)
            label_drops = labelrun.land_drops(
                labelrun.parse_label_drops(args.label_drop or ()),
                planned,
                final,
                labelrun.pending_infra(text.splitlines(), jobs),
                {} if gold is None else gold.verdicts,
            )
        except (labelrun.LabelResultsError, OSError, ValueError) as e:
            return _fail(f"--labels refused: {e}")
        labels = labelrun.evidence(final, rows, label_drops)
        record = label_run_record(text, labels, (r.run_id for r in final.values()), len(final), label_drops)
    manifest, _ = build_manifest(
        data, entries_mod.SELECTED, utc_now(), pool_choice=args.pool, gold=gold, labels=labels, label_record=record
    )
    if args.freeze:
        try:
            manifest = freeze(manifest)
        except NotFreezableError as e:
            return _fail(f"not frozen: {e}")
    write_text_atomic(ev / "manifest.json", dump(manifest))
    for entry in manifest.entries:
        c = entry.counts
        print(
            f"{entry.key}: resolved {c.resolved}/{c.instances}; on the pool {c.pool_resolved} resolved, "
            f"{c.pool_unresolved} unresolved, {c.empty_patches} empty; final {c.pool_final}"
        )
    print(f"provisional labels on the pool: {label_counts(manifest.records)}")
    print(f"final labels on the pool: {label_counts(manifest.records, provisional=False)}")
    if manifest.labels is not None:
        print(f"label runs: {manifest.labels.model_dump(mode='json', exclude={'drops'})}")
        for d in manifest.labels.drops:
            print(
                f"label drop (the founder's): {d.instance_id}: {d.reason}; {len(d.jobs)} jobs INFRA on every run, "
                f"{len(d.records)} records DROPPED"
            )
    print(f"status {manifest.status}; blocked on: {list(manifest.blocked_on)}")
    return 0


def cmd_images(args: argparse.Namespace) -> int:
    from receipts.eval.images import resolve_all
    from receipts.eval.sources import make_client

    ev, _ = _paths(args)
    ids = (ev / "pools" / "swe_abs_251.txt").read_text().split()
    with make_client() as client:
        images = resolve_all(client, ids, utc_now(), workers=args.workers)
    write_text_atomic(ev / "images.json", pretty_json(images.model_dump(mode="json")))
    print(
        f"{len(images.images)} images pinned; unique compressed layers {images.unique_compressed_bytes / 2**30:.1f} GiB"
    )
    return 0


def cmd_patches(args: argparse.Namespace) -> int:
    from receipts.eval.patches import pull_entry
    from receipts.eval.sources import make_client

    ev, data = _paths(args)
    ids = (ev / "pools" / "swe_abs_251.txt").read_text().split()
    with make_client() as client:
        for e in entries_mod.SELECTED:
            index = pull_entry(e, ids, data, client, workers=args.workers)
            reports = sum(1 for p in index.patches if p.has_report)
            print(f"{e.key}: {len(index.patches)} patches and {reports} reports indexed")
    return 0


def _plan(name: str) -> SamplePlan:
    from receipts.eval.sampler import CORE_PLAN, SPEC_PLAN

    return {"spec": SPEC_PLAN, "core": CORE_PLAN}[name]


def supply_lines(manifest: Manifest, pool_text: str, plan: SamplePlan, seed: str) -> list[str]:
    """Candidates per label and side against what the plan draws, under the split this seed gives."""
    from receipts.eval.sampler import Side, candidates_from_manifest, pool_from_manifest, pool_ids_for, side_count
    from receipts.eval.sampler import split_pool as split

    sides = split(pool_from_manifest(manifest, pool_ids_for(manifest, pool_text)), seed, plan.calibration_pct)
    supply = Counter((c.label.value, sides[c.instance_id].value) for c in candidates_from_manifest(manifest))
    out = []
    for label, n in sorted(plan.full.items()):
        for side in (Side.CALIBRATION, Side.TEST):
            need = side_count(n, side, plan.calibration_pct)
            out.append(f"  {label.value} {side.value}: {supply[(label.value, side.value)]} candidates for {need} items")
    return out


def cmd_sample(args: argparse.Namespace) -> int:
    from receipts.eval.manifest import content_sha256, load
    from receipts.eval.sampler import InfeasibleSampleError, SampleCheckError, draw_from_manifest, pilot

    ev, _ = _paths(args)
    manifest = load((ev / "manifest.json").read_text())
    pool_text = (ev / "pools" / f"{manifest.pool_choice or 'swe_abs_251'}.txt").read_text()
    plan = _plan(args.plan)
    seed = args.seed or content_sha256(manifest)
    try:
        sample = draw_from_manifest(manifest, pool_text, plan, seed=args.seed)
    except SampleCheckError as e:
        return _fail(f"refused: {e}")
    except InfeasibleSampleError as e:
        print(f"receipts eval: infeasible under plan {args.plan}: {e}", file=sys.stderr)
        for line in supply_lines(manifest, pool_text, plan, seed):
            print(line, file=sys.stderr)
        return 3
    frozen = sample.status == "frozen"
    out = Path(args.out).resolve() if args.out else ev / ("sample.json" if frozen else "sample-provisional.json")
    write_text_atomic(out, pretty_json(sample.model_dump(mode="json")))
    print("supply (candidates per label and side, the plan's items):")
    for line in supply_lines(manifest, pool_text, plan, seed):
        print(line)
    for core_only in (True, False):
        items = sample.scope(core_only=core_only)
        print(("core" if core_only else "full"), dict(Counter((i.label.value, i.side.value) for i in items)))
        print("  repos", dict(Counter(i.repo for i in items).most_common()))
        print("  models", dict(Counter(i.model for i in items).most_common()))
    cal = sum(1 for s in sample.split.values() if s.value == "calibration")
    print(f"split: {cal} of {len(sample.split)} pool instances in calibration")
    print("pilot:", [f"{i.label.value}:{i.entry.split('_', 1)[-1]}:{i.instance_id}" for i in pilot(sample)])
    print(f"wrote {out} ({sample.status}, plan {args.plan})")
    return 0


def _verified_sample(ev: Path) -> tuple[Manifest, str, Sample]:
    """The manifest and its sample (sample.json once frozen), after verify_sample: the exact seeded draw."""
    from receipts.eval.manifest import load
    from receipts.eval.sampler import Sample, verify_sample

    manifest = load((ev / "manifest.json").read_text())
    name = "sample.json" if manifest.status == "frozen" else "sample-provisional.json"
    sample_text = (ev / name).read_text()
    sample = Sample.model_validate_json(sample_text)
    pool_text = (ev / "pools" / f"{manifest.pool_choice or 'swe_abs_251'}.txt").read_text()
    verify_sample(sample, manifest, pool_text)
    return manifest, sample_text, sample


def cmd_pilot(args: argparse.Namespace) -> int:
    from receipts.eval.images import ImageSet
    from receipts.eval.pilot import ARMS_FILE, RUN_FILE, PilotCheckError, build_pilot, verify_pilot
    from receipts.eval.sampler import SampleCheckError, pilot
    from receipts.eval.specs import SPECS_FILE
    from receipts.eval.specs import load as load_specs
    from receipts.eval.targets import TargetError, build_targets

    ev, data = _paths(args)
    try:
        manifest, sample_text, sample = _verified_sample(ev)
    except SampleCheckError as e:
        return _fail(f"the sample is not the seeded draw from this manifest: {e}")
    images = ImageSet.model_validate_json((ev / "images.json").read_text())
    specs = load_specs((ev / SPECS_FILE).read_text())
    try:
        targets = build_targets(pilot(sample), manifest, _rows(manifest, data), images, data, specs)
        cfg, run_text, arms_text = build_pilot(sample, sample_text, manifest, targets)
        verify_pilot(cfg, sample, sample_text, manifest, targets, run_text, arms_text)
    except (TargetError, PilotCheckError) as e:
        return _fail(f"pilot refused: {e}")
    root = ev.parent
    write_text_atomic(ev / "pilot.json", pretty_json(cfg.model_dump(mode="json")))
    write_text_atomic(root / RUN_FILE, run_text)
    write_text_atomic(root / ARMS_FILE, arms_text)
    modes = Counter((t.label.value, t.temperatures.value) for t in cfg.targets)
    print(f"pilot: {len(cfg.targets)} targets ({cfg.status}); temperatures {dict(sorted(modes.items()))}")
    print(f"arms: {[(a.name, a.attacker_model, a.step_limit, a.seed) for a in cfg.arms]}")
    print(f"wrote {ev / 'pilot.json'}, {root / RUN_FILE} (sha256 {cfg.run_file_sha256[:12]}) and {root / ARMS_FILE}")
    return 0


def cmd_targets(args: argparse.Namespace) -> int:
    from receipts.eval.images import ImageSet
    from receipts.eval.sampler import SampleCheckError
    from receipts.eval.specs import SPECS_FILE
    from receipts.eval.specs import load as load_specs
    from receipts.eval.targets import TargetError, build_targets, write_targets

    ev, data = _paths(args)
    try:
        manifest, _, sample = _verified_sample(ev)
    except SampleCheckError as e:
        return _fail(f"the sample is not the seeded draw from this manifest: {e}")
    images = ImageSet.model_validate_json((ev / "images.json").read_text())
    specs = load_specs((ev / SPECS_FILE).read_text())
    try:
        targets = build_targets(sample.items, manifest, _rows(manifest, data), images, data, specs)
    except TargetError as e:
        return _fail(f"targets refused: {e}")
    out = data / "targets" / ("sample.jsonl" if manifest.status == "frozen" else "sample-provisional.jsonl")
    digest = write_targets(out, targets)
    print(f"wrote {len(targets)} targets to {out} (sha256 {digest[:12]})")
    return 0


def cmd_quote(args: argparse.Namespace) -> int:
    from receipts.eval.vmplan import quote

    ev, _ = _paths(args)
    q = quote(args.instance)
    write_text_atomic(ev / "vm" / "quote.json", pretty_json(q.model_dump(mode="json")))
    for c in q.cases:
        hours = sum(c.run_minutes.values()) / 60
        print(f"{c.case}: {c.sandbox_minutes_per_target} sandbox-min/target, {hours:.1f} VM hours, total {c.total_usd}")
        day = c.pilot_day
        blocks = "; ".join(f"{b.what} on {b.instance_type} {b.minutes} min" for b in day.blocks)
        print(f"  {day.date}: {blocks}; eval VM {day.eval_vm_hours} h; compute {day.compute_usd}")
    print(f"label runs before the pilot: {q.label_runs}")
    p = q.pilot
    print(f"pilot: {p.targets} targets, --jobs {p.jobs} on {p.instance_type}, {p.waves} waves of {'+'.join(p.arms)}:")
    for name, pc in p.cases.items():
        print(
            f"  {name}: {pc.target_run_minutes} min a target run; eval VM {pc.eval_vm_minutes} min {pc.eval_vm_usd}; "
            f"GPU VM {pc.gpu_vm_minutes} min {pc.gpu_vm_usd}"
        )
    return 0


def cmd_cloud_init(args: argparse.Namespace) -> int:
    from receipts.eval.cloudinit import render
    from receipts.eval.hashing import write_bytes_atomic
    from receipts.eval.images import ImageSet

    ev, _ = _paths(args)
    images = ImageSet.model_validate_json((ev / "images.json").read_text())
    ids = Path(args.instances).read_text().split() if args.instances else [i.instance_id for i in images.images]
    data, gz = render(images.subset(ids))
    out = ev / "vm" / ("cloud-init.yaml.gz" if gz else "cloud-init.yaml")
    other = ev / "vm" / ("cloud-init.yaml" if gz else "cloud-init.yaml.gz")
    write_bytes_atomic(out, data)
    other.unlink(missing_ok=True)
    print(f"wrote {out} ({len(data)} bytes{', gzip' if gz else ''}) for {len(ids)} images")
    return 0


def cmd_vm_files(args: argparse.Namespace) -> int:
    from receipts.eval.vmlaunch import files

    ev, _ = _paths(args)
    for name, text in files().items():
        write_text_atomic(ev / "vm" / name, text)
        print(f"wrote {ev / 'vm' / name}")
    return 0


def cmd_runsc_check(args: argparse.Namespace) -> int:
    from receipts.eval.images import ImageSet
    from receipts.eval.manifest import load
    from receipts.eval.runsc_check import (
        PreflightError,
        RuntimeMismatchError,
        SwebenchGrader,
        load_rows,
        run_check,
        subprocess_docker,
    )

    ev, data = _paths(args)
    images = ImageSet.model_validate_json((ev / "images.json").read_text())
    ids = Path(args.instances).read_text().split()
    if args.rows:
        rows = load_rows(Path(args.rows))
    else:
        pin = load((ev / "manifest.json").read_text()).derived_file("swe-bench-verified:rows")
        rows = load_rows(data / pin.path, expect_sha256=pin.sha256)
    try:
        report = run_check(
            rows,
            images.subset(ids),
            subprocess_docker,
            SwebenchGrader(),
            Path(args.out),
            run_id=args.run_id,
            timeout_s=args.timeout,
            workers=_workers(args),
            max_drop_pct=args.max_drop_pct,
        )
    except (PreflightError, RuntimeMismatchError) as e:
        return _fail(f"the VM is not ready: {e}")
    except ValueError as e:
        return _fail(f"refused: {e}")
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.verdicts.get("infra"):
        return _fail(f"{report.verdicts['infra']} instances hit an infrastructure error; rerun them")
    return 3 if report.drop_pct_over_limit else 0


def cmd_labels(args: argparse.Namespace) -> int:
    from receipts.eval import labelrun
    from receipts.eval.images import ImageSet
    from receipts.eval.manifest import PatchIndexEntry, load, load_patch_index, patch_text
    from receipts.eval.rows import load_abs
    from receipts.eval.runsc_check import (
        PreflightError,
        ResultLog,
        RuntimeMismatchError,
        SwebenchGrader,
        preflight,
        subprocess_docker,
    )

    ev, data = _paths(args)
    manifest = load((ev / "manifest.json").read_text())
    rows = _rows(manifest, data)
    abs_pin = manifest.derived_file("swe-abs:rows")
    abs_rows = load_abs(data / abs_pin.path, expect_sha256=abs_pin.sha256)
    images = ImageSet.model_validate_json((ev / "images.json").read_text()).by_instance
    only = Path(args.only).read_text().split() if args.only else None
    jobs = labelrun.plan(plan_inputs(manifest, rows), only=only)
    indexes: dict[str, dict[str, PatchIndexEntry]] = {}

    def patch(entry: str, iid: str) -> str:
        if entry not in indexes:
            indexes[entry] = load_patch_index(data, entry)
        return patch_text(data, indexes[entry][iid])

    try:
        work = labelrun.materials(jobs, rows, abs_rows, images, patch)
        preflight(subprocess_docker, sorted({m.image for m in work}, key=lambda i: i.instance_id))
        summary = labelrun.run_labels(
            work,
            subprocess_docker,
            SwebenchGrader(),
            ResultLog(labelrun.results_path(Path(args.out))),
            run_id=args.run_id,
            timeout_s=args.timeout,
            workers=_workers(args),
        )
    except (PreflightError, RuntimeMismatchError) as e:
        return _fail(f"the VM is not ready: {e}")
    except labelrun.LabelResultsError as e:
        return _fail(f"refused: {e}")
    print(json.dumps(summary.model_dump(mode="json"), indent=2, sort_keys=True))
    if summary.infra:
        return _fail(f"{len(summary.infra)} jobs hit an infrastructure error; run `labels` again to retry them")
    return 0


def _no_wiring(command: str) -> int:
    return _fail(
        f"{command} needs the run phase's sandbox and runners, which the integrator wires in "
        "(receipts.eval.cli.register(sub, wiring=Wiring(...)) in src/receipts/cli.py)"
    )


def cmd_r24(args: argparse.Namespace, *, wiring: Wiring | None = None) -> int:
    from receipts.eval import r24
    from receipts.eval.images import ImageSet, RegistryError
    from receipts.eval.manifest import load
    from receipts.eval.runsc_check import PreflightError, RuntimeMismatchError, preflight, subprocess_docker
    from receipts.eval.specs import SPECS_FILE
    from receipts.eval.specs import load as load_specs

    if wiring is None:
        return _no_wiring("r24")
    ev, data = _paths(args)
    manifest = load((ev / "manifest.json").read_text())
    rows = _rows(manifest, data)
    pins = ImageSet.model_validate_json((ev / "images.json").read_text())
    images = {i.instance_id: i.ref for i in pins.images}
    table = load_specs((ev / SPECS_FILE).read_text())
    specs = {(s.repo, s.version): s for s in table.specs}
    ids = Path(args.instances).read_text().split()
    for line in r24.naming_lines(r24.naming_before_run(rows, specs, ids)):  # before the run starts
        print(line, file=sys.stderr)
    diagnose = _harness_diagnosis(ev, data, rows) if args.diagnose else None
    workers = _workers(args)
    try:
        preflight(subprocess_docker, pins.subset(ids))  # the VM's runsc is gVisor and runc is not, before any instance
        report = r24.run_check(
            rows,
            images,
            specs,
            ids,
            wiring.sandbox,
            wiring.runners,
            Path(args.out),
            run_id=args.run_id,
            timeout_s=args.timeout,
            workers=workers,
            max_drop_pct=args.max_drop_pct,
            diagnose=diagnose,
        )
    except (PreflightError, RuntimeMismatchError) as e:
        return _fail(f"the run phase is not on gVisor: {e}")
    except (ValueError, RegistryError) as e:
        return _fail(f"refused: {e}")
    print(json.dumps(report.model_dump(mode="json"), indent=2, sort_keys=True))
    if report.unnamed:
        print(
            f"{len(report.unnamed)} instances have {sum(report.unnamed.values())} SWE-bench ids that name no case the "
            "runner reports (Django docstring names, pytest ids cut at a space); each result lists them"
        )
    if report.patch_fallbacks:
        print(
            f"{len(report.patch_fallbacks)} instances needed more than `git apply --verbose` for gold plus the test "
            f"patch (REG-569): {report.patch_fallbacks}"
        )
    if report.xfail_counted:
        print(
            f"{len(report.xfail_counted)} instances had expected cases that failed as expected, which R24 counted as "
            f"passing, as swebench 4.1.0 grades XFAIL in their repositories: {report.xfail_counted}"
        )
    if report.xfail_skipped:
        print(
            f"{len(report.xfail_skipped)} instances had expected cases that failed as expected where swebench 4.1.0's "
            f"log parser writes no XFAIL and grades them failed, so they stayed skipped: {report.xfail_skipped}"
        )
    if report.unchecked:
        stuck = [i for i in report.unchecked if i in set(report.unnameable)]
        rest = len(report.unchecked) - len(stuck)
        message = f"{len(report.unchecked)} instances are unchecked"
        if rest:
            message += f"; rerun the {rest} a rerun can settle with --instances and a new --out"
        if stuck:
            message += (
                f"; {', '.join(stuck)} {'has' if len(stuck) == 1 else 'have'} no FAIL_TO_PASS id R24 can name, so "
                + r24.UNNAMEABLE_HINT
            )
        if report.probe_failed and len(report.probe_failed) == report.instances:
            message += "; every kernel probe read nothing, so check that the wired sandbox runs runsc"
        return _fail(message)
    return 3 if report.drop_pct_over_limit else 0


def _harness_diagnosis(ev: Path, data: Path, rows: Mapping[str, Row]) -> Callable[[str], str]:
    """R24's drop reason from the harness gold check: the same gold as root, the way SWE-bench runs it."""
    from receipts.eval.images import ImageSet
    from receipts.eval.runsc_check import SwebenchGrader, check_instance, subprocess_docker

    images = ImageSet.model_validate_json((ev / "images.json").read_text()).by_instance
    grader = SwebenchGrader()

    def diagnose(iid: str) -> str:
        r = check_instance(rows[iid].harness_dict(), images[iid], subprocess_docker, grader, timeout_s=1800, run_id="d")
        if r.kept:
            return "harness (root): kept, so the run phase (uid 65534, read-only root) breaks it: back to the founder"
        return f"harness (root): {r.verdict.value}"

    return diagnose


REPLAY_TRIES = 2  # a gold replay the sandbox could not answer is retried once, in a fresh container


def parse_replay_drops(values: Sequence[str]) -> dict[str, str]:
    """The founder's --replay-drop values, `<finding id>=<reason>`, by finding id (a finding id never holds '=')."""
    out: dict[str, str] = {}
    for v in values:
        fid, sep, reason = v.partition("=")
        if not sep or not fid.strip() or not reason.strip():
            raise ValueError(f"--replay-drop {v!r}: want <finding id>=<reason>")
        if fid.strip() in out:
            raise ValueError(f"--replay-drop {fid.strip()}: named twice")
        out[fid.strip()] = reason.strip()
    return out


def _read_replay_cache(path: Path) -> dict[str, score_types.GoldReplay]:
    """The cached gold replays: (outcome, failure type, and the run's output tails and problem) by replay key. A line
    whose replay is infra (ERROR with type INFRA or none, which an older cache may hold) is ignored, so that replay
    runs again; a line written before the output was kept reads with empty tails."""
    from receipts.contracts import FailureType, Outcome
    from receipts.eval.score import GoldReplay, is_infra

    cache: dict[str, GoldReplay] = {}
    if not path.is_file():
        return cache
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        outcome = Outcome(rec["outcome"])
        ftype = None if rec.get("failure_type") is None else FailureType(rec["failure_type"])
        if not is_infra(outcome, ftype):
            said = {k: str(rec.get(k) or "") for k in ("stdout_tail", "stderr_tail", "problem")}
            cache[str(rec["key"])] = GoldReplay(outcome=outcome, failure_type=ftype, **said)
    return cache


def cmd_score(args: argparse.Namespace, *, wiring: Wiring | None = None) -> int:
    from receipts.contracts import Repro, Side, Target
    from receipts.eval import score as score_mod
    from receipts.eval.manifest import load
    from receipts.eval.pilot import PilotCheckError, PilotConfig, check_files, read_run_file
    from receipts.eval.score import GoldReplay, is_infra
    from receipts.eval.specs import changed_files

    try:
        drops = parse_replay_drops(args.replay_drop or ())
    except ValueError as e:
        return _fail(f"score refused: {e}")
    ev, data = _paths(args)
    root = ev.parent
    cfg = PilotConfig.model_validate_json((ev / "pilot.json").read_text())
    run_text = (root / cfg.run_file).read_text()
    arms_text = (root / cfg.arms_file).read_text()
    try:
        check_files(cfg, run_text, arms_text)
    except PilotCheckError as e:
        return _fail(f"score refused: {e}")
    if cfg.status != "frozen" and not args.provisional:
        return _fail("the pilot is provisional (drawn before the freeze); pass --provisional to score it anyway")
    targets = [line.target for line in read_run_file(run_text)]
    rows = _rows(load((ev / "manifest.json").read_text()), data)
    missing = [t.id for t in targets if t.id.split("~", 1)[-1] not in rows]
    if missing:
        return _fail(f"score refused: no dataset row for {missing[:3]}")
    ledger_dir = Path(args.ledgers)
    ledgers = {}
    try:
        for name, path in score_mod.ledger_paths(ledger_dir, cfg.arms).items():
            ledgers[name] = score_mod.read_ledger(path.read_text(), str(path))
    except (OSError, score_mod.ScoreError) as e:
        return _fail(f"score refused: {e}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    cache_path = out / "gold-replays.jsonl"
    cache = _read_replay_cache(cache_path)
    run_phase: list[tuple[Sandbox, Mapping[RunnerKind, SuiteRunner]]] = []

    def replay_once(target: Target, repro: Repro, gold: str) -> GoldReplay:
        sandbox, runners = run_phase[0]
        try:
            ws = sandbox.prepare(target, Side.GOLD, patch=gold)
            try:
                result = sandbox.run_test(
                    ws,
                    score_mod.repro_tests(repro),
                    runner=runners[target.test_runner],
                    changed_files=changed_files(gold),
                    timeout_s=args.timeout,
                    files=repro.files,
                    seed=0,
                    fresh=True,
                    expect=list(repro.cases) or None,  # contract v3: the repro's own cases, when it names them
                )
            finally:
                sandbox.close(ws)
        except Exception as e:  # the sandbox's own failure says nothing about the repro on gold
            return GoldReplay(outcome=None, detail=f"{type(e).__name__}: {e}"[-500:])
        if is_infra(result.outcome, result.failure_type):
            return GoldReplay(outcome=None, detail=(result.problem or result.stderr_tail or result.stdout_tail)[-500:])
        keep = score_mod.GOLD_OUTPUT_CHARS
        return GoldReplay(
            outcome=result.outcome,
            failure_type=result.failure_type,
            stdout_tail=result.stdout_tail[-keep:],
            stderr_tail=result.stderr_tail[-keep:],
            problem=result.problem[-keep:],
        )

    def replay(target: Target, repro: Repro) -> GoldReplay:
        key = score_mod.replay_key(target.id, repro)
        if key in cache:
            return cache[key]
        if wiring is None:
            raise score_mod.ScoreError(f"{target.id}: no cached gold replay, and no sandbox is wired to run one")
        if not run_phase:
            run_phase.append((wiring.sandbox(), wiring.runners()))
        gold = rows[target.id.split("~", 1)[1]].patch
        r = GoldReplay(outcome=None)
        for _ in range(REPLAY_TRIES):
            r = replay_once(target, repro, gold)
            if r.scored:
                break
        if not r.scored:
            return r  # unscored infra: never cached, so the next score replays it
        cache[key] = r
        with cache_path.open("a", encoding="utf-8") as fh:
            rec = {"key": key, "target": target.id, "outcome": r.outcome.value if r.outcome else None,
                   "failure_type": None if r.failure_type is None else r.failure_type.value,
                   "stdout_tail": r.stdout_tail, "stderr_tail": r.stderr_tail, "problem": r.problem}  # fmt: skip
            fh.write(json.dumps(rec) + "\n")
        return r

    try:
        result = score_mod.score(cfg, targets, ledgers, replay, drops=drops)
    except score_mod.ScoreError as e:
        return _fail(f"score refused: {e}")
    write_text_atomic(out / "score.json", pretty_json(result.model_dump(mode="json")))
    for a in result.arms:
        print(
            f"{a.arm} ({a.attacker_model}): {a.blockers} blockers, {a.catches} targets caught {a.catches_by_label} "
            f"({a.catch_findings} findings), {a.invalid_repros} invalid repros ({a.invalid_repro_pct}%), "
            f"{len(a.unscored)} unscored, {len(a.dropped)} dropped by the founder, cost ${a.cost_micro_usd / 1e6:.2f} "
            f"(p50 ${a.cost_per_target_micro_usd.p50 / 1e6:.2f} per target), {a.compute_hours} compute hours"
        )
        fv = a.fix_validity
        print(
            f"{a.arm} fix-validity on gold (REG-577, outside REG-561's rule): {fv.candidates} candidates, "
            f"{fv.catches} targets caught {fv.catches_by_label} ({fv.catch_findings} findings), {fv.invalid_repros} "
            f"invalid repros, {len(fv.unscored)} unscored, {len(fv.dropped)} dropped by the founder"
        )
    for fid, why in result.replay_drops.items():
        print(f"replay drop (the founder's): {fid}: {why}")
    print(f"REG-561: {'met' if result.rule.met else 'not met'}: {result.rule.why}")
    print(f"wrote {out / 'score.json'}")
    waiting = [
        f"{n} {what} replays"
        for n, what in ((result.unscored, "blocker"), (result.fix_validity_unscored, "fix-validity"))
        if n
    ]
    if waiting:
        return _fail(
            f"{' and '.join(waiting)} on gold hit infra twice and are unscored; run score again with the same --out "
            "(settled replays are cached), or, if one keeps failing, the founder may drop it by name with "
            "--replay-drop <finding id>=<reason>"
        )
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    """EV3: report.json and report.md from a score.json of format /3. Pure arithmetic: no sandbox, no model."""
    from receipts.eval import report as report_mod

    try:
        result = report_mod.report(report_mod.read_score(Path(args.score).read_text()), seed=args.seed)
    except (OSError, report_mod.ReportError) as e:
        return _fail(f"report refused: {e}")
    out = Path(args.out)
    write_text_atomic(out / "report.json", pretty_json(result.model_dump(mode="json")))
    write_text_atomic(out / "report.md", report_mod.render_markdown(result))
    for a in result.arms:
        r = a.rates
        print(
            f"{a.arm}: WDR {r.wdr.k}/{r.wdr.n}, false alarms {r.false_alarm.k}/{r.false_alarm.n}, precision "
            f"{r.precision.k}/{r.precision.n}, MCC {r.mcc if r.mcc is None else round(r.mcc, 3)}"
        )
    print(
        f"wrote {out / 'report.json'} and {out / 'report.md'}{'' if result.final else ' (not final: replays pending)'}"
    )
    return 0


def _usage(parser: argparse.ArgumentParser) -> Callable[[argparse.Namespace], int]:
    def handler(_: argparse.Namespace) -> int:
        parser.print_help(sys.stderr)
        return 2

    return handler


WORKERS_HELP = "sandboxes at once (default: this machine's vCPUs over 2, within its memory at 4g each)"


def cmd_demo_target(args: argparse.Namespace) -> int:
    from receipts.eval.demo import demo_target, write_run_file

    try:
        write_run_file(Path(args.out), demo_target(Path(args.pins), args.image))
    except (OSError, ValueError) as e:
        return _fail(f"demo target refused: {e}")
    print(f"wrote {args.out}")
    return 0


def cmd_demo_fixcheck(args: argparse.Namespace, wiring: Wiring | None = None) -> int:
    from receipts.contracts import Ledger
    from receipts.eval.demo import fix_check
    from receipts.eval.pilot import read_run_file
    from receipts.surface.ledger import JsonlLedger, LedgerError

    if wiring is None:
        return _fail("demo fixcheck needs the run phase's sandbox and runners")
    try:
        targets = read_run_file(Path(args.target).read_text())
        if len(targets) != 1:
            raise ValueError("--target must contain exactly one RunItem")
        target = targets[0].target
        ledger: Ledger = JsonlLedger(Path(args.ledger), read_only=True)
        history = [entry.finding for entry in ledger.entries(target.id) if entry.finding.id == args.finding]
        if not history:
            raise ValueError("the finding is absent from the target's ledger")
        result = fix_check(
            target,
            history[-1],
            Path(args.fix).read_bytes().decode("utf-8"),
            sandbox=wiring.sandbox(),
            runner=wiring.runners()[target.test_runner],
        )
        write_text_atomic(Path(args.out), result.model_dump_json(indent=2) + "\n")
    except (OSError, ValueError, LedgerError, RuntimeError) as e:
        return _fail(f"demo fixcheck refused: {e}")
    print(f"wrote {args.out}: passes_on_fix={result.passes_on_fix}")
    return 0 if result.passes_on_fix else 1


def build_parser(
    parser: argparse.ArgumentParser | None = None, wiring: Wiring | None = None
) -> argparse.ArgumentParser:
    p = parser or argparse.ArgumentParser(prog="receipts eval", description="Lane eval: Tier A data, sample and VM.")
    p.add_argument("--root", help="repository root (default: found from the working directory)")
    p.set_defaults(func=_usage(p))
    sub = p.add_subparsers(dest="eval_command", required=True)
    demo = sub.add_parser("demo", help="the pinned Tier B demo target and its separate later-fix check")
    demo.set_defaults(func=_usage(demo))
    demo_sub = demo.add_subparsers(dest="demo_command", required=True)
    dt = demo_sub.add_parser("target", help="validate the demo pins and write one spec-temperature RunItem")
    dt.add_argument("--pins", required=True, help="directory containing pins.json, template and both diffs")
    dt.add_argument("--image", required=True, help="digest-pinned image reference read on the eval VM")
    dt.add_argument("--out", required=True, help="output run JSONL")
    dt.set_defaults(func=cmd_demo_target)
    df = demo_sub.add_parser("fixcheck", help="replay a blocker's repro five times on base plus PR plus fix")
    df.add_argument("--target", required=True, help="one RunItem JSONL from demo target")
    df.add_argument("--ledger", required=True, help="existing arm ledger, opened read-only")
    df.add_argument("--finding", required=True, help="blocker finding id")
    df.add_argument("--fix", required=True, help="combined base plus PR plus fix diff")
    df.add_argument("--out", required=True, help="output FixCheck JSON")
    df.set_defaults(func=functools.partial(cmd_demo_fixcheck, wiring=wiring))
    sub.add_parser("fetch", help="download the pinned sources into eval/data/ and check their sha256").set_defaults(
        func=cmd_fetch
    )
    sub.add_parser("rows", help="convert the two pinned datasets to rows JSONL (needs --extra eval)").set_defaults(
        func=cmd_rows
    )
    sub.add_parser("pool", help="derive the SWE-ABS pools into eval/pools/").set_defaults(func=cmd_pool)
    sub.add_parser("test-specs", help="SWE-bench's test commands for the pool, from the pinned wheel").set_defaults(
        func=cmd_test_specs
    )
    m = sub.add_parser("manifest", help="build eval/manifest.json from eval/data/")
    m.add_argument(
        "--pool", choices=["swe_abs_251", "swe_abs_206"], default="swe_abs_251", help="the founder's choice: 251"
    )
    m.add_argument("--freeze", action="store_true", help="mark the manifest frozen; refuses while anything is missing")
    m.add_argument("--gold-runsc", help="R24's output directory (receipts eval r24)")
    m.add_argument("--gold-rerun", action="append", help="a rerun's output directory; replaces INFRA verdicts only")
    m.add_argument("--gold-drop", help="the founder's drops: <instance_id><TAB><reason> per line, unchecked only")
    m.add_argument("--labels", help="the label runner's output directory (receipts eval labels)")
    m.add_argument(
        "--label-drop",
        action="append",
        metavar="INSTANCE=REASON",
        help=(
            "the founder's: stop waiting on an instance R24 kept whose unfinished label jobs were INFRA on every run; "
            "the records that waited on them become DROPPED, with the reason in the manifest (needs --labels and "
            "--gold-runsc; repeatable)"
        ),
    )
    m.set_defaults(func=cmd_manifest)
    i = sub.add_parser("images", help="pin the Epoch image digests for the pool into eval/images.json")
    i.add_argument("--workers", type=int, default=8)
    i.set_defaults(func=cmd_images)
    pt = sub.add_parser("patches", help="pull every entry's patch.diff and report.json for the pool into eval/data/")
    pt.add_argument("--workers", type=int, default=8)
    pt.set_defaults(func=cmd_patches)
    sm = sub.add_parser("sample", help="draw the seeded, capped Tier A sample and its 40/60 split")
    sm.add_argument("--plan", choices=["spec", "core"], default="spec", help="spec: 150+150+75; core: 100+100+50")
    sm.add_argument("--seed", help="drafts only: a seed other than the manifest's content sha256 (refused when frozen)")
    sm.add_argument("--out", help="output path (default eval/sample.json, or eval/sample-provisional.json for a draft)")
    sm.set_defaults(func=cmd_sample)
    pl = sub.add_parser("pilot", help="write eval/pilot.json, the run file and the arms file from the verified sample")
    pl.set_defaults(func=cmd_pilot)
    tg = sub.add_parser("targets", help="write every verified sample item as a contracts.Target into eval/data/")
    tg.set_defaults(func=cmd_targets)
    q = sub.add_parser("quote", help="write the VM plan and cost quote to eval/vm/quote.json")
    q.add_argument("--instance", default="m7i.8xlarge", help="the pilot's and the rehearsal's instance type")
    q.set_defaults(func=cmd_quote)
    ci = sub.add_parser("cloud-init", help="render the VM user data from eval/images.json into eval/vm/")
    ci.add_argument("--instances", help="file of instance ids to pull (default: every pinned image)")
    ci.set_defaults(func=cmd_cloud_init)
    vf = sub.add_parser("vm-files", help="write the eval and GPU VM launch inputs to eval/vm/")
    vf.set_defaults(func=cmd_vm_files)
    rc = sub.add_parser("runsc-check", help="on the eval VM: the harness gold check, as root (explains R24 drops)")
    rc.add_argument("--instances", required=True, help="file of instance ids, e.g. eval/pools/swe_abs_251.txt")
    rc.add_argument("--rows", help="SWE-bench Verified rows JSONL (default: the one the manifest pinned)")
    rc.add_argument("--out", required=True, help="directory for results.jsonl, kept.txt, dropped.tsv, report.json")
    rc.add_argument("--run-id", required=True)
    rc.add_argument("--timeout", type=float, default=1800, help="seconds per eval script (the harness default)")
    rc.add_argument("--workers", type=int, help=WORKERS_HELP)
    rc.add_argument("--max-drop-pct", type=int, default=25, help="exit 3 when more than this share is dropped")
    rc.set_defaults(func=cmd_runsc_check)
    lb = sub.add_parser("labels", help="on the eval VM: our SWE-ABS and FAIL_TO_PASS runs (REG-564); resumable")
    lb.add_argument("--out", required=True, help="directory for results.jsonl (append-only; a rerun resumes it)")
    lb.add_argument("--run-id", required=True)
    lb.add_argument("--only", help="file of instance ids: run only their jobs")
    lb.add_argument("--timeout", type=float, default=1800, help="seconds per eval script")
    lb.add_argument("--workers", type=int, help=WORKERS_HELP)
    lb.set_defaults(func=cmd_labels)
    r = sub.add_parser("r24", help="on the eval VM: gold in the run phase under runsc (R24); needs the wiring")
    r.add_argument("--instances", required=True, help="file of instance ids, e.g. eval/pools/swe_abs_251.txt")
    r.add_argument("--out", required=True, help="a new directory for results.jsonl, dropped.tsv and report.json")
    r.add_argument("--run-id", required=True)
    r.add_argument(
        "--timeout",
        type=float,
        default=1800,
        help=(
            "seconds for one run_test of the test patch's files; a run of the named cases' modules (a test patch "
            "with no test file) gets 2 s per expected case, never less than this and at most 7200 s unless this is "
            "more"
        ),
    )
    r.add_argument("--workers", type=int, help=WORKERS_HELP)
    r.add_argument("--max-drop-pct", type=int, default=25, help="exit 3 when more than this share is dropped")
    r.add_argument("--diagnose", action="store_true", help="run the harness gold check on every drop, as its reason")
    r.set_defaults(func=functools.partial(cmd_r24, wiring=wiring))
    sc = sub.add_parser("score", help="score the pilot's per-arm ledgers against gold and apply REG-561's rule")
    sc.add_argument("--ledgers", required=True, help="directory with one <arm>.jsonl ledger per arm")
    sc.add_argument("--out", required=True, help="directory for score.json and the gold-replay cache")
    sc.add_argument("--timeout", type=float, default=600, help="seconds for one gold replay")
    sc.add_argument("--provisional", action="store_true", help="score a pilot drawn before the freeze (a dry run)")
    sc.add_argument(
        "--replay-drop",
        action="append",
        metavar="FINDING=REASON",
        help=(
            "the founder's: drop a blocker's or a fix-validity candidate's gold replay that is still unscored infra "
            "after its retry; that finding counts on neither side, with the reason in score.json (repeatable)"
        ),
    )
    sc.set_defaults(func=functools.partial(cmd_score, wiring=wiring))
    rp = sub.add_parser("report", help="rates with intervals, comparisons and costs from score.json (EV3)")
    rp.add_argument("--score", required=True, help="score.json from `receipts eval score` (format /3)")
    rp.add_argument("--seed", type=int, default=0, help="the bootstrap's seed")
    rp.add_argument("--out", required=True, help="directory for report.json and report.md")
    rp.set_defaults(func=cmd_report)
    return p


def register(subparsers: argparse._SubParsersAction[argparse.ArgumentParser], wiring: Wiring | None = None) -> None:
    """Hook for receipts.cli: `receipts eval ...`. Every parser sets func to a handler that returns the exit code.
    `wiring` gives `r24` and `score` the run phase's sandbox and runners; without it they exit 2 and say so."""
    build_parser(subparsers.add_parser("eval", help="lane eval: Tier A data, sample and VM"), wiring)


def main(argv: Sequence[str] | None = None, wiring: Wiring | None = None) -> int:
    args = build_parser(wiring=wiring).parse_args(argv)
    code: int = args.func(args)
    return code


if __name__ == "__main__":
    sys.exit(main())
