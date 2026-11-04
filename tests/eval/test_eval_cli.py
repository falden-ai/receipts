"""The lane's CLI: every command dispatches through `func` and returns its exit code, so receipts.cli can wire it in;
a frozen manifest takes no seed; pilot and targets refuse a sample that is not the seeded draw; r24 and score run on
the wired run phase (faked here) and refuse without it, r24 after the VM preflight; a gold replay the sandbox could
not answer is retried, never cached and never scored; runsc-check's and labels' failures map to exit codes, not
tracebacks; worker counts come from the machine."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

import receipts.cli as top
from receipts.contracts import (
    CatchKind,
    FailureType,
    Finding,
    LedgerEntry,
    Lens,
    Outcome,
    ReplayKey,
    Repro,
    RunnerKind,
    Side,
    StageCost,
    Status,
)
from receipts.eval import cli
from receipts.eval import runsc_check as rc
from receipts.eval.hashing import pretty_json, sha256_bytes, sha256_file, sha256_text
from receipts.eval.images import ImageRef, ImageSet
from receipts.eval.labels import Label
from receipts.eval.manifest import DerivedRecord, PatchIndex, PatchIndexEntry, dump, patch_index_path
from receipts.eval.pilot import PilotConfig, read_run_file, verify_pilot
from receipts.eval.r24 import R24Result, R24Verdict
from receipts.eval.rows import ABS_ROWS_PATH, ROWS_PATH, AbsRow, Row
from receipts.eval.runsc_check import PreflightError, Report
from receipts.eval.sampler import Sample, rank
from receipts.eval.score import GOLD_OUTPUT_CHARS
from receipts.eval.specs import SPECS_FILE, RepoSpec, SpecTable, runner_kind
from receipts.eval.specs import dump as dump_specs

from .conftest import FakeRunPhase, fake_runners, make_manifest, run_result

INFRA = run_result(Outcome.ERROR, failure_type=FailureType.INFRA, stderr_tail="docker: container is not running")


@pytest.fixture
def preflights(monkeypatch: pytest.MonkeyPatch) -> list[list[str]]:
    """The VM preflight, faked: it records the images it was asked to prove (no Docker on this machine)."""
    seen: list[list[str]] = []

    def fake(docker: object, images: list[ImageRef]) -> None:
        seen.append([i.instance_id for i in images])

    monkeypatch.setattr(rc, "preflight", fake)
    return seen


def leaves(parser: argparse.ArgumentParser) -> list[tuple[str, argparse.ArgumentParser]]:
    out = []
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for name, sub in action.choices.items():
                out.append((name, sub))
                out.extend(leaves(sub))
    return out


def test_every_eval_parser_sets_a_handler_and_the_top_level_cli_dispatches_to_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = argparse.ArgumentParser()
    sub = root.add_subparsers(dest="command", required=True)
    cli.register(sub)
    found = leaves(root)
    assert {n for n, _ in found} >= {
        "eval", "manifest", "sample", "pilot", "targets", "vm-files", "runsc-check", "patches", "labels", "r24",
        "score", "test-specs",
    }  # fmt: skip
    for name, parser in found:
        assert callable(parser.get_default("func")), name
    monkeypatch.setattr(top, "LANE_COMMANDS", (cli.register,))
    assert top.main(["eval", "--root", str(tmp_path), "quote"]) == 0
    assert (tmp_path / "eval" / "vm" / "quote.json").is_file()
    assert top.main(["eval", "--root", str(tmp_path), "vm-files"]) == 0
    assert (tmp_path / "eval" / "vm" / "run-instances.json").is_file()
    assert (tmp_path / "eval" / "vm" / "gpu-cloud-init.yaml").is_file()


def write_root(tmp_path: Path, manifest: Any, pool_text: str) -> Path:
    (tmp_path / "eval" / "pools").mkdir(parents=True, exist_ok=True)
    (tmp_path / "eval" / "manifest.json").write_text(dump(manifest))
    (tmp_path / "eval" / "pools" / "swe_abs_251.txt").write_text(pool_text)
    return tmp_path


def test_plant_a_seed_for_a_frozen_manifest_is_refused(tmp_path: Path, synthetic_manifest: Any) -> None:
    root = write_root(tmp_path, *synthetic_manifest(frozen=True))
    assert cli.main(["--root", str(root), "sample", "--plan", "core", "--seed", "lucky-7"]) == 2
    assert not (root / "eval" / "sample.json").exists()
    assert cli.main(["--root", str(root), "sample", "--plan", "core"]) == 0
    sample = Sample.model_validate_json((root / "eval" / "sample.json").read_text())
    assert sample.status == "frozen" and sample.seed == sample.manifest_content_sha256


def test_an_infeasible_plan_exits_3_with_the_supply_it_found(
    tmp_path: Path, synthetic_manifest: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    ids = [f"django__django-{n}" for n in range(30)]
    root = write_root(tmp_path, *synthetic_manifest(frozen=False, ids=ids))
    assert cli.main(["--root", str(root), "sample", "--plan", "spec"]) == 3
    assert "candidates for 60 items" in capsys.readouterr().err


def test_plant_r24_reruns_or_drops_without_a_base_are_refused(tmp_path: Path) -> None:
    (tmp_path / "eval" / "pools").mkdir(parents=True)
    (tmp_path / "eval" / "pools" / "swe_abs_251.txt").write_text("django__django-1\n")
    assert cli.main(["--root", str(tmp_path), "manifest", "--gold-rerun", str(tmp_path)]) == 2
    assert cli.main(["--root", str(tmp_path), "manifest", "--gold-drop", str(tmp_path / "x")]) == 2


def test_plant_an_r24_result_for_another_pool_is_refused_by_the_manifest_command(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    (tmp_path / "eval" / "pools").mkdir(parents=True)
    (tmp_path / "eval" / "pools" / "swe_abs_251.txt").write_text("django__django-1\n")
    ref = ImageRef(
        instance_id="django__django-1",
        image="ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1",
        digest="sha256:" + "e" * 64,
        media_type="application/vnd.docker.distribution.manifest.v2+json",
    )
    images = ImageSet(resolved_at="2026-10-27T00:00:00Z", images=(ref,), unique_compressed_bytes=0)
    (tmp_path / "eval" / "images.json").write_text(pretty_json(images.model_dump(mode="json")))
    out = tmp_path / "runs" / "r24"
    out.mkdir(parents=True)
    stray = R24Result(
        run_id="d1",
        instance_id="sympy__sympy-9",
        target_id="gold~sympy__sympy-9",
        image="x",
        runner=RunnerKind.SYMPY,
        tests=("t.py",),
        tests_from="test_patch",
        expected=1,
        verdict=R24Verdict.KEPT,
        attempts=({"gvisor": True, "outcome": "pass"},),  # type: ignore[arg-type]
    )
    (out / "results.jsonl").write_text(stray.model_dump_json() + "\n")
    assert cli.main(["--root", str(tmp_path), "manifest", "--gold-runsc", str(out)]) == 2
    assert "not on the pool" in capsys.readouterr().err
    kept = stray.model_copy(update={"instance_id": "django__django-1", "target_id": "gold~django__django-1"})
    (out / "results.jsonl").write_text(kept.model_dump_json() + "\n")  # its own instance, but another image
    assert cli.main(["--root", str(tmp_path), "manifest", "--gold-runsc", str(out)]) == 2
    assert "not its pin in eval/images.json" in capsys.readouterr().err


# ---------------------------------------------------------------- a full synthetic root: rows, patches, images, specs


def full_root(
    tmp_path: Path, *, frozen: bool = True, f2p: dict[str, tuple[str, ...]] | None = None
) -> tuple[Path, Any]:
    manifest, pool_text = make_manifest(frozen=frozen)
    data = tmp_path / "eval" / "data"
    ids = sorted({r.instance_id for r in manifest.records})
    rows = {}
    ids_for = {
        RunnerKind.DJANGO: (("test_fix (app.tests.T)",), ("test_old (app.tests.T)",)),
        RunnerKind.SYMPY: (("test_fix",), ("test_old",)),
        RunnerKind.PYTEST: (("tests/test_it.py::test_fix",), ("tests/test_it.py::test_old",)),
    }
    for iid in ids:
        repo_name = manifest.records[[r.instance_id for r in manifest.records].index(iid)].repo
        fails, p2p = ids_for[runner_kind(repo_name)]
        rows[iid] = Row(
            FAIL_TO_PASS=(f2p or {}).get(iid, fails),
            PASS_TO_PASS=p2p,
            base_commit="a" * 40,
            created_at="2020-01-01T00:00:00Z",
            difficulty="<15 min fix",
            environment_setup_commit="b" * 40,
            hints_text="",
            instance_id=iid,
            patch=f"diff --git a/src/{iid}.py b/src/{iid}.py\n+gold\n",
            problem_statement="It breaks.",
            repo=manifest.records[[r.instance_id for r in manifest.records].index(iid)].repo,
            test_patch="diff --git a/tests/test_it.py b/tests/test_it.py\n+t\n",
            version="3.0",
        )
    (data / ROWS_PATH).parent.mkdir(parents=True)
    (data / ROWS_PATH).write_text("".join(r.model_dump_json() + "\n" for r in rows.values()))
    (data / ABS_ROWS_PATH).parent.mkdir(parents=True)
    abs_rows = [
        AbsRow(
            FAIL_TO_PASS=r.FAIL_TO_PASS,
            PASS_TO_PASS=r.PASS_TO_PASS,
            base_commit=r.base_commit,
            instance_id=r.instance_id,
            original_test_patch=r.test_patch,
            patch=r.patch,
            repo=r.repo,
            test_patch="diff --git a/t b/t\n+abs\n",
            version=r.version,
        )
        for r in rows.values()
    ]
    (data / ABS_ROWS_PATH).write_text("".join(a.model_dump_json() + "\n" for a in abs_rows))
    records = []
    by_entry: dict[str, list[PatchIndexEntry]] = {}
    for r in manifest.records:
        body = f"diff --git a/src/{r.instance_id}.py b/src/{r.instance_id}.py\n+{r.entry}\n".encode()
        rel = f"patches/{r.entry}/{r.instance_id}/patch.diff"
        (data / rel).parent.mkdir(parents=True, exist_ok=True)
        (data / rel).write_bytes(body)
        by_entry.setdefault(r.entry, []).append(
            PatchIndexEntry(instance_id=r.instance_id, sha256=sha256_bytes(body), size=len(body), path=rel)
        )
        records.append(r.model_copy(update={"patch_sha256": sha256_bytes(body), "patch_bytes": len(body)}))
    for entry, patches in by_entry.items():
        index = PatchIndex(entry=entry, store="s3-logs", patches=tuple(patches))
        patch_index_path(data, entry).write_text(pretty_json(index.model_dump(mode="json")))
    derived = tuple(
        DerivedRecord(
            name=name, derived_from="x", path=path, sha256=sha256_file(data / path), size=(data / path).stat().st_size,
            how="synthetic",
        )
        for name, path in (("swe-bench-verified:rows", ROWS_PATH), ("swe-abs:rows", ABS_ROWS_PATH))
    )  # fmt: skip
    manifest = manifest.model_copy(update={"records": tuple(records), "derived": derived})
    write_root(tmp_path, manifest, pool_text)
    images = ImageSet(
        resolved_at="2026-10-27T00:00:00Z",
        images=tuple(
            ImageRef(
                instance_id=i,
                image=f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{i}",
                digest="sha256:" + sha256_text(i),
                media_type="application/vnd.docker.distribution.manifest.v2+json",
            )
            for i in ids
        ),
        unique_compressed_bytes=0,
    )
    (tmp_path / "eval" / "images.json").write_text(pretty_json(images.model_dump(mode="json")))
    repos = sorted({r.repo for r in rows.values()})
    specs = SpecTable(
        swebench_version="4.1.0",
        wheel_sha256="1" * 64,
        constants_sha256="2" * 64,
        specs=tuple(RepoSpec(repo=x, version="3.0", test_cmd=f"test {x}", runner=runner_kind(x)) for x in repos),
    )
    (tmp_path / "eval" / SPECS_FILE).write_text(dump_specs(specs))
    assert cli.main(["--root", str(tmp_path), "sample", "--plan", "core"]) == 0
    return tmp_path, manifest


def test_pilot_and_targets_write_the_run_file_and_arms_file_from_a_verified_sample(tmp_path: Path) -> None:
    root, _ = full_root(tmp_path)
    assert cli.main(["--root", str(root), "pilot"]) == 0
    cfg = PilotConfig.model_validate_json((root / "eval" / "pilot.json").read_text())
    run_text = (root / cfg.run_file).read_text()
    arms_text = (root / cfg.arms_file).read_text()
    lines = read_run_file(run_text)
    assert len(lines) == 20 and all(line.target.test_command.startswith("test ") for line in lines)
    assert {line.target.test_runner for line in lines} >= {RunnerKind.DJANGO, RunnerKind.SYMPY}
    from receipts.eval.manifest import load

    manifest = load((root / "eval" / "manifest.json").read_text())
    sample_text = (root / "eval" / "sample.json").read_text()
    verify_pilot(
        cfg,
        Sample.model_validate_json(sample_text),
        sample_text,
        manifest,
        [x.target for x in lines],
        run_text,
        arms_text,
    )
    assert cli.main(["--root", str(root), "targets"]) == 0
    assert len((root / "eval" / "data" / "targets" / "sample.jsonl").read_text().splitlines()) == 250


def test_plant_a_hand_edited_sample_is_refused_by_pilot_and_targets(tmp_path: Path) -> None:
    """The re-review's note: pilot and targets built from sample.json without verify_sample, so nothing gated a
    frozen sample. A swapped item (rank recomputed) passes check_sample and must fail verify_sample here."""
    root, manifest = full_root(tmp_path)
    path = root / "eval" / "sample.json"
    sample = Sample.model_validate_json(path.read_text())
    chosen = {(i.entry, i.instance_id) for i in sample.items}
    victim = sample.items[0]
    spare = next(
        r
        for r in manifest.records
        if (r.entry, r.instance_id) not in chosen
        and r.final is victim.label
        and r.repo == victim.repo
        and r.entry == victim.entry
        and sample.split.get(r.instance_id) is victim.side
    )
    forged_item = victim.model_copy(
        update={"instance_id": spare.instance_id, "rank": rank(sample.seed, spare.entry, spare.instance_id)}
    )
    forged = sample.model_copy(update={"items": (forged_item, *sample.items[1:])})
    path.write_text(pretty_json(forged.model_dump(mode="json")))
    assert cli.main(["--root", str(root), "pilot"]) == 2
    assert cli.main(["--root", str(root), "targets"]) == 2
    assert not (root / "eval" / "pilot.json").exists()
    assert not (root / "eval" / "data" / "targets" / "sample.jsonl").exists()


# ---------------------------------------------------------------- r24 and score on the wired run phase


def wiring(phase: FakeRunPhase) -> cli.Wiring:
    return cli.Wiring(sandbox=lambda: phase, runners=fake_runners)  # type: ignore[arg-type, return-value]


def test_r24_without_the_wiring_exits_2_and_says_who_wires_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = full_root(tmp_path)
    ids = root / "ids.txt"
    ids.write_text("django__django-2000\n")
    args = ["--root", str(root), "r24", "--instances", str(ids), "--out", str(root / "runs" / "r24"), "--run-id", "d1"]
    assert cli.main(args) == 2 and "integrator wires" in capsys.readouterr().err


def test_r24_runs_the_real_path_through_the_wired_sandbox(tmp_path: Path, preflights: list[list[str]]) -> None:
    root, _ = full_root(tmp_path)
    ids = root / "ids.txt"
    ids.write_text("django__django-2000\nsympy__sympy-2001\npsf__requests-2002\n")
    phase = FakeRunPhase({"gold~psf__requests-2002": Outcome.FAIL})
    out = root / "runs" / "r24"
    args = ["--root", str(root), "r24", "--instances", str(ids), "--out", str(out), "--run-id", "d1", "--workers", "1"]
    assert cli.main(args, wiring=wiring(phase)) == 3  # 1 of 3 dropped is over the 25% that flags a systemic break
    assert preflights == [["django__django-2000", "psf__requests-2002", "sympy__sympy-2001"]]
    assert len((out / "results.jsonl").read_text().splitlines()) == 3
    assert {side for _, side, _ in phase.prepared} == {Side.GOLD}
    assert json.loads((out / "report.json").read_text())["dropped"] == {"psf__requests-2002": "gold_fails"}
    infra = FakeRunPhase({"gold~django__django-2000": Outcome.ERROR})
    out2 = root / "runs" / "r24-b"
    args2 = [
        "--root",
        str(root),
        "r24",
        "--instances",
        str(ids),
        "--out",
        str(out2),
        "--run-id",
        "d2",
        "--workers",
        "1",
    ]
    assert cli.main(args2, wiring=wiring(infra)) == 2  # unchecked instances: rerun them


def test_plant_r24_on_a_vm_that_fails_its_preflight_runs_no_instance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Under runsc a probe that exits 1 is only that instance's INFRA, so the VM itself is proven first: runsc is
    gVisor and runc is not, before any instance runs."""
    root, _ = full_root(tmp_path)
    ids = root / "ids.txt"
    ids.write_text("django__django-2000\n")

    def not_ready(docker: object, images: object) -> None:
        raise rc.RuntimeMismatchError("the runsc probe's kernel log lacks 'Starting gVisor...'")

    monkeypatch.setattr(rc, "preflight", not_ready)
    phase = FakeRunPhase()
    out = root / "runs" / "r24"
    args = ["--root", str(root), "r24", "--instances", str(ids), "--out", str(out), "--run-id", "d1"]
    assert cli.main(args, wiring=wiring(phase)) == 2 and "not on gVisor" in capsys.readouterr().err
    assert phase.prepared == [] and not (out / "results.jsonl").exists()


def test_plant_r24_for_an_instance_with_no_pinned_image_is_refused_not_a_traceback(
    tmp_path: Path, preflights: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    root, _ = full_root(tmp_path)
    ids = root / "ids.txt"
    ids.write_text("django__django-2000\ndjango__django-99999\n")
    phase = FakeRunPhase()
    args = ["--root", str(root), "r24", "--instances", str(ids), "--out", str(root / "runs" / "r24"), "--run-id", "d1"]
    assert cli.main(args, wiring=wiring(phase)) == 2 and "no pinned image" in capsys.readouterr().err
    assert phase.prepared == [] and preflights == []


def test_r24_takes_its_worker_count_from_the_machine(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflights: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    """REG-568: worker counts come from the VM's vCPU count, never a constant."""
    from receipts.eval import r24, vmplan

    seen: list[int] = []
    real = r24.run_check

    def spy(*a: Any, **kw: Any) -> Any:
        seen.append(kw["workers"])
        return real(*a, **kw)

    monkeypatch.setattr(r24, "run_check", spy)
    monkeypatch.setattr(vmplan, "machine_vcpus", lambda: 64)
    monkeypatch.setattr(vmplan, "machine_memory_bytes", lambda: 256 * 1024**3)
    root, _ = full_root(tmp_path)
    ids = root / "ids.txt"
    ids.write_text("django__django-2000\n")
    args = ["--root", str(root), "r24", "--instances", str(ids), "--out", str(root / "a"), "--run-id", "d1"]
    assert cli.main(args, wiring=wiring(FakeRunPhase())) == 0
    assert "workers: 32 (64 vCPUs here, 2 per sandbox)" in capsys.readouterr().err
    again = [*args[:6], str(root / "b"), *args[7:], "--workers", "3"]
    assert cli.main(again, wiring=wiring(FakeRunPhase())) == 0
    assert seen == [32, 3]


def blocker(target_id: str, *, cost: int = 100_000, attack: str = "a1", cases: tuple[str, ...] = ()) -> LedgerEntry:
    repro = Repro(
        id="r1",
        attack_id="a1",
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        kind=CatchKind.FIX_VALIDITY,
        files={"tests/test_repro.py": "def test_x():\n    assert f(2) == 4\n"},
        command=("python", "-m", "pytest", "tests/test_repro.py"),
        cases=cases,
    )
    runs = (*(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5)), run_result(Outcome.PASS, Side.SURROGATE))
    finding = Finding(
        id=f"{target_id}/fixed_x/{attack}",
        target_id=target_id,
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        status=Status.BLOCKER,
        kind=CatchKind.FIX_VALIDITY,
        repro=repro,
        runs=runs,
        quote="It breaks.",
        replay=ReplayKey(base_sha="a" * 40, patch_sha="b" * 64, harness_hash="h", seed=0, image_digest="sha256:x"),
        costs=(StageCost(stage="attack", model="muse-spark-1.3", micro_usd=cost, seconds=30.0),),
        wall_seconds=100.0,
    )
    return LedgerEntry(at="2026-10-28T12:00:00Z", receipts_version="0.0.1", finding=finding)


def scored_root(tmp_path: Path) -> tuple[Path, PilotConfig, Path]:
    root, _ = full_root(tmp_path)
    assert cli.main(["--root", str(root), "pilot"]) == 0
    cfg = PilotConfig.model_validate_json((root / "eval" / "pilot.json").read_text())
    ledgers = root / "runs" / "ledgers"
    ledgers.mkdir(parents=True)
    ids = [t.target_id for t in cfg.targets]
    spark = [blocker(t) for t in ids[:4]] + [blocker(ids[1], attack="a2")]  # two blockers on one target
    (ledgers / "spark.jsonl").write_text("".join(b.model_dump_json() + "\n" for b in spark))
    (ledgers / "glimmer.jsonl").write_text("".join(blocker(t, cost=0).model_dump_json() + "\n" for t in ids[:3]))
    return root, cfg, ledgers


def test_score_judges_every_blocker_on_gold_and_applies_the_rule(tmp_path: Path) -> None:
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    phase = FakeRunPhase({ids[0]: Outcome.FAIL})  # spark's first and glimmer's first blocker fail on gold
    out = root / "runs" / "score"
    assert (
        cli.main(["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)], wiring=wiring(phase))
        == 0
    )
    score = json.loads((out / "score.json").read_text())
    spark, glimmer = score["arms"]
    # REG-561 counts caught targets; the per-finding count sits beside it, and the rate stays per finding
    assert (spark["blockers"], spark["catches"], spark["catch_findings"], spark["invalid_repros"]) == (5, 3, 4, 1)
    assert spark["invalid_repro_pct"] == 20.0
    assert (glimmer["blockers"], glimmer["catches"], glimmer["invalid_repros"]) == (3, 2, 1)
    assert glimmer["invalid_repro_pct"] == 33.3 and score["rule"]["met"] is False  # 33.3% is not under 30%
    assert (score["rule"]["baseline_catches"], score["rule"]["baseline_catch_findings"]) == (3, 4)
    assert spark["cost_micro_usd"] == 500_000 and glimmer["cost_micro_usd"] == 0
    assert len(phase.prepared) == 4  # spark and glimmer share three repros: four replays, not seven
    assert all(side is Side.GOLD for _, side, _ in phase.prepared)
    assert all(p and p.startswith("diff --git a/src/") for _, _, p in phase.prepared)  # the gold patch
    # the replays are cached: a second score needs no sandbox at all
    assert cli.main(["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]) == 0


def test_plant_a_gold_replay_that_hits_infra_is_retried_and_never_counted_invalid(tmp_path: Path) -> None:
    """Blocker 4: one Docker hiccup on a gold replay was cached and counted as an invalid repro on every later run.
    Here the first replay of each repro errors with INFRA and the next passes: no invalid repro, and only the passes
    are cached."""
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    phase = FakeRunPhase({t: [INFRA, Outcome.PASS] for t in ids[:4]})
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    assert cli.main(args, wiring=wiring(phase)) == 0
    score = json.loads((out / "score.json").read_text())
    spark, glimmer = score["arms"]
    assert (spark["catches"], spark["invalid_repros"], spark["unscored"]) == (4, 0, [])
    assert (glimmer["catches"], glimmer["invalid_repros"], glimmer["invalid_repro_pct"]) == (3, 0, 0.0)
    assert len(phase.ran) == 8  # four repros, each tried twice
    assert all(call["fresh"] is True for call in phase.ran)  # CL4: every replay runs in a fresh container
    cached = [json.loads(x) for x in (out / "gold-replays.jsonl").read_text().splitlines()]
    assert [c["outcome"] for c in cached] == ["pass"] * 4 and all(c["failure_type"] is None for c in cached)


def test_plant_a_gold_replay_that_stays_infra_is_unscored_exits_2_and_is_replayed_next_time(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    broken = FakeRunPhase({ids[0]: [INFRA, RuntimeError("docker: daemon gone")]})
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    assert cli.main(args, wiring=wiring(broken)) == 2 and "unscored" in capsys.readouterr().err
    score = json.loads((out / "score.json").read_text())
    spark, glimmer = score["arms"]
    assert spark["unscored"] == [f"{ids[0]}/fixed_x/a1"] and glimmer["unscored"] == [f"{ids[0]}/fixed_x/a1"]
    assert (spark["invalid_repros"], glimmer["invalid_repros"]) == (0, 0)  # never an invalid repro
    assert score["unscored"] == 2 and score["rule"]["judged"] is False and score["rule"]["met"] is False
    cached = [json.loads(x)["target"] for x in (out / "gold-replays.jsonl").read_text().splitlines()]
    assert ids[0] not in cached and len(cached) == 3
    fixed = FakeRunPhase()
    assert cli.main(args, wiring=wiring(fixed)) == 0  # the next run replays only the unscored one
    assert [r["target"] for r in fixed.ran] == [ids[0]]


def test_an_old_cache_line_with_an_infra_replay_is_not_read_as_an_answer(tmp_path: Path) -> None:
    root, _, ledgers = scored_root(tmp_path)
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    assert cli.main(args, wiring=wiring(FakeRunPhase())) == 0
    path = out / "gold-replays.jsonl"
    lines = [json.loads(x) for x in path.read_text().splitlines()]
    lines[0] = {"key": lines[0]["key"], "target": lines[0]["target"], "outcome": "error"}  # the round-2 format
    path.write_text("".join(json.dumps(x) + "\n" for x in lines))
    again = FakeRunPhase()
    assert cli.main(args, wiring=wiring(again)) == 0
    assert [r["target"] for r in again.ran] == [lines[0]["target"]]


def test_plant_a_provisional_pilot_is_scored_only_with_provisional(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CL1: a pilot drawn before the freeze is a dry run; scoring it needs --provisional."""
    root, _ = full_root(tmp_path, frozen=False)
    assert cli.main(["--root", str(root), "pilot"]) == 0
    cfg = PilotConfig.model_validate_json((root / "eval" / "pilot.json").read_text())
    assert cfg.status == "provisional"
    ledgers = root / "runs" / "ledgers"
    ledgers.mkdir(parents=True)
    for arm in ("spark", "glimmer"):
        (ledgers / f"{arm}.jsonl").write_text(blocker(cfg.targets[0].target_id).model_dump_json() + "\n")
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    assert cli.main(args, wiring=wiring(FakeRunPhase())) == 2 and "provisional" in capsys.readouterr().err
    assert not (out / "score.json").exists()
    assert cli.main([*args, "--provisional"], wiring=wiring(FakeRunPhase())) == 0
    assert json.loads((out / "score.json").read_text())["pilot_status"] == "provisional"


@pytest.mark.parametrize("plant", ["no ledger", "no wiring", "a stranger's finding"])
def test_plant_a_score_that_cannot_be_trusted_is_refused(tmp_path: Path, plant: str) -> None:
    root, _, ledgers = scored_root(tmp_path)
    wired: cli.Wiring | None = wiring(FakeRunPhase())
    if plant == "no ledger":
        (ledgers / "glimmer.jsonl").unlink()
    elif plant == "no wiring":
        wired = None
    else:
        entry = blocker("20260217_other~django__django-9999")
        (ledgers / "glimmer.jsonl").write_text(entry.model_dump_json() + "\n")
    out = root / "runs" / "score"
    assert cli.main(["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)], wiring=wired) == 2
    assert not (out / "score.json").exists()


# ---------------------------------------------------------------- runsc-check exit codes


def runsc_root(tmp_path: Path) -> Path:
    iid = "django__django-1"
    ref = ImageRef(
        instance_id=iid,
        image=f"ghcr.io/epoch-research/swe-bench.eval.x86_64.{iid}",
        digest="sha256:" + "e" * 64,
        media_type="application/vnd.docker.distribution.manifest.v2+json",
    )
    (tmp_path / "eval").mkdir()
    images = ImageSet(resolved_at="2026-10-27T00:00:00Z", images=(ref,), unique_compressed_bytes=0)
    (tmp_path / "eval" / "images.json").write_text(pretty_json(images.model_dump(mode="json")))
    (tmp_path / "ids.txt").write_text(f"{iid}\n")
    (tmp_path / "rows.jsonl").write_text(json.dumps({"instance_id": iid, "patch": "p"}) + "\n")
    return tmp_path


def run_check_cmd(root: Path) -> int:
    return cli.main(
        ["--root", str(root), "runsc-check", "--instances", str(root / "ids.txt"), "--rows", str(root / "rows.jsonl"),
         "--out", str(root / "out"), "--run-id", "t"]
    )  # fmt: skip


@pytest.mark.parametrize(
    ("outcome", "code"),
    [
        (PreflightError("runsc is not registered with Docker"), 2),
        (Report(run_id="t", instances=1, kept=0, verdicts={"infra": 1}, dropped={"x": "infra"}, max_drop_pct=25), 2),
        (Report(run_id="t", instances=1, kept=0, verdicts={"gold_fails": 1}, dropped={"x": "g"}, max_drop_pct=25), 3),
        (Report(run_id="t", instances=1, kept=1, verdicts={"kept": 1}, dropped={}, max_drop_pct=25), 0),
    ],
)
def test_runsc_check_maps_its_outcomes_to_exit_codes_not_tracebacks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, outcome: object, code: int
) -> None:
    def fake_run_check(*args: Any, **kwargs: Any) -> Report:
        if isinstance(outcome, Exception):
            raise outcome
        assert isinstance(outcome, Report)
        return outcome

    monkeypatch.setattr(rc, "run_check", fake_run_check)
    monkeypatch.setattr(rc, "SwebenchGrader", lambda: object())
    assert run_check_cmd(runsc_root(tmp_path)) == code


@pytest.mark.parametrize(("infra", "code"), [((), 0), (("swe_abs_gold~gold~django__django-2000",), 2)])
def test_plant_labels_exits_2_while_any_job_is_infra(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, infra: tuple[str, ...], code: int
) -> None:
    """CL3: a label run with INFRA jobs left exited 0, and the freeze could start on an unfinished run."""
    from receipts.eval import labelrun

    root, _ = full_root(tmp_path, frozen=False)
    monkeypatch.setattr(rc, "preflight", lambda docker, images: None)
    monkeypatch.setattr(rc, "SwebenchGrader", lambda: object())
    summary = labelrun.RunSummary(run_id="d1", planned=1, already_final=0, ran=1, statuses={}, infra=infra)
    monkeypatch.setattr(labelrun, "run_labels", lambda *a, **kw: summary)
    out = root / "runs" / "labels"
    assert cli.main(["--root", str(root), "labels", "--out", str(out), "--run-id", "d1"]) == code


def test_the_label_plan_comes_from_the_manifest(tmp_path: Path) -> None:
    root, manifest = full_root(tmp_path, frozen=False)
    from receipts.eval.rows import load as load_rows

    rows = load_rows(root / "eval" / "data" / ROWS_PATH, expect_sha256=sha256_file(root / "eval" / "data" / ROWS_PATH))
    dropped = manifest.records[0].instance_id
    marked = manifest.model_copy(
        update={
            "records": tuple(
                r.model_copy(update={"gold_runsc": False}) if r.instance_id == dropped else r for r in manifest.records
            )
        }
    )
    inputs = cli.plan_inputs(marked, rows)
    assert dropped in inputs.dropped and cli.plan_inputs(marked, rows, with_drops=False).dropped == frozenset()
    assert len(inputs.resolved) == sum(1 for r in manifest.records if r.resolved is True)
    assert inputs.gold_sha256[dropped] == sha256_text(rows[dropped].patch)
    # an unresolved patch is planned a FAIL_TO_PASS run unless its published report settled it
    unresolved = [r for r in manifest.records if r.resolved is False]
    settled = unresolved[0].model_copy(update={"report_sha256": "c" * 64, "f2p_source": "report"})
    unsettled = unresolved[1].model_copy(update={"report_sha256": "d" * 64, "f2p_source": None})
    ours = unresolved[2].model_copy(update={"f2p_source": "run", "fail_to_pass_passed": False})
    planned = cli.plan_inputs(marked.model_copy(update={"records": (settled, unsettled, ours)}), rows).no_report
    assert [(e, i) for e, i, _ in planned] == [(unsettled.entry, unsettled.instance_id), (ours.entry, ours.instance_id)]


# ---------------------------------------------------------------- round 4: the founder's way out of a persistent INFRA


def test_plant_a_label_drop_needs_the_label_results_and_r24(tmp_path: Path) -> None:
    (tmp_path / "eval" / "pools").mkdir(parents=True)
    (tmp_path / "eval" / "pools" / "swe_abs_251.txt").write_text("django__django-1\n")
    assert cli.main(["--root", str(tmp_path), "manifest", "--label-drop", "django__django-1=x"]) == 2
    assert (
        cli.main(["--root", str(tmp_path), "manifest", "--labels", str(tmp_path), "--label-drop", "django__django-1=x"])
        == 2
    )


def r24_all_kept(root: Path) -> Path:
    """An R24 results file in which every pool instance's gold passed in the run phase on its pinned image."""
    pins = ImageSet.model_validate_json((root / "eval" / "images.json").read_text()).by_instance
    out = root / "runs" / "r24"
    out.mkdir(parents=True)
    lines = [
        R24Result(run_id="d1", instance_id=i, target_id=f"gold~{i}", image=pins[i].ref, runner=RunnerKind.DJANGO,
                  tests=("t.py",), tests_from="test_patch", expected=1, verdict=R24Verdict.KEPT,
                  attempts=({"gvisor": True, "outcome": "pass"},)).model_dump_json()  # type: ignore[arg-type]
        for i in (root / "eval" / "pools" / "swe_abs_251.txt").read_text().split()
    ]  # fmt: skip
    (out / "results.jsonl").write_text("\n".join(lines) + "\n")
    return out


def test_plant_the_manifest_command_lands_a_founder_label_drop_and_records_its_reason(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The wiring of the label drop: the label results' INFRA-only jobs reach land_drops, the landed drop reaches the
    evidence (its records DROPPED) and the manifest's label record (its reason), nothing drops without the flag, and a
    drop that cannot land exits 2. build_manifest needs the real pinned sources, so a recorder stands in for it."""
    from receipts.eval import labelrun
    from receipts.eval import manifest as manifest_mod
    from receipts.eval.rows import load as load_rows

    root, draft = full_root(tmp_path, frozen=False)
    r24_out = r24_all_kept(root)
    seen: list[tuple[Any, Any]] = []

    def recorder(*args: Any, gold: Any = None, labels: Any = None, label_record: Any = None, **kw: Any) -> Any:
        seen.append((labels, label_record))
        verdicts = {} if gold is None else gold.verdicts
        recs = tuple(r.model_copy(update={"gold_runsc": verdicts.get(r.instance_id)}) for r in draft.records)
        return draft.model_copy(update={"records": recs, "labels": label_record}), None

    monkeypatch.setattr(manifest_mod, "build_manifest", recorder)
    rows = load_rows(root / "eval" / "data" / ROWS_PATH, expect_sha256=sha256_file(root / "eval" / "data" / ROWS_PATH))
    kept = draft.model_copy(update={"records": tuple(r.model_copy(update={"gold_runsc": True}) for r in draft.records)})
    jobs = labelrun.plan(cli.plan_inputs(kept, rows, with_drops=False))
    victim = next(j.instance_id for j in jobs if j.kind is labelrun.JobKind.SWE_ABS_PATCH)

    def line(j: labelrun.Job, status: labelrun.JobStatus) -> str:
        done = status is labelrun.JobStatus.DONE
        return labelrun.JobResult(run_id="d1", job=j, image="img", status=status, log_found=done,
                                  statuses={"t": "PASSED"} if done else {}).model_dump_json()  # fmt: skip

    lines: list[str] = []
    for j in jobs:
        if j.instance_id == victim and j.kind is labelrun.JobKind.SWE_ABS_GOLD:
            lines += [line(j, labelrun.JobStatus.INFRA)] * 2  # INFRA on both runs
        else:
            lines.append(line(j, labelrun.JobStatus.DONE))
    (root / "runs" / "labels").mkdir(parents=True)
    (root / "runs" / "labels" / "results.jsonl").write_text("\n".join(lines) + "\n")
    base = ["--root", str(root), "manifest", "--gold-runsc", str(r24_out), "--labels", str(root / "runs" / "labels")]
    assert cli.main(base) == 0
    assert seen[-1][0].dropped == {} and seen[-1][1].drops == ()  # nothing drops without the founder's flag
    reason = "gVisor kills the sandbox at the 4g cap on every run (founder, 10/28)"
    assert cli.main([*base, "--label-drop", f"{victim}={reason}"]) == 0
    labels, record = seen[-1]
    waiting = tuple(sorted(f"{j.entry}~{victim}" for j in jobs
                           if j.kind is labelrun.JobKind.SWE_ABS_PATCH and j.instance_id == victim))  # fmt: skip
    assert labels.dropped == dict.fromkeys(waiting, reason)
    [d] = record.drops
    assert (d.instance_id, d.reason, d.jobs, d.records) == (victim, reason, (f"swe_abs_gold~gold~{victim}",), waiting)
    assert f"label drop (the founder's): {victim}: {reason}" in capsys.readouterr().out
    settled = next(i for i in sorted({j.instance_id for j in jobs}) if i != victim)
    assert cli.main([*base, "--label-drop", f"{settled}=x"]) == 2
    assert "every label job on it has a final result" in capsys.readouterr().err


def test_plant_a_founder_replay_drop_lets_score_judge_and_records_the_reason(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Persistent INFRA's exit path for a gold replay (round-3 review): a replay infra on every run left REG-561 'not
    judged' and score exiting 2, forever. The founder drops it by finding id with a reason: it counts on neither side,
    the rule is judged, and score.json says what was dropped and why. Without the flag nothing is dropped."""
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    stuck = f"{ids[0]}/fixed_x/a1"
    assert cli.main(args, wiring=wiring(FakeRunPhase({ids[0]: [INFRA]}))) == 2  # never dropped on its own
    reason = "gVisor kills the replay at the 4g cap (founder, 10/28)"
    assert cli.main([*args, "--replay-drop", f"{stuck}={reason}"], wiring=wiring(FakeRunPhase({ids[0]: [INFRA]}))) == 0
    score = json.loads((out / "score.json").read_text())
    spark, glimmer = score["arms"]
    assert spark["dropped"] == glimmer["dropped"] == {stuck: reason} and score["replay_drops"] == {stuck: reason}
    assert (spark["unscored"], score["unscored"], spark["catches"], spark["invalid_repros"]) == ([], 0, 3, 0)
    assert score["rule"]["judged"] is True and score["rule"]["founder_dropped"] == 2
    assert "the founder dropped (score --replay-drop)" in score["rule"]["why"]
    cached = [json.loads(x)["target"] for x in (out / "gold-replays.jsonl").read_text().splitlines()]
    assert ids[0] not in cached  # a dropped replay is never cached as an answer
    assert f"replay drop (the founder's): {stuck}: {reason}" in capsys.readouterr().out
    # refused: a drop on a replay that scored, a drop on no blocker, a drop with no reason (before any replay runs)
    assert (
        cli.main([*args, "--replay-drop", f"{ids[1]}/fixed_x/a1=x"], wiring=wiring(FakeRunPhase({ids[0]: [INFRA]})))
        == 2
    )
    assert "its replay on gold scored" in capsys.readouterr().err
    assert cli.main([*args, "--replay-drop", f"{stuck}=x", "--replay-drop", "a~b/fixed_x/a9=y"],
                    wiring=wiring(FakeRunPhase({ids[0]: [INFRA]}))) == 2  # fmt: skip
    assert "no arm has a blocker finding with that id" in capsys.readouterr().err
    idle = FakeRunPhase()
    assert cli.main([*args, "--replay-drop", stuck], wiring=wiring(idle)) == 2 and idle.ran == []


def test_plant_score_leaves_a_dropped_replay_out_of_the_invalid_repro_rate(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The round-4 review's probe, through `receipts eval score`: glimmer has four blockers, two caught on gold, one
    invalid and one whose replay stays infra and is dropped by the founder. The pre-registered rate is 1 of the 3
    scored, 33.3%, so REG-561 is not met; with the drop left in the denominator it read 25.0% and met."""
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    (ledgers / "glimmer.jsonl").write_text("".join(blocker(t, cost=0).model_dump_json() + "\n" for t in ids[:4]))
    stuck = f"{ids[0]}/fixed_x/a1"
    reason = "gVisor kills the replay at the 4g cap (founder, 10/28)"
    out = root / "runs" / "score"
    args = [
        "--root",
        str(root),
        "score",
        "--ledgers",
        str(ledgers),
        "--out",
        str(out),
        "--replay-drop",
        f"{stuck}={reason}",
    ]
    assert cli.main(args, wiring=wiring(FakeRunPhase({ids[0]: [INFRA], ids[1]: Outcome.FAIL}))) == 0
    score = json.loads((out / "score.json").read_text())
    spark, glimmer = score["arms"]
    assert (glimmer["blockers"], glimmer["catches"], glimmer["invalid_repros"]) == (4, 2, 1)
    assert glimmer["dropped"] == {stuck: reason} and glimmer["unscored"] == []
    assert glimmer["invalid_repro_pct"] == 33.3  # 1 invalid over the 3 scored, never over all 4
    assert (spark["blockers"], spark["catches"], spark["invalid_repros"], spark["invalid_repro_pct"]) == (5, 2, 2, 50.0)
    rule = score["rule"]
    assert rule["judged"] is True and rule["met"] is False and rule["candidate_invalid_repro_pct"] == 33.3
    assert "invalid-repro rate is 33.3%, not under 30%" in rule["why"]
    assert "REG-561: not met" in capsys.readouterr().out


def test_plant_a_replay_drop_named_twice_or_with_an_empty_reason_is_refused_before_any_replay(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The round-4 review's XR1 and XR2: parse_replay_drops refuses a finding named twice and a reason that is empty
    after '=', but no test said so. Both are refused, with exit 2, before any replay runs."""
    assert cli.parse_replay_drops(["a~b/fixed_x/a1=x", "c~d/fixed_x/a1=y"]) == {
        "a~b/fixed_x/a1": "x",
        "c~d/fixed_x/a1": "y",
    }
    with pytest.raises(ValueError, match="named twice"):
        cli.parse_replay_drops(["a~b/fixed_x/a1=x", "a~b/fixed_x/a1=y"])
    for bad in ("a~b/fixed_x/a1=", "a~b/fixed_x/a1=   ", "=because", "a~b/fixed_x/a1"):
        with pytest.raises(ValueError, match="want <finding id>=<reason>"):
            cli.parse_replay_drops([bad])
    root, cfg, ledgers = scored_root(tmp_path)
    stuck = f"{cfg.targets[0].target_id}/fixed_x/a1"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(root / "runs" / "score")]
    for extra in (["--replay-drop", f"{stuck}=x", "--replay-drop", f"{stuck}=y"], ["--replay-drop", f"{stuck}=  "]):
        idle = FakeRunPhase()
        assert cli.main([*args, *extra], wiring=wiring(idle)) == 2 and idle.ran == []
        assert "score refused" in capsys.readouterr().err
    assert not (root / "runs" / "score" / "score.json").exists()


def test_a_gold_replay_runs_the_repros_own_cases_as_expect(tmp_path: Path) -> None:
    """Contract v3: a repro that names its cases replays with them as expect, so PASS on gold means each of them ran and
    passed there; a repro that names none replays as before, with no expect."""
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    named = blocker(ids[0], cases=("tests/test_repro.py::test_x",))
    (ledgers / "spark.jsonl").write_text(named.model_dump_json() + "\n" + blocker(ids[1]).model_dump_json() + "\n")
    (ledgers / "glimmer.jsonl").write_text("")
    phase = FakeRunPhase()
    out = root / "runs" / "score"
    assert (
        cli.main(["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)], wiring=wiring(phase))
        == 0
    )
    by = {r["target"]: r for r in phase.ran}
    assert by[ids[0]]["expect"] == ["tests/test_repro.py::test_x"] and by[ids[1]]["expect"] is None


def test_plant_r24_lists_what_it_cannot_name_before_the_run_and_says_what_unblocks_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """The round-3 review's first note: an instance whose FAIL_TO_PASS ids SWE-bench cut at a space is unchecked on
    every run, and the closing hint said to rerun it, which cannot help. R24 judges it as before; the listing now comes
    before the VM preflight, and the hint names the founder's --gold-drop (or rule) for it and a rerun for the rest."""
    root, _ = full_root(tmp_path, f2p={"astropy__astropy-2002": ("tests/test_it.py::test_fix[(1,",)})
    ids = root / "ids.txt"
    ids.write_text("astropy__astropy-2002\ndjango__django-2000\npsf__requests-2001\n")
    before: list[str] = []

    def preflight(docker: object, images: object) -> None:
        before.append(capsys.readouterr().err)

    monkeypatch.setattr(rc, "preflight", preflight)
    out = root / "runs" / "r24"
    args = ["--root", str(root), "r24", "--instances", str(ids), "--out", str(out), "--run-id", "d1", "--workers", "1"]
    assert cli.main(args, wiring=wiring(FakeRunPhase({"gold~psf__requests-2001": INFRA}))) == 2
    assert "before the run: 1 instances have SWE-bench ids R24 cannot name yet" in before[0]
    assert "astropy__astropy-2002: none of its FAIL_TO_PASS ids can be named" in before[0]
    err = capsys.readouterr().err
    assert "2 instances are unchecked; rerun the 1 a rerun can settle with --instances and a new --out" in err
    assert "astropy__astropy-2002 has no FAIL_TO_PASS id R24 can name, so no rerun can change that" in err
    assert "a founder --gold-drop" in err and "the founder's rule for ids SWE-bench cut at a space" in err
    assert [n["instance_id"] for n in json.loads((out / "unnamed.json").read_text())] == ["astropy__astropy-2002"]


def test_plant_r24_prints_the_expected_failures_it_counted_and_the_ones_it_left_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, preflights: list[list[str]], capsys: pytest.CaptureFixture[str]
) -> None:
    """The round-5 review's V18: cmd_r24's two lines on expected failures were covered only through the report's
    fields. They name the instances whose expected failures R24 counted as passing (swebench 4.1.0 grades XFAIL there)
    and those it left skipped (swebench's parser writes no XFAIL there, so the harness grades them failed)."""
    from receipts.eval import r24

    def canned(*a: Any, **kw: Any) -> r24.R24Report:
        return r24.R24Report(
            run_id="d1",
            instances=2,
            verdicts={"gold_skips": 1, "kept": 1},
            dropped={"django__django-2000": "gold_skips"},
            unchecked=(),
            xfail_counted={"psf__requests-2002": 1},
            xfail_skipped={"django__django-2000": 2},
            max_drop_pct=25,
        )

    monkeypatch.setattr(r24, "run_check", canned)
    root, _ = full_root(tmp_path)
    ids = root / "ids.txt"
    ids.write_text("django__django-2000\npsf__requests-2002\n")
    args = ["--root", str(root), "r24", "--instances", str(ids), "--out", str(root / "runs" / "r24"), "--run-id", "d1"]
    assert cli.main(args, wiring=wiring(FakeRunPhase())) == 3  # 1 of 2 dropped is over the 25% limit
    out = capsys.readouterr().out
    assert (
        "1 instances had expected cases that failed as expected, which R24 counted as passing, as swebench 4.1.0 "
        "grades XFAIL in their repositories: {'psf__requests-2002': 1}"
    ) in out
    assert (
        "1 instances had expected cases that failed as expected where swebench 4.1.0's log parser writes no XFAIL "
        "and grades them failed, so they stayed skipped: {'django__django-2000': 2}"
    ) in out


# ---------------------------------------------------------------- round 3: score's fix-validity number (REG-577)


def candidate(target_id: str, *, attack: str = "f1") -> LedgerEntry:
    """What the skeptic writes for a fix-validity repro it could not settle: s_sat_inconclusive after 5 counted
    failures on the patch and 0 passes in 5 base runs. Its test file is not the blockers', so it replays on its own."""
    repro = Repro(
        id="r1",
        attack_id=attack,
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        kind=CatchKind.FIX_VALIDITY,
        files={"tests/test_fv.py": "def test_fv():\n    assert g(2) == 4\n"},
        command=("python", "-m", "pytest", "tests/test_fv.py"),
    )
    runs = (
        *(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5)),
        *(run_result(Outcome.FAIL, Side.BASE) for _ in range(5)),
    )
    finding = Finding(
        id=f"{target_id}/fixed_x/{attack}",
        target_id=target_id,
        claim_id="fixed_x",
        lens=Lens.CORRECTNESS,
        status=Status.S_SAT_INCONCLUSIVE,
        kind=CatchKind.FIX_VALIDITY,
        repro=repro,
        runs=runs,
        detail="SK1: fails on the patch in all 5 runs and passes on base in 0 of 5",
        costs=(StageCost(stage="attack", model="muse-glimmer-30b", micro_usd=0, seconds=30.0),),
    )
    return LedgerEntry(at="2026-10-29T12:00:00Z", receipts_version="0.0.1", finding=finding)


def add(path: Path, *entries: LedgerEntry) -> None:
    with path.open("a", encoding="utf-8") as fh:
        fh.write("".join(e.model_dump_json() + "\n" for e in entries))


def test_score_counts_fix_validity_replays_apart_and_prints_each_arms_fix_validity_number(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The round-2 review's notes 1 to 3, through `receipts eval score`. Score.replays counts the blockers' gold
    replays as it did before REG-577 (4 here, where it read 7), and the candidates' replays are counted apart (3: a
    repro on a GOOD target that both arms found, a repro on a BAD target that two of glimmer's findings share, and one
    that fails on gold). Each arm gets a line with its fix-validity number, and score.json flags a fix-validity catch
    on a GOOD target as GOOD label disputed. The blocker numbers and REG-561's rule do not move."""
    root, cfg, ledgers = scored_root(tmp_path)
    rest = cfg.targets[4:]
    good = next(t.target_id for t in rest if t.label is Label.GOOD)
    bad = next(t.target_id for t in rest if t.label is Label.BAD)
    other = next(t.target_id for t in rest if t.target_id not in (good, bad))
    add(ledgers / "spark.jsonl", candidate(good))
    add(ledgers / "glimmer.jsonl", candidate(good), candidate(bad), candidate(bad, attack="f2"), candidate(other))
    phase = FakeRunPhase({other: Outcome.FAIL})
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    assert cli.main(args, wiring=wiring(phase)) == 0
    score = json.loads((out / "score.json").read_text())
    assert score["replays"] == 4  # the blockers' alone, as before REG-577
    assert (score["fix_validity_replays"], score["fix_validity_unscored"]) == (3, 0)
    assert len(phase.prepared) == 7  # each distinct repro once, whichever number needs it
    spark, glimmer = score["arms"]
    fv = glimmer["fix_validity"]
    assert (fv["candidates"], fv["catches"], fv["catch_findings"], fv["invalid_repros"]) == (4, 2, 3, 1)
    assert fv["catches_by_label"] == {"BAD": 1, "GOOD": 1} and fv["good_label_disputed"] == 1
    assert (fv["invalid_on_gold"], fv["unscored"], fv["dropped"]) == ({"fail": 1}, [], {})
    assert (spark["fix_validity"]["catches_by_label"], spark["fix_validity"]["good_label_disputed"]) == ({"GOOD": 1}, 1)
    assert (spark["blockers"], spark["catches"], spark["invalid_repros"], spark["good_label_disputed"]) == (5, 4, 0, 0)
    assert (glimmer["blockers"], glimmer["catches"], glimmer["invalid_repros"]) == (3, 3, 0)
    assert score["rule"]["judged"] is True and score["rule"]["candidate_catches"] == 3
    printed = capsys.readouterr().out
    assert (
        "glimmer fix-validity on gold (REG-577, outside REG-561's rule): 4 candidates, 2 targets caught "
        "{'BAD': 1, 'GOOD': 1} (3 findings), 1 invalid repros, 0 unscored, 0 dropped by the founder"
    ) in printed
    assert (
        "spark fix-validity on gold (REG-577, outside REG-561's rule): 1 candidates, 1 targets caught {'GOOD': 1} "
        "(1 findings), 0 invalid repros, 0 unscored, 0 dropped by the founder"
    ) in printed


def test_plant_an_unscored_fix_validity_replay_exits_2_like_a_blockers_until_it_scores_or_the_founder_drops_it(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The round-2 review's note 2: a fix-validity replay that stayed infra let score exit 0. It is handled as a
    blocker's is: score.json is written, the replay is never cached, score exits 2 and says what is unscored, the
    next score replays only that one, and if it stays infra the founder may drop it by finding id. REG-561's rule is
    judged throughout, since the fix-validity number never enters it."""
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    add(ledgers / "glimmer.jsonl", candidate(ids[5]), candidate(ids[6]))
    stuck = f"{ids[5]}/fixed_x/f1"
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    assert cli.main(args, wiring=wiring(FakeRunPhase({ids[5]: [INFRA, RuntimeError("docker: daemon gone")]}))) == 2
    err = capsys.readouterr().err
    assert "1 fix-validity replays on gold hit infra twice and are unscored" in err and "--replay-drop" in err
    assert "blocker replays" not in err
    score = json.loads((out / "score.json").read_text())
    fv = score["arms"][1]["fix_validity"]
    assert (fv["unscored"], fv["invalid_repros"], fv["catches"], score["fix_validity_unscored"]) == ([stuck], 0, 1, 1)
    assert score["unscored"] == 0 and score["rule"]["judged"] is True
    cached = [json.loads(x)["target"] for x in (out / "gold-replays.jsonl").read_text().splitlines()]
    assert ids[5] not in cached and ids[6] in cached
    # with a blocker's replay unscored too, the message names both
    both = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(root / "runs" / "both")]
    assert cli.main(both, wiring=wiring(FakeRunPhase({ids[0]: [INFRA], ids[5]: [INFRA]}))) == 2
    assert "2 blocker replays and 1 fix-validity replays on gold hit infra twice" in capsys.readouterr().err
    # the founder's drop: on neither side of the fix-validity number, and outside REG-561's founder_dropped
    reason = "gVisor kills the replay at the 4g cap (founder, 10/29)"
    assert cli.main([*args, "--replay-drop", f"{stuck}={reason}"], wiring=wiring(FakeRunPhase({ids[5]: [INFRA]}))) == 0
    score = json.loads((out / "score.json").read_text())
    fv = score["arms"][1]["fix_validity"]
    assert (fv["unscored"], fv["dropped"], fv["invalid_repros"]) == ([], {stuck: reason}, 0)
    assert score["fix_validity_unscored"] == 0 and score["replay_drops"] == {stuck: reason}
    assert score["rule"]["judged"] is True and score["rule"]["founder_dropped"] == 0
    assert f"replay drop (the founder's): {stuck}: {reason}" in capsys.readouterr().out
    # without the drop, the next score replays only the unscored one; once it scored, a drop on it is refused
    fixed = FakeRunPhase()
    assert cli.main(args, wiring=wiring(fixed)) == 0
    assert [r["target"] for r in fixed.ran] == [ids[5]]
    assert json.loads((out / "score.json").read_text())["arms"][1]["fix_validity"]["catches"] == 2
    assert cli.main([*args, "--replay-drop", f"{stuck}=x"], wiring=wiring(FakeRunPhase())) == 2
    assert "its replay on gold scored" in capsys.readouterr().err


# ---------------------------------------------------------------- fix round 5: the gold replay's own output, cached


def test_a_gold_replay_s_output_is_cached_with_it_bounded_and_an_old_line_still_reads(tmp_path: Path) -> None:
    # The 10/29 diagnosis had to infer why 33 repros failed on gold, since the cache kept only the outcome.
    root, cfg, ledgers = scored_root(tmp_path)
    ids = [t.target_id for t in cfg.targets]
    said = run_result(
        Outcome.FAIL,
        stdout_tail="x" * 3_000 + "1 failed",
        stderr_tail="AssertionError: assert 3 == 4",
        problem="",
    )
    out = root / "runs" / "score"
    args = ["--root", str(root), "score", "--ledgers", str(ledgers), "--out", str(out)]
    assert cli.main(args, wiring=wiring(FakeRunPhase({ids[0]: said}))) == 0
    path = out / "gold-replays.jsonl"
    lines = {x["target"]: x for x in map(json.loads, path.read_text().splitlines())}
    first = lines[ids[0]]
    assert (first["outcome"], first["stderr_tail"], first["problem"]) == ("fail", "AssertionError: assert 3 == 4", "")
    assert first["stdout_tail"] == ("x" * 3_000 + "1 failed")[-GOLD_OUTPUT_CHARS:]  # its last characters only
    assert lines[ids[1]]["stdout_tail"] == "" and lines[ids[1]]["outcome"] == "pass"
    cache = cli._read_replay_cache(path)
    assert cache[first["key"]].stderr_tail == "AssertionError: assert 3 == 4"
    # A line as the cache wrote it before (key, target, outcome, failure type) still reads, with empty tails.
    old = {k: first[k] for k in ("key", "target", "outcome", "failure_type")}
    path.write_text(json.dumps(old) + "\n")
    (replay,) = cli._read_replay_cache(path).values()
    assert (replay.outcome, replay.failure_type, replay.stdout_tail, replay.stderr_tail) == (
        Outcome.FAIL,
        FailureType.ASSERTION,
        "",
        "",
    )
