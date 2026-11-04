"""Demo inputs, offline fix replay, IR-1 and an explicitly opted-in local-image smoke test."""

from __future__ import annotations

import io
import json
import os
import re
import shlex
import shutil
import subprocess
import tarfile
from pathlib import Path

import pytest

from receipts.attack.sandbox import DockerSandbox, docker_available, refused_env_names, workspace_root
from receipts.contracts import (
    RUN_PHASE_LIMITS,
    Arm,
    CatchKind,
    Finding,
    LedgerEntry,
    Lens,
    Outcome,
    PatchMethod,
    ReplayKey,
    Repro,
    RunnerKind,
    Side,
    Status,
    Target,
    Tier,
)
from receipts.core.pipeline import RunItem
from receipts.eval import cli
from receipts.eval.demo import FixCheck, demo_target, fix_check, load_pins, write_run_file
from receipts.eval.hashing import sha256_text

from .conftest import FakeRunPhase, fake_runners, run_result

PINS_DIR = Path(__file__).resolve().parents[2] / "demo" / "pandas-64677"
IMAGE = "127.0.0.1:5000/receipts-demo/pandas-64677@sha256:" + "e" * 64


@pytest.fixture
def pins_dir(tmp_path: Path) -> Path:
    for name in ("pins.json", "patch.diff", "fix.diff", "target.template.json"):
        shutil.copy2(PINS_DIR / name, tmp_path / name)
    # Build evidence may already pin the real VM image. Unit tests use a synthetic digest.
    pins = json.loads((tmp_path / "pins.json").read_text())
    pins["image"]["ref"] = None
    (tmp_path / "pins.json").write_text(json.dumps(pins))
    return tmp_path


def blocker(target: Target, *, cases: bool = True) -> Finding:
    repro = Repro(
        id="repro",
        attack_id="attack",
        claim_id="bool",
        lens=Lens.REGRESSION,
        kind=CatchKind.REGRESSION,
        files={"test_demo_repro.py": "def test_identity():\n    assert True\n"},
        command=("python", "-m", "pytest", "test_demo_repro.py"),
        cases=("test_demo_repro.py::test_identity",) if cases else (),
    )
    assert target.patch_sha is not None
    assert target.image is not None
    return Finding(
        id=f"{target.id}/bool/attack",
        target_id=target.id,
        claim_id="bool",
        lens=Lens.REGRESSION,
        kind=CatchKind.REGRESSION,
        status=Status.BLOCKER,
        repro=repro,
        runs=tuple(run_result(Outcome.FAIL, Side.PATCH) for _ in range(5))
        + tuple(run_result(Outcome.PASS, Side.BASE) for _ in range(5)),
        quote="Verified result dtype is bool",
        replay=ReplayKey(
            base_sha=target.base_sha,
            patch_sha=target.patch_sha,
            harness_hash="demo-test",
            seed=0,
            image_digest=target.image.split("@", 1)[1],
        ),
    )


def test_target_round_trip_pins_patch_and_run_item(pins_dir: Path) -> None:
    target = demo_target(pins_dir, IMAGE)
    pins = load_pins(pins_dir)
    assert Target.model_validate_json(target.model_dump_json()) == target
    assert target.id == "demo~pandas-64677"
    assert target.tier is Tier.B
    assert target.repo == pins.repo.value == "pandas-dev/pandas"
    assert target.base_sha == pins.base_sha.value == "82315e7fb597d53f797f62d2a83f9ee9dcad5042"
    assert target.patch_sha == pins.merge_sha.value == "d27e2f320150e3b064f3256a8bada7478e89da53"
    assert target.patch.encode() == (pins_dir / "patch.diff").read_bytes()
    assert target.image == IMAGE
    out = pins_dir / "run.jsonl"
    write_run_file(out, target)
    assert len(out.read_text().splitlines()) == 1
    item = RunItem.model_validate_json(out.read_text())
    assert item.target == target
    assert item.temperatures.value == "spec"
    arms = json.loads((PINS_DIR.parent / "arms.json").read_text())
    assert [Arm.model_validate(arm) for arm in arms] == [Arm(name="spark", attacker_model="muse-spark-1.3")]


@pytest.mark.parametrize("name", ["patch", "fix"])
def test_hash_mismatch_is_refused(pins_dir: Path, name: str) -> None:
    path = pins_dir / "pins.json"
    pins = json.loads(path.read_text())
    pins[f"{name}_diff_sha256"]["value"] = "0" * 64
    path.write_text(json.dumps(pins))
    with pytest.raises(ValueError, match=f"{name}.diff: sha256"):
        demo_target(pins_dir, IMAGE)


def test_target_contains_only_the_pr_claim(pins_dir: Path) -> None:
    target = demo_target(pins_dir, IMAGE)
    text = "\n".join((target.title, target.body, target.issue, target.commit_messages)).lower()
    for phrase in (
        "65100",
        "65101",
        "unobserved categorical groups",
        "boolean identity value",
        "use_bool_fastpath",
        "test_numpy_bool_unobserved_categorical_group_returns_identity",
        "groupby.any()`/`all()` returns `nan`",
    ):
        assert phrase not in text
    assert target.issue == target.commit_messages == target.session_summary == ""
    assert "24,928" in target.body


@pytest.mark.parametrize("image", ["sha256:" + "a" * 64, "pandas:latest", "", "-bad@sha256:" + "a" * 64])
def test_image_must_be_a_named_digest(pins_dir: Path, image: str) -> None:
    with pytest.raises(ValueError):
        demo_target(pins_dir, image)


def test_pinned_image_cannot_be_overridden(pins_dir: Path) -> None:
    path = pins_dir / "pins.json"
    pins = json.loads(path.read_text())
    pins["image"]["ref"] = IMAGE.replace("e" * 64, "a" * 64)
    path.write_text(json.dumps(pins))
    with pytest.raises(ValueError, match="disagrees"):
        demo_target(pins_dir, IMAGE)


@pytest.mark.parametrize("field,value", [("base_sha", "a" * 40), ("patch", "wrong"), ("issue", "future issue")])
def test_template_must_match_the_pins_and_have_no_issue(pins_dir: Path, field: str, value: str) -> None:
    path = pins_dir / "target.template.json"
    template = json.loads(path.read_text())
    template[field] = value
    path.write_text(json.dumps(template))
    with pytest.raises(ValueError, match="disagrees"):
        demo_target(pins_dir, IMAGE)


@pytest.mark.parametrize("last", [Outcome.PASS, Outcome.FAIL, Outcome.ERROR, Outcome.SKIPPED, Outcome.TIMEOUT])
@pytest.mark.parametrize("cases", [True, False])
def test_fix_check_requires_five_passes_and_closes(pins_dir: Path, last: Outcome, cases: bool) -> None:
    target = demo_target(pins_dir, IMAGE)
    finding = blocker(target, cases=cases)
    assert finding.repro is not None
    phase = FakeRunPhase({target.id: [Outcome.PASS] * 4 + [last]})
    diff = (pins_dir / "fix.diff").read_bytes().decode()
    result = fix_check(target, finding, diff, sandbox=phase, runner=fake_runners()[RunnerKind.PYTEST])
    assert result.passes_on_fix is (last is Outcome.PASS)
    assert len(result.runs) == 5
    assert result.finding_id == finding.id
    assert result.fix_diff_sha256 == sha256_text(diff)
    assert result.image_digest == "sha256:" + "e" * 64
    assert phase.prepared == [(target.id, Side.GOLD, diff)]
    assert phase.closed == 1
    assert [r["seed"] for r in phase.ran] == list(range(5))
    for call in phase.ran:
        assert call["fresh"] is True
        assert call["files"] == finding.repro.files
        assert call["expect"] == (finding.repro.cases or None)
        assert call["tests"] == ("test_demo_repro.py",)
        assert call["changed_files"] == [
            "doc/source/whatsnew/v3.1.0.rst",
            "pandas/core/groupby/groupby.py",
            "pandas/tests/groupby/test_reductions.py",
        ]
    assert FixCheck.model_validate_json(result.model_dump_json()) == result
    with pytest.raises(ValueError, match="agree"):
        FixCheck.model_validate({**result.model_dump(), "passes_on_fix": not result.passes_on_fix})


def test_nonblocker_and_wrong_replay_are_refused_before_prepare(pins_dir: Path) -> None:
    target = demo_target(pins_dir, IMAGE)
    finding = blocker(target)
    assert finding.replay is not None
    phase = FakeRunPhase()
    for bad in (
        Finding.model_validate({**finding.model_dump(), "status": Status.NOT_REPRODUCED}),
        Finding.model_validate(
            {**finding.model_dump(), "replay": {**finding.replay.model_dump(), "base_sha": "a" * 40}}
        ),
    ):
        with pytest.raises(ValueError):
            fix_check(target, bad, "diff", sandbox=phase, runner=fake_runners()[RunnerKind.PYTEST])
    with pytest.raises(ValueError, match="five"):
        fix_check(target, finding, "diff", sandbox=phase, runner=fake_runners()[RunnerKind.PYTEST], runs=4)
    assert not phase.prepared


def test_fix_check_closes_on_runtime_error(pins_dir: Path) -> None:
    target = demo_target(pins_dir, IMAGE)
    phase = FakeRunPhase({target.id: RuntimeError("sandbox stopped")})
    with pytest.raises(RuntimeError, match="sandbox stopped"):
        fix_check(target, blocker(target), "diff", sandbox=phase, runner=fake_runners()[RunnerKind.PYTEST])
    assert phase.closed == 1


@pytest.mark.parametrize("last,code", [(Outcome.PASS, 0), (Outcome.FAIL, 1)])
def test_cli_target_and_read_only_fixcheck(pins_dir: Path, last: Outcome, code: int) -> None:
    out = pins_dir / "target.jsonl"
    assert cli.main(["demo", "target", "--pins", str(pins_dir), "--image", IMAGE, "--out", str(out)]) == 0
    target = RunItem.model_validate_json(out.read_text()).target
    finding = blocker(target)
    ledger = pins_dir / "spark.jsonl"
    entry = LedgerEntry(at="2026-10-30T00:00:00Z", receipts_version="test", finding=finding)
    ledger.write_text(entry.model_dump_json() + "\n")
    before = ledger.read_bytes()
    ledger.chmod(0o444)
    phase = FakeRunPhase({target.id: [Outcome.PASS] * 4 + [last]})
    result_path = pins_dir / "fixcheck.json"
    args = [
        "demo", "fixcheck", "--target", str(out), "--ledger", str(ledger), "--finding", finding.id,
        "--fix", str(pins_dir / "fix.diff"), "--out", str(result_path),
    ]  # fmt: skip
    wiring = cli.Wiring(sandbox=lambda: phase, runners=fake_runners)
    assert cli.main(args, wiring=wiring) == code
    assert FixCheck.model_validate_json(result_path.read_text()).passes_on_fix is (code == 0)
    assert ledger.read_bytes() == before
    assert cli.main(args) == 2
    args[args.index(finding.id)] = target.id + "/absent/attack"
    assert cli.main(args, wiring=wiring) == 2
    args[args.index(target.id + "/absent/attack")] = finding.id
    result_path.unlink()
    phase.outcomes[target.id] = [RuntimeError("runsc unavailable")]
    assert cli.main(args, wiring=wiring) == 2
    assert not result_path.exists()
    assert phase.closed == 2
    # A later reclassification takes precedence over the earlier blocker.
    ledger.chmod(0o644)
    latest = Finding.model_validate({**finding.model_dump(), "status": Status.NOT_REPRODUCED})
    ledger.write_text(
        entry.model_dump_json() + "\n" + entry.model_copy(update={"finding": latest}).model_dump_json() + "\n"
    )
    prepares = len(phase.prepared)
    assert cli.main(args, wiring=wiring) == 2
    assert len(phase.prepared) == prepares


def test_cli_bad_input_is_exit_two(pins_dir: Path) -> None:
    assert (
        cli.main(
            [
                "demo",
                "target",
                "--pins",
                str(pins_dir),
                "--image",
                "unpinned",
                "--out",
                str(pins_dir / "out"),
            ]
        )
        == 2
    )
    assert not (pins_dir / "out").exists()


def test_tier_b_without_checkout_copies_its_image_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    import receipts.attack.sandbox as sandbox_mod

    tree = tmp_path / "image-tree"
    tree.mkdir()
    (tree / "module.py").write_text("base\n")
    target = Target(id="image-b", tier=Tier.B, repo="o/r", base_sha="a" * 40, patch="patch", image=IMAGE)
    sandbox = DockerSandbox(runsc=True, work_dir=tmp_path)
    copied: list[tuple[str, str]] = []

    def image_tree(image: str, root: str) -> Path:
        copied.append((image, root))
        return tree

    def apply(repo: Path, diff: str, **kwargs: object) -> PatchMethod:
        assert diff == target.patch
        (repo / "module.py").write_text("patched\n")
        return PatchMethod.GIT_APPLY

    monkeypatch.setattr(sandbox, "_image_tree", image_tree)
    monkeypatch.setattr(sandbox_mod, "apply_patch", apply)
    root = workspace_root(target)
    host, method, _ = sandbox._tree_for(target, Side.PATCH, target.patch, IMAGE, "runsc", root)
    assert copied == [(IMAGE, "/workspace")]
    assert host is not None and host != tree
    assert (host / "module.py").read_text() == "patched\n"
    assert (tree / "module.py").read_text() == "base\n"
    assert method is PatchMethod.GIT_APPLY


def _docker(*args: str, stdin: bytes | None = None, timeout: float = 120) -> bytes:
    result = subprocess.run(["docker", *args], input=stdin, capture_output=True, timeout=timeout, check=False)
    assert result.returncode == 0, (result.stdout + result.stderr).decode(errors="replace")[-12000:]
    return result.stdout


def test_both_demo_recipes_install_git_for_fix_and_surrogate_inspection() -> None:
    assert "--no-install-recommends build-essential git" in (PINS_DIR / "Dockerfile").read_text()
    assert "git" in json.loads((PINS_DIR / "pins.json").read_text())["apt"]["packages"]
    other = PINS_DIR.parent / "check-demo"
    assert "--no-install-recommends git" in (other / "Dockerfile").read_text()
    assert all("git --version" in (folder / "RUNBOOK.md").read_text() for folder in (PINS_DIR, other))


def test_demo_recipes_use_no_heredoc_and_the_pandas_build_imports_the_workspace_alone() -> None:
    # The eval VM's docker has no buildx, and its legacy builder runs a RUN heredoc with an empty body: the pandas
    # image's extension copy never ran there, and the image built "successfully" with no extension (FAL-643).
    dockerfiles = sorted(PINS_DIR.parent.glob("*/Dockerfile"))
    assert [path.parent.name for path in dockerfiles] == ["check-demo", "pandas-64677"]
    for path in dockerfiles:
        assert "<<" not in path.read_text(), f"{path}: a heredoc never runs under the eval VM's legacy builder"
    # A build that leaves the workspace unable to import pandas fails the build, not only the smoke test.
    assert "PYTHONPATH=/workspace python -B -c" in (PINS_DIR / "Dockerfile").read_text()
    assert "assert extensions > 0" in (PINS_DIR / "Dockerfile").read_text()


def test_the_pandas_workspace_keeps_pytest_rootdir_at_the_workspace_root() -> None:
    # The wheel installs pandas/pyproject.toml for pandas.test(). Copied into /workspace/pandas it became pytest's
    # rootdir, so every case id lost its pandas/ prefix and every SK1 replay reported its case never ran (FAL-643).
    text = (PINS_DIR / "Dockerfile").read_text()
    assert 'source == installed / "pyproject.toml"' in text
    assert 'assert not (tree / "pyproject.toml").exists()' in text
    # The build fails unless pytest, run from /workspace, collects ids under pandas/.
    assert "--collect-only -q" in text
    assert "grep -q '^pandas/tests/groupby/test_api.py::'" in text


@pytest.mark.docker
@pytest.mark.skipif(os.environ.get("RECEIPTS_RUN_DOCKER") != "1", reason="Docker is opt-in; hold the shared heavy lock")
def test_docker_pandas_base_import_suite_and_both_diffs(tmp_path: Path) -> None:
    if not docker_available():
        pytest.skip("Docker is not installed or its daemon is unavailable")
    image = os.environ.get("RECEIPTS_DEMO_IMAGE") or load_pins(PINS_DIR).image.ref
    assert image, "build the demo image or set RECEIPTS_DEMO_IMAGE to an existing local image"
    info = json.loads(_docker("image", "inspect", image))[0]
    image_env = {entry.partition("=")[0]: entry.partition("=")[2] for entry in info["Config"].get("Env") or []}
    assert not refused_env_names(image_env)
    created = _docker("create", "--pull=never", "--network=none", "--read-only", "--user=65534:65534", image, "true")
    container = created.decode().strip()
    tree = tmp_path / "tree"
    tree.mkdir()
    try:
        _docker("cp", f"{container}:/workspace/.", str(tree), timeout=300)
    finally:
        _docker("rm", "-f", container)
    assert _docker(
        "run", "--rm", "--pull=never", "--network=none", "--read-only", "--user=65534:65534", image, "git", "--version"
    ).startswith(b"git version ")
    for name in ("patch.diff", "fix.diff"):
        result = subprocess.run(
            ["git", "-C", str(tree), "apply", "--check", str(PINS_DIR / name)],
            capture_output=True,
            text=True,
            check=False,
            env={
                "PATH": os.environ["PATH"],
                "GIT_CONFIG_GLOBAL": "/dev/null",
                "GIT_CONFIG_SYSTEM": "/dev/null",
                "GIT_CEILING_DIRECTORIES": str(tmp_path),
            },
        )
        assert result.returncode == 0, result.stderr
    stream = io.BytesIO()
    with tarfile.open(fileobj=stream, mode="w") as archive:
        archive.add(tree, arcname=".")
    command = Target.model_validate_json((PINS_DIR / "target.template.json").read_bytes()).test_command
    probe = (
        "import json,pathlib,subprocess,sys,time,xml.etree.ElementTree as ET; "
        "import pandas; assert pandas.__file__.startswith('/workspace/'); "
        "print('RECEIPTS-IMPORT '+pandas.__version__+' '+pandas.__file__, flush=True); "
        "size=int(subprocess.check_output(['du','-sb','/workspace']).split()[0]); assert size < 700000000; "
        "start=time.monotonic(); "
        f"rc=subprocess.call({[*shlex.split(command), 'pandas/tests/groupby', '--junitxml=/tmp/groupby.xml']!r}); "
        "wall=time.monotonic()-start; "
        "suites=ET.parse('/tmp/groupby.xml').getroot(); "
        "counts={k:sum(int(s.get(k,0)) for s in suites.iter('testsuite')) "
        "for k in ['tests','failures','errors','skipped']}; "
        # pytest's rootdir is /workspace, so every case id reads pandas/tests/... as a replay's expected case does:
        # a pandas/pyproject.toml in the workspace made it tests/... and every SK1 replay 'never ran' (FAL-643).
        # Every testcase with a classname (a module skipped at collection has none) must be under pandas.tests.
        "names=[c.get('classname') for c in suites.iter('testcase') if c.get('classname')]; "
        "assert names and all(n.startswith('pandas.tests.groupby.') for n in names), names[:3]; "
        "dpkg=pathlib.Path('/usr/local/share/receipts-demo-dpkg.txt').read_text(); "
        "apt={line.split()[1]:line.split()[2] for line in dpkg.splitlines() if line.startswith('ii ')}; "
        "print('RECEIPTS-EVIDENCE '+json.dumps(dict(workspace_bytes=size,wall_seconds=wall,suite_counts=counts,"
        "apt_versions=apt,import_path=pandas.__file__,version=pandas.__version__)), flush=True); sys.exit(rc)"
    )
    script = "mkdir -p /tmp/home && export HOME=/tmp/home && tar --no-same-owner -xf - -C /workspace && exec python -c "
    output = _docker(
        "run",
        "--rm",
        "-i",
        "--pull=never",
        *RUN_PHASE_LIMITS.run_flags(),
        "-w",
        "/workspace",
        image,
        "sh",
        "-c",
        script + shlex.quote(probe),
        stdin=stream.getvalue(),
        timeout=1800,
    ).decode(errors="replace")
    matches = re.findall(r"^RECEIPTS-EVIDENCE (.+)$", output, flags=re.MULTILINE)
    assert len(matches) == 1
    evidence = json.loads(matches[0])
    assert evidence["suite_counts"]["tests"] > 0
    assert evidence["suite_counts"]["failures"] == evidence["suite_counts"]["errors"] == 0
    assert "build-essential" in evidence["apt_versions"]
    assert "git" in evidence["apt_versions"]
    print(json.dumps(evidence, indent=2))
    if path := os.environ.get("RECEIPTS_DEMO_EVIDENCE"):
        Path(path).write_text(json.dumps(evidence, indent=2) + "\n")
