"""`receipts demo` rebuilds the pandas demo's receipts from the shipped records with no network and no key."""

from __future__ import annotations

import json
import os
import shutil
import socket
import subprocess
import sys
from pathlib import Path

import pytest

from receipts import cli
from receipts.surface.demo_command import DEFAULT_EVIDENCE, PAGES

REPO = Path(__file__).resolve().parents[2]
# Every credential the README's security posture names, and the AWS and GitHub variables a client could pick up.
CREDENTIALS = (
    "FALDEN_META_API_KEY",
    "MODAL_PROXY_TOKEN_ID",
    "MODAL_PROXY_TOKEN_SECRET",
    "RECEIPTS_LINEAR_API_KEY",
    "RECEIPTS_GITHUB_APP_ID",
    "RECEIPTS_GITHUB_APP_INSTALLATION_ID",
    "RECEIPTS_GITHUB_APP_KEY_FILE",
    "AWS_ACCESS_KEY_ID",
    "AWS_SECRET_ACCESS_KEY",
    "AWS_SESSION_TOKEN",
    "AWS_PROFILE",
    "GITHUB_TOKEN",
)
CANARY = "canary-credential-0f3b"

# Runs in a fresh interpreter, so every import of receipts.cli and what it pulls in happens under the watch. It swaps
# os.environ for a recorder, blocks the network and every way to start a process, and only then imports the CLI.
CHILD = r"""
import json
import os
import socket
import subprocess
import sys

report_path, argv = sys.argv[1], sys.argv[2:]
read = []
blocked = []


class WatchedEnviron(dict):
    def __getitem__(self, key):
        read.append(key)
        return super().__getitem__(key)

    def get(self, key, default=None):
        read.append(key)
        return super().get(key, default)

    def pop(self, key, *default):
        read.append(key)
        return super().pop(key, *default)

    def setdefault(self, key, default=None):
        read.append(key)
        return super().setdefault(key, default)

    def __contains__(self, key):
        read.append(str(key))
        return super().__contains__(key)

    # Scanning or copying the whole environment counts as reading every key.
    def __iter__(self):
        read.append("*")
        return super().__iter__()

    def keys(self):
        read.append("*")
        return super().keys()

    def items(self):
        read.append("*")
        return super().items()

    def values(self):
        read.append("*")
        return super().values()

    def copy(self):
        read.append("*")
        return dict(super().items())


def refuse(what):
    def blocked_call(*args, **kwargs):
        blocked.append(what)
        raise AssertionError(f"receipts demo tried {what}")
    return blocked_call


os.environ = WatchedEnviron(os.environ)
for name in ("connect", "connect_ex"):
    setattr(socket.socket, name, refuse(f"socket.{name}"))
socket.create_connection = refuse("socket.create_connection")
socket.getaddrinfo = refuse("socket.getaddrinfo")
subprocess.Popen._execute_child = refuse("subprocess")
for name in ("system", "popen", "fork", "forkpty", "posix_spawn", "posix_spawnp", "execv", "execve", "execvp",
             "execvpe", "execl", "execle", "execlp", "execlpe", "spawnv", "spawnve", "spawnvp", "spawnvpe"):
    if hasattr(os, name):
        setattr(os, name, refuse(f"os.{name}"))

from receipts import cli  # noqa: E402

code = cli.main(argv)
with open(report_path, "w", encoding="utf-8") as report:
    json.dump({"code": code, "read": read, "blocked": blocked}, report)
"""


def _run_child(tmp_path: Path, *argv: str) -> tuple[dict[str, object], subprocess.CompletedProcess[str]]:
    script = tmp_path / "watched_demo.py"
    script.write_text(CHILD, encoding="utf-8")
    report = tmp_path / "report.json"
    done = subprocess.run(
        [sys.executable, str(script), str(report), *argv],
        cwd=REPO,
        env={**os.environ, **dict.fromkeys(CREDENTIALS, CANARY)},
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert report.exists(), f"the watched run died before reporting:\n{done.stderr}"
    return json.loads(report.read_text(encoding="utf-8")), done


def _no_network(*_: object, **__: object) -> None:
    raise AssertionError("receipts demo opened a network connection")


def _no_process(*_: object, **__: object) -> None:
    raise AssertionError("receipts demo started a process")


@pytest.fixture
def offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(socket.socket, "connect", _no_network)
    monkeypatch.setattr(socket.socket, "connect_ex", _no_network)
    monkeypatch.setattr(socket, "create_connection", _no_network)
    monkeypatch.setattr(socket, "getaddrinfo", _no_network)
    monkeypatch.setattr(subprocess.Popen, "_execute_child", _no_process)
    monkeypatch.chdir(REPO)


def test_demo_writes_both_receipts_offline_without_reading_a_credential(tmp_path: Path) -> None:
    out = tmp_path / "demo"
    result, done = _run_child(tmp_path, "demo", "--out-dir", str(out))
    assert result["code"] == 0, done.stderr
    assert result["blocked"] == [], f"receipts demo tried {result['blocked']}"
    touched = set(result["read"])  # type: ignore[call-overload]
    assert "*" not in touched, "receipts demo scanned or copied the whole environment"
    assert not touched & set(CREDENTIALS), f"receipts demo read a credential: {sorted(touched & set(CREDENTIALS))}"

    run = (out / "pandas-64677.html").read_text(encoding="utf-8")
    fix = (out / "pandas-64677-fix-5.html").read_text(encoding="utf-8")
    assert "9 blockers" in run and "d27e2f320150e3b064f3256a8bada7478e89da53" in run
    assert "1 blocker" in fix and "correctness-9.1" in fix
    for page in (run, fix):
        assert CANARY not in page
    assert CANARY not in done.stdout + done.stderr
    assert "no model call" in done.stdout


def test_demo_outside_the_repository_says_where_to_run_it(
    offline: None, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    assert cli.main(["demo", "--out-dir", str(tmp_path / "out")]) == 2
    assert "repository root" in capsys.readouterr().err
    assert not (tmp_path / "out").exists()


def test_demo_refuses_a_ledger_without_the_demo_target(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence = tmp_path / "evidence"
    for page in ("run-2026-10-31T134142Z/spark.jsonl",):
        (evidence / page).parent.mkdir(parents=True)
        (evidence / page).write_text("", encoding="utf-8")
    assert cli.main(["demo", "--evidence", str(evidence), "--out-dir", str(tmp_path / "out")]) == 3
    assert "demo~pandas-64677" in capsys.readouterr().err


def _copy_evidence(tmp_path: Path, pages: int = len(PAGES)) -> Path:
    evidence = tmp_path / "evidence"
    for page in PAGES[:pages]:
        (evidence / page.ledger).parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / DEFAULT_EVIDENCE / page.ledger, evidence / page.ledger)
    return evidence


def test_demo_warns_on_a_torn_last_line_and_leaves_the_ledger_alone(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence = _copy_evidence(tmp_path)
    ledger = evidence / PAGES[0].ledger
    with ledger.open("ab") as handle:
        handle.write(b'{"torn": ')
    before = ledger.read_bytes()
    assert cli.main(["demo", "--evidence", str(evidence), "--out-dir", str(tmp_path / "out")]) == 0
    err = capsys.readouterr().err
    assert f"receipts demo: {ledger}: skipped torn last line" in err
    assert err.count("skipped torn last line") == 1
    assert ledger.read_bytes() == before


@pytest.mark.parametrize("inside", ["demo", "demo/pandas-64677/evidence", "demo/pandas-64677/evidence/run-x"])
def test_demo_refuses_to_write_into_the_shipped_records(
    offline: None, inside: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    root = tmp_path / "repo"  # a copy of the shipped records, so a broken guard cannot write into the real ones
    shutil.copytree(_copy_evidence(tmp_path), root / DEFAULT_EVIDENCE)
    monkeypatch.chdir(root)
    before = sorted(str(path) for path in (root / "demo").rglob("*"))
    assert cli.main(["demo", "--out-dir", inside]) == 2
    assert "must be outside demo/" in capsys.readouterr().err
    assert sorted(str(path) for path in (root / "demo").rglob("*")) == before


def test_demo_refuses_an_out_dir_inside_a_named_evidence_folder(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence = _copy_evidence(tmp_path)
    assert cli.main(["demo", "--evidence", str(evidence), "--out-dir", str(evidence / "pages")]) == 2
    assert "must be outside demo/" in capsys.readouterr().err
    assert not (evidence / "pages").exists()


def test_demo_names_the_page_it_wrote_when_the_second_fails(
    offline: None, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    evidence = _copy_evidence(tmp_path, pages=1)
    out = tmp_path / "out"
    assert cli.main(["demo", "--evidence", str(evidence), "--out-dir", str(out)]) == 2
    err = capsys.readouterr().err
    assert f"already wrote {out / PAGES[0].out_name} ({PAGES[0].label})" in err
