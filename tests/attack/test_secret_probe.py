"""The hostile probe's detectors, and the hostile repository's own test, must actually fire, so the sandbox result
"found nothing" is meaningful.

Each detector test plants a secret, a credentials file, a listening loopback socket, a resolvable name or a second
network interface and shows the detector flags it; each check in the hostile test_probe.py is shown failing on a
report that carries the matching leak. Loopback and /etc/hosts only; no external network is contacted. The fixture
files are loaded by path, exactly as the sandbox runs them.
"""

from __future__ import annotations

import importlib.util
import json
import socket
from pathlib import Path
from types import ModuleType

import pytest

_FIXTURE = Path(__file__).parent / "fixtures" / "conftest_secret"


def _load(name: str) -> ModuleType:
    spec = importlib.util.spec_from_file_location(f"{name}_fixture", _FIXTURE / f"{name}.py")
    assert spec is not None and spec.loader is not None
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


probe = _load("secret_probe")
hostile_test = _load("test_probe")


def test_secret_env_is_flagged_by_name_and_value() -> None:
    env = {
        "PATH": "/usr/bin",
        "HOME": "/root",
        "LANG": "C",
        "MY_API_KEY": "redacted",
        "BORING_NAME": "ghp_" + "a" * 36,
    }
    found = probe.find_secret_env(env)
    assert "MY_API_KEY" in found  # flagged by name
    assert "BORING_NAME" in found  # flagged by value shape
    assert "PATH" not in found and "HOME" not in found and "LANG" not in found


def test_benign_env_flags_nothing() -> None:
    assert probe.find_secret_env({"PATH": "/usr/bin", "PWD": "/work", "TERM": "xterm"}) == []


def test_home_credentials_file_is_flagged(tmp_path: Path) -> None:
    (tmp_path / ".aws").mkdir()
    (tmp_path / ".aws" / "credentials").write_text("[default]\naws_access_key_id = AKIAEXAMPLE\n")
    found = probe.find_home_secrets(tmp_path)
    assert str(tmp_path / ".aws" / "credentials") in found


def test_empty_home_flags_nothing(tmp_path: Path) -> None:
    assert probe.find_home_secrets(tmp_path) == []


def test_connect_detects_a_reachable_loopback_listener() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]
        results = probe.try_connect([("127.0.0.1", port)], timeout=2.0)
    assert results[0]["connected"] is True  # the detector reports a real connection


def test_connect_reports_a_closed_port_as_unreachable() -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as srv:
        srv.bind(("127.0.0.1", 0))
        port = srv.getsockname()[1]  # bound but never listening, then released
    results = probe.try_connect([("127.0.0.1", port)], timeout=2.0)
    assert results[0]["connected"] is False
    assert results[0]["error"]


def test_resolve_detects_a_name_the_host_can_resolve() -> None:
    # localhost comes from the hosts file; no resolver on the network is asked.
    assert probe.try_resolve(["localhost"])[0]["resolved"] is True


def test_interfaces_lists_what_the_kernel_shows(tmp_path: Path) -> None:
    (tmp_path / "lo").mkdir()
    (tmp_path / "eth0").mkdir()
    assert probe.interfaces(tmp_path) == ["eth0", "lo"]
    assert probe.interfaces(tmp_path / "missing") == []


def test_targets_and_names_come_from_the_environment() -> None:
    env = {"PROBE_TARGETS": "10.0.0.7:8080, bad, host.example:443", "PROBE_DNS": "listener, ,other"}
    assert probe.targets_from_env(env) == [("10.0.0.7", 8080), ("host.example", 443), *probe.DEFAULT_TARGETS]
    assert probe.dns_names_from_env(env) == ["listener", "other"]
    assert probe.targets_from_env({}) == list(probe.DEFAULT_TARGETS)


def test_a_config_file_overlays_the_probes_settings_and_nothing_else(tmp_path: Path) -> None:
    config = tmp_path / "probe_config.json"
    config.write_text(json.dumps({"PROBE_TARGETS": "10.0.0.9:80", "PATH": "/evil", "PROBE_DNS": "listener"}))
    merged = probe.settings({"PATH": "/usr/bin", "PROBE_DNS": "old"}, config)
    assert merged == {"PATH": "/usr/bin", "PROBE_TARGETS": "10.0.0.9:80", "PROBE_DNS": "listener"}
    assert probe.settings({"A": "1"}, tmp_path / "missing.json") == {"A": "1"}


def test_every_home_is_scanned(tmp_path: Path) -> None:
    run_home, named = tmp_path / "run-home", tmp_path / "host-home"
    for home in (run_home, named):
        (home / ".ssh").mkdir(parents=True)
    (named / ".ssh" / "id_ed25519").write_text("not a real key\n")
    homes = probe.homes_to_scan({"HOME": str(run_home), "PROBE_HOMES": f"{named}, "})
    assert homes[0] == run_home and named in homes and Path("/root") in homes
    # Only the homes this test made: a Linux host's real /home and /root hold secrets of their own.
    mine = [home for home in homes if home.is_relative_to(tmp_path)]
    assert probe.find_home_secrets(*mine) == [str(named / ".ssh" / "id_ed25519")]


def test_build_report_shape(tmp_path: Path) -> None:
    report = probe.build_report({"PATH": "/usr/bin"}, tmp_path, [("127.0.0.1", 9)], ran_from="conftest")
    assert set(report) == {
        "ran_from",
        "env_names",
        "secret_env",
        "home",
        "home_secrets",
        "connections",
        "dns",
        "interfaces",
    }
    assert report["home"] == str(tmp_path)
    assert report["ran_from"] == "conftest"
    assert report["secret_env"] == [] and report["home_secrets"] == [] and report["dns"] == []
    assert report["env_names"] == ["PATH"]


# -- the hostile repository's own test has teeth ---------------------------------------------

_CLEAN: dict[str, object] = {
    "ran_from": "conftest",
    "home": "/",
    "env_names": ["PATH"],
    "secret_env": ["GPG_KEY"],  # a public value baked into the python image; not a host secret family
    "home_secrets": [],
    "connections": [{"target": "10.0.0.7:8080", "connected": False, "error": "OSError"}],
    "dns": [{"name": "listener", "resolved": False, "error": "gaierror"}],
    "interfaces": ["lo"],
}


def _check(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, **changes: object) -> None:
    path = tmp_path / "probe_report.json"
    path.write_text(json.dumps({**_CLEAN, **changes}))
    monkeypatch.setenv("PROBE_REPORT", str(path))
    hostile_test.test_sandbox_leaked_nothing()


def test_the_hostile_test_passes_on_a_clean_report(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _check(tmp_path, monkeypatch)


@pytest.mark.parametrize(
    ("changes", "what"),
    [
        ({"ran_from": "script"}, "did not run at collection"),
        ({"home_secrets": ["/root/.aws/credentials"]}, "home credential"),
        ({"home": "/nonexistent"}, "does not exist"),
        ({"connections": [{"target": "10.0.0.7:8080", "connected": True, "error": ""}]}, "network reachable"),
        ({"dns": [{"name": "listener", "resolved": True, "error": ""}]}, "names resolved"),
        ({"interfaces": ["eth0", "lo"]}, "network interfaces"),
        ({"secret_env": ["FALDEN_SECRET_SENTINEL"]}, "host secret family"),
    ],
)
def test_the_hostile_test_fails_on_each_leak(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, changes: dict[str, object], what: str
) -> None:
    with pytest.raises(AssertionError, match=what):
        _check(tmp_path, monkeypatch, **changes)
