"""The hostile repository's own test: it passes only when the conftest ran at collection and its probe found no
secret, resolved no name, reached nothing and saw no interface but loopback.

The attack lane's Docker test runs this file with pytest inside the sandbox, where it must pass, and inside a
networked control container, where it must fail on its assertions. The lane's own suite never collects it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

_HOST_SECRET_FAMILIES = ("FALDEN", "AWS_", "ANTHROPIC", "OPENAI", "GITHUB_TOKEN", "GH_TOKEN")


def test_sandbox_leaked_nothing() -> None:
    report_path = os.environ.get("PROBE_REPORT") or os.path.join(os.getcwd(), "probe_report.json")
    report = json.loads(Path(report_path).read_text())
    assert report["ran_from"] == "conftest", "the hostile conftest did not run at collection"
    # The home scan means something only if the run's HOME exists (nobody's default, /nonexistent, does not).
    assert Path(report["home"]).is_dir(), f"the run's HOME {report['home']} does not exist; the home scan read nothing"
    assert report["home_secrets"] == [], f"home credential files readable: {report['home_secrets']}"
    connected = [c["target"] for c in report["connections"] if c["connected"]]
    assert connected == [], f"network reachable from the sandbox: {connected}"
    resolved = [d["name"] for d in report["dns"] if d["resolved"]]
    assert resolved == [], f"names resolved from the sandbox: {resolved}"
    assert set(report["interfaces"]) <= {"lo"}, f"network interfaces in the sandbox: {report['interfaces']}"
    # A benign env baked into the base image (for example a public GPG_KEY) may be flagged, but no host secret family.
    leaked = [n for n in report["secret_env"] if any(fam in n for fam in _HOST_SECRET_FAMILIES)]
    assert leaked == [], f"host secret family visible in the sandbox: {leaked}"
