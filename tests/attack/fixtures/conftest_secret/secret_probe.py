"""A hostile repository's probe. Stdlib only, so it runs inside the sandbox with nothing but the repo present.

It stands in for an agent-authored (or injected) repository that tries, at import time, to read anything that looks
like a key or token from the environment, to read credential files under the home directory, to resolve a name and
to open a network connection. Inside the sandbox all of it must come up empty: no secret is visible, no name
resolves, no connection succeeds and the only network interface is loopback.

The targets come from a ``probe_config.json`` the harness writes next to this file, or from the run's environment
(``PROBE_TARGETS`` as ``host:port`` pairs, ``PROBE_DNS`` as names and ``PROBE_HOMES`` as extra home directories,
comma-separated), so a test can aim the probe at a listener it has shown to be reachable from a networked container.
Credential files are looked for under the run's HOME, root's home, every /home/<user> and the named homes. The same
functions are imported by the attack lane's unit tests, where a planted secret, a planted credentials file and a
listening loopback socket make each detector fire. The report never contains a secret value, only the names and
paths that were reachable, so the report itself cannot leak one.
"""

from __future__ import annotations

import contextlib
import json
import os
import re
import socket
from collections.abc import Mapping, Sequence
from pathlib import Path

# Environment variable NAMES that look like a credential.
_SECRET_NAME = re.compile(
    r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|CREDENTIAL|APIKEY|PRIVATE|SESSION|AUTH|"
    r"AWS_|FALDEN_|ANTHROPIC|OPENAI|GITHUB_TOKEN|GH_TOKEN)",
    re.IGNORECASE,
)
# Values that look like a credential regardless of the variable name.
_SECRET_VALUE = re.compile(
    r"(sk-[A-Za-z0-9]{8,}|ghp_[A-Za-z0-9]{20,}|gho_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{20,}|"
    r"AKIA[0-9A-Z]{12,}|xox[baprs]-[A-Za-z0-9-]{10,}|-----BEGIN [A-Z ]*PRIVATE KEY-----)"
)
# Paths under the home directory that usually hold credentials.
_HOME_CANDIDATES = (
    ".aws/credentials",
    ".aws/config",
    ".config/gcloud/application_default_credentials.json",
    ".ssh/id_rsa",
    ".ssh/id_ed25519",
    ".netrc",
    ".env",
    ".git-credentials",
    ".npmrc",
    ".pypirc",
    ".docker/config.json",
)
DEFAULT_TARGETS: tuple[tuple[str, int], ...] = (("127.0.0.1", 9), ("192.0.2.1", 443))


def find_secret_env(environ: Mapping[str, str]) -> list[str]:
    """Names (never values) of environment variables that look like a credential."""
    found: list[str] = []
    for name, value in environ.items():
        if _SECRET_NAME.search(name) or _SECRET_VALUE.search(value or ""):
            found.append(name)
    return sorted(found)


def find_home_secrets(home: Path, *others: Path) -> list[str]:
    """Credential files under ``home`` (and each further directory) that exist and can be read (the leak signal), by
    path, contents never read."""
    found: list[str] = []
    for base in (home, *others):
        for rel in _HOME_CANDIDATES:
            path = base / rel
            try:
                readable = path.is_file() and os.access(path, os.R_OK)
            except OSError:
                readable = False
            if readable and str(path) not in found:
                found.append(str(path))
    return found


def homes_to_scan(environ: Mapping[str, str]) -> list[Path]:
    """Every home a credential could sit under: the run's HOME, root's, each /home/<user>, and any the harness names
    in ``PROBE_HOMES`` (for instance the host's own home path, which must not exist inside)."""
    homes = [Path(environ.get("HOME", "/root")), Path("/root")]
    with contextlib.suppress(OSError):
        homes += sorted(p for p in Path("/home").iterdir() if p.is_dir())
    homes += [Path(p) for p in environ.get("PROBE_HOMES", "").split(",") if p.strip()]
    return list(dict.fromkeys(homes))


def targets_from_env(environ: Mapping[str, str]) -> list[tuple[str, int]]:
    """``PROBE_TARGETS`` (``host:port,host:port``) plus the fixed defaults: a closed loopback port and TEST-NET-1."""
    targets: list[tuple[str, int]] = []
    for item in environ.get("PROBE_TARGETS", "").split(","):
        host, sep, port = item.strip().rpartition(":")
        if sep and host and port.isdigit():
            targets.append((host, int(port)))
    return [*targets, *DEFAULT_TARGETS]


def dns_names_from_env(environ: Mapping[str, str]) -> list[str]:
    return [name.strip() for name in environ.get("PROBE_DNS", "").split(",") if name.strip()]


def try_connect(targets: Sequence[tuple[str, int]], *, timeout: float = 2.0) -> list[dict[str, object]]:
    """Try to open each TCP target. Reports whether the connection succeeded, and the error if not."""
    results: list[dict[str, object]] = []
    for host, port in targets:
        entry: dict[str, object] = {"target": f"{host}:{port}", "connected": False, "error": ""}
        try:
            with socket.create_connection((host, port), timeout=timeout):
                entry["connected"] = True
        except OSError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
    return results


def try_resolve(names: Sequence[str]) -> list[dict[str, object]]:
    """Try to resolve each name. A sandbox with no network has no resolver to ask."""
    results: list[dict[str, object]] = []
    for name in names:
        entry: dict[str, object] = {"name": name, "resolved": False, "error": ""}
        try:
            socket.getaddrinfo(name, None)
            entry["resolved"] = True
        except OSError as exc:
            entry["error"] = f"{type(exc).__name__}: {exc}"
        results.append(entry)
    return results


def interfaces(sys_net: Path = Path("/sys/class/net")) -> list[str]:
    return sorted(os.listdir(sys_net)) if sys_net.is_dir() else []


def build_report(
    environ: Mapping[str, str],
    home: Path,
    targets: Sequence[tuple[str, int]],
    dns_names: Sequence[str] = (),
    *,
    ran_from: str = "script",
    extra_homes: Sequence[Path] = (),
) -> dict[str, object]:
    return {
        "ran_from": ran_from,
        "env_names": sorted(environ),  # names only, so the harness can confirm no host variable crossed in
        "secret_env": find_secret_env(environ),
        "home": str(home),
        "home_secrets": find_home_secrets(home, *extra_homes),
        "connections": try_connect(targets),
        "dns": try_resolve(dns_names),
        "interfaces": interfaces(),
    }


def settings(environ: Mapping[str, str], config: Path) -> dict[str, str]:
    """The probe's own settings (PROBE_TARGETS, PROBE_DNS, PROBE_HOMES): the environment's, overlaid by a
    ``probe_config.json`` the harness wrote next to this file, if there is one."""
    merged = dict(environ)
    if config.is_file():
        data = json.loads(config.read_text())
        merged.update({str(k): str(v) for k, v in data.items() if str(k).startswith("PROBE_")})
    return merged


def main(ran_from: str = "script") -> int:
    environ = dict(os.environ)
    probe = settings(environ, Path(__file__).parent / "probe_config.json")
    homes = homes_to_scan(probe)
    report = build_report(
        environ,
        homes[0],
        targets_from_env(probe),
        dns_names_from_env(probe),
        ran_from=ran_from,
        extra_homes=homes[1:],
    )
    out_path = environ.get("PROBE_REPORT") or os.path.join(os.getcwd(), "probe_report.json")
    Path(out_path).write_text(json.dumps(report))
    print("PROBE_REPORT_JSON " + json.dumps(report))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
