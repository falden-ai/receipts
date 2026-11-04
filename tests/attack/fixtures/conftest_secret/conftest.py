"""The hostile repository's conftest. pytest imports it at collection, before any test runs, and it runs the probe
that tries to read secrets, resolve names and reach the network.

A real attacker's conftest would run unconditionally. This one runs only when a ``probe_config.json`` sits next to it
(or ``RECEIPTS_PROBE_RUN=1`` is in its environment). The fixture directory ships no such file: only the attack lane's
Docker tests write one, as a run file when they run pytest inside the sandbox through ``run_test`` (and into the
networked control container that proves the probe can see a network), so no other test run can execute this hostile
code by accident. The lane conftest also keeps pytest from collecting this directory.
"""

from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(__file__)

if os.environ.get("RECEIPTS_PROBE_RUN") == "1" or os.path.exists(os.path.join(_HERE, "probe_config.json")):
    sys.path.insert(0, _HERE)
    import secret_probe

    secret_probe.main(ran_from="conftest")
