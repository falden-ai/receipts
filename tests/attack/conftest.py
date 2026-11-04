"""Attack-lane test configuration: register the ``docker`` marker, keep the hostile fixtures out of collection, and
skip Docker-backed tests cleanly unless they are opted in on a reachable daemon.

The fixtures tree holds a hostile ``conftest.py`` and a ``test_probe.py`` meant to run inside the sandbox, not here,
so collection never recurses into it. Docker tests run only with ``RECEIPTS_RUN_DOCKER=1`` and a live daemon; the
founder's heavy lock should be held around them, which the lane does in the shell, not in-process.
"""

from __future__ import annotations

import os

import pytest

from receipts.attack.sandbox import docker_available

collect_ignore = ["fixtures"]
collect_ignore_glob = ["fixtures/*"]


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line("markers", "docker: needs a reachable docker daemon; run under the shared heavy lock")


def pytest_runtest_setup(item: pytest.Item) -> None:
    if "docker" not in item.keywords:
        return
    if os.environ.get("RECEIPTS_RUN_DOCKER") != "1":
        pytest.skip("docker tests are opt-in: set RECEIPTS_RUN_DOCKER=1 and hold the heavy lock")
    if not docker_available():
        pytest.skip("docker daemon is not reachable")
