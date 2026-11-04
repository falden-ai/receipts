"""Guards for every core test: no real Meta key is visible, no request can reach the network, and the default spend
file is a fresh one in the test's own temporary folder, never one in the worktree."""

from __future__ import annotations

from pathlib import Path

import httpx
import pytest


@pytest.fixture(autouse=True)
def _no_real_key_network_or_spend_file(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Removes the key variable for the test's duration, makes httpx's real transport fail any request, so a test
    that forgot its fake transport fails loudly instead of calling the API, and points RECEIPTS_SPEND_FILE at a file
    that does not exist yet, so a default client starts unarmed."""
    monkeypatch.delenv("FALDEN_META_API_KEY", raising=False)
    monkeypatch.setenv("RECEIPTS_SPEND_FILE", str(tmp_path / "default-spend" / "spend.jsonl"))

    def refuse(self: httpx.HTTPTransport, request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"a core test tried to reach the network: {request.method} {request.url}")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse)
