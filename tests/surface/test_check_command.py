"""CLI registration and exit codes, with local run files and fake GitHub responses only."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import NoReturn

import httpx
import pytest

from receipts.surface import check_command
from receipts.surface.check_command import register
from receipts.surface.checks import APP_ID, INSTALLATION_ID, KEY_FILE, AppConfig, PostedCheck, post_check
from receipts.surface.ledger import JsonlLedger
from tests.surface.test_checks import REPO, SHA, TOKEN, make_entry, make_target, token_answer
from tests.surface.test_checks import rsa_key as rsa_key


def arguments(tmp_path: Path, *extra: str, patch: str | None = None) -> argparse.Namespace:
    ledger_path = tmp_path / "ledger.jsonl"
    ledger = JsonlLedger(ledger_path)
    ledger.append(make_entry())
    target_path = tmp_path / "run.jsonl"
    target = make_target() if patch is None else make_target(patch=patch)
    target_path.write_text(json.dumps({"target": target.model_dump(mode="json"), "temperatures": "spec"}) + "\n")
    parser = argparse.ArgumentParser(prog="receipts")
    register(parser.add_subparsers(dest="command", required=True))
    return parser.parse_args(
        [
            "check",
            "post",
            "--ledger",
            str(ledger_path),
            "--targets",
            str(target_path),
            "--target-id",
            "demo-pr-1",
            "--repo",
            REPO,
            "--head-sha",
            SHA,
            *extra,
        ]
    )


def call(args: argparse.Namespace) -> int:
    result: int = args.func(args)
    return result


class NoEnvironment(Mapping[str, str]):
    def __getitem__(self, key: str) -> NoReturn:
        raise AssertionError("dry run read an environment variable")

    def __iter__(self) -> NoReturn:
        raise AssertionError("dry run enumerated the environment")

    def __len__(self) -> NoReturn:
        raise AssertionError("dry run measured the environment")


def no_post(*args: object, **kwargs: object) -> NoReturn:
    raise AssertionError("unexpected call to poster")


def test_dry_run_never_reads_environment_or_calls_poster(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = arguments(tmp_path)
    before = args.ledger.read_bytes()
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", NoEnvironment())
        patch.setattr(check_command, "post_check", no_post)
        assert call(args) == 0
    captured = capsys.readouterr()
    payload = json.loads(captured.out)
    assert payload["name"] == "Falden Receipts"
    assert payload["conclusion"] == "failure"
    assert payload["head_sha"] == SHA
    assert captured.err == ""
    assert args.ledger.read_bytes() == before
    assert not args.ledger.with_suffix(".jsonl.torn").exists()


@pytest.mark.parametrize("missing", [APP_ID, INSTALLATION_ID, KEY_FILE])
def test_missing_exact_variable_exits_two_without_posting(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], missing: str
) -> None:
    args = arguments(tmp_path, "--post")
    values = {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: str(tmp_path / "key.pem")}
    values.pop(missing)
    monkeypatch.setattr(os, "environ", values)
    monkeypatch.setattr(check_command, "post_check", no_post)
    assert call(args) == 2
    captured = capsys.readouterr()
    assert missing in captured.err
    assert captured.out == ""


@pytest.mark.parametrize("name", [APP_ID, INSTALLATION_ID])
def test_invalid_app_id_does_not_echo_its_value(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], name: str
) -> None:
    args = arguments(tmp_path, "--post")
    values = {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: "unused"}
    values[name] = "TEST-ONLY-INVALID-ID-SENTINEL"
    monkeypatch.setattr(os, "environ", values)
    monkeypatch.setattr(check_command, "post_check", no_post)
    assert call(args) == 2
    captured = capsys.readouterr()
    assert name in captured.err
    assert values[name] not in captured.err + captured.out


@pytest.mark.parametrize("json_output", [False, True])
def test_post_prints_id_and_url_without_credentials(
    tmp_path: Path,
    rsa_key: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    json_output: bool,
) -> None:
    args = arguments(tmp_path, "--post", *(["--json"] if json_output else []))
    monkeypatch.setattr(os, "environ", {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: str(rsa_key)})
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json=token_answer())
        assert request.method == "POST"
        assert request.url.path == f"/repos/{REPO}/check-runs"
        return httpx.Response(201, json={"id": 42, "html_url": f"https://github.com/{REPO}/runs/42"})

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:

        def fake_post(config: AppConfig, repo: str, payload: dict[str, object]) -> PostedCheck:
            return post_check(config, repo, payload, client=client)

        monkeypatch.setattr(check_command, "post_check", fake_post)
        assert call(args) == 0
    captured = capsys.readouterr()
    assert "42" in captured.out
    assert f"https://github.com/{REPO}/runs/42" in captured.out
    if json_output:
        assert json.loads(captured.out)["id"] == 42
    assert captured.err == ""
    assert TOKEN not in captured.out
    assert rsa_key.read_text() not in captured.out
    assert calls[0].headers["Authorization"] not in captured.out
    assert [request.method for request in calls] == ["POST", "POST"]


@pytest.mark.parametrize(
    "patch",
    [
        "--- a/package/prices.py\n+++ b/package/prices.py\n@@ -1 +1,0 @@\n-old()\n",
        "--- a/tests/test_prices.py\n+++ b/tests/test_prices.py\n@@ -1 +1 @@\n-old()\n+new()\n",
    ],
)
def test_post_does_not_refuse_a_blocker_without_an_annotatable_line(
    tmp_path: Path, rsa_key: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str], patch: str
) -> None:
    args = arguments(tmp_path, "--post", "--json", patch=patch)
    monkeypatch.setattr(os, "environ", {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: str(rsa_key)})
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json=token_answer())
        assert request.method == "POST"
        assert request.url.path == f"/repos/{REPO}/check-runs"
        payload = json.loads(request.content)
        assert payload["conclusion"] == "failure"
        assert payload["output"]["annotations"] == []
        assert "Blocker `demo-pr-1/fixed_x/a1` has no line annotation" in payload["output"]["summary"]
        assert "no added non-test line that can be annotated safely" in payload["output"]["summary"]
        return httpx.Response(201, json={"id": 42, "html_url": f"https://github.com/{REPO}/runs/42"})

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:

        def fake_post(config: AppConfig, repo: str, payload: dict[str, object]) -> PostedCheck:
            return post_check(config, repo, payload, client=client)

        monkeypatch.setattr(check_command, "post_check", fake_post)
        assert call(args) == 0
    captured = capsys.readouterr()
    assert captured.err == ""
    assert json.loads(captured.out) == {"id": 42, "html_url": f"https://github.com/{REPO}/runs/42"}
    assert [request.method for request in calls] == ["POST", "POST"]


def test_repo_missing_from_token_exits_three_without_check_call(
    tmp_path: Path, rsa_key: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = arguments(tmp_path, "--post")
    monkeypatch.setattr(os, "environ", {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: str(rsa_key)})
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(201, json=token_answer(repo="third-party/receipts-demo"))

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:

        def fake_post(config: AppConfig, repo: str, payload: dict[str, object]) -> PostedCheck:
            return post_check(config, repo, payload, client=client)

        monkeypatch.setattr(check_command, "post_check", fake_post)
        assert call(args) == 3
    captured = capsys.readouterr()
    assert "does not include --repo" in captured.err
    assert captured.out == ""
    assert len(calls) == 1


def test_mode_644_is_refused_before_a_github_request(
    tmp_path: Path, rsa_key: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = arguments(tmp_path, "--post")
    rsa_key.chmod(0o644)
    monkeypatch.setattr(os, "environ", {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: str(rsa_key)})
    with httpx.Client(
        transport=httpx.MockTransport(lambda r: pytest.fail("GitHub was called")), trust_env=False
    ) as client:

        def fake_post(config: AppConfig, repo: str, payload: dict[str, object]) -> PostedCheck:
            return post_check(config, repo, payload, client=client)

        monkeypatch.setattr(check_command, "post_check", fake_post)
        assert call(args) == 2
    captured = capsys.readouterr()
    assert "mode 600 or stricter" in captured.err
    assert rsa_key.read_text() not in captured.err + captured.out


def test_github_error_exits_one_and_never_prints_echoed_credentials(
    tmp_path: Path, rsa_key: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    args = arguments(tmp_path, "--post")
    monkeypatch.setattr(os, "environ", {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: str(rsa_key)})
    pem = rsa_key.read_text()

    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": pem + TOKEN + request.headers["Authorization"]})

    with httpx.Client(transport=httpx.MockTransport(respond), trust_env=False) as client:

        def fake_post(config: AppConfig, repo: str, payload: dict[str, object]) -> PostedCheck:
            return post_check(config, repo, payload, client=client)

        monkeypatch.setattr(check_command, "post_check", fake_post)
        assert call(args) == 1
    captured = capsys.readouterr()
    assert captured.err == "check post: GitHub check posting failed\n"
    assert captured.out == ""
    assert pem not in captured.err
    assert TOKEN not in captured.err


@pytest.mark.parametrize(
    ("field", "value"),
    [("repo", "third-party/receipts-demo"), ("head_sha", "c" * 40), ("head_sha", "bad"), ("target_id", "absent")],
)
def test_target_selection_and_commit_binding_fail_before_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, field: str, value: str
) -> None:
    args = arguments(tmp_path, "--post")
    setattr(args, field, value)
    with monkeypatch.context() as patch:
        patch.setattr(os, "environ", NoEnvironment())
        patch.setattr(check_command, "post_check", no_post)
        assert call(args) == 1


def test_duplicate_target_ids_are_refused(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    args.targets.write_text(args.targets.read_text() * 2)
    assert call(args) == 1


def test_invalid_run_item_does_not_echo_model_text(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    args = arguments(tmp_path)
    args.targets.write_text('{"target": "TEST-ONLY-PRIVATE-SENTINEL"}\n')
    assert call(args) == 1
    captured = capsys.readouterr()
    assert "line 1" in captured.err
    assert "TEST-ONLY-PRIVATE-SENTINEL" not in captured.err + captured.out


def test_missing_ledger_is_not_created(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    args.ledger.unlink()
    assert call(args) == 1
    assert not args.ledger.exists()


def test_torn_ledger_is_not_modified_or_silently_posted(tmp_path: Path) -> None:
    args = arguments(tmp_path)
    content = args.ledger.read_bytes() + b'{"v":2'
    args.ledger.write_bytes(content)
    assert call(args) == 1
    assert args.ledger.read_bytes() == content
