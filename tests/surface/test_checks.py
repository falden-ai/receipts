"""SU1 checks against synthetic evidence, MockTransport, and throwaway RSA keys. No network."""

from __future__ import annotations

import base64
import json
import os
import subprocess
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx
import pytest

from receipts.contracts import (
    CatchKind,
    FailureType,
    Finding,
    LedgerEntry,
    Lens,
    Outcome,
    ReplayKey,
    Repro,
    RunResult,
    Side,
    Status,
    Target,
    Tier,
    UncheckedReason,
)
from receipts.surface.checks import (
    APP_ID,
    CHECK_NAME,
    INSTALLATION_ID,
    KEY_FILE,
    AppConfig,
    AppConfigError,
    CheckInputError,
    RepositoryRefusedError,
    app_jwt,
    check_payload,
    installation_token,
    post_check,
)
from receipts.surface.ledger import InvalidEntryError
from receipts.surface.tickets import TrackerError
from receipts.surface.trackers import GitHubHTTP

SHA = "b" * 40
REPO = "falden-demo/receipts-demo"
PATCH = """diff --git a/package/prices.py b/package/prices.py
--- a/package/prices.py
+++ b/package/prices.py
@@ -1,3 +1,3 @@
 def price(quantity):
-    return quantity * 100
+    return quantity * 99
 # end
diff --git a/tests/test_prices.py b/tests/test_prices.py
--- a/tests/test_prices.py
+++ b/tests/test_prices.py
@@ -1 +1 @@
-assert price(1) == 100
+assert price(1) == 99
"""
HOSTILE = (
    "![image](https://evil.example/p) [link](http://evil.example/x) javascript:alert(1) "
    "www.evil.example bare.example \N{RIGHT-TO-LEFT OVERRIDE}\N{ZERO WIDTH SPACE} "
    "\n```\n````\n<img src=//evil.example/i>"
)
TOKEN = "test-only-installation-token-sentinel"


def make_target(*, patch: str = PATCH, target_id: str = "demo-pr-1") -> Target:
    return Target(
        id=target_id,
        tier=Tier.LIVE,
        repo=REPO,
        base_sha="a" * 40,
        patch_sha=SHA,
        patch=patch,
        image="falden-demo/receipts-demo@sha256:" + "c" * 64,
    )


def make_entry(
    status: Status = Status.BLOCKER,
    *,
    suffix: str = "a1",
    stdout: str = '  File "/workspace/package/prices.py", line 2, in price\n    raise AssertionError',
    stderr: str = "",
    hostile: str = "",
    target_id: str = "demo-pr-1",
    kind: CatchKind = CatchKind.REGRESSION,
) -> LedgerEntry:
    fail = RunResult(
        side=Side.PATCH,
        outcome=Outcome.FAIL,
        failure_type=FailureType.ASSERTION,
        fresh_container=True,
        duration_s=0.1,
        stdout_tail=stdout + hostile,
        stderr_tail=stderr + hostile,
    )
    passed = RunResult(side=Side.BASE, outcome=Outcome.PASS, fresh_container=True, duration_s=0.1)
    if kind is not CatchKind.REGRESSION:
        passed = passed.model_copy(update={"side": Side.SURROGATE})
    finding = Finding(
        id=f"{target_id}/fixed_x/{suffix}",
        target_id=target_id,
        claim_id="fixed_x",
        lens=Lens.REGRESSION,
        status=status,
        kind=kind,
        unchecked_reason=UncheckedReason.BUDGET if status is Status.UNCHECKED else None,
        quote="The discount applies at five items. " + hostile,
        detail="A quantity of five must receive the discount. " + hostile,
        repro=Repro(
            id="r1",
            attack_id="a1",
            claim_id="fixed_x",
            lens=Lens.REGRESSION,
            kind=kind,
            files={"tests/test_repro.py": "def test_price():\n    assert price(5) == 450\n# " + hostile},
            command=("python", "-m", "pytest", "tests/test_repro.py"),
        ),
        runs=(fail,) * 5 + (passed,) * 5,
        replay=ReplayKey(
            base_sha="a" * 40,
            patch_sha=SHA,
            harness_hash="test-harness",
            seed=7,
            image_digest="sha256:" + "c" * 64,
        ),
    )
    return LedgerEntry(at="2026-10-30T12:00:00Z", receipts_version="0.0.1", finding=finding)


def output(payload: dict[str, object]) -> dict[str, Any]:
    value = payload["output"]
    assert isinstance(value, dict)
    return value


@pytest.fixture
def rsa_key(tmp_path: Path) -> Path:
    """Generated for this test only. Never a real App key."""
    key = tmp_path / "throwaway-test-only.pem"
    result = subprocess.run(
        ["openssl", "genpkey", "-algorithm", "RSA", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(key)],
        capture_output=True,
        timeout=30,
        check=False,
    )
    assert result.returncode == 0
    key.chmod(0o600)
    return key


def config(key: Path) -> AppConfig:
    return AppConfig(app_id="123", installation_id="456", key_file=key)


def token_answer(*, repo: str = REPO, seconds: int = 3_500, token: str = TOKEN) -> dict[str, object]:
    return {
        "token": token,
        "expires_at": (datetime.now(UTC) + timedelta(seconds=seconds)).isoformat(),
        "repositories": [{"full_name": repo}],
    }


def existing_run(run_id: int = 42, *, app_id: int = 123, sha: str = SHA) -> dict[str, object]:
    return {"id": run_id, "name": CHECK_NAME, "head_sha": sha, "app": {"id": app_id}}


@pytest.mark.parametrize(
    ("statuses", "conclusion"),
    [
        ((Status.BLOCKER,), "failure"),
        ((Status.BLOCKER, Status.SUSPECTED), "failure"),
        ((Status.SUSPECTED,), "neutral"),
        ((Status.SUSPECTED, Status.UNCHECKED), "neutral"),
        ((Status.FLAKY,), "neutral"),
        ((Status.POSSIBLY_TOXIC,), "neutral"),
        ((Status.S_SAT_INCONCLUSIVE,), "neutral"),
        ((Status.FLAKY, Status.POSSIBLY_TOXIC, Status.S_SAT_INCONCLUSIVE), "neutral"),
        ((Status.REFUTED, Status.FLAKY), "neutral"),
        ((Status.REFUTED, Status.POSSIBLY_TOXIC), "neutral"),
        ((Status.NOT_REPRODUCED, Status.S_SAT_INCONCLUSIVE), "neutral"),
        ((Status.BLOCKER, Status.FLAKY, Status.POSSIBLY_TOXIC, Status.S_SAT_INCONCLUSIVE), "failure"),
        ((Status.REFUTED, Status.NOT_REPRODUCED), "success"),
        ((Status.UNCHECKED,), "neutral"),
        ((), "neutral"),
    ],
)
def test_conclusions(statuses: tuple[Status, ...], conclusion: str) -> None:
    payload = check_payload([make_entry(s, suffix=str(i)) for i, s in enumerate(statuses)], make_target(), head_sha=SHA)
    assert payload["conclusion"] == conclusion
    assert payload["external_id"] == "demo-pr-1"
    assert "actions" not in payload
    assert "details_url" not in payload
    if not statuses or set(statuses) == {Status.UNCHECKED}:
        assert "nothing was checked" in output(payload)["title"].lower()


@pytest.mark.parametrize("status", [Status.FLAKY, Status.POSSIBLY_TOXIC, Status.S_SAT_INCONCLUSIVE])
def test_inconclusive_result_is_named_in_title(status: Status) -> None:
    payload = check_payload([make_entry(status)], make_target(), head_sha=SHA)
    assert "1 " + status.value.replace("_", " ") in output(payload)["title"]


def test_latest_state_wins_in_append_order_and_other_targets_are_ignored() -> None:
    old = make_entry().model_copy(update={"at": "2026-10-31T12:00:00Z"})
    new = make_entry(Status.REFUTED)
    payload = check_payload([old, new, make_entry(target_id="other")], make_target(), head_sha=SHA)
    assert payload["conclusion"] == "success"
    assert output(payload)["annotations"] == []
    assert "refuted: 1" in output(payload)["summary"]


def test_blocker_summary_contains_evidence_and_unchecked_reasons() -> None:
    payload = check_payload([make_entry(), make_entry(Status.UNCHECKED, suffix="u")], make_target(), head_sha=SHA)
    assert output(payload)["title"] == "1 blocker: regression catch, 5 of 5 runs"
    summary = output(payload)["summary"]
    for expected in (
        "regression catch",
        "`demo-pr-1/fixed_x/a1`",
        "The discount applies",
        "tests/test_repro.py",
        "assert price(5) == 450",
        "| patch | 5 | fail | assertion |",
        "| base | 10 | pass | none |",
        '"harness_hash":"test-harness"',
        '"seed":7',
        "Unchecked lens: regression; reason: budget",
    ):
        assert expected in summary


def test_unvalidated_blocker_is_refused() -> None:
    bad = make_entry()
    bad = bad.model_copy(update={"finding": bad.finding.model_copy(update={"runs": ()})})
    with pytest.raises(InvalidEntryError):
        check_payload([bad], make_target(), head_sha=SHA)


@pytest.mark.parametrize("field", ["base_sha", "patch_sha"])
def test_replay_for_another_revision_is_refused(field: str) -> None:
    entry = make_entry()
    assert entry.finding.replay is not None
    replay = entry.finding.replay.model_copy(update={field: "d" * 40})
    entry = entry.model_copy(update={"finding": entry.finding.model_copy(update={"replay": replay})})
    with pytest.raises(CheckInputError, match="replay key does not match"):
        check_payload([entry], make_target(), head_sha=SHA)


@pytest.mark.parametrize(
    "tail",
    [
        '  File "/workspace/package/prices.py", line 2, in price',
        '  File "/workspace/package/prices.py", line 2, in price\n'
        '  File "/workspace/tests/test_prices.py", line 1, in test_price',
        '  File "/workspace/package/prices.py", line 2, in price\n'
        '  File "/workspace/package/prices.py", line 3, in unchanged',
    ],
)
def test_annotation_uses_added_non_test_frame_and_skips_deeper_ineligible_frames(tail: str) -> None:
    payload = check_payload([make_entry(stdout=tail)], make_target(), head_sha=SHA)
    [note] = output(payload)["annotations"]
    assert note["path"] == "package/prices.py"
    assert note["start_line"] == note["end_line"] == 2
    assert note["annotation_level"] == "failure"
    assert "not located" not in note["message"]


def test_deepest_eligible_frame_in_either_output_is_used() -> None:
    patch = (
        PATCH
        + """diff --git a/package/other.py b/package/other.py
--- a/package/other.py
+++ b/package/other.py
@@ -9 +9 @@
-old()
+new()
"""
    )
    entry = make_entry(stderr='  File "/workspace/package/other.py", line 9, in new')
    [note] = output(check_payload([entry], make_target(patch=patch), head_sha=SHA))["annotations"]
    assert (note["path"], note["start_line"]) == ("package/other.py", 9)


@pytest.mark.parametrize("name", ["tests.py", "prices_tests.py", "prices_test.py", "conftest.py"])
def test_test_modules_outside_test_directories_are_skipped(name: str) -> None:
    patch = (
        PATCH
        + f"""diff --git a/package/{name} b/package/{name}
--- a/package/{name}
+++ b/package/{name}
@@ -1 +1 @@
-old()
+new()
"""
    )
    entry = make_entry(stderr=f'  File "/workspace/package/{name}", line 1, in test_price')
    [note] = output(check_payload([entry], make_target(patch=patch), head_sha=SHA))["annotations"]
    assert note["path"] == "package/prices.py"
    assert "not located" not in note["message"]


@pytest.mark.parametrize(
    "tail",
    [
        "AssertionError: no native frame",
        '  File "/workspace/package/prices.py", line 1, in unchanged',
        '  File "/workspace/tests/test_prices.py", line 1, in test_price',
        '  File "/elsewhere/package/prices.py", line 2, in outside',
        '  File "/workspace-other/package/prices.py", line 2, in prefix',
        '  File "/workspace/../package/prices.py", line 2, in traversal',
        '  File "package/prices.py", line 2, in relative',
    ],
)
def test_annotation_falls_back_without_trusting_external_or_unchanged_frames(tail: str) -> None:
    [note] = output(check_payload([make_entry(stdout=tail)], make_target(), head_sha=SHA))["annotations"]
    assert note["path"] == "package/prices.py"
    assert note["start_line"] == 2
    assert "line not located from the traceback" in note["message"]


def test_diff_parser_counts_hunks_and_understands_quoted_paths() -> None:
    patch = """diff --git "a/package/with space.py" "b/package/with space.py"
--- "a/package/with space.py"
+++ "b/package/with space.py"
@@ -1,0 +2,3 @@
+++ b/forged.py
+@@ -1 +99 @@
+new()
"""
    entry = make_entry(stdout='  File "/workspace/package/with space.py", line 4, in new')
    [note] = output(check_payload([entry], make_target(patch=patch), head_sha=SHA))["annotations"]
    assert note["path"] == "package/with space.py"
    assert note["start_line"] == 4
    assert "not located" not in note["message"]


def test_bare_empty_context_line_counts_toward_annotation_line_number() -> None:
    patch = "--- a/package/prices.py\n+++ b/package/prices.py\n@@ -1,2 +1,2 @@\n\n-old()\n+new()\n"
    entry = make_entry(stdout='  File "/workspace/package/prices.py", line 2, in new')
    [note] = output(check_payload([entry], make_target(patch=patch), head_sha=SHA))["annotations"]
    assert note["path"] == "package/prices.py"
    assert note["start_line"] == note["end_line"] == 2
    assert "not located" not in note["message"]


@pytest.mark.parametrize(
    "patch",
    [
        "--- a/package/prices.py\n+++ b/package/prices.py\n@@ -1 +1,0 @@\n-old()\n",
        "--- a/tests/test_prices.py\n+++ b/tests/test_prices.py\n@@ -1 +1 @@\n-old()\n+new()\n",
        "--- a/../outside.py\n+++ b/../outside.py\n@@ -1 +1 @@\n-old()\n+new()\n",
        "diff --git a/package/prices.py b/package/prices.py\nold mode 100644\nnew mode 100755\n",
        (
            "diff --git a/package/data.bin b/package/data.bin\n"
            "Binary files a/package/data.bin and b/package/data.bin differ\n"
        ),
        "--- a/package/prices.py\n+++ /dev/null\n@@ -1 +0,0 @@\n-old()\n",
    ],
)
def test_no_safe_added_line_omits_annotation_without_refusing_failed_check(patch: str) -> None:
    entry = make_entry()
    payload = check_payload([entry], make_target(patch=patch), head_sha=SHA)
    assert payload["conclusion"] == "failure"
    fields = output(payload)
    assert fields["annotations"] == []
    assert f"Blocker `{entry.finding.id}` has no line annotation" in fields["summary"]
    assert "no added non-test line that can be annotated safely" in fields["summary"]


def test_all_unannotated_blockers_are_named_before_evidence_is_clipped(rendered: Callable[[str], str]) -> None:
    entries = [make_entry(suffix=str(i), hostile=HOSTILE * 400) for i in range(5)]
    target = make_target(patch="--- a/package/prices.py\n+++ b/package/prices.py\n@@ -1 +1,0 @@\n-old()\n")
    payload = check_payload(entries, target, head_sha=SHA)
    fields = output(payload)
    assert payload["conclusion"] == "failure"
    assert fields["annotations"] == []
    summary = fields["summary"]
    assert len(summary) <= 60_000
    assert "Summary clipped" in summary
    for entry in entries:
        assert f"Blocker `{entry.finding.id}` has no line annotation" in summary
    assert "http" not in rendered(summary)
    assert "bare.example" not in rendered(summary)


def test_r16_canary_is_inert_everywhere(rendered: Callable[[str], str]) -> None:
    payload = check_payload([make_entry(hostile=HOSTILE)], make_target(), head_sha=SHA)
    fields = output(payload)
    outside = rendered(fields["summary"]).lower()
    for link in ("http", "www.", "javascript:", "bare.example"):
        assert link not in outside
        assert link not in fields["title"].lower()
        assert link not in fields["annotations"][0]["message"].lower()
    serialized = json.dumps(payload, ensure_ascii=False)
    assert "\N{RIGHT-TO-LEFT OVERRIDE}" not in serialized
    assert "\N{ZERO WIDTH SPACE}" not in serialized
    assert "\\u202e" in serialized and "\\u200b" in serialized
    assert "[link removed]" in fields["summary"]


def test_summary_clips_at_safe_markdown_boundaries(rendered: Callable[[str], str]) -> None:
    entries = [make_entry(suffix=str(i), hostile=HOSTILE * 400) for i in range(5)]
    summary = output(check_payload(entries, make_target(), head_sha=SHA))["summary"]
    assert len(summary) <= 60_000
    assert "Summary clipped" in summary
    assert "bare.example" not in rendered(summary)
    assert "http" not in rendered(summary)


def decode_segment(segment: str) -> bytes:
    return base64.urlsafe_b64decode(segment + "=" * (-len(segment) % 4))


def test_jwt_has_rs256_claims_and_verifies_with_throwaway_public_key(rsa_key: Path, tmp_path: Path) -> None:
    now = int(time.time())
    jwt = app_jwt(config(rsa_key), now=now)
    header, claims, signature = jwt.split(".")
    assert json.loads(decode_segment(header)) == {"alg": "RS256", "typ": "JWT"}
    assert json.loads(decode_segment(claims)) == {"iss": "123", "iat": now - 60, "exp": now + 540}
    public = tmp_path / "test-public.pem"
    subprocess.run(
        ["openssl", "pkey", "-in", str(rsa_key), "-pubout", "-out", str(public)], capture_output=True, check=True
    )
    sig = tmp_path / "signature"
    sig.write_bytes(decode_segment(signature))
    result = subprocess.run(
        ["openssl", "dgst", "-sha256", "-verify", str(public), "-signature", str(sig)],
        input=f"{header}.{claims}".encode(),
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0
    assert result.stdout == b"Verified OK\n"


def test_each_openssl_call_reads_the_complete_key_from_a_shared_offset(
    rsa_key: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pem = rsa_key.read_bytes()
    reads: list[bytes] = []
    commands: list[str] = []

    def consume_key(
        args: list[str],
        *,
        input: bytes,
        capture_output: bool,
        pass_fds: tuple[int, ...],
        timeout: int,
        check: bool,
    ) -> subprocess.CompletedProcess[bytes]:
        [fd] = pass_fds
        assert f"/dev/fd/{fd}" in args
        # macOS /dev/fd duplicates the open descriptor and shares its offset.
        # Consume the descriptor itself to reproduce that behavior on Linux too.
        reads.append(os.read(fd, len(pem) + 1))
        commands.append(args[1])
        return subprocess.CompletedProcess(args, 0, stdout=b"test-only-signature", stderr=b"")

    monkeypatch.setattr(subprocess, "run", consume_key)
    assert len(app_jwt(config(rsa_key)).split(".")) == 3
    assert commands == ["dgst", "rsa"]
    assert [read == pem for read in reads] == [True, True]


@pytest.mark.parametrize("mode", [0o600, 0o400])
def test_key_mode_owner_only_is_accepted(rsa_key: Path, mode: int) -> None:
    rsa_key.chmod(mode)
    assert len(app_jwt(config(rsa_key)).split(".")) == 3


def test_key_mode_644_and_symlinks_are_refused(rsa_key: Path, tmp_path: Path) -> None:
    rsa_key.chmod(0o644)
    with pytest.raises(AppConfigError, match="mode 600 or stricter"):
        app_jwt(config(rsa_key))
    rsa_key.chmod(0o600)
    link = tmp_path / "link.pem"
    link.symlink_to(rsa_key)
    with pytest.raises(AppConfigError, match="regular key file"):
        app_jwt(config(link))


def test_invalid_key_is_not_echoed_in_errors_or_reprs(tmp_path: Path) -> None:
    key = tmp_path / "invalid.pem"
    sentinel = "TEST-ONLY-NOT-A-SECRET-PEM-SENTINEL"
    key.write_text(sentinel)
    key.chmod(0o600)
    with pytest.raises(AppConfigError) as err:
        app_jwt(config(key))
    assert sentinel not in str(err.value) + repr(err.value) + repr(config(key))


def test_ec_key_is_not_mislabeled_rs256(tmp_path: Path) -> None:
    key = tmp_path / "test-only-ec.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "EC", "-pkeyopt", "ec_paramgen_curve:P-256", "-out", str(key)],
        capture_output=True,
        check=True,
    )
    key.chmod(0o600)
    with pytest.raises(AppConfigError):
        app_jwt(config(key))


def test_pss_only_key_is_not_mislabeled_rs256(tmp_path: Path) -> None:
    key = tmp_path / "test-only-rsa-pss.pem"
    subprocess.run(
        ["openssl", "genpkey", "-algorithm", "RSA-PSS", "-pkeyopt", "rsa_keygen_bits:2048", "-out", str(key)],
        capture_output=True,
        check=True,
    )
    key.chmod(0o600)
    with pytest.raises(AppConfigError, match="could not sign"):
        app_jwt(config(key))


def test_config_reads_only_exact_variable_names(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    values = {APP_ID: "123", INSTALLATION_ID: "456", KEY_FILE: str(tmp_path / "private.pem")}
    monkeypatch.setattr(os, "environ", values)
    assert AppConfig.from_environ().app_id == "123"
    values.pop(INSTALLATION_ID)
    with pytest.raises(AppConfigError, match=INSTALLATION_ID):
        AppConfig.from_environ()


def test_token_request_is_downscoped_and_jwt_is_separate_from_installation_token(rsa_key: Path) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.github.com/app/installations/456/access_tokens"
        assert request.method == "POST"
        assert json.loads(request.content) == {"repositories": ["receipts-demo"], "permissions": {"checks": "write"}}
        assert request.headers["Authorization"].startswith("Bearer eyJ")
        return httpx.Response(201, json=token_answer())

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        assert installation_token(config(rsa_key), REPO, client=client) == TOKEN


def test_missing_repository_refuses_before_any_check_call(rsa_key: Path) -> None:
    calls: list[httpx.Request] = []

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(201, json=token_answer(repo="other-owner/receipts-demo"))

    with httpx.Client(transport=httpx.MockTransport(respond)) as client, pytest.raises(RepositoryRefusedError):
        post_check(config(rsa_key), REPO, check_payload([], make_target(), head_sha=SHA), client=client)
    assert len(calls) == 1
    assert calls[0].url.path.endswith("/access_tokens")


@pytest.mark.parametrize("seconds", [-1, 3_700])
def test_token_lifetime_outside_one_hour_is_refused(rsa_key: Path, seconds: int) -> None:
    with (
        httpx.Client(
            transport=httpx.MockTransport(lambda r: httpx.Response(201, json=token_answer(seconds=seconds)))
        ) as client,
        pytest.raises(TrackerError, match="one-hour lifetime"),
    ):
        installation_token(config(rsa_key), REPO, client=client)


@pytest.mark.parametrize("refuted", [False, True])
def test_repost_creates_a_fresh_check_without_duplicate_or_stale_annotations(rsa_key: Path, refuted: bool) -> None:
    calls: list[httpx.Request] = []
    runs: dict[int, dict[str, Any]] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        assert request.url.host == "api.github.com"
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json=token_answer())
        assert request.headers["Authorization"] == "Bearer " + TOKEN
        if request.method == "GET":
            assert request.url.params["check_name"] == CHECK_NAME
            assert request.url.params["app_id"] == "123"
            assert request.url.path == f"/repos/{REPO}/commits/{SHA}/check-runs"
            return httpx.Response(200, json={"total_count": len(runs), "check_runs": [existing_run(i) for i in runs]})
        body = json.loads(request.content)
        if request.method == "PATCH":
            run_id = int(request.url.path.rsplit("/", 1)[1])
            # GitHub appends annotations on update; it never replaces or clears them.
            body["output"]["annotations"] = runs[run_id]["output"]["annotations"] + body["output"]["annotations"]
            runs[run_id].update(body)
        else:
            assert request.method == "POST"
            assert request.url.path == f"/repos/{REPO}/check-runs"
            assert body["head_sha"] == SHA
            run_id = 42 + len(runs)
            runs[run_id] = body
        return httpx.Response(200, json={"id": run_id, "html_url": f"https://github.com/{REPO}/runs/{run_id}"})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        entries = [make_entry()]
        payload = check_payload(entries, make_target(), head_sha=SHA)
        first = post_check(config(rsa_key), REPO, payload, client=client)
        if refuted:
            entries.append(make_entry(Status.REFUTED))
        latest_payload = check_payload(entries, make_target(), head_sha=SHA)
        second = post_check(config(rsa_key), REPO, latest_payload, client=client)
    assert runs[second.id]["output"]["annotations"] == output(latest_payload)["annotations"]
    assert runs[second.id]["conclusion"] == ("success" if refuted else "failure")
    assert len(runs[first.id]["output"]["annotations"]) == 1
    assert first.id != second.id
    assert first.html_url != second.html_url
    assert len(runs) == 2
    assert [r.method for r in calls] == ["POST", "POST", "POST", "POST"]
    assert TOKEN not in repr(first) + repr(GitHubHTTP(TOKEN)) + repr(config(rsa_key))


def test_annotation_batches_append_only_to_the_run_created_by_each_post(rsa_key: Path) -> None:
    sizes: list[int] = []
    calls: list[tuple[str, str]] = []
    runs: dict[int, list[object]] = {}

    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json=token_answer())
        calls.append((request.method, request.url.path))
        body = json.loads(request.content)
        batch = body["output"]["annotations"]
        sizes.append(len(batch))
        if request.method == "POST":
            assert request.url.path == f"/repos/{REPO}/check-runs"
            assert body["head_sha"] == SHA
            run_id = 42 + len(runs)
            runs[run_id] = batch
        else:
            assert request.method == "PATCH"
            assert "head_sha" not in body
            run_id = int(request.url.path.rsplit("/", 1)[1])
            runs[run_id].extend(batch)
        return httpx.Response(200, json={"id": run_id, "html_url": f"https://github.com/{REPO}/runs/{run_id}"})

    payload = check_payload([make_entry(suffix=str(i)) for i in range(101)], make_target(), head_sha=SHA)
    with httpx.Client(transport=httpx.MockTransport(respond)) as client:
        first = post_check(config(rsa_key), REPO, payload, client=client)
        second = post_check(config(rsa_key), REPO, payload, client=client)
    assert first.id != second.id
    assert sizes == [50, 50, 1, 50, 50, 1]
    assert runs[first.id] == runs[second.id] == output(payload)["annotations"]
    assert calls == [
        (method, f"/repos/{REPO}/check-runs" + (f"/{run.id}" if method == "PATCH" else ""))
        for run in (first, second)
        for method in ("POST", "PATCH", "PATCH")
    ]


@pytest.mark.parametrize(
    "url",
    [
        "https://evil.example/runs/42",
        "https://[",
        f"https://github.com/{REPO}/runs/42?token={TOKEN}",
        f"https://github.com/{REPO}/runs/{TOKEN}",
    ],
)
def test_posted_url_cannot_leak_a_credential_or_redirect_to_another_host(rsa_key: Path, url: str) -> None:
    def respond(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/access_tokens"):
            return httpx.Response(201, json=token_answer())
        assert request.method == "POST"
        assert request.url.path == f"/repos/{REPO}/check-runs"
        return httpx.Response(201, json={"id": 42, "html_url": url})

    with httpx.Client(transport=httpx.MockTransport(respond)) as client, pytest.raises(TrackerError) as err:
        post_check(config(rsa_key), REPO, check_payload([], make_target(), head_sha=SHA), client=client)
    assert TOKEN not in str(err.value) + repr(err.value)


@pytest.mark.parametrize("status", [302, 401, 429, 500])
def test_remote_errors_and_redirects_never_expose_credentials(rsa_key: Path, status: int) -> None:
    calls: list[httpx.Request] = []
    pem = rsa_key.read_text()

    def respond(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            json={"message": request.headers["Authorization"] + TOKEN + pem},
            headers={"Location": "https://evil.example/steal"},
        )

    with httpx.Client(transport=httpx.MockTransport(respond)) as client, pytest.raises(TrackerError) as err:
        post_check(config(rsa_key), REPO, check_payload([], make_target(), head_sha=SHA), client=client)
    assert len(calls) == 1
    assert TOKEN not in str(err.value) + repr(err.value)
    assert pem not in str(err.value) + repr(err.value)
    assert calls[0].headers["Authorization"] not in str(err.value) + repr(err.value)
    assert err.value.__suppress_context__
