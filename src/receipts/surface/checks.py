"""GitHub check output and a posting-only App. No repository code runs in this process."""

from __future__ import annotations

import base64
import json
import os
import re
import shlex
import stat
import subprocess
import time
from collections import Counter
from collections.abc import Iterable, Iterator, Mapping
from datetime import datetime
from pathlib import Path, PurePosixPath
from typing import Literal, cast
from urllib.parse import urlsplit

import httpx
from pydantic import BaseModel, ConfigDict, Field, StrictInt, ValidationError

from receipts.agentloop.diff import changed_files
from receipts.contracts import Finding, LedgerEntry, Outcome, Side, Status, Target, Tier
from receipts.surface.ledger import validated
from receipts.surface.markdown import code_block, code_span, plain, strip_links
from receipts.surface.text import visible
from receipts.surface.tickets import TrackerError
from receipts.surface.trackers import GitHubHTTP

CHECK_NAME = "Falden Receipts"
SUMMARY_LIMIT = 60_000
ANNOTATION_BATCH = 50
APP_ID = "RECEIPTS_GITHUB_APP_ID"
INSTALLATION_ID = "RECEIPTS_GITHUB_APP_INSTALLATION_ID"
KEY_FILE = "RECEIPTS_GITHUB_APP_KEY_FILE"
_SHA = re.compile(r"[0-9a-fA-F]{40}")
_REPO = re.compile(r"[A-Za-z0-9][A-Za-z0-9-]*/[A-Za-z0-9_.-]+")
_HUNK = re.compile(r"^@@ -\d+(?:,(\d+))? \+(\d+)(?:,(\d+))? @@")
_FRAME = re.compile(r'^\s*File "([^"\n]+)", line ([0-9]+)', re.MULTILINE)
_CLIPPED = "Summary clipped; the full evidence remains in the ledger."
_INCONCLUSIVE = (Status.FLAKY, Status.POSSIBLY_TOXIC, Status.S_SAT_INCONCLUSIVE)


class CheckInputError(ValueError):
    """Input cannot safely describe a check on this head commit."""


class AppConfigError(ValueError):
    """Missing or unsafe App configuration, with no credential values in its message."""


class RepositoryRefusedError(TrackerError):
    """The installation token is not scoped to the requested repository."""


class AppConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    app_id: str = Field(pattern=r"^[1-9][0-9]*$")
    installation_id: str = Field(pattern=r"^[1-9][0-9]*$")
    key_file: Path = Field(repr=False)

    @classmethod
    def from_environ(cls) -> AppConfig:
        values: list[str] = []
        for name in (APP_ID, INSTALLATION_ID, KEY_FILE):
            value = os.environ.get(name)
            if not value:
                raise AppConfigError(f"missing {name}")
            if name != KEY_FILE and re.fullmatch(r"[1-9][0-9]*", value) is None:
                raise AppConfigError(f"{name} must be a positive integer")
            values.append(value)
        return cls(app_id=values[0], installation_id=values[1], key_file=Path(values[2]))


class _Response(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)


class _Repository(_Response):
    full_name: str


class _InstallationToken(_Response):
    token: str = Field(min_length=1, repr=False)
    expires_at: datetime
    repositories: tuple[_Repository, ...]


class PostedCheck(_Response):
    id: StrictInt = Field(gt=0)
    html_url: str


def validate_destination(target: Target, repo: str, head_sha: str) -> None:
    """Bind the receipt to the exact repository and patch commit before acquiring credentials."""
    if _REPO.fullmatch(repo) is None or repo.split("/")[1] in (".", ".."):
        raise CheckInputError("--repo must be OWNER/NAME")
    if _SHA.fullmatch(head_sha) is None:
        raise CheckInputError("--head-sha must be a full 40-character commit SHA")
    if target.repo.casefold() != repo.casefold():
        raise CheckInputError("--repo does not match the target repository")
    if target.patch_sha is None or target.patch_sha.casefold() != head_sha.casefold():
        raise CheckInputError("--head-sha does not match the target patch SHA")


def check_payload(entries: Iterable[LedgerEntry], target: Target, *, head_sha: str) -> dict[str, object]:
    """Latest ledger state wins. All model text is inert in GitHub's markdown renderer."""
    validate_destination(target, target.repo, head_sha)
    latest: dict[str, Finding] = {}
    for raw in entries:
        finding = validated(raw).finding
        if finding.target_id == target.id:
            latest[finding.id] = finding
    findings = tuple(latest.values())
    for finding in findings:
        if finding.replay is not None and (
            finding.replay.base_sha.casefold() != target.base_sha.casefold()
            or finding.replay.patch_sha.casefold() != head_sha.casefold()
        ):
            raise CheckInputError("the finding replay key does not match the target base and patch SHAs")
    counts = Counter(f.status for f in findings)
    blockers = tuple(f for f in findings if f.status is Status.BLOCKER)
    notes = tuple(annotation(f, target) for f in blockers)
    unannotated = tuple(f.id for f, note in zip(blockers, notes, strict=True) if note is None)
    nothing_checked = not findings or all(f.status is Status.UNCHECKED for f in findings)
    inconclusive = any(counts[status] for status in _INCONCLUSIVE)
    neutral = counts[Status.SUSPECTED] or nothing_checked or inconclusive
    conclusion = "failure" if blockers else "neutral" if neutral else "success"
    if nothing_checked:
        title = f"Nothing was checked; {counts[Status.UNCHECKED]} unchecked"
    elif len(blockers) == 1:
        finding = blockers[0]
        patch = [r for r in finding.runs if r.side is Side.PATCH]
        title = f"1 blocker: {_kind(finding)}, {sum(r.outcome is Outcome.FAIL for r in patch)} of {len(patch)} runs"
    elif blockers:
        title = f"{len(blockers)} blockers; {counts[Status.SUSPECTED]} suspected, {counts[Status.UNCHECKED]} unchecked"
    else:
        title = f"No blockers; {counts[Status.SUSPECTED]} suspected, {counts[Status.UNCHECKED]} unchecked"
        if inconclusive:
            title += "; " + ", ".join(
                f"{counts[status]} {status.value.replace('_', ' ')}" for status in _INCONCLUSIVE if counts[status]
            )
    return {
        "name": CHECK_NAME,
        "head_sha": head_sha,
        "external_id": target.id,
        "status": "completed",
        "conclusion": conclusion,
        "output": {
            "title": plain(title),
            "summary": _summary(findings, counts, unannotated),
            "annotations": [note for note in notes if note is not None],
        },
    }


def _kind(finding: Finding) -> str:
    return f"{finding.kind.value.replace('_', ' ')} catch" if finding.kind is not None else "unknown catch"


def _summary(findings: tuple[Finding, ...], counts: Counter[Status], unannotated: tuple[str, ...]) -> str:
    parts: list[str] = []
    size = 0
    for part in _summary_parts(findings, counts, unannotated):
        if size + len(part) + 2 + len(_CLIPPED) + 2 > SUMMARY_LIMIT:
            parts.append(_CLIPPED)
            break
        parts.append(part)
        size += len(part) + 2
    return "\n\n".join(parts)


def _summary_parts(
    findings: tuple[Finding, ...], counts: Counter[Status], unannotated: tuple[str, ...]
) -> Iterator[str]:
    yield "## Falden Receipts"
    yield "\n".join(f"- {status.value.replace('_', ' ')}: {counts[status]}" for status in Status)
    for finding_id in unannotated:
        yield (
            f"Blocker {code_span(strip_links(finding_id), limit=500)} has no line annotation: "
            "the patch has no added non-test line that can be annotated safely."
        )
    for finding in findings:
        if finding.status is Status.UNCHECKED:
            reason = finding.unchecked_reason.value if finding.unchecked_reason else "unknown"
            yield f"Unchecked lens: {finding.lens.value}; reason: {reason}; finding: {code_span(finding.id, limit=500)}"
            if finding.detail:
                yield code_block(strip_links(finding.detail), limit=4_000)
        elif finding.status is Status.BLOCKER:
            yield f"### {_kind(finding)}: {code_span(strip_links(finding.id), limit=500)}"
            yield "Quoted claim or issue sentence:"
            yield code_block(strip_links(finding.quote or ""), limit=6_000)
            if finding.detail:
                yield "Detail:\n\n" + code_block(strip_links(finding.detail), limit=4_000)
            if finding.repro is not None:
                yield "Repro command: " + code_span(shlex.join(finding.repro.command), limit=2_000)
                for name, content in finding.repro.files.items():
                    yield "Repro file: " + code_span(strip_links(name), limit=500)
                    yield code_block(strip_links(content), limit=10_000)
            yield "\n".join(
                ["| Side | Run | Outcome | Failure |", "| --- | ---: | --- | --- |"]
                + [
                    f"| {run.side.value} | {i} | {run.outcome.value} | "
                    f"{run.failure_type.value if run.failure_type else 'none'} |"
                    for i, run in enumerate(finding.runs, 1)
                ]
            )
            if finding.replay is not None:
                yield "Replay key:\n\n" + code_block(finding.replay.model_dump_json(), "json", limit=4_000)
            failed = next((r for r in finding.runs if r.side is Side.PATCH and r.outcome is Outcome.FAIL), None)
            if failed is not None:
                yield "First failing patch run, stdout:\n\n" + code_block(strip_links(failed.stdout_tail), limit=6_000)
                yield "First failing patch run, stderr:\n\n" + code_block(strip_links(failed.stderr_tail), limit=6_000)


def _non_test(path: str) -> bool:
    parts = PurePosixPath(path).parts
    return bool(parts) and not any(
        part.lower() in ("test", "tests", "testing", "tests.py", "conftest.py")
        or part.lower().startswith("test_")
        or part.lower().endswith("_test.py")
        or part.lower().endswith("_tests.py")
        for part in parts
    )


def _safe_path(path: str) -> bool:
    return (
        bool(path)
        and not path.startswith("/")
        and not any(part in ("", ".", "..") for part in path.split("/"))
        and visible(path, keep="") == path
        and "\\" not in path
    )


def _added_lines(patch: str) -> dict[str, list[int]]:
    """Consume hunk lengths so added content cannot impersonate a diff header."""
    result: dict[str, list[int]] = {}
    path: str | None = None
    old_left = new_left = number = 0
    for line in patch.split("\n"):
        if old_left > 0 or new_left > 0:
            if line.startswith("+"):
                if path is not None and new_left > 0:
                    result.setdefault(path, []).append(number)
                number += 1
                new_left -= 1
                continue
            if line.startswith("-"):
                old_left -= 1
                continue
            if line.startswith(" ") or line == "":
                number += 1
                old_left -= 1
                new_left -= 1
                continue
            if line.startswith("\\"):
                continue
            old_left = new_left = 0
        if line.startswith(("diff --git ", "+++ ")):
            paths = changed_files(line)
            path = paths[0] if len(paths) == 1 and _safe_path(paths[0]) and _non_test(paths[0]) else None
        match = _HUNK.match(line)
        if match:
            old_left = int(match[1] or "1")
            number = int(match[2])
            new_left = int(match[3] or "1")
    return result


def annotation(finding: Finding, target: Target) -> dict[str, object] | None:
    """Locate a safe added line, or omit the annotation without changing the blocker's status."""
    added = _added_lines(target.patch)
    root = "/testbed/" if target.tier is Tier.A else "/workspace/"
    failed = next((r for r in finding.runs if r.side is Side.PATCH and r.outcome is Outcome.FAIL), None)
    located: tuple[str, int] | None = None
    if failed is not None:
        for frame in _FRAME.finditer(failed.stdout_tail + "\n" + failed.stderr_tail):
            absolute, number = frame[1], int(frame[2])
            if absolute.startswith(root):
                path = absolute[len(root) :]
                if _safe_path(path) and number in added.get(path, ()):
                    located = path, number
    fallback = located is None
    if located is None:
        located = next(((path, lines[0]) for path, lines in added.items() if lines), None)
    if located is None:
        return None
    path, number = located
    message = f"{_kind(finding)}: {finding.id}. {finding.quote or ''} {finding.detail or ''}"
    if fallback:
        message = "line not located from the traceback. " + message
    return {
        "path": path,
        "start_line": number,
        "end_line": number,
        "annotation_level": "failure",
        "title": plain(_kind(finding)),
        "message": plain(strip_links(visible(message, keep="")), limit=8_000),
    }


def _base64(data: bytes) -> str:
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def app_jwt(config: AppConfig, *, now: int | None = None) -> str:
    """Sign RS256 using the checked, already open key descriptor. No PEM enters an argument or log."""
    timestamp = int(time.time()) if now is None else now
    header = _base64(b'{"alg":"RS256","typ":"JWT"}')
    claims = _base64(json.dumps({"iat": timestamp - 60, "exp": timestamp + 540, "iss": config.app_id}).encode())
    signing_input = f"{header}.{claims}"
    try:
        fd = os.open(config.key_file, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    except OSError:
        raise AppConfigError(f"{KEY_FILE} must name a readable regular key file") from None
    try:
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or stat.S_IMODE(info.st_mode) & ~0o600:
            raise AppConfigError(f"{KEY_FILE} must be a regular file with mode 600 or stricter")
        try:
            os.lseek(fd, 0, os.SEEK_SET)
            signed = subprocess.run(
                ["openssl", "dgst", "-sha256", "-sign", f"/dev/fd/{fd}", "-sigopt", "rsa_padding_mode:pkcs1"],
                input=signing_input.encode("ascii"),
                capture_output=True,
                pass_fds=(fd,),
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise AppConfigError("OpenSSL could not sign the App JWT") from None
        if signed.returncode != 0:
            raise AppConfigError("OpenSSL could not sign the App JWT")
        # Reject an EC or other non-RSA key even if openssl dgst can sign with it.
        try:
            # /dev/fd shares the file offset on macOS, so signing may have consumed it.
            os.lseek(fd, 0, os.SEEK_SET)
            rsa = subprocess.run(
                ["openssl", "rsa", "-in", f"/dev/fd/{fd}", "-check", "-noout"],
                input=b"",
                capture_output=True,
                pass_fds=(fd,),
                timeout=15,
                check=False,
            )
        except (OSError, subprocess.SubprocessError):
            raise AppConfigError("OpenSSL could not validate the RSA key") from None
        if rsa.returncode != 0:
            raise AppConfigError("the App JWT requires an RSA private key")
    finally:
        os.close(fd)
    return f"{signing_input}.{_base64(signed.stdout)}"


def _request(
    http: GitHubHTTP,
    method: Literal["GET", "POST", "PATCH"],
    path: str,
    *,
    params: Mapping[str, str] | None = None,
    body: Mapping[str, object] | None = None,
) -> object:
    try:
        # The shared transport delegates to httpx, which supports PATCH. Its annotation predates check runs.
        return http.request(cast(Literal["GET", "POST"], method), path, params=params, body=body)
    except (TrackerError, ValueError):
        # A remote error can echo an Authorization value. Do not expose its text or chained exception.
        raise TrackerError("GitHub check request failed") from None


def installation_token(config: AppConfig, repo: str, *, client: httpx.Client, now: int | None = None) -> str:
    jwt = app_jwt(config, now=now)
    raw = _request(
        GitHubHTTP(jwt, client=client),
        "POST",
        f"/app/installations/{config.installation_id}/access_tokens",
        body={"repositories": [repo.split("/")[1]], "permissions": {"checks": "write"}},
    )
    try:
        answer = _InstallationToken.model_validate(raw)
    except ValidationError:
        raise TrackerError("GitHub returned an invalid installation token answer") from None
    if repo.casefold() not in {r.full_name.casefold() for r in answer.repositories}:
        raise RepositoryRefusedError("the installation token does not include --repo")
    timestamp = time.time() if now is None else now
    if answer.expires_at.tzinfo is None or not 0 < answer.expires_at.timestamp() - timestamp <= 3_600:
        raise TrackerError("GitHub returned an installation token outside its one-hour lifetime")
    if any(ch.isspace() for ch in answer.token):
        raise TrackerError("GitHub returned an invalid installation token")
    return answer.token


def post_check(
    config: AppConfig, repo: str, payload: Mapping[str, object], *, client: httpx.Client | None = None
) -> PostedCheck:
    """Create a fresh check; append extra annotation batches only to this run. Never follow redirects."""
    if client is None:
        with httpx.Client(timeout=30.0, trust_env=False) as owned_client:
            return post_check(config, repo, payload, client=owned_client)
    sha = payload.get("head_sha")
    if _REPO.fullmatch(repo) is None or not isinstance(sha, str) or _SHA.fullmatch(sha) is None:
        raise CheckInputError("invalid repository or check head SHA")
    token = installation_token(config, repo, client=client)
    http = GitHubHTTP(token, client=client)
    output = payload.get("output")
    if not isinstance(output, dict) or not isinstance(output.get("annotations"), list):
        raise CheckInputError("a check payload must contain output and annotations")
    annotations = output["annotations"]
    batches = [annotations[i : i + ANNOTATION_BATCH] for i in range(0, len(annotations), ANNOTATION_BATCH)] or [[]]
    result: PostedCheck | None = None
    for batch in batches:
        body = dict(payload)
        body["output"] = {**output, "annotations": batch}
        if result is None:
            raw = _request(http, "POST", f"/repos/{repo}/check-runs", body=body)
        else:
            body.pop("head_sha", None)
            raw = _request(http, "PATCH", f"/repos/{repo}/check-runs/{result.id}", body=body)
        try:
            result = PostedCheck.model_validate(raw)
        except ValidationError:
            raise TrackerError("GitHub returned an invalid posted check run") from None
        try:
            parts = urlsplit(result.html_url)
        except ValueError:
            raise TrackerError("GitHub returned an invalid check-run URL") from None
        if (
            parts.scheme != "https"
            or parts.netloc != "github.com"
            or parts.path.casefold() != f"/{repo}/runs/{result.id}".casefold()
            or parts.query not in ("", "check_suite_focus=true")
            or parts.fragment
        ):
            raise TrackerError("GitHub returned an unexpected check-run URL")
    assert result is not None
    return result
