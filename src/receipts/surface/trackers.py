"""The HTTP transports the ticket writers post through, over httpx.

Each takes its token from the caller and never reads one itself. The token goes only in the Authorization header of
the tracker's own API host: never to an upload URL, never through a redirect (none is followed), and never into a log
line, an exception or a repr. Linear's personal API keys go in the header as they are; GitHub's tokens go as Bearer.
Tests drive these with httpx.MockTransport, so no test reaches the network.

Sources, read 2026-10-27: Linear's GraphQL schema (github.com/linear/linear, packages/sdk/src/schema.graphql) for
attachmentsForURL, fileUpload, issueCreate, issue and attachmentCreate; linear.app/developers/attachments (attachment
metadata and its attributes); linear.app/developers/how-to-upload-a-file-to-linear (a PUT to the pre-signed URL with
the returned headers); linear.app/developers/rate-limiting (HTTP 429, the RATELIMITED code, Retry-After).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal
from urllib.parse import urlsplit

import httpx

from receipts.surface.text import visible
from receipts.surface.tickets import RateLimitedError, TrackerError

LINEAR_API = "https://api.linear.app/graphql"
GITHUB_API = "https://api.github.com"
GITHUB_VERSION = "2022-11-28"
MESSAGE_LIMIT = 300  # characters of a tracker's own error text kept in an exception


class LinearHTTP:
    """The Linear transport: GraphQL over POST, and the upload's PUT to a pre-signed URL on an allowed host. Linear
    does not document the upload host, so only hosts under linear.app are allowed unless the caller adds one."""

    def __init__(
        self,
        api_key: str,
        *,
        client: httpx.Client | None = None,
        upload_hosts: Sequence[str] = ("linear.app",),
        timeout_s: float = 30.0,
    ) -> None:
        if not api_key or any(ch.isspace() for ch in api_key):
            raise ValueError("a Linear API key is one token with no whitespace")
        self._key = api_key
        self._client = client or httpx.Client(timeout=timeout_s)
        self._upload_hosts = tuple(h.lower().strip(".") for h in upload_hosts)

    def __repr__(self) -> str:
        return f"LinearHTTP(api={LINEAR_API!r}, upload_hosts={self._upload_hosts!r})"

    def graphql(self, query: str, variables: Mapping[str, object]) -> Mapping[str, object]:
        try:
            response = self._client.post(
                LINEAR_API,
                json={"query": query, "variables": dict(variables)},
                headers={"Authorization": self._key},
                follow_redirects=False,
            )
        except httpx.HTTPError as err:
            raise TrackerError(f"Linear did not answer: {type(err).__name__}") from err
        payload = _json(response)
        errors = payload.get("errors") if isinstance(payload, dict) else None
        if response.status_code == 429 or _has_code(errors, "RATELIMITED"):
            raise RateLimitedError("Linear rate limited the call", retry_after=_retry_after(response))
        if errors:
            raise TrackerError(f"Linear refused the call: {_messages(errors)}")
        if not response.is_success:
            raise TrackerError(f"Linear answered HTTP {response.status_code}")
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, dict):
            raise TrackerError("Linear's answer has no data")
        return data

    def upload(self, url: str, headers: Mapping[str, str], content: bytes) -> None:
        parts = urlsplit(url)
        host = (parts.hostname or "").lower()
        allowed = any(host == h or host.endswith("." + h) for h in self._upload_hosts)
        if parts.scheme != "https" or not allowed or parts.username or parts.password:
            raise TrackerError(f"the upload URL's host {visible(host) or '(none)'} is not one this writer uploads to")
        request = self._client.build_request("PUT", url, content=content, headers=dict(headers))
        request.headers.pop("Authorization", None)  # a pre-signed URL never gets the API key, whatever the client holds
        try:
            response = self._client.send(request, follow_redirects=False)
        except httpx.HTTPError as err:
            raise TrackerError(f"the upload did not answer: {type(err).__name__}") from err
        if not response.is_success:
            raise TrackerError(f"the upload answered HTTP {response.status_code}")


class GitHubHTTP:
    """The GitHub transport: the REST API with a Bearer token, JSON in and out."""

    def __init__(
        self,
        token: str,
        *,
        client: httpx.Client | None = None,
        api: str = GITHUB_API,
        timeout_s: float = 30.0,
    ) -> None:
        if not token or any(ch.isspace() for ch in token):
            raise ValueError("a GitHub token is one token with no whitespace")
        if urlsplit(api).scheme != "https":
            raise ValueError("the GitHub API is reached over https")
        self._token = token
        self._api = api.rstrip("/")
        self._client = client or httpx.Client(timeout=timeout_s)

    def __repr__(self) -> str:
        return f"GitHubHTTP(api={self._api!r})"

    def request(
        self,
        method: Literal["GET", "POST"],
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        body: Mapping[str, object] | None = None,
    ) -> object:
        segments = path.split("/")
        if not path.startswith("/") or "" in segments[1:] or "." in segments or ".." in segments or "?" in path:
            raise ValueError(f"an API path is /a/b with plain segments: {path!r}")
        headers = {
            "Authorization": f"Bearer {self._token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": GITHUB_VERSION,
            "User-Agent": "falden-receipts",
        }
        try:
            response = self._client.request(
                method,
                self._api + path,
                params=dict(params) if params else None,
                json=dict(body) if body is not None else None,
                headers=headers,
                follow_redirects=False,
            )
        except httpx.HTTPError as err:
            raise TrackerError(f"GitHub did not answer: {type(err).__name__}") from err
        payload = _json(response)
        limited = response.status_code == 429 or (
            response.status_code == 403 and response.headers.get("x-ratelimit-remaining") == "0"
        )
        if limited:
            raise RateLimitedError("GitHub rate limited the call", retry_after=_retry_after(response))
        if not response.is_success:
            message = payload.get("message") if isinstance(payload, dict) else None
            detail = f": {_clean(message)}" if isinstance(message, str) else ""
            raise TrackerError(f"GitHub answered HTTP {response.status_code}{detail}")
        return payload


def _json(response: httpx.Response) -> object:
    if not response.content:
        return {}
    try:
        return response.json()
    except ValueError:
        if response.is_success:
            raise TrackerError("the tracker's answer is not JSON") from None
        return {}


def _has_code(errors: object, code: str) -> bool:
    if not isinstance(errors, list):
        return False
    for error in errors:
        extensions = error.get("extensions") if isinstance(error, dict) else None
        if isinstance(extensions, dict) and extensions.get("code") == code:
            return True
    return False


def _messages(errors: object) -> str:
    if not isinstance(errors, list):
        return "an unreadable error"
    texts = [str(e.get("message")) for e in errors if isinstance(e, dict) and e.get("message")]
    return _clean("; ".join(texts) or "an error with no message")


def _clean(text: str) -> str:
    one_line = " ".join(visible(text).split())
    return one_line if len(one_line) <= MESSAGE_LIMIT else one_line[: MESSAGE_LIMIT - 3] + "..."


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        seconds = float(value) if value is not None else None
    except ValueError:
        return None  # an HTTP date: the caller waits its own default
    return seconds if seconds is not None and 0 <= seconds < 86_400 else None
