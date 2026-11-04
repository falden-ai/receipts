"""The public files Tier A is built from, each pinned to a revision and a sha256, and the one downloader for them.

Nothing here is fetched from an address a caller supplies: every URL is built from constants, and every request,
redirects included, is checked against the same rules. AWS service endpoints (any host under amazonaws.com) are
refused, with one exception the founder approved on 2026-10-27: anonymous HTTPS GETs of the public bucket
s3://swe-bench-submissions, which holds the agent patches and their evaluation reports. A request to that host must
be a plain GET with no credentials of any kind; the client never reads proxy settings or .netrc from the
environment. A content delivery network that happens to run on AWS, such as Hugging Face's, is not an AWS service
endpoint and is allowed.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from pathlib import Path
from urllib.parse import urlsplit

import httpx
from pydantic import Field

from receipts.contracts import Frozen
from receipts.eval.hashing import SHA256, sha256_bytes, sha256_file, write_bytes_atomic

SWE_ABS_REPO_COMMIT = "26b4c736ca3d566d8395d7999304d0264abe7992"  # OpenAgentEval/SWE-ABS main, 2026-10-26
SWE_ABS_DATASET_REVISION = "79fc07a8f4ce5e3720e069948073f5f910b7577b"  # OpenAgentLab/SWE-Bench_Verified_ABS
SWE_BENCH_VERIFIED_REVISION = "c104f840cc67f8b6eec6f759ebc8b2693d585d4a"  # princeton-nlp/SWE-bench_Verified
EXPERIMENTS_COMMIT = "40f164d5b8f1d249bf95a6df8b74b577fd8e519d"  # SWE-bench/experiments main, 2026-09-03

SWE_ABS_PAPER = "https://arxiv.org/html/2603.00520v1"

_GH_MEDIA = f"https://media.githubusercontent.com/media/OpenAgentEval/SWE-ABS/{SWE_ABS_REPO_COMMIT}"
_EXPERIMENTS_RAW = f"https://raw.githubusercontent.com/SWE-bench/experiments/{EXPERIMENTS_COMMIT}/evaluation/verified"


class Source(Frozen):
    """One pinned upstream file."""

    name: str = Field(min_length=1)
    url: str = Field(pattern=r"^https://")
    path: str = Field(min_length=1)  # where it lands, relative to eval/data/
    revision: str = Field(min_length=7)
    sha256: str = Field(pattern=SHA256)
    size: int = Field(gt=0)
    note: str = ""


def _experiments(entry: str, file: str, sha256: str, size: int) -> Source:
    return Source(
        name=f"experiments:{entry}:{file}",
        url=f"{_EXPERIMENTS_RAW}/{entry}/{file}",
        path=f"experiments/{entry}/{file}",
        revision=EXPERIMENTS_COMMIT,
        sha256=sha256,
        size=size,
    )


SOURCES: tuple[Source, ...] = (
    Source(
        name="swe-abs:verdicts",
        url=f"{_GH_MEDIA}/data/swe_abs_per_instance_verdicts.json",
        path="swe-abs/swe_abs_per_instance_verdicts.json",
        revision=SWE_ABS_REPO_COMMIT,
        sha256="8c9d31cca70b02852144111e85f72bfe749d6a95745556b156b984c5d650228d",
        size=6178762,
        note="SWE-ABS per-patch re-evaluation of 39 agents on SWE-bench Verified (Git LFS, added 2026-10-26)",
    ),
    Source(
        name="swe-abs:key-map-top30",
        url=f"{_GH_MEDIA}/data/key_map_top30.json",
        path="swe-abs/key_map_top30.json",
        revision=SWE_ABS_REPO_COMMIT,
        sha256="8d13b351320e723bf46f6175a25bffa22b4cadd2b9e67024ac0fbbf5096a0704",
        size=2051,
        note="the 30 leaderboard agents the paper's Table 2 and Table 6 are computed over",
    ),
    Source(
        name="swe-abs:dataset",
        url=(
            "https://huggingface.co/datasets/OpenAgentLab/SWE-Bench_Verified_ABS/resolve/"
            f"{SWE_ABS_DATASET_REVISION}/data/test-00000-of-00001.parquet"
        ),
        path="hf/SWE-Bench_Verified_ABS/test-00000-of-00001.parquet",
        revision=SWE_ABS_DATASET_REVISION,
        sha256="12df02a43cb624a88bbfadf25493f18b7daf551a5d6061d47109cd5b88bc6f97",
        size=3168881,
        note="500 rows; test_patch is the SWE-ABS test patch, original_test_patch the SWE-bench one",
    ),
    Source(
        name="swe-bench-verified:dataset",
        url=(
            "https://huggingface.co/datasets/princeton-nlp/SWE-bench_Verified/resolve/"
            f"{SWE_BENCH_VERIFIED_REVISION}/data/test-00000-of-00001.parquet"
        ),
        path="hf/SWE-bench_Verified/test-00000-of-00001.parquet",
        revision=SWE_BENCH_VERIFIED_REVISION,
        sha256="a45b1fe4e2f0c8390b2b2938ac83e92ed5979000856808f3679c07812e9e6dcd",
        size=2096679,
        note="the FAIL_TO_PASS and PASS_TO_PASS lists the published results were graded against",
    ),
    _experiments(
        "20260217_mini-v2.0.0_glm-5-high",
        "metadata.yaml",
        "51caca169f8ea93795ea15d67cdb3c78740612dbc4040a5d30cbf65e0d660013",
        684,
    ),
    _experiments(
        "20260217_mini-v2.0.0_glm-5-high",
        "per_instance_details.json",
        "cd139949a7a4044709f28d628c0a6ae9861da3e334033ae736a92a44d1d0a57c",
        53220,
    ),
    _experiments(
        "20260217_mini-v2.0.0_claude-4-6-opus",
        "metadata.yaml",
        "57c9de0e30a7299338f659f0526e8e6bceb6e4e59e7c5a533528f3b7c3017264",
        826,
    ),
    _experiments(
        "20260217_mini-v2.0.0_claude-4-6-opus",
        "per_instance_details.json",
        "c669276c9b7c23ffe967df0d7cc9f1f6406001ae02095761ddb2ac067103430d",
        52805,
    ),
    _experiments(
        "20260217_mini-v2.0.0_minimax-2-5-high",
        "metadata.yaml",
        "9d11723206b90a1a3dc02b0a3c2c3870e8c487b756c36aff13120ad397e7abbc",
        761,
    ),
    _experiments(
        "20260217_mini-v2.0.0_minimax-2-5-high",
        "per_instance_details.json",
        "62a26211006b557f1a31371ee6cf3119be5bdaed43dcce3fe6efedc0912c012c",
        53456,
    ),
    _experiments(
        "20260217_mini-v2.0.0_gemini-3-flash-high",
        "metadata.yaml",
        "1f0567e39e292e9216cb669def83e6f6a9bd02ee3d9731c4f23cad13ac42816b",
        827,
    ),
    _experiments(
        "20260217_mini-v2.0.0_gemini-3-flash-high",
        "per_instance_details.json",
        "6e49e555537839726c02dbbefd45cdf63bbed8c138eaf349d783381ca0716e33",
        53102,
    ),
    _experiments(
        "20250807_openhands_gpt5",
        "metadata.yaml",
        "ae9131e63a403487454cd7d55e632704f646451b5bcaffe8075fafd1bd99d6c5",
        510,
    ),
    _experiments(
        "20250807_openhands_gpt5",
        "results/results.json",
        "4aba4bbf158e4cfb76d717cd617954749058cefecd364aec117bb45415f9b907",
        10707,
    ),
    Source(
        name="swebench:wheel",
        url=(
            "https://files.pythonhosted.org/packages/36/67/981d8b642ac3eac7c8a7b7832ff8b2fb74f96b28b5fcd9a8979879e5c46d/"
            "swebench-4.1.0-py3-none-any.whl"
        ),
        path="pypi/swebench-4.1.0-py3-none-any.whl",
        revision="swebench==4.1.0",
        sha256="1243776f720047cc9e20a427f7a52b75c13a07abda6154fb60fe77f82ec8af57",
        size=157231,
        note="the harness the VM grades with; its per-version test commands become Target.test_command",
    ),
)


def source(name: str) -> Source:
    for s in SOURCES:
        if s.name == name:
            return s
    raise KeyError(name)


class FetchRefusedError(RuntimeError):
    """A download this lane is not allowed to make."""


class IntegrityError(RuntimeError):
    """A file whose bytes are not the pinned ones."""


# The founder's approval, 2026-10-27: anonymous HTTPS reads of the public bucket s3://swe-bench-submissions. Only its
# virtual-hosted address is allowed; every other AWS endpoint, path-style addresses of this bucket included, is refused.
APPROVED_AWS_HOSTS: frozenset[str] = frozenset({"swe-bench-submissions.s3.amazonaws.com"})
_CREDENTIAL_HEADERS = ("authorization", "x-amz-security-token", "x-amz-content-sha256", "x-amz-date", "cookie")


def is_aws_endpoint(host: str) -> bool:
    host = host.lower().rstrip(".")
    return host == "amazonaws.com" or host.endswith(".amazonaws.com")


def check_url(url: str) -> str:
    parts = urlsplit(url)
    if parts.scheme != "https":
        raise FetchRefusedError(f"{url}: only https downloads are allowed")
    host = (parts.hostname or "").lower().rstrip(".")
    if not host:
        raise FetchRefusedError(f"{url}: no host")
    if parts.username or parts.password:
        raise FetchRefusedError(f"{url}: credentials in a URL are refused")
    if is_aws_endpoint(host) and host not in APPROVED_AWS_HOSTS:
        raise FetchRefusedError(
            f"{url}: {host} is an AWS service endpoint, and only {sorted(APPROVED_AWS_HOSTS)} is approved"
        )
    return url


def check_request(request: httpx.Request) -> None:
    """Every request: its URL passes `check_url`, and one to an approved AWS host is an anonymous GET."""
    check_url(str(request.url))
    host = (request.url.host or "").lower().rstrip(".")
    if host in APPROVED_AWS_HOSTS:
        if request.method != "GET":
            raise FetchRefusedError(f"{request.method} {request.url}: only GET is approved for {host}")
        sent = [h for h in _CREDENTIAL_HEADERS if h in request.headers]
        if sent:
            raise FetchRefusedError(f"{request.url}: no credentials go to {host}, refusing headers {sent}")


def make_client(*, transport: httpx.BaseTransport | None = None) -> httpx.Client:
    """An HTTP client that re-checks every request, redirects included, with `check_request`. It ignores the
    environment (no proxy variables, no .netrc), so nothing outside this code can add credentials to a request."""
    return httpx.Client(
        follow_redirects=True,
        timeout=httpx.Timeout(60.0, connect=20.0),
        headers={"User-Agent": "falden-receipts-eval/0.0.1"},
        event_hooks={"request": [check_request]},
        transport=transport,
        trust_env=False,
    )


def verify_bytes(src: Source, data: bytes) -> None:
    if len(data) != src.size:
        raise IntegrityError(f"{src.name}: {len(data)} bytes, expected {src.size}")
    got = sha256_bytes(data)
    if got != src.sha256:
        raise IntegrityError(f"{src.name}: sha256 {got}, expected {src.sha256}")


def verify_file(src: Source, data_dir: Path) -> Path:
    path = data_dir / src.path
    if not path.is_file():
        raise IntegrityError(f"{src.name}: {path} is missing; run `receipts eval fetch`")
    size = path.stat().st_size
    if size != src.size:
        raise IntegrityError(f"{src.name}: {path} has {size} bytes, expected {src.size}")
    got = sha256_file(path)
    if got != src.sha256:
        raise IntegrityError(f"{src.name}: {path} sha256 {got}, expected {src.sha256}")
    return path


def fetch(src: Source, data_dir: Path, client: httpx.Client, *, retries: int = 3) -> Path:
    """Download one pinned file into data_dir, unless the right bytes are already there."""
    path = data_dir / src.path
    if path.is_file():
        try:
            return verify_file(src, data_dir)
        except IntegrityError:
            path.unlink()
    last: Exception | None = None
    for attempt in range(retries):
        try:
            response = client.get(src.url)
            response.raise_for_status()
            data = response.content
            verify_bytes(src, data)
            write_bytes_atomic(path, data)
            return path
        except (httpx.TransportError, httpx.HTTPStatusError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"{src.name}: download failed after {retries} attempts: {last}")


def fetch_all(data_dir: Path, client: httpx.Client, sources: Iterable[Source] = SOURCES) -> list[Path]:
    return [fetch(s, data_dir, client) for s in sources]
