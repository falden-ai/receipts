"""Epoch AI's SWE-bench images (https://epoch.ai/blog/swebench-docker), pinned by digest.

Epoch publishes one image per instance at ghcr.io/epoch-research/swe-bench.eval.<arch>.<instance_id>, tagged only
`latest`. A tag can move, so the eval never pulls by tag: `resolve` reads the manifest the tag points to today, checks
that its sha256 equals the digest the registry reports, and records `name@sha256:...`. The VM pulls those references
and nothing else.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from concurrent.futures import ThreadPoolExecutor

import httpx
from pydantic import Field

from receipts.contracts import Frozen
from receipts.eval.hashing import sha256_bytes
from receipts.eval.pool import INSTANCE_ID

REGISTRY = "ghcr.io"
NAMESPACE = "epoch-research"
ARCH = "x86_64"
DIGEST = r"^sha256:[0-9a-f]{64}$"
MANIFEST_TYPES = (
    "application/vnd.oci.image.index.v1+json",
    "application/vnd.docker.distribution.manifest.list.v2+json",
    "application/vnd.docker.distribution.manifest.v2+json",
    "application/vnd.oci.image.manifest.v1+json",
)


def repository(instance_id: str, arch: str = ARCH) -> str:
    """The image name without the registry: epoch-research/swe-bench.eval.x86_64.<instance_id>."""
    return f"{NAMESPACE}/swe-bench.eval.{arch}.{instance_id}"


class ImageRef(Frozen):
    instance_id: str = Field(pattern=INSTANCE_ID)
    image: str = Field(pattern=r"^ghcr\.io/epoch-research/swe-bench\.eval\.x86_64\.[A-Za-z0-9_.-]+$")
    digest: str = Field(pattern=DIGEST)
    media_type: str
    compressed_bytes: int | None = Field(default=None, ge=0)  # sum of layer sizes, for single-platform manifests

    @property
    def ref(self) -> str:
        return f"{self.image}@{self.digest}"


class RegistryError(RuntimeError):
    """A registry answer that cannot be trusted as a pin."""


def _token(client: httpx.Client, repo: str) -> str:
    r = client.get(f"https://{REGISTRY}/token", params={"scope": f"repository:{repo}:pull", "service": REGISTRY})
    r.raise_for_status()
    token = r.json().get("token")
    if not isinstance(token, str) or not token:
        raise RegistryError(f"{repo}: no pull token")
    return token


def resolve(client: httpx.Client, instance_id: str, tag: str = "latest") -> tuple[ImageRef, dict[str, int]]:
    """The pinned reference for one instance, and its layers (digest -> compressed size)."""
    repo = repository(instance_id)
    token = _token(client, repo)
    r = client.get(
        f"https://{REGISTRY}/v2/{repo}/manifests/{tag}",
        headers={"Authorization": f"Bearer {token}", "Accept": ", ".join(MANIFEST_TYPES)},
    )
    r.raise_for_status()
    body = r.content
    digest = r.headers.get("docker-content-digest", "")
    computed = f"sha256:{sha256_bytes(body)}"
    if digest != computed:
        raise RegistryError(f"{repo}: registry digest {digest!r} does not match the manifest bytes ({computed})")
    media_type = r.headers.get("content-type", "").split(";")[0].strip()
    if media_type not in MANIFEST_TYPES:
        raise RegistryError(f"{repo}: unexpected manifest type {media_type!r}")
    manifest = json.loads(body)
    layers: dict[str, int] = {}
    if isinstance(manifest, dict) and isinstance(manifest.get("layers"), list):
        layers = {str(layer["digest"]): int(layer["size"]) for layer in manifest["layers"]}
    ref = ImageRef(
        instance_id=instance_id,
        image=f"{REGISTRY}/{repo}",
        digest=digest,
        media_type=media_type,
        compressed_bytes=sum(layers.values()) if layers else None,
    )
    return ref, layers


def resolve_all(client: httpx.Client, instance_ids: Iterable[str], resolved_at: str, *, workers: int = 8) -> ImageSet:
    ids = sorted(set(instance_ids))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        results = list(pool.map(lambda i: resolve(client, i), ids))
    unique: dict[str, int] = {}
    for _, layers in results:
        unique.update(layers)
    return ImageSet(
        resolved_at=resolved_at,
        images=tuple(ref for ref, _ in results),
        unique_compressed_bytes=sum(unique.values()),
    )


class ImageSet(Frozen):
    """eval/images.json."""

    registry: str = REGISTRY
    resolved_at: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
    source: str = "https://epoch.ai/blog/swebench-docker"
    images: tuple[ImageRef, ...]
    unique_compressed_bytes: int = Field(ge=0)  # every distinct layer counted once: the real download size

    @property
    def by_instance(self) -> dict[str, ImageRef]:
        return {i.instance_id: i for i in self.images}

    def subset(self, instance_ids: Iterable[str]) -> list[ImageRef]:
        """The pins for these instances; refuses any instance without one."""
        index = self.by_instance
        wanted = sorted(set(instance_ids))
        missing = [i for i in wanted if i not in index]
        if missing:
            raise RegistryError(f"no pinned image for {len(missing)} instances, e.g. {missing[:3]}")
        return [index[i] for i in wanted]
