"""Digest pins are taken only when the manifest bytes hash to the digest the registry claims. MockTransport only."""

from __future__ import annotations

import json

import httpx
import pytest
from pydantic import ValidationError

from receipts.eval.hashing import sha256_bytes
from receipts.eval.images import ImageRef, ImageSet, RegistryError, repository, resolve, resolve_all
from receipts.eval.sources import make_client

MANIFEST = json.dumps(
    {
        "schemaVersion": 2,
        "mediaType": "application/vnd.docker.distribution.manifest.v2+json",
        "config": {"digest": "sha256:" + "c" * 64, "size": 10},
        "layers": [{"digest": "sha256:" + "1" * 64, "size": 100}, {"digest": "sha256:" + "2" * 64, "size": 50}],
    }
).encode()
TYPE = "application/vnd.docker.distribution.manifest.v2+json"


def registry(digest: str | None = None, media_type: str = TYPE) -> httpx.MockTransport:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/token":
            return httpx.Response(200, json={"token": "t"})
        assert request.headers["authorization"] == "Bearer t"
        headers = {"docker-content-digest": digest or f"sha256:{sha256_bytes(MANIFEST)}", "content-type": media_type}
        return httpx.Response(200, content=MANIFEST, headers=headers)

    return httpx.MockTransport(handler)


def test_a_pin_is_the_manifest_digest_and_its_layers() -> None:
    with make_client(transport=registry()) as client:
        ref, layers = resolve(client, "django__django-1")
    assert ref.digest == f"sha256:{sha256_bytes(MANIFEST)}"
    assert ref.ref == f"ghcr.io/{repository('django__django-1')}@{ref.digest}"
    assert ref.compressed_bytes == 150 and len(layers) == 2


def test_plant_a_registry_digest_that_does_not_match_the_bytes_is_refused() -> None:
    with make_client(transport=registry(digest="sha256:" + "0" * 64)) as client, pytest.raises(RegistryError):
        resolve(client, "django__django-1")


def test_plant_an_unexpected_manifest_type_is_refused() -> None:
    with make_client(transport=registry(media_type="text/html")) as client, pytest.raises(RegistryError):
        resolve(client, "django__django-1")


def test_shared_layers_are_counted_once() -> None:
    with make_client(transport=registry()) as client:
        images = resolve_all(client, ["django__django-1", "django__django-2"], "2026-10-27T00:00:00Z", workers=2)
    assert len(images.images) == 2
    assert images.unique_compressed_bytes == 150


def test_plant_a_tag_or_a_short_digest_cannot_be_a_pin() -> None:
    good = {"instance_id": "django__django-1", "media_type": TYPE}
    with pytest.raises(ValidationError):
        ImageRef(
            image="ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1:latest",
            digest="sha256:" + "a" * 64,
            **good,
        )
    with pytest.raises(ValidationError):
        ImageRef(image="ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1", digest="sha256:abc", **good)
    with pytest.raises(ValidationError):
        ImageRef(image="docker.io/library/ubuntu", digest="sha256:" + "a" * 64, **good)


def test_a_subset_refuses_instances_without_a_pin() -> None:
    ref = ImageRef(
        instance_id="django__django-1",
        image="ghcr.io/epoch-research/swe-bench.eval.x86_64.django__django-1",
        digest="sha256:" + "a" * 64,
        media_type=TYPE,
    )
    images = ImageSet(resolved_at="2026-10-27T00:00:00Z", images=(ref,), unique_compressed_bytes=0)
    assert images.subset(["django__django-1"]) == [ref]
    with pytest.raises(RegistryError, match="no pinned image"):
        images.subset(["django__django-1", "sympy__sympy-2"])
