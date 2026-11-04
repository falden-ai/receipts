"""One contracts.ModelClient over several: each request goes to the client that serves its model. The pipeline hands
attackers one client, and each arm's attacker model picks its server: Meta's API for Muse Spark, Falden's own vLLM
for Muse Glimmer 30B."""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType

from receipts.contracts import ModelClient, ModelRefusedError, ModelRequest, ModelResponse, check_model


class ModelRouter:
    """contracts.ModelClient that sends each request to the client registered for its model id. A model with no
    client is refused before any call, as is any id off the allowlist."""

    def __init__(self, routes: Mapping[str, ModelClient]) -> None:
        for model in routes:
            check_model(model)
        self._routes = MappingProxyType(dict(routes))

    @property
    def models(self) -> tuple[str, ...]:
        return tuple(sorted(self._routes))

    def complete(self, request: ModelRequest) -> ModelResponse:
        model = check_model(request.model)
        client = self._routes.get(model)
        if client is None:
            raise ModelRefusedError(f"{model}: no client serves it in this run (served: {', '.join(self.models)})")
        return client.complete(request)
