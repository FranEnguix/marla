"""Remote API Plan Maker backend -- not implemented in this release.

The config schema's ``model.backend: local | remote`` is a real,
spec-defined choice, but the first release only ships the local HF
transformers backend (see ``local_backend.py``). This stub keeps
``backend: remote`` a valid, documented configuration value that fails
loudly and early rather than silently pretending to produce advice.
"""

from __future__ import annotations

from marla.models.plan_maker_backend import BackendResponse


class RemoteApiBackend:
    def __init__(self, model_name: str):
        self.model_name = model_name

    async def generate(self, prompt: str, legal_action_ids: list[str]) -> BackendResponse:
        raise NotImplementedError(
            "model.backend: 'remote' is not implemented in this MARLA release. "
            "Use model.backend: 'local' (HF transformers, see models/local_backend.py), "
            "or implement a remote backend satisfying PlanMakerBackend and wire it "
            "into models/backend_factory.py."
        )
