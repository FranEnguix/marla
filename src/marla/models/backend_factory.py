"""Constructs the configured Plan Maker backend (spec section 18: device resolution is local)."""

from __future__ import annotations

from marla.config.models import PlanMakerModelConfig
from marla.models.local_backend import LocalTransformersBackend
from marla.models.plan_maker_backend import PlanMakerBackend
from marla.models.remote_backend import RemoteApiBackend
from marla.runtime.device import DeviceRequest


def build_backend(model_config: PlanMakerModelConfig, device: DeviceRequest) -> PlanMakerBackend:
    if model_config.backend == "local":
        return LocalTransformersBackend(
            model_name=model_config.name, device=device, max_new_tokens=model_config.max_new_tokens
        )
    if model_config.backend == "remote":
        return RemoteApiBackend(model_name=model_config.name)
    raise ValueError(f"Unknown Plan Maker model backend: {model_config.backend!r}")
