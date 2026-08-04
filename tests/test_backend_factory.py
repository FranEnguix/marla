from types import SimpleNamespace

import pytest

from marla.models.backend_factory import build_backend
from marla.models.remote_backend import RemoteApiBackend


@pytest.mark.asyncio
async def test_remote_backend_is_constructed_but_raises_on_generate():
    model_config = SimpleNamespace(backend="remote", name="some-remote-model")
    backend = build_backend(model_config, device="cpu")
    assert isinstance(backend, RemoteApiBackend)
    with pytest.raises(NotImplementedError):
        await backend.generate("prompt", legal_action_ids=[])


def test_unknown_backend_raises_value_error():
    model_config = SimpleNamespace(backend="carrier-pigeon", name="x")
    with pytest.raises(ValueError):
        build_backend(model_config, device="cpu")
