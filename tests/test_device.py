import pytest
import torch

from marla.runtime.device import DeviceResolutionError, resolve_device


def test_cpu_always_resolves_to_cpu():
    resolved = resolve_device("cpu")
    assert resolved.requested == "cpu"
    assert resolved.resolved == "cpu"


def test_auto_resolves_to_cuda_if_available_else_cpu():
    resolved = resolve_device("auto")
    assert resolved.requested == "auto"
    if torch.cuda.is_available():
        assert resolved.resolved == "cuda"
    else:
        assert resolved.resolved == "cpu"


@pytest.mark.skipif(torch.cuda.is_available(), reason="only meaningful without CUDA")
def test_gpu_raises_when_cuda_unavailable():
    with pytest.raises(DeviceResolutionError):
        resolve_device("gpu")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.gpu
def test_gpu_resolves_when_cuda_available():
    resolved = resolve_device("gpu")
    assert resolved.resolved == "cuda"


def test_unknown_device_request_raises():
    with pytest.raises(ValueError):
        resolve_device("tpu")  # type: ignore[arg-type]
