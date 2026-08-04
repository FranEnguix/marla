"""Resolve the ``device: cpu | gpu | auto`` configuration option.

Each machine/process resolves its own device independently (this matters in
distributed mode, where the Plan Maker may run on different hardware than
the RL Orchestrator). Remote API model backends are never subject to these
checks -- only locally executed ML models are.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

DeviceRequest = Literal["cpu", "gpu", "auto"]
ResolvedDeviceName = Literal["cpu", "cuda"]


class DeviceResolutionError(RuntimeError):
    """Raised when ``device: gpu`` is requested but CUDA is unavailable/unusable."""


@dataclass(frozen=True)
class ResolvedDevice:
    """Requested vs. actually-resolved device, recorded for reproducibility."""

    requested: DeviceRequest
    resolved: ResolvedDeviceName

    @property
    def torch_device(self) -> "object":
        import torch

        return torch.device(self.resolved)


def resolve_device(requested: DeviceRequest) -> ResolvedDevice:
    """Resolve ``requested`` into a concrete device, per spec section 18.

    - ``cpu``: always resolves to CPU.
    - ``gpu``: must successfully initialize CUDA (availability check plus an
      actual tensor allocation), otherwise raises :class:`DeviceResolutionError`
      so startup fails before the agent reports ready.
    - ``auto``: CUDA if available, else CPU.
    """
    import torch

    if requested == "cpu":
        return ResolvedDevice(requested=requested, resolved="cpu")

    if requested == "gpu":
        _require_cuda(torch)
        return ResolvedDevice(requested=requested, resolved="cuda")

    if requested == "auto":
        if torch.cuda.is_available():
            try:
                _require_cuda(torch)
                return ResolvedDevice(requested=requested, resolved="cuda")
            except DeviceResolutionError:
                return ResolvedDevice(requested=requested, resolved="cpu")
        return ResolvedDevice(requested=requested, resolved="cpu")

    raise ValueError(f"Unknown device request: {requested!r}")


def _require_cuda(torch_module: "object") -> None:
    if not torch_module.cuda.is_available():
        raise DeviceResolutionError(
            "device: gpu was requested but CUDA is not available on this machine"
        )
    try:
        # Force real allocation, not just availability reporting.
        probe = torch_module.zeros(1, device="cuda")
        del probe
    except Exception as exc:  # pragma: no cover - depends on hardware/driver state
        raise DeviceResolutionError(
            f"device: gpu was requested but CUDA initialization failed: {exc}"
        ) from exc
