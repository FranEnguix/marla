"""Checkpoint save/load for the recurrent PPO policy + optimizer.

Also carries what's needed to genuinely *resume* training rather than
restart it (research/aamas2027's staged-training design): the next
episode seed a resumed ``RolloutCollector`` should continue from (instead
of repeating the same episode-seed sequence a fresh run would), and torch's
global RNG state at save time (so resumed stochastic action/query sampling
continues the same stream rather than starting a new one from
process-launch entropy). Both are optional/additive -- a checkpoint saved
before this existed loads exactly as before, just without resume support.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from marla.learning.recurrent_policy import RecurrentPolicy


@dataclass(frozen=True)
class CheckpointMetadata:
    update_count: int
    environment_steps: int
    config_hash: str
    # None for a checkpoint saved before resume support existed, or for a
    # baseline/non-resumable save path that never set them.
    next_episode_seed: int | None = None
    rng_state_restored: bool = False


def save_checkpoint(
    path: str | Path,
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    update_count: int,
    environment_steps: int,
    config_hash: str,
    next_episode_seed: int | None = None,
    rng_state: torch.Tensor | None = None,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "policy_state_dict": policy.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "update_count": update_count,
            "environment_steps": environment_steps,
            "config_hash": config_hash,
            "next_episode_seed": next_episode_seed,
            "rng_state": rng_state,
        },
        path,
    )


def peek_checkpoint_metadata(path: str | Path) -> CheckpointMetadata:
    """Read a checkpoint's metadata without needing a policy to load into --
    used to compute how many rollouts remain to a resumed run's target
    step count, before the policy has been constructed yet."""
    data = torch.load(Path(path), map_location="cpu", weights_only=False)
    return CheckpointMetadata(
        update_count=data["update_count"],
        environment_steps=data["environment_steps"],
        config_hash=data["config_hash"],
        next_episode_seed=data.get("next_episode_seed"),
    )


def load_checkpoint(
    path: str | Path,
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer | None = None,
    map_location: str | torch.device = "cpu",
    restore_rng_state: bool = False,
) -> CheckpointMetadata:
    """``restore_rng_state=True`` sets torch's *global* RNG stream to the
    saved state (``torch.set_rng_state``) -- only meaningful when actually
    resuming a training run; evaluation call sites must never pass this
    (deterministic eval doesn't sample at all, and mutating the global RNG
    as a side effect of "just loading a checkpoint" would be surprising).
    """
    data = torch.load(Path(path), map_location=map_location, weights_only=False)
    policy.load_state_dict(data["policy_state_dict"])
    if optimizer is not None and data.get("optimizer_state_dict") is not None:
        optimizer.load_state_dict(data["optimizer_state_dict"])

    rng_state = data.get("rng_state")
    rng_state_restored = False
    if restore_rng_state and rng_state is not None:
        torch.set_rng_state(rng_state.cpu().to(torch.uint8))
        rng_state_restored = True

    return CheckpointMetadata(
        update_count=data["update_count"],
        environment_steps=data["environment_steps"],
        config_hash=data["config_hash"],
        next_episode_seed=data.get("next_episode_seed"),
        rng_state_restored=rng_state_restored,
    )
