"""Checkpoint save/load for the recurrent PPO policy + optimizer + LR
scheduler.

Also carries what's needed to genuinely *resume* training rather than
restart it (research/aamas2027's staged-training design): the next
episode seed a resumed ``RolloutCollector`` should continue from (instead
of repeating the same episode-seed sequence a fresh run would), torch's
global RNG state at save time (so resumed stochastic action/query sampling
continues the same stream rather than starting a new one from
process-launch entropy), and the LR scheduler's own ``state_dict()`` (so a
resumed run's learning-rate trajectory continues exactly, not from a fresh
schedule -- see ``learning/lr_scheduler.py``). All three are
optional/additive: a checkpoint saved without them (e.g. a
non-resumable/evaluation-only save) still loads, just without resume
support for the missing piece.

Research-alpha policy: this is NOT trying to stay compatible with
checkpoints saved by older MARLA alpha versions. A checkpoint missing a
field this module now expects for RESUME (e.g. no ``scheduler_state_dict``)
fails loudly and specifically when a resume is actually requested, rather
than silently guessing a schedule state -- see ``load_checkpoint``'s
``restore_rng_state`` branch. Plain (non-resuming) evaluation loads of an
older checkpoint still work: the policy weights themselves are the only
thing evaluation needs.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from marla.learning.action_encoder import POLICY_REPRESENTATION_VERSION
from marla.learning.recurrent_policy import RecurrentPolicy


class PolicyRepresentationMismatchError(Exception):
    """Raised when loading a checkpoint saved under a different
    ``policy_representation_version`` -- e.g. one saved before the
    action/target compatibility features
    (:mod:`marla.environment.action_compatibility`) existed. Deliberately
    raised *before* ``load_state_dict`` so the failure is a clear,
    MARLA-specific message naming both versions, not a bare PyTorch tensor
    shape-mismatch stack trace. Never silently partially loads.
    """


class CheckpointResumeError(Exception):
    """Raised when a caller asks to *resume* training
    (``restore_rng_state=True``) from a checkpoint that lacks a field the
    current alpha requires for an exact resume (e.g. no
    ``scheduler_state_dict``, because it predates the LR-scheduler
    framework). Historical-checkpoint compatibility is explicitly not a
    goal (research-alpha policy) -- such a checkpoint can still be loaded
    for plain evaluation (``restore_rng_state=False``), just not resumed.
    """


@dataclass(frozen=True)
class CheckpointMetadata:
    update_count: int
    environment_steps: int
    config_hash: str
    # None for a checkpoint saved before resume support existed, or for a
    # baseline/non-resumable save path that never set them.
    next_episode_seed: int | None = None
    rng_state_restored: bool = False
    # None for a checkpoint saved before this field existed (implicitly
    # version 1: parameter-presence-only action features, no target/action
    # compatibility layer) -- see POLICY_REPRESENTATION_VERSION.
    policy_representation_version: int | None = None
    # Architecture-ablation identity (v2/v3-target/v3-full) -- the version
    # integer alone is not enough once two policies can share a version but
    # differ in which optional visible-progress components are enabled
    # (different RecurrentCore input width). None for a checkpoint saved
    # before these flags existed.
    visible_target_progress_enabled: bool | None = None
    visible_subnet_exploration_enabled: bool | None = None
    # Training-run hyperparameters, not policy-architecture properties --
    # the saved policy's *shape* is identical regardless of these values,
    # so none of them feed into POLICY_REPRESENTATION_VERSION or the
    # ablation-flag mismatch check below (a checkpoint saved with one value
    # is weight-compatible with a policy/run using a different one).
    # Recorded purely so a checkpoint can be identified later without
    # separately tracking down its run's config.yaml. None for a
    # checkpoint saved before the corresponding feature existed.
    critic_refinement_epochs: int | None = None
    num_envs: int | None = None
    steps_per_env: int | None = None
    # LR scheduler resume state (research-readiness alpha: replaces the old
    # ad-hoc linear-only schedule -- see learning/lr_scheduler.py).
    # `scheduler_type` / `scheduler_config` are informational (what kind of
    # schedule this run used, and its exact parameters, e.g. for an audit
    # or for rebuilding a matching scheduler from a fresh config);
    # `scheduler_state_dict` is the actual PyTorch state needed to resume
    # its trajectory exactly (last_epoch, base_lrs, ...); `learning_rate`
    # is the optimizer's LR at save time, for quick inspection without
    # reconstructing a scheduler.
    scheduler_type: str | None = None
    scheduler_config: dict[str, Any] | None = None
    scheduler_state_dict: dict[str, Any] | None = None
    learning_rate: float | None = None


def save_checkpoint(
    path: str | Path,
    policy: RecurrentPolicy,
    optimizer: torch.optim.Optimizer,
    update_count: int,
    environment_steps: int,
    config_hash: str,
    next_episode_seed: int | None = None,
    rng_state: torch.Tensor | None = None,
    critic_refinement_epochs: int | None = None,
    num_envs: int | None = None,
    steps_per_env: int | None = None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
    scheduler_config: dict[str, Any] | None = None,
) -> None:
    """``scheduler``/``scheduler_config`` are optional only for callers
    that genuinely have no scheduler (rare -- direct unit tests of the
    checkpoint format itself). Every production call site
    (``learning/trainer.py``, ``metrics/writer.py``) always passes the
    run's real scheduler and its resolved config, so a checkpoint written
    by a normal training run always carries full scheduler-resume state.
    """
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
            "policy_representation_version": POLICY_REPRESENTATION_VERSION,
            "visible_target_progress_enabled": policy.visible_target_progress_enabled,
            "visible_subnet_exploration_enabled": policy.visible_subnet_exploration_enabled,
            "critic_refinement_epochs": critic_refinement_epochs,
            "num_envs": num_envs,
            "steps_per_env": steps_per_env,
            "scheduler_type": scheduler_config.get("type") if scheduler_config else None,
            "scheduler_config": scheduler_config,
            "scheduler_state_dict": scheduler.state_dict() if scheduler is not None else None,
            "learning_rate": optimizer.param_groups[0]["lr"],
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
        policy_representation_version=data.get("policy_representation_version"),
        visible_target_progress_enabled=data.get("visible_target_progress_enabled"),
        visible_subnet_exploration_enabled=data.get("visible_subnet_exploration_enabled"),
        critic_refinement_epochs=data.get("critic_refinement_epochs"),
        num_envs=data.get("num_envs"),
        steps_per_env=data.get("steps_per_env"),
        scheduler_type=data.get("scheduler_type"),
        scheduler_config=data.get("scheduler_config"),
        scheduler_state_dict=data.get("scheduler_state_dict"),
        learning_rate=data.get("learning_rate"),
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
    It is also the signal this function uses to decide whether scheduler
    resume state is REQUIRED: a resume without it raises
    :class:`CheckpointResumeError` rather than silently starting the LR
    schedule over (research-alpha policy: no old-checkpoint compatibility
    shims for resume; a plain, non-resuming evaluation load of the same
    checkpoint still works).
    """
    data = torch.load(Path(path), map_location=map_location, weights_only=False)

    saved_version = data.get("policy_representation_version")
    if saved_version != POLICY_REPRESENTATION_VERSION:
        raise PolicyRepresentationMismatchError(
            f"Checkpoint {path} was saved under policy_representation_version="
            f"{saved_version!r} (None means a pre-versioning checkpoint, saved before the "
            f"action/target compatibility features existed), but this build of MARLA uses "
            f"version {POLICY_REPRESENTATION_VERSION}. The two are not weight-compatible "
            "(ActionEncoder's input width differs) and must never be silently partially "
            "loaded. This checkpoint is historical/pilot-only for the current architecture; "
            "resume/evaluate it with a matching MARLA version instead."
        )

    # A version match alone is not sufficient once two policies can share a
    # version but differ in the v2/v3-target/v3-full ablation flags -- those
    # change RecurrentCore's/GraphEncoder's input width just as much as a
    # version bump would. Checked before any load_state_dict call, same as
    # the version check above.
    saved_target_progress = data.get("visible_target_progress_enabled")
    saved_subnet_exploration = data.get("visible_subnet_exploration_enabled")
    if (
        saved_target_progress != policy.visible_target_progress_enabled
        or saved_subnet_exploration != policy.visible_subnet_exploration_enabled
    ):
        raise PolicyRepresentationMismatchError(
            f"Checkpoint {path} was saved with visible_target_progress_enabled="
            f"{saved_target_progress!r}, visible_subnet_exploration_enabled="
            f"{saved_subnet_exploration!r} (None means a checkpoint saved before these flags "
            f"existed), but the policy being loaded into has visible_target_progress_enabled="
            f"{policy.visible_target_progress_enabled!r}, visible_subnet_exploration_enabled="
            f"{policy.visible_subnet_exploration_enabled!r}. These are architecture-ablation "
            "flags (v2/v3-target/v3-full) that change RecurrentCore's/GraphEncoder's "
            "input width -- the two are not weight-compatible and must never be silently "
            "partially loaded. Load with a policy configured for the same ablation instead."
        )

    scheduler_state_dict = data.get("scheduler_state_dict")
    if restore_rng_state and scheduler_state_dict is None:
        raise CheckpointResumeError(
            f"Checkpoint {path} has no scheduler_state_dict -- it predates the LR-scheduler "
            "framework (research-alpha policy: resuming from a pre-scheduler checkpoint is not "
            "supported, to avoid silently restarting the LR schedule's trajectory). Load it for "
            "plain evaluation instead (restore_rng_state=False), or resume from a checkpoint "
            "saved by the current MARLA version."
        )

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
        policy_representation_version=saved_version,
        visible_target_progress_enabled=saved_target_progress,
        visible_subnet_exploration_enabled=saved_subnet_exploration,
        critic_refinement_epochs=data.get("critic_refinement_epochs"),
        num_envs=data.get("num_envs"),
        steps_per_env=data.get("steps_per_env"),
        scheduler_type=data.get("scheduler_type"),
        scheduler_config=data.get("scheduler_config"),
        scheduler_state_dict=scheduler_state_dict,
        learning_rate=data.get("learning_rate"),
    )
