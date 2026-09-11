"""PyTorch-native learning-rate schedulers for PPO training.

Replaces the earlier ad-hoc, MARLA-specific ``compute_learning_rate``
linear-only formula (research-alpha cleanup: the previous "3e-4 linear
decay" was a development convenience, never a scientifically frozen
choice -- see ``research/EXPERIMENT_PLAN.md``'s Stage 1). Every supported
schedule maps directly onto a standard ``torch.optim.lr_scheduler`` class
rather than a parallel MARLA-specific reimplementation, so scheduler
behavior, edge cases, and ``state_dict()``/``load_state_dict()`` semantics
are exactly PyTorch's own, not a custom approximation of them.

**Stepping semantics** (must stay consistent everywhere a scheduler is
built or stepped): call ``scheduler.step()`` exactly once per COMPLETED
PPO update, never once per minibatch/epoch and never once per environment
step. A scheduler is constructed once per training run (or reconstructed
+ ``load_state_dict``-restored once, on resume) at ``last_epoch=-1``;
PyTorch's own ``LRScheduler.__init__`` performs an implicit "step 0" that
sets the optimizer to the schedule's initial-epoch LR before any
``.step()`` call, so the FIRST PPO update always uses the schedule's
initial (epoch-0) rate with no special-casing needed at the call site.

**Horizon semantics**: schedulers whose behavior depends on a total
duration (``linear``, ``cosine``) are built with
``total_iters``/``T_max = max(compute_num_rollouts(ppo_config) - 1, 1)``
-- i.e. the number of ``.step()`` calls that will actually occur across
the FULL run (one fewer than the number of PPO updates, since there is no
step after the final update), derived from the run's GLOBAL environment-
step budget and effective batch size, never from optimizer-minibatch
counts or from how many rollouts remain in a particular resumed process
invocation (``compute_num_rollouts`` always reflects the run's full,
original horizon since it is a pure function of ``total_environment_steps``,
which does not change across a resume).
"""

from __future__ import annotations

import torch

from marla.config.models import (
    CosineSchedulerConfig,
    ExponentialSchedulerConfig,
    LinearSchedulerConfig,
    PPOConfig,
    StepSchedulerConfig,
    compute_num_rollouts,
)

LRScheduler = torch.optim.lr_scheduler.LRScheduler


def build_scheduler(optimizer: torch.optim.Optimizer, ppo_config: PPOConfig) -> LRScheduler:
    """Builds the scheduler named by ``ppo_config.optimizer.scheduler``,
    with its horizon (where applicable) derived from
    ``compute_num_rollouts(ppo_config)`` -- the actual number of PPO
    updates this run's ``total_environment_steps``/``effective_batch_size``
    imply, per the module docstring's "Horizon semantics".
    """
    scheduler_config = ppo_config.optimizer.scheduler
    total_updates = compute_num_rollouts(ppo_config)
    horizon = max(total_updates - 1, 1)

    if isinstance(scheduler_config, LinearSchedulerConfig):
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=scheduler_config.end_factor, total_iters=horizon
        )
    if isinstance(scheduler_config, CosineSchedulerConfig):
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=horizon, eta_min=scheduler_config.eta_min)
    if isinstance(scheduler_config, StepSchedulerConfig):
        return torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_config.step_size, gamma=scheduler_config.gamma)
    if isinstance(scheduler_config, ExponentialSchedulerConfig):
        return torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=scheduler_config.gamma)

    # ConstantSchedulerConfig (and, defensively, any future type this
    # module doesn't yet recognize by name -- fail via the type-narrowing
    # above rather than silently guessing "constant" for something new):
    if scheduler_config.type != "constant":
        raise ValueError(f"Unsupported scheduler config: {scheduler_config!r}")
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda _step: 1.0)


def current_learning_rate(optimizer: torch.optim.Optimizer) -> float:
    """The optimizer's current (single, shared) learning rate. MARLA does
    not use separate actor/critic parameter groups (spec: out of scope for
    this task) -- every param group shares one LR, so reading group 0 is
    exact, not an approximation.
    """
    return optimizer.param_groups[0]["lr"]
