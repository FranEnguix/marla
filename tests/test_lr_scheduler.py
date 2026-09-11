"""PyTorch-native LR scheduler semantics (research-readiness alpha
cleanup, replacing the old ad-hoc linear-only ``compute_learning_rate``).
"""

from __future__ import annotations

import math

import pytest
import torch

from marla.config.models import (
    ConstantSchedulerConfig,
    CosineSchedulerConfig,
    ExponentialSchedulerConfig,
    LinearSchedulerConfig,
    OptimizerConfig,
    PPOConfig,
    StepSchedulerConfig,
)
from marla.learning.lr_scheduler import build_scheduler, current_learning_rate


def _ppo_config(scheduler, total_environment_steps=10_000, steps_per_env=1_000, num_envs=1) -> PPOConfig:
    return PPOConfig(
        total_environment_steps=total_environment_steps,
        steps_per_env=steps_per_env,
        num_envs=num_envs,
        epochs=4,
        minibatch_sequences=8,
        gamma=0.99,
        gae_lambda=0.95,
        clip_epsilon=0.2,
        value_coefficient=0.5,
        query_entropy_coefficient=0.01,
        action_entropy_coefficient=0.01,
        max_grad_norm=0.5,
        optimizer=OptimizerConfig(learning_rate=1e-3, scheduler=scheduler),
    )


def _optimizer(lr=1e-3) -> torch.optim.Optimizer:
    model = torch.nn.Linear(2, 2)
    return torch.optim.Adam(model.parameters(), lr=lr)


# 10 total_environment_steps / 1000 steps_per_env -> 10 PPO updates -> horizon = 9 steps.


def test_constant_scheduler_never_changes_lr():
    optimizer = _optimizer(1e-3)
    scheduler = build_scheduler(optimizer, _ppo_config(ConstantSchedulerConfig()))
    assert current_learning_rate(optimizer) == pytest.approx(1e-3)
    for _ in range(15):
        scheduler.step()
        assert current_learning_rate(optimizer) == pytest.approx(1e-3)


def test_linear_scheduler_starts_at_initial_rate_and_reaches_end_factor_at_final_update():
    optimizer = _optimizer(1e-3)
    scheduler = build_scheduler(optimizer, _ppo_config(LinearSchedulerConfig(end_factor=0.1)))
    assert current_learning_rate(optimizer) == pytest.approx(1e-3)
    for _ in range(9):  # 9 steps = the 9 update-boundaries across 10 total updates
        scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(1e-4)  # end_factor=0.1 * 1e-3


def test_linear_scheduler_end_factor_zero_matches_old_decay_to_zero_semantics():
    optimizer = _optimizer(1e-3)
    scheduler = build_scheduler(optimizer, _ppo_config(LinearSchedulerConfig(end_factor=0.0)))
    for _ in range(9):
        scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(0.0)


def test_linear_scheduler_never_negative_and_holds_after_horizon():
    optimizer = _optimizer(1e-3)
    scheduler = build_scheduler(optimizer, _ppo_config(LinearSchedulerConfig(end_factor=0.0)))
    for _ in range(50):  # far past the 9-step horizon
        scheduler.step()
        assert current_learning_rate(optimizer) >= 0.0
    assert current_learning_rate(optimizer) == pytest.approx(0.0)


def test_cosine_scheduler_starts_at_initial_rate_and_reaches_eta_min_at_final_update():
    optimizer = _optimizer(1e-3)
    scheduler = build_scheduler(optimizer, _ppo_config(CosineSchedulerConfig(eta_min=1e-5)))
    assert current_learning_rate(optimizer) == pytest.approx(1e-3)
    for _ in range(9):
        scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(1e-5, abs=1e-9)


def test_step_scheduler_decays_by_gamma_every_step_size_updates():
    optimizer = _optimizer(1e-3)
    scheduler = build_scheduler(optimizer, _ppo_config(StepSchedulerConfig(step_size=2, gamma=0.5)))
    assert current_learning_rate(optimizer) == pytest.approx(1e-3)
    for _ in range(2):
        scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(5e-4)
    for _ in range(2):
        scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(2.5e-4)


def test_exponential_scheduler_decays_by_gamma_every_update():
    optimizer = _optimizer(1e-3)
    scheduler = build_scheduler(optimizer, _ppo_config(ExponentialSchedulerConfig(gamma=0.9)))
    assert current_learning_rate(optimizer) == pytest.approx(1e-3)
    scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(9e-4)
    scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(8.1e-4)


def test_horizon_is_derived_from_global_environment_step_budget_not_minibatch_count():
    # 20_480 total steps / (4 envs * 512 steps_per_env) = 10 PPO updates -> horizon = 9.
    optimizer = _optimizer(1e-3)
    ppo = _ppo_config(LinearSchedulerConfig(end_factor=0.0), total_environment_steps=20_480, steps_per_env=512, num_envs=4)
    scheduler = build_scheduler(optimizer, ppo)
    for _ in range(9):
        scheduler.step()
    assert current_learning_rate(optimizer) == pytest.approx(0.0)
    # One step short of the horizon must NOT yet be fully decayed.
    optimizer2 = _optimizer(1e-3)
    scheduler2 = build_scheduler(optimizer2, ppo)
    for _ in range(8):
        scheduler2.step()
    assert current_learning_rate(optimizer2) > 0.0


def test_every_scheduler_type_produces_a_finite_positive_lr_across_its_horizon():
    configs = [
        ConstantSchedulerConfig(),
        LinearSchedulerConfig(end_factor=0.05),
        CosineSchedulerConfig(eta_min=1e-6),
        StepSchedulerConfig(step_size=3, gamma=0.5),
        ExponentialSchedulerConfig(gamma=0.95),
    ]
    for scheduler_config in configs:
        optimizer = _optimizer(1e-3)
        scheduler = build_scheduler(optimizer, _ppo_config(scheduler_config))
        for _ in range(20):
            scheduler.step()
            lr = current_learning_rate(optimizer)
            assert math.isfinite(lr)
            assert lr >= 0.0


def test_scheduler_state_dict_round_trips_and_resumes_the_exact_trajectory():
    optimizer_a = _optimizer(1e-3)
    ppo = _ppo_config(CosineSchedulerConfig(eta_min=1e-5))
    scheduler_a = build_scheduler(optimizer_a, ppo)
    for _ in range(4):
        scheduler_a.step()
    saved_state = scheduler_a.state_dict()

    # Simulate a resume: fresh optimizer + fresh scheduler, restored from
    # state_dict, with the optimizer's LR explicitly synced to the
    # restored schedule's current value -- exactly what
    # learning/checkpoint.py's load_checkpoint does (PyTorch schedulers
    # don't sync the optimizer's LR as a side effect of load_state_dict
    # alone, only on the next .step() call).
    optimizer_b = _optimizer(1e-3)
    scheduler_b = build_scheduler(optimizer_b, ppo)
    scheduler_b.load_state_dict(saved_state)
    optimizer_b.param_groups[0]["lr"] = scheduler_b.get_last_lr()[0]

    assert current_learning_rate(optimizer_b) == pytest.approx(current_learning_rate(optimizer_a))
    for _ in range(5):
        scheduler_a.step()
        scheduler_b.step()
        assert current_learning_rate(optimizer_b) == pytest.approx(current_learning_rate(optimizer_a))
    assert scheduler_a.state_dict()["last_epoch"] == scheduler_b.state_dict()["last_epoch"]
