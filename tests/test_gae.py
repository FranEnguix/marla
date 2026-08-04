import math

import pytest

from marla.learning.gae import compute_gae


def test_single_terminated_step():
    advantages, returns = compute_gae(
        rewards=[1.0], values=[2.0], terminated=[True], truncated=[False],
        bootstrap_values=[None], gamma=0.9, gae_lambda=0.95,
    )
    assert math.isclose(advantages[0], -1.0)
    assert math.isclose(returns[0], 1.0)


def test_single_truncated_step_uses_bootstrap():
    advantages, returns = compute_gae(
        rewards=[1.0], values=[2.0], terminated=[False], truncated=[True],
        bootstrap_values=[5.0], gamma=0.9, gae_lambda=0.95,
    )
    expected_delta = 1.0 + 0.9 * 5.0 - 2.0
    assert math.isclose(advantages[0], expected_delta)
    assert math.isclose(returns[0], expected_delta + 2.0)


def test_last_row_of_window_without_termination_requires_bootstrap():
    advantages, returns = compute_gae(
        rewards=[1.0], values=[2.0], terminated=[False], truncated=[False],
        bootstrap_values=[5.0], gamma=0.9, gae_lambda=0.95,
    )
    expected_delta = 1.0 + 0.9 * 5.0 - 2.0
    assert math.isclose(advantages[0], expected_delta)


def test_missing_bootstrap_value_raises():
    with pytest.raises(ValueError):
        compute_gae(
            rewards=[1.0], values=[2.0], terminated=[False], truncated=[True],
            bootstrap_values=[None], gamma=0.9, gae_lambda=0.95,
        )
    with pytest.raises(ValueError):
        compute_gae(
            rewards=[1.0], values=[2.0], terminated=[False], truncated=[False],
            bootstrap_values=[None], gamma=0.9, gae_lambda=0.95,
        )


def test_mismatched_lengths_raise():
    with pytest.raises(ValueError):
        compute_gae(
            rewards=[1.0, 2.0], values=[1.0], terminated=[False, True],
            truncated=[False, False], bootstrap_values=[None, None], gamma=0.9, gae_lambda=0.95,
        )


def test_two_step_episode_undiscounted():
    advantages, returns = compute_gae(
        rewards=[1.0, 1.0], values=[0.0, 0.0], terminated=[False, True],
        truncated=[False, False], bootstrap_values=[None, None], gamma=1.0, gae_lambda=1.0,
    )
    assert math.isclose(advantages[1], 1.0)
    assert math.isclose(advantages[0], 2.0)
    assert math.isclose(returns[0], 2.0)
    assert math.isclose(returns[1], 1.0)


def test_episode_boundary_resets_accumulation_mid_buffer():
    """A second episode packed into the same buffer must not see the first
    episode's future rewards leak into its advantages."""
    rewards = [1.0, 1.0, 10.0, 10.0]
    values = [0.0, 0.0, 0.0, 0.0]
    terminated = [False, True, False, True]
    truncated = [False, False, False, False]
    bootstrap_values = [None, None, None, None]

    advantages, _ = compute_gae(rewards, values, terminated, truncated, bootstrap_values, gamma=1.0, gae_lambda=1.0)

    # episode 1 (indices 0-1) must be unaffected by episode 2's huge rewards
    assert math.isclose(advantages[0], 2.0)
    assert math.isclose(advantages[1], 1.0)
    assert math.isclose(advantages[2], 20.0)
    assert math.isclose(advantages[3], 10.0)


def test_zero_gae_lambda_reduces_to_one_step_td_error():
    advantages, _ = compute_gae(
        rewards=[1.0, 1.0, 1.0], values=[0.5, 0.5, 0.5], terminated=[False, False, True],
        truncated=[False, False, False], bootstrap_values=[None, None, None], gamma=0.9, gae_lambda=0.0,
    )
    # with lambda=0, GAE == one-step TD error at every non-terminal step
    assert math.isclose(advantages[0], 1.0 + 0.9 * 0.5 - 0.5)
    assert math.isclose(advantages[1], 1.0 + 0.9 * 0.5 - 0.5)
