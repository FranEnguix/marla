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


def test_wrong_bootstrap_value_at_a_boundary_changes_advantages_through_the_whole_segment():
    """Documents why a wrong V_{t+1} at a rollout-cutoff/truncation boundary
    (e.g. the recurrent-bootstrap-query bug: a bootstrap probe built from
    the wrong previous_query silently substitutes a different V_{t+1})
    matters scientifically, not just cosmetically: compute_gae() itself is
    unchanged and correct here -- this only demonstrates its known,
    intended sensitivity to its bootstrap_values input, using two
    different-but-plausible bootstrap values at the same boundary (standing
    in for a "query=True" vs "query=False" V(s_{t+1})), across a 3-step
    non-terminated window (rollout cutoff, not truncated -- values[-1]
    still needs a bootstrap per compute_gae's own contract)."""
    rewards = [1.0, 1.0, 1.0]
    values = [0.5, 0.5, 0.5]
    terminated = [False, False, False]
    truncated = [False, False, False]
    gamma, gae_lambda = 0.9, 0.95

    bootstrap_query_false = [None, None, 5.0]  # only the last (window-final) row needs one
    bootstrap_query_true = [None, None, 8.0]  # a different, equally plausible V(s_{t+1})

    adv_false, _ = compute_gae(rewards, values, terminated, truncated, bootstrap_query_false, gamma, gae_lambda)
    adv_true, _ = compute_gae(rewards, values, terminated, truncated, bootstrap_query_true, gamma, gae_lambda)

    # delta_t at the boundary itself: r_t + gamma * V_{t+1} - V_t -- directly
    # shifted by exactly gamma * (the bootstrap difference).
    delta_false = rewards[2] + gamma * bootstrap_query_false[2] - values[2]
    delta_true = rewards[2] + gamma * bootstrap_query_true[2] - values[2]
    assert math.isclose(adv_false[2], delta_false)
    assert math.isclose(adv_true[2], delta_true)
    assert not math.isclose(adv_false[2], adv_true[2])
    assert math.isclose(adv_true[2] - adv_false[2], gamma * (bootstrap_query_true[2] - bootstrap_query_false[2]))

    # And it propagates backward through gamma*lambda into every earlier
    # advantage of the same non-terminated segment -- not just the boundary
    # row itself (GAE's own recursive definition, exercised here, not
    # reimplemented): advantages[t] = delta_t + gamma*lambda*advantages[t+1].
    assert not math.isclose(adv_false[1], adv_true[1])
    assert not math.isclose(adv_false[0], adv_true[0])
    expected_diff_at_2 = gamma * (bootstrap_query_true[2] - bootstrap_query_false[2])
    assert math.isclose(adv_true[1] - adv_false[1], gamma * gae_lambda * expected_diff_at_2)
    assert math.isclose(adv_true[0] - adv_false[0], (gamma * gae_lambda) ** 2 * expected_diff_at_2)


def test_zero_gae_lambda_reduces_to_one_step_td_error():
    advantages, _ = compute_gae(
        rewards=[1.0, 1.0, 1.0], values=[0.5, 0.5, 0.5], terminated=[False, False, True],
        truncated=[False, False, False], bootstrap_values=[None, None, None], gamma=0.9, gae_lambda=0.0,
    )
    # with lambda=0, GAE == one-step TD error at every non-terminal step
    assert math.isclose(advantages[0], 1.0 + 0.9 * 0.5 - 0.5)
    assert math.isclose(advantages[1], 1.0 + 0.9 * 0.5 - 0.5)
