from marla.environment.finish import compute_finish_reward


def test_finish_reward_when_objective_satisfied():
    assert compute_finish_reward(True, completion_reward=1.0, premature_finish_penalty=-1.0) == 1.0


def test_finish_reward_when_objective_not_satisfied():
    assert compute_finish_reward(False, completion_reward=1.0, premature_finish_penalty=-1.0) == -1.0


def test_finish_reward_uses_configured_values():
    assert compute_finish_reward(True, completion_reward=5.0, premature_finish_penalty=-2.5) == 5.0
    assert compute_finish_reward(False, completion_reward=5.0, premature_finish_penalty=-2.5) == -2.5


# --- Proportional (per-remaining-target) premature-FINISH penalty ---


def test_default_per_target_coefficient_preserves_old_fixed_penalty_semantics():
    """Omitting the new argument entirely (as every caller predating this
    feature does) must behave exactly as before: a pure fixed penalty,
    independent of how many targets remain.
    """
    assert compute_finish_reward(False, completion_reward=1.0, premature_finish_penalty=-1.0) == -1.0
    # Passing remaining_sensitive_targets without a non-zero coefficient
    # must not change anything either -- the term multiplies to zero.
    assert (
        compute_finish_reward(
            False, completion_reward=1.0, premature_finish_penalty=-1.0, remaining_sensitive_targets=7
        )
        == -1.0
    )


def test_aamas_style_pure_proportional_penalty_one_remaining_target():
    """AAMAS configuration: base penalty is 0.0, -20 per remaining target."""
    reward = compute_finish_reward(
        False,
        completion_reward=1.0,
        premature_finish_penalty=0.0,
        premature_finish_penalty_per_remaining_target=-20.0,
        remaining_sensitive_targets=1,
    )
    assert reward == -20.0


def test_aamas_style_pure_proportional_penalty_two_remaining_targets():
    reward = compute_finish_reward(
        False,
        completion_reward=1.0,
        premature_finish_penalty=0.0,
        premature_finish_penalty_per_remaining_target=-20.0,
        remaining_sensitive_targets=2,
    )
    assert reward == -40.0


def test_aamas_style_pure_proportional_penalty_three_remaining_targets():
    reward = compute_finish_reward(
        False,
        completion_reward=1.0,
        premature_finish_penalty=0.0,
        premature_finish_penalty_per_remaining_target=-20.0,
        remaining_sensitive_targets=3,
    )
    assert reward == -60.0


def test_base_plus_per_target_penalty_add_rather_than_replace():
    reward = compute_finish_reward(
        False,
        completion_reward=1.0,
        premature_finish_penalty=-1.0,
        premature_finish_penalty_per_remaining_target=-20.0,
        remaining_sensitive_targets=2,
    )
    assert reward == -41.0


def test_successful_finish_ignores_per_target_coefficient_entirely():
    """A satisfied objective always returns completion_reward, regardless
    of the per-target coefficient or remaining count passed in -- there are
    no remaining targets to penalize once the objective holds.
    """
    reward = compute_finish_reward(
        True,
        completion_reward=1.0,
        premature_finish_penalty=0.0,
        premature_finish_penalty_per_remaining_target=-20.0,
        remaining_sensitive_targets=3,
    )
    assert reward == 1.0
