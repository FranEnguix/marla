from marla.environment.finish import compute_finish_reward


def test_finish_reward_when_objective_satisfied():
    assert compute_finish_reward(True, completion_reward=1.0, premature_finish_penalty=-1.0) == 1.0


def test_finish_reward_when_objective_not_satisfied():
    assert compute_finish_reward(False, completion_reward=1.0, premature_finish_penalty=-1.0) == -1.0


def test_finish_reward_uses_configured_values():
    assert compute_finish_reward(True, completion_reward=5.0, premature_finish_penalty=-2.5) == 5.0
    assert compute_finish_reward(False, completion_reward=5.0, premature_finish_penalty=-2.5) == -2.5
