from pathlib import Path

from marla.environment.nasimemu_adapter import NasimEmuAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
UNI_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/uni.v2.yaml").resolve())


def make_adapter(scenario=SMALL_SCENARIO, max_episode_steps=20):
    return NasimEmuAdapter(
        scenario=scenario,
        max_episode_steps=max_episode_steps,
        completion_reward=1.0,
        premature_finish_penalty=-1.0,
    )


def test_reset_returns_at_least_one_host_and_finish_is_always_legal():
    adapter = make_adapter()
    state = adapter.reset(seed=1)
    assert len(state.host_addresses) >= 1
    assert state.step_idx == 0

    actions = adapter.legal_actions(state)
    assert any(a.is_finish for a in actions)


def test_same_seed_is_reproducible():
    adapter_a = make_adapter(scenario=UNI_SCENARIO)
    adapter_b = make_adapter(scenario=UNI_SCENARIO)

    state_a = adapter_a.reset(seed=42)
    state_b = adapter_b.reset(seed=42)

    assert state_a.host_addresses == state_b.host_addresses
    assert (state_a.raw_observation == state_b.raw_observation).all()


def test_different_seeds_can_differ():
    adapter = make_adapter(scenario=UNI_SCENARIO)
    state_a = adapter.reset(seed=1)
    adapter2 = make_adapter(scenario=UNI_SCENARIO)
    state_b = adapter2.reset(seed=2)
    # Not a strict guarantee for every possible pair, but true for these seeds
    # against this scenario generator; documents expected seed sensitivity.
    assert state_a.raw_observation.shape != state_b.raw_observation.shape or not (
        state_a.raw_observation == state_b.raw_observation
    ).all()


def test_step_advances_exactly_once_and_never_terminates_internally():
    adapter = make_adapter()
    state = adapter.reset(seed=1)
    actions = adapter.legal_actions(state)
    non_finish = next(a for a in actions if not a.is_finish)

    result = adapter.step(non_finish)
    assert result.terminated is False
    assert result.state.step_idx == state.step_idx + 1


def test_truncation_after_max_episode_steps():
    adapter = make_adapter(max_episode_steps=3)
    state = adapter.reset(seed=1)

    for expected_step in range(1, 4):
        actions = adapter.legal_actions(state)
        non_finish = next(a for a in actions if not a.is_finish)
        result = adapter.step(non_finish)
        state = result.state
        assert result.state.step_idx == expected_step
        if expected_step < 3:
            assert result.truncated is False
        else:
            assert result.truncated is True


def test_finish_terminates_without_advancing_nasim_step_idx():
    adapter = make_adapter()
    state = adapter.reset(seed=1)
    actions = adapter.legal_actions(state)
    finish = next(a for a in actions if a.is_finish)

    result = adapter.step(finish)
    assert result.terminated is True
    assert result.truncated is False
    assert result.state is None
    assert result.info["finish"] is True


def test_finish_reward_matches_objective_satisfaction():
    adapter = make_adapter(scenario=UNI_SCENARIO)
    state = adapter.reset(seed=7)
    assert adapter.objective_satisfied(state) is False  # uni.v2 is not solved at reset

    actions = adapter.legal_actions(state)
    finish = next(a for a in actions if a.is_finish)
    result = adapter.step(finish)

    assert result.info["objective_satisfied"] is False
    assert result.nasimemu_reward == -1.0  # premature_finish_penalty


def test_objective_satisfied_is_boolean():
    adapter = make_adapter()
    state = adapter.reset(seed=1)
    assert isinstance(adapter.objective_satisfied(state), bool)
