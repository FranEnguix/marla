from pathlib import Path

from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.nasimemu_adapter import NasimEmuAdapter

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())
UNI_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/uni.v2.yaml").resolve())


def make_adapter(scenario=SMALL_SCENARIO, max_episode_steps=20, premature_finish_penalty_per_remaining_target=0.0):
    return NasimEmuAdapter(
        scenario=scenario,
        max_episode_steps=max_episode_steps,
        completion_reward=1.0,
        premature_finish_penalty=-1.0,
        premature_finish_penalty_per_remaining_target=premature_finish_penalty_per_remaining_target,
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
    assert adapter.objective_satisfied() is False  # uni.v2 is not solved at reset

    actions = adapter.legal_actions(state)
    finish = next(a for a in actions if a.is_finish)
    result = adapter.step(finish)

    assert result.info["objective_satisfied"] is False
    assert result.nasimemu_reward == -1.0  # premature_finish_penalty


def test_objective_satisfied_is_boolean():
    adapter = make_adapter()
    adapter.reset(seed=1)
    assert isinstance(adapter.objective_satisfied(), bool)


def test_step_absorbs_a_nasimemu_action_space_assertion_error(monkeypatch):
    # Regression test: NASimEmu's own _translate_action() asserts that the
    # constructed action is a member of this episode's precomputed action
    # space -- a hard precondition, not the graceful "attempt failed, pay
    # the cost" its own step() applies to every other infeasible attempt.
    # A real 100k-step run crashed on exactly this after ~11,000 episodes
    # (a scenario-wide exploit that wasn't valid for one specific episode's
    # host arrangement); MARLA must absorb it like any other failed
    # attempt instead of letting the whole run die.
    adapter = make_adapter(max_episode_steps=20)
    state = adapter.reset(seed=1)
    actions = adapter.legal_actions(state)
    non_finish = next(a for a in actions if not a.is_finish)

    def _raise_assertion_error(*args, **kwargs):
        raise AssertionError("Failed to execute <fake action>")

    monkeypatch.setattr(adapter._env, "step", _raise_assertion_error)
    result = adapter.step(non_finish)

    assert result.terminated is False
    assert result.state is not None
    assert result.state.step_idx == state.step_idx + 1  # still advances, like a real step
    assert result.state.host_addresses == state.host_addresses  # unchanged: nothing actually ran
    assert result.nasimemu_reward < 0  # a real cost was still charged
    assert result.info["invalid_action"] is True
    assert result.info["action_id"] == non_finish.action_id


# --- Proportional (per-remaining-target) premature-FINISH penalty ---
# uni.v2.yaml seed=7 has exactly 2 sensitive targets, both not yet
# compromised at reset (asserted below rather than assumed), so these
# tests directly manipulate real HostVector.access to control exactly how
# many remain -- not a mock of NASimEmu's own state machinery.


def test_sensitive_target_status_all_remaining_at_reset():
    adapter = make_adapter(scenario=UNI_SCENARIO)
    adapter.reset(seed=7)
    total, remaining = adapter.sensitive_target_status()
    assert total == 2
    assert remaining == 2  # nothing compromised yet


def test_sensitive_target_status_root_access_reduces_remaining():
    adapter = make_adapter(scenario=UNI_SCENARIO)
    adapter.reset(seed=7)
    addresses = adapter._env.env.network.sensitive_addresses
    state = adapter._env.env.current_state

    state.get_host(addresses[0]).access = AccessLevel.ROOT
    total, remaining = adapter.sensitive_target_status()
    assert total == 2
    assert remaining == 1  # one target now satisfied

    state.get_host(addresses[1]).access = AccessLevel.ROOT
    total, remaining = adapter.sensitive_target_status()
    assert total == 2
    assert remaining == 0


def test_objective_satisfied_and_zero_remaining_agree_independent_of_finish():
    """Regression test (spec section 24): objective_satisfied() and
    sensitive_target_status()'s remaining count must always agree
    (remaining == 0 iff objective_satisfied() is True), and this is
    checked by directly mutating simulator access state -- never via
    FINISH, which performs no simulator action at all (see
    NasimEmuAdapter.step's own docstring) -- to prove the invariant holds
    from the simulator's own state alone, not from any FINISH-related
    bookkeeping. This is what lets rollout.py compute
    objective_became_satisfied/sensitive_targets_remaining identically for
    every step regardless of selected.is_finish.
    """
    adapter = make_adapter(scenario=UNI_SCENARIO)
    adapter.reset(seed=7)
    addresses = adapter._env.env.network.sensitive_addresses
    state = adapter._env.env.current_state

    assert adapter.objective_satisfied() is False
    total, remaining = adapter.sensitive_target_status()
    assert (remaining == 0) == adapter.objective_satisfied()

    state.get_host(addresses[0]).access = AccessLevel.ROOT
    assert adapter.objective_satisfied() is False  # one of two still missing
    _total, remaining = adapter.sensitive_target_status()
    assert (remaining == 0) == adapter.objective_satisfied()

    state.get_host(addresses[1]).access = AccessLevel.ROOT
    assert adapter.objective_satisfied() is True  # both now rooted -- no FINISH involved
    _total, remaining = adapter.sensitive_target_status()
    assert remaining == 0
    assert (remaining == 0) == adapter.objective_satisfied()


def test_sensitive_target_status_user_access_still_counts_as_remaining():
    """USER access must NOT satisfy a target -- only ROOT does."""
    adapter = make_adapter(scenario=UNI_SCENARIO)
    adapter.reset(seed=7)
    addresses = adapter._env.env.network.sensitive_addresses
    state = adapter._env.env.current_state

    state.get_host(addresses[0]).access = AccessLevel.USER
    total, remaining = adapter.sensitive_target_status()
    assert total == 2
    assert remaining == 2  # USER-only access does not reduce "remaining"


def test_finish_reward_pure_per_remaining_target_penalty_one_remaining():
    adapter = make_adapter(
        scenario=UNI_SCENARIO, premature_finish_penalty_per_remaining_target=-20.0
    )
    adapter._premature_finish_penalty = 0.0  # AAMAS-style: no fixed component
    state = adapter.reset(seed=7)
    addresses = adapter._env.env.network.sensitive_addresses
    adapter._env.env.current_state.get_host(addresses[0]).access = AccessLevel.ROOT  # 1 of 2 done

    finish = next(a for a in adapter.legal_actions(state) if a.is_finish)
    result = adapter.step(finish)

    assert result.info["objective_satisfied"] is False
    assert result.info["sensitive_targets_total"] == 2
    assert result.info["sensitive_targets_remaining"] == 1
    assert result.nasimemu_reward == -20.0


def test_finish_reward_pure_per_remaining_target_penalty_two_remaining():
    adapter = make_adapter(
        scenario=UNI_SCENARIO, premature_finish_penalty_per_remaining_target=-20.0
    )
    adapter._premature_finish_penalty = 0.0
    state = adapter.reset(seed=7)  # both sensitive targets still remaining

    finish = next(a for a in adapter.legal_actions(state) if a.is_finish)
    result = adapter.step(finish)

    assert result.info["sensitive_targets_remaining"] == 2
    assert result.nasimemu_reward == -40.0


def test_finish_reward_default_per_target_coefficient_matches_pre_existing_behavior():
    """Without opting in, an ordinary premature FINISH is unaffected by how
    many sensitive targets remain -- exactly the pre-existing fixed-penalty
    behavior (regression guard for the new feature's backward compatibility).
    """
    adapter = make_adapter(scenario=UNI_SCENARIO)  # per-target coefficient defaults to 0.0
    state = adapter.reset(seed=7)

    finish = next(a for a in adapter.legal_actions(state) if a.is_finish)
    result = adapter.step(finish)

    assert result.nasimemu_reward == -1.0  # premature_finish_penalty alone


def test_ordinary_non_finish_action_reward_unaffected_by_per_target_coefficient():
    """The per-target coefficient must only ever apply on a premature
    FINISH -- never as a per-step tax on ordinary actions."""
    adapter_plain = make_adapter(scenario=UNI_SCENARIO)
    adapter_penalized = make_adapter(scenario=UNI_SCENARIO, premature_finish_penalty_per_remaining_target=-20.0)

    state_plain = adapter_plain.reset(seed=7)
    state_penalized = adapter_penalized.reset(seed=7)
    non_finish_plain = next(a for a in adapter_plain.legal_actions(state_plain) if not a.is_finish)
    non_finish_penalized = next(a for a in adapter_penalized.legal_actions(state_penalized) if not a.is_finish)
    assert non_finish_plain.action_id == non_finish_penalized.action_id

    result_plain = adapter_plain.step(non_finish_plain)
    result_penalized = adapter_penalized.step(non_finish_penalized)
    assert result_plain.nasimemu_reward == result_penalized.nasimemu_reward


def test_step_absorbed_invalid_action_can_still_truncate_the_episode(monkeypatch):
    adapter = make_adapter(max_episode_steps=1)
    state = adapter.reset(seed=1)
    actions = adapter.legal_actions(state)
    non_finish = next(a for a in actions if not a.is_finish)

    monkeypatch.setattr(adapter._env, "step", lambda *a, **k: (_ for _ in ()).throw(AssertionError("x")))
    result = adapter.step(non_finish)

    assert result.truncated is True
