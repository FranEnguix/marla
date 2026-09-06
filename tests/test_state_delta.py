"""compute_state_delta correctness (spec section 3's observable state-change
fields).

Regression coverage for a real bug (found by cross-validating state_delta's
output against environment/observation_summary.py's independently-derived
per-host dicts on a real scenario): the original implementation compared
``before``/``after`` host rows by *array position*
(``enumerate(before.host_addresses)``), which silently misses every host
address that only exists in ``after`` -- exactly what happens whenever a
scan reveals a brand-new subnet. NASimEmu appends wholly new host rows to
``EnvironmentState.host_rows``/``host_addresses`` mid-episode in that case
(verified empirically: a 3-host snapshot growing to 6 hosts, the 3 new rows
arriving already ``discovered=True``), rather than flipping a ``discovered``
bit on a row that was present all along. A positional comparison never even
looks at the new rows, so ``new_hosts_discovered``/``new_subnets_discovered``
were silently 0 for this (very common -- most bundled scenarios randomize
host counts per subnet) case.
"""

from __future__ import annotations

from pathlib import Path

from nasimemu.nasim.envs.host_vector import HostVector
from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.nasimemu_adapter import EnvironmentState, NasimEmuAdapter
from marla.environment.state_delta import AccessGain, compute_state_delta

REPO_ROOT = Path(__file__).resolve().parent.parent
SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_user_three_subnets.v2.yaml").resolve())


def _make_adapter() -> NasimEmuAdapter:
    return NasimEmuAdapter(scenario=SCENARIO, max_episode_steps=100, completion_reward=1.0, premature_finish_penalty=-1.0)


def test_no_change_between_identical_snapshots():
    adapter = _make_adapter()
    state = adapter.reset(seed=1)
    delta = compute_state_delta(state, state)
    assert delta.new_hosts_discovered == 0
    assert delta.new_subnets_discovered == 0
    assert delta.new_services_confirmed == 0
    assert delta.new_processes_confirmed == 0
    assert delta.access_gain == AccessGain.NONE


def test_finish_transition_with_no_next_state_is_a_no_op():
    adapter = _make_adapter()
    state = adapter.reset(seed=1)
    delta = compute_state_delta(state, None)
    assert delta == compute_state_delta(state, None)  # deterministic
    assert delta.new_hosts_discovered == 0
    assert delta.access_gain == AccessGain.NONE


def test_regression_new_subnet_appended_mid_episode_is_counted():
    """The exact bug scenario: scan/exploit/privesc on the entry subnet's
    hosts (no new rows), then a subnet_scan that reveals a new subnet,
    appending wholly new host rows already marked discovered.
    """
    adapter = _make_adapter()
    state = adapter.reset(seed=1)
    action_order = ["subnet_scan", "os_scan", "service_scan", "process_scan", "exploit", "privilege_escalation"]

    deltas = []
    for step in range(7):
        legal = adapter.legal_actions(state)
        non_finish = [a for a in legal if not a.is_finish]
        action_type = action_order[step % len(action_order)]
        candidates = [a for a in non_finish if a.action_type == action_type]
        action = candidates[0] if candidates else non_finish[step % len(non_finish)]

        before_host_count = len(state.host_addresses)
        transition = adapter.step(action)
        if transition.state is None:
            break
        delta = compute_state_delta(state, transition.state)
        deltas.append((action.action_type, before_host_count, len(transition.state.host_addresses), delta))
        state = transition.state
        if transition.terminated or transition.truncated:
            break

    # At least one step must have grown the host-row array (a new subnet
    # revealed) -- if this scenario's random layout ever stops producing
    # that within 7 fixed-order steps, the test fixture itself needs
    # revisiting, not the assertion below.
    growth_steps = [(t, before, after, d) for t, before, after, d in deltas if after > before]
    assert growth_steps, "fixture assumption broke: no host-row growth observed in 7 steps"

    for action_type, before_count, after_count, delta in growth_steps:
        new_rows = after_count - before_count
        assert delta.new_hosts_discovered >= new_rows, (
            f"{action_type}: array grew by {new_rows} rows but "
            f"new_hosts_discovered={delta.new_hosts_discovered} (positional-comparison bug would report 0 here)"
        )
        assert delta.new_subnets_discovered >= 1


def _real_host_row(seed: int = 1):
    """A real, correctly-initialized HostVector row from an actual reset --
    HostVector's class-level layout (address_space_bounds, service/process
    index maps) is only populated by real scenario vectorization, so a
    hand-built synthetic row would need to reverse-engineer that layout;
    copying a real one and mutating it via HostVector's own setters/index
    maps is both correct and scenario-agnostic.
    """
    adapter = _make_adapter()
    state = adapter.reset(seed=seed)
    return state, HostVector(state.host_rows[0])


def _set_service(host: HostVector, name: str, value: bool) -> None:
    idx = HostVector._get_service_idx(HostVector.service_idx_map[name])
    host.vector[idx] = 1.0 if value else 0.0


def _set_process(host: HostVector, name: str, value: bool) -> None:
    idx = HostVector._get_process_idx(HostVector.process_idx_map[name])
    host.vector[idx] = 1.0 if value else 0.0


def test_existing_host_service_confirmation_is_detected():
    state, host = _real_host_row()
    before_row = host.copy()
    service_name = next(iter(HostVector.service_idx_map))
    _set_service(before_row, service_name, False)
    after_row = before_row.copy()
    _set_service(after_row, service_name, True)

    before = EnvironmentState(
        raw_observation=None, host_rows=[before_row.numpy()], host_addresses=[state.host_addresses[0]],
    )
    after = EnvironmentState(
        raw_observation=None, host_rows=[after_row.numpy()], host_addresses=[state.host_addresses[0]],
    )
    delta = compute_state_delta(before, after)
    assert delta.new_services_confirmed == 1
    assert delta.new_processes_confirmed == 0
    assert delta.new_hosts_discovered == 0
    assert delta.access_gain == AccessGain.NONE


def test_existing_host_process_confirmation_is_detected():
    state, host = _real_host_row()
    before_row = host.copy()
    process_name = next(iter(HostVector.process_idx_map), None)
    if process_name is None:
        return  # this scenario has no processes defined -- nothing to test here
    _set_process(before_row, process_name, False)
    after_row = before_row.copy()
    _set_process(after_row, process_name, True)

    before = EnvironmentState(raw_observation=None, host_rows=[before_row.numpy()], host_addresses=[state.host_addresses[0]])
    after = EnvironmentState(raw_observation=None, host_rows=[after_row.numpy()], host_addresses=[state.host_addresses[0]])
    delta = compute_state_delta(before, after)
    assert delta.new_processes_confirmed == 1


def test_access_gain_none_to_user():
    state, host = _real_host_row()
    before_row = host.copy()
    before_row.access = int(AccessLevel.NONE)
    after_row = before_row.copy()
    after_row.access = int(AccessLevel.USER)

    before = EnvironmentState(raw_observation=None, host_rows=[before_row.numpy()], host_addresses=[state.host_addresses[0]])
    after = EnvironmentState(raw_observation=None, host_rows=[after_row.numpy()], host_addresses=[state.host_addresses[0]])
    assert compute_state_delta(before, after).access_gain == AccessGain.USER


def test_access_gain_user_to_root():
    state, host = _real_host_row()
    before_row = host.copy()
    before_row.access = int(AccessLevel.USER)
    after_row = before_row.copy()
    after_row.access = int(AccessLevel.ROOT)

    before = EnvironmentState(raw_observation=None, host_rows=[before_row.numpy()], host_addresses=[state.host_addresses[0]])
    after = EnvironmentState(raw_observation=None, host_rows=[after_row.numpy()], host_addresses=[state.host_addresses[0]])
    assert compute_state_delta(before, after).access_gain == AccessGain.ROOT


def test_access_gain_none_directly_to_root():
    # Some exploits grant root directly (spec's rationale for AccessGain
    # being an achieved-level enum, not a delta-of-one boolean).
    state, host = _real_host_row()
    before_row = host.copy()
    before_row.access = int(AccessLevel.NONE)
    after_row = before_row.copy()
    after_row.access = int(AccessLevel.ROOT)

    before = EnvironmentState(raw_observation=None, host_rows=[before_row.numpy()], host_addresses=[state.host_addresses[0]])
    after = EnvironmentState(raw_observation=None, host_rows=[after_row.numpy()], host_addresses=[state.host_addresses[0]])
    assert compute_state_delta(before, after).access_gain == AccessGain.ROOT


def test_access_cannot_decrease_is_reported_as_no_gain():
    state, host = _real_host_row()
    before_row = host.copy()
    before_row.access = int(AccessLevel.ROOT)
    after_row = before_row.copy()
    after_row.access = int(AccessLevel.USER)  # never happens in real NASimEmu, but the function must not misreport it as a gain

    before = EnvironmentState(raw_observation=None, host_rows=[before_row.numpy()], host_addresses=[state.host_addresses[0]])
    after = EnvironmentState(raw_observation=None, host_rows=[after_row.numpy()], host_addresses=[state.host_addresses[0]])
    assert compute_state_delta(before, after).access_gain == AccessGain.NONE


def test_wholly_new_host_address_counts_as_discovered_with_its_own_facts():
    """A host address present in `after` but entirely absent from `before`
    -- the exact shape of the real regression above, isolated as a unit
    test independent of any particular scenario's random layout.
    """
    state, host = _real_host_row()
    existing_row = host.copy()
    new_row = host.copy()
    new_row.access = int(AccessLevel.USER)
    service_name = next(iter(HostVector.service_idx_map))
    _set_service(new_row, service_name, True)

    existing_address = state.host_addresses[0]
    new_address = (existing_address[0] + 1, 0)  # a different subnet entirely

    before = EnvironmentState(raw_observation=None, host_rows=[existing_row.numpy()], host_addresses=[existing_address])
    after = EnvironmentState(
        raw_observation=None,
        host_rows=[existing_row.numpy(), new_row.numpy()],
        host_addresses=[existing_address, new_address],
    )
    delta = compute_state_delta(before, after)
    assert delta.new_hosts_discovered == 1
    assert delta.new_subnets_discovered == 1
    assert delta.new_services_confirmed == 1  # the new host's own already-true service counts
    assert delta.access_gain == AccessGain.USER  # the new host's own access level
