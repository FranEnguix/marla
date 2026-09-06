"""Differential validation (spec section 8): for small, tractable fixtures,
cross-check MARLA's symbolic solvability checker against an *independent*,
brute-force exhaustive search over the REAL NASimEmu simulator's state
graph (not the symbolic abstraction) -- i.e. answering "does some real
action sequence reach ``goal_reached()``?" by literally trying every
reachable state, using NASimEmu's own ``Network.perform_action``.

This is test infrastructure only (never used by the production checker,
which must stay a closed-form symbolic proof, not brute force -- see
``marla.scenario.solvability``'s own module docstring on why sampling/
enumeration doesn't scale to real V2 randomization). It exists purely to
catch a systematic mismatch between the symbolic model and NASimEmu's
actual mechanics that a hand-reasoned model could miss. No NASimEmu file
is modified to make this possible.

Every fixture uses ``prob: 1.0`` on every exploit/privesc so transitions
are fully deterministic (NASimEmu's own randomness is ``np.random.rand() >
action.prob``, always false at ``prob=1.0``) -- required for a plain BFS
over concrete states to be exact and reproducible.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import nasimemu.nasim as nasim
from nasimemu.nasim.envs.environment import NASimEnv

from marla.scenario.solvability import check_solvability
from marla.scenario.spec import load_scenario_spec


def _exhaustive_ground_truth(scenario_path: Path, max_states: int = 200_000) -> bool:
    """Brute-force BFS over every state reachable from the initial state,
    using NASimEmu's own ``Network.perform_action`` for every transition.
    Returns True iff some reachable state has all sensitive hosts rooted.
    """
    scenario = nasim.load_scenario(str(scenario_path))
    env = NASimEnv(scenario, fully_obs=True, flat_actions=True)
    start_state = env.network.reset(env.current_state)

    visited = {start_state}
    queue = deque([start_state])
    explored = 0

    while queue:
        state = queue.popleft()
        explored += 1
        if explored > max_states:
            raise RuntimeError(
                f"Exhaustive search exceeded {max_states} states -- fixture is not small enough "
                "for differential testing; shrink it rather than raising this limit."
            )
        if env.network.all_sensitive_hosts_compromised(state):
            return True
        for action in env.action_space.actions:
            next_state, _result = env.network.perform_action(state, action)
            if next_state not in visited:
                visited.add(next_state)
                queue.append(next_state)

    return False


def _write(tmp_path: Path, content: str, name: str = "scenario.yaml") -> Path:
    # Deliberately a *V1*-format fixture (no ".v2." in the filename, so
    # ``nasim.load_scenario``/``marla.scenario.spec.load_scenario_spec``
    # both dispatch to the plain ``ScenarioLoader``, not ``ScenarioLoaderV2``.
    # V2's loader, when given an *explicit* (non-``_random``)
    # ``host_configurations`` dict, never converts its string tuple keys
    # ("(1, 0)") to real tuples (``ScenarioLoaderV2._parse_hosts`` uses the
    # raw dict key as-is; contrast ``ScenarioLoader._parse_hosts``, which
    # calls ``eval(address)`` -- verified directly: loading such a file
    # crashes inside ``Scenario._permute_subnets`` with "too many values to
    # unpack"). That is a pre-existing NASimEmu quirk in a code path real
    # bundled scenarios never exercise (they always use ``_random`` for V2),
    # not something to fix here (the task requires not modifying NASimEmu
    # unless unavoidable, and it is avoidable: V1 format supports exactly
    # the explicit/concrete host_configurations these differential fixtures
    # need, correctly).
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


# Shared 2-subnet static fixture builder: internet(0) -> subnet1(1, entry,
# 1 host) -> subnet2(2, sensitive, 1 host). Everything explicit/concrete
# so there is exactly one realization to compare -- differential testing
# is about the symbolic ABSTRACTION matching real mechanics, not about V2
# randomization (already covered exhaustively/symbolically by
# test_scenario_adversarial.py).
def _static_scenario(subnet2_os: str, subnet2_services: list[str], privesc_yaml: str) -> str:
    services = ["1_linux_proftpd", "2_windows_elasticsearch", "3_windows_wp_ninja"]
    services_yaml = "\n".join(f"  - {s}" for s in services)
    subnet2_services_yaml = "[" + ", ".join(subnet2_services) + "]"
    return (
        "subnets: [1, 1]\n"
        "topology: [[1, 1, 0],\n"
        "           [1, 1, 1],\n"
        "           [0, 1, 1]]\n"
        "sensitive_hosts:\n"
        '  "(2, 0)": 100\n'
        "os:\n  - linux\n  - windows\n"
        f"services:\n{services_yaml}\n"
        "processes:\n  - ~\n"
        "exploits:\n"
        "  e_proftpd:\n"
        "    service: 1_linux_proftpd\n"
        "    os: linux\n"
        "    prob: 1.0\n"
        "    cost: 1\n"
        "    access: user\n"
        "  e_elasticsearch:\n"
        "    service: 2_windows_elasticsearch\n"
        "    os: windows\n"
        "    prob: 1.0\n"
        "    cost: 1\n"
        "    access: root\n"
        "  e_wp_ninja:\n"
        "    service: 3_windows_wp_ninja\n"
        "    os: windows\n"
        "    prob: 1.0\n"
        "    cost: 1\n"
        "    access: user\n"
        f"{privesc_yaml}"
        "service_scan_cost: 1\nos_scan_cost: 1\nsubnet_scan_cost: 1\nprocess_scan_cost: 1\n"
        "host_configurations:\n"
        '  "(1, 0)": {os: linux, services: [1_linux_proftpd], processes: []}\n'
        f'  "(2, 0)": {{os: {subnet2_os}, services: {subnet2_services_yaml}, processes: []}}\n'
        "firewall:\n"
        "  (0, 1): [_all]\n"
        "  (1, 0): [_all]\n"
        "  (1, 2): [_all]\n"
        "  (2, 1): [_all]\n"
    )


_LINUX_PRIVESC = (
    "privilege_escalation:\n"
    "  pe_kernel:\n"
    "    process: ~\n    os: linux\n    prob: 1.0\n    cost: 1\n    access: root\n"
)
_NO_WINDOWS_PRIVESC = _LINUX_PRIVESC  # linux-only privesc, no windows one
_WITH_WINDOWS_PRIVESC = _LINUX_PRIVESC.rstrip("\n") + (
    "\n  pe_windows:\n    process: ~\n    os: windows\n    prob: 1.0\n    cost: 1\n    access: root\n"
)


def test_differential_known_unrootable_host_matches_real_simulator(tmp_path):
    """Windows sensitive host: wp_ninja (USER only) + no windows privesc --
    symbolic checker says PROVEN_UNSOLVABLE; ground truth (real simulator,
    exhaustive search) must independently agree that goal_reached() is
    NEVER achievable from any reachable state.
    """
    content = _static_scenario("windows", ["3_windows_wp_ninja"], _NO_WINDOWS_PRIVESC)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    symbolic_result = check_solvability(spec)
    ground_truth = _exhaustive_ground_truth(path)

    assert symbolic_result.status.value == "proven_unsolvable"
    assert ground_truth is False
    assert symbolic_result.universally_solvable == ground_truth


def test_differential_user_plus_privesc_matches_real_simulator(tmp_path):
    """Windows sensitive host: wp_ninja (USER) + a windows privesc to ROOT
    -- symbolic checker says PROVEN_SOLVABLE; ground truth must agree a
    real action sequence reaches goal_reached().
    """
    content = _static_scenario("windows", ["3_windows_wp_ninja"], _WITH_WINDOWS_PRIVESC)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    symbolic_result = check_solvability(spec)
    ground_truth = _exhaustive_ground_truth(path)

    assert symbolic_result.status.value == "proven_solvable"
    assert ground_truth is True
    assert symbolic_result.universally_solvable == ground_truth


def test_differential_direct_root_exploit_matches_real_simulator(tmp_path):
    """Windows sensitive host with elasticsearch -- direct ROOT exploit,
    no privesc needed at all.
    """
    content = _static_scenario("windows", ["2_windows_elasticsearch"], _NO_WINDOWS_PRIVESC)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    symbolic_result = check_solvability(spec)
    ground_truth = _exhaustive_ground_truth(path)

    assert symbolic_result.status.value == "proven_solvable"
    assert ground_truth is True
    assert symbolic_result.universally_solvable == ground_truth


def test_differential_linux_user_plus_privesc_matches_real_simulator(tmp_path):
    """Linux sensitive host: proftpd (USER) + linux kernel privesc -> ROOT."""
    content = _static_scenario("linux", ["1_linux_proftpd"], _LINUX_PRIVESC)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    symbolic_result = check_solvability(spec)
    ground_truth = _exhaustive_ground_truth(path)

    assert symbolic_result.status.value == "proven_solvable"
    assert ground_truth is True
    assert symbolic_result.universally_solvable == ground_truth


def test_differential_firewall_blocked_route_matches_real_simulator(tmp_path):
    """A locally-rootable sensitive host behind a firewall that blocks its
    only service -- symbolic checker must call this PROVEN_UNSOLVABLE (a
    network failure, not a host failure), and the real simulator's
    exhaustive search must independently confirm goal_reached() is never
    achievable (the host is never even exploitable, let alone rootable).
    """
    content = (
        "subnets: [1, 1]\n"
        "topology: [[1, 1, 0],\n"
        "           [1, 1, 1],\n"
        "           [0, 1, 1]]\n"
        "sensitive_hosts:\n"
        '  "(2, 0)": 100\n'
        "os:\n  - windows\n"
        "services:\n  - 2_windows_elasticsearch\n"
        "processes:\n  - ~\n"
        "exploits:\n"
        "  e_elasticsearch:\n"
        "    service: 2_windows_elasticsearch\n"
        "    os: windows\n    prob: 1.0\n    cost: 1\n    access: root\n"
        "privilege_escalation: {}\n"
        "service_scan_cost: 1\nos_scan_cost: 1\nsubnet_scan_cost: 1\nprocess_scan_cost: 1\n"
        "host_configurations:\n"
        '  "(1, 0)": {os: windows, services: [], processes: []}\n'
        '  "(2, 0)": {os: windows, services: [2_windows_elasticsearch], processes: []}\n'
        "firewall:\n"
        "  (0, 1): [_all]\n"
        "  (1, 0): [_all]\n"
        "  (1, 2): []\n"
        "  (2, 1): [_all]\n"
    )
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    symbolic_result = check_solvability(spec)
    ground_truth = _exhaustive_ground_truth(path)

    assert symbolic_result.status.value == "proven_unsolvable"
    assert not symbolic_result.host_rootability_failures
    assert len(symbolic_result.network_reachability_failures) == 1
    assert ground_truth is False
    assert symbolic_result.universally_solvable == ground_truth
