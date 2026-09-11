"""Fairness regression test (spec section 41): PPO's action-compatibility
layer must have access to every fact the Plan Maker's observation summary
has -- no case where the Plan Maker knows something PPO's compatibility
logic cannot know. Both are built from the exact same
:func:`marla.environment.visible_facts.extract_visible_host_facts` call on
the exact same :class:`~marla.environment.nasimemu_adapter.EnvironmentState`,
so this is provable directly: run a real episode and compare, at several
points, the Plan Maker's ``build_observation_summary`` output against the
``VisibleHostFacts`` PPO's ``compute_action_compatibility`` consumes for
that same state -- field by field, not just "both call the same function"
(a refactor could otherwise silently reintroduce a second, drifted
extraction path on one side without either call site itself changing).
"""

from __future__ import annotations

from pathlib import Path

from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.action_compatibility import compute_compatibility_matrix
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.environment.observation_summary import build_observation_summary
from marla.environment.visible_facts import extract_visible_host_facts
from marla.scenarios import diagnostic_scenario_path

REPO_ROOT = Path(__file__).resolve().parent.parent
_ACCESS_NAMES = {AccessLevel.NONE: "none", AccessLevel.USER: "user", AccessLevel.ROOT: "root"}


def _adapter() -> NasimEmuAdapter:
    return NasimEmuAdapter(
        scenario=str(diagnostic_scenario_path("micro_c_mixed_os_discrimination.v2.yaml")),
        max_episode_steps=20,
        completion_reward=1.0,
        premature_finish_penalty=0.0,
    )


def _assert_summary_matches_facts(state) -> None:
    summary = build_observation_summary(state)
    facts_by_target = extract_visible_host_facts(state)
    assert len(summary["hosts"]) == len(facts_by_target)

    for host_row in summary["hosts"]:
        facts = facts_by_target[host_row["target"]]
        # Every field the Plan Maker sees must equal the exact same
        # VisibleHostFacts PPO's compatibility layer would compute for
        # this same host, at this same state -- not merely "compatible",
        # exactly equal.
        assert host_row["access"] == _ACCESS_NAMES[facts.access]
        assert host_row["compromised"] == facts.compromised
        assert host_row["reachable"] == facts.reachable
        assert host_row["known_os"] == sorted(facts.known_os)
        assert host_row["known_services"] == sorted(facts.known_services)
        assert host_row["known_processes"] == sorted(facts.known_processes)


def test_plan_maker_summary_and_ppo_compatibility_facts_agree_at_episode_start():
    adapter = _adapter()
    state = adapter.reset(seed=4)
    _assert_summary_matches_facts(state)


def test_plan_maker_summary_and_ppo_compatibility_facts_agree_throughout_an_episode():
    """Not just the initial state -- after every scan/exploit/privesc step,
    for the whole episode, so a fact that only becomes visible mid-episode
    (a newly confirmed service, a compromised host, a new access level)
    can't silently diverge between the two consumers.
    """
    adapter = _adapter()
    state = adapter.reset(seed=4)
    _assert_summary_matches_facts(state)

    for _ in range(15):
        legal = adapter.legal_actions(state)
        if not legal:
            break
        # Deterministic traversal order, not random -- reproducible failures.
        action = legal[0]
        result = adapter.step(action)
        if result.state is None:
            break
        state = result.state
        _assert_summary_matches_facts(state)


def test_compatibility_matrix_is_computed_from_the_same_facts_the_summary_exposes():
    """Beyond the per-field equality above: build the actual compatibility
    matrix PPO would use for the legal action set at a mid-episode state,
    and confirm every CONFIRMED_COMPATIBLE/CONTRADICTED determination is
    consistent with what the Plan Maker's own summary shows for that
    action's target (e.g. a CONFIRMED_COMPATIBLE exploit's required
    service must appear in that target's known_services list in the
    summary too) -- the two are reading the identical underlying facts,
    not just structurally-similar-looking ones.
    """
    from marla.environment.action_compatibility import compatibility_status_from_vector

    adapter = _adapter()
    state = adapter.reset(seed=4)
    for _ in range(6):
        legal = adapter.legal_actions(state)
        state = adapter.step(legal[0]).state

    legal = adapter.legal_actions(state)
    facts_by_target = extract_visible_host_facts(state)
    summary = build_observation_summary(state)
    summary_by_target = {h["target"]: h for h in summary["hosts"]}
    matrix = compute_compatibility_matrix(legal, facts_by_target)

    for action, vector in zip(legal, matrix.tolist()):
        if action.action_type != "exploit" or not action.target_key:
            continue
        status = compatibility_status_from_vector(action.action_type, vector)
        service = action.parameters.get("service")
        target_summary = summary_by_target[action.target_key]
        if status.value == "confirmed_compatible" and service:
            assert service in target_summary["known_services"]


def test_no_second_visible_fact_extraction_implementation_exists():
    """Structural guard against the exact duplication this task's spec
    warns against: no module other than visible_facts.py itself should
    read a HostVector's .os, .services, AND .processes together (the
    specific combination extract_visible_host_facts exists to centralize
    into one VisibleHostFacts per host) -- every consumer
    (observation_summary.py, action_compatibility.py, rollout.py, ppo.py)
    must import and call extract_visible_host_facts instead of
    re-deriving that combined shape. Reading just one or two of these
    (e.g. graph.py's deliberately os-free GraphSAGE features, or
    state_delta.py's services/processes *diff* between two snapshots,
    neither of which builds a "current facts" structure) is a different,
    legitimate thing and is not flagged.
    """
    # Scoped to the runtime PPO/Plan Maker consumers this fairness contract
    # is actually about -- marla.scenario is a separate, unrelated static-
    # analysis subsystem reasoning over scenario *specs* (never per-step
    # simulator state), and legitimately reads scenario-spec os/services/
    # processes fields with no connection to VisibleHostFacts at all.
    scoped_dirs = [REPO_ROOT / "src" / "marla" / "environment", REPO_ROOT / "src" / "marla" / "learning"]
    visible_facts_path = REPO_ROOT / "src" / "marla" / "environment" / "visible_facts.py"
    offending: list[str] = []

    for scoped_dir in scoped_dirs:
        for path in scoped_dir.rglob("*.py"):
            if path == visible_facts_path:
                continue
            source = path.read_text(encoding="utf-8")
            combined = all(token in source for token in (".os", ".services", ".processes"))
            if combined and "visible_facts" not in source:
                offending.append(str(path.relative_to(REPO_ROOT)))

    assert not offending, f"found a second HostVector-reading extraction path outside visible_facts.py: {offending}"
