import pytest

from marla.knowledge.retriever import (
    KnowledgeBase,
    KnowledgeRule,
    compute_observation_flags,
    load_knowledge_base,
    resolve_knowledge_path,
    retrieve_rules,
)


def test_resolve_package_reference_points_at_shipped_file():
    path = resolve_knowledge_path("package://marla/knowledge/nasimemu_rules.yaml")
    assert path.is_file()
    assert path.name == "nasimemu_rules.yaml"


def test_resolve_rejects_unsupported_package_reference():
    with pytest.raises(ValueError):
        resolve_knowledge_path("package://not-marla/foo.yaml")


def test_load_shipped_knowledge_base():
    kb = load_knowledge_base("package://marla/knowledge/nasimemu_rules.yaml")
    assert kb.version == "nasimemu-rules-v1"
    assert len(kb.rules) > 0
    ids = [r.id for r in kb.rules]
    assert len(ids) == len(set(ids))  # unique rule IDs


def test_shipped_knowledge_base_tells_finish_to_score_high_once_objective_complete():
    kb = load_knowledge_base("package://marla/knowledge/nasimemu_rules.yaml")
    obs = {"hosts": [], "sensitive_hosts_total": 1, "sensitive_hosts_with_root_access": 1}
    matched = retrieve_rules(kb, obs, {"finish"})
    assert "finish-when-all-sensitive-hosts-captured" in [r.id for r in matched]


def _host(access="none", reachable=True, known_services=0, known_processes=0, is_objective_target=False):
    return {
        "target": "host-1-0",
        "access": access,
        "reachable": reachable,
        "known_services": known_services,
        "known_processes": known_processes,
        "is_objective_target": is_objective_target,
    }


def test_services_unknown_flag():
    obs = {"hosts": [_host(reachable=True, known_services=0)]}
    assert "services_unknown" in compute_observation_flags(obs)

    obs2 = {"hosts": [_host(reachable=True, known_services=3)]}
    assert "services_unknown" not in compute_observation_flags(obs2)


def test_user_access_present_flag():
    assert "user_access_present" in compute_observation_flags({"hosts": [_host(access="user")]})
    assert "user_access_present" in compute_observation_flags({"hosts": [_host(access="root")]})
    assert "user_access_present" not in compute_observation_flags({"hosts": [_host(access="none")]})


def test_root_access_missing_flag():
    assert "root_access_missing" in compute_observation_flags({"hosts": [_host(access="user")]})
    assert "root_access_missing" not in compute_observation_flags({"hosts": [_host(access="root")]})


def test_empty_observation_has_no_positive_flags_but_root_missing_is_vacuously_true():
    flags = compute_observation_flags({"hosts": []})
    assert flags == {"root_access_missing"}


def test_all_sensitive_hosts_captured_flag():
    obs = {"hosts": [], "sensitive_hosts_total": 2, "sensitive_hosts_with_root_access": 2}
    assert "all_sensitive_hosts_captured" in compute_observation_flags(obs)

    obs_incomplete = {"hosts": [], "sensitive_hosts_total": 2, "sensitive_hosts_with_root_access": 1}
    assert "all_sensitive_hosts_captured" not in compute_observation_flags(obs_incomplete)

    obs_none_sensitive = {"hosts": [], "sensitive_hosts_total": 0, "sensitive_hosts_with_root_access": 0}
    assert "all_sensitive_hosts_captured" not in compute_observation_flags(obs_none_sensitive)


def test_services_unknown_flag_treats_empty_list_the_same_as_zero_count():
    # build_observation_summary now reports known_services as a list of
    # confirmed names, not a count -- an empty list must trigger this flag
    # exactly like the old count-of-zero did.
    obs = {"hosts": [{"target": "host-1-0", "access": "none", "reachable": True, "known_services": []}]}
    assert "services_unknown" in compute_observation_flags(obs)


def _kb(*rules: KnowledgeRule) -> KnowledgeBase:
    return KnowledgeBase(version="test-v1", rules=tuple(rules))


def test_retrieve_matches_rule_with_no_triggers_regardless_of_context():
    kb = _kb(KnowledgeRule(id="always", observation_flags=(), legal_action_types=(), text="t"))
    matched = retrieve_rules(kb, {"hosts": []}, {"finish"})
    assert [r.id for r in matched] == ["always"]


def test_retrieve_requires_all_observation_flags():
    kb = _kb(
        KnowledgeRule(id="needs-both", observation_flags=("user_access_present", "root_access_missing"), legal_action_types=(), text="t")
    )
    # only root_access_missing is active (no hosts) -> should NOT match
    assert retrieve_rules(kb, {"hosts": []}, set()) == []
    # both active -> should match
    assert [r.id for r in retrieve_rules(kb, {"hosts": [_host(access="user")]}, set())] == ["needs-both"]


def test_retrieve_requires_any_legal_action_type_overlap():
    kb = _kb(KnowledgeRule(id="scan-rule", observation_flags=(), legal_action_types=("service_scan", "exploit"), text="t"))
    assert retrieve_rules(kb, {"hosts": []}, {"finish"}) == []
    assert [r.id for r in retrieve_rules(kb, {"hosts": []}, {"exploit"})] == ["scan-rule"]


def test_retrieve_preserves_stable_yaml_order():
    kb = _kb(
        KnowledgeRule(id="b", observation_flags=(), legal_action_types=(), text="t"),
        KnowledgeRule(id="a", observation_flags=(), legal_action_types=(), text="t"),
    )
    result = retrieve_rules(kb, {"hosts": []}, set())
    assert [r.id for r in result] == ["b", "a"]  # file order, not alphabetical


def test_retrieve_is_deterministic_across_repeated_calls():
    kb = load_knowledge_base("package://marla/knowledge/nasimemu_rules.yaml")
    obs = {"hosts": [_host(access="user", known_services=0)]}
    types = {"service_scan", "exploit", "process_scan", "privilege_escalation", "finish"}
    first = [r.id for r in retrieve_rules(kb, obs, types)]
    second = [r.id for r in retrieve_rules(kb, obs, types)]
    assert first == second
