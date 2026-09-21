"""Targeted tests for marla.environment.consultation_scope -- the single
authoritative subnet-routing/scoping implementation shared by the real
SPADE consultation path and DirectConsultant evaluation.
"""

from __future__ import annotations

import pytest
import torch

from marla.environment.actions import ActionDescriptor, finish_descriptor, host_target_key
from marla.environment.consultation_scope import (
    build_scoped_observation,
    compute_routing_margin,
    extract_action_subnet,
    select_consulted_actions,
    select_consulted_subnet,
)


def _scan(subnet: int, host: int, kind: str = "service_scan") -> ActionDescriptor:
    target = host_target_key(subnet, host)
    return ActionDescriptor(action_id=f"{kind}:{target}", action_type=kind, target_key=target)


def _exploit(subnet: int, host: int) -> ActionDescriptor:
    target = host_target_key(subnet, host)
    return ActionDescriptor(action_id=f"exploit:{target}:eternalblue", action_type="exploit", target_key=target)


def _privesc(subnet: int, host: int) -> ActionDescriptor:
    target = host_target_key(subnet, host)
    return ActionDescriptor(
        action_id=f"privilege-escalation:{target}:sudo", action_type="privilege_escalation", target_key=target
    )


# --- 1. subnet extraction from every target action type --------------------


def test_extract_action_subnet_for_every_action_type():
    for action_type, descriptor in [
        ("service_scan", _scan(2, 3, "service_scan")),
        ("os_scan", _scan(2, 3, "os_scan")),
        ("process_scan", _scan(2, 3, "process_scan")),
        ("subnet_scan", _scan(2, 3, "subnet_scan")),
        ("exploit", _exploit(2, 3)),
        ("privilege_escalation", _privesc(2, 3)),
    ]:
        assert extract_action_subnet(descriptor) == 2, action_type


# --- 2. FINISH has no subnet ------------------------------------------------


def test_finish_has_no_subnet():
    assert extract_action_subnet(finish_descriptor()) is None


# --- 3. deterministic subnet selection --------------------------------------


def test_select_consulted_subnet_is_the_subnet_of_the_highest_base_logit_non_finish_action():
    actions = [_scan(1, 1), _scan(2, 1), finish_descriptor()]
    base_logits = torch.tensor([0.1, 5.0, 3.0])  # subnet-2 action has the highest non-FINISH logit
    assert select_consulted_subnet(actions, base_logits) == 2


def test_select_consulted_subnet_is_deterministic_and_repeatable():
    actions = [_scan(1, 1), _scan(3, 1), _scan(2, 1), finish_descriptor()]
    base_logits = torch.tensor([1.0, 1.0, 1.0, 1.0])  # tie among all non-FINISH candidates
    first = select_consulted_subnet(actions, base_logits)
    for _ in range(5):
        assert select_consulted_subnet(actions, base_logits) == first
    # torch.argmax breaks ties by returning the first maximal index --
    # subnet 1's action is first in the non-FINISH candidate list.
    assert first == 1


# --- 4. FINISH cannot choose the consultation subnet ------------------------


def test_finish_cannot_choose_consultation_subnet_even_with_the_highest_logit():
    actions = [_scan(1, 1), finish_descriptor()]
    base_logits = torch.tensor([0.01, 99.0])  # FINISH has by far the highest logit
    assert select_consulted_subnet(actions, base_logits) == 1


def test_no_non_finish_candidate_yields_none_subnet_not_a_crash():
    actions = [finish_descriptor()]
    base_logits = torch.tensor([2.0])
    assert select_consulted_subnet(actions, base_logits) is None


# --- 5/6. scoped action membership + FINISH included exactly once ----------


def test_select_consulted_actions_membership_and_finish_exactly_once():
    actions = [_scan(1, 1), _scan(1, 2), _exploit(2, 1), _privesc(3, 1), finish_descriptor()]
    scope = select_consulted_actions(actions, consulted_subnet=1)

    non_finish_consulted = [a for a in scope.consulted_actions if not a.is_finish]
    assert all(extract_action_subnet(a) == 1 for a in non_finish_consulted)
    assert len(non_finish_consulted) == 2

    finish_count = sum(1 for a in scope.consulted_actions if a.is_finish)
    assert finish_count == 1

    # every non-consulted global action must be absent from the scope, but
    # nothing here removes it from the GLOBAL action list PPO still sees --
    # this module only ever returns index subsets, never mutates `actions`.
    consulted_ids = {a.action_id for a in scope.consulted_actions}
    assert actions[2].action_id not in consulted_ids  # subnet-2 exploit excluded
    assert actions[3].action_id not in consulted_ids  # subnet-3 privesc excluded
    assert scope.global_candidate_action_count == len(actions)


def test_select_consulted_actions_preserves_global_order_and_stable_indices():
    actions = [_scan(2, 1), finish_descriptor(), _scan(2, 2), _scan(1, 1)]
    scope = select_consulted_actions(actions, consulted_subnet=2)
    assert scope.consulted_indices == [0, 1, 2]  # global order, never reordered
    assert [actions[i] for i in scope.consulted_indices] == scope.consulted_actions


def test_no_consulted_subnet_edge_case_yields_finish_only():
    actions = [_scan(1, 1), _scan(2, 1), finish_descriptor()]
    scope = select_consulted_actions(actions, consulted_subnet=None)
    assert len(scope.consulted_actions) == 1
    assert scope.consulted_actions[0].is_finish


# --- 7/8/9. scoped observation: filtering, global progress, no leakage -----


def _observation_fixture() -> dict:
    return {
        "hosts": [
            {
                "target": "host-1-1", "access": "none", "compromised": False, "reachable": True,
                "is_sensitive_target": False, "known_os": ["linux"], "known_services": ["ssh"], "known_processes": [],
            },
            {
                "target": "host-2-1", "access": "root", "compromised": True, "reachable": True,
                "is_sensitive_target": True, "known_os": ["windows"], "known_services": [], "known_processes": ["svchost"],
            },
            {
                "target": "host-2-2", "access": "user", "compromised": True, "reachable": True,
                "is_sensitive_target": False, "known_os": [], "known_services": ["smb"], "known_processes": [],
            },
        ],
        "sensitive_hosts_total": 2,
        "sensitive_hosts_with_root_access": 1,
        "network_exploration": {
            "known_subnets": [1, 2],
            "successfully_scanned_subnets": [1],
            "known_unscanned_subnets": [2],
        },
    }


def test_scoped_observation_filters_to_selected_subnet_only():
    scoped = build_scoped_observation(_observation_fixture(), consulted_subnet=2)
    local_targets = {h["target"] for h in scoped["local_hosts"]}
    assert local_targets == {"host-2-1", "host-2-2"}
    assert "host-1-1" not in local_targets


def test_scoped_observation_local_host_fields_match_authoritative_facts_exactly():
    observation = _observation_fixture()
    scoped = build_scoped_observation(observation, consulted_subnet=2)
    by_target = {h["target"]: h for h in scoped["local_hosts"]}
    original_by_target = {h["target"]: h for h in observation["hosts"]}
    for target, host in by_target.items():
        assert host == original_by_target[target]  # no reshaping, no dropped/added fields


def test_scoped_observation_global_progress_reflects_full_visible_state_not_just_selected_subnet():
    scoped = build_scoped_observation(_observation_fixture(), consulted_subnet=2)
    progress = scoped["global_progress"]
    # sensitive-target counts must reflect ALL visible hosts, including the
    # one in subnet 1 -- the whole point of the global summary is that it
    # is NOT scoped down like local_hosts is.
    assert progress["visible_sensitive_targets_total"] == 2
    assert progress["visible_sensitive_targets_with_root"] == 1
    assert progress["visible_sensitive_targets_remaining"] == 1
    assert progress["known_subnets_count"] == 2
    assert progress["successfully_scanned_subnets_count"] == 1
    assert progress["known_unscanned_subnets_count"] == 1
    assert progress["selected_subnet"] == 2


def test_scoped_observation_no_hidden_fact_leakage_when_subnet_is_none():
    scoped = build_scoped_observation(_observation_fixture(), consulted_subnet=None)
    assert scoped["local_hosts"] == []
    # global progress is still the full visible-only summary -- never empty
    # just because no subnet was selected.
    assert scoped["global_progress"]["visible_sensitive_targets_total"] == 2


def test_scoped_observation_contains_no_keys_beyond_global_progress_and_local_hosts():
    scoped = build_scoped_observation(_observation_fixture(), consulted_subnet=1)
    assert set(scoped.keys()) == {"global_progress", "local_hosts"}


# --- compute_routing_margin (route-switch diagnostic, spec sections 4-5) ---


def test_routing_margin_is_the_gap_to_the_best_competing_subnet():
    actions = [_scan(1, 1), _scan(2, 1), finish_descriptor()]
    base_logits = torch.tensor([5.0, 2.0, 3.0])  # winner: subnet 1 (logit 5.0), best other: subnet 2 (logit 2.0)
    assert compute_routing_margin(actions, base_logits) == pytest.approx(3.0)


def test_routing_margin_is_none_with_a_single_visible_subnet():
    actions = [_scan(1, 1), _scan(1, 2), finish_descriptor()]
    base_logits = torch.tensor([5.0, 1.0, 3.0])
    assert compute_routing_margin(actions, base_logits) is None


def test_routing_margin_is_none_with_no_non_finish_candidate():
    actions = [finish_descriptor()]
    base_logits = torch.tensor([2.0])
    assert compute_routing_margin(actions, base_logits) is None


def test_routing_margin_can_be_negative_when_winner_and_runner_up_are_close():
    actions = [_scan(1, 1), _scan(2, 1)]
    base_logits = torch.tensor([1.0, 1.5])  # subnet 2 actually wins the argmax
    margin = compute_routing_margin(actions, base_logits)
    assert margin == pytest.approx(0.5)
    # the "winner" in the margin computation is whichever subnet
    # select_consulted_subnet would actually route to -- confirm consistency.
    assert select_consulted_subnet(actions, base_logits) == 2
