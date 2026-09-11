"""Deterministic representation tests for action-compatibility features and
status (spec section 29) -- written before wiring this into the encoder,
to pin down the exact semantics independent of any neural-network
behavior. ``unknown`` vs. ``mismatch`` vs. ``confirmed_compatible`` vs.
``not_applicable`` must never be conflated (spec section 6).
"""

from __future__ import annotations

from nasimemu.nasim.envs.utils import AccessLevel

from marla.environment.action_compatibility import (
    COMPATIBILITY_FEATURE_DIM,
    CompatibilityStatus,
    compatibility_status_from_vector,
    compute_action_compatibility,
    compute_compatibility_matrix,
)
from marla.environment.actions import ActionDescriptor, finish_descriptor
from marla.environment.visible_facts import VisibleHostFacts

TARGET = "host-2-0"


def _facts(
    reachable=True,
    compromised=False,
    access=AccessLevel.NONE,
    known_os=frozenset(),
    known_services=frozenset(),
    known_processes=frozenset(),
) -> dict[str, VisibleHostFacts]:
    return {
        TARGET: VisibleHostFacts(
            target_key=TARGET,
            reachable=reachable,
            compromised=compromised,
            access=access,
            known_os=known_os,
            known_services=known_services,
            known_processes=known_processes,
        )
    }


def _exploit(service="9200_windows_elasticsearch", os="windows") -> ActionDescriptor:
    return ActionDescriptor(
        action_id=f"exploit:{TARGET}:e_x", action_type="exploit", target_key=TARGET,
        parameters={"service": service, "os": os},
    )


def _privesc(process=None, os="windows") -> ActionDescriptor:
    return ActionDescriptor(
        action_id=f"privilege-escalation:{TARGET}:pe_x", action_type="privilege_escalation", target_key=TARGET,
        parameters={"process": process, "os": os},
    )


# --- Service match / unknown -----------------------------------------------


def test_service_match_when_known_present():
    facts = _facts(known_services=frozenset({"9200_windows_elasticsearch"}))
    vec, status = compute_action_compatibility(_exploit(), facts)
    assert vec[2] == 1.0  # required_service_known_present
    assert status != CompatibilityStatus.CONTRADICTED


def test_service_unknown_when_not_yet_discovered():
    facts = _facts(known_services=frozenset())  # nothing scanned yet
    vec, status = compute_action_compatibility(_exploit(), facts)
    assert vec[2] == 0.0
    assert status == CompatibilityStatus.UNKNOWN  # NOT contradicted


# --- OS match / mismatch / unknown -----------------------------------------


def test_os_match_when_observed_os_equals_requirement():
    facts = _facts(known_os=frozenset({"windows"}), known_services=frozenset({"9200_windows_elasticsearch"}))
    vec, status = compute_action_compatibility(_exploit(os="windows"), facts)
    assert vec[4] == 1.0  # required_os_known_present
    assert vec[5] == 0.0  # required_os_known_incompatible
    assert status == CompatibilityStatus.CONFIRMED_COMPATIBLE


def test_os_mismatch_when_observed_os_conflicts():
    facts = _facts(known_os=frozenset({"windows"}))
    vec, status = compute_action_compatibility(_exploit(service="1_linux_x", os="linux"), facts)
    assert vec[4] == 0.0
    assert vec[5] == 1.0  # required_os_known_incompatible
    assert status == CompatibilityStatus.CONTRADICTED


def test_os_unknown_when_no_os_scan_yet():
    facts = _facts(known_os=frozenset())
    vec, status = compute_action_compatibility(_exploit(), facts)
    assert vec[4] == 0.0
    assert vec[5] == 0.0  # NOT mismatch -- genuinely unknown
    assert status == CompatibilityStatus.UNKNOWN


def test_unknown_is_never_collapsed_into_mismatch():
    """The single most important anti-conflation assertion (spec section 6)."""
    facts = _facts(known_os=frozenset())
    _vec, status = compute_action_compatibility(_exploit(), facts)
    assert status != CompatibilityStatus.CONTRADICTED
    assert status == CompatibilityStatus.UNKNOWN


# --- Process match / unknown (privesc) -------------------------------------


def test_process_match_when_known_present():
    facts = _facts(compromised=True, known_processes=frozenset({"httpd"}), known_os=frozenset({"windows"}))
    vec, status = compute_action_compatibility(_privesc(process="httpd", os="windows"), facts)
    assert vec[3] == 1.0
    assert status == CompatibilityStatus.CONFIRMED_COMPATIBLE


def test_process_unknown_when_not_yet_discovered():
    facts = _facts(compromised=True, known_os=frozenset({"windows"}))
    vec, status = compute_action_compatibility(_privesc(process="httpd", os="windows"), facts)
    assert vec[3] == 0.0
    assert status == CompatibilityStatus.UNKNOWN


def test_process_none_requirement_is_unconstrained():
    """process=None (NASimEmu's `~` sentinel) means "no process required
    at all" -- must not be treated as an unmet requirement."""
    facts = _facts(compromised=True, known_os=frozenset({"windows"}))
    vec, status = compute_action_compatibility(_privesc(process=None, os="windows"), facts)
    assert vec[0] == 0.0  # has_service_or_process_requirement: False (process is None)
    assert status == CompatibilityStatus.CONFIRMED_COMPATIBLE


# --- Reachability / access / compromised precondition ----------------------


def test_unreachable_target_is_contradicted():
    facts = _facts(reachable=False, known_os=frozenset({"windows"}), known_services=frozenset({"9200_windows_elasticsearch"}))
    _vec, status = compute_action_compatibility(_exploit(), facts)
    assert status == CompatibilityStatus.CONTRADICTED


def test_privesc_on_uncompromised_host_is_contradicted():
    """NASimEmu's Network.perform_action requires host_compromised for
    privilege_escalation -- a real, positively-observable precondition."""
    facts = _facts(compromised=False, known_os=frozenset({"windows"}))
    vec, status = compute_action_compatibility(_privesc(process=None, os="windows"), facts)
    assert vec[9] == 0.0  # target_compromised
    assert status == CompatibilityStatus.CONTRADICTED


def test_privesc_on_compromised_host_is_not_contradicted_by_that_alone():
    facts = _facts(compromised=True, known_os=frozenset({"windows"}))
    vec, status = compute_action_compatibility(_privesc(process=None, os="windows"), facts)
    assert vec[9] == 1.0
    assert status == CompatibilityStatus.CONFIRMED_COMPATIBLE


# --- No hidden facts (anti-information-leak) --------------------------------


def test_true_hidden_state_never_leaks_as_a_match():
    """Critical regression test: even if we (as the test author) know the
    real underlying host runs elasticsearch, the compatibility layer must
    only ever see VisibleHostFacts -- which, here, does NOT include it
    (never observed). It must be treated as unknown, never matched.
    """
    # Simulates "true state has the service, but it was never scanned":
    # VisibleHostFacts simply doesn't carry it -- there is no back door.
    facts = _facts(known_os=frozenset({"windows"}), known_services=frozenset())  # NOT observed
    vec, status = compute_action_compatibility(_exploit(service="9200_windows_elasticsearch"), facts)
    assert vec[2] == 0.0  # required_service_known_present: NOT matched
    assert status == CompatibilityStatus.UNKNOWN


# --- Scans / FINISH: not_applicable -----------------------------------------


def test_scan_action_status_is_not_applicable():
    facts = _facts(known_os=frozenset({"windows"}))
    scan = ActionDescriptor(action_id=f"os-scan:{TARGET}", action_type="os_scan", target_key=TARGET, parameters={})
    _vec, status = compute_action_compatibility(scan, facts)
    assert status == CompatibilityStatus.NOT_APPLICABLE


def test_finish_status_is_not_applicable():
    from marla.environment.actions import finish_descriptor

    _vec, status = compute_action_compatibility(finish_descriptor(), {})
    assert status == CompatibilityStatus.NOT_APPLICABLE


# --- Fixed width -------------------------------------------------------------


def test_feature_vector_is_always_fixed_width():
    facts = _facts(known_os=frozenset({"windows"}), known_services=frozenset({"x"}))
    for descriptor in (_exploit(), _privesc(), ActionDescriptor("os-scan:h", "os_scan", TARGET, {})):
        vec, _status = compute_action_compatibility(descriptor, facts)
        assert len(vec) == COMPATIBILITY_FEATURE_DIM


def test_compatibility_matrix_shape_and_empty_case():
    facts = _facts()
    matrix = compute_compatibility_matrix([_exploit(), _privesc()], facts)
    assert matrix.shape == (2, COMPATIBILITY_FEATURE_DIM)
    empty = compute_compatibility_matrix([], facts)
    assert empty.shape == (0, COMPATIBILITY_FEATURE_DIM)


# --- Status reconstruction from a stored feature vector (spec section 15) ---
# decisions.csv / probability-mass diagnostics only have a StepRecord's
# already-computed compatibility_features tensor + legal_action_descriptors
# on hand, never the original VisibleHostFacts -- compatibility_status_from_vector
# must reproduce compute_action_compatibility's own status exactly, for
# every case above, or the three systems (encoder features, decision
# metrics, probability-mass diagnostics) would drift apart.


def _assert_reconstruction_matches(descriptor: ActionDescriptor, facts: dict[str, VisibleHostFacts]) -> None:
    vec, status = compute_action_compatibility(descriptor, facts)
    assert compatibility_status_from_vector(descriptor.action_type, vec) == status


def test_status_reconstruction_matches_for_every_documented_case():
    cases = [
        (_exploit(), _facts(known_services=frozenset({"9200_windows_elasticsearch"}))),
        (_exploit(), _facts()),
        (_exploit(os="windows"), _facts(known_os=frozenset({"windows"}), known_services=frozenset({"9200_windows_elasticsearch"}))),
        (_exploit(service="1_linux_x", os="linux"), _facts(known_os=frozenset({"windows"}))),
        (_exploit(), _facts(known_os=frozenset())),
        (_privesc(process="httpd", os="windows"), _facts(compromised=True, known_processes=frozenset({"httpd"}), known_os=frozenset({"windows"}))),
        (_privesc(process="httpd", os="windows"), _facts(compromised=True, known_os=frozenset({"windows"}))),
        (_privesc(process=None, os="windows"), _facts(compromised=True, known_os=frozenset({"windows"}))),
        (_exploit(), _facts(reachable=False, known_os=frozenset({"windows"}), known_services=frozenset({"9200_windows_elasticsearch"}))),
        (_privesc(process=None, os="windows"), _facts(compromised=False, known_os=frozenset({"windows"}))),
        (_privesc(process=None, os="windows"), _facts(compromised=True, known_os=frozenset({"windows"}))),
        (ActionDescriptor(f"os-scan:{TARGET}", "os_scan", TARGET, {}), _facts(known_os=frozenset({"windows"}))),
        (finish_descriptor(), {}),
    ]
    for descriptor, facts in cases:
        _assert_reconstruction_matches(descriptor, facts)


def test_status_reconstruction_rejects_wrong_width():
    import pytest

    with pytest.raises(ValueError):
        compatibility_status_from_vector("exploit", [0.0] * (COMPATIBILITY_FEATURE_DIM - 1))
