"""Adversarial tests targeting the universal (``for all realizations``)
quantifier specifically -- acceptance-critical per the task that added
them. Each test is designed to break a checker that (incorrectly) unions
capabilities across mutually-exclusive random realizations, ignores
firewall/pivot semantics, or assumes a larger subnet "probably" contains a
usable host.

Every fixture uses a single fixed OS (``os: [linux]``) to eliminate
OS-choice randomness and isolate exactly the service-selection randomness
each test is about -- these tests are about the *pivot/firewall/subnet-size*
quantifier, not host-rootability (already covered by
``test_scenario_solvability.py``'s A-D/G-I). Service names follow
NASimEmu's real ``<port>_<os>_<name>`` convention (``is_for_os`` reads the
OS from the second underscore-separated token, see
``ScenarioLoaderV2._parse_host_configs``/``marla.scenario.spec.is_for_os``)
-- a fixture using non-conforming names would be silently treated as
incompatible with every OS, which is a fixture bug, not a checker bug (this
is exactly the mistake this file's first draft made and caught via a
failing test, not by inspection).
"""

from __future__ import annotations

from pathlib import Path
from textwrap import dedent, indent

from marla.scenario.solvability import check_solvability
from marla.scenario.spec import load_scenario_spec


def _write(tmp_path: Path, content: str, name: str = "scenario.v2.yaml") -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


def _block(text: str) -> str:
    """De-indent a triple-quoted block back to column 0, then re-indent by
    exactly 2 spaces -- safe to call regardless of the caller's own source
    indentation (unlike bare ``dedent``, which silently produces the wrong
    result when a block's own lines aren't uniformly indented relative to
    its closing ``\"\"\"``, e.g. because IDE/editor auto-indent left the
    closing line at a different depth -- verified against this exact
    failure mode while writing this file).
    """
    return indent(dedent(text).strip("\n") + "\n", "  ")


# Topology: internet(0) -> subnet1(1, entry, public) -> subnet2(2, pivot) ->
# subnet3(3, sensitive). subnet1/subnet3 fixed size 1 throughout; only
# subnet2's size/services vary per test.
_TOPOLOGY = dedent(
    """\
    topology: [[1, 1, 0, 0],
               [1, 1, 1, 0],
               [0, 1, 1, 1],
               [0, 0, 1, 1]]
    """
)

# 3-part names: <port>_<os>_<name> -- 'linux' or 'any' in the middle token,
# matching real bundled scenarios exactly (see is_for_os).
SVC_A = "1_linux_service_a"
SVC_B = "2_linux_service_b"
SVC_MARKER = "9_any_sensitive_marker"


def _scenario(subnet2_size: str, services: list[str], exploits_yaml: str, firewall_yaml: str) -> str:
    services_yaml = "\n".join(f"  - {s}" for s in services)
    return (
        "address_space_bounds: (4, 5)\n"
        f"subnets: [1, {subnet2_size}, 1]\n"
        f"{_TOPOLOGY}"
        "sensitive_hosts:\n"
        "  1: 0.\n"
        "  2: 0.\n"
        "  3: 1.0\n"
        "os:\n  - linux\n"
        f"services:\n{services_yaml}\n"
        f"  - {SVC_MARKER}\n"
        "processes:\n  - ~\n"
        f"exploits:\n{exploits_yaml}"
        "privilege_escalation:\n"
        "  pe_kernel:\n"
        "    process: ~\n"
        "    os: linux\n"
        "    prob: 1.0\n"
        "    cost: 1\n"
        "    access: root\n"
        f"sensitive_services:\n  - {SVC_MARKER}\n"
        "service_scan_cost: 1\n"
        "os_scan_cost: 1\n"
        "subnet_scan_cost: 1\n"
        "process_scan_cost: 1\n"
        "host_configurations: _random\n"
        f"{firewall_yaml}"
    )


def _exploit(name: str, service: str, os: str | None, access: str) -> str:
    return _block(
        f"""\
        {name}:
          service: {service}
          os: {os if os is not None else "~"}
          prob: 1.0
          cost: 1
          access: {access}
        """
    )


_MARKER_EXPLOIT = _exploit("e_marker", SVC_MARKER, None, "user")
# The sensitive host's guaranteed sensitive_services member has a USER
# exploit + pe_kernel privesc -> always rootable once reached, isolating
# every test below to purely network/pivot semantics, not host rootability.


def _firewall(edge_to_subnet3: str) -> str:
    # "firewall:" itself is a TOP-LEVEL key (unlike exploits/privesc entries
    # inserted via _block, which nest under an already-open top-level key)
    # -- only its children get the 2-space indent.
    return (
        "firewall:\n"
        "  (0, 1): [_all]\n"
        "  (1, 0): [_all]\n"
        "  (1, 2): [_all]\n"
        "  (2, 1): [_all]\n"
        f"  (2, 3): {edge_to_subnet3}\n"
        "  (3, 2): [_all]\n"
    )


# --- Test A: mutually exclusive pivot capabilities (the sensitive host in
# subnet 3 draws EITHER service_a or service_b; firewall on the only route
# into subnet3 permits only service_a) ------------------------------------


def test_A_mutually_exclusive_pivot_capabilities_is_unsolvable(tmp_path):
    exploits = (
        _MARKER_EXPLOIT
        + _exploit("e_service_a", SVC_A, "linux", "user")
        + _exploit("e_service_b", SVC_B, "linux", "user")
    )
    firewall = _firewall(f"[{SVC_A}]")  # only service_a permitted into subnet3
    content = _scenario("1-1", [SVC_A, SVC_B], exploits, firewall)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)
    assert spec.randomized is True

    result = check_solvability(spec)
    assert result.status.value == "proven_unsolvable", result.to_dict()
    assert len(result.network_reachability_failures) == 1


# --- Test B: both alternatives independently valid -> PROVEN_SOLVABLE ----


def test_B_all_pivot_alternatives_valid_is_solvable(tmp_path):
    exploits = (
        _MARKER_EXPLOIT
        + _exploit("e_service_a", SVC_A, "linux", "user")
        + _exploit("e_service_b", SVC_B, "linux", "user")
    )
    firewall = _firewall("[_all]")  # both permitted
    content = _scenario("1-1", [SVC_A, SVC_B], exploits, firewall)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    result = check_solvability(spec)
    assert result.status.value == "proven_solvable", result.to_dict()
    assert result.universally_solvable is True


# --- Test C: minimum legal subnet size (1-5) can still be unsolvable -----


def test_C_minimum_subnet_size_realization_is_unsolvable(tmp_path):
    """subnet2 ranges 1-5 hosts (irrelevant here -- the constrained edge is
    2->3, and subnet3 is fixed size 1); the sensitive host in subnet3 can
    legally draw only service_b, which the firewall blocks -- must still be
    PROVEN_UNSOLVABLE regardless of subnet2's size range.
    """
    exploits = (
        _MARKER_EXPLOIT
        + _exploit("e_service_a", SVC_A, "linux", "user")
        + _exploit("e_service_b", SVC_B, "linux", "user")
    )
    firewall = _firewall(f"[{SVC_A}]")
    content = _scenario("1-5", [SVC_A, SVC_B], exploits, firewall)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)
    assert spec.subnets[2].size_min == 1 and spec.subnets[2].size_max == 5

    result = check_solvability(spec)
    assert result.status.value == "proven_unsolvable", result.to_dict()


# --- Test D: every legally-drawable host works regardless of size -------


def test_D_size_does_not_matter_when_every_host_class_works(tmp_path):
    exploits = (
        _MARKER_EXPLOIT
        + _exploit("e_service_a", SVC_A, "linux", "user")
        + _exploit("e_service_b", SVC_B, "linux", "user")
    )
    firewall = _firewall("[_all]")
    content = _scenario("1-5", [SVC_A, SVC_B], exploits, firewall)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    result = check_solvability(spec)
    assert result.status.value == "proven_solvable", result.to_dict()


# --- Test E: firewall-specific failure (locally exploitable, blocked) ---


def test_E_firewall_blocks_the_only_service_the_target_host_can_run(tmp_path):
    """Only ONE non-sensitive service exists (service_a); the sensitive
    host is locally exploitable both via it and via its guaranteed
    marker service (has a real exploit for each), but the firewall on the
    only route into subnet3 permits NEITHER -- proves local exploitability
    is not being confused with network reachability. (Blocking only
    service_a while leaving the always-present marker service permitted
    would make this scenario genuinely solvable via the marker service --
    caught by this test's own first draft.)
    """
    exploits = _MARKER_EXPLOIT + _exploit("e_service_a", SVC_A, "linux", "user")
    firewall = _firewall("[]")  # blocks every service on the only route into subnet3
    content = _scenario("1-1", [SVC_A], exploits, firewall)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    result = check_solvability(spec)
    assert result.status.value == "proven_unsolvable", result.to_dict()
    assert len(result.network_reachability_failures) == 1
    assert result.network_reachability_failures[0].target_subnet_id == 3


# --- Test F: alternate topology path -- one route blocked, one works ----


def test_F_alternate_topology_path_is_solvable_if_one_route_works(tmp_path):
    """A(1) -> B(2) -> D(4) and A(1) -> C(3) -> D(4); B's host is
    concretely unexploitable, C's is concretely fine.

    Uses an explicit (non-``_random``) ``host_configurations`` dict, i.e. a
    concrete/static realization, rather than V2 ``_random`` generation --
    NASimEmu's real ``_random`` generator draws every host's services from
    one *global*, OS-filtered pool shared identically across every subnet
    (``ScenarioLoaderV2._parse_host_configs`` has no subnet-specific
    restriction at all), so "B may legally draw a broken service, C never
    can" is not constructible as two different randomized subnets in the
    same real V2 generator -- if a broken service is a legal draw anywhere,
    it is a legal (if unlikely) draw for every plain subnet's host,
    including C. The topological existence-of-an-alternate-path logic this
    test targets is about the fixed-point reachability search itself, not
    about V2 randomization (already covered by Tests A-E), so a concrete
    realization isolates exactly that.
    """
    topology = dedent(
        """\
        topology: [[1, 1, 0, 0, 0],
                   [1, 1, 1, 1, 0],
                   [0, 1, 1, 0, 1],
                   [0, 1, 0, 1, 1],
                   [0, 0, 1, 1, 1]]
        """
    )
    svc_b_broken = "1_linux_broken"
    svc_c_ok = "2_linux_ok"
    content = (
        "address_space_bounds: (5, 2)\n"
        "subnets: [1, 1, 1, 1]\n"
        f"{topology}"
        "sensitive_hosts:\n"
        "  1: 0.\n  2: 0.\n  3: 0.\n  4: 1.0\n"
        "os:\n  - linux\n"
        f"services:\n  - {svc_b_broken}\n  - {svc_c_ok}\n  - {SVC_MARKER}\n"
        "processes:\n  - ~\n"
        f"exploits:\n{_MARKER_EXPLOIT}{_exploit('e_service_c', svc_c_ok, 'linux', 'user')}"
        "privilege_escalation:\n"
        "  pe_kernel:\n"
        "    process: ~\n    os: linux\n    prob: 1.0\n    cost: 1\n    access: root\n"
        "service_scan_cost: 1\nos_scan_cost: 1\nsubnet_scan_cost: 1\nprocess_scan_cost: 1\n"
        "host_configurations:\n"
        f"  (1, 0): {{os: linux, services: [{svc_c_ok}], processes: []}}\n"
        f"  (2, 0): {{os: linux, services: [{svc_b_broken}], processes: []}}\n"
        f"  (3, 0): {{os: linux, services: [{svc_c_ok}], processes: []}}\n"
        f"  (4, 0): {{os: linux, services: [{SVC_MARKER}], processes: []}}\n"
        "firewall: _subnets\n"
    )
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)
    assert spec.randomized is False
    from dataclasses import replace

    spec.static_hosts[(4, 0)] = replace(spec.static_hosts[(4, 0)], is_sensitive=True)

    result = check_solvability(spec)
    assert result.status.value == "proven_solvable", result.to_dict()
    assert not result.network_reachability_failures


# --- Test J: all sensitive hosts individually rootable, but a required
# intermediate (non-sensitive) subnet cannot be compromised -------------


def test_J_unsolvable_intermediate_pivot_despite_all_sensitive_hosts_rootable(tmp_path):
    """The sensitive host itself is perfectly rootable (guaranteed exploit
    + privesc) -- the ONLY problem is that subnet2 (a non-sensitive,
    intermediate bridge subnet) can legally draw a host with a service
    that has no exploit at all, blocking the only path to subnet3. Proves
    "every target locally rootable" does not imply "scenario solvable".
    """
    svc_unexploitable = "1_linux_unexploitable"
    exploits = _MARKER_EXPLOIT  # subnet2's own service has NO exploit at all
    firewall = _firewall("[_all]")
    content = _scenario("1-1", [svc_unexploitable], exploits, firewall)
    path = _write(tmp_path, content)
    spec = load_scenario_spec(path)

    result = check_solvability(spec)
    assert result.status.value == "proven_unsolvable", result.to_dict()
    # The sensitive host class itself has no rootability problem -- this
    # is purely a network/pivot failure.
    assert not result.host_rootability_failures
    assert len(result.network_reachability_failures) == 1
    assert result.network_reachability_failures[0].target_subnet_id == 3
