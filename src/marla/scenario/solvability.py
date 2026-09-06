"""Universal scenario-solvability checker for MARLA's ``capture_target``
objective (spec: "MARLA must never start a training/evaluation run on a
scenario for which one or more valid NASimEmu-generated realizations cannot
reach the configured success objective").

This is an **offline feasibility analysis**, not a simulation of PPO or of
MARLA's own observation/action representation (see module docstring of
:mod:`marla.environment.actions` for why that representation is a separate
concern -- a scenario can be structurally solvable even if today's policy
architecture would struggle to learn it, and that is out of scope here).

Formal model implemented
-------------------------
A host configuration (an OS plus a service set and a process set) is
**rootable** iff it admits at least one of:

    1. a direct ROOT-access exploit for an installed, OS-compatible service, or
    2. a USER-access exploit for an installed, OS-compatible service,
       *followed by* a compatible privilege-escalation action (OS-compatible,
       and whose required process -- if any -- is installed) that grants ROOT.

A scenario realization is **solvable** iff every sensitive host's subnet is
attacker-reachable (see below) and every sensitive host's configuration is
rootable. A scenario file is **universally solvable** iff *every*
realization NASimEmu's loader could legally produce from it is solvable.

Network reachability mirrors NASimEmu's own real propagation rule
(``nasim.envs.network.Network._update_reachable``/``subnet_public``/
``traffic_permitted``): a subnet is reachable if it is directly connected
to the internet in the topology matrix, or if a subnet already known
reachable is topologically connected to it *and* at least one host in the
target subnet can be compromised (any exploit access level -- pivoting
only requires a successful exploit, not ROOT) via a service the firewall
actually permits from that (or any other currently-reachable) subnet. This
is computed as a fixed-point relaxation, not a single BFS pass, because
real NASimEmu traffic can originate from *any* already-compromised or
public host, not just the specific neighbor a plain BFS would expand from
next -- a subnet whose only viable service is blocked from one reachable
neighbor but permitted from another (reachable slightly later) must still
end up reachable. Subnet-level firewall rules (``firewall: _subnets``, or
an explicit per-subnet-pair dict) are honored exactly; host-level
(per-address) firewall overrides -- a rarer, hand-authored-scenario
feature -- are not yet incorporated into pivot/reachability reasoning (see
this module's own limitations note), though they are still respected by
:func:`_host_rootability`'s caller data model. NASimEmu's own ``_random``
V2 host generation never sets per-address firewalls at all (grep-verified
in ``ScenarioLoaderV2._parse_host_configs``), so this is not a gap for any
randomized scenario -- only a hand-authored static scenario using them
narrowly could be affected.

V2 randomization
-----------------
For a V2 scenario with ``host_configurations: _random``, NASimEmu draws,
per host: a uniform-random OS, then a uniformly-random-sized (skewed
towards small) *subset* of that OS's compatible non-sensitive services (and,
for a sensitive host, exactly one uniformly-chosen compatible entry from
``sensitive_services``, always appended). Because rootability is monotonic
in the service/process set actually installed on a host (installing more
services or processes can only ever add attack paths, never remove one --
:func:`_host_rootability` only ever tests presence, never absence), the
*worst case* over every legally-drawable subset is always realized by some
legally-drawable **size-1** subset (NASimEmu's own subset-size distribution,
``skew_dist``, always allows size 1). This is why this checker reasons about
the finite set of single-service/single-process draws rather than
enumerating every subset (2^n) or sampling seeds -- it is an exact
worst-case argument, not an approximation: if a size-1 draw is unrootable,
that draw is one of the legal outcomes NASimEmu can produce, so the
scenario is not universally solvable; if every size-1 draw (and hence, by
monotonicity, every larger draw) is rootable, the scenario is universally
solvable with respect to rootability.

Monotonicity proof: verified assumptions (spec section 9)
------------------------------------------------------------
This checker's entire V2 argument rests on "the worst case is always a
legally-drawable size-1 draw." Each assumption behind that claim was
checked directly against ``ScenarioLoaderV2._parse_host_configs`` (not
merely asserted because it sounds plausible), and holds **only** for that
generator as it exists today:

- **Size-1 is always legally drawable.** ``skew_dist(n)`` samples from
  ``np.arange(n) + 1``, i.e. every candidate size is in ``[1, n]`` --
  size 0 is never possible (services list is asserted non-empty at parse
  time) and size 1 is always in range.
- **OS-dependent filtering (``is_for_os``) is applied identically for
  every subnet.** ``_parse_host_configs`` loops over every ``(subnet,
  host)`` address uniformly, with no subnet-specific branching at all --
  confirmed directly (see :mod:`marla.scenario.spec`'s docstring and
  ``tests/test_scenario_adversarial.py``'s Test F comment, which had to
  switch to a static fixture specifically because a "subnet-specific"
  service pool is *not constructible* in the real ``_random`` generator).
- **The guaranteed ``sensitive_services`` member is drawn independently of
  the non-sensitive subset**, and is only ever present at all when the
  host is sensitive. Critically, "this subnet's host is sensitive" is
  itself probabilistic per subnet (``sensitive_hosts: {subnet: prob}``) --
  a purely non-sensitive host (drawing *only* from ``non_sensitive_pool``,
  with **no** sensitive-service fallback) remains a legal, undiminished
  possibility for any subnet whose probability is below 1.0. An earlier
  version of :func:`_pivot_guaranteed_given_allowed_services` didn't track
  this distinction and would let a subnet "borrow" viability from a
  sensitive service that might not even be present on the host actually
  drawn -- fixed; see that function's own docstring and Test A/C in
  ``tests/test_scenario_adversarial.py``, which construct exactly the
  counterexample (a firewall permitting only one of two legally-drawable
  services) that catches a checker unioning across draws instead of
  requiring every draw to survive.
- **Duplicate services/processes cannot occur** (asserted at parse time:
  ``len(services) == len(set(services))``), so "drawing the same service
  twice" is not a possibility this model needs to account for separately.
- **Process generation follows the identical size-1-always-possible
  pattern** as services, and is otherwise irrelevant to a host lacking any
  process-requiring privesc.
- **Firewall permission is a fixed scenario property, never randomized**
  (``_parse_firewall`` has no ``random.*``/``np.random.*`` call) -- so
  taking the union of firewall-permitted services across multiple
  *already-established-reachable* subnets (in
  :func:`_network_reachability_randomized`) is combining facts that hold
  identically in *every* realization, not combining capabilities that may
  only coexist in different realizations. This is a different kind of
  union than the one above, and is safe for exactly that reason -- see
  that function's own docstring.
- **Drawing one service never changes what else can be drawn or what
  exploits exist** -- ``random.sample`` draws are independent per host,
  and the exploit/privesc *definitions* are static scenario data,
  unaffected by any host's realized configuration.

If a future NASimEmu generator version violates any of these (e.g.
correlated draws, subnet-specific pools, or randomized firewalls), this
checker's V2 randomized path must be revisited -- it would no longer be
entitled to claim ``PROVEN_SOLVABLE``/``PROVEN_UNSOLVABLE`` and should
return ``UNKNOWN`` instead of silently overclaiming. As of this writing no
such generator exists; :func:`check_solvability` already returns
``UNKNOWN`` for the one case it can detect structurally (an unsupported
``objective`` type) and each site above notes the specific NASimEmu
mechanism it was checked against, so a future audit has something exact to
re-verify rather than a general assurance.
"""

from __future__ import annotations

import hashlib
from collections import deque
from itertools import product

from marla.scenario.models import (
    HostRootabilityFailure,
    NetworkReachabilityFailure,
    ScenarioSolvabilityResult,
    SolvabilityStatus,
    StepBoundEstimate,
)
from marla.scenario.spec import ScenarioSpec, StaticHostSpec, is_for_os

VALIDATOR_VERSION = "1.0"
USER, ROOT = 1, 2

SUPPORTED_OBJECTIVES = ("capture_target",)


def scenario_content_hash(spec: ScenarioSpec) -> str:
    return hashlib.sha256(spec.raw_content.encode("utf-8")).hexdigest()


def _host_rootability(
    spec: ScenarioSpec, os: str, services: frozenset[str], processes: frozenset[str]
) -> tuple[bool, tuple[str, ...], str | None]:
    """Direct port of the formal model in the module docstring."""
    attack_paths: list[str] = []
    direct_root = False
    has_user_or_root = False
    for e in spec.exploits:
        if e.service in services and (e.os is None or e.os == os):
            if e.access == ROOT:
                direct_root = True
                attack_paths.append(f"{e.name} (service={e.service}) -> ROOT")
            elif e.access == USER:
                has_user_or_root = True
                attack_paths.append(f"{e.name} (service={e.service}) -> USER")

    privesc_to_root = False
    for p in spec.privescs:
        if (p.os is None or p.os == os) and (p.process is None or p.process in processes):
            if p.access == ROOT:
                privesc_to_root = True
                attack_paths.append(f"{p.name} -> ROOT (privesc, process={p.process or 'any'})")

    rootable = direct_root or (has_user_or_root and privesc_to_root)
    if rootable:
        missing = None
    elif not has_user_or_root and not direct_root:
        missing = (
            f"no exploit reaches USER or ROOT access for OS={os} given the installed "
            f"services {sorted(services) or '(none)'}"
        )
    else:
        missing = (
            f"USER access is achievable ({', '.join(sorted(set(a.split(' ')[0] for a in attack_paths)))}) "
            f"but no compatible privilege escalation to ROOT exists for OS={os}"
        )
    return rootable, tuple(attack_paths), missing


def _pivot_capable(spec: ScenarioSpec, os: str, services: frozenset[str]) -> bool:
    """Whether this (os, services) combination yields at least one
    successful exploit (any access level) -- the only requirement to
    pivot onward, per NASimEmu's own
    ``Network._update_reachable``/``perform_action`` (triggered by any
    ``action.is_exploit() and action_obs.success``, regardless of the
    resulting access level).
    """
    return any(e.service in services and (e.os is None or e.os == os) for e in spec.exploits)


# --- Randomized (V2, host_configurations: _random) analysis ---------------


def _service_pools(spec: ScenarioSpec, os: str) -> tuple[list[str], list[str]]:
    """(non_sensitive_pool, sensitive_pool) of services legally drawable for
    a host of this OS, mirroring ``ScenarioLoaderV2._parse_host_configs``
    exactly (``is_for_os`` filtering, sensitive services excluded from the
    "non-sensitive" pool)."""
    compatible = [s for s in spec.services if is_for_os(s, os)]
    sensitive_pool = [s for s in spec.sensitive_services if is_for_os(s, os)]
    non_sensitive_pool = [s for s in compatible if s not in sensitive_pool]
    return non_sensitive_pool, sensitive_pool


def _process_pool(spec: ScenarioSpec, os: str) -> list[str | None]:
    return [p for p in spec.processes if is_for_os(p, os)]


def _check_pivot_guarantee(spec: ScenarioSpec) -> tuple[bool, list[str]]:
    """True iff EVERY legally-drawable single-service host, for EVERY OS in
    the scenario, achieves at least USER access -- i.e. every possible host
    (sensitive or not, any subnet) is guaranteed exploitable and can
    therefore always pivot onward. See module docstring: worst case is
    always a size-1 draw, by monotonicity.
    """
    notes = []
    guaranteed = True
    for os in spec.os_list:
        non_sensitive_pool, _sensitive_pool = _service_pools(spec, os)
        if not non_sensitive_pool:
            guaranteed = False
            notes.append(
                f"OS={os} has no non-sensitive service NASimEmu could ever assign to a host "
                "of this OS -- host generation for this OS cannot be reasoned about safely."
            )
            continue
        for s in non_sensitive_pool:
            if not _pivot_capable(spec, os, frozenset({s})):
                guaranteed = False
                notes.append(
                    f"OS={os}, service={s}: no exploit exists for this (service, OS) pair. "
                    "A host that legally draws only this service cannot be compromised at all, "
                    "so it cannot be guaranteed to pivot onward to hosts/subnets behind it."
                )
    return guaranteed, notes


def _sensitive_host_failure_classes(
    spec: ScenarioSpec, sensitive_subnet_ids: list[int]
) -> list[HostRootabilityFailure]:
    """Exhaustive worst-case search over sensitive-host configuration
    classes (spec sections 4-5): for each OS, every (chosen sensitive
    service, single non-sensitive service, single process) combination
    NASimEmu could legally draw for a sensitive host of that OS. Bounded by
    ``len(os) * len(sensitive_services) * len(services) * len(processes)``,
    always small in practice (tens, not millions) -- see module docstring
    for why size-1 draws are the true worst case, not an approximation.
    """
    if not sensitive_subnet_ids:
        # No subnet can ever hold a sensitive host (sensitive probability
        # is 0 everywhere) -- the objective is vacuously satisfied by
        # every realization. That is a *different*, already-documented
        # problem (a vacuous objective, see research/aamas2027/AUDIT.md),
        # not a rootability failure: there is nothing to root, so nothing
        # here is unrootable.
        return []

    failures: list[HostRootabilityFailure] = []
    for os in sorted(spec.os_list):
        non_sensitive_pool, sensitive_pool = _service_pools(spec, os)
        if not sensitive_pool or not non_sensitive_pool:
            # Either this OS can never host a sensitive host (no compatible
            # sensitive service -- NASimEmu's own loader would raise
            # ``assert len(sensitive_services) > 0`` before ever reaching
            # this state) or it has no service pool at all -- both are
            # generation-level defects the pivot-guarantee check already
            # surfaces; nothing further to check for rootability here.
            continue

        process_pool = _process_pool(spec, os)
        process_choices: list[frozenset[str]] = (
            [frozenset({p}) for p in sorted(process_pool, key=lambda x: (x is None, x))]
            if process_pool
            else [frozenset()]
        )

        found: HostRootabilityFailure | None = None
        for ss, ns, procs in product(sorted(sensitive_pool), sorted(non_sensitive_pool), process_choices):
            services = frozenset({ss, ns})
            rootable, attack_paths, missing = _host_rootability(spec, os, services, procs)
            if not rootable:
                found = HostRootabilityFailure(
                    os=os,
                    services=services,
                    processes=procs,
                    available_attack_paths=attack_paths,
                    missing_capability=missing or "unknown",
                    affected_subnet_ids=tuple(sensitive_subnet_ids),
                )
                break
        if found is not None:
            failures.append(found)
    return failures


def _permitted_services_on_edge(spec: ScenarioSpec, src: int, dst: int) -> frozenset[str] | None:
    """``None`` means every service is permitted (the ``_subnets``/``_all``
    marker); a concrete frozenset restricts traffic to just those services.
    An edge with no firewall entry at all is treated as fully blocked
    (defensive -- a scenario that loads through NASimEmu's own loader
    always has an entry for every topologically-connected pair, see
    ``ScenarioLoaderV2._contains_all_required_firewalls``).
    """
    if src == dst:
        return None
    return spec.firewall.get((src, dst), frozenset())


def _pivot_guaranteed_given_allowed_services(
    spec: ScenarioSpec, allowed_services: frozenset[str] | None, dst_sensitive_prob: float = 0.0
) -> tuple[bool, list[str]]:
    """Like :func:`_check_pivot_guarantee`, restricted to the services a
    firewall permits on one specific subnet edge (``None`` = unrestricted).

    **Quantifier note (do not weaken this):** the service NASimEmu draws
    for a host is unaffected by the firewall -- only which OS-compatible
    services *exist* to draw from matters for the draw itself; the
    firewall only gates whether an already-installed service's exploit
    traffic can actually reach the host from a given source. So this must
    check, for **every** service the generator could legally draw (the
    full ``non_sensitive_pool``, unfiltered), whether *that exact draw*
    is both installed-and-permitted -- filtering the pool down to the
    permitted subset *before* checking would silently ask "does some
    permitted service work" (an existential claim: true if *any*
    realization succeeds) instead of "does the worst-case draw survive"
    (the universal claim actually required: true only if *every*
    realization succeeds). A prior version of this function filtered
    first and was caught by an adversarial test constructing exactly this
    counterexample (firewall permits only ``service_a``; a legal
    single-service draw of ``service_b`` is unreachable) --
    ``tests/test_scenario_adversarial.py``'s Test A/C.

    ``dst_sensitive_prob``: the destination subnet's configured
    sensitive-host probability. A host there is guaranteed to also carry
    one member of ``sensitive_services`` **only** when this is exactly
    1.0 (every legally-drawable host in that subnet is sensitive) --
    below 1.0, a purely non-sensitive host (whose only services come from
    ``non_sensitive_pool``, with no sensitive-service fallback at all) is
    itself a legal realization, so that fallback must not be assumed.
    """
    notes = []
    guaranteed = True
    for os in spec.os_list:
        non_sensitive_pool, sensitive_pool = _service_pools(spec, os)
        if not non_sensitive_pool:
            guaranteed = False
            notes.append(f"OS={os} has no non-sensitive service NASimEmu could ever assign to a host of this OS.")
            continue

        def _viable(service: str) -> bool:
            permitted = allowed_services is None or service in allowed_services
            return permitted and _pivot_capable(spec, os, frozenset({service}))

        # Only usable as a fallback if EVERY host in this subnet is
        # guaranteed sensitive (prob == 1.0) AND every possible choice of
        # guaranteed sensitive service is itself viable (the specific
        # element of sensitive_pool drawn is also random).
        sensitive_fallback_always_viable = (
            dst_sensitive_prob >= 1.0 and bool(sensitive_pool) and all(_viable(ss) for ss in sensitive_pool)
        )

        for s in non_sensitive_pool:
            if _viable(s):
                continue
            if sensitive_fallback_always_viable:
                # This non-sensitive draw alone doesn't work, but every
                # host here is guaranteed to ALSO carry a viable sensitive
                # service regardless -- still guaranteed overall.
                continue
            guaranteed = False
            if not _pivot_capable(spec, os, frozenset({s})):
                notes.append(f"OS={os}, service={s}: no exploit exists for this (service, OS) pair.")
            else:
                notes.append(
                    f"OS={os}, service={s}: has a working exploit, but the firewall on this edge "
                    "does not permit it -- a legal draw of only this service is unreachable."
                )
    return guaranteed, notes


def _network_reachability_randomized(
    spec: ScenarioSpec, pivot_notes: list[str]
) -> tuple[list[int], list[NetworkReachabilityFailure]]:
    """Subnet-level fixed-point mirroring ``Network.subnet_public`` /
    ``Network._update_reachable``/``traffic_permitted`` exactly: a subnet is
    reachable if directly public, or if pivoting through it is guaranteed
    using the union of services permitted by the firewall from its
    currently-reachable neighbors (real NASimEmu lets *any* already-
    compromised or public host serve as the traffic source, not just one --
    hence a fixed-point relaxation, re-checked every pass as more neighbors
    become reachable, rather than a single one-shot BFS pass).

    ``pivot_guaranteed``/``pivot_notes`` (the *global*, firewall-unaware
    scan from :func:`_check_pivot_guarantee`) are used only to annotate the
    result with a scenario-wide defect note (e.g. some (os, service) pair
    has zero exploits anywhere) -- they must NOT gate whether this
    function even attempts the relaxation below. An earlier version did
    gate on it, which was *unsound in the safe direction but wrong in
    completeness*: it caused a real topology with two alternate routes,
    one genuinely broken and one genuinely fine, to be reported
    unreachable even via the fine route, merely because some unrelated
    subnet's hypothetical host could draw a broken service -- caught by
    ``tests/test_scenario_adversarial.py``'s Test F. Each edge's own
    per-edge check (:func:`_pivot_guaranteed_given_allowed_services`,
    called below) already independently accounts for exactly this
    (service, OS) defect wherever it actually applies, so gating the whole
    relaxation on the same defect a second time was redundant and overly
    conservative, not an extra safety margin.
    """
    topology = spec.topology
    n = len(topology)
    reachable = [topology[s][0] == 1 for s in range(n)]
    parent: dict[int, int] = {}

    changed = True
    while changed:
        changed = False
        for dst in range(n):
            if reachable[dst]:
                continue
            in_neighbors = [p for p in range(n) if topology[p][dst] == 1 and reachable[p]]
            if not in_neighbors:
                continue
            allowed: set[str] = set()
            unrestricted = False
            for p in in_neighbors:
                permitted = _permitted_services_on_edge(spec, p, dst)
                if permitted is None:
                    unrestricted = True
                    break
                allowed |= permitted
            edge_allowed = None if unrestricted else frozenset(allowed)
            guaranteed, _ = _pivot_guaranteed_given_allowed_services(
                spec, edge_allowed, spec.subnets[dst].sensitive_prob
            )
            if guaranteed:
                reachable[dst] = True
                parent[dst] = in_neighbors[0]
                changed = True

    failures: list[NetworkReachabilityFailure] = []
    for subnet in spec.subnets:
        if subnet.id == 0 or not subnet.is_sensitive_capable:
            continue
        if not reachable[subnet.id]:
            path = [subnet.id]
            cur = subnet.id
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()

            structural_in_neighbors = [p for p in range(n) if topology[p][subnet.id] == 1]
            if not structural_in_neighbors:
                reason = "no topology edge leads to this subnet at all"
            else:
                # Name the specific blocking edge(s): every structural
                # in-edge either isn't itself reachable, or its firewall
                # restriction blocks every service a legal single-service
                # draw could produce.
                per_edge_notes = []
                for p in structural_in_neighbors:
                    permitted = _permitted_services_on_edge(spec, p, subnet.id)
                    _, edge_notes = _pivot_guaranteed_given_allowed_services(
                        spec, permitted, subnet.sensitive_prob
                    )
                    if edge_notes:
                        per_edge_notes.append(f"edge {p}->{subnet.id}: {'; '.join(edge_notes)}")
                reason = (
                    "; ".join(per_edge_notes)
                    if per_edge_notes
                    else "no topology path connects this subnet to the attacker's foothold "
                    "(none of its structural predecessor subnets is itself reachable)"
                )
                if pivot_notes:
                    reason += "; scenario-wide: " + "; ".join(pivot_notes)
            failures.append(
                NetworkReachabilityFailure(
                    target_subnet_id=subnet.id,
                    required_path=tuple(path) if len(path) > 1 else (0, subnet.id),
                    blocking_edge=(path[-2], path[-1]) if len(path) > 1 else (0, subnet.id),
                    reason=reason,
                )
            )
    reachable_ids = [s for s in range(n) if reachable[s]]
    return reachable_ids, failures


def _check_randomized(spec: ScenarioSpec) -> ScenarioSolvabilityResult:
    sensitive_subnet_ids = [s.id for s in spec.subnets if s.is_sensitive_capable]
    max_count = sum(s.size_max for s in spec.subnets if s.is_sensitive_capable)

    pivot_guaranteed, pivot_notes = _check_pivot_guarantee(spec)
    _reachable_ids, network_failures = _network_reachability_randomized(spec, pivot_notes)
    host_failures = _sensitive_host_failure_classes(spec, sensitive_subnet_ids)

    universally_solvable = not host_failures and not network_failures
    status = SolvabilityStatus.PROVEN_SOLVABLE if universally_solvable else SolvabilityStatus.PROVEN_UNSOLVABLE

    notes = list(pivot_notes) if not pivot_guaranteed else []
    if not sensitive_subnet_ids:
        notes.append(
            "No subnet in this scenario has a nonzero sensitive-host probability -- the "
            "capture_target objective is vacuously satisfied (0 sensitive hosts to root) on "
            "every realization. This is reported as solvable (nothing is unrootable), but a "
            "vacuous objective is a separate scenario-design problem this checker does not flag."
        )
    if universally_solvable:
        notes.append(
            "Universal solvability was proven by exhaustive worst-case analysis over every "
            "OS x sensitive-service x single-service x single-process combination NASimEmu's "
            "_random host generation can legally draw (monotonicity argument: larger service/"
            "process sets only add attack paths) -- not by sampling seeds."
        )

    return ScenarioSolvabilityResult(
        status=status,
        universally_solvable=universally_solvable,
        scenario_path=str(spec.path),
        scenario_format=spec.format,
        objective="capture_target",
        randomized=True,
        host_rootability_failures=host_failures,
        network_reachability_failures=network_failures,
        sensitive_subnet_ids=sensitive_subnet_ids,
        sensitive_target_count_range=(0, max_count),
        structurally_solvable=universally_solvable,
        step_bound=None,
        notes=notes
        + [
            "No numeric step-count bound is computed for randomized scenarios: the number and "
            "location of sensitive hosts is only known probabilistically, so any single number "
            "would not be a genuinely proven bound."
        ],
        scenario_hash=scenario_content_hash(spec),
        validator_version=VALIDATOR_VERSION,
    )


# --- Static (V1, or V2 with explicit host_configurations) analysis --------


def _check_static(spec: ScenarioSpec) -> ScenarioSolvabilityResult:
    assert spec.static_hosts is not None
    hosts: dict[tuple[int, int], StaticHostSpec] = spec.static_hosts

    # Concrete sensitive addresses: for V1 this is already resolved
    # (StaticHostSpec.is_sensitive); for a V2 file with an explicit
    # host_configurations dict, sensitivity is likewise per-address (no
    # separate probability step applies once host_configurations is a
    # concrete dict -- see loader_v2.py: sensitive_hosts there is a
    # subnet->probability map used only to *decide* addresses, which for an
    # already-explicit host_configurations dict is moot; MARLA treats such
    # a file's sensitivity as whatever ``sensitive_hosts`` in the YAML
    # resolves to for this one concrete network, i.e. deterministic here).
    sensitive_addrs = {addr for addr, h in hosts.items() if h.is_sensitive}

    topology = spec.topology
    n = len(topology)
    reachable = [topology[s][0] == 1 for s in range(n)]
    parent: dict[int, int] = {}
    # Which hosts are exploitable (any access) given service/OS match,
    # independent of firewall (firewall gates *which source subnet* can
    # reach them, checked separately below), seeded from public subnets and
    # grown as new subnets unlock.
    exploitable: dict[tuple[int, int], tuple[bool, tuple[str, ...], str | None]] = {
        addr: _host_rootability(spec, h.os, h.services, h.processes) for addr, h in hosts.items()
    }
    # Which services each pivot-capable host could be exploited through
    # (any exploit matching its OS+installed services), used to check
    # firewall permission per candidate source subnet below.
    pivot_services: dict[tuple[int, int], frozenset[str]] = {
        addr: frozenset(
            e.service for e in spec.exploits if e.service in h.services and (e.os is None or e.os == h.os)
        )
        for addr, h in hosts.items()
    }

    def _dst_pivotable_from(dst_subnet: int, src_subnet: int) -> bool:
        permitted = _permitted_services_on_edge(spec, src_subnet, dst_subnet)
        for addr, services in pivot_services.items():
            if addr[0] != dst_subnet or not services:
                continue
            if permitted is None or (services & permitted):
                return True
        return False

    changed = True
    while changed:
        changed = False
        for dst in range(n):
            if reachable[dst]:
                continue
            viable_sources = [
                p for p in range(n) if topology[p][dst] == 1 and reachable[p] and _dst_pivotable_from(dst, p)
            ]
            if viable_sources:
                reachable[dst] = True
                parent[dst] = viable_sources[0]
                changed = True

    network_failures: list[NetworkReachabilityFailure] = []
    host_failures: list[HostRootabilityFailure] = []

    for addr in sorted(sensitive_addrs):
        subnet_id = addr[0]
        if not reachable[subnet_id]:
            path = [subnet_id]
            cur = subnet_id
            while cur in parent:
                cur = parent[cur]
                path.append(cur)
            path.reverse()
            network_failures.append(
                NetworkReachabilityFailure(
                    target_subnet_id=subnet_id,
                    required_path=tuple(path) if len(path) > 1 else (0, subnet_id),
                    blocking_edge=(path[-2], path[-1]) if len(path) > 1 else (0, subnet_id),
                    reason=(
                        f"no host in a currently-reachable subnet upstream of subnet {subnet_id} "
                        "has any working exploit for its OS/service combination"
                    ),
                )
            )
            continue

        rootable, attack_paths, missing = exploitable[addr]
        if not rootable:
            host = hosts[addr]
            host_failures.append(
                HostRootabilityFailure(
                    os=host.os,
                    services=host.services,
                    processes=host.processes,
                    available_attack_paths=attack_paths,
                    missing_capability=missing or "unknown",
                    affected_subnet_ids=(subnet_id,),
                )
            )

    universally_solvable = not host_failures and not network_failures
    status = SolvabilityStatus.PROVEN_SOLVABLE if universally_solvable else SolvabilityStatus.PROVEN_UNSOLVABLE

    step_bound = None
    if universally_solvable and sensitive_addrs:
        try:
            from nasimemu.nasim.envs.utils import get_minimal_steps_to_goal

            lower = get_minimal_steps_to_goal([list(row) for row in topology], list(sensitive_addrs))
            step_bound = StepBoundEstimate(
                lower_bound=int(lower),
                lower_bound_proven=True,
                upper_bound=int(lower) + 2 * len(sensitive_addrs),
                upper_bound_proven=False,
            )
        except Exception:
            step_bound = None

    return ScenarioSolvabilityResult(
        status=status,
        universally_solvable=universally_solvable,
        scenario_path=str(spec.path),
        scenario_format=spec.format,
        objective="capture_target",
        randomized=False,
        host_rootability_failures=host_failures,
        network_reachability_failures=network_failures,
        sensitive_subnet_ids=sorted({addr[0] for addr in sensitive_addrs}),
        sensitive_target_count_range=(len(sensitive_addrs), len(sensitive_addrs)),
        structurally_solvable=universally_solvable,
        step_bound=step_bound,
        notes=[],
        scenario_hash=scenario_content_hash(spec),
        validator_version=VALIDATOR_VERSION,
    )


def check_solvability(spec: ScenarioSpec, objective: str = "capture_target") -> ScenarioSolvabilityResult:
    """Entry point: analyze whether every realization ``spec`` can produce
    has a valid path to ROOT on every sensitive host (spec section 20: the
    caller decides the objective; only ``capture_target`` is implemented
    today, matching :class:`marla.config.models.ObjectiveConfig`).
    """
    if objective not in SUPPORTED_OBJECTIVES:
        return ScenarioSolvabilityResult(
            status=SolvabilityStatus.UNKNOWN,
            universally_solvable=False,
            scenario_path=str(spec.path),
            scenario_format=spec.format,
            objective=objective,
            randomized=spec.randomized,
            notes=[f"Objective {objective!r} is not supported by the solvability checker yet."],
            scenario_hash=scenario_content_hash(spec),
            validator_version=VALIDATOR_VERSION,
        )

    if spec.randomized:
        return _check_randomized(spec)
    return _check_static(spec)
