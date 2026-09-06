Scenario solvability
======================

MARLA never starts a training or evaluation run on a scenario for which
one or more valid NASimEmu-generated realizations cannot reach the
configured success objective. This page defines what that means and how
it's checked/repaired -- see :doc:`cli` for the exact commands
(:ref:`scenario-solvability`, :ref:`scenario-repair`) and their output.

Why this exists
-----------------

A NASimEmu "V2" scenario file (``*.v2.yaml``) is a *generator*, not a fixed
network: subnet sizes can be ranges, sensitive-host placement is
per-subnet-probabilistic, and ``host_configurations: _random`` draws each
host's OS and installed services/processes independently at every episode
reset. It is entirely possible for a scenario to be solvable *on average*
-- most generated realizations have a real attack path -- while a nonzero
fraction of legal realizations have none at all (e.g. a sensitive host
whose randomly-drawn OS and service set happens to admit no path to ROOT).
Training against such a scenario means an unknown fraction of episodes are
unwinnable by construction, silently capping achievable success and
confounding any comparison between conditions/hyperparameters. MARLA's
validator (:mod:`marla.scenario`) exists to catch this *before* spending
any compute on it.

Formal definition
-------------------

For MARLA's current (only) objective, ``capture_target``, a scenario
**realization** is solvable iff there exists a valid sequence of NASimEmu
actions that reaches ROOT access on every sensitive host. A scenario
**file** is **universally solvable** iff *every* realization NASimEmu's
own loader (:mod:`nasimemu.nasim.scenarios.loader_v2`) could legally
produce from it is solvable.

Rootability is derived directly from NASimEmu's real action-precondition
semantics (:mod:`nasimemu.nasim.envs.host_vector`,
:mod:`nasimemu.nasim.envs.network`), not approximated: a host configuration
(OS + installed services + installed processes) is rootable iff it admits

1. a direct ROOT-access exploit for an installed, OS-compatible service, or
2. a USER-access exploit for an installed, OS-compatible service,
   followed by a compatible privilege-escalation action (OS-compatible,
   and whose required process, if any, is installed) that grants ROOT.

Network reachability mirrors NASimEmu's real pivoting rule
(``Network._update_reachable``/``subnet_public``): a subnet is reachable
if directly connected to the internet in the topology matrix, or if an
already-reachable subnet is topologically connected to it *and* at least
one host there can be compromised (any exploit access level -- pivoting
only needs a successful exploit, not ROOT), subject to the scenario's
firewall rules. A scenario is not universally solvable if some
sensitive-capable subnet cannot be proven reachable this way, even if
every host in it would be individually rootable once reached.

Universal V2 solvability, not sampled seeds
----------------------------------------------

The checker does **not** sample seeds and declare success if none of them
failed -- a scenario can go thousands of seeds between failing
realizations (the bundled ``sm_entry_user_three_subnets.v2.yaml`` fails on
roughly 1 in 3 windows-OS sensitive-host draws, which is still easy to miss
sampling casually). Instead it reasons about the finite space of host
configuration classes NASimEmu's ``_random`` generation can produce.
Because rootability is monotonic in the installed service/process set
(installing more can only ever add attack paths, never remove one), the
worst case over every legally-drawable subset is always realized by some
legally-drawable *single-service* draw -- NASimEmu's own subset-size
distribution always allows size 1. So the checker enumerates every
OS x sensitive-service x single-service x single-process combination
(bounded by the scenario's own small service/process/OS counts, not by the
network size or number of possible seeds) and proves either that all of
them are rootable (``PROVEN_SOLVABLE``) or exhibits one that is not
(``PROVEN_UNSOLVABLE``, with that exact combination reported). This is an
exact proof, not an approximation, and stays cheap regardless of how many
concrete realizations the scenario could produce.

Three possible outcomes -- ``PROVEN_SOLVABLE``, ``PROVEN_UNSOLVABLE``, and
``UNKNOWN`` (a scenario shape the checker cannot yet reason about
exhaustively, e.g. an unsupported objective) -- are always distinguished
explicitly. ``marla run`` treats anything other than ``PROVEN_SOLVABLE`` as
unsafe to run.

Structural solvability vs. episode length
--------------------------------------------

The checker establishes that a valid action sequence *exists*; it does not
require that sequence to fit within ``environment.max_episode_steps``, and
never claims an exact minimum action count unless genuinely proven. For a
fully static (non-randomized) scenario it additionally reports a
conservative, explicitly-labeled step-count lower bound (the shortest
subnet-hop path to every sensitive host, a real necessary lower bound on
actions -- see ``nasimemu.nasim.envs.utils.get_minimal_steps_to_goal``) and
an unproven upper-bound estimate; for a randomized scenario, where the
number and location of sensitive hosts is only known probabilistically, no
single number would be a genuinely proven bound, so none is given.

Solvability vs. learnability
-------------------------------

This checker answers "does a solution exist?", never "can the current
policy architecture find it within a training budget?" A scenario can be
structurally ``PROVEN_SOLVABLE`` and still be extremely hard to *learn*
(sparse reward, long horizon, large action space) -- that is a separate,
policy/algorithm-level question this subsystem deliberately does not
address (see ``marla.environment.actions``' own module docstring for why
MARLA's action *representation* is likewise kept separate from this
offline scenario analysis).

.. _repair-policy:

Repair policy
---------------

``marla scenario repair`` (:mod:`marla.scenario.repair`) only implements
the minimal-modification priority the checker's own failure classes
actually need in practice: **add the missing attack capability** -- a
missing ROOT-capable exploit for an already-installed service, or (the more
common case in practice) a missing privilege escalation for an OS that can
currently only reach USER. It:

- never invents a new service, process, or OS -- it only ever reuses ones
  already present in the scenario, and only where they can legally occur
  on the affected host class (spec: never add an exploit that can never
  match a generated host);
- uses a clearly artificial identifier for anything it adds (e.g.
  ``marla_repair_windows_root_privesc``), and documents in the repair
  report that it is a benchmark-solvability construct, not a real
  vulnerability;
- edits the scenario's real YAML structure (never blind text substitution)
  and preserves every unrelated line, including comments, byte-for-byte;
- never touches topology, sensitive-host probabilities, host counts, or
  unrelated exploits/privescs when a local vulnerability addition suffices
  -- broader changes (firewall/topology) are reserved for failure classes
  this repository's scenarios have not been found to need, and repair
  raises rather than silently attempting one it wasn't designed to verify;
- never overwrites the source scenario, and never overwrites an existing
  output path without ``--overwrite``;
- only ever reports success after the generated file both loads through
  NASimEmu's real loader and is independently re-checked as
  ``PROVEN_SOLVABLE`` by this same checker -- see :ref:`scenario-repair`.

Reproducibility metadata
---------------------------

Every run that passes the preflight check records a compact
``scenario_validation`` object in its ``metadata.json``
(``status``, ``validator_version``, ``scenario_hash``) -- enough to prove
later that a paper experiment used a validated scenario, without storing
the (potentially large) full diagnostic structure for a run that never
needed it.

Runs from before a scenario was repaired
-------------------------------------------

Any run executed before this validation subsystem existed -- or before a
particular scenario was repaired -- has no such metadata, and may have
been trained/evaluated against a scenario later proven not universally
solvable. Such runs are never retroactively rewritten to pretend they used
a repaired scenario (that would fabricate provenance); instead, mark them
prominently, in the research documentation/manifest that catalogs them
(never inside the immutable run artifacts themselves), as **PILOT /
INVALID FOR FINAL COMPARISON / UNSOLVABLE SOURCE SCENARIO** -- see
``research/aamas2027/manifest.yaml``'s per-run ``status`` field for the
convention this repository uses, and its own revision history for a worked
example. An analysis script that discovers run directories automatically
should check for this marker and exclude such runs from any figure or
table presented as a final result, while still leaving their data on disk
for whatever it remains valid evidence of (e.g. a bug-diagnosis
investigation unrelated to the scenario's solvability).
