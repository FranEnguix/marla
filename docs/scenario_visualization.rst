.. _scenario-visualization:

Scenario figures
================

``marla scenario plot`` draws a NASimEmu scenario as figures meant for
papers -- vector PDF and SVG plus a 300 DPI PNG, with a deterministic
layout -- in two ways:

* **without a seed**, the scenario **definition**: what the YAML specifies,
  read directly from it without generating any network -- the topology,
  every subnet's host-count range, where sensitive targets can occur and,
  in the verbose figure, every probability that drives generation;
* **with** ``--seed``, one **realization**: the exact network NASimEmu
  generates for that seed, as used by MARLA's training and evaluation.

Both produce a *simplified* and a *verbose* figure by default.

.. code-block:: bash

   # Plot the YAML scenario definition
   marla scenario plot scenario.yaml

   # Plot a specific realization
   marla scenario plot scenario.yaml --seed 11

   # Only the simplified diagram
   marla scenario plot scenario.yaml --mode simplified

   # Only the verbose diagram
   marla scenario plot scenario.yaml --mode verbose

   # Save to a chosen directory
   marla scenario plot scenario.yaml \
       --output-dir plots/scenarios/

.. list-table::
   :header-rows: 1
   :widths: 24 76

   * - Command
     - Figure contents
   * - No seed, simplified
     - Scenario topology, each subnet's host-count range and target range.
   * - No seed, verbose
     - The same, plus the probability distributions of NASimEmu's generator
       (host counts, targets, OS, services and processes given the OS,
       sensitive services), the actions and the firewall rules.
   * - Seeded, simplified
     - The actual sampled network: its hosts and which are sensitive
       targets.
   * - Seeded, verbose
     - The actual sampled network with every host's sampled OS, services,
       processes and value (simulator ground truth -- see below).

``SCENARIO`` is a scenario YAML path or a ``marla://<file name>`` reference
(as for ``marla scenario check``). Options:

* ``--seed N`` (repeatable) or ``--seeds 11,12,20-29``: draw realizations
  instead of the definition.
* ``--mode both|simplified|verbose`` (default ``both``).
* ``--format all|pdf,svg,png`` (default ``all``; any comma-separated subset).
* ``--dpi`` (PNG, default 300).
* ``--output-dir`` (default ``plots/scenarios/``); ``--run-dir RUN`` plots
  the scenario recorded in a ``marla run`` directory into
  ``RUN/plots/scenarios/``.
* ``--size natural|single|double|<inches>`` (see below), ``--caption``
  (prints scenario, seed and counts under the figure; off by default so
  figures drop straight into a paper), ``--overwrite``.

Scenario definitions (no seed)
------------------------------

Nothing is sampled: :mod:`marla.scenario.definition` parses the YAML and
derives every probability in closed form from the exact rules of
NASimEmu's loader, so the figure describes all networks the scenario can
produce. The same file always gives the same figure.

**Simplified.** Each subnet is a container -- concrete hosts are not drawn,
because their number is random -- showing ``Hosts: 2–5`` (or ``Hosts: 3``
when fixed) and ``Targets: 0–2`` / ``Targets: none``; a diamond marks the
subnets that can hold sensitive targets. Counts are ranges over every
network the generator can produce. A host count the YAML does not
determine is shown as ``unspecified`` (and listed under ``warnings`` in the
manifest), never guessed.

**Verbose.** The same topology, and in every subnet: the host-count
distribution (sizes ``"a-b"`` are uniform over ``a..b``), the probability
that each host is a sensitive target (independently per host, as the YAML
specifies) with the resulting target range, and the share of its hosts
running each sensitive service. Below the diagram:

* a **host attributes** table for scenarios with
  ``host_configurations: _random``. Every host draws its OS uniformly from
  the OS list. Given the OS, it receives ``k`` of the ``n`` non-sensitive
  services available to that OS, a uniformly random subset with
  ``P(k) ∝ 0.5^k``; the table therefore lists ``P(present | OS)`` for each
  service, labelled as conditional on the OS, and the footnote gives
  ``P(k)``. Service presences are *not* independent and not mutually
  exclusive. A sensitive host additionally gets a sensitive service;
  another host gets one with probability 10%. Processes follow the same
  rule; ``processes: [~]`` means none are defined, and no process
  probabilities are shown;
* an **actions** table: every exploit and privilege escalation with the
  service/process and OS it requires, its success probability, the access
  it grants and its cost, plus the scan costs;
* the subnet firewall rules, when a link does not permit every service.

These are the generator's probabilities, **before** MARLA's zero-target
regeneration: MARLA discards every generated network without a sensitive
target and draws again, so the networks used in experiments are
conditioned on at least one target. The figure states
``P(no target)`` for the scenario (for ``evidence_b2_two_hosts``, 25%).

Static scenarios (NASimEmu v1 files, or v2 files with an explicit
``host_configurations`` mapping) list every host as the YAML defines it.

Subnet numbers in a definition are the YAML's. NASimEmu randomly permutes
the non-internet subnet IDs of every realization, so a realization's
"Subnet 1" is not necessarily the YAML's subnet 1; both kinds of figure
place each YAML subnet in the same position.

Seeded realizations
-------------------

A realization is rebuilt by :mod:`marla.environment.scenario_realization`
with the generator MARLA itself uses -- ``NasimEmuAdapter.reset(seed=...)``,
including the zero-target regeneration -- then read from the simulator. No
attribute is sampled independently.

* **Simplified**: each subnet's actual hosts, labelled with their NASimEmu
  address ``(subnet,host)`` -- the addresses in MARLA's logs, e.g.
  ``host-2-0`` -- with sensitive targets as vermillion diamonds and a bold
  address (never color alone); no host attributes.
* **Verbose**: one card per host with its sampled OS, services, processes
  (``n/a`` when the scenario defines none), value, and the exploits and
  privilege escalations whose host-side preconditions hold on it.

If the requested seed's network had no target and was regenerated, the
figure shows the regenerated network and its file name carries the
**effective seed**; the manifest records both. The caller's
``random``/``numpy`` states are restored afterwards, so plotting never
changes training or evaluation.

**Ground truth, not observations.** A seeded verbose figure shows offline
simulator ground truth for documentation. The RL agent does not see it: it
observes host attributes only through its own scans and exploits (see
:mod:`marla.environment.visible_facts`). Every such figure says so under
its legend.

Three kinds of seed are kept apart: the **training seed**
(``experiment.seed``) selects a run's own episode-seed streams and RNGs but
is not itself a realization seed; the **requested seed** is the scenario
seed an episode starts from (for MARLA evaluations, the evaluation episode
seed); the **effective seed** is the one that generated the network.

Reading the figures
-------------------

* Links join **subnets**, never hosts: NASimEmu has no host-to-host links.
  Hosts in one subnet reach each other; subnets connect through the
  topology matrix. A plain line is a bidirectional link whose subnet
  firewall permits every service; an arrowhead marks a one-way link; a
  firewall mark (``FW``; numbered ``FW1``, ``FW2``, ... in verbose figures,
  with a rules table listing what each direction permits) means the link
  does not permit every service.
* A link is *connectivity*, not reachability at a given moment: which hosts
  an agent can reach also depends on what it has compromised and
  discovered (internet-facing subnets are reachable from the start).
* Subnets are laid out top to bottom by their distance from the internet,
  so an attack path reads downwards.

Batches from experiment records
-------------------------------

.. code-block:: text

   marla scenario plot-batch [SCENARIO] [--from-episodes PATH ...] [--eval-only] [--split NAME]
                             [--from-census CSV] [--seeds LIST] [--mode both] [--jobs N]
                             [--skip-existing | --overwrite] [--no-dedupe] ...

draws the realizations an experiment used. Seeds come from MARLA
``episodes.csv`` files (``--from-episodes``, a file or a directory searched
recursively; with ``SCENARIO`` only that scenario's rows), a scenario
census (``--from-census``: ``seed,host,os,services,sensitive``, one row
per host) and/or ``--seeds``. The same seed evaluated by several
conditions, training seeds or policy modes is drawn **once**, and
different seeds that generate the identical network share one figure.
Every reconstruction is verified against the records -- census host
attributes, the episodes' ``sensitive_targets_total``, the scenario hash
``marla run`` recorded -- and a seed that does not match is reported and
gets no figure (exit status 1).

``marla summarize RUN --scenarios`` draws the run's scenario definition and
the realizations of its evaluation episodes into ``RUN/plots/scenarios/``.
It is opt-in: ordinary summarization never renders scenarios.

Output directory and manifest
-----------------------------

All figures sit directly in one directory -- no per-scenario or per-seed
subdirectories:

.. code-block:: text

   <scenario>.{pdf,svg,png}                                definition, simplified
   <scenario>_verbose.{pdf,svg,png}                        definition, verbose
   <scenario>_seed_<effective seed>.{pdf,svg,png}          realization, simplified
   <scenario>_seed_<effective seed>_verbose.{pdf,svg,png}  realization, verbose
   manifest.json

``<scenario>`` is the scenario file name without ``.yaml``/``.v2``
(characters unsafe in file names become ``_``); a *different* scenario file
with the same name gets ``_<first 8 hex digits of its sha256>`` appended.
Existing files are never replaced silently: a run that would overwrite one
stops before writing anything unless ``--overwrite`` (or, for batches,
``--skip-existing``) is given.

``manifest.json`` (schema ``marla.scenario_figures/2``) records every
figure: its ``kind`` (``definition`` or ``realization``), scenario paths
and sha256, the files, size, smallest font and layout self-check of each
view, and counts (definitions: host, target and per-subnet ranges and
``P(no target)``; realizations: subnets, hosts, targets and the subnet
permutation). For realizations it also lists every requested seed that
generates the network, with its effective seed, regeneration attempt,
verification and provenance (episodes, training seeds, splits, conditions,
experiments), and marks one ``"paper_representative": true`` realization
per scenario by a fixed rule that never looks at policy results (the most
frequent hosts/targets-per-subnet signature, then the smallest requested
seed). The manifest is merged, never truncated; a schema-1 manifest is
migrated on first use.

Formats, sizes and fonts
------------------------

* **PDF**: vector, fonts embedded as TrueType (text stays selectable).
* **SVG**: vector, text kept as editable ``<text>`` elements (DejaVu Sans;
  a viewer without it substitutes a similar sans-serif).
* **PNG**: 300 DPI by default.

Exports are byte-for-byte reproducible. Text is DejaVu Sans (bundled with
matplotlib) at its print size, 6 to 7.5 pt, and figures grow with the
network instead of shrinking text: ``--size natural`` (default).
``--size single`` / ``double`` / ``<inches>`` produces a figure exactly that
wide (3.33 in and 7.0 in, the ACM/AAMAS column and text widths), scaling
down (never up) and refusing to push text below 5 pt. Simplified
definition figures (about 2.6 in wide) fit a single column; verbose ones
(up to about 7 in) are meant for an appendix or supplementary material.

Including a figure in LaTeX
---------------------------

.. code-block:: latex

   \begin{figure}[t]
     \centering
     \includegraphics{figures/evidence_c_multi_target.pdf}
     \caption{Scenario C: two DMZ hosts reachable from the internet, each a
       sensitive target with probability 0.5, and two internal targets.}
     \label{fig:scenario-c}
   \end{figure}

   % appendix: the full probabilistic definition, text width
   \begin{figure*}[t]
     \centering
     \includegraphics{figures/evidence_c_multi_target_verbose.pdf}
     \caption{Scenario C definition with NASimEmu's generation probabilities.}
   \end{figure*}

Include the PDF without ``width=`` so its text keeps its designed size; for
a column-wide figure, render with ``--size single`` instead of scaling in
LaTeX.

Limitations
-----------

* Only subnet-level structure is drawn; firewall rules appear as marks plus
  a rules table, not as per-service edges.
* Exploit entries on seeded verbose cards state that the *host-side*
  preconditions hold, exactly as NASimEmu checks them; success also depends
  on reachability, subnet firewalls, access level and probability.
* NASimEmu defines no entry, pivot or attacker hosts: the attacker starts on
  the internet, and no host roles beyond "sensitive target" are drawn.
* Definition probabilities describe NASimEmu's v2 loader (``_random``); a
  scenario format it does not support is shown as ``unspecified``.

Regenerating the evidence-aware study's figures
------------------------------------------------

.. code-block:: text

   cd research/comparisons/evidence_aware_action_selection
   scripts/plot_scenarios.sh definitions     # the 4 definitions (8 figures)
   scripts/plot_scenarios.sh realizations    # every tested realization
   # add --overwrite to re-render existing files
