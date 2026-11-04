.. _scenario-visualization:

Scenario figures
================

``marla scenario plot`` and ``marla scenario plot-batch`` draw the network
NASimEmu actually generates for a scenario **and a seed** -- not the
scenario's probability distributions -- as figures meant for papers:
vector PDF and SVG plus a 300 DPI PNG, with a deterministic layout.

Basic usage
-----------

.. code-block:: text

   marla scenario plot SCENARIO --seed 11
   marla scenario plot SCENARIO --seed 11 --verbose
   marla scenario plot SCENARIO --seeds 11,12,20-29 --view both

``SCENARIO`` is a scenario YAML path or a ``marla://<file name>`` reference
(as for ``marla scenario check``). This writes, into ``plots/scenarios/``
(or ``--output-dir``):

.. code-block:: text

   evidence_c_multi_target_seed_11.pdf
   evidence_c_multi_target_seed_11.svg
   evidence_c_multi_target_seed_11.png
   manifest.json

The **default view** shows the network structure only: the internet (where
the attacker starts), each subnet as a box titled with its subnet ID and,
if the YAML has ``subnet_labels``, its label (``Subnet 1 · DMZ``), the hosts
inside it labelled with their NASimEmu address ``(subnet,host)``, which
hosts are sensitive targets, and the subnet links. It never shows a host's
OS, services or processes.

``--verbose`` (or ``--view verbose``) draws the same topology with one card
per host listing what that realization sampled: OS, services, processes,
value, the exploits and privilege escalations whose host-side
preconditions hold on that host (with their access level, and success
probability when below 1), and any host firewall rules. Subnet links whose
firewall does not permit every service carry numbered marks (``FW1``,
``FW2``, ...) explained in a rules table under the legend.

Other options: ``--format pdf,svg,png`` (any subset), ``--dpi`` (PNG,
default 300), ``--size natural|single|double|<inches>`` (see below),
``--caption`` (prints scenario, seed and counts under the figure; off by
default so figures drop straight into a paper), ``--overwrite``, and
``--run-dir RUN`` (plots the scenario recorded in a ``marla run``
directory's ``metadata.json`` into ``RUN/plots/scenarios/``).

Reading the figures
-------------------

* Hosts are circles; **sensitive targets** (the hosts the
  ``capture_target`` objective needs ROOT on) are vermillion **diamonds**,
  with a bold address, and their verbose cards carry a "target" tag -- so
  target status never depends on color alone.
* Links join **subnets**, never hosts: NASimEmu has no host-to-host links.
  Hosts in one subnet reach each other; subnets connect through the
  topology matrix. A plain line is a bidirectional link whose subnet
  firewall permits every service; an arrowhead marks a one-way link (a
  non-symmetric topology); a firewall mark means some direction of that
  link permits only some services (the verbose rules table lists exactly
  which, per direction, "none" when nothing is permitted).
* A link is *connectivity*, not reachability at a given moment: which hosts
  an agent can reach also depends on what it has compromised and
  discovered so far (internet-facing subnets are reachable from the start,
  others after compromising a host in a connected subnet and, to be
  discovered, a subnet scan from it).
* The subnets are laid out top to bottom by their distance from the
  internet, so an attack path reads downwards.

Ground truth, not observations
------------------------------

Everything a verbose figure lists is **offline simulator ground truth** for
scientific documentation. The RL agent never sees it: it observes host
attributes only through its own scans and exploits (see
:mod:`marla.environment.visible_facts`). Every verbose figure says so in a
note under the legend, and ``manifest.json`` repeats it. Nothing about a
policy, its weights or its training outcome is ever drawn.

Exact reconstruction of a seeded realization
--------------------------------------------

A randomized scenario (``host_configurations: _random``, subnet size
ranges like ``2-5``, per-subnet sensitivity probabilities) is a
distribution. The figure must show the one realization an episode with
that seed played on, so :mod:`marla.environment.scenario_realization`
never samples anything itself: it calls the same
``NasimEmuAdapter.reset(seed=...)`` that training and evaluation call, then
reads the generated network from the simulator. This includes:

* **Zero-target regeneration.** If a seed's realization has no sensitive
  target, MARLA discards it and regenerates from a deterministic
  replacement seed (see :meth:`~marla.environment.nasimemu_adapter.NasimEmuAdapter.reset`).
  The figure shows the regenerated network; its file name carries that
  **effective seed**, and the manifest records both the requested and the
  effective seed and the attempt number.
* **Subnet permutation.** NASimEmu randomly permutes the non-internet
  subnet IDs of every generated network (the YAML's DMZ may be subnet 1 in
  one seed and subnet 2 in the next, and every host address moves with
  it). The permutation is recovered without changing NASimEmu's random
  stream and checked against the realized topology; it lets the layout
  place each YAML subnet in the **same position for every seed** while the
  titles and host labels show the realization's real IDs and addresses
  (the ones in MARLA's logs, e.g. ``host-2-0``).
* **No side effects.** The caller's ``random``/``numpy`` states are restored
  afterwards; plotting never changes training or evaluation.

Three kinds of seed are kept apart: the **training seed**
(``experiment.seed``) selects a run's own episode-seed streams and RNGs but
is not itself a realization seed; the **requested seed** is the scenario
seed an episode starts from (for MARLA evaluations, the evaluation episode
seed); the **effective seed** is the one that generated the network.

Batches from experiment records
-------------------------------

.. code-block:: text

   marla scenario plot-batch [SCENARIO] [--from-episodes PATH ...] [--eval-only] [--split NAME]
                             [--from-census CSV] [--seeds LIST] [--view both] [--jobs N]
                             [--skip-existing | --overwrite] [--no-dedupe] ...

Seeds can come from:

* ``--from-episodes PATH`` -- a MARLA ``episodes.csv``, or a directory
  searched recursively for them (training runs, evaluation outputs). Each
  row contributes its ``seed`` and ``scenario``. With ``SCENARIO``, only
  that scenario's rows are used. The same seed evaluated by several
  conditions, training seeds or policy modes is plotted **once**; the
  manifest lists them all under that seed's provenance.
* ``--from-census CSV`` -- a scenario census (``seed,host,os,services,
  sensitive`` and optional ``split``, one row per host), e.g. the one the
  evidence-aware study wrote. Every host of every seed is checked against it.
* ``--seeds LIST`` for ``SCENARIO``.

Every reconstruction is verified against what was recorded: census host
attributes, the episodes' ``sensitive_targets_total``, and the scenario
hash ``marla run`` recorded for that run (a scenario file edited since the
run is refused). A seed that does not match is **reported and gets no
figure** (``unreconstructed`` in the manifest; exit status 1).

By default a batch renders identical realizations once: different seeds
can generate exactly the same network, and that figure then lists all of
them (``--no-dedupe`` renders each effective seed). Batches show progress
and a summary, and run ``--jobs`` rendering processes in parallel.

``marla summarize RUN --scenarios`` does the same for a run's own
evaluation episodes (into ``RUN/plots/scenarios/``, skipping figures that
already exist). It is opt-in: ordinary summarization never renders
scenarios.

Output directory and manifest
-----------------------------

All figures of an output directory sit directly in it -- no per-scenario or
per-seed subdirectories:

.. code-block:: text

   <scenario>_seed_<effective seed>.{pdf,svg,png}           default view
   <scenario>_seed_<effective seed>_verbose.{pdf,svg,png}   verbose view
   manifest.json

``<scenario>`` is the scenario file name without ``.yaml``/``.v2``
(characters unsafe in file names become ``_``). A *different* scenario file
with the same name gets ``_<first 8 hex digits of its sha256>`` appended.
Existing files are never replaced silently: a run that would overwrite one
stops before writing anything unless ``--overwrite`` (replace) or, for
batches, ``--skip-existing`` (keep) is given.

``manifest.json`` makes every figure traceable to its exact realization:
the scenario's file paths and sha256; per figure, every requested seed that
generates it with its effective seed, regeneration attempt, verification
result and provenance (sources, episodes, training seeds, splits,
conditions, experiments); the subnet permutation; counts of subnets, hosts
and targets; the files of each view, figure size, smallest font, and the
layout self-check (overlaps, clipping, edges through boxes -- expected
empty) and edge-crossing count; formats, DPI and warnings. It is merged,
never truncated, when more figures are added.

**Representative figures.** Per scenario, exactly one figure is marked
``"paper_representative": true``, chosen by a fixed rule that never looks
at any policy's results: the most frequent structural signature (hosts
and targets in each YAML subnet, weighted by requested seeds), and among
those the figure of the smallest requested seed. The rule and the chosen
signature's frequency are recorded under ``representative_selection``.
Representative figures are ordinary files of the directory.

Formats, sizes and fonts
------------------------

* **PDF**: vector, fonts embedded as TrueType (text stays selectable).
* **SVG**: vector, text kept as editable ``<text>`` elements (font family
  DejaVu Sans; a viewer without it substitutes a similar sans-serif).
* **PNG**: 300 DPI by default (``--dpi``).

Exports are byte-for-byte reproducible: the same scenario, seed and options
give identical files.

Text is set in DejaVu Sans (bundled with matplotlib) at its print size --
6 to 7.5 pt -- and the figure grows with the network rather than shrinking
its text: ``--size natural`` (default). ``--size single`` / ``double`` /
``<inches>`` produces a figure exactly that wide (3.33 in and 7.0 in, the
ACM/AAMAS column and text widths), scaling the drawing down (never up) and
refusing if that would push text below 5 pt. Large verbose figures are
meant as supplementary material at their natural size.

Including a figure in LaTeX
---------------------------

.. code-block:: latex

   \begin{figure}[t]
     \centering
     \includegraphics{figures/evidence_c_multi_target_seed_3000000001.pdf}
     \caption{Scenario C, held-out realization 3000000001: two DMZ hosts
       reachable from the internet and two internal hosts; diamonds are
       sensitive targets.}
   \end{figure}

Include the PDF without a ``width=`` so its text keeps its designed size;
for a column-wide figure, render with ``--size single`` instead of scaling
in LaTeX.

Limitations
-----------

* Only subnet-level structure is drawn. Firewall rules are shown as marks
  plus a rules table (verbose view), not as per-service edges.
* Exploit/privilege-escalation entries on verbose cards state that the
  *host-side* preconditions (service or process, and OS) hold, exactly as
  NASimEmu checks them; whether an attempt succeeds also depends on
  reachability, subnet firewalls, access level and the action's success
  probability.
* NASimEmu defines no entry, pivot or attacker hosts: the attacker starts
  on the internet, and no host roles beyond "sensitive target" are drawn.
* Very large networks produce large verbose figures (columns per subnet are
  reduced when a layer grows too wide, but no information is ever dropped).

Regenerating the evidence-aware study's figures
------------------------------------------------

.. code-block:: text

   cd research/comparisons/evidence_aware_action_selection
   scripts/plot_scenarios.sh            # add --overwrite to re-render

which runs, for each experiment ``E`` and its scenario ``S``:

.. code-block:: text

   marla scenario plot-batch scenarios/S --from-census evaluation/scenario_census_E.csv \
       --from-episodes evaluation --output-dir plots/scenarios --jobs 8
