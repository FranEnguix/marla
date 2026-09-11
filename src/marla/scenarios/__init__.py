"""MARLA-owned NASimEmu scenarios.

Every scenario MARLA ships that has passed the current universal
structural-solvability validator (:mod:`marla.scenario`) for its declared
objective lives under :data:`solvable`, and its filename always contains
``.solvable.v2.yaml`` (or, for a non-randomized scenario, ``.solvable.yaml``).

**What ``.solvable`` means -- and does not mean**

``*.solvable.v2.yaml`` means exactly this:

    This scenario file passed ``marla scenario check`` (validator version
    recorded in ``metadata.json``/the checker's own result) for the
    ``capture_target`` objective: every realization NASimEmu's loader can
    legally produce from it has at least one valid action sequence
    reaching ROOT on every sensitive host.

It does **not** mean:

- PPO (or any policy) will necessarily learn a solution;
- every, or any, training run against it will succeed;
- the scenario is easy;
- a solution exists within any particular ``environment.max_episode_steps``
  budget (structural solvability and step-budget feasibility are tracked
  separately -- see :class:`marla.scenario.models.StepBoundEstimate`).

Structural solvability and learnability are different questions; only the
first is what this convention certifies. See :doc:`/scenario_solvability`
for the full definition and the reasoning behind it.

The filename alone is never trusted: ``marla scenario check`` (and the
``marla run`` preflight) always re-validates the actual file content, not
its name -- a scenario renamed to end in ``.solvable.v2.yaml`` without
actually passing validation still fails.

This package is part of the installed ``marla`` distribution (see
``pyproject.toml``'s wheel packaging), so :func:`solvable_scenario_path`
resolves correctly whether MARLA is used from an editable source checkout
or a normal ``pip install``.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path


def solvable_scenarios_dir() -> Path:
    """Directory containing every scenario MARLA has validated and
    committed as solvable."""
    return Path(str(resources.files("marla.scenarios") / "solvable"))


def solvable_scenario_path(name: str) -> Path:
    """Resolve one committed solvable scenario by filename, e.g.
    ``"sm_entry_user_three_subnets.solvable.v2.yaml"``.

    Raises
    ------
    FileNotFoundError
        If no such file exists under :func:`solvable_scenarios_dir`.
    """
    path = solvable_scenarios_dir() / name
    if not path.is_file():
        available = sorted(p.name for p in solvable_scenarios_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"No MARLA-owned solvable scenario named {name!r}. Available: {available}"
        )
    return path


def diagnostic_scenarios_dir() -> Path:
    """Directory containing MARLA's small, deterministic, hand-authored
    micro-scenarios used to prove the action/target compatibility
    representation fix (:mod:`marla.environment.action_compatibility`)
    actually removes the aliasing problem it targets, and for cheap
    diagnostic PPO learnability runs (``research/diagnostics/``). These
    are deliberately not part of :func:`solvable_scenarios_dir` --
    tiny, purpose-built fixtures, not scenarios meant for real AAMAS
    training/evaluation.
    """
    return Path(str(resources.files("marla.scenarios") / "diagnostic"))


def diagnostic_scenario_path(name: str) -> Path:
    """Resolve one committed diagnostic micro-scenario by filename, e.g.
    ``"micro_a_direct_root.v2.yaml"``.

    Raises
    ------
    FileNotFoundError
        If no such file exists under :func:`diagnostic_scenarios_dir`.
    """
    path = diagnostic_scenarios_dir() / name
    if not path.is_file():
        available = sorted(p.name for p in diagnostic_scenarios_dir().glob("*.yaml"))
        raise FileNotFoundError(
            f"No MARLA-owned diagnostic scenario named {name!r}. Available: {available}"
        )
    return path
