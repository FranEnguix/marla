"""Run-config-aware preflight solvability check (spec sections 7, 20).

Used by ``marla run`` before any agent is started. Deliberately thin: all
the actual reasoning lives in :mod:`marla.scenario.solvability`; this module
only knows how to go from a loaded :class:`~marla.config.models.Config` to
the scenario path and the configured objective's *name* (not its
semantics -- today only ``capture_target`` is implemented, but this module
never hardcodes that, so a future objective type just needs a new branch in
``check_solvability``, not a CLI change).
"""

from __future__ import annotations

from pathlib import Path

from marla.config.models import Config
from marla.scenario.models import ScenarioSolvabilityResult
from marla.scenario.solvability import check_solvability
from marla.scenario.spec import load_scenario_spec


def preflight_check(config: Config, config_dir: Path, resolved_scenario_path: str) -> ScenarioSolvabilityResult:
    """Analyze the scenario ``marla run`` is about to use, under the
    experiment's own configured objective.
    """
    spec = load_scenario_spec(resolved_scenario_path)
    return check_solvability(spec, objective=config.objective.type)
