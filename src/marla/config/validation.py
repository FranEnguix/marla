"""Validation helpers that go beyond what Pydantic field types can express.

Structural/schema validation lives in :mod:`marla.config.models`. This module
covers cross-cutting or filesystem-dependent checks (e.g. does the scenario
file actually exist) that are still worth surfacing to ``marla validate``
but are not part of the Pydantic schema itself.
"""

from __future__ import annotations

from pathlib import Path

from marla.config.models import Config
from marla.scenarios.uri import MARLA_SCENARIO_URI_PREFIX, ScenarioReferenceError, resolve_scenario_reference


def validate_scenario_reference(config: Config, config_dir: Path) -> list[str]:
    """Return a list of human-readable warnings/errors about the scenario reference.

    Three forms (see :mod:`marla.scenarios.uri`): a ``marla://<name>``
    reference (a MARLA-owned packaged scenario -- resolved and existence-
    checked against MARLA's own package data, never ``config_dir``), a
    plain filesystem path ending in ``.yaml`` (resolved relative to
    ``config_dir`` if not already absolute), or any other string (a
    NASimEmu procedurally generated benchmark name looked up at runtime --
    we can only meaningfully check existence for the first two cases).
    """
    problems: list[str] = []
    scenario = config.environment.scenario

    if scenario.startswith(MARLA_SCENARIO_URI_PREFIX) or scenario.endswith(".yaml"):
        try:
            resolved = resolve_scenario_reference(scenario, config_dir)
        except ScenarioReferenceError as exc:
            problems.append(f"environment.scenario '{scenario}' is invalid: {exc}")
            return problems
        if not Path(resolved).is_file():
            problems.append(
                f"environment.scenario '{scenario}' does not exist "
                f"(resolved to '{resolved}')"
            )

    return problems


def validate_knowledge_references(config: Config) -> list[str]:
    """Sanity-check ``agents[].knowledge.path`` values.

    Accepted forms: ``package://<path-inside-marla-package>`` or a plain
    filesystem path. Actual resolution/loading happens in
    :mod:`marla.knowledge.retriever` (Milestone 8); here we only reject
    obviously malformed references early.
    """
    problems: list[str] = []
    for agent in config.agents:
        path = agent.knowledge.path
        if not path:
            problems.append(f"agents[{agent.alias}].knowledge.path must not be empty")
            continue
        if not (path.startswith("package://") or path.startswith("/") or "/" in path):
            problems.append(
                f"agents[{agent.alias}].knowledge.path '{path}' is not a "
                "recognizable package:// or filesystem reference"
            )
    return problems


def validate_config_semantics(config: Config, config_dir: Path) -> list[str]:
    """Run all filesystem/cross-cutting checks and return combined problems."""
    problems: list[str] = []
    problems.extend(validate_scenario_reference(config, config_dir))
    problems.extend(validate_knowledge_references(config))
    return problems
