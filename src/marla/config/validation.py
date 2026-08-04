"""Validation helpers that go beyond what Pydantic field types can express.

Structural/schema validation lives in :mod:`marla.config.models`. This module
covers cross-cutting or filesystem-dependent checks (e.g. does the scenario
file actually exist) that are still worth surfacing to ``marla validate``
but are not part of the Pydantic schema itself.
"""

from __future__ import annotations

from pathlib import Path

from marla.config.models import Config


def validate_scenario_reference(config: Config, config_dir: Path) -> list[str]:
    """Return a list of human-readable warnings/errors about the scenario reference.

    NASimEmu treats a scenario ending in ``.yaml`` as a path to a static
    scenario file, and any other string as the name of a procedurally
    generated benchmark looked up at runtime. We can only meaningfully
    check existence for the file case here.
    """
    problems: list[str] = []
    scenario = config.environment.scenario

    if scenario.endswith(".yaml"):
        scenario_path = Path(scenario)
        if not scenario_path.is_absolute():
            scenario_path = (config_dir / scenario_path).resolve()
        if not scenario_path.is_file():
            problems.append(
                f"environment.scenario '{scenario}' does not exist "
                f"(resolved to '{scenario_path}')"
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
