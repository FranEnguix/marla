# Coding Conventions

## Purpose

Set the engineering rules for MARLA implementation so that the codebase remains readable, typed, testable, reproducible, and aligned with the documented architecture across all MVPs.

## Final Design Decision

Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) is implemented in Python 3.10 with strict engineering standards. Public code must be typed, linted, formatted, documented, and tested. The project should use a layered structure that separates environment adapters, agent logic, policy logic, data models, prompts, experiment tooling, and documentation.

Linting and formatting are mandatory through Ruff. Type checking is mandatory through mypy in strict mode. Docstrings should use Sphinx RTD style. Testing should use pytest and include unit, integration, and deterministic replay tests. Data models should be implemented with Pydantic or with dataclasses plus an explicit validation layer.

## MVP Evolution

### MVP #1

Establish the project structure, typing rules, logging discipline, and minimum test strategy. The first MVP should already set the standards that later MVPs must inherit.

### MVP #2

Extend conventions to agent lifecycle code, transport adapters, and message handling. Communication behavior must be covered by integration tests.

### MVP #3

Apply the same standards to specialist roles, prompt handling, structured outputs, and blackboard integration. Optional subsystems should be toggleable without code duplication.

### MVP #4

Strengthen conventions around reward versioning, structured runtime traces, and ablation configuration management. Dynamic behavior must remain inspectable and testable.

### MVP #5

Stabilize the codebase as a coherent research platform with clear boundaries, maintainable modules, and testable interfaces across all major subsystems.

## Confirmed Constraints

- Python 3.10 is mandatory.
- Ruff is mandatory.
- mypy strict mode is mandatory.
- Public functions should be typed.
- Unjustified use of `Any` is not acceptable.
- Sphinx RTD-style docstrings are the standard.
- pytest is the test runner.
- Test suites must include unit tests, integration tests, and deterministic replay tests where applicable.
- The repository structure includes `docs/`, `configs/`, `scenarios/`, `prompts/`, `src/marla/`, `tests/`, and `scripts/`.
- Optional features should be controlled by configuration or feature flags rather than separate code paths.
- Implementation should remain layered and modular rather than highly entangled.

## Deferred / Future Work

- Final decision between Pydantic and dataclass-first data modeling.
- Naming conventions for prompt files and experiment artifacts.
- Coverage thresholds by test category.
- Pre-commit hook policy and CI pipeline details.
- Packaging and release conventions if the project becomes externally distributed.

## Risks / Drift Warnings

The biggest risk is allowing early MVP code to become a permanent prototype layer. If conventions are relaxed during the first phases, later cleanup will be expensive and architecture drift will accelerate.

Another risk is weak typing around messages, blackboard records, and reward logic. Those areas are central to MARLA and must remain explicit.

A final risk is feature sprawl. If future-MVP abstractions are implemented too early without a real need, the codebase will become harder to understand and maintain.
