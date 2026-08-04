# Interfaces and Data Models

## Purpose

Define the concrete contracts between MARLA subsystems so that implementation remains modular, typed, testable, and stable across MVPs. This file is the bridge from conceptual architecture to implementation-ready schemas and APIs.

## Final Design Decision

Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) must use explicit typed interfaces for agent messages, environment interaction, policy decisions, blackboard state, reward adjustments, and experiment logging. Data exchange should be structured first and human-readable second. Free-form text may still exist inside specialist reasoning traces, but downstream control flow must depend on typed fields and well-defined message contracts.

The system should define canonical domain objects such as observation snapshots, action candidates, validated action sets, reward signals, episode summaries, and experiment records. Communication in the SPADE/XMPP runtime should be wrapped in message envelopes that preserve sender, receiver, type, correlation identifiers, timestamps, and payload schema version.

## MVP Evolution

### MVP #1

Define the minimum interface surface needed for baseline RL and environment wrapping. This includes observation ingestion, action execution, step result handling, and experiment logging.

### MVP #2

Introduce message envelopes and agent-facing interfaces aligned with SPADE/XMPP. The data model should now distinguish domain payloads from transport metadata.

### MVP #3

Add schemas for specialist proposals, plan suggestions, action refinements, and blackboard annotations. These should be versioned from the start to reduce drift.

### MVP #4

Add reward-adjustment proposal schemas, Gatekeeper filtering outputs, and richer structured logging for dynamic adaptation and governance behavior.

### MVP #5

Stabilize the full schema set for the integrated architecture, including persistent experiment records, event streams, and ablation metadata.

## Confirmed Constraints

- Interfaces must be typed and implementation-friendly.
- Public APIs should avoid unstructured `Any`-style payloads.
- Message contracts must support SPADE/XMPP transport.
- Core interface families include environment wrapper API, policy API, blackboard API, reward API, and logging/event API.
- Schema versioning should exist from the beginning, even if lightweight.
- Structured logging is mandatory for reproducibility and analysis.
- Event and message models must be traceable across runs and episodes.
- Human-readable traces are useful, but typed payloads are authoritative.

## Deferred / Future Work

- Final schema definitions for every domain object.
- Choice between Pydantic-first and dataclass-plus-validation implementation.
- Backward compatibility policy across MVPs.
- Stable identifiers for actions, proposals, and blackboard records.
- Formal event taxonomy for all runtime behaviors.

## Risks / Drift Warnings

If interfaces are not stabilized early, the codebase will accumulate hidden coupling between agents, wrappers, and training logic. That will make later MVP integration expensive and error-prone.

Another risk is relying on natural-language specialist outputs directly in control flow. Doing so would reduce determinism, make testing harder, and blur the contract between advisory reasoning and executable behavior.

A final risk is mixing transport and domain logic. SPADE/XMPP details should not leak into every business object, or the system will become difficult to evolve.
