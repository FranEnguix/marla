# MARLA Project Overview

## Purpose

Provide a concise, authoritative overview of Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) so that a new contributor, reviewer, or reader can understand what the project is, why it exists, and what the intended end result looks like without reading the rest of the documentation first.

## Final Design Decision

MARLA is a research-oriented offensive AI platform for simulated environments that can later be emulated. It is designed to operate in NASimEmu scenarios and to study how a reinforcement learning attacker can be augmented by advisory agents with specialized reasoning roles. MARLA is a RL system whose core acting policy is owned by the RL Orchestrator, while the other agents provide advisory, filtering, refinement, and reward-shaping support.

The end state of the project is a publication-ready experimental platform that supports training, planning, evaluation, and execution in offensive cyber scenarios. The system must be modular, ablatable, reproducible in baseline mode, and compatible with Python 3.10, SPADE, XMPP, and NASimEmu.

## MVP Evolution

### MVP #1

Establish the minimal MARLA baseline: a single RL-driven attacker interacting with NASimEmu on a local machine. The purpose of this phase is to confirm that the environment wrapper, training loop, logging pipeline, and experiment execution are stable before introducing agent coordination complexity.

### MVP #2

Embed the baseline into the intended agent architecture using SPADE. The focus is not yet sophisticated advisory behavior, but rather proving that the RL Orchestrator, message passing, and system lifecycle can run inside the target multi-agent execution model without changing the meaning of the environment interaction.

### MVP #3

Introduce LLM specialist agents as advisory components. At this stage, Plan Maker and Action Evolver can participate through controlled scheduling and blackboard-mediated coordination. The goal is to test whether specialist advice can improve decision quality or produce richer experimental traces without replacing the central RL policy.

### MVP #4

Introduce controlled dynamic reward shaping through the Reward Optimizer and stronger governance behavior through the Gatekeeper. This MVP marks the transition from a mostly advisory architecture to a more adaptive one, while preserving strict logging, versioning, and ablation support.

### MVP #5

Consolidate the full MARLA architecture as an integrated experimental platform. This phase should support end-to-end experiments, comparative ablations, runtime advisory scheduling, persistent experiment records, and a clear path to publication-quality evaluation.

## Confirmed Constraints

- MARLA is for simulated environments only.
- The system models attackers only in the current scope.
- Defenders are out of scope for now and belong to future work.
- The RL Orchestrator is the only agent allowed to interact directly with NASimEmu.
- Specialist agents are advisory rather than environment-facing.
- SPADE and XMPP are mandatory architectural dependencies.
- Core execution must support a single-machine setup.
- Python 3.10 is fixed due to environment constraints.
- The project must support both offline baseline mode and optional LLM-augmented mode.
- Every major subsystem must be independently ablatable and measurable.

## Deferred / Future Work

- Defender modeling and attacker-defender co-simulation.
- Hybrid Gatekeeper validation using rule-based and LLM-assisted methods.
- More advanced memory backends beyond in-memory blackboard plus structured logs.
- Richer scenario families, cross-scenario transfer, and curriculum strategies.
- Distributed execution beyond a single machine baseline.

## Risks / Drift Warnings

The biggest risk is architectural drift between the documented final MARLA vision and the currently implemented MVP. If the team starts treating early MVP behavior as if it already reflects the final architecture, the documentation will become misleading and experiments will be hard to interpret.

A second risk is allowing specialist agents or LLM reasoning to silently become the true control logic. MARLA is defined as centralized RL with advisory agents, so the RL Orchestrator must remain the owner of the acting policy unless this design contract is explicitly revised.

A third risk is forgetting that MARLA is staged by MVP. Features such as online reward shaping, Gatekeeper validation, and LLM runtime participation should not be assumed to exist from the first implementation phase.
