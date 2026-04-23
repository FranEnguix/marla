# LLM Role in System

## Purpose

Define exactly what LLM-based specialists are allowed to do, what they must never do, how they participate at runtime, and how their use is bounded so that Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) does not drift into an uncontrolled LLM-driven architecture.

## Final Design Decision

LLM specialists are allowed to participate in the MARLA runtime under hybrid scheduling. They are not necessarily invoked at every step. Instead, each specialist has its own activation policy, such as on-demand invocation or periodic invocation every k steps. Each specialist must support timeout behavior and disable mode so that MARLA can run both LLM-assisted and non-LLM baselines.

LLMs may propose plans, refine candidate actions, critique poor options, summarize observations, and propose reward-shaping adjustments according to role. Within the simulation boundary only, they may generate executable action content or action refinements that correspond to simulator-valid behavior. They must not directly execute environment actions, directly own the acting policy, or bypass the RL Orchestrator. In early MVP stages, some validation behavior may be absent, but the architectural boundary remains in force.

## MVP Evolution

### MVP #1

No LLM dependency. The baseline must function without specialist invocation so that the core RL and simulator integration can be tested in isolation.

### MVP #2

The runtime may introduce LLM agent scaffolding or stubs, but the system should still not depend on LLMs for correct baseline operation.

### MVP #3

Introduce runtime advisory use for Plan Maker and Action Evolver under controlled scheduling. Outputs should already be structured enough to integrate safely with the rest of the system.

### MVP #4

Introduce Reward Optimizer into the LLM-assisted runtime and strengthen the governance path for specialist outputs. Timeouts, logging, and traceability become increasingly important.

### MVP #5

Support the full LLM-assisted architecture with hybrid scheduling, optional disable mode, and robust logging. The system should make it easy to compare LLM-disabled, partially enabled, and fully enabled configurations.

## Confirmed Constraints

- LLM participation is optional, not mandatory for core operation.
- Core MARLA must run offline without requiring an LLM.
- LLM specialists are runtime advisors, not environment executors.
- LLM outputs may be role-specific: planning, refinement, reward suggestions, state summarization, and critique.
- LLM-triggered steps must obey timeout rules.
- Hard timeout is 15 seconds.
- The system must provide fallback behavior when LLM invocation fails or times out.
- LLM-generated content is permitted only in simulation context.
- Specialist outputs must never bypass the RL Orchestrator's execution authority.
- Disable mode must exist for ablation and reproducibility purposes.

## Deferred / Future Work

- More precise prompt contracts and prompt versioning rules.
- Local-model versus remote-model deployment strategies.
- Stronger structured-output guarantees and schema-level validation.
- Hybrid Gatekeeper plus LLM validation in later architecture phases.
- Cost accounting for LLM usage in long experiments.

## Risks / Drift Warnings

The main risk is role inflation. If LLM specialists begin making de facto final decisions, MARLA will stop being a centralized RL system and become an opaque LLM-controlled architecture.

A second risk is runtime fragility. If experiments depend on LLM latency or availability, the platform becomes hard to benchmark and compare.

A third risk is boundary erosion around offensive generation. Simulation-only permission must remain explicit and enforced in documentation, interfaces, and implementation.
