# MARLA Architecture Constraints

## Purpose

Capture the non-negotiable engineering and research constraints that the MARLA architecture must satisfy, including runtime limits, determinism rules, LLM boundaries, reproducibility expectations, and security boundaries.

## Final Design Decision

Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) must run on a single machine under Python 3.10 and remain compatible with NASimEmu, SPADE, and XMPP. The core system must be able to run fully offline in a baseline mode without LLM support. LLM integration is optional and must degrade gracefully if disabled, unavailable, or too slow.

The architecture supports two reproducibility modes. Strict Reproducibility Mode is the paper-grade mode and requires fixed seeds, deterministic environment behavior, fixed reward configuration, and either no LLM participation or replayed LLM outputs. Adaptive Research Mode allows LLM participation and dynamic reward shaping, but every deviation from strict reproducibility must be explicitly logged.

The RL Orchestrator is the only environment-facing component. LLMs may advise, critique, refine, or reshape, but they must not directly execute simulator actions. Runtime latency is bounded. The system must support hard timeouts and fallback behavior so that the environment loop does not stall indefinitely.

## MVP Evolution

### MVP #1

Prioritize simplicity and determinism. LLMs should be absent or fully disabled. The architecture should focus on reproducible RL baseline execution, stable environment integration, and structured logging.

### MVP #2

Add SPADE/XMPP lifecycle and messaging constraints. The architecture must now support agent startup, teardown, message routing, and local communication reliability without compromising baseline execution semantics.

### MVP #3

Allow bounded LLM runtime participation under hybrid scheduling. The architecture must now enforce timeout handling, structured output validation at the interface level, and disable mode for ablations.

### MVP #4

Introduce dynamic reward shaping as a constrained subsystem. Reward updates must be versioned, logged at every step where relevant, and separated from raw environment reward. This phase introduces stronger auditability requirements.

### MVP #5

Consolidate all constraints into a stable research platform. The final architecture must support offline baseline execution, LLM-augmented execution, reproducibility modes, comparative experiments, and publication-grade artifact generation.

## Confirmed Constraints

- Python 3.10 is fixed.
- Single-machine execution is the baseline assumption.
- CPU-only operation must be supported.
- GPU use is optional, not required.
- SPADE and local XMPP support are mandatory.
- Core execution must work offline.
- LLM use is optional and can be local or online.
- Baseline step latency target is below 200 ms.
- Typical non-LLM augmented steps should remain below 500 ms.
- LLM-triggered steps should target below 5 seconds.
- Hard timeout for LLM participation is 15 seconds.
- The environment loop must never block indefinitely due to LLM failure.
- Reproducibility matters and must be supported in strict mode.
- Every run must log seed, configuration, model version, reward version, and LLM mode.
- Offensive content generation is limited to simulation use only.
- Reward signals must distinguish among environment reward, shaped reward, and reward used for learning.

## Deferred / Future Work

- More precise latency budgets for different scenario sizes.
- Benchmarking memory and CPU envelopes by MVP.
- Database-backed experiment storage if JSON logs become insufficient.
- Formal deterministic replay support for LLM-assisted runs.
- Stronger sandboxing or policy controls for specialist outputs.

## Risks / Drift Warnings

The most serious risk is losing control of reproducibility. If dynamic reward shaping, stochastic LLM behavior, and asynchronous messaging are introduced without strict logging and mode separation, the resulting experiments may become impossible to interpret or reproduce.

Another risk is hidden runtime coupling. If the RL loop becomes dependent on LLM latency, MARLA will stop being robust as an experimental platform and become too fragile for controlled evaluation.

A final risk is weakening the security boundary. If simulation-only allowances are not enforced at the architectural level, contributors may accidentally design interfaces that are too close to real-world offensive tooling.
