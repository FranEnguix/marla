# RL Design

## Purpose

Define the reinforcement learning formulation used by MARLA, including policy ownership, training and inference behavior, reward handling, exploration constraints, and the relationship between RL and the advisory agents.

## Final Design Decision

Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) is a single-agent reinforcement learning system. The RL Orchestrator owns the acting policy and is the only component that may execute actions in NASimEmu. Specialist agents do not own competing policies for environment interaction and are not treated as independent RL actors in the current design.

The RL policy may consume environment observations and MARLA-generated advisory context, but the system must preserve a clear distinction between the learned policy and the advisory layer. Reward handling is split into three concepts: the raw environment reward from NASimEmu, the MARLA-shaped reward proposed or derived at runtime, and the final reward signal used in policy updates. Reward shaping may change online during both training and inference, but only in the later MVP stages and only with full versioning and step-level logging.

## MVP Evolution

### MVP #1

Implement a plain RL baseline with no reward optimizer and no required LLM participation. The priority is to establish a stable training loop, action execution, replayability, and baseline performance metrics.

### MVP #2

Keep the learning formulation unchanged while embedding the policy inside the SPADE/XMPP runtime. The purpose is architectural migration, not algorithmic expansion.

### MVP #3

Allow advisory inputs from specialists such as Plan Maker and Action Evolver. The RL Orchestrator may use these inputs as context or proposal sets, but the central acting policy must remain the final selector.

### MVP #4

Introduce dynamic reward shaping and stronger action governance. Reward Optimizer may influence the reward signal under strict logging, and Gatekeeper may constrain candidate actions before final selection.

### MVP #5

Stabilize the integrated RL design as a modular research platform that supports baseline training, advisory augmentation, reward adaptation, structured ablations, and comparative evaluation.

## Confirmed Constraints

- MARLA is not true multi-agent RL.
- RL Orchestrator owns the policy.
- The action chosen for simulator execution must always originate from the Orchestrator.
- Reward shaping may change online in training and inference, but only with explicit versioning and logging.
- Early MVPs must support a reward-static baseline.
- Exploration may be constrained by architecture rules and validated action subsets.
- Every RL variant must remain independently ablatable and measurable.
- The system must support clear comparisons such as RL only, RL plus planning, RL plus planning and action evolution, RL plus reward shaping, and full MARLA.

## Deferred / Future Work

- Detailed algorithm choice and policy architecture selection.
- How advisory context is encoded for policy consumption.
- Formal curriculum strategies across scenario families.
- Action masking and constraint-aware exploration policies.
- More detailed definitions of milestone-based rewards and auxiliary metrics.

## Risks / Drift Warnings

The biggest risk is blurring the line between learned policy and external advice. If the RL Orchestrator merely follows specialist outputs, then MARLA would no longer be a centralized RL system in a meaningful sense.

A second risk is uncontrolled reward shaping. Dynamic reward changes can invalidate comparisons if the system does not log exactly what reward was active and how it was used.

A third risk is weak ablation design. If variants cannot be turned on and off cleanly, the eventual research claims will be difficult to justify.
