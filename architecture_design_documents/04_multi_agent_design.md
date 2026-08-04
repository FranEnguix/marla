# Multi-Agent Design

## Purpose

Define the MARLA multi-agent architecture, including agent roles, communication patterns, coordination model, authority boundaries, memory semantics, and staged evolution across MVPs.

## Final Design Decision

Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) uses a message-based multi-agent architecture implemented with SPADE and XMPP, but the acting policy remains centralized. The RL Orchestrator is the sole owner of environment-facing execution. The other agents are specialists that provide advisory functions rather than independent execution authority.

The specialist roles are:
- Plan Maker, which produces high-level attack directions and strategic suggestions.
- Action Evolver, which refines or mutates candidate actions or action representations when requested.
- Reward Optimizer, which proposes reward-shaping adjustments under controlled conditions.
- Gatekeeper, which filters, constrains, transforms, or rejects proposals, but never generates decisions of its own.

Shared memory is modeled as a two-layer system: an in-memory episode blackboard for fast coordination and a persistent experiment store for logs, traces, seeds, reward versions, and other cross-episode artifacts. Direct communication among specialist agents is not allowed in the MVP design; coordination flows through the blackboard and, in later phases, through Gatekeeper-mediated control.

## MVP Evolution

### MVP #1

Implement only the minimal acting role required for baseline RL execution. The architectural target may be acknowledged, but specialist roles need not yet be behaviorally present.

### MVP #2

Run the acting system inside SPADE/XMPP so that the intended agent runtime becomes real. The focus is on process boundaries, lifecycle management, message flow, and local orchestration rather than on sophisticated specialist logic.

### MVP #3

Introduce Plan Maker and Action Evolver as advisory agents. They consume blackboard state, produce proposals, and operate under controlled scheduling. The RL Orchestrator remains the final selector of actions.

### MVP #4

Introduce Reward Optimizer and a stronger Gatekeeper. Gatekeeper behavior moves beyond pass-through, and reward adaptation becomes part of the runtime architecture. Logging and proposal governance become more important in this phase.

### MVP #5

Consolidate the full agent ecosystem with stable coordination patterns, blackboard semantics, persistent experiment records, and clear ablation support. The final design should make it easy to activate or deactivate specialist roles without breaking the rest of the system.

## Confirmed Constraints

- SPADE/XMPP is mandatory.
- RL Orchestrator is the only agent allowed to interact directly with NASimEmu.
- Plan Maker, Action Evolver, and Reward Optimizer are advisory roles.
- Gatekeeper never creates decisions; it only filters, constrains, or transforms.
- Direct specialist-to-specialist communication is not allowed in MVP stages.
- Shared memory has both transient episode scope and persistent experiment scope.
- Coordination must remain traceable and auditable.
- The architecture must support ablation by enabling or disabling individual specialist roles.

## Deferred / Future Work

- Direct specialist communication under tightly controlled policies.
- Distributed deployment across more than one machine.
- Alternative memory backends such as Redis or a relational/graph store.
- Hybrid Gatekeeper behavior using rule-based and LLM-assisted validation.
- Richer negotiation or revision protocols between specialists and Gatekeeper.

## Risks / Drift Warnings

A major risk is turning the architecture into de facto decentralized control while still claiming centralized RL. If specialists start selecting actions implicitly, the documented identity of MARLA will no longer match the implementation.

Another risk is communication sprawl. Allowing ad hoc direct messages among specialists too early would reduce traceability and make debugging far harder.

A final risk is treating the blackboard as an undefined dumping ground. If shared memory semantics are not disciplined, coordination will become opaque and difficult to reproduce.
