# Open Questions

## Purpose

Track unresolved design and implementation questions in a disciplined way so that Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) evolves intentionally rather than through undocumented assumptions. This file also records which foundational decisions are already closed.

## Final Design Decision

Core architectural identity is no longer open. MARLA is defined as a centralized RL system augmented by advisory agents, implemented with SPADE/XMPP, operating in NASimEmu, and evolving through staged MVPs. This file is therefore not a place for re-litigating the core design contract unless a deliberate architecture revision is proposed.

Instead, this file should hold scoped open questions about implementation details, evaluation methodology, policy design, scheduling heuristics, reward mechanisms, schema definitions, logging policies, and future extensions. Every open question should be actionable, attributable, and linked to an MVP or roadmap milestone.

## MVP Evolution

### MVP #1

Focus open questions on the baseline: environment wrapper details, action indexing, training loop shape, initial metrics, and logging format.

### MVP #2

Add open questions related to agent runtime behavior, local XMPP setup, transport reliability, lifecycle management, and message envelope design.

### MVP #3

Track unresolved choices around specialist trigger policies, prompt design, proposal schema structure, and blackboard semantics.

### MVP #4

Track questions about reward adaptation mechanisms, Gatekeeper rule taxonomy, versioning strategy, and fairness of experimental comparison.

### MVP #5

Focus on research-grade questions such as ablation matrix completeness, scenario diversity, generalization claims, and long-term maintainability of the integrated platform.

## Confirmed Constraints

- This file must not replace the authoritative design files.
- Questions should be linked to concrete implementation or research decisions.
- Every question should identify status, owner, and impact.
- Closed decisions should be recorded separately from open items.
- MVP linkage is required whenever possible.
- Questions that affect experiment validity should be prioritized.

## Deferred / Future Work

- Introduce a more formal decision log if the project grows in team size.
- Add references to issue trackers or ADR-style records.
- Classify questions by severity, dependency, and experiment impact.
- Add review cadence for unresolved items.

## Risks / Drift Warnings

If open questions are not tracked explicitly, contributors will answer them informally in code, which leads to inconsistent implementations and weak experimental claims.

A second risk is reopening already-closed architecture decisions without a formal revision process. That would destabilize both the codebase and the documentation.

A final risk is allowing low-impact questions to crowd out high-impact ones. This file should help focus attention, not become a dumping ground for every uncertainty.
