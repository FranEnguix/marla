# NASimEmu Notes

## Purpose

Summarize the NASimEmu environment as used by MARLA, including what the simulator provides, how observations and actions appear, what information is safe to use in decision logic, and what limitations must be respected.

## Final Design Decision

Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) treats NASimEmu as the external environment and source of legal actions, observations, episode progression, and base rewards. Decision logic must use only the standard observation exposed by the simulator and any MARLA-derived summaries built from that observation. Hidden or debug-only fields, including true state views such as `s_true`, are forbidden in decision logic and may only be used for debugging, offline inspection, or post hoc analysis.

The simulator provides a large legal action space composed of scans, exploits, privilege escalations, and related operations over hosts and subnets. MARLA may internally rank, constrain, or filter that action space, but the legal action definitions belong to the environment.

## MVP Evolution

### MVP #1

Use NASimEmu in the most direct way possible. Consume the standard observation, access the legal action space, execute actions through a minimal wrapper, and record the raw step outputs needed for baseline training and evaluation.

### MVP #2

Keep the environment semantics unchanged while moving access behind the RL Orchestrator and SPADE-based runtime. The goal is architectural integration, not environment reinterpretation.

### MVP #3

Introduce MARLA-side semantic summaries and advisory reasoning based on the observation, but still prohibit hidden-state leakage. Specialists may reason over structured summaries derived from the legal observation.

### MVP #4

Allow reward shaping and stronger proposal filtering while preserving the simulator as the authority over legal actions, raw rewards, and episode termination. MARLA may reinterpret reward but not simulator legality.

### MVP #5

Stabilize the environment interface for all experiments, ensuring that observations, actions, derived summaries, and logging are consistently represented across ablations and scenario families.

## Confirmed Constraints

- NASimEmu is the environment provider for MARLA.
- Scenario definitions include subnets, topology, services, exploits, privilege escalation actions, costs, and related environment metadata.
- Observations are partial and must be treated as the only decision-safe state input from the environment.
- The legal action space can be large and is environment-defined.
- Step results include reward, done flag, and auxiliary `info`.
- Hidden state fields such as `s_true` are never allowed in MARLA decision logic.
- Debug and offline analysis may inspect hidden fields, but training and inference may not use them.
- MARLA may derive higher-level summaries from observations, but those summaries must remain grounded in decision-safe inputs.
- Emulator limitations and action semantics must be preserved rather than rewritten by advisory agents.

## Deferred / Future Work

- A formal mapping from raw observation tensors to semantic world-model objects.
- Scenario family taxonomy and scenario metadata normalization.
- Standardized wrappers for observation summarization and action indexing.
- Environment stress tests across increasing topology and host-space sizes.
- Better documentation of NASimEmu-specific corner cases relevant to MARLA.

## Risks / Drift Warnings

The main risk is state leakage. If true state or debug-only fields enter training, evaluation results will be invalid and difficult to defend in a paper.

A second risk is overinterpreting the simulator. MARLA can build derived summaries, but those summaries must not silently assume information that is not actually observable.

A third risk is letting MARLA-side action filtering obscure the distinction between environment legality and architecture preference. Contributors must always be able to tell whether an action is impossible in NASimEmu or merely discouraged by MARLA.
