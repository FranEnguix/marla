# MARLA Problem Definition

## Purpose

Define precisely what problem MARLA is solving, what type of system it is, what outputs it must produce, and what is considered success or failure in both engineering and research terms.

## Final Design Decision

Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA) models an attacking system operating in a simulated network environment. It is not a defender, and it does not currently include defender agents, blue-team responses, or attacker-defender game dynamics. The system's role is to learn and execute offensive behavior in simulation for research purposes.

The project combines four goals at once: training, evaluation, planning, and simulated execution. Training refers to learning an action-selection policy for the RL Orchestrator. Evaluation refers to measuring the performance of the baseline and ablated variants. Planning refers to producing structured advisory proposals about what to do next. Simulated execution refers to actually applying selected actions inside NASimEmu.

A MARLA run must produce, at minimum, action traces, episode metrics, experiment metadata, and comparative records sufficient to support ablation studies and reproducible analysis.

## MVP Evolution

### MVP #1

Define the problem narrowly as baseline RL-based attack execution in NASimEmu. The main question is whether a single RL attacker can operate correctly in the simulator and produce stable training and evaluation outputs.

### MVP #2

Extend the problem to include the same attacker embedded in a message-based multi-agent runtime. The key question becomes whether the intended execution architecture can host the baseline behavior without semantic distortion.

### MVP #3

Extend the problem to include specialist advisory reasoning. The question is whether plan proposals and action refinement can improve decision support, trace quality, or experimental insight while preserving centralized RL control.

### MVP #4

Extend the problem to include online reward adaptation and stronger governance over proposals. The question becomes whether adaptive reward shaping and action filtering produce better learning behavior or better control properties.

### MVP #5

Define the full problem as evaluating an integrated, modular, ablatable research platform for offensive AI in simulation, combining centralized RL, advisory specialists, governance filtering, structured memory, and robust experiment analysis.

## Confirmed Constraints

- Agents are attackers only.
- Defenders are future work.
- The system exists for research in simulation, not real-world deployment.
- The project must support training, planning, evaluation, and simulated execution.
- The environment-facing action authority belongs only to the RL Orchestrator.
- Outputs must include metrics, traces, and comparison artifacts.
- LLM-generated content is allowed only within the simulation context and not as a real-world operational tool.

## Deferred / Future Work

- Formal attacker-defender formulations.
- Adversarial adaptation against simulated defenders.
- Multi-objective formulations balancing stealth, efficiency, and objective completion.
- Broader success definitions tied to curriculum or scenario families.
- Formal task taxonomies for specific offensive workflows beyond the current simulator action model.

## Risks / Drift Warnings

If this file is ignored, the project may be interpreted inconsistently by different contributors. Some may treat MARLA as a planning assistant, others as a true MARL system, and others as a direct LLM-driven attacker. That would undermine the validity of both the implementation and the paper.

A second risk is output ambiguity. If required outputs are not fixed, experiments may produce incompatible logs, and later ablations will not be comparable.

A third risk is scope inflation. Adding defenders, co-evolution, or real-world assumptions without formally revising the problem definition would create hidden research questions that the current architecture is not designed to answer.
