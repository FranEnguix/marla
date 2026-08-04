# MARLA: Complete Implementation Specification

**Package:** `marla`  
**CLI command:** `marla`  
**Python:** 3.10 only  
**Primary environment:** NASimEmu simulation  
**Primary comparison:** recurrent PPO baseline vs recurrent PPO assisted by a Plan Maker through a Gatekeeper  
**Deployment:** local single-process SPADE runtime or distributed multi-machine SPADE/XMPP deployment

This document is the implementation contract for the first MARLA release. It is intended to be read before coding begins.


# 1. Project Goal

MARLA (Multi-Agent Reinforcement Learning Architecture) is a research platform for studying whether a reinforcement-learning cyber agent can improve its NASimEmu decisions by selectively consulting external language-model-based support agents.

The first release must support two matched variants.

## Baseline

```text
RL Orchestrator <-> NASimEmu
```

The baseline contains only:

- RL Orchestrator;
- recurrent PPO model;
- NASimEmu simulator;
- experiment metrics.

It does **not** start the Gatekeeper or Plan Maker.

## Assisted variant

```text
RL Orchestrator -> Gatekeeper -> Plan Maker
RL Orchestrator <- Gatekeeper <- Plan Maker
RL Orchestrator <-> NASimEmu
```

The assisted variant adds:

- a learned binary query gate;
- Gatekeeper-mediated Plan Maker consultation;
- a fixed consultation cost;
- confidence-score processing;
- a learned trust coefficient;
- a learned advice scale.

Only the RL Orchestrator is trained. The Plan Maker is frozen.


# 2. Authoritative Decisions

1. MARLA targets Python `>=3.10,<3.11`.
2. MARLA is published as the PyPI package `marla`.
3. The installed entry point is the `marla` command.
4. The CLI uses Typer.
5. NASimEmu simulation is the first-release priority.
6. Emulation is secondary and must not delay simulation correctness.
7. Execution modes are `local` and `distributed`.
8. Local mode runs all configured SPADE agents in one `spade.run()` main function and one asyncio event loop.
9. Distributed mode starts only agents named with repeatable `--agent` options.
10. Agents use SPADE messages and JIDs. Distributed traffic traverses a reachable XMPP server.
11. There is no Coordinator component.
12. The RL Orchestrator coordinates startup and shutdown.
13. Direct lifecycle messages are allowed: `READY_CHECK`, `READY`, `START_EXPERIMENT`, `STOP_EXPERIMENT`, and `EXPERIMENT_FAILED`.
14. All Plan Maker advisory requests and responses pass through the Gatekeeper.
15. The Gatekeeper is absent from the baseline and mandatory when a support agent is enabled.
16. There is no Blackboard.
17. The RL Orchestrator owns episode, recurrent, rollout, and metrics state.
18. The Plan Maker does not use knowledge or trajectories from previous MARLA runs.
19. The Plan Maker uses the current observation, current legal actions, experiment objective, and static versioned NASimEmu knowledge.
20. The static knowledge is retrieved with deterministic lightweight RAG.
21. Torch Geometric is used for graph encoding.
22. The RL model is recurrent PPO with a graph encoder, GRU, dynamic candidate-action scorer, query gate, trust head, and value critic.
23. Consultation is part of one compound decision; it is not a separate NASimEmu action.
24. NASimEmu advances exactly once per compound decision.
25. Consultation uses a fixed configured cost in the initial experiments.
26. No elapsed response timeout is used.
27. A healthy connection waits for the response; disconnect or agent shutdown fails the run.
28. Response elapsed time is logged.
29. Gatekeeper correction is bounded by `max_schema_revisions`.
30. Exhausting revisions produces `payload: null`; the base policy is used and the cost is still charged.
31. Device values are `cpu`, `gpu`, and `auto`, applying to locally executed ML models.
32. Metrics use ordinary CSV, JSON, and text logs.


# 3. Components

## 3.1 RL Orchestrator

The RL Orchestrator owns:

- NASimEmu creation, reset, and step calls;
- current visible observation;
- legal action construction;
- stable action IDs;
- Torch Geometric graph conversion;
- recurrent PPO model;
- GRU hidden state;
- rollout buffer and GAE;
- PPO optimization;
- query decision;
- final environment-action decision;
- lifecycle coordination;
- checkpoints;
- central metrics files.

It is the only component that interacts directly with NASimEmu.

## 3.2 Gatekeeper

The Gatekeeper:

- validates advisory request schemas;
- forwards valid requests to the Plan Maker;
- validates Plan Maker responses;
- verifies run/request/action-ID correlation;
- requests schema corrections;
- delivers one accepted or rejected advisory artifact to the RL Orchestrator.

It does not validate or execute final NASimEmu actions or `FINISH`.

## 3.3 Plan Maker

The Plan Maker:

- loads a static NASimEmu knowledge base;
- deterministically retrieves relevant rules;
- constructs a prompt;
- invokes a configured local or remote language model;
- returns one independent score in `[0,1]` for every legal action ID.

It has no reward, critic, PPO gradient, mutable cross-run memory, hidden simulator access, or direct environment access.

## 3.4 NASimEmu Adapter

The adapter isolates library-specific behavior and exposes:

```python
class NasimEmuAdapter:
    def reset(self, seed: int | None = None) -> "EnvironmentState": ...
    def legal_actions(self, state: "EnvironmentState") -> list["ActionDescriptor"]: ...
    def to_pyg_data(self, state: "EnvironmentState") -> "torch_geometric.data.Data": ...
    def step(self, action: "ActionDescriptor") -> "TransitionResult": ...
    def objective_satisfied(self, state: "EnvironmentState") -> bool: ...
```


# 4. Execution and Deployment

## 4.1 Local mode

```yaml
execution:
  mode: local
```

Local mode:

- starts one Python process;
- creates all configured agents inside one async `main`;
- starts them from one `spade.run(main())`;
- retains separate JIDs and SPADE behaviors;
- prohibits direct inter-agent Python method calls;
- uses the same messages and schemas as distributed mode;
- keeps NASimEmu and metrics in the RL Orchestrator.

Baseline local process:

```text
SPADE main
└── RL Orchestrator
    └── NASimEmu
```

Assisted local process:

```text
SPADE main
├── RL Orchestrator
│   └── NASimEmu
├── Gatekeeper
└── Plan Maker
```

Blocking Plan Maker inference must not block the shared asyncio loop. A blocking local backend should run via `asyncio.to_thread`; remote APIs should use async clients where possible.

## 4.2 Distributed mode

```yaml
execution:
  mode: distributed
```

Each invocation starts selected agents:

```bash
marla run experiment.yaml --agent rl_orchestrator --agent gatekeeper
```

On another machine:

```bash
marla run experiment.yaml --agent plan_maker_1
```

Rules:

- `--agent` is repeatable;
- aliases or full configured JIDs are accepted;
- duplicates and unknown values fail validation;
- local mode rejects `--agent`;
- distributed mode requires at least one `--agent`;
- every process loads the same configuration;
- distributed mode requires an explicit shared `experiment.run_id`;
- only the process containing `rl_orchestrator` starts NASimEmu and writes central metrics;
- all machines connect to the same reachable XMPP server;
- all messages carry the shared run ID and mismatches are rejected.


# 5. Lifecycle without a Coordinator

The RL Orchestrator coordinates the experiment.

## Startup

1. Each process loads and validates the same YAML.
2. Each selected agent initializes SPADE and local resources.
3. Local models resolve the configured device and load before readiness.
4. Non-Orchestrator agents wait.
5. RL Orchestrator performs presence checks.
6. RL Orchestrator sends `READY_CHECK`.
7. Required agents return `READY` with run ID, alias, JID, schema version, model version, and resolved device.
8. RL Orchestrator verifies all required participants.
9. RL Orchestrator sends `START_EXPERIMENT`.
10. Training or evaluation begins.

## Normal shutdown

1. Save final checkpoint.
2. Flush metrics.
3. Send `STOP_EXPERIMENT`.
4. Stop behaviors and disconnect agents.
5. Write `summary.json`.
6. Exit successfully.

## Failure

An explicit disconnect, agent shutdown, model initialization failure, XMPP authentication error, or unrecoverable environment error fails the run. The detecting participant sends `EXPERIMENT_FAILED` when possible. The RL Orchestrator records the failure, sends `STOP_EXPERIMENT`, flushes metrics, and exits nonzero.

There is no elapsed-time response timeout. Elapsed response duration is nevertheless measured.


# 6. No Blackboard

The RL Orchestrator uses private Python state.

```python
@dataclass
class EpisodeContext:
    run_id: str
    episode_id: int
    observation_id: str
    observation: object
    graph_observation: object
    legal_actions: list["ActionDescriptor"]
    previous_action_id: str | None
    previous_training_reward: float
    previous_query: bool
    environment_step: int
    done: bool
```

The Gatekeeper stores only transient request correlation:

```python
@dataclass
class PendingRequest:
    run_id: str
    request_id: str
    source_observation_id: str
    episode_id: int
    environment_step: int
    expected_agent_alias: str
    legal_action_ids: tuple[str, ...]
    schema_revision_count: int
```

A pending entry is removed after final acceptance or rejection.


# 7. Stable Actions and FINISH

Every current action has a semantic stable ID. Do not correlate advice by vector position alone.

Examples:

```text
service-scan:host-3
process-scan:host-3
exploit:host-3:exploit-e2
privilege-escalation:host-3:privesc-p1
finish
```

```python
@dataclass(frozen=True)
class ActionDescriptor:
    action_id: str
    action_type: str
    target_key: str | None
    parameters: dict[str, object]
    is_finish: bool
```

`FINISH` is a MARLA wrapper action, always included in both the RL candidate set and Plan Maker request.

- If the objective is satisfied, apply the configured completion reward.
- Otherwise apply the configured premature-finish penalty.
- It immediately terminates the episode.
- Gatekeeper does not validate it as a final action.


# 8. Plan Maker Knowledge and RAG

The Plan Maker is a frozen function:

\[
M_\psi(o_t, \mathcal{A}_t, K) \rightarrow c_t
\]

where each score satisfies:

\[
c_{t,i}\in[0,1]
\]

Scores do not need to sum to one.

## Knowledge policy

Allowed:

- current visible observation;
- public descriptions of current legal actions;
- configured experiment objective;
- static NASimEmu rules.

Forbidden:

- prior-run trajectories;
- previous evaluation outcomes;
- hidden vulnerabilities or services;
- hidden success probabilities;
- future state;
- mutable cross-run memory.

## Static knowledge examples

The packaged knowledge should explain:

- scan services before choosing service-dependent exploits when services are unknown;
- process scans may be prerequisites for privilege escalation;
- exploit applicability depends on visible prerequisites;
- root access can be more valuable than user access when the objective benefits;
- if user access exists and root remains useful, prioritize required process scan or privilege escalation;
- balance action probability, cost, prerequisites, and expected access;
- give low confidence to actions with unsatisfied visible prerequisites;
- give `FINISH` high confidence only when the objective is satisfied or continuing has poor expected value.

## Deterministic lightweight RAG

Use:

```text
src/marla/knowledge/nasimemu_rules.yaml
```

Example:

```yaml
version: nasimemu-rules-v1
rules:
  - id: scan-before-exploit
    triggers:
      observation_flags: [services_unknown]
      legal_action_types: [service_scan, exploit]
    text: >
      When services on a reachable host are unknown, prioritize service
      scanning before selecting a service-dependent exploit.

  - id: user-to-root
    triggers:
      observation_flags: [user_access_present, root_access_missing]
      legal_action_types: [process_scan, privilege_escalation]
    text: >
      When user access exists and root access is useful for the objective,
      prioritize required information gathering and privilege escalation.
```

The retriever evaluates deterministic predicates and returns a stable ordered rule subset. Retrieved rule IDs are stored with consultation artifacts.


# 9. Plan Maker Prompt and Messages

## Request body

```json
{
  "schema_version": "1.0",
  "run_id": "ppo-plan-maker-seed-42",
  "request_id": "request-000001",
  "episode_id": 47,
  "step": 12,
  "source_observation_id": "observation-47-12",
  "objective": {
    "type": "capture_target",
    "description": "Obtain the configured target access"
  },
  "observation": {},
  "legal_actions": [
    {
      "action_id": "service-scan:host-3",
      "type": "service_scan",
      "target": "host-3",
      "parameters": {}
    },
    {
      "action_id": "finish",
      "type": "finish",
      "target": null,
      "parameters": {}
    }
  ]
}
```

## Recommended prompt

```text
You are the MARLA Plan Maker. You provide advisory confidence scores for
currently legal NASimEmu actions. You do not execute actions.

NASIMEMU KNOWLEDGE
{retrieved_rules}

EXPERIMENT OBJECTIVE
{objective}

CURRENT VISIBLE OBSERVATION
{observation}

LEGAL ACTIONS
{legal_actions}

Assign an independent confidence in the inclusive range [0,1] to every
supplied action ID. Scores are not required to sum to one. Do not add or
omit action IDs. Return strict JSON and no explanatory text.
```

## Response body

```json
{
  "schema_version": "1.0",
  "run_id": "ppo-plan-maker-seed-42",
  "request_id": "request-000001",
  "scores": {
    "service-scan:host-3": 0.81,
    "finish": 0.03
  },
  "model_version": "plan-maker-slm-v1",
  "prompt_version": "nasimemu-plan-maker-v1",
  "knowledge_version": "nasimemu-rules-v1",
  "inference_latency_ms": 812,
  "retrieved_rule_ids": ["scan-before-exploit"]
}
```


# 10. SPADE Message Rules

Every message includes metadata:

```text
performative
message_type
schema_version
run_id
conversation_id
request_id
sender_alias
receiver_alias
```

Use SPADE `thread` or equivalent conversation metadata plus body-level `request_id`.

Advisory route:

```text
RL Orchestrator -> Gatekeeper -> Plan Maker
Plan Maker -> Gatekeeper -> RL Orchestrator
```

The RL Orchestrator ignores direct Plan Maker advisory messages.

The Gatekeeper validates:

- JSON structure;
- schema version;
- run ID;
- request ID;
- expected sender;
- exact legal action-ID coverage;
- absence of unknown IDs;
- numeric score types;
- all scores in `[0,1]`;
- required model/prompt/knowledge versions.

On invalid response, Gatekeeper asks for correction. After `max_schema_revisions`, it emits:

```json
{
  "response_status": "schema_rejected",
  "validation": {
    "status": "rejected",
    "reason": "maximum_schema_revisions_exceeded"
  },
  "payload": null
}
```

The RL Orchestrator then uses base logits and still charges consultation cost.


# 11. Recurrent PPO Architecture

```text
NASimEmu observation
        |
        v
Torch Geometric graph encoder
        |
        v
global graph embedding
        |
        v
GRU recurrent state
        |
        +--> dynamic base action scorer
        +--> binary query gate
        +--> value critic

After accepted advice:
base logits + beta * alpha * normalized advice -> final logits
```

Only the RL Orchestrator parameters are optimized.


# 12. Torch Geometric Graph Encoder

At step \(t\):

\[
G_t=(V_t,E_t,X_t)
\]

- \(V_t\): visible host/subnet nodes;
- \(E_t\): visible membership or connectivity;
- \(X_t\): visible node features.

Candidate node features:

- node type;
- discovered/reachable flags;
- access level: none, user, root;
- known service and process counts;
- scan-status flags;
- target-objective flag when visible;
- normalized visible numeric fields.

Do not encode hidden simulator properties.

Use a first GraphSAGE implementation:

```python
class GraphEncoder(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, layers: int):
        super().__init__()
        self.layers = nn.ModuleList(
            [SAGEConv(input_dim, hidden_dim)]
            + [SAGEConv(hidden_dim, hidden_dim) for _ in range(layers - 1)]
        )

    def forward(self, data: Batch) -> tuple[Tensor, Tensor]:
        h = data.x
        for layer in self.layers:
            h = F.relu(layer(h, data.edge_index))
        g = global_mean_pool(h, data.batch)
        return h, g
```

The adapter supplies `node_key_to_index` so action targets can retrieve node embeddings. Non-target actions such as `FINISH` use a learned no-target embedding.

Torch Geometric `Batch` is used for recurrent PPO minibatches.


# 13. GRU State and Dynamic Action Scoring

Recurrent state:

\[
z_t=\operatorname{GRUCell}(x_t,z_{t-1})
\]

with:

\[
x_t=[g_t,E(a_{t-1}),\tilde r_{t-1},q_{t-1}]
\]

At episode start:

- hidden state is zero;
- previous action is a learned start token;
- previous reward is zero;
- previous query is zero.

For each action:

\[
e_{t,i}=f_{action}(E_{type}(a_{t,i}),h_{target},E_{parameters})
\]

Base logit:

\[
b_{t,i}=w^T\tanh(W_z z_t+W_e e_{t,i})
\]

Base policy:

\[
\pi_t^0(a_{t,i})=\operatorname{softmax}(b_t)_i
\]

Only legal actions are scored. Padded minibatch entries are masked with a large negative logit and excluded from entropy and log-probability calculations.


# 14. Learned Query Gate

The query decision occurs before seeing advice.

Base entropy:

\[
H_t^0=-\sum_i\pi_t^0(a_i)\log\pi_t^0(a_i)
\]

Top-two margin:

\[
\Delta_t^0=b_{t,(1)}-b_{t,(2)}
\]

Use zero if only one action exists.

Query probability:

\[
p_t^q=\sigma(f_q[z_t,H_t^0,\Delta_t^0,N_t,\kappa])
\]

Sample:

\[
q_t\sim\operatorname{Bernoulli}(p_t^q)
\]

- `q_t=0`: use base action policy.
- `q_t=1`: synchronously consult through Gatekeeper.

Configuration:

```yaml
consultation:
  mode: learned
  cost: 0.10
  max_schema_revisions: 3
```

Baseline:

```yaml
consultation:
  mode: disabled
```


# 15. Advice Processing and Learned Trust

Confidence values are not assumed calibrated probabilities.

Clip:

\[
c'_{t,i}=\operatorname{clip}(c_{t,i},\epsilon,1-\epsilon)
\]

Convert to log-odds:

\[
u_{t,i}=\log\frac{c'_{t,i}}{1-c'_{t,i}}
\]

Normalize:

\[
\hat c_{t,i}=\frac{u_{t,i}-\bar u_t}{\sigma(u_t)+\epsilon}
\]

If variance is effectively zero, normalized advice is all zeros. If correlation is undefined, use zero.

Advice summary:

\[
S(c_t)=[mean,std,max,top1-top2,H(softmax(u_t))]
\]

Agreement features:

- whether base and Plan Maker top actions agree;
- correlation between base logits and Plan Maker evidence.

Trust:

\[
\beta_t=\sigma(f_\beta[z_t,S(c_t),A_t])
\]

Global advice scale:

\[
\alpha=\operatorname{softplus}(\bar\alpha)
\]

Adjusted logits:

\[
b^{PM}_{t,i}=b_{t,i}+\beta_t\alpha\hat c_{t,i}
\]

Advised policy:

\[
\pi_t^{PM}=\operatorname{softmax}(b_t^{PM})
\]

Rejected advice forces `beta=0` for that step.


# 16. Compound Decision, Reward, and PPO

Conditional action policy:

\[
\pi(a_t|z_t,q_t,c_t)=
\begin{cases}
\pi_t^0(a_t),&q_t=0\\
\pi_t^{PM}(a_t),&q_t=1
\end{cases}
\]

The RL Orchestrator samples one action and advances NASimEmu exactly once.

Training reward:

\[
\tilde r_t=r_t^{NASimEmu}-\kappa q_t
\]

The cost is charged on every attempted consultation, including rejected responses or advice later ignored by low trust.

Joint log-probability:

\[
\log p_t=
\log\pi^q(q_t|z_t)
+(1-q_t)\log\pi^0(a_t|z_t)
+q_t\log\pi^{PM}(a_t|z_t,c_t)
\]

PPO ratio:

\[
\rho_t=\exp(\log p_t^{new}-\log p_t^{old})
\]

Actor loss:

\[
L_{actor}=-E[\min(\rho_t\hat A_t,clip(\rho_t,1-\epsilon,1+\epsilon)\hat A_t)]
\]

Total minimized loss:

\[
L=L_{actor}+c_vL_{value}-c_{e,q}H(\pi^q)-c_{e,a}H(\pi^a)
\]

Use GAE, advantage normalization, gradient clipping, optional value clipping, and approximate-KL monitoring.


# 17. Rollout Buffer

Each compound step stores:

```text
run_id
episode_id
environment_step
observation_id
graph_data
node_key_to_index
legal_action_descriptors
legal_action_ids_in_order
initial_gru_hidden_state
previous_action_features
previous_training_reward
previous_query
sampled_query
old_query_probability
old_query_log_probability
plan_maker_scores_in_action_order
plan_maker_validation_status
plan_maker_response_status
plan_maker_request_id
plan_maker_artifact_path
normalized_advice
beta
alpha
base_logits
final_logits
selected_action_index
old_action_log_probability
old_joint_log_probability
critic_value
nasimemu_reward
consultation_cost
training_reward
terminated
truncated
```

Critical invariant:

> The exact Plan Maker score vector collected during rollout must be reused during every PPO epoch. Never call the Plan Maker while recomputing new policy probabilities.

For recurrent training:

- split rollouts into ordered sequences;
- store initial hidden state for each sequence;
- never propagate hidden state across episode boundaries;
- pad short sequences;
- mask padded entries;
- shuffle sequences, not individual transitions.


# 18. Device Handling

```yaml
device: auto  # cpu | gpu | auto
```

`cpu`:

- all locally executed ML models use CPU.

`gpu`:

- all required locally executed ML models must initialize CUDA;
- startup fails if CUDA or the requested GPU is unavailable;
- startup fails if model allocation fails;
- an agent must not send `READY` after failed initialization.

`auto`:

- use CUDA when locally available;
- otherwise use CPU.

Remote API models are unaffected. In distributed mode, each process resolves device independently. Record requested and resolved device for each local model.


# 19. Configuration Examples

## Assisted

```yaml
schema_version: "1.0"

experiment:
  name: ppo_plan_maker
  run_id: ppo-plan-maker-seed-42
  phase: training
  seed: 42

execution:
  mode: local

device: auto

xmpp:
  server: xmpp.example.org

environment:
  mode: simulation
  scenario: scenarios/university.yaml
  max_episode_steps: 200

objective:
  type: capture_target
  description: Obtain the configured target access

policy:
  algorithm: recurrent_ppo
  graph_encoder:
    type: graphsage
    hidden_size: 128
    layers: 2
  action_encoder:
    hidden_size: 128
    action_type_embedding_size: 32
  recurrent:
    hidden_size: 128
    sequence_length: 32
  ppo:
    rollout_steps: 512
    epochs: 4
    minibatch_sequences: 8
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    value_coefficient: 0.5
    query_entropy_coefficient: 0.01
    action_entropy_coefficient: 0.01
    max_grad_norm: 0.5
    learning_rate: 0.0003

consultation:
  mode: learned
  cost: 0.10
  max_schema_revisions: 3

rl_orchestrator:
  alias: rl_orchestrator
  jid: rl-orchestrator@xmpp.example.org

gatekeeper:
  alias: gatekeeper
  jid: gatekeeper@xmpp.example.org

agents:
  - alias: plan_maker_1
    jid: plan-maker@xmpp.example.org
    role: plan_maker
    model:
      backend: local
      name: configured-small-language-model
    prompt_version: nasimemu-plan-maker-v1
    knowledge:
      path: package://marla/knowledge/nasimemu_rules.yaml
      version: nasimemu-rules-v1

metrics:
  output_directory: runs
  record_decisions: true

reproducibility:
  deterministic_torch: true
```

## Baseline

Remove `gatekeeper` and `agents`, and use:

```yaml
consultation:
  mode: disabled
```

## Distributed

```yaml
execution:
  mode: distributed
```

An explicit shared `experiment.run_id` is mandatory.


# 20. Package and CLI

Use a `src` layout:

```text
src/marla/
├── __init__.py
├── __main__.py
├── cli.py
├── config/
├── runtime/
├── agents/
├── behaviours/
├── messaging/
├── environment/
├── learning/
├── knowledge/
├── models/
├── metrics/
└── utils/
```

Recommended detailed modules:

```text
config/loader.py
config/models.py
config/validation.py
runtime/local.py
runtime/distributed.py
runtime/lifecycle.py
runtime/device.py
agents/orchestrator.py
agents/gatekeeper.py
agents/plan_maker.py
messaging/schemas.py
messaging/builders.py
messaging/parsers.py
environment/nasimemu_adapter.py
environment/graph.py
environment/actions.py
environment/finish.py
learning/graph_encoder.py
learning/action_encoder.py
learning/recurrent_policy.py
learning/query_gate.py
learning/advice.py
learning/critic.py
learning/rollout.py
learning/gae.py
learning/ppo.py
knowledge/retriever.py
knowledge/nasimemu_rules.yaml
models/plan_maker_backend.py
models/local_backend.py
models/remote_backend.py
metrics/writer.py
metrics/summary.py
```

`pyproject.toml`:

```toml
[project]
name = "marla"
version = "0.1.0"
requires-python = ">=3.10,<3.11"
dependencies = [
  "typer",
  "pydantic",
  "PyYAML",
  "spade",
  "numpy",
  "pandas",
  "torch",
  "torch-geometric"
]

[project.scripts]
marla = "marla.cli:app"
```

Required CLI:

```text
marla run CONFIG
marla validate CONFIG
marla summarize RUN_OR_DIRECTORY
marla version
```

`marla run` accepts repeatable `--agent` only in distributed mode.

Also support:

```bash
python -m marla --help
```


# 21. Simple Metrics

Run directory:

```text
runs/<experiment-name>/<run-id>/
├── config.yaml
├── metadata.json
├── episodes.csv
├── decisions.csv
├── updates.csv
├── summary.json
├── marla.log
├── checkpoints/
└── artifacts/plan_maker/
```

No database, Redis, MLflow, or Parquet service is required.

## metadata.json

Store:

- run ID;
- experiment name and variant;
- phase and seed;
- MARLA, Python, NASimEmu, Torch, Torch Geometric, and SPADE versions;
- OS;
- execution mode;
- requested/resolved devices;
- scenario and objective;
- aliases and JIDs;
- model, prompt, and knowledge versions;
- configuration hash;
- Git commit when available;
- start/end timestamps and final status.

Names, aliases, and JIDs are stored directly.

## episodes.csv

```text
run_id
variant
episode_id
seed
scenario
goal_success
nasimemu_return
training_return
benchmark_return
environment_steps
rl_decisions
steps_to_goal
episode_seconds
consultation_count
consultation_cost
schema_rejection_count
finish_reason
```

## decisions.csv

```text
run_id
episode_id
environment_step
observation_id
legal_action_count
query_probability
queried
consultation_cost
request_id
response_status
response_latency_ms
schema_revision_count
base_policy_entropy
base_top_two_margin
base_top_action_id
plan_maker_top_action_id
final_top_action_id
selected_action_id
selected_action_base_rank
selected_action_plan_maker_rank
beta
alpha
advice_changed_top_action
action_success
nasimemu_reward
training_reward
terminated
truncated
artifact_path
```

## updates.csv

```text
run_id
update
environment_steps
policy_loss
value_loss
query_entropy
action_entropy
approximate_kl
clip_fraction
explained_variance
gradient_norm
learning_rate
mean_beta
mean_query_probability
actual_query_rate
elapsed_training_seconds
checkpoint_id
```

## summary.json

At minimum:

- episode count;
- goal-success rate;
- mean and median benchmark return;
- mean environment steps;
- mean steps to goal;
- mean episode duration;
- total and mean consultations;
- mean consultation cost;
- mean Plan Maker latency;
- mean beta;
- advice-changed-top-action rate;
- schema-rejection rate;
- total training environment steps;
- total training time.

The RL Orchestrator is the only central metrics writer. Full Plan Maker request/response artifacts are stored as `artifacts/plan_maker/<request-id>.json`.


# 22. Reproducibility and Comparison

Set and store seeds for:

- Python;
- NumPy;
- Torch CPU;
- Torch CUDA;
- NASimEmu;
- action sampling;
- query sampling;
- minibatch shuffling.

Baseline and assisted variants must match:

- graph encoder;
- GRU;
- candidate-action scorer;
- critic;
- PPO hyperparameters;
- scenario set;
- seed set;
- training environment-step budget;
- evaluation episode limit.

The assisted variant adds only:

- query gate;
- advice processor;
- trust head;
- advice scale;
- Gatekeeper;
- Plan Maker.

Report the assisted model's additional parameter count.

Primary comparison metrics:

1. goal-success rate;
2. fixed benchmark return;
3. environment steps to goal;
4. episode wall time;
5. training environment steps and wall time;
6. query rate;
7. consultation cost;
8. mean trust coefficient.

Benchmark return should exclude consultation cost so task performance remains comparable; training return includes consultation cost.


# 23. Testing

## Unit tests

Cover:

- config validation;
- Python version check;
- device resolution;
- stable action IDs;
- `FINISH`;
- NASimEmu graph conversion;
- Torch Geometric shapes;
- target-node lookup;
- candidate masks;
- GRU reset;
- query distribution;
- confidence clipping and normalization;
- constant-vector handling;
- trust bounds and positive alpha;
- PPO joint log-probability;
- GAE;
- recurrent masks;
- deterministic knowledge retrieval;
- message schema validation;
- CSV records.

## Integration tests

Baseline smoke test:

- CPU;
- small scenario;
- short rollout;
- no Gatekeeper or Plan Maker.

Assisted local smoke test:

- one SPADE main;
- mock deterministic Plan Maker;
- accepted and rejected response paths.

Distributed smoke test:

- RL Orchestrator and Gatekeeper in one process;
- Plan Maker in another;
- common XMPP server;
- shared run ID;
- clean startup and shutdown.

## Critical PPO regression

Verify:

- consultation cost changes training return;
- exact stored advice is reused;
- Plan Maker is never called during optimization;
- sequence masks prevent cross-episode state leakage;
- PPO update changes trainable parameters.


# 24. Implementation Milestones

1. Package, Python 3.10 enforcement, Typer CLI, config validation.
2. NASimEmu adapter, stable actions, `FINISH`, graph conversion.
3. Torch Geometric GraphSAGE, action encoder, GRU, critic, base policy.
4. Recurrent rollout, GAE, PPO, checkpoints, baseline metrics.
5. Local SPADE runtime and lifecycle.
6. Gatekeeper, schemas, correction loop, mock Plan Maker.
7. Query gate, advice processing, beta/alpha, joint PPO probability.
8. Static NASimEmu knowledge and deterministic RAG.
9. Real local or remote Plan Maker backend.
10. Distributed `--agent` runtime and multi-machine test.
11. Matched baseline/assisted research runs and summary reporting.


# 25. First-Release Non-Goals

Do not block version 0.1 on:

- DQN;
- Decision Transformer;
- Action Evolver;
- Reward Optimizer;
- Blackboard;
- Coordinator;
- full emulation;
- multiple support agents;
- persistent Plan Maker memory;
- automatic remote deployment;
- Kubernetes;
- vector databases;
- Redis;
- MLflow;
- Parquet;
- dashboards;
- automatic sweep orchestration;
- environment vectorization.


# 26. Acceptance Criteria

MARLA 0.1 is complete when:

- it installs under Python 3.10;
- `marla --help` and `python -m marla --help` work;
- baseline starts without Gatekeeper or Plan Maker;
- baseline trains recurrent PPO in NASimEmu simulation;
- graph encoding uses Torch Geometric;
- checkpoints save and load;
- assisted local mode starts all agents in one SPADE main;
- learned query decisions are part of one compound environment step;
- all advice passes through Gatekeeper;
- exact action IDs are validated;
- accepted scores influence logits through learned beta and alpha;
- rejected responses fall back to base policy;
- consultation cost is charged;
- exact advice is stored and reused during PPO replay;
- distributed mode starts selected aliases/JIDs on separate machines;
- shared run ID and readiness handshake work;
- disconnects fail cleanly;
- CSV/JSON metrics support baseline-versus-assisted comparison.


# 27. Core Invariant

At every environment step, the recurrent PPO RL Orchestrator forms a base policy over the current legal NASimEmu actions plus `FINISH`. In the assisted variant, it samples a binary query decision before seeing advice. If it queries, the Plan Maker returns a Gatekeeper-validated confidence for every legal action ID using only the current visible observation and static NASimEmu knowledge. The RL Orchestrator computes a bounded learned trust coefficient and applies the advice as a residual adjustment to its base logits. It then selects exactly one action, advances NASimEmu exactly once, subtracts consultation cost when applicable, and stores the complete compound decision for recurrent PPO training.

The Plan Maker is frozen and advisory.  
The Gatekeeper validates advisory communication, not environment actions.  
The RL Orchestrator is the only environment-facing agent.  
The baseline excludes Gatekeeper and Plan Maker.  
The same package supports local and distributed execution.
