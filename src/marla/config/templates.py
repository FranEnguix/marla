"""Starter configuration templates for ``marla init`` (see ``marla.cli``).

These are deliberately scaled down from ``examples/`` (which mirror the
spec's production-realistic settings): the goal here is a fast first run
that proves the platform works end-to-end, not a real research result.
"""

from __future__ import annotations

from pathlib import Path

_SCENARIO_RELATIVE_PATH = Path("NASimEmu") / "scenarios" / "sm_entry_dmz_one_subnet.v2.yaml"

BASELINE_TEMPLATE = """\
schema_version: "1.0"

experiment:
  name: baseline_quickstart
  run_id: baseline-quickstart-1
  phase: training
  seed: 42

execution:
  mode: local

device: auto

xmpp:
  server: localhost

environment:
  mode: simulation
  scenario: {scenario}
  max_episode_steps: 50

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
    sequence_length: 16
  ppo:
    # Small on purpose: enough rollouts to see the training loop actually
    # learn something, fast enough to finish in well under a minute on CPU.
    # Scale these up (see examples/baseline.yaml) for a real research run.
    total_environment_steps: 1280
    rollout_steps: 128
    epochs: 2
    minibatch_sequences: 4
    gamma: 0.99
    gae_lambda: 0.95
    clip_epsilon: 0.2
    value_coefficient: 0.5
    query_entropy_coefficient: 0.01
    action_entropy_coefficient: 0.01
    max_grad_norm: 0.5
    learning_rate: 0.0003

consultation:
  mode: disabled

rl_orchestrator:
  alias: rl_orchestrator
  jid: rl-orchestrator@localhost
  password_env: MARLA_RL_ORCHESTRATOR_PASSWORD

metrics:
  output_directory: runs
  record_decisions: true
  # Periodic deterministic (greedy) evaluation episodes, so `marla summarize`
  # has a real train-vs-eval reward/episode-length curve out of the box.
  # Cheap here since there's no Plan Maker to consult -- the assisted
  # template below leaves this at 0 (disabled) since each eval episode
  # would also consult the Plan Maker, and see MetricsConfig's docstring.
  eval_episodes: 2
  eval_every_rollouts: 2

reproducibility:
  deterministic_torch: true
"""

ASSISTED_TEMPLATE = """\
schema_version: "1.0"

experiment:
  name: assisted_quickstart
  run_id: assisted-quickstart-1
  phase: training
  seed: 42

execution:
  mode: local

# Forced to cpu, not auto: a small model doing this few tokens of generation
# is plenty fast on CPU, and GPU health/availability varies a lot across
# machines -- a first-run "does the platform work" demo shouldn't gamble on
# it. Switch to auto/gpu once you've confirmed the pipeline runs and want
# the speed for a real, longer training run.
device: cpu

xmpp:
  server: localhost

environment:
  mode: simulation
  scenario: {scenario}
  max_episode_steps: 50

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
    sequence_length: 16
  ppo:
    # Smaller than baseline.yaml's, on top of being small for the same
    # reason: a real consultation is a real, slow-on-CPU model call, so the
    # total step count directly bounds how many of those a quickstart run
    # can rack up.
    total_environment_steps: 384
    rollout_steps: 128
    epochs: 2
    minibatch_sequences: 4
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
  jid: rl-orchestrator@localhost
  password_env: MARLA_RL_ORCHESTRATOR_PASSWORD

gatekeeper:
  alias: gatekeeper
  jid: gatekeeper@localhost
  password_env: MARLA_GATEKEEPER_PASSWORD

agents:
  - alias: plan_maker_1
    jid: plan-maker@localhost
    password_env: MARLA_PLAN_MAKER_1_PASSWORD
    role: plan_maker
    model:
      backend: local
      # Same production-scale model as examples/assisted.yaml. A smaller
      # 0.5B model was tried first for a faster download/first run, but in
      # practice it doesn't reliably score every one of the required action
      # IDs (observed: 10/11 covered, then 1/11 after a correction retry) --
      # it reasons sensibly about which actions look promising, it just
      # doesn't reliably enumerate the full required set. This model follows
      # "score every one of these N items" far more reliably. ~3GB one-time
      # download.
      name: Qwen/Qwen2.5-1.5B-Instruct
      # The expected advisory response is a small JSON object, nowhere near
      # the default 512-token generation budget; each unnecessary token is
      # real wall-clock time, multiplied by however many consultations this
      # run makes and by the correction loop's retries on top of that. A
      # complete, correctly-shaped mapping for this scenario's 11 actions
      # needs ~200 tokens; 256 leaves real headroom without being wasteful.
      max_new_tokens: 256
    prompt_version: nasimemu-plan-maker-v1
    knowledge:
      path: package://marla/knowledge/nasimemu_rules.yaml
      version: nasimemu-rules-v1

metrics:
  output_directory: runs
  record_decisions: true
  # eval_episodes defaults to 0 (disabled): a periodic deterministic
  # evaluation pass would consult the Plan Maker again for each eval
  # episode's queries, doubling real model-inference cost for this
  # quickstart. See baseline_quickstart.yaml's metrics section, and
  # MetricsConfig's docstring, for what enabling it gets you.

reproducibility:
  deterministic_torch: true
"""

TEMPLATES: dict[str, str] = {
    "baseline.yaml": BASELINE_TEMPLATE,
    "assisted.yaml": ASSISTED_TEMPLATE,
}


def find_nasimemu_scenario(start: Path, max_levels: int = 5) -> Path | None:
    """Search ``start`` and its ancestors for the bundled NASimEmu scenario used
    by the templates.

    NASimEmu is a separate checked-out project alongside this repo, not part
    of the installed ``marla`` package, so there is no fixed path to it --
    this mirrors how a developer would locate it by eye, starting from the
    current working directory.
    """
    current = start.resolve()
    for _ in range(max_levels + 1):
        candidate = current / _SCENARIO_RELATIVE_PATH
        if candidate.is_file():
            return candidate
        if current.parent == current:
            break
        current = current.parent
    return None


def render_templates(scenario_path: str) -> dict[str, str]:
    """Render every template with ``scenario_path`` substituted in."""
    return {name: template.format(scenario=scenario_path) for name, template in TEMPLATES.items()}
