"""Pydantic models for the MARLA experiment configuration schema.

These models are the authoritative, validated representation of an
experiment YAML file (see ``MARLA_complete_implementation_specification.md``,
section 19). Every field name mirrors the YAML key exactly. Unknown keys are
rejected (``extra="forbid"``) so typos fail fast instead of being silently
ignored.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

SUPPORTED_CONFIG_SCHEMA_VERSIONS = ("1.0",)


class MarlaBaseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ExperimentConfig(MarlaBaseModel):
    name: str
    run_id: str | None = None
    phase: Literal["training", "evaluation"] = "training"
    seed: int


class ExecutionConfig(MarlaBaseModel):
    mode: Literal["local", "distributed"]


class XmppConfig(MarlaBaseModel):
    server: str


class AgentIdentityConfig(MarlaBaseModel):
    """Identity of a SPADE agent: alias, JID, and where to find its password."""

    alias: str
    jid: str
    # Name of an environment variable holding the XMPP password. Never the
    # password itself: secrets are referenced, not embedded, so saved
    # configs can be stored/shared without redaction gymnastics.
    password_env: str | None = None


class EnvironmentConfig(MarlaBaseModel):
    mode: Literal["simulation"] = "simulation"
    scenario: str
    max_episode_steps: int = Field(gt=0)


class ObjectiveConfig(MarlaBaseModel):
    type: Literal["capture_target"]
    description: str
    # objective_satisfied() == NASimEmu's native goal: all sensitive/value
    # hosts compromised (see environment/nasimemu_adapter.py). Not shown in
    # the spec's example YAML; defaulted here so those examples still
    # validate as-is while remaining overridable per experiment.
    completion_reward: float = 1.0
    premature_finish_penalty: float = -1.0


class GraphEncoderConfig(MarlaBaseModel):
    type: Literal["graphsage"] = "graphsage"
    hidden_size: int = Field(gt=0)
    layers: int = Field(gt=0)


class ActionEncoderConfig(MarlaBaseModel):
    hidden_size: int = Field(gt=0)
    action_type_embedding_size: int = Field(gt=0)


class RecurrentConfig(MarlaBaseModel):
    hidden_size: int = Field(gt=0)
    sequence_length: int = Field(gt=0)


class PPOConfig(MarlaBaseModel):
    # Not shown in the spec's example YAML but required to know when a
    # training run should stop; kept explicit rather than defaulted since
    # it materially affects reproducibility/comparability between runs.
    total_environment_steps: int = Field(gt=0)
    rollout_steps: int = Field(gt=0)
    epochs: int = Field(gt=0)
    minibatch_sequences: int = Field(gt=0)
    gamma: float = Field(gt=0, le=1)
    gae_lambda: float = Field(ge=0, le=1)
    clip_epsilon: float = Field(gt=0)
    value_coefficient: float = Field(ge=0)
    query_entropy_coefficient: float = Field(ge=0)
    action_entropy_coefficient: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    learning_rate: float = Field(gt=0)


class PolicyConfig(MarlaBaseModel):
    algorithm: Literal["recurrent_ppo"] = "recurrent_ppo"
    graph_encoder: GraphEncoderConfig
    action_encoder: ActionEncoderConfig
    recurrent: RecurrentConfig
    ppo: PPOConfig


class ConsultationConfig(MarlaBaseModel):
    mode: Literal["disabled", "learned"]
    cost: float = 0.0
    max_schema_revisions: int | None = None

    @model_validator(mode="after")
    def _learned_requires_fields(self) -> "ConsultationConfig":
        if self.mode == "learned":
            if self.max_schema_revisions is None:
                raise ValueError(
                    "consultation.max_schema_revisions is required when "
                    "consultation.mode == 'learned'"
                )
            if self.max_schema_revisions < 0:
                raise ValueError("consultation.max_schema_revisions must be >= 0")
            if self.cost < 0:
                raise ValueError("consultation.cost must be >= 0")
        return self


class PlanMakerModelConfig(MarlaBaseModel):
    backend: Literal["local", "remote"]
    name: str
    # Not in the spec's example configs; added because the local backend's
    # generation length is otherwise a fixed, uncontrollable 512 tokens per
    # call -- fine for a large production model, but a real cost driver for
    # a small model doing a compact JSON advisory response (up to 4x the
    # correction-loop retries on top). A *floor*, not a fixed value: the
    # local backend also estimates a per-request minimum from the actual
    # number and length of legal action IDs (a response is one JSON entry
    # per action, so a scenario with more actions -- or the same episode
    # later on, once more hosts are discovered -- needs more tokens than
    # this alone might provide) and uses whichever is larger. Default of
    # 512 preserves prior behavior for existing configs.
    max_new_tokens: int = 512


class PlanMakerKnowledgeConfig(MarlaBaseModel):
    path: str
    version: str


class PlanMakerAgentConfig(MarlaBaseModel):
    alias: str
    jid: str
    password_env: str | None = None
    role: Literal["plan_maker"]
    model: PlanMakerModelConfig
    prompt_version: str
    knowledge: PlanMakerKnowledgeConfig


class MetricsConfig(MarlaBaseModel):
    output_directory: str = "runs"
    record_decisions: bool = True
    # Periodic deterministic (greedy) evaluation episodes, run with the
    # current policy weights between rollouts (an EvalCallback-style pass,
    # not a genuine held-out generalization test -- MARLA's config has a
    # single environment.scenario, not a train/test scenario split; see
    # learning/rollout.py's run_evaluation_episodes). 0 disables it, which
    # is the default: this adds real wall-clock cost (extra episodes, and
    # in assisted mode extra Plan Maker consultations), so existing configs
    # opt in rather than getting it for free.
    eval_episodes: int = Field(default=0, ge=0)
    eval_every_rollouts: int = Field(default=1, gt=0)


class ReproducibilityConfig(MarlaBaseModel):
    deterministic_torch: bool = True


class Config(MarlaBaseModel):
    schema_version: str
    experiment: ExperimentConfig
    execution: ExecutionConfig
    device: Literal["cpu", "gpu", "auto"] = "auto"
    xmpp: XmppConfig
    environment: EnvironmentConfig
    objective: ObjectiveConfig
    policy: PolicyConfig
    consultation: ConsultationConfig
    rl_orchestrator: AgentIdentityConfig
    gatekeeper: AgentIdentityConfig | None = None
    agents: list[PlanMakerAgentConfig] = Field(default_factory=list)
    metrics: MetricsConfig = Field(default_factory=MetricsConfig)
    reproducibility: ReproducibilityConfig = Field(default_factory=ReproducibilityConfig)

    @model_validator(mode="after")
    def _check_schema_version(self) -> "Config":
        if self.schema_version not in SUPPORTED_CONFIG_SCHEMA_VERSIONS:
            raise ValueError(
                f"Unsupported schema_version '{self.schema_version}'. "
                f"Supported: {SUPPORTED_CONFIG_SCHEMA_VERSIONS}"
            )
        return self

    @model_validator(mode="after")
    def _check_distributed_run_id(self) -> "Config":
        if self.execution.mode == "distributed" and not self.experiment.run_id:
            raise ValueError(
                "experiment.run_id is mandatory when execution.mode == 'distributed'"
            )
        return self

    @model_validator(mode="after")
    def _check_baseline_excludes_support_agents(self) -> "Config":
        if self.consultation.mode == "disabled":
            if self.gatekeeper is not None:
                raise ValueError(
                    "gatekeeper must not be configured when consultation.mode == "
                    "'disabled' (baseline must not start the Gatekeeper)"
                )
            if self.agents:
                raise ValueError(
                    "agents must not be configured when consultation.mode == "
                    "'disabled' (baseline must not start the Plan Maker)"
                )
        else:  # learned
            if self.gatekeeper is None:
                raise ValueError(
                    "gatekeeper is required when consultation.mode == 'learned'"
                )
            plan_makers = [a for a in self.agents if a.role == "plan_maker"]
            if not plan_makers:
                raise ValueError(
                    "at least one agents[] entry with role 'plan_maker' is "
                    "required when consultation.mode == 'learned'"
                )
        return self

    @model_validator(mode="after")
    def _check_unique_aliases_and_jids(self) -> "Config":
        aliases = [self.rl_orchestrator.alias]
        jids = [self.rl_orchestrator.jid]
        if self.gatekeeper is not None:
            aliases.append(self.gatekeeper.alias)
            jids.append(self.gatekeeper.jid)
        for agent in self.agents:
            aliases.append(agent.alias)
            jids.append(agent.jid)

        if len(aliases) != len(set(aliases)):
            raise ValueError(f"agent aliases must be unique, got: {aliases}")
        if len(jids) != len(set(jids)):
            raise ValueError(f"agent JIDs must be unique, got: {jids}")
        return self
