"""Pydantic models for the MARLA experiment configuration schema.

These models are the authoritative, validated representation of an
experiment YAML file (see spec section 19). Every field name mirrors the
YAML key exactly. Unknown keys are rejected (``extra="forbid"``) so typos
fail fast instead of being silently ignored.
"""

from __future__ import annotations

from typing import Annotated, Literal, Union

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
    # Effective reward for selecting FINISH before the objective is
    # satisfied is a base term plus a per-remaining-target term:
    #
    #   premature_finish_penalty
    #   + premature_finish_penalty_per_remaining_target * remaining_sensitive_targets
    #
    # "remaining" means a sensitive/value host not yet at ROOT access --
    # USER access does not count (see NasimEmuAdapter.sensitive_target_status,
    # matching NASimEmu's own all_sensitive_hosts_compromised()). The
    # per-target term defaults to 0.0, so an older config using only the
    # fixed `premature_finish_penalty` keeps its exact old semantics
    # unchanged. To get a *pure* proportional penalty with no fixed
    # component (e.g. -20 per remaining target and nothing else), set
    # `premature_finish_penalty: 0.0` alongside a non-zero per-target value
    # -- there is no separate "mode" switch; the two terms simply add.
    premature_finish_penalty: float = -1.0
    premature_finish_penalty_per_remaining_target: float = 0.0


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
    # Architecture ablation switches (spec: v2 / v3-target / v3-full), never
    # divergent codebases -- see marla.environment.visible_facts and
    # RecurrentPolicy's own docstring. Defaults preserve the FINISH-
    # diagnostics phase's existing v3-target behavior (target progress on,
    # exploration progress off) for every config that doesn't mention
    # these explicitly.
    visible_target_progress: bool = True
    visible_subnet_exploration: bool = False


class ConstantSchedulerConfig(MarlaBaseModel):
    """No learning-rate scheduler: the optimizer's ``learning_rate`` is used
    unchanged for the entire run. Implemented as a real (no-op)
    ``torch.optim.lr_scheduler.LambdaLR`` under the hood -- see
    ``learning/lr_scheduler.py`` -- so every run has a uniform
    schedule/checkpoint code path, "constant" is simply the schedule whose
    multiplier is always 1.0, not a special ``None`` case threaded through
    training and checkpointing.
    """

    type: Literal["constant"] = "constant"


class LinearSchedulerConfig(MarlaBaseModel):
    """Linear decay from the optimizer's ``learning_rate`` down to
    ``end_factor * learning_rate``, reaching exactly ``end_factor`` at the
    final PPO update of the run (``torch.optim.lr_scheduler.LinearLR``,
    ``start_factor=1.0``; horizon derived from the actual number of PPO
    updates the run will perform -- see ``learning/lr_scheduler.py``, spec
    section 8). No default: the old implicit "decays to exactly 0" behavior
    is no longer assumed -- every config using this schedule must say what
    fraction of the initial rate it decays to.
    """

    type: Literal["linear"] = "linear"
    end_factor: float = Field(ge=0, le=1)


class CosineSchedulerConfig(MarlaBaseModel):
    """Cosine annealing from the optimizer's ``learning_rate`` down to
    ``eta_min``, reaching ``eta_min`` exactly at the final PPO update
    (``torch.optim.lr_scheduler.CosineAnnealingLR``; horizon derived the
    same way as ``linear``, above).
    """

    type: Literal["cosine"] = "cosine"
    eta_min: float = Field(ge=0, default=0.0)


class StepSchedulerConfig(MarlaBaseModel):
    """Multiplicative decay by ``gamma`` every ``step_size`` PPO updates
    (``torch.optim.lr_scheduler.StepLR``). ``step_size`` is in PPO updates,
    not environment steps or optimizer minibatch steps -- see
    "Scheduler stepping semantics" in ``learning/lr_scheduler.py``.
    """

    type: Literal["step"] = "step"
    step_size: int = Field(gt=0)
    gamma: float = Field(gt=0, le=1)


class ExponentialSchedulerConfig(MarlaBaseModel):
    """Multiplicative decay by ``gamma`` every PPO update
    (``torch.optim.lr_scheduler.ExponentialLR``).
    """

    type: Literal["exponential"] = "exponential"
    gamma: float = Field(gt=0, le=1)


SchedulerConfig = Annotated[
    Union[
        ConstantSchedulerConfig,
        LinearSchedulerConfig,
        CosineSchedulerConfig,
        StepSchedulerConfig,
        ExponentialSchedulerConfig,
    ],
    Field(discriminator="type"),
]


class OptimizerConfig(MarlaBaseModel):
    """Only Adam is supported in this release -- an unrecognized ``type``
    is rejected outright (Pydantic's ``Literal`` does this for free) rather
    than silently falling back to some other optimizer.

    ``learning_rate`` and ``scheduler`` live here (not on ``PPOConfig``)
    because they are optimizer-level concerns, not PPO-algorithm concerns
    -- and because a scheduler wraps a specific optimizer instance, keeping
    them adjacent in both the YAML and this model avoids splitting one
    conceptual decision across two unrelated config sections. Both are
    REQUIRED, with no default for ``scheduler``: spec section 6 requires
    every experiment config to state its schedule explicitly rather than
    silently inheriting a hidden default (``marla init`` templates set an
    explicit value; there is no config-wide fallback).
    """

    type: Literal["adam"] = "adam"
    learning_rate: float = Field(gt=0)
    # Matches the Adam paper's numerical-stability term added inside the
    # denominator, not a learning-rate-like quantity -- torch's default
    # (1e-8) is tuned for supervised learning with typically well-scaled
    # gradients; PPO's advantage-scaled policy gradient is noisier, and a
    # slightly larger eps (1e-5, the same value used by OpenAI Baselines /
    # CleanRL's PPO implementations) improves numerical stability there.
    eps: float = Field(default=1.0e-5, gt=0)
    scheduler: SchedulerConfig


class PPOConfig(MarlaBaseModel):
    # Not shown in the spec's example YAML but required to know when a
    # training run should stop; kept explicit rather than defaulted since
    # it materially affects reproducibility/comparability between runs.
    total_environment_steps: int = Field(gt=0)
    # Environment transitions collected per PPO update, PER independent
    # environment stream (see `num_envs` below). For `num_envs=1`, "per
    # stream" and "total per update" are the same number -- the canonical
    # research baseline uses num_envs=4, so most current configs collect
    # `num_envs * steps_per_env` transitions per update; see
    # `effective_batch_size` below.
    steps_per_env: int = Field(gt=0)
    epochs: int = Field(gt=0)
    minibatch_sequences: int = Field(gt=0)
    gamma: float = Field(gt=0, le=1)
    gae_lambda: float = Field(ge=0, le=1)
    clip_epsilon: float = Field(gt=0)
    value_coefficient: float = Field(ge=0)
    query_entropy_coefficient: float = Field(ge=0)
    action_entropy_coefficient: float = Field(ge=0)
    max_grad_norm: float = Field(gt=0)
    optimizer: OptimizerConfig
    # Experimental (spec: the State-N critic-calibration collapse
    # investigation) -- number of EXTRA value-head-only optimizer passes
    # run immediately after the normal ``epochs``-pass actor+critic PPO
    # update completes, using the exact same rollout/GAE return targets
    # (never Monte Carlo/oracle data). Every non-critic parameter is
    # frozen during these passes (see learning.ppo.critic_refinement_update
    # for the exact isolation mechanism) -- the actor and shared backbone
    # never change during this phase. Default 0 is an exact behavioral
    # no-op: every existing config/checkpoint is unaffected. Never
    # overloads `epochs`, which remains the actor+critic joint-update
    # count unchanged.
    critic_refinement_epochs: int = Field(ge=0, default=0)
    # Number of INDEPENDENT environment streams collected, under the SAME
    # frozen policy parameters, before one PPO update runs. Effective
    # transitions per PPO update = `num_envs * steps_per_env` (see
    # `effective_batch_size` below). The provisional canonical research
    # baseline is num_envs=4 (see research/EXPERIMENT_PLAN.md); num_envs=1
    # collapses to a single-stream collector with no other behavioral
    # change. Each stream gets its own NasimEmuAdapter, its own
    # recurrent-hidden-state chain, and its own deterministically-derived
    # environment-reset seed range (see
    # `marla.learning.rollout.env_base_seed`) -- never a slice of one
    # continuous simulator trajectory. GAE is computed independently per
    # stream (never across a stream boundary) before the resulting
    # advantages/returns are combined for one shared PPO update -- see
    # `marla.learning.trainer.run_training_loop`'s multi-env branch.
    num_envs: int = Field(ge=1, default=1)


def effective_batch_size(ppo_config: "PPOConfig") -> int:
    """Total environment transitions collected (across every independent
    environment stream) before one PPO update runs -- ``num_envs *
    steps_per_env``. For ``num_envs=1`` this is exactly ``steps_per_env``.
    """
    return ppo_config.num_envs * ppo_config.steps_per_env


def compute_num_rollouts(ppo_config: "PPOConfig") -> int:
    """``ceil(total_environment_steps / effective_batch_size)`` -- the
    single source of truth for how many collect/PPO-update cycles a
    training run needs, used by ``cli.py``, ``learning/trainer.py``'s
    scheduler-horizon derivation, and every research driver script.
    ``total_environment_steps`` is a GLOBAL budget across every
    environment stream, never per-stream -- for ``num_envs=4`` this must
    NOT be computed as if only one stream's steps counted, which would
    silently run 4x the intended total.
    """
    import math

    return math.ceil(ppo_config.total_environment_steps / effective_batch_size(ppo_config))


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


class ResourceMonitoringConfig(MarlaBaseModel):
    """CPU/RAM/GPU telemetry (:mod:`marla.monitoring.resources`), sampled
    on a background thread throughout training/evaluation. Enabled by
    default -- the sampling overhead is small (measured; see
    ``research/RESEARCH_READINESS.md``) and computational cost is now a
    first-class research metric, not an opt-in extra.
    """

    enabled: bool = True
    sampling_interval_seconds: float = Field(default=1.0, gt=0)


class MetricsConfig(MarlaBaseModel):
    output_directory: str = "runs"
    record_decisions: bool = True
    resource_monitoring: ResourceMonitoringConfig = Field(default_factory=ResourceMonitoringConfig)
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


class CarbonConfig(MarlaBaseModel):
    """Per-agent energy/CO2-equivalent emissions tracking
    (:mod:`marla.monitoring.carbon`, CodeCarbon-backed). ``enabled``
    defaults to ``False`` (unlike ``metrics.resource_monitoring``, which
    defaults on): CodeCarbon is an OPTIONAL dependency (the ``carbon``
    extra), not a hard one like ``psutil`` -- defaulting this on would
    make every existing config/test suddenly require it installed. The
    research HPO study configs (spec: Optuna PPO tuning) turn this on
    explicitly.
    """

    enabled: bool = False
    # "process" isolates CPU/RAM estimates to this process (matching "one
    # agent = one MARLA run"); GPU power is still measured at device
    # level regardless of tracking_mode (CodeCarbon/NVML limitation, not
    # a MARLA one) -- see monitoring/carbon.py's module docstring. This is
    # exactly why HPO agents must run strictly sequentially on one device,
    # never concurrently: see learning/optuna_study.py's n_jobs=1.
    tracking_mode: Literal["process", "machine"] = "process"
    measure_power_secs: float = Field(default=1.0, gt=0)
    # Explicit carbon-location overrides (spec section 8): when ANY of
    # these is set, an OfflineEmissionsTracker is built with them instead
    # of CodeCarbon's own online geo-IP-based auto-resolution. None of
    # these being set does NOT mean "no location" -- it means "let
    # CodeCarbon auto-resolve it", which is the default, normal path.
    country_iso_code: str | None = None
    region: str | None = None
    cloud_provider: str | None = None
    cloud_region: str | None = None


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
    carbon: CarbonConfig = Field(default_factory=CarbonConfig)

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
    def _check_multi_env_requires_local_execution(self) -> "Config":
        # runtime/distributed.py's process-per-agent design constructs a
        # single NasimEmuAdapter and has no notion of multiple independent
        # environment streams (spec: the 2048-transition, 4-independent-
        # environment-stream ablation is a LOCAL-mode-only experiment).
        # Rejected explicitly here rather than silently under-collecting
        # (num_rollouts would already assume num_envs streams per update,
        # but only one adapter would ever be built) if this combination is
        # ever attempted.
        if self.policy.ppo.num_envs > 1 and self.execution.mode != "local":
            raise ValueError(
                f"policy.ppo.num_envs={self.policy.ppo.num_envs} > 1 requires execution.mode == 'local' -- "
                "the multi-environment collector is not implemented for distributed mode."
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
