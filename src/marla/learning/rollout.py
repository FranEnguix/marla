"""Recurrent rollout collection (spec sections 16-17).

Runs the compound-decision loop against a live :class:`NasimEmuAdapter` and
records everything PPO needs to replay each transition without ever calling
the environment or the Plan Maker again. Baseline instances are constructed
with ``consultation_enabled=False`` and never touch the query-gate/advice
machinery at all; assisted instances receive a ``consult_fn`` callback that
performs the actual Gatekeeper round-trip (kept out of this module so it
stays decoupled from SPADE -- see ``marla.agents.advisory_client``).
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

import torch
from torch_geometric.data import Data

from marla.environment.actions import ActionDescriptor
from marla.environment.nasimemu_adapter import EnvironmentState, NasimEmuAdapter
from marla.environment.observation_summary import build_observation_summary
from marla.evaluation.overrides import EvaluationOverrides
from marla.learning.decision import (
    FinalDecision,
    compute_final_decision,
    compute_joint_log_probability,
    compute_query_probability,
)
from marla.learning.recurrent_policy import RecurrentPolicy, RecurrentState

logger = logging.getLogger(__name__)

# Purely a progress-visibility cadence (spec has no elapsed timeout to
# respect here) -- frequent enough that a slow run doesn't look hung
# between the sparser per-rollout/per-consultation log lines, rare enough
# not to spam a fast one.
STEP_LOG_INTERVAL = 10


@dataclass(frozen=True)
class ConsultationResult:
    """What a ``consult_fn`` callback returns; decoupled from SPADE/Gatekeeper types."""

    status: str  # "accepted" | "schema_rejected"
    scores: dict[str, float] | None
    request_id: str
    latency_ms: float | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None


ConsultFn = Callable[[list[ActionDescriptor], int, int, str, dict], Awaitable[ConsultationResult]]


@dataclass
class StepRecord:
    """One compound decision, with everything needed to replay it under PPO."""

    run_id: str
    episode_id: int
    environment_step: int
    observation_id: str

    graph_data: Data
    node_key_to_index: dict[str, int]
    legal_action_descriptors: list[ActionDescriptor]

    initial_gru_hidden_state: torch.Tensor  # z_{t-1}, detached
    previous_action_embedding: torch.Tensor  # E(a_{t-1}), detached
    previous_training_reward: float
    previous_query: bool

    base_logits: torch.Tensor  # detached, (N,)
    final_logits: torch.Tensor  # detached, (N,) -- == base_logits unless accepted advice was applied
    selected_action_index: int
    old_action_log_probability: float
    old_joint_log_probability: float
    critic_value: float

    nasimemu_reward: float
    consultation_cost: float
    training_reward: float
    terminated: bool
    truncated: bool

    # Only meaningful when terminated is False and (truncated or this is the
    # last record of a rollout collection window): the value estimate of the
    # state immediately following this step, needed to bootstrap GAE across
    # a rollout-window cutoff. See learning/gae.py.
    bootstrap_value: float | None = None

    # Assisted-variant fields; left at defaults in the baseline.
    sampled_query: bool = False
    old_query_probability: float | None = None
    old_query_log_probability: float | None = None
    plan_maker_scores_in_action_order: list[float] | None = None
    plan_maker_validation_status: str | None = None
    plan_maker_response_status: str | None = None
    plan_maker_request_id: str | None = None
    plan_maker_latency_ms: float | None = None
    plan_maker_artifact_path: str | None = None
    plan_maker_input_tokens: int | None = None
    plan_maker_output_tokens: int | None = None
    plan_maker_total_tokens: int | None = None
    normalized_advice: list[float] | None = None
    beta: float | None = None
    alpha: float | None = None


@dataclass
class EpisodeSummary:
    run_id: str
    episode_id: int
    seed: int
    goal_success: bool
    nasimemu_return: float
    training_return: float
    environment_steps: int
    steps_to_goal: int | None
    episode_seconds: float
    finish_reason: str
    consultation_count: int = 0
    consultation_cost: float = 0.0
    schema_rejection_count: int = 0
    # Which training rollout this episode belongs to (training episode) or
    # immediately followed (eval episode) -- lets metrics/plots bucket
    # returns by training progress instead of raw episode index. None for
    # episodes summarized outside run_training_loop (e.g. direct
    # RolloutCollector use in tests).
    rollout: int | None = None
    # True for a deterministic (greedy) evaluation episode from
    # run_evaluation_episodes, contributing nothing to the PPO buffer.
    # False for an ordinary stochastic training episode.
    is_eval: bool = False


class RolloutCollector:
    """Collects fixed-length rollout windows, resuming across calls.

    One :class:`NasimEmuAdapter`/:class:`RecurrentPolicy` pair per instance,
    matching MARLA's non-vectorized, single-environment design.
    """

    def __init__(
        self,
        adapter: NasimEmuAdapter,
        policy: RecurrentPolicy,
        run_id: str,
        base_seed: int,
        consultation_enabled: bool = False,
        consultation_cost: float = 0.0,
        consult_fn: ConsultFn | None = None,
        stop_event: asyncio.Event | None = None,
        deterministic: bool = False,
        overrides: EvaluationOverrides | None = None,
    ) -> None:
        if consultation_enabled and consult_fn is None:
            raise ValueError("consult_fn is required when consultation_enabled=True")

        self._adapter = adapter
        self._policy = policy
        self._run_id = run_id
        self._seed_counter = base_seed
        self._consultation_enabled = consultation_enabled
        self._consultation_cost = consultation_cost
        self._consult_fn = consult_fn
        self._stop_event = stop_event
        # Greedy (argmax) action/query selection instead of sampling, for a
        # deterministic evaluation pass (see run_evaluation_episodes) -- not
        # used by ordinary training rollout collection.
        self._deterministic = deterministic
        # Evaluation-only ablation hooks (research/aamas2027). None on every
        # real `marla run` training/eval path (trainer.py never passes this)
        # -- see marla.evaluation.overrides for what each field does.
        self._overrides = overrides
        self._fallback_rng = random.Random(overrides.fallback_rng_seed) if overrides is not None else None

        self._episode_id = 0
        self._state: EnvironmentState | None = None
        self._rstate: RecurrentState | None = None
        self._episode_seed = 0
        self._episode_start_time = 0.0
        self._episode_nasimemu_return = 0.0
        self._episode_training_return = 0.0
        self._episode_steps = 0
        self._steps_to_goal: int | None = None
        self._episode_consultation_count = 0
        self._episode_consultation_cost_total = 0.0
        self._episode_schema_rejection_count = 0

    def _start_new_episode(self) -> None:
        seed = self._seed_counter
        self._seed_counter += 1
        self._state = self._adapter.reset(seed=seed)
        self._rstate = self._policy.initial_recurrent_state()
        self._episode_id += 1
        self._episode_seed = seed
        self._episode_start_time = time.monotonic()
        self._episode_nasimemu_return = 0.0
        self._episode_training_return = 0.0
        self._episode_steps = 0
        self._steps_to_goal = None
        self._episode_consultation_count = 0
        self._episode_consultation_cost_total = 0.0
        self._episode_schema_rejection_count = 0

    async def collect(self, num_steps: int) -> tuple[list[StepRecord], list[EpisodeSummary]]:
        if self._state is None:
            self._start_new_episode()

        records: list[StepRecord] = []
        summaries: list[EpisodeSummary] = []

        for i in range(num_steps):
            if self._stop_event is not None and self._stop_event.is_set():
                logger.info("  stop requested; ending this rollout early at step %d/%d", i, num_steps)
                break

            if i % STEP_LOG_INTERVAL == 0:
                logger.info("  step %d/%d (episode %d)", i + 1, num_steps, self._episode_id)

            state = self._state
            rstate = self._rstate
            assert state is not None and rstate is not None

            legal_actions = self._adapter.legal_actions(state)
            graph_obs = self._adapter.to_pyg_data(state)
            step_out = self._policy.step(graph_obs, legal_actions, rstate)

            decision = await self._decide(legal_actions, step_out, state)
            action_index = decision["action_index"]
            selected = legal_actions[action_index]

            transition = self._adapter.step(selected)
            consultation_cost = decision["consultation_cost"]
            training_reward = transition.nasimemu_reward - consultation_cost

            # A stop request arriving during this step's (possibly slow,
            # blocking) consultation makes this record the last one of the
            # window too, exactly like reaching the end of num_steps -- GAE
            # requires a bootstrap value for whichever record ends up last
            # in a non-terminated window (see learning/gae.py), and the stop
            # is checked again at the top of the loop before the next step.
            stop_requested = self._stop_event is not None and self._stop_event.is_set()
            is_last_in_window = i == num_steps - 1 or stop_requested
            bootstrap_value: float | None = None
            if not transition.terminated and (transition.truncated or is_last_in_window):
                bootstrap_value = self._bootstrap_value(transition.state, step_out, action_index, training_reward)

            record = StepRecord(
                run_id=self._run_id,
                episode_id=self._episode_id,
                environment_step=self._episode_steps,
                observation_id=f"observation-{self._episode_id}-{self._episode_steps}",
                graph_data=graph_obs.data,
                node_key_to_index=graph_obs.node_key_to_index,
                legal_action_descriptors=legal_actions,
                initial_gru_hidden_state=rstate.z.detach().clone(),
                previous_action_embedding=rstate.previous_action_embedding.detach().clone(),
                previous_training_reward=rstate.previous_reward,
                previous_query=bool(rstate.previous_query),
                base_logits=step_out.base_logits.detach().clone(),
                final_logits=decision["final_logits"],
                selected_action_index=action_index,
                old_action_log_probability=decision["action_log_prob"],
                old_joint_log_probability=decision["joint_log_prob"],
                critic_value=float(step_out.value.detach()),
                nasimemu_reward=transition.nasimemu_reward,
                consultation_cost=consultation_cost,
                training_reward=training_reward,
                terminated=transition.terminated,
                truncated=transition.truncated,
                bootstrap_value=bootstrap_value,
                sampled_query=decision["sampled_query"],
                old_query_probability=decision["query_probability"],
                old_query_log_probability=decision["query_log_prob"],
                plan_maker_scores_in_action_order=decision["plan_maker_scores"],
                plan_maker_validation_status=decision["plan_maker_validation_status"],
                plan_maker_response_status=decision["plan_maker_response_status"],
                plan_maker_request_id=decision["plan_maker_request_id"],
                plan_maker_latency_ms=decision["plan_maker_latency_ms"],
                plan_maker_input_tokens=decision["plan_maker_input_tokens"],
                plan_maker_output_tokens=decision["plan_maker_output_tokens"],
                plan_maker_total_tokens=decision["plan_maker_total_tokens"],
                normalized_advice=decision["normalized_advice"],
                beta=decision["beta"],
                alpha=decision["alpha"],
            )
            records.append(record)

            self._episode_nasimemu_return += transition.nasimemu_reward
            self._episode_training_return += training_reward
            self._episode_steps += 1
            if decision["sampled_query"]:
                self._episode_consultation_count += 1
                self._episode_consultation_cost_total += consultation_cost
                if decision["plan_maker_response_status"] == "schema_rejected":
                    self._episode_schema_rejection_count += 1
            if selected.is_finish and transition.info.get("objective_satisfied"):
                self._steps_to_goal = self._episode_steps

            if transition.terminated or transition.truncated:
                summaries.append(
                    EpisodeSummary(
                        run_id=self._run_id,
                        episode_id=self._episode_id,
                        seed=self._episode_seed,
                        goal_success=bool(transition.info.get("objective_satisfied", False)),
                        nasimemu_return=self._episode_nasimemu_return,
                        training_return=self._episode_training_return,
                        environment_steps=self._episode_steps,
                        steps_to_goal=self._steps_to_goal,
                        episode_seconds=time.monotonic() - self._episode_start_time,
                        finish_reason="finish" if selected.is_finish else "truncated",
                        consultation_count=self._episode_consultation_count,
                        consultation_cost=self._episode_consultation_cost_total,
                        schema_rejection_count=self._episode_schema_rejection_count,
                        is_eval=self._deterministic,
                    )
                )
                self._state = None
                self._rstate = None
                if i < num_steps - 1:
                    self._start_new_episode()
            else:
                self._state = transition.state
                self._rstate = self._policy.advance_recurrent_state(
                    step_out, action_index, training_reward, query=decision["sampled_query"]
                )

            await asyncio.sleep(0)  # cooperative yield: let other local-mode agents run

        return records, summaries

    async def _decide(self, legal_actions: list[ActionDescriptor], step_out, state: EnvironmentState) -> dict:
        """Runs steps 5-7 of the core invariant: query gate, consultation, advice, selection."""
        if not self._consultation_enabled:
            probs = step_out.base_probs.detach()
            distribution = torch.distributions.Categorical(probs=probs)
            sampled = probs.argmax() if self._deterministic else distribution.sample()
            action_index = int(sampled.item())
            action_log_prob = float(distribution.log_prob(sampled))
            return {
                "action_index": action_index,
                "final_logits": step_out.base_logits.detach().clone(),
                "action_log_prob": action_log_prob,
                "joint_log_prob": action_log_prob,
                "consultation_cost": 0.0,
                "sampled_query": False,
                "query_probability": None,
                "query_log_prob": None,
                "plan_maker_scores": None,
                "plan_maker_validation_status": None,
                "plan_maker_response_status": None,
                "plan_maker_request_id": None,
                "plan_maker_latency_ms": None,
                "plan_maker_input_tokens": None,
                "plan_maker_output_tokens": None,
                "plan_maker_total_tokens": None,
                "normalized_advice": None,
                "beta": None,
                "alpha": None,
            }

        query_probability = compute_query_probability(
            self._policy, step_out, len(legal_actions), self._consultation_cost
        )
        # query_probability is always computed and recorded (for comparison)
        # regardless of query_mode -- only which of NO_QUERY/ALWAYS_QUERY/the
        # gate's own decision actually gets *sampled* is overridden.
        if self._overrides is not None and self._overrides.query_mode == "never":
            sampled_query = False
        elif self._overrides is not None and self._overrides.query_mode == "always":
            sampled_query = True
        elif self._deterministic:
            sampled_query = bool(query_probability.detach().item() >= 0.5)
        else:
            query_dist = torch.distributions.Bernoulli(probs=query_probability.detach())
            sampled_query = bool(query_dist.sample().item())

        consultation_cost = 0.0
        plan_maker_confidence = None
        plan_maker_scores: list[float] | None = None
        plan_maker_request_id: str | None = None
        plan_maker_response_status: str | None = None
        plan_maker_validation_status: str | None = None
        plan_maker_latency_ms: float | None = None
        plan_maker_input_tokens: int | None = None
        plan_maker_output_tokens: int | None = None
        plan_maker_total_tokens: int | None = None

        if sampled_query:
            assert self._consult_fn is not None
            consultation_cost = self._consultation_cost
            observation = build_observation_summary(state)
            logger.info(
                "  step %d (episode %d): querying Plan Maker, %d legal action(s)...",
                self._episode_steps, self._episode_id, len(legal_actions),
            )
            consult_start = time.monotonic()
            result = await self._consult_fn(
                legal_actions, self._episode_id, self._episode_steps,
                f"observation-{self._episode_id}-{self._episode_steps}",
                observation,
            )
            logger.info(
                "  step %d (episode %d): Plan Maker responded in %.1fs: %s",
                self._episode_steps, self._episode_id, time.monotonic() - consult_start, result.status,
            )
            plan_maker_request_id = result.request_id
            plan_maker_response_status = result.status
            plan_maker_latency_ms = result.latency_ms
            plan_maker_input_tokens = result.input_tokens
            plan_maker_output_tokens = result.output_tokens
            plan_maker_total_tokens = result.total_tokens
            if result.status == "accepted":
                assert result.scores is not None
                plan_maker_validation_status = "accepted"
                plan_maker_confidence = torch.tensor(
                    [result.scores[a.action_id] for a in legal_actions],
                    dtype=torch.float32,
                    device=step_out.z.device,
                )
                plan_maker_scores = plan_maker_confidence.tolist()
            else:
                plan_maker_validation_status = "rejected"

        final_decision = compute_final_decision(self._policy, step_out, sampled_query, plan_maker_confidence)

        # BETA_ZERO/BETA_ONE: only meaningful when advice was actually
        # obtained -- compute_final_decision already leaves normalized_advice
        # None for "not queried" and "queried but rejected", so this is a
        # deliberate no-op in both of those cases, not a gap. Never touches
        # decision.py/advice.py: it only recombines fields FinalDecision
        # already exposes.
        if (
            self._overrides is not None
            and self._overrides.beta_override is not None
            and final_decision.normalized_advice is not None
        ):
            beta_value = step_out.base_logits.new_tensor(self._overrides.beta_override)
            final_decision = FinalDecision(
                final_logits=step_out.base_logits + beta_value * final_decision.alpha * final_decision.normalized_advice,
                beta=beta_value,
                alpha=final_decision.alpha,
                normalized_advice=final_decision.normalized_advice,
            )

        if self._overrides is not None and self._overrides.action_selection == "plan_maker_argmax":
            # PLAN_MAKER_ONLY: ignore the policy's distribution entirely.
            # Fixed, seeded fallback on schema_rejected -- a documented
            # fallback rule, never a re-use of any learned policy output.
            if plan_maker_confidence is not None:
                action_index = int(torch.argmax(plan_maker_confidence).item())
            else:
                assert self._fallback_rng is not None
                action_index = self._fallback_rng.randrange(len(legal_actions))
        else:
            final_probs = torch.softmax(final_decision.final_logits.detach(), dim=-1)
            if self._deterministic:
                action_index = int(final_probs.argmax().item())
            else:
                action_dist = torch.distributions.Categorical(probs=final_probs)
                action_index = int(action_dist.sample().item())

        joint = compute_joint_log_probability(
            query_probability.detach(), sampled_query, final_decision.final_logits.detach(), action_index
        )

        return {
            "action_index": action_index,
            "final_logits": final_decision.final_logits.detach().clone(),
            "action_log_prob": float(joint.action_log_prob),
            "joint_log_prob": float(joint.joint_log_prob),
            "consultation_cost": consultation_cost,
            "sampled_query": sampled_query,
            "query_probability": float(query_probability.detach()),
            "query_log_prob": float(joint.query_log_prob),
            "plan_maker_scores": plan_maker_scores,
            "plan_maker_validation_status": plan_maker_validation_status,
            "plan_maker_response_status": plan_maker_response_status,
            "plan_maker_request_id": plan_maker_request_id,
            "plan_maker_latency_ms": plan_maker_latency_ms,
            "plan_maker_input_tokens": plan_maker_input_tokens,
            "plan_maker_output_tokens": plan_maker_output_tokens,
            "plan_maker_total_tokens": plan_maker_total_tokens,
            "normalized_advice": final_decision.normalized_advice.detach().tolist()
            if final_decision.normalized_advice is not None
            else None,
            "beta": float(final_decision.beta.detach()) if final_decision.beta is not None else None,
            "alpha": float(final_decision.alpha.detach()) if final_decision.alpha is not None else None,
        }

    def _bootstrap_value(self, next_state, step_out, action_index: int, training_reward: float) -> float:
        assert next_state is not None
        next_rstate = self._policy.advance_recurrent_state(step_out, action_index, training_reward, query=False)
        next_legal = self._adapter.legal_actions(next_state)
        next_graph_obs = self._adapter.to_pyg_data(next_state)
        with torch.no_grad():
            probe = self._policy.step(next_graph_obs, next_legal, next_rstate)
        return float(probe.value)


# Kept far above any realistic training seed range (base_seed is normally a
# small int from experiment.seed) so a deterministic evaluation pass never
# regenerates a scenario instance a training episode already used.
EVAL_SEED_OFFSET = 1_000_000_000


async def run_evaluation_episodes(
    policy: RecurrentPolicy,
    adapter: NasimEmuAdapter,
    run_id: str,
    num_episodes: int,
    seed_start: int,
    consultation_enabled: bool = False,
    consultation_cost: float = 0.0,
    consult_fn: ConsultFn | None = None,
) -> list[EpisodeSummary]:
    """Runs ``num_episodes`` deterministic (greedy) episodes with the
    current policy weights and reports their returns, contributing nothing
    to the PPO buffer -- an ``EvalCallback``-style periodic evaluation pass
    (see e.g. stable-baselines3), not a genuine held-out generalization
    test: MARLA's config has a single ``environment.scenario``, not a
    train/test scenario split, so this measures the current policy's
    performance without exploration noise on the same scenario training
    uses, not generalization to unseen scenarios.

    Reuses ``adapter`` rather than building a second instance: training and
    evaluation never run concurrently (this is awaited strictly between
    rollout-collection phases in ``run_training_loop``), and
    ``NasimEmuAdapter`` holds no state of its own between ``reset()`` calls.
    """
    policy.eval()
    try:
        collector = RolloutCollector(
            adapter=adapter,
            policy=policy,
            run_id=run_id,
            base_seed=seed_start,
            consultation_enabled=consultation_enabled,
            consultation_cost=consultation_cost,
            consult_fn=consult_fn,
            deterministic=True,
        )
        with torch.no_grad():
            _, summaries = await collector.collect(num_episodes * adapter.max_episode_steps)
    finally:
        policy.train()
    return summaries[:num_episodes]
