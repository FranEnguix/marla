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
from dataclasses import dataclass, field

import torch
from torch_geometric.data import Data

from marla.environment.action_compatibility import compute_compatibility_matrix
from marla.environment.actions import ActionDescriptor
from marla.environment.nasimemu_adapter import EnvironmentState, NasimEmuAdapter
from marla.environment.observation_summary import build_observation_summary
from marla.environment.state_delta import AccessGain, StateDelta, compute_state_delta
from marla.environment.visible_facts import (
    all_visible_sensitive_targets_rooted,
    assemble_visible_progress,
    extract_visible_host_facts,
    extract_visible_network_exploration,
)
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
    # (N, COMPATIBILITY_FEATURE_DIM) -- the exact tensor passed to
    # ActionEncoder at collection time (spec section 5/11). Training-buffer
    # state only, like graph_data/legal_action_descriptors above: discarded
    # with the rest of this rollout's StepRecords after the PPO update,
    # never persisted to decisions.csv. PPO replay (learning/ppo.py) reuses
    # this exact tensor rather than recomputing it from (by then stale,
    # and inaccessible without calling the simulator again) live state.
    compatibility_features: torch.Tensor
    # (VISIBLE_PROGRESS_DIM,) -- the exact visible-only global-progress
    # vector passed to RecurrentCore at collection time (FINISH-
    # learnability investigation). Same training-buffer-only/replay-reuse
    # contract as compatibility_features above.
    visible_progress: torch.Tensor

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

    # Cheap, scalar-only additions for decisions.csv (spec section 3) --
    # none of these retain any tensor or full observation, only what's
    # needed to reconstruct a compact metrics row after PPO replay.
    selected_action_type: str = ""
    objective_satisfied: bool = False
    objective_became_satisfied: bool = False
    # Compact scalars only (spec: no hidden simulator state/large
    # observations persisted merely to compute these) -- see
    # NasimEmuAdapter.sensitive_target_status. sensitive_targets_with_root
    # is derived (total - remaining), not separately queried.
    sensitive_targets_total: int | None = None
    sensitive_targets_with_root: int | None = None
    sensitive_targets_remaining: int | None = None
    # Decision-time diagnostic truth (FINISH investigation): the objective/
    # target status that existed WHEN THE POLICY CHOSE THIS STEP'S ACTION --
    # i.e. queried from the live simulator before this step's action was
    # applied, never inferred retrospectively from the post-action
    # transition above. This answers "was FINISH already the correct
    # action at this decision?", a different question from
    # objective_satisfied/objective_became_satisfied (which describe the
    # state AFTER this step's action). Diagnostic simulator truth only --
    # never fed to the policy as an input; see
    # marla.environment.action_compatibility/visible_facts for what the
    # policy is actually allowed to see.
    objective_satisfied_before_action: bool = False
    sensitive_targets_remaining_before_action: int | None = None
    sensitive_targets_with_root_before_action: int | None = None
    # Observable subnet-exploration progress (spec sections 3-24), decision-
    # time, visible-only (built from marla.environment.visible_facts
    # .VisibleNetworkExploration -- known subnets = have >=1 visible host;
    # scanned = successful SubnetScan at least once). known_subnets_total
    # is the size of the KNOWN set, never the true scenario subnet count.
    known_subnets_total_before_action: int = 0
    known_subnets_scanned_before_action: int = 0
    known_exploration_frontier_remaining_before_action: bool = False
    # Visible-only "are all currently-confirmed-sensitive targets already
    # rooted" (None when no sensitive target is visible yet -- see
    # all_visible_sensitive_targets_rooted's own docstring for why that
    # must never collapse into True). Used for the frontier-conditional
    # FINISH diagnostic (P(FINISH | visible targets complete, frontier=...)),
    # which is deliberately a DIFFERENT question from the true-objective
    # one above (objective_satisfied_before_action).
    all_visible_sensitive_targets_rooted_before_action: bool | None = None
    has_visible_sensitive_target_before_action: bool = False
    state_delta: StateDelta = field(default_factory=lambda: StateDelta(0, 0, 0, 0, AccessGain.NONE))
    # Set after collection, once the rollout's total step count is known
    # (mirrors EpisodeSummary.rollout below) -- None only for a StepRecord
    # never passed through run_training_loop (e.g. constructed directly in
    # a unit test).
    global_environment_step: int | None = None
    rollout: int | None = None
    # Multi-environment PPO collection (spec: the 2048-transition, 4-
    # independent-environment-stream ablation) -- which of N independent
    # environment streams this record came from, purely for diagnostic/
    # reporting purposes (spec section 41's per-environment metrics).
    # None for every single-environment run (the default, unchanged
    # behavior) -- never read by GAE, chunking, or PPO update logic,
    # which all key exclusively on `episode_id` (see
    # RolloutCollector.__init__'s `episode_id_offset` for how episode_id
    # stays globally unique across streams without this field's help).
    env_index: int | None = None

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
    """Success semantics (do not conflate these -- see each field's own
    docstring): ``objective_reached`` is true the instant the simulator
    state satisfies the objective, independent of FINISH; ``successful_finish``
    (and its synonym ``episode_success``) additionally requires the policy
    to have explicitly selected FINISH while that held. A timeout after the
    objective was reached is ``objective_reached=True`` but
    ``successful_finish=False`` -- a real, important diagnostic case, not a
    plain failure indistinguishable from never having reached the objective
    at all.
    """

    run_id: str
    episode_id: int
    seed: int
    # DEPRECATED: kept only for backward compatibility with existing
    # analysis code that reads this column. Identical to
    # ``successful_finish`` -- NEVER to ``objective_reached`` -- and new
    # code should read ``successful_finish``/``episode_success`` instead,
    # which say what they mean without relying on institutional memory of
    # what "goal_success" was defined to mean.
    goal_success: bool
    nasimemu_return: float
    training_return: float
    environment_steps: int
    # First step at which the objective became satisfied, regardless of
    # whether/when FINISH was later selected -- decoupled from finish_step
    # below (spec section 5). None if the objective was never satisfied.
    steps_to_goal: int | None
    # The step at which FINISH was selected, whether or not that made the
    # episode a success (a premature FINISH still has a finish_step, just
    # no steps_to_goal). None for a truncated (timeout) episode.
    finish_step: int | None
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
    # finish_step - steps_to_goal, only for a genuinely successful episode
    # (successful_finish) where both exist -- see spec section 5. Not
    # meaningful (left None) for a premature finish or a timeout.
    finish_delay_steps: int | None = None
    # True for a deterministic (greedy) evaluation episode from
    # run_evaluation_episodes, contributing nothing to the PPO buffer.
    # False for an ordinary stochastic training episode.
    is_eval: bool = False
    # True the moment adapter.objective_satisfied() ever becomes True
    # during this episode (all sensitive targets rooted), regardless of
    # whether/when FINISH was selected afterward. Equivalent to
    # ``steps_to_goal is not None`` -- kept as its own explicit boolean so
    # analysis code doesn't have to re-derive it from a timing field.
    objective_reached: bool = False
    # True only if the policy explicitly selected FINISH while the
    # objective held (identical value to the deprecated ``goal_success``).
    successful_finish: bool = False
    # MARLA's protocol defines "successful episode" as an explicit FINISH
    # after the objective is satisfied -- so this is always identical to
    # ``successful_finish``, never to ``objective_reached``. Kept as a
    # separate, explicitly-documented field rather than expecting every
    # reader to know that definition (spec section 20).
    episode_success: bool = False
    # Final simulator state at episode end (whether ended by FINISH or by
    # truncation) -- a direct query, not a running/cached value, so it is
    # correct even when FINISH (which itself performs no simulator action)
    # ends the episode.
    sensitive_targets_total: int | None = None
    sensitive_targets_with_root_final: int | None = None
    sensitive_targets_remaining_final: int | None = None


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
        episode_id_offset: int = 0,
        env_index: int | None = None,
    ) -> None:
        """``episode_id_offset``/``env_index``: multi-environment PPO
        collection support (spec: the 2048-transition, 4-independent-
        environment-stream ablation). Both default to values that make
        this collector byte-identical to before they existed:
        ``episode_id_offset=0`` means episode IDs start at 1 exactly as
        always; ``env_index=None`` leaves ``StepRecord.env_index`` unset
        on every record this collector produces. When multiple
        ``RolloutCollector`` instances are combined into one PPO batch
        (see ``learning.rollout.env_seed_and_episode_offset`` and
        ``learning.trainer.run_training_loop``'s multi-env branch), each
        instance is given a DISTINCT, non-overlapping
        ``episode_id_offset`` so ``episode_id`` remains a globally unique
        episode identifier across every stream in the combined batch --
        this is what lets every existing "group by episode_id" code path
        (GAE-adjacent chunking, decision-row building, the value-
        calibration/probe state-selection tooling) keep working completely
        UNCHANGED on a multi-env batch, with zero risk of two different
        environments' episode 1 being confused for one another.
        ``env_index`` is purely a diagnostic/reporting field (per-
        environment metrics, spec section 41) -- no GAE, chunking, or PPO
        logic ever reads it.
        """
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
        self._episode_id_offset = episode_id_offset
        self._env_index = env_index

        self._episode_id = episode_id_offset
        self._state: EnvironmentState | None = None
        self._rstate: RecurrentState | None = None
        self._episode_seed = 0
        self._episode_start_time = 0.0
        self._episode_nasimemu_return = 0.0
        self._episode_training_return = 0.0
        self._episode_steps = 0
        self._steps_to_goal: int | None = None
        self._finish_step: int | None = None
        self._objective_satisfied_so_far = False
        self._episode_consultation_count = 0
        self._episode_consultation_cost_total = 0.0
        self._episode_schema_rejection_count = 0

    @property
    def next_episode_seed(self) -> int:
        """The seed the *next* episode this collector starts will use.

        A resumed training run passes this (from the prior invocation's
        final value, via ``CheckpointMetadata.next_episode_seed``) as the
        new collector's ``base_seed``, so episode seeds keep progressing
        forward instead of repeating the same sequence a fresh run with
        ``base_seed=experiment.seed`` would collect.
        """
        return self._seed_counter

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
        self._finish_step = None
        self._objective_satisfied_so_far = False
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
            graph_obs = self._adapter.to_pyg_data(
                state, include_subnet_scan_feature=self._policy.visible_subnet_exploration_enabled
            )
            # Computed once, here, from the live pre-action state -- passed
            # explicitly into policy.step() and stored verbatim on this
            # step's StepRecord, so PPO replay reuses the identical tensor
            # rather than recomputing it later (spec section 11; the
            # simulator must never be called again during replay).
            facts_by_target = extract_visible_host_facts(state)
            compatibility_matrix = compute_compatibility_matrix(legal_actions, facts_by_target)
            # Visible-only global progress summary (FINISH-learnability
            # investigation, spec section 18; extended with observable
            # subnet-exploration progress) -- computed from the exact same
            # state as compatibility_matrix above, passed explicitly into
            # policy.step() and stored verbatim on this step's StepRecord,
            # for the same replay-consistency reason. assemble_visible_progress
            # only includes each component the policy was actually
            # constructed with (spec: disabling a component removes it
            # from the input, never just zeroes it).
            exploration = extract_visible_network_exploration(state)
            visible_progress = assemble_visible_progress(
                facts_by_target, exploration,
                self._policy.visible_target_progress_enabled, self._policy.visible_subnet_exploration_enabled,
            )
            # FINISH diagnostic: the objective/target status AT THE MOMENT
            # the policy is about to decide, queried from the live
            # simulator before this step's action is applied below -- never
            # inferred after the fact from the resulting transition. This
            # is diagnostic simulator truth (for decisions.csv only), not a
            # policy input: nothing here is passed into policy.step().
            objective_satisfied_before_action = self._adapter.objective_satisfied()
            sensitive_targets_total_before, sensitive_targets_remaining_before = self._adapter.sensitive_target_status()
            # Exploration diagnostics (spec sections 21-24) -- built from
            # the SAME VisibleNetworkExploration used for the policy input
            # above (never a second interpretation), plus the visible-only
            # "are all currently-visible sensitive targets rooted" flag
            # used for the frontier-conditional FINISH diagnostic.
            known_subnets_total_before_action = len(exploration.known_subnets)
            known_subnets_scanned_before_action = len(exploration.successfully_scanned_subnets)
            known_exploration_frontier_remaining_before_action = exploration.known_exploration_frontier_remaining
            all_visible_sensitive_targets_rooted_before_action = all_visible_sensitive_targets_rooted(facts_by_target)
            has_visible_sensitive_target_before_action = any(f.is_sensitive_target for f in facts_by_target.values())
            step_out = self._policy.step(graph_obs, legal_actions, rstate, compatibility_matrix, visible_progress)

            decision = await self._decide(legal_actions, step_out, state)
            action_index = decision["action_index"]
            selected = legal_actions[action_index]

            transition = self._adapter.step(selected)
            consultation_cost = decision["consultation_cost"]
            training_reward = transition.nasimemu_reward - consultation_cost

            # Called explicitly, not read from transition.info: the info
            # dict only ever carries "objective_satisfied" on the FINISH
            # branch (see NasimEmuAdapter.step), but steps_to_goal now needs
            # this on *every* step to detect the exact transition step,
            # independent of whether/when FINISH is later selected (spec
            # section 5). Cheap: a direct query of the underlying env's own
            # current state, not derived from any stored snapshot.
            objective_satisfied_now = self._adapter.objective_satisfied()
            objective_became_satisfied = objective_satisfied_now and not self._objective_satisfied_so_far
            sensitive_targets_total, sensitive_targets_remaining = self._adapter.sensitive_target_status()
            state_delta = compute_state_delta(state, transition.state)

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
                compatibility_features=compatibility_matrix,
                visible_progress=visible_progress,
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
                selected_action_type=selected.action_type,
                objective_satisfied=objective_satisfied_now,
                objective_became_satisfied=objective_became_satisfied,
                sensitive_targets_total=sensitive_targets_total,
                sensitive_targets_with_root=sensitive_targets_total - sensitive_targets_remaining,
                sensitive_targets_remaining=sensitive_targets_remaining,
                objective_satisfied_before_action=objective_satisfied_before_action,
                sensitive_targets_remaining_before_action=sensitive_targets_remaining_before,
                sensitive_targets_with_root_before_action=(
                    sensitive_targets_total_before - sensitive_targets_remaining_before
                ),
                known_subnets_total_before_action=known_subnets_total_before_action,
                known_subnets_scanned_before_action=known_subnets_scanned_before_action,
                known_exploration_frontier_remaining_before_action=known_exploration_frontier_remaining_before_action,
                all_visible_sensitive_targets_rooted_before_action=all_visible_sensitive_targets_rooted_before_action,
                has_visible_sensitive_target_before_action=has_visible_sensitive_target_before_action,
                state_delta=state_delta,
                env_index=self._env_index,
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
            if objective_became_satisfied:
                self._steps_to_goal = self._episode_steps
            self._objective_satisfied_so_far = self._objective_satisfied_so_far or objective_satisfied_now
            if selected.is_finish:
                self._finish_step = self._episode_steps

            if transition.terminated or transition.truncated:
                # Unchanged fundamental rule: success requires an explicit
                # FINISH selection while the objective holds, not merely
                # having satisfied it at some point (see steps_to_goal
                # above, which tracks that separately for credit-assignment
                # analysis, not for this pass/fail determination).
                # successful_finish (and its deprecated name, goal_success)
                # requires an EXPLICIT FINISH while the objective holds --
                # unchanged rule. objective_reached is independent of
                # FINISH entirely: it is true the instant the simulator
                # state satisfies the objective (self._objective_satisfied_so_far,
                # already updated above to include this step), whether or
                # not FINISH ever gets selected afterward -- this is what
                # distinguishes "timeout after reaching the objective"
                # (objective_reached=True, successful_finish=False) from
                # "never reached it at all" (both False), a distinction
                # `goal_success` alone could not express.
                successful_finish = selected.is_finish and objective_satisfied_now
                objective_reached = self._objective_satisfied_so_far
                finish_delay_steps = (
                    self._finish_step - self._steps_to_goal
                    if successful_finish and self._steps_to_goal is not None and self._finish_step is not None
                    else None
                )
                sensitive_targets_total, sensitive_targets_remaining = self._adapter.sensitive_target_status()
                summaries.append(
                    EpisodeSummary(
                        run_id=self._run_id,
                        episode_id=self._episode_id,
                        seed=self._episode_seed,
                        goal_success=successful_finish,
                        nasimemu_return=self._episode_nasimemu_return,
                        training_return=self._episode_training_return,
                        environment_steps=self._episode_steps,
                        steps_to_goal=self._steps_to_goal,
                        finish_step=self._finish_step,
                        finish_delay_steps=finish_delay_steps,
                        episode_seconds=time.monotonic() - self._episode_start_time,
                        finish_reason="finish" if selected.is_finish else "truncated",
                        consultation_count=self._episode_consultation_count,
                        consultation_cost=self._episode_consultation_cost_total,
                        schema_rejection_count=self._episode_schema_rejection_count,
                        is_eval=self._deterministic,
                        objective_reached=objective_reached,
                        successful_finish=successful_finish,
                        episode_success=successful_finish,
                        sensitive_targets_total=sensitive_targets_total,
                        sensitive_targets_with_root_final=sensitive_targets_total - sensitive_targets_remaining,
                        sensitive_targets_remaining_final=sensitive_targets_remaining,
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
        next_graph_obs = self._adapter.to_pyg_data(
            next_state, include_subnet_scan_feature=self._policy.visible_subnet_exploration_enabled
        )
        next_facts = extract_visible_host_facts(next_state)
        next_compatibility = compute_compatibility_matrix(next_legal, next_facts)
        next_exploration = extract_visible_network_exploration(next_state)
        next_visible_progress = assemble_visible_progress(
            next_facts, next_exploration,
            self._policy.visible_target_progress_enabled, self._policy.visible_subnet_exploration_enabled,
        )
        with torch.no_grad():
            probe = self._policy.step(next_graph_obs, next_legal, next_rstate, next_compatibility, next_visible_progress)
        return float(probe.value)


# Kept far above any realistic training seed range (base_seed is normally a
# small int from experiment.seed) so a deterministic evaluation pass never
# regenerates a scenario instance a training episode already used.
EVAL_SEED_OFFSET = 1_000_000_000

# Multi-environment PPO collection (spec: the 2048-transition, 4-
# independent-environment-stream ablation). Each of `num_envs` independent
# RolloutCollector instances needs (a) a distinct, non-overlapping
# base_seed range so no two streams' episodes ever reuse the same
# NASimEmu scenario-generation seed, and (b) a distinct, non-overlapping
# episode_id range so `episode_id` remains a globally unique episode
# identifier once all streams' records are combined into one PPO batch
# (see RolloutCollector.__init__'s own docstring). Both strides are kept
# far above any realistic per-stream episode count in a single collection
# window (mirrors EVAL_SEED_OFFSET's own reasoning above) -- deterministic,
# documented, reproducible.
ENV_SEED_STRIDE = 10_000_000
EPISODE_ID_STRIDE = 10_000_000


def env_base_seed(training_seed: int, env_index: int) -> int:
    """The ``base_seed`` one of ``num_envs`` independent RolloutCollector
    streams should use (spec section 14). Env 0 always gets exactly
    ``training_seed`` -- so a single-environment run (``num_envs=1``)
    uses the SAME base_seed the pre-multi-env ``RolloutCollector`` always
    used, byte-identical (spec section 6). Every other stream gets a
    distinct, non-overlapping range.
    """
    return training_seed + env_index * ENV_SEED_STRIDE


def env_episode_id_offset(env_index: int) -> int:
    """The ``episode_id_offset`` one of ``num_envs`` independent
    RolloutCollector streams should use. Env 0 always gets 0 (episode IDs
    start at 1, exactly the pre-multi-env convention -- spec section 6),
    every other stream gets a distinct, non-overlapping range.
    """
    return env_index * EPISODE_ID_STRIDE


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

    ``adapter`` must be a *different instance* than any adapter a training
    collector might resume mid-episode on -- ``NasimEmuAdapter`` wraps a
    genuinely mutable, stateful ``NASimEmuEnv`` (the live scenario
    instance, step index, ...), and this function's own ``reset()``/
    ``step()`` calls would otherwise silently overwrite that live state out
    from under a training collector's paused, resumed-next-rollout episode
    -- observed in practice for scenarios whose host count varies across
    resets (a "ranges of hosts" scenario file, which most bundled ones are)
    as a shape mismatch several steps later, not an error at the point of
    interference itself. ``run_training_loop`` builds a dedicated eval
    adapter for exactly this reason; only a caller with no concurrent
    training collector on the same adapter (e.g. a standalone evaluation
    script) may safely pass an adapter it also uses elsewhere, and only
    between full episodes.
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
