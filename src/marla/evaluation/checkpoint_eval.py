"""Load a completed run's checkpoint and evaluate it -- optionally on a
different scenario, optionally under an :class:`EvaluationOverrides`
ablation -- with no gradient step ever taken.

Composes existing, already-tested building blocks rather than
reimplementing anything: ``config.loader.load_config`` (the run's own saved
``config.yaml``, so policy shape is always read off what actually trained,
never guessed), ``learning.checkpoint.load_checkpoint``,
``learning.recurrent_policy.RecurrentPolicy``,
``environment.nasimemu_adapter.NasimEmuAdapter``, and
``learning.rollout.RolloutCollector`` (with the ``overrides`` hook added
alongside this module). See ``research/aamas2027/AUDIT.md`` sections 1-4
and the plan's "Architecture" section for why this lives here instead of
in the training-facing CLI.

No ``torch.optim`` import anywhere in this module, and no ``.backward()``/
``.step()`` call -- structurally impossible to take a gradient step here
(research/aamas2027/VALIDATION.md check 9).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from marla.config.loader import load_config
from marla.config.models import Config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.evaluation.advisory_cache import AdvisoryCache
from marla.evaluation.direct_consult import DirectConsultant
from marla.evaluation.overrides import EvaluationOverrides
from marla.learning.checkpoint import load_checkpoint
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import ConsultFn, EpisodeSummary, RolloutCollector, StepRecord
from marla.messaging.schemas import AdvisoryObjective
from marla.runtime.device import DeviceRequest, resolve_device


@dataclass
class EvaluationRunResult:
    records: list[StepRecord]
    summaries: list[EpisodeSummary]
    consultation_enabled: bool
    scenario: str
    seed_start: int
    num_episodes: int
    cache_hits: int = 0
    cache_misses: int = 0


def load_run_config(run_dir: Path) -> Config:
    """The exact config a run trained with -- policy shape is read off this,
    never assumed, so a mismatched ``consultation_enabled`` can't silently
    build the wrong-shaped policy before ``load_checkpoint``'s strict
    ``state_dict`` load would catch it anyway."""
    return load_config(run_dir / "config.yaml")


def _build_consult_fn(config: Config, run_id: str, device: DeviceRequest, cache: AdvisoryCache | None) -> ConsultFn:
    from marla.knowledge.retriever import load_knowledge_base
    from marla.models.backend_factory import build_backend

    assert config.agents, "consultation_enabled requires exactly one plan_maker agent in config.agents"
    plan_maker_cfg = config.agents[0]
    knowledge_base = load_knowledge_base(plan_maker_cfg.knowledge.path)
    if knowledge_base.version != plan_maker_cfg.knowledge.version:
        raise ValueError(
            f"Configured knowledge.version {plan_maker_cfg.knowledge.version!r} does not match "
            f"the loaded knowledge file's version {knowledge_base.version!r}"
        )
    backend = build_backend(plan_maker_cfg.model, device)
    return DirectConsultant(
        backend=backend,
        knowledge_base=knowledge_base,
        model_version=plan_maker_cfg.model.name,
        objective=AdvisoryObjective(type=config.objective.type, description=config.objective.description),
        run_id=run_id,
        max_schema_revisions=config.consultation.max_schema_revisions or 0,
        cache=cache,
    )


def build_policy(
    config: Config, consultation_enabled: bool, device: torch.device, checkpoint_path: Path | None, seed: int | None = None
) -> RecurrentPolicy:
    """A policy matching ``config.policy`` -- loaded from ``checkpoint_path``
    if given, otherwise freshly initialized (seeded if ``seed`` is given) and
    never trained. The untrained path exists for PLAN_MAKER_ONLY, which uses
    a policy purely as a logging scaffold (see overrides.PLAN_MAKER_ONLY):
    action selection never consults it, so using an untrained network
    removes any ambiguity about whether RL learning influenced that
    condition."""
    if seed is not None:
        torch.manual_seed(seed)
    policy = RecurrentPolicy(config.policy, consultation_enabled=consultation_enabled).to(device)
    if checkpoint_path is not None:
        load_checkpoint(checkpoint_path, policy, optimizer=None, map_location=device)
    policy.eval()
    return policy


async def evaluate_checkpoint(
    run_dir: Path,
    seed_start: int,
    num_episodes: int,
    overrides: EvaluationOverrides | None = None,
    scenario_path: str | None = None,
    device: DeviceRequest = "auto",
    policy_override: RecurrentPolicy | None = None,
    advisory_cache_path: Path | None = None,
    run_id: str | None = None,
) -> EvaluationRunResult:
    """Evaluate ``run_dir``'s checkpoint (or ``policy_override``, for a
    scaffold that was never trained) on ``num_episodes`` deterministic
    episodes starting at seed ``seed_start`` (a contiguous range --
    ``RolloutCollector`` increments its own seed counter by 1 per episode,
    exactly matching how every other MARLA eval seed set in this program is
    specified). No optimizer is constructed; nothing here calls
    ``.backward()``/``.step()``.
    """
    config = load_run_config(run_dir)
    resolved_device = resolve_device(device)
    consultation_enabled = config.consultation.mode == "learned"
    resolved_run_id = run_id or (config.experiment.run_id or config.experiment.name)

    if policy_override is not None:
        policy = policy_override
        policy.eval()
    else:
        policy = build_policy(
            config, consultation_enabled, resolved_device.torch_device, checkpoint_path=run_dir / "checkpoint.pt"
        )

    target_scenario = scenario_path or config.environment.scenario
    adapter = NasimEmuAdapter(
        scenario=target_scenario,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
        premature_finish_penalty_per_remaining_target=config.objective.premature_finish_penalty_per_remaining_target,
    )

    consult_fn: ConsultFn | None = None
    consultant: DirectConsultant | None = None
    if consultation_enabled:
        cache = AdvisoryCache(advisory_cache_path) if advisory_cache_path is not None else None
        consultant = _build_consult_fn(config, resolved_run_id, device, cache)
        consult_fn = consultant

    collector = RolloutCollector(
        adapter=adapter,
        policy=policy,
        run_id=resolved_run_id,
        base_seed=seed_start,
        consultation_enabled=consultation_enabled,
        consultation_cost=config.consultation.cost,
        consult_fn=consult_fn,
        deterministic=True,
        overrides=overrides,
    )
    with torch.no_grad():
        records, summaries = await collector.collect(num_episodes * adapter.max_episode_steps)

    # collect() runs a fixed *step* budget, not a fixed *episode* count --
    # short episodes can complete more than num_episodes within that budget.
    # Keep records and summaries consistent with each other (decisions.csv
    # must only ever contain decisions from episodes episodes.csv also
    # reports) by truncating both to the same set of episode_ids, not just
    # slicing summaries.
    kept_summaries = summaries[:num_episodes]
    kept_episode_ids = {s.episode_id for s in kept_summaries}
    kept_records = [r for r in records if r.episode_id in kept_episode_ids]

    return EvaluationRunResult(
        records=kept_records,
        summaries=kept_summaries,
        consultation_enabled=consultation_enabled,
        scenario=target_scenario,
        seed_start=seed_start,
        num_episodes=num_episodes,
        cache_hits=consultant.cache_hits if consultant is not None else 0,
        cache_misses=consultant.cache_misses if consultant is not None else 0,
    )
