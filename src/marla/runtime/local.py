"""Local execution mode (spec section 4.1).

All configured agents run in one Python process, started from one
``spade.run()`` main function, sharing one asyncio event loop -- but they
still communicate exclusively through real SPADE messages over a real (here,
embedded) XMPP server, never through direct Python method calls between
agents.
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path

import spade
import torch

from marla.config.models import Config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.trainer import build_policy_and_optimizer
from marla.scenarios.uri import resolve_scenario_reference

logger = logging.getLogger(__name__)


class LocalRunError(Exception):
    """Raised when a local-mode run ends in failure (mirrors EXPERIMENT_FAILED)."""


def resolve_scenario_path(config: Config, config_dir: Path) -> str:
    """Resolves ``config.environment.scenario`` to whatever a
    ``NasimEmuAdapter`` actually needs: a real filesystem path for a
    ``marla://...`` reference (see :mod:`marla.scenarios.uri`) or a
    ``.yaml`` filesystem path, or the reference unchanged for a NASimEmu
    named/procedural scenario. This is the one place ``marla run`` (local
    and, via this function, distributed mode) turns a persisted config's
    scenario reference into something NASimEmu can load -- the config
    itself, on disk, keeps whatever reference it had (a ``marla://`` URI
    is never rewritten to the resolved path it happened to materialize to
    on this machine).
    """
    return resolve_scenario_reference(config.environment.scenario, config_dir)


def resolve_password(password_env: str | None, alias: str) -> str:
    if password_env is None:
        raise LocalRunError(f"Agent {alias!r} has no password_env configured")
    password = os.environ.get(password_env)
    if not password:
        raise LocalRunError(f"Environment variable {password_env!r} for agent {alias!r} is not set")
    return password


def resolve_run_dir(config: Config) -> Path:
    """``<metrics.output_directory>/<experiment-name>/<run-id>/`` (spec section 21)."""
    run_id = config.experiment.run_id or config.experiment.name
    return Path(config.metrics.output_directory) / config.experiment.name / run_id


def resolve_debug_dir(config: Config, run_id: str) -> Path:
    """Where ``--debug`` writes every Plan Maker query/response pair (see PlanMakerAgent).

    Deliberately its own top-level ``debug/`` directory, not nested under
    ``metrics.output_directory`` (``runs/`` by default) -- it holds ad hoc
    inspection files for a human, not run metrics/results, and is easy to
    overlook several directories deep.
    """
    debug_dir = Path("debug") / run_id
    debug_dir.mkdir(parents=True, exist_ok=True)
    return debug_dir


async def _build_and_run(
    config: Config,
    config_dir: Path,
    num_rollouts: int,
    debug: bool = False,
    resume_from: Path | None = None,
    run_dir: Path | None = None,
) -> "object":
    from marla.agents.gatekeeper import GatekeeperAgent
    from marla.agents.orchestrator import RLOrchestratorAgent
    from marla.agents.plan_maker import PlanMakerAgent
    from marla.knowledge.retriever import load_knowledge_base
    from marla.learning.checkpoint import load_checkpoint
    from marla.models.backend_factory import build_backend
    from marla.runtime.device import resolve_device
    from marla.runtime.lifecycle import register_sigint_handler

    resolved_device = resolve_device(config.device)
    scenario_path = resolve_scenario_path(config, config_dir)
    run_id = config.experiment.run_id or config.experiment.name
    stop_event = asyncio.Event()
    register_sigint_handler(stop_event)

    # Without this, policy weight initialization and every stochastic
    # action/query draw (torch.distributions.Categorical/Bernoulli, used
    # throughout learning/rollout.py) are governed by process-startup
    # entropy, not experiment.seed -- two runs with the "same" seed would
    # not actually be reproducible or comparable. Mirrors what the
    # test-only run_baseline_training() already does (learning/trainer.py).
    # Harmless when resuming: load_checkpoint's restore_rng_state below
    # overwrites this immediately afterward with the checkpoint's own RNG
    # state, and the weights this seeds are about to be overwritten too.
    torch.manual_seed(config.experiment.seed)
    policy, optimizer = build_policy_and_optimizer(
        config, resolved_device.torch_device, consultation_enabled=config.consultation.mode == "learned"
    )

    initial_environment_steps = 0
    initial_update_count = 0
    initial_scheduler_state: dict | None = None
    resumed_seed = config.experiment.seed
    if resume_from is not None:
        metadata = load_checkpoint(
            resume_from, policy, optimizer, map_location=resolved_device.torch_device, restore_rng_state=True
        )
        initial_environment_steps = metadata.environment_steps
        initial_update_count = metadata.update_count
        initial_scheduler_state = metadata.scheduler_state_dict
        if metadata.next_episode_seed is not None:
            resumed_seed = metadata.next_episode_seed
        print(
            f"Resumed from {resume_from}: {initial_environment_steps} environment step(s) / "
            f"{initial_update_count} update(s) already done, continuing episode seeds from "
            f"{resumed_seed}, RNG state restored: {metadata.rng_state_restored}, "
            f"LR scheduler ({metadata.scheduler_type}) restored from step {metadata.scheduler_state_dict.get('last_epoch') if metadata.scheduler_state_dict else 'n/a'}",
            flush=True,
        )

    # One independent NasimEmuAdapter per environment stream (spec: the
    # 2048-transition, 4-independent-environment-stream ablation) --
    # num_envs=1 (every existing config) builds exactly one, matching
    # pre-multi-env behavior exactly (never a slice of one shared adapter).
    adapters = [
        NasimEmuAdapter(
            scenario=scenario_path,
            max_episode_steps=config.environment.max_episode_steps,
            completion_reward=config.objective.completion_reward,
            premature_finish_penalty=config.objective.premature_finish_penalty,
            premature_finish_penalty_per_remaining_target=config.objective.premature_finish_penalty_per_remaining_target,
        )
        for _ in range(config.policy.ppo.num_envs)
    ]
    adapter = adapters if len(adapters) > 1 else adapters[0]

    required_participants: dict[str, str] = {}
    support_agents = []

    if config.consultation.mode == "learned":
        assert config.gatekeeper is not None
        assert len(config.agents) == 1  # first release: exactly one Plan Maker (spec non-goals)
        plan_maker_cfg = config.agents[0]
        required_participants[config.gatekeeper.alias] = config.gatekeeper.jid
        required_participants[plan_maker_cfg.alias] = plan_maker_cfg.jid

        gatekeeper_password = resolve_password(config.gatekeeper.password_env, config.gatekeeper.alias)
        gatekeeper = GatekeeperAgent(
            jid=config.gatekeeper.jid,
            password=gatekeeper_password,
            alias=config.gatekeeper.alias,
            run_id=run_id,
            orchestrator_alias=config.rl_orchestrator.alias,
            orchestrator_jid=config.rl_orchestrator.jid,
            plan_maker_alias=plan_maker_cfg.alias,
            plan_maker_jid=plan_maker_cfg.jid,
            max_schema_revisions=config.consultation.max_schema_revisions,
        )

        knowledge_base = load_knowledge_base(plan_maker_cfg.knowledge.path)
        if knowledge_base.version != plan_maker_cfg.knowledge.version:
            raise LocalRunError(
                f"Configured knowledge.version {plan_maker_cfg.knowledge.version!r} does not match "
                f"the loaded knowledge file's version {knowledge_base.version!r}"
            )
        backend = build_backend(plan_maker_cfg.model, config.device)
        plan_maker_password = resolve_password(plan_maker_cfg.password_env, plan_maker_cfg.alias)
        debug_dir = resolve_debug_dir(config, run_id) if debug else None
        if debug_dir is not None:
            print(f"--debug: Plan Maker queries/responses will be written to {debug_dir}/", flush=True)
        plan_maker = PlanMakerAgent(
            jid=plan_maker_cfg.jid,
            password=plan_maker_password,
            alias=plan_maker_cfg.alias,
            run_id=run_id,
            gatekeeper_alias=config.gatekeeper.alias,
            gatekeeper_jid=config.gatekeeper.jid,
            backend=backend,
            model_version=plan_maker_cfg.model.name,
            knowledge_base=knowledge_base,
            resolved_device=getattr(backend, "resolved_device", "n/a"),
            debug_dir=debug_dir,
        )
        support_agents = [gatekeeper, plan_maker]

    orchestrator_password = resolve_password(config.rl_orchestrator.password_env, config.rl_orchestrator.alias)

    orchestrator = RLOrchestratorAgent(
        jid=config.rl_orchestrator.jid,
        password=orchestrator_password,
        alias=config.rl_orchestrator.alias,
        run_id=run_id,
        config=config,
        policy=policy,
        optimizer=optimizer,
        adapter=adapter,
        device=resolved_device.torch_device,
        seed=resumed_seed,
        num_rollouts=num_rollouts,
        initial_environment_steps=initial_environment_steps,
        initial_update_count=initial_update_count,
        initial_scheduler_state=initial_scheduler_state,
        required_participants=required_participants,
        gatekeeper_alias=config.gatekeeper.alias if config.gatekeeper else None,
        gatekeeper_jid=config.gatekeeper.jid if config.gatekeeper else None,
        stop_event=stop_event,
        run_dir=run_dir,
    )

    for agent in support_agents:
        await agent.start()
    await orchestrator.start()
    await spade.wait_until_finished(orchestrator)
    return orchestrator


def run_local(
    config: Config,
    config_dir: Path,
    num_rollouts: int,
    embedded_xmpp_server: bool = True,
    debug: bool = False,
    resume_from: Path | None = None,
    run_dir: Path | None = None,
):
    """Run a local-mode experiment; blocks until the RL Orchestrator finishes.

    Raises :class:`LocalRunError` if the run ended in failure. Returns the
    stopped :class:`RLOrchestratorAgent` (holding ``training_result``) on
    success.

    ``resume_from``: a checkpoint.pt to load policy/optimizer/RNG state
    from and continue training rather than starting fresh (research/
    aamas2027's staged training) -- ``num_rollouts`` here must already be
    computed as the *remaining* rollouts to the config's target step count,
    not the full target (see cli.py's ``run`` command).

    SPADE's ``Container`` is a process-wide singleton, and its event loop is
    closed at the end of ``spade.run()`` -- calling ``run_local`` more than
    once in the same process reuses a closed loop and fails. This matches
    real usage (``marla run`` is always a fresh process); tests that need
    more than one local-mode run must isolate each in its own subprocess.
    """
    if config.execution.mode != "local":
        raise ValueError("run_local requires execution.mode == 'local'")

    result: dict[str, object] = {}

    async def main() -> None:
        try:
            result["orchestrator"] = await _build_and_run(
                config, config_dir, num_rollouts, debug=debug, resume_from=resume_from, run_dir=run_dir
            )
        except Exception as exc:
            # SPADE's container.run() swallows exceptions raised in main()
            # (it only logs them) rather than propagating them to run_local's
            # caller, so this must be captured explicitly or a construction
            # failure (e.g. a bad model name, a missing password_env) would
            # silently leave `orchestrator` unset instead of failing loudly.
            result["error"] = exc
            raise

    spade.run(main(), embedded_xmpp_server=embedded_xmpp_server)

    if "error" in result:
        exc = result["error"]
        raise LocalRunError(str(exc)) from exc

    orchestrator = result.get("orchestrator")
    if orchestrator is not None and getattr(orchestrator, "failure", None) is not None:
        raise LocalRunError(str(orchestrator.failure)) from orchestrator.failure
    return orchestrator
