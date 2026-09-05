"""Distributed execution mode (spec section 4.2).

Each ``marla run`` process starts only the agents named via repeatable
``--agent`` options (by alias or JID), connecting to the configured,
externally-reachable XMPP server -- distributed mode never uses the
embedded ``pyjabber`` server, that is a local-mode convenience only. Every
process loads the identical configuration and shares ``experiment.run_id``
(already enforced by config validation); only the process that starts
``rl_orchestrator`` owns NASimEmu and writes central metrics.
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path

import spade
import torch

from marla.config.models import Config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.trainer import build_policy_and_optimizer
from marla.runtime.local import resolve_debug_dir, resolve_password, resolve_scenario_path

logger = logging.getLogger(__name__)


class DistributedRunError(Exception):
    """Raised when a distributed-mode process ends in failure."""


async def _build_and_run(
    config: Config, config_dir: Path, selected_aliases: set[str], num_rollouts: int, debug: bool = False
) -> list:
    from marla.agents.gatekeeper import GatekeeperAgent
    from marla.agents.orchestrator import RLOrchestratorAgent
    from marla.agents.plan_maker import PlanMakerAgent
    from marla.knowledge.retriever import load_knowledge_base
    from marla.models.backend_factory import build_backend
    from marla.runtime.device import resolve_device
    from marla.runtime.lifecycle import register_sigint_handler

    run_id = config.experiment.run_id
    assert run_id  # enforced by config validation in distributed mode
    agents: list = []
    stop_event = asyncio.Event()
    register_sigint_handler(stop_event)

    plan_maker_cfg = config.agents[0] if config.agents else None

    required_participants: dict[str, str] = {}
    if config.consultation.mode == "learned":
        assert config.gatekeeper is not None and plan_maker_cfg is not None
        required_participants[config.gatekeeper.alias] = config.gatekeeper.jid
        required_participants[plan_maker_cfg.alias] = plan_maker_cfg.jid

    if config.rl_orchestrator.alias in selected_aliases:
        resolved_device = resolve_device(config.device)
        scenario_path = resolve_scenario_path(config, config_dir)
        # See runtime/local.py's identical call for why: without this,
        # experiment.seed does not actually control policy init or
        # stochastic action/query sampling on the real run path.
        torch.manual_seed(config.experiment.seed)
        policy, optimizer = build_policy_and_optimizer(
            config, resolved_device.torch_device, consultation_enabled=config.consultation.mode == "learned"
        )
        adapter = NasimEmuAdapter(
            scenario=scenario_path,
            max_episode_steps=config.environment.max_episode_steps,
            completion_reward=config.objective.completion_reward,
            premature_finish_penalty=config.objective.premature_finish_penalty,
        )
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
            seed=config.experiment.seed,
            num_rollouts=num_rollouts,
            required_participants=required_participants,
            gatekeeper_alias=config.gatekeeper.alias if config.gatekeeper else None,
            gatekeeper_jid=config.gatekeeper.jid if config.gatekeeper else None,
            stop_event=stop_event,
        )
        agents.append(orchestrator)

    if config.gatekeeper is not None and config.gatekeeper.alias in selected_aliases:
        assert plan_maker_cfg is not None
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
            enable_disconnect_detection=True,
        )
        agents.append(gatekeeper)

    for agent_cfg in config.agents:
        if agent_cfg.alias not in selected_aliases:
            continue
        assert config.gatekeeper is not None
        knowledge_base = load_knowledge_base(agent_cfg.knowledge.path)
        if knowledge_base.version != agent_cfg.knowledge.version:
            raise DistributedRunError(
                f"Configured knowledge.version {agent_cfg.knowledge.version!r} does not match "
                f"the loaded knowledge file's version {knowledge_base.version!r}"
            )
        backend = build_backend(agent_cfg.model, config.device)
        plan_maker_password = resolve_password(agent_cfg.password_env, agent_cfg.alias)
        debug_dir = resolve_debug_dir(config, run_id) if debug else None
        if debug_dir is not None:
            print(f"--debug: Plan Maker queries/responses will be written to {debug_dir}/", flush=True)
        plan_maker = PlanMakerAgent(
            jid=agent_cfg.jid,
            password=plan_maker_password,
            alias=agent_cfg.alias,
            run_id=run_id,
            gatekeeper_alias=config.gatekeeper.alias,
            gatekeeper_jid=config.gatekeeper.jid,
            backend=backend,
            model_version=agent_cfg.model.name,
            knowledge_base=knowledge_base,
            resolved_device=getattr(backend, "resolved_device", "n/a"),
            enable_disconnect_detection=True,
            debug_dir=debug_dir,
        )
        agents.append(plan_maker)

    if not agents:
        raise DistributedRunError(f"No configured agents matched the selected aliases: {sorted(selected_aliases)}")

    # Non-orchestrator agents first: if this process also hosts the RL
    # Orchestrator, its lifecycle behaviour starts sending READY_CHECK the
    # moment it is added, and a message to an agent whose behaviours aren't
    # registered yet is simply lost (no message queuing across processes
    # either without server-side persistence) -- see
    # OrchestratorLifecycleBehaviour's retry loop for the cross-process case.
    ordered = sorted(agents, key=lambda a: getattr(a, "alias", "") == config.rl_orchestrator.alias)
    for agent in ordered:
        await agent.start()
    await spade.wait_until_finished(agents)
    return agents


def run_distributed(
    config: Config, config_dir: Path, selected_aliases: set[str], num_rollouts: int, debug: bool = False
):
    """Entry point for ``marla run`` in distributed mode.

    Starts only the agents in ``selected_aliases`` and connects to the
    configured (real, externally reachable) XMPP server. Blocks until every
    locally-started agent finishes; raises :class:`DistributedRunError` if
    any of them ended in failure.

    Like :func:`marla.runtime.local.run_local`, this can only be called once
    per process (SPADE's ``Container`` is a process-wide singleton) -- which
    matches real usage, since ``marla run`` is always a fresh process.
    """
    if config.execution.mode != "distributed":
        raise ValueError("run_distributed requires execution.mode == 'distributed'")
    if not config.experiment.run_id:
        raise ValueError("distributed mode requires experiment.run_id")

    result: dict[str, object] = {}

    async def main() -> None:
        try:
            result["agents"] = await _build_and_run(config, config_dir, selected_aliases, num_rollouts, debug=debug)
        except Exception as exc:
            result["error"] = exc
            raise

    spade.run(main(), embedded_xmpp_server=False)

    if "error" in result:
        exc = result["error"]
        raise DistributedRunError(str(exc)) from exc

    agents = result.get("agents", [])
    failed = [agent for agent in agents if getattr(agent, "failure", None) is not None]
    if failed:
        raise DistributedRunError("; ".join(f"{a.alias}: {a.failure}" for a in failed))

    return next((a for a in agents if getattr(a, "training_result", None) is not None), None)
