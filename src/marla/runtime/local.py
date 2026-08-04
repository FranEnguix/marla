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

from marla.config.models import Config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.trainer import build_policy_and_optimizer

logger = logging.getLogger(__name__)


class LocalRunError(Exception):
    """Raised when a local-mode run ends in failure (mirrors EXPERIMENT_FAILED)."""


def resolve_scenario_path(config: Config, config_dir: Path) -> str:
    scenario = config.environment.scenario
    if scenario.endswith(".yaml"):
        path = Path(scenario)
        if not path.is_absolute():
            path = (config_dir / path).resolve()
        return str(path)
    return scenario


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


async def _build_and_run(config: Config, config_dir: Path, num_rollouts: int, debug: bool = False) -> "object":
    from marla.agents.gatekeeper import GatekeeperAgent
    from marla.agents.orchestrator import RLOrchestratorAgent
    from marla.agents.plan_maker import PlanMakerAgent
    from marla.knowledge.retriever import load_knowledge_base
    from marla.models.backend_factory import build_backend
    from marla.runtime.device import resolve_device
    from marla.runtime.lifecycle import register_sigint_handler

    resolved_device = resolve_device(config.device)
    scenario_path = resolve_scenario_path(config, config_dir)
    run_id = config.experiment.run_id or config.experiment.name
    stop_event = asyncio.Event()
    register_sigint_handler(stop_event)

    policy, optimizer = build_policy_and_optimizer(
        config, resolved_device.torch_device, consultation_enabled=config.consultation.mode == "learned"
    )
    adapter = NasimEmuAdapter(
        scenario=scenario_path,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

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
        seed=config.experiment.seed,
        num_rollouts=num_rollouts,
        required_participants=required_participants,
        gatekeeper_alias=config.gatekeeper.alias if config.gatekeeper else None,
        gatekeeper_jid=config.gatekeeper.jid if config.gatekeeper else None,
        stop_event=stop_event,
    )

    for agent in support_agents:
        await agent.start()
    await orchestrator.start()
    await spade.wait_until_finished(orchestrator)
    return orchestrator


def run_local(
    config: Config, config_dir: Path, num_rollouts: int, embedded_xmpp_server: bool = True, debug: bool = False
):
    """Run a local-mode experiment; blocks until the RL Orchestrator finishes.

    Raises :class:`LocalRunError` if the run ended in failure. Returns the
    stopped :class:`RLOrchestratorAgent` (holding ``training_result``) on
    success.

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
            result["orchestrator"] = await _build_and_run(config, config_dir, num_rollouts, debug=debug)
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
