"""Runs one assisted-variant training scenario (real RL Orchestrator + real
Gatekeeper + deterministic mock Plan Maker) via SPADE, for Milestone 7
integration testing. Invoked as a subprocess (SPADE's Container is a
process-wide singleton, see runtime/local.py).

Usage: python -m tests.support.run_assisted_training_scenario <config_path> <scenario_path> <num_rollouts> <strategy>
"""

from __future__ import annotations

import json
import math
import sys

import spade

from marla.agents.gatekeeper import GatekeeperAgent
from marla.agents.orchestrator import RLOrchestratorAgent
from marla.config.loader import load_config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.trainer import build_policy_and_optimizer
from marla.runtime.device import resolve_device
from tests.support.mock_plan_maker import MockPlanMakerAgent


async def run_scenario(config_path: str, scenario_path: str, num_rollouts: int, strategy: str) -> dict:
    config = load_config(config_path)
    resolved_device = resolve_device(config.device)
    run_id = config.experiment.run_id or config.experiment.name

    assert config.gatekeeper is not None
    assert len(config.agents) == 1
    plan_maker_cfg = config.agents[0]

    policy, optimizer = build_policy_and_optimizer(config, resolved_device.torch_device, consultation_enabled=True)
    adapter = NasimEmuAdapter(
        scenario=scenario_path,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    orchestrator = RLOrchestratorAgent(
        config.rl_orchestrator.jid,
        "pass-orchestrator",
        alias=config.rl_orchestrator.alias,
        run_id=run_id,
        config=config,
        policy=policy,
        optimizer=optimizer,
        adapter=adapter,
        device=resolved_device.torch_device,
        seed=config.experiment.seed,
        num_rollouts=num_rollouts,
        required_participants={
            config.gatekeeper.alias: config.gatekeeper.jid,
            plan_maker_cfg.alias: plan_maker_cfg.jid,
        },
        gatekeeper_alias=config.gatekeeper.alias,
        gatekeeper_jid=config.gatekeeper.jid,
    )
    gatekeeper = GatekeeperAgent(
        config.gatekeeper.jid,
        "pass-gatekeeper",
        alias=config.gatekeeper.alias,
        run_id=run_id,
        orchestrator_alias=config.rl_orchestrator.alias,
        orchestrator_jid=config.rl_orchestrator.jid,
        plan_maker_alias=plan_maker_cfg.alias,
        plan_maker_jid=plan_maker_cfg.jid,
        max_schema_revisions=config.consultation.max_schema_revisions,
    )
    plan_maker = MockPlanMakerAgent(
        plan_maker_cfg.jid,
        "pass-plan-maker",
        alias=plan_maker_cfg.alias,
        run_id=run_id,
        gatekeeper_alias=config.gatekeeper.alias,
        gatekeeper_jid=config.gatekeeper.jid,
        strategy=strategy,
    )

    await gatekeeper.start()
    await plan_maker.start()
    await orchestrator.start()

    await spade.wait_until_finished(orchestrator)

    result = orchestrator.training_result

    def _finite(value) -> bool:
        # mean_beta/mean_query_probability can be legitimately None (no
        # queried/accepted records in that rollout), and checkpoint_id is
        # always None until checkpointing is wired into the run loop.
        if value is None:
            return True
        return not (math.isnan(value) or math.isinf(value))

    if result is None:
        return {"failure": str(orchestrator.failure) if orchestrator.failure else "no training_result"}

    queried_records = [r for r in result.all_records if r.sampled_query]
    accepted_records = [r for r in queried_records if r.plan_maker_validation_status == "accepted"]

    return {
        "failure": str(orchestrator.failure) if orchestrator.failure else None,
        "environment_steps": result.environment_steps,
        "num_episodes": len(result.episode_summaries),
        "num_updates": len(result.update_metrics),
        "any_queried": len(queried_records) > 0,
        "any_accepted": len(accepted_records) > 0,
        "accepted_betas": [r.beta for r in accepted_records],
        "accepted_alphas": [r.alpha for r in accepted_records],
        "total_consultation_cost": sum(r.consultation_cost for r in result.all_records),
        "update_metrics_all_finite": all(_finite(v) for m in result.update_metrics for v in m.values()),
    }


def main() -> None:
    config_path, scenario_path, num_rollouts, strategy = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4]
    result: dict = {}

    async def _entry() -> None:
        nonlocal result
        result = await run_scenario(config_path, scenario_path, num_rollouts, strategy)

    spade.run(_entry(), embedded_xmpp_server=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
