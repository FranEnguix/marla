"""Runs one Gatekeeper + mock Plan Maker + stub-orchestrator scenario in-process.

Invoked as a subprocess (see ``tests/test_gatekeeper.py``) because SPADE's
``Container`` is a process-wide singleton and its event loop is closed at
the end of ``spade.run()`` -- a second in-process run would reuse a closed
loop. Prints a single JSON line to stdout with the outcome.

Usage: python -m tests.support.run_gatekeeper_scenario <strategy> <max_schema_revisions>
"""

from __future__ import annotations

import json
import sys

import spade

from marla.agents.gatekeeper import GatekeeperAgent
from tests.support.mock_plan_maker import MockPlanMakerAgent
from tests.support.orchestrator_stub import StubOrchestratorAgent

RUN_ID = "gatekeeper-test-run"


async def run_scenario(strategy: str, max_schema_revisions: int) -> dict:
    orchestrator = StubOrchestratorAgent(
        "orchestrator@localhost",
        "pass-orchestrator",
        alias="rl_orchestrator",
        run_id=RUN_ID,
        gatekeeper_alias="gatekeeper",
        gatekeeper_jid="gatekeeper@localhost",
        plan_maker_alias="plan_maker_1",
        plan_maker_jid="plan-maker@localhost",
    )
    gatekeeper = GatekeeperAgent(
        "gatekeeper@localhost",
        "pass-gatekeeper",
        alias="gatekeeper",
        run_id=RUN_ID,
        orchestrator_alias="rl_orchestrator",
        orchestrator_jid="orchestrator@localhost",
        plan_maker_alias="plan_maker_1",
        plan_maker_jid="plan-maker@localhost",
        max_schema_revisions=max_schema_revisions,
    )
    plan_maker = MockPlanMakerAgent(
        "plan-maker@localhost",
        "pass-plan-maker",
        alias="plan_maker_1",
        run_id=RUN_ID,
        gatekeeper_alias="gatekeeper",
        gatekeeper_jid="gatekeeper@localhost",
        strategy=strategy,
    )

    await gatekeeper.start()
    await plan_maker.start()
    await orchestrator.start()

    await spade.wait_until_finished(orchestrator)

    outcome = orchestrator.outcome
    return {
        "failure": str(orchestrator.failure) if orchestrator.failure else None,
        "status": outcome.status if outcome else None,
        "scores": outcome.payload.scores if (outcome and outcome.payload) else None,
        "retrieved_rule_ids": outcome.payload.retrieved_rule_ids if (outcome and outcome.payload) else None,
    }


def main() -> None:
    strategy = sys.argv[1]
    max_schema_revisions = int(sys.argv[2])
    result: dict = {}

    async def _entry() -> None:
        nonlocal result
        result = await run_scenario(strategy, max_schema_revisions)

    spade.run(_entry(), embedded_xmpp_server=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
