"""Runs a stub orchestrator + real Gatekeeper + real PlanMakerAgent via SPADE,
for Milestone 9 integration testing. The backend is injectable: a fake
backend for fast, deterministic tests of the message-handling/parsing
plumbing, or a genuinely tiny real HF model for one true end-to-end check.

Invoked as a subprocess (SPADE's Container is a process-wide singleton, see
runtime/local.py).

Usage: python -m tests.support.run_real_plan_maker_scenario <backend_kind> <max_schema_revisions>
"""

from __future__ import annotations

import json
import sys

import spade

from marla.agents.gatekeeper import GatekeeperAgent
from marla.agents.plan_maker import PlanMakerAgent
from marla.knowledge.retriever import load_knowledge_base
from marla.models.plan_maker_backend import BackendResponse
from tests.support.orchestrator_stub import StubOrchestratorAgent

RUN_ID = "plan-maker-test-run"


class FakeBackend:
    """Returns fixed canned text regardless of prompt content."""

    def __init__(self, text: str):
        self._text = text

    async def generate(self, prompt: str, legal_action_ids: list[str]) -> BackendResponse:
        return BackendResponse(raw_text=self._text, latency_ms=1.0)


def _build_backend(kind: str):
    if kind == "fake_valid":
        return FakeBackend(json.dumps({"service-scan:host-1-0": 0.8, "finish": 0.1}))
    if kind == "fake_garbage":
        return FakeBackend("I'm sorry, I cannot comply with this request.")
    if kind == "fake_fenced_json":
        return FakeBackend(
            "Here you go:\n```json\n" + json.dumps({"service-scan:host-1-0": 0.6, "finish": 0.2}) + "\n```"
        )
    if kind == "real_tiny":
        from marla.models.local_backend import LocalTransformersBackend

        return LocalTransformersBackend(model_name="sshleifer/tiny-gpt2", device="cpu", max_new_tokens=16)
    raise ValueError(f"Unknown backend_kind: {kind!r}")


async def run_scenario(backend_kind: str, max_schema_revisions: int) -> dict:
    backend = _build_backend(backend_kind)
    knowledge_base = load_knowledge_base("package://marla/knowledge/nasimemu_rules.yaml")

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
    plan_maker = PlanMakerAgent(
        "plan-maker@localhost",
        "pass-plan-maker",
        alias="plan_maker_1",
        run_id=RUN_ID,
        gatekeeper_alias="gatekeeper",
        gatekeeper_jid="gatekeeper@localhost",
        backend=backend,
        model_version="test-model-v1",
        knowledge_base=knowledge_base,
        resolved_device="cpu",
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
    backend_kind = sys.argv[1]
    max_schema_revisions = int(sys.argv[2])
    result: dict = {}

    async def _entry() -> None:
        nonlocal result
        result = await run_scenario(backend_kind, max_schema_revisions)

    spade.run(_entry(), embedded_xmpp_server=True)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
