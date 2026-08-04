"""A minimal RL-Orchestrator stand-in for Gatekeeper integration tests.

Exercises real lifecycle coordination (READY_CHECK/READY/START_EXPERIMENT/
STOP_EXPERIMENT) plus one Gatekeeper consultation round-trip, without
needing the full NASimEmu/policy machinery -- that wiring is Milestone 7's
job. This lets Milestone 6 test the Gatekeeper end-to-end now.
"""

from __future__ import annotations

import asyncio

from spade.agent import Agent
from spade.behaviour import OneShotBehaviour

from marla.agents.advisory_client import AdvisoryOutcome, AdvisoryResponseListenerBehaviour, send_advisory_request
from marla.agents.lifecycle_behaviours import (
    FailureListenerBehaviour,
    ReadyListenerBehaviour,
    message_type_template,
)
from marla.messaging.builders import build_message, new_id
from marla.messaging.schemas import (
    MESSAGE_SCHEMA_VERSION,
    AdvisoryActionDescriptor,
    AdvisoryObjective,
    MessageMetadata,
    MessageType,
)
from marla.runtime.lifecycle import LifecycleFailedError, wait_for_all_ready


class StubMainBehaviour(OneShotBehaviour):
    async def run(self) -> None:
        agent: StubOrchestratorAgent = self.agent
        conversation_id = new_id("lifecycle")
        required = {agent.gatekeeper_alias: agent.gatekeeper_jid, agent.plan_maker_alias: agent.plan_maker_jid}

        for alias, jid in required.items():
            metadata = MessageMetadata(
                performative="request", message_type=MessageType.READY_CHECK,
                schema_version=MESSAGE_SCHEMA_VERSION, run_id=agent.run_id,
                conversation_id=conversation_id, request_id=new_id("request"),
                sender_alias=agent.alias, receiver_alias=alias,
            )
            await self.send(build_message(jid, metadata))

        try:
            await wait_for_all_ready(set(required), agent.events)
        except LifecycleFailedError as exc:
            agent.failure = exc
            await agent.stop()
            return

        for alias, jid in required.items():
            metadata = MessageMetadata(
                performative="inform", message_type=MessageType.START_EXPERIMENT,
                schema_version=MESSAGE_SCHEMA_VERSION, run_id=agent.run_id,
                conversation_id=conversation_id, request_id=new_id("request"),
                sender_alias=agent.alias, receiver_alias=alias,
            )
            await self.send(build_message(jid, metadata))

        agent.outcome = await send_advisory_request(
            self,
            gatekeeper_jid=agent.gatekeeper_jid,
            gatekeeper_alias=agent.gatekeeper_alias,
            run_id=agent.run_id,
            sender_alias=agent.alias,
            episode_id=1,
            step=1,
            source_observation_id="observation-1-1",
            objective=AdvisoryObjective(type="capture_target", description="test objective"),
            observation={},
            legal_actions=[
                AdvisoryActionDescriptor(action_id="service-scan:host-1-0", type="service_scan", target="host-1-0"),
                AdvisoryActionDescriptor(action_id="finish", type="finish", target=None),
            ],
            pending=agent.pending,
        )

        for alias, jid in required.items():
            metadata = MessageMetadata(
                performative="inform", message_type=MessageType.STOP_EXPERIMENT,
                schema_version=MESSAGE_SCHEMA_VERSION, run_id=agent.run_id,
                conversation_id=conversation_id, request_id=new_id("request"),
                sender_alias=agent.alias, receiver_alias=alias,
            )
            await self.send(build_message(jid, metadata))

        await agent.stop()


class StubOrchestratorAgent(Agent):
    def __init__(
        self,
        jid: str,
        password: str,
        *,
        alias: str,
        run_id: str,
        gatekeeper_alias: str,
        gatekeeper_jid: str,
        plan_maker_alias: str,
        plan_maker_jid: str,
    ):
        super().__init__(jid, password)
        self.alias = alias
        self.run_id = run_id
        self.gatekeeper_alias = gatekeeper_alias
        self.gatekeeper_jid = gatekeeper_jid
        self.plan_maker_alias = plan_maker_alias
        self.plan_maker_jid = plan_maker_jid

        self.events: asyncio.Queue | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.failure: Exception | None = None
        self.outcome: AdvisoryOutcome | None = None

    async def setup(self) -> None:
        self.events = asyncio.Queue()

        self.presence.on_subscribe = lambda peer_jid: self.presence.approve_subscription(peer_jid)
        self.presence.on_unavailable = self._handle_unavailable
        self.presence.set_available()
        self.presence.subscribe(self.gatekeeper_jid)
        self.presence.subscribe(self.plan_maker_jid)

        self.add_behaviour(
            ReadyListenerBehaviour(self.events, self.run_id), message_type_template(MessageType.READY)
        )
        self.add_behaviour(
            FailureListenerBehaviour(self.events, self.run_id),
            message_type_template(MessageType.EXPERIMENT_FAILED),
        )
        self.add_behaviour(
            AdvisoryResponseListenerBehaviour(self.pending), message_type_template(MessageType.ADVISORY_RESPONSE)
        )
        self.add_behaviour(StubMainBehaviour())

    def _handle_unavailable(self, peer_jid: str, presence_info=None, last_presence=None) -> None:
        from marla.runtime.lifecycle import FailureSignal

        bare_jid = str(peer_jid).split("/")[0]
        for alias, jid in ((self.gatekeeper_alias, self.gatekeeper_jid), (self.plan_maker_alias, self.plan_maker_jid)):
            if str(jid).split("/")[0] == bare_jid:
                assert self.events is not None
                self.events.put_nowait(FailureSignal(alias=alias, reason="disconnected"))
                return
