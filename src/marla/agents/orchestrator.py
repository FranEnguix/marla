"""The RL Orchestrator SPADE agent (spec section 3.1).

Owns NASimEmu, the recurrent PPO policy, and lifecycle coordination -- there
is no Coordinator component. The baseline variant has no
``required_participants`` at all, so lifecycle collapses to "send nothing,
wait for nothing, train immediately." The assisted variant additionally
consults the Gatekeeper mid-training via a real ``await`` on the same event
loop (see ``learning/trainer.py`` and ``agents/advisory_client.py``).
"""

from __future__ import annotations

import asyncio
import logging

import torch
from spade.agent import Agent
from spade.behaviour import OneShotBehaviour

from marla.agents.advisory_client import AdvisoryResponseListenerBehaviour, send_advisory_request
from marla.agents.lifecycle_behaviours import (
    FailureListenerBehaviour,
    ReadyListenerBehaviour,
    disable_reconnect_on_missed_ping,
    message_type_template,
)
from marla.config.models import Config
from marla.environment.actions import ActionDescriptor
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.recurrent_policy import RecurrentPolicy
from marla.learning.rollout import ConsultationResult
from marla.learning.trainer import TrainingResult, run_training_loop
from marla.messaging.builders import build_message, new_id
from marla.messaging.schemas import (
    MESSAGE_SCHEMA_VERSION,
    AdvisoryActionDescriptor,
    AdvisoryObjective,
    MessageMetadata,
    MessageType,
)
from marla.runtime.lifecycle import FailureSignal, LifecycleFailedError, wait_for_all_ready

logger = logging.getLogger(__name__)


class OrchestratorLifecycleBehaviour(OneShotBehaviour):
    """Runs startup -> training -> shutdown exactly once (spec section 5)."""

    READY_CHECK_RETRY_SECONDS = 5.0
    STOP_EXPERIMENT_GRACE_SECONDS = 1.0

    async def run(self) -> None:
        agent: RLOrchestratorAgent = self.agent
        conversation_id = new_id("lifecycle")

        try:
            if agent.required_participants:
                await self._wait_for_ready_with_retries(agent, conversation_id)
            else:
                await self._send_to_all(MessageType.READY_CHECK, conversation_id, performative="request")
        except LifecycleFailedError as exc:
            logger.error("Lifecycle failed waiting for participants: %s", exc)
            agent.failure = exc
            await self._stop_experiment(agent, conversation_id)
            await agent.stop()
            return

        await self._send_to_all(MessageType.START_EXPERIMENT, conversation_id, performative="inform")

        try:
            # Native async, on this same event loop: the assisted variant's
            # per-step consultation is a real await on the Gatekeeper
            # round-trip, which requires staying on this loop rather than
            # running in a background thread (spec section 4.1).
            agent.training_result = await run_training_loop(
                agent.policy,
                agent.optimizer,
                agent.adapter,
                agent.run_id,
                agent.config.policy.ppo,
                agent.config.policy.recurrent.sequence_length,
                agent.num_rollouts,
                agent.device,
                agent.seed,
                consultation_enabled=agent.consultation_enabled,
                consultation_cost=agent.consultation_cost,
                consult_fn=self._consult if agent.consultation_enabled else None,
                stop_event=agent.stop_event,
            )
        except Exception as exc:  # unrecoverable environment/model error
            logger.exception("Training failed for run %s", agent.run_id)
            agent.failure = exc
            await self._stop_experiment(agent, conversation_id)
            await agent.stop()
            return

        await self._stop_experiment(agent, conversation_id)
        await agent.stop()

    async def _stop_experiment(self, agent: "RLOrchestratorAgent", conversation_id: str) -> None:
        # From here on, participants disconnecting is the run ending as
        # intended, not a failure -- a disconnect reported after this point
        # must not overwrite a training result (or an already-recorded
        # failure) with a spurious "peer disconnected" one.
        agent.expecting_peer_disconnect = True
        await self._send_to_all(MessageType.STOP_EXPERIMENT, conversation_id, performative="inform")
        if agent.required_participants:
            # STOP_EXPERIMENT has no application-level ack; a short grace
            # period gives each participant's own StopExperimentBehaviour a
            # chance to dequeue it and arm its own "expect a disconnect"
            # flag before this agent actually disconnects. Without it, a
            # short training run (few steps, small model) can finish and
            # tear down fast enough that the disconnect reaches a
            # participant before the message announcing it does, and gets
            # mistaken for a real crash.
            await asyncio.sleep(self.STOP_EXPERIMENT_GRACE_SECONDS)

    async def _wait_for_ready_with_retries(self, agent: "RLOrchestratorAgent", conversation_id: str) -> None:
        """Resend READY_CHECK periodically until every participant is ready.

        In distributed mode, a required participant's process may not have
        registered its READY_CHECK responder yet when the first attempt is
        sent -- there is no cross-process message queuing without
        server-side persistence, so a message arriving too early is simply
        lost. Retrying (rather than sending once and waiting forever) closes
        that startup race without weakening "no elapsed timeout": this loop
        never gives up, it just keeps knocking until someone answers.
        """
        required = set(agent.required_participants)
        ready_task = asyncio.ensure_future(wait_for_all_ready(required, agent.events))
        try:
            while not ready_task.done():
                await self._send_to_all(MessageType.READY_CHECK, conversation_id, performative="request")
                done, _pending = await asyncio.wait({ready_task}, timeout=self.READY_CHECK_RETRY_SECONDS)
                if ready_task in done:
                    break
            await ready_task
        finally:
            if not ready_task.done():
                ready_task.cancel()

    async def _consult(
        self,
        legal_actions: list[ActionDescriptor],
        episode_id: int,
        step: int,
        source_observation_id: str,
        observation: dict,
    ) -> ConsultationResult:
        agent: RLOrchestratorAgent = self.agent
        outcome = await send_advisory_request(
            self,
            gatekeeper_jid=agent.gatekeeper_jid,
            gatekeeper_alias=agent.gatekeeper_alias,
            run_id=agent.run_id,
            sender_alias=agent.alias,
            episode_id=episode_id,
            step=step,
            source_observation_id=source_observation_id,
            objective=AdvisoryObjective(
                type=agent.config.objective.type, description=agent.config.objective.description
            ),
            observation=observation,
            legal_actions=[
                AdvisoryActionDescriptor(
                    action_id=a.action_id, type=a.action_type, target=a.target_key, parameters=a.parameters
                )
                for a in legal_actions
            ],
            pending=agent.pending,
        )
        scores = outcome.payload.scores if outcome.payload is not None else None
        return ConsultationResult(
            status=outcome.status,
            scores=scores,
            request_id=outcome.request_id,
            latency_ms=outcome.latency_seconds * 1000,
        )

    async def _send_to_all(self, message_type: MessageType, conversation_id: str, performative: str) -> None:
        agent: RLOrchestratorAgent = self.agent
        for alias, jid in agent.required_participants.items():
            metadata = MessageMetadata(
                performative=performative,
                message_type=message_type,
                schema_version=MESSAGE_SCHEMA_VERSION,
                run_id=agent.run_id,
                conversation_id=conversation_id,
                request_id=new_id("request"),
                sender_alias=agent.alias,
                receiver_alias=alias,
            )
            await self.send(build_message(jid, metadata))


class RLOrchestratorAgent(Agent):
    """The only agent that owns NASimEmu and writes central experiment metrics."""

    def __init__(
        self,
        jid: str,
        password: str,
        *,
        alias: str,
        run_id: str,
        config: Config,
        policy: RecurrentPolicy,
        optimizer: torch.optim.Optimizer,
        adapter: NasimEmuAdapter,
        device: torch.device,
        seed: int,
        num_rollouts: int,
        required_participants: dict[str, str] | None = None,
        gatekeeper_alias: str | None = None,
        gatekeeper_jid: str | None = None,
        stop_event: asyncio.Event | None = None,
    ):
        super().__init__(jid, password)
        self.alias = alias
        self.run_id = run_id
        self.config = config
        self.policy = policy
        self.optimizer = optimizer
        self.adapter = adapter
        self.device = device
        self.seed = seed
        self.num_rollouts = num_rollouts
        self.required_participants = dict(required_participants or {})

        self.consultation_enabled = config.consultation.mode == "learned"
        self.consultation_cost = config.consultation.cost
        self.gatekeeper_alias = gatekeeper_alias
        self.gatekeeper_jid = gatekeeper_jid
        if self.consultation_enabled and (gatekeeper_alias is None or gatekeeper_jid is None):
            raise ValueError("gatekeeper_alias/gatekeeper_jid are required when consultation.mode == 'learned'")

        self.events: asyncio.Queue | None = None
        self.pending: dict[str, asyncio.Future] = {}
        self.failure: Exception | None = None
        self.training_result: TrainingResult | None = None
        self._peers_seen_available: set[str] = set()
        self.expecting_peer_disconnect = False
        # Set by a Ctrl+C handler (see runtime/local.py, runtime/distributed.py);
        # checked between steps so a stop finishes the in-flight step/consultation
        # cleanly rather than tearing down mid-computation, then follows the same
        # STOP_EXPERIMENT/agent.stop() path as a normal, successful completion.
        self.stop_event = stop_event or asyncio.Event()

    async def setup(self) -> None:
        disable_reconnect_on_missed_ping(self)
        self.events = asyncio.Queue()

        self.presence.on_subscribe = lambda peer_jid: self.presence.approve_subscription(peer_jid)
        self.presence.on_available = self._handle_available
        self.presence.on_unavailable = self._handle_unavailable
        self.presence.set_available()
        for jid in self.required_participants.values():
            self.presence.subscribe(jid)

        self.add_behaviour(
            ReadyListenerBehaviour(self.events, self.run_id), message_type_template(MessageType.READY)
        )
        self.add_behaviour(
            FailureListenerBehaviour(self.events, self.run_id),
            message_type_template(MessageType.EXPERIMENT_FAILED),
        )
        if self.consultation_enabled:
            self.add_behaviour(
                AdvisoryResponseListenerBehaviour(self.pending),
                message_type_template(MessageType.ADVISORY_RESPONSE),
            )
        self.add_behaviour(OrchestratorLifecycleBehaviour())

    def _handle_available(self, peer_jid: str, presence_info=None, last_presence=None) -> None:
        self._peers_seen_available.add(str(peer_jid).split("/")[0])

    def _handle_unavailable(self, peer_jid: str, presence_info=None, last_presence=None) -> None:
        # A peer that has not come online yet also reports "unavailable"
        # (e.g. a presence probe answered before startup ordering has
        # settled); only a peer previously seen available counts as having
        # genuinely disconnected.
        bare_jid = str(peer_jid).split("/")[0]
        if bare_jid not in self._peers_seen_available or self.expecting_peer_disconnect:
            return
        for alias, participant_jid in self.required_participants.items():
            if str(participant_jid).split("/")[0] == bare_jid:
                assert self.events is not None
                self.events.put_nowait(FailureSignal(alias=alias, reason="disconnected"))
                return
