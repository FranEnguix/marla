"""Regression tests for OrchestratorLifecycleBehaviour's training/failure race.

No real SPADE agent or XMPP connection needed: these exercise
_run_training_watching_for_failure and _next_failure directly against a
plain asyncio.Queue, the same "SPADE-free pure logic" pattern used by
runtime/lifecycle.py's own tests.
"""

import asyncio
from types import SimpleNamespace

import pytest

from marla.agents.orchestrator import OrchestratorLifecycleBehaviour
from marla.runtime.lifecycle import FailureSignal, LifecycleFailedError, ReadySignal


@pytest.mark.asyncio
async def test_run_training_watching_for_failure_returns_result_on_normal_completion():
    behaviour = OrchestratorLifecycleBehaviour()
    agent = SimpleNamespace(events=asyncio.Queue())

    async def training_call():
        return "training-result"

    result = await behaviour._run_training_watching_for_failure(agent, training_call())
    assert result == "training-result"


@pytest.mark.asyncio
async def test_run_training_watching_for_failure_raises_when_a_participant_disconnects_mid_training():
    # Regression test: before this fix, a Gatekeeper/Plan Maker disconnect
    # detected mid-training was enqueued onto agent.events but never read --
    # wait_for_all_ready only drains that queue during the pre-training
    # handshake -- so a hung consultation (or any in-progress training) just
    # ran forever instead of failing. training_call here simulates exactly
    # that: a consultation await that would never return on its own.
    behaviour = OrchestratorLifecycleBehaviour()
    agent = SimpleNamespace(events=asyncio.Queue())

    async def training_call():
        await asyncio.sleep(3600)

    async def disconnect_shortly_after():
        await asyncio.sleep(0.01)
        await agent.events.put(FailureSignal(alias="gatekeeper", reason="disconnected"))

    disconnect_task = asyncio.ensure_future(disconnect_shortly_after())
    with pytest.raises(LifecycleFailedError) as exc_info:
        await behaviour._run_training_watching_for_failure(agent, training_call())
    assert exc_info.value.alias == "gatekeeper"
    await disconnect_task


@pytest.mark.asyncio
async def test_run_training_watching_for_failure_propagates_training_loop_exceptions():
    behaviour = OrchestratorLifecycleBehaviour()
    agent = SimpleNamespace(events=asyncio.Queue())

    async def training_call():
        raise RuntimeError("environment blew up")

    with pytest.raises(RuntimeError, match="environment blew up"):
        await behaviour._run_training_watching_for_failure(agent, training_call())


@pytest.mark.asyncio
async def test_next_failure_skips_ready_signal_duplicates_and_returns_the_failure():
    queue: asyncio.Queue = asyncio.Queue()
    await queue.put(
        ReadySignal(alias="gatekeeper", jid="gk@localhost", schema_version="1.0", model_version="v1", resolved_device="cpu")
    )
    await queue.put(FailureSignal(alias="gatekeeper", reason="disconnected"))

    result = await OrchestratorLifecycleBehaviour._next_failure(queue)
    assert result == FailureSignal(alias="gatekeeper", reason="disconnected")
