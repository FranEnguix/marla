import asyncio
import os
import signal

import pytest

from marla.runtime.lifecycle import FailureSignal, LifecycleFailedError, ReadySignal, register_sigint_handler, wait_for_all_ready


def _ready(alias: str) -> ReadySignal:
    return ReadySignal(alias=alias, jid=f"{alias}@localhost", schema_version="1.0", model_version="v1", resolved_device="cpu")


@pytest.mark.asyncio
async def test_empty_required_aliases_returns_immediately():
    events: asyncio.Queue = asyncio.Queue()
    result = await wait_for_all_ready(set(), events)
    assert result == {}


@pytest.mark.asyncio
async def test_waits_for_all_required_aliases():
    events: asyncio.Queue = asyncio.Queue()
    await events.put(_ready("gatekeeper"))
    await events.put(_ready("plan_maker_1"))

    result = await wait_for_all_ready({"gatekeeper", "plan_maker_1"}, events)
    assert set(result) == {"gatekeeper", "plan_maker_1"}
    assert result["gatekeeper"].resolved_device == "cpu"


@pytest.mark.asyncio
async def test_ignores_ready_events_for_unrequired_aliases():
    events: asyncio.Queue = asyncio.Queue()
    await events.put(_ready("someone_else"))
    await events.put(_ready("gatekeeper"))

    result = await wait_for_all_ready({"gatekeeper"}, events)
    assert set(result) == {"gatekeeper"}


@pytest.mark.asyncio
async def test_failure_signal_for_required_alias_raises():
    events: asyncio.Queue = asyncio.Queue()
    await events.put(FailureSignal(alias="gatekeeper", reason="disconnected"))

    with pytest.raises(LifecycleFailedError) as excinfo:
        await wait_for_all_ready({"gatekeeper"}, events)
    assert excinfo.value.alias == "gatekeeper"
    assert "disconnected" in str(excinfo.value)


@pytest.mark.asyncio
async def test_failure_signal_for_unrequired_alias_is_ignored():
    events: asyncio.Queue = asyncio.Queue()
    await events.put(FailureSignal(alias="someone_else", reason="oops"))
    await events.put(_ready("gatekeeper"))

    result = await wait_for_all_ready({"gatekeeper"}, events)
    assert set(result) == {"gatekeeper"}


@pytest.mark.asyncio
async def test_waits_indefinitely_until_event_arrives():
    events: asyncio.Queue = asyncio.Queue()

    async def deliver_late():
        await asyncio.sleep(0.05)
        await events.put(_ready("gatekeeper"))

    task = asyncio.create_task(deliver_late())
    result = await wait_for_all_ready({"gatekeeper"}, events)
    await task
    assert set(result) == {"gatekeeper"}


@pytest.mark.asyncio
async def test_register_sigint_handler_sets_event_on_sigint():
    stop_event = asyncio.Event()
    register_sigint_handler(stop_event)

    os.kill(os.getpid(), signal.SIGINT)
    await asyncio.sleep(0.1)  # give the loop a chance to dispatch the signal

    assert stop_event.is_set()


@pytest.mark.asyncio
async def test_register_sigint_handler_is_one_shot():
    stop_event = asyncio.Event()
    register_sigint_handler(stop_event)

    os.kill(os.getpid(), signal.SIGINT)
    await asyncio.sleep(0.1)
    assert stop_event.is_set()

    # The handler removes itself once fired; nothing left to remove.
    loop = asyncio.get_running_loop()
    assert loop.remove_signal_handler(signal.SIGINT) is False
