"""Startup/shutdown/failure coordination logic (spec section 5), SPADE-free.

There is no Coordinator component; the RL Orchestrator coordinates startup
and shutdown directly. This module holds the *pure* decision logic --
"given these READY/failure events, are we ready to start, and if not, why
did we fail" -- independent of SPADE/XMPP wiring, so it can be unit tested
with a plain ``asyncio.Queue`` instead of a real agent connection. The SPADE
glue that turns real messages and presence callbacks into the events this
module consumes lives in :mod:`marla.agents.lifecycle_behaviours`.

There is no elapsed-time response timeout (spec section 5/10): this module
waits indefinitely on the event queue. Only an explicit failure/disconnect
event ends the wait early.
"""

from __future__ import annotations

import asyncio
import signal
from dataclasses import dataclass


@dataclass(frozen=True)
class ReadySignal:
    """A participant reported READY."""

    alias: str
    jid: str
    schema_version: str
    model_version: str
    resolved_device: str


@dataclass(frozen=True)
class FailureSignal:
    """A participant failed, disconnected, or reported EXPERIMENT_FAILED."""

    alias: str
    reason: str


LifecycleEvent = ReadySignal | FailureSignal


class LifecycleFailedError(Exception):
    """Raised when a required participant fails or disconnects during startup."""

    def __init__(self, alias: str, reason: str):
        self.alias = alias
        self.reason = reason
        super().__init__(f"{alias}: {reason}")


def register_sigint_handler(stop_event: asyncio.Event) -> None:
    """Arm a one-shot Ctrl+C handler that requests a graceful stop.

    A synchronous, blocking Plan Maker consultation can't safely be
    interrupted mid-computation (there is no clean cancellation point inside
    a torch forward pass), so the first SIGINT just sets ``stop_event``
    (checked between environment steps -- see
    ``learning.rollout.RolloutCollector.collect``) and removes itself. A
    stopped run still finishes the in-flight step and then follows the same
    STOP_EXPERIMENT/agent.stop() shutdown path as a normal completion,
    rather than tearing down mid-message. A second Ctrl+C falls through to
    Python's default KeyboardInterrupt handling, for an immediate,
    unconditional exit if the graceful stop doesn't return promptly enough.
    """
    loop = asyncio.get_running_loop()

    def _handle_sigint() -> None:
        print(
            "\nCtrl+C received: stopping after the current step (press Ctrl+C again to force-quit)...",
            flush=True,
        )
        stop_event.set()
        loop.remove_signal_handler(signal.SIGINT)

    loop.add_signal_handler(signal.SIGINT, _handle_sigint)


async def wait_for_all_ready(
    required_aliases: set[str], events: "asyncio.Queue[LifecycleEvent]"
) -> dict[str, ReadySignal]:
    """Block until every required alias has signaled READY.

    Raises :class:`LifecycleFailedError` on the first failure/disconnect
    event concerning a required alias. Events for aliases outside
    ``required_aliases`` are ignored (defensive; should not occur in
    practice). Returns immediately with an empty dict if ``required_aliases``
    is empty (the baseline variant has no participants to wait for).
    """
    ready: dict[str, ReadySignal] = {}
    if not required_aliases:
        return ready

    while set(ready) != required_aliases:
        event = await events.get()
        if isinstance(event, FailureSignal):
            if event.alias in required_aliases:
                raise LifecycleFailedError(event.alias, event.reason)
        elif isinstance(event, ReadySignal):
            if event.alias in required_aliases:
                ready[event.alias] = event
        else:
            raise TypeError(f"Unknown lifecycle event: {event!r}")

    return ready
