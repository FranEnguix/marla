# Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA)

MARLA is a research-oriented offensive AI platform for simulated environments. It is designed to operate in NASimEmu scenarios and to study how a centralized reinforcement learning attacker can be augmented by advisory agents with specialized reasoning roles.

Concretely: a reinforcement-learning cyber agent (recurrent PPO over NASimEmu) that can
improve its decisions by selectively consulting an external, frozen,
language-model-based Plan Maker through a schema-validating Gatekeeper.

See `MARLA_complete_implementation_specification.md` for the full design, and
`architecture_design_documents/` for the original design documents.

## Install

```bash
pip install -e ".[dev]"
```

## CLI

```bash
marla --help
marla validate experiment.yaml
marla run experiment.yaml
marla summarize runs/<experiment-name>/<run-id>
marla version
python -m marla --help
```

## Known limitation: embedded XMPP server flakiness in assisted mode

`execution: local` runs use SPADE's built-in embedded XMPP server
(`pyjabber`) so `marla run` works with zero setup -- `xmpp.server` can just
be `localhost`. For the **baseline** variant (no Gatekeeper/Plan Maker) this
is fully reliable since there is no presence-subscription traffic at all.

For the **assisted** variant, `pyjabber` has an observed race condition in
its roster/presence-subscription handling: with 3+ agents connecting and
subscribing to each other's presence, roughly 1-in-4 runs raise an
unhandled `sqlite`/`asyncio` error during startup or (less harmfully)
during shutdown after the run's actual result was already produced. This is
a `pyjabber` robustness issue, not a MARLA correctness issue -- but for long
real research runs where a crash mid-training would be costly, point
`xmpp.server` at a real, separately-deployed XMPP server (e.g. Prosody or
ejabberd) instead of relying on the embedded one. Distributed mode already
requires a real reachable XMPP server, so this only matters for assisted
*local* runs.

The same `pyjabber` race shows up in **distributed** mode too, and more
reliably, when run as a standalone (non-embedded) server: the full
multi-agent handshake needs two concurrent presence subscriptions
(Gatekeeper -> Orchestrator, Plan Maker -> Gatekeeper) where the
single-agent baseline case needs none, and standalone `pyjabber` failed
this consistently in testing (see `tests/test_distributed.py`'s
`test_distributed_multiagent_completes`, marked `xfail` for this reason).
The full multi-agent distributed handshake -- READY_CHECK/READY/
START_EXPERIMENT/STOP_EXPERIMENT and real ADVISORY_REQUEST/ADVISORY_RESPONSE
round trips across separate processes -- was verified working end-to-end
against a properly configured Prosody server; use a real XMPP server for
distributed assisted runs, not standalone `pyjabber`.

## Fixed: keepalive/reconnect cascade during real Plan Maker inference

SPADE enables a XEP-0199 keepalive ping (every 55s) that **reconnects the
client if a ping times out**. The local Plan Maker backend runs
`generate()` synchronously on the shared event loop by design (see
`local_backend.py`'s module docstring: `asyncio.to_thread` reproducibly
hangs for this call in this environment) -- a real model doing real
inference against a real (and, as an episode progresses, growing) prompt
can block that loop long enough to miss a ping. The resulting reconnect
re-registers the agent (SPADE's default `auto_register=True`), and if the
loop is blocked *again* at that moment, registration itself times out,
crashing the run with an unrelated-looking `RegistrationException` --
this was the actual cause of a run appearing to hang or crash partway
through, not a Plan Maker or model bug. Fixed by disabling ping-triggered
reconnection for all agents (`agents/lifecycle_behaviours.py`'s
`disable_reconnect_on_missed_ping`): a genuinely dropped peer is still
caught by the presence-based disconnect detection used in distributed
mode, which doesn't depend on ping timing.
