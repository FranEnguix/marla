# Multi-Agent Reinforcement Learning Architecture for offensive AI (MARLA)

MARLA is a research-oriented offensive AI platform for simulated environments. It is designed to operate in NASimEmu scenarios and to study how a centralized reinforcement learning attacker can be augmented by advisory agents with specialized reasoning roles.

Concretely: a reinforcement-learning cyber agent (recurrent PPO over NASimEmu) that can
improve its decisions by selectively consulting an external, frozen,
language-model-based Plan Maker through a schema-validating Gatekeeper.

See `docs/` for the full documentation (installation, configuration
reference, architecture, CLI, and metrics/plots) -- build it locally with
`pip install -e ".[docs]"` and `sphinx-build -b html docs docs/_build/html`.

## Install

```bash
pip install -e ".[dev]"

# NASimEmu is a separate editable install (not a regular pip dependency
# -- see docs/installation.rst for why); do this too, or `marla run`
# has nothing to run against:
pip install -e ./gym-0.21.0   # vendored, buildable gym==0.21.0
pip install -e ./NASimEmu
```

Optional extras: `local-lm` (a real local Plan Maker model), `gpu` (NVML
GPU utilization/memory telemetry), `carbon` (CodeCarbon energy/CO2eq
tracking), `optuna` (persistent hyperparameter studies), `docs` (build the
Sphinx docs locally). Combine as needed, e.g.
`pip install -e ".[dev,gpu,carbon,optuna]"`.

## CLI

```bash
marla --help
marla init [DIRECTORY]                          # generate starter configs
marla validate experiment.yaml
marla run experiment.yaml [--resume CHECKPOINT]
marla summarize runs/<experiment-name>/<run-id>
marla scenario check|repair SCENARIO
marla optimize study.yaml                        # requires the `optuna` extra
marla study status|summarize study.yaml
marla version
python -m marla --help
```

See `docs/cli.rst` for every command's full reference.

## Training, evaluation, and scenarios

An experiment is one YAML file (`examples/baseline.yaml` for PPO-only,
`examples/assisted.yaml` for the Plan-Maker-advised variant); `marla run`
validates it, runs a scenario solvability preflight, then trains. Scenarios
are referenced by a `marla://<name>.yaml` URI (packaged, pre-validated,
resolves the same regardless of install method or working directory), a
filesystem path, or a NASimEmu-generated benchmark name -- see
`docs/configuration.rst`'s `environment` section and
`docs/scenario_solvability.rst`.

`policy.ppo.num_envs` collects multiple independent environment streams
per PPO update (multi-environment rollout collection); `policy.ppo.optimizer.scheduler`
configures a PyTorch-native learning-rate schedule (`constant`/`linear`/
`cosine`/`step`/`exponential`), stepped once per completed PPO update and
saved/restored exactly across `marla run --resume`. Both are documented in
full in `docs/configuration.rst`'s `policy` section.

`metrics.eval_episodes`/`eval_every_rollouts` run periodic deterministic
evaluation episodes between training rollouts, at zero cost when disabled
(`eval_episodes: 0`, the default). `marla summarize RUN_DIRECTORY` prints
the run's aggregate statistics and writes every plot to `RUN_DIRECTORY/plots/`
-- see `docs/metrics.rst` for what each plot/CSV column means.

## Resource and carbon telemetry

`metrics.resource_monitoring` (on by default) records CPU/RAM/GPU usage
throughout every run (`resources.csv`, `resource_summary.json`). The
optional `carbon` config block adds per-run estimated energy (kWh) and
CO2-equivalent emissions tracking via [CodeCarbon](https://github.com/mlco2/codecarbon)
(local CSV output only -- nothing is ever uploaded to CodeCarbon's hosted
API); `marla summarize` prints an energy/CO2eq line when it was enabled
for that run. See `docs/configuration.rst`'s `metrics`/`carbon` sections.

## Hyperparameter studies

`marla optimize study.yaml` runs (or resumes) a persistent, SQLite-backed
Optuna hyperparameter study -- safe to interrupt and rerun without losing
progress or re-running completed trials. `marla study status`/`marla study
summarize` inspect a study without running anything. Requires the
`optuna` extra. See `docs/cli.rst`'s Optuna commands section.

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
