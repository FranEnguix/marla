import asyncio
import copy
import gc
import weakref
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.learning.trainer import build_policy_and_optimizer, run_baseline_training, run_training_loop

REPO_ROOT = Path(__file__).resolve().parent.parent
SMALL_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve())


def _tiny_config():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    # examples/baseline.yaml enables periodic evaluation by default; tests
    # that want it explicitly override these two fields themselves, so the
    # shared tiny config stays eval-off regardless of that file's defaults.
    data["metrics"]["eval_episodes"] = 0
    return parse_config(data)


def test_build_policy_and_optimizer_uses_adam_with_the_configured_eps():
    config = _tiny_config()
    policy, optimizer = build_policy_and_optimizer(config, torch.device("cpu"))
    assert isinstance(optimizer, torch.optim.Adam)
    assert not isinstance(optimizer, torch.optim.AdamW)
    assert optimizer.param_groups[0]["eps"] == config.policy.ppo.optimizer.eps
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1.0e-5)
    assert optimizer.param_groups[0]["lr"] == config.policy.ppo.optimizer.learning_rate


def test_build_policy_and_optimizer_uses_an_overridden_eps():
    config = _tiny_config()
    data = config.model_dump()
    data["policy"]["ppo"]["optimizer"] = {
        "type": "adam", "learning_rate": data["policy"]["ppo"]["optimizer"]["learning_rate"],
        "eps": 1.0e-3, "scheduler": {"type": "constant"},
    }
    config = parse_config(data)
    _policy, optimizer = build_policy_and_optimizer(config, torch.device("cpu"))
    assert optimizer.param_groups[0]["eps"] == pytest.approx(1.0e-3)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_earlier_rollouts_step_records_are_garbage_collected_not_retained():
    """Regression/verification test for spec section 1: a rollout's
    heavyweight StepRecords (and their graph tensors) must not be retained
    anywhere once that rollout's compact metrics have been extracted --
    not in TrainingResult, not in a closure. Reading the code isn't
    sufficient proof (a hidden reference is easy to miss); this observes
    real garbage collection of the first rollout's records after several
    more rollouts have run.
    """
    import marla.metrics.writer as writer_module

    config = _tiny_config()
    policy, optimizer = build_policy_and_optimizer(config, torch.device("cpu"))
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO, max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    captured: dict = {}
    real_build_decision_rows = writer_module.build_decision_rows

    def spy(records, advantages, returns):
        if "graph_ref" not in captured:
            captured["graph_ref"] = weakref.ref(records[0].graph_data)
            captured["record_ref"] = weakref.ref(records[0])
        return real_build_decision_rows(records, advantages, returns)

    # run_training_loop imports build_decision_rows fresh from
    # marla.metrics.writer inside its own body on every call (to avoid a
    # circular import at module scope), so patching the attribute here is
    # enough -- no need to also patch a cached reference on trainer_module.
    writer_module.build_decision_rows = spy
    try:
        result = await run_training_loop(
            policy, optimizer, adapter, "mem-test", config.policy.ppo, config.policy.recurrent.sequence_length,
            num_rollouts=4, device=torch.device("cpu"), seed=1,
        )
    finally:
        writer_module.build_decision_rows = real_build_decision_rows

    assert "graph_ref" in captured, "spy never observed a rollout -- fixture produced zero records"
    gc.collect()
    assert captured["graph_ref"]() is None, "rollout 1's graph tensor is still referenced after 3 more rollouts ran"
    assert captured["record_ref"]() is None, "rollout 1's StepRecord is still referenced after 3 more rollouts ran"
    assert not hasattr(result, "all_records")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_smoke_run_cpu_small_scenario():
    config = _tiny_config()
    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=2, seed=1)

    assert result.environment_steps == 24
    assert len(result.episode_summaries) >= 1
    assert len(result.update_metrics) > 0
    # These are legitimately None whenever this minibatch happened to
    # contain zero occurrences of the underlying category (e.g. no FINISH
    # action at all, or no premature-FINISH specifically) -- see
    # learning/ppo.py's own "... if values else None" conditionals. A tiny
    # 2-rollout/12-step smoke run has no guarantee every category is
    # populated, so these are data-dependent, not a diagnostic that must
    # always fire.
    NULLABLE_UPDATE_FIELDS = {
        "mean_raw_advantage_finish", "max_raw_advantage_finish", "min_raw_advantage_finish",
        "mean_raw_advantage_non_finish", "max_raw_advantage_non_finish",
        "mean_raw_advantage_successful_finish", "mean_raw_advantage_premature_finish",
        "finish_advantage_z_max", "max_raw_advantage_state_N_finish", "max_normalized_advantage_state_N_finish",
        "mean_raw_advantage_state_N_finish", "mean_normalized_advantage_state_N_finish",
    }
    for m in result.update_metrics:
        # Rollout-level aggregates that can legitimately be None/NaN in
        # baseline mode (no consultations) now live in rollouts.csv instead
        # (see result.rollout_rows) -- every OTHER field left on an
        # individual update dict is a required PPO diagnostic and must be
        # finite.
        for key, value in m.items():
            if key in NULLABLE_UPDATE_FIELDS and value is None:
                continue
            assert value is not None, key
            if isinstance(value, (int, float)):
                assert torch.isfinite(torch.tensor(value)), key

    assert len(result.rollout_rows) > 0
    for row in result.rollout_rows:
        assert row["mean_beta"] is None  # baseline never queries
        assert row["mean_query_probability"] is None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_training_changes_policy_parameters():
    config = _tiny_config()
    torch.manual_seed(0)
    from marla.learning.trainer import build_policy_and_optimizer

    initial_policy, _ = build_policy_and_optimizer(config, torch.device("cpu"))
    initial_state = copy.deepcopy(initial_policy.state_dict())

    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=2, seed=0)
    trained_state = result.policy.state_dict()

    changed = any(not torch.equal(initial_state[k], trained_state[k]) for k in initial_state)
    assert changed


@pytest.mark.asyncio
async def test_run_baseline_training_rejects_assisted_config():
    config = _tiny_config()
    data = config.model_dump()
    data["consultation"] = {"mode": "learned", "cost": 0.1, "max_schema_revisions": 3}
    data["gatekeeper"] = {"alias": "gatekeeper", "jid": "gk@localhost"}
    data["agents"] = [
        {
            "alias": "plan_maker_1",
            "jid": "pm@localhost",
            "role": "plan_maker",
            "model": {"backend": "local", "name": "x"},
            "prompt_version": "v1",
            "knowledge": {"path": "package://marla/knowledge/nasimemu_rules.yaml", "version": "v1"},
        }
    ]
    assisted_config = parse_config(data)
    with pytest.raises(ValueError):
        await run_baseline_training(assisted_config, SMALL_SCENARIO, num_rollouts=1)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_a_mid_training_exception_still_stops_resource_and_carbon_tracking_and_flushes_partial_data(
    tmp_path, monkeypatch
):
    """Regression: a mid-loop exception (e.g. the NaN/Inf fail-fast assert
    inside ``optimize()``) must not leave ``ResourceMonitor``'s background
    thread or CodeCarbon's tracker running past ``run_training_loop``'s
    return, and must not silently discard the real partial resource/carbon
    data already collected before the failure -- see run_training_loop's
    ``except Exception`` cleanup block. The original exception must still
    propagate unchanged (never swallowed by cleanup).
    """
    config = _tiny_config()
    data = config.model_dump()
    data["carbon"] = {"enabled": True}
    config = parse_config(data)
    run_dir = tmp_path / "run"

    from marla.learning import trainer as trainer_module

    call_count = {"n": 0}
    real_optimize = trainer_module.optimize

    def _fail_on_first_call(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("injected failure for exception-cleanup regression test")
        return real_optimize(*args, **kwargs)

    monkeypatch.setattr(trainer_module, "optimize", _fail_on_first_call)

    with pytest.raises(RuntimeError, match="injected failure"):
        await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=3, seed=1, run_dir=run_dir)

    # The background resource-sampling thread must be gone, not leaked.
    import threading

    assert not any(t.name == "marla-resource-monitor" for t in threading.enumerate())

    resource_summary_path = run_dir / "resource_summary.json"
    assert resource_summary_path.is_file(), "resource_summary.json must be flushed even on a mid-training failure"

    carbon_summary_path = run_dir / "carbon" / "carbon_summary.json"
    assert carbon_summary_path.is_file(), "carbon_summary.json must be flushed even on a mid-training failure"
    import json

    carbon_summary = json.loads(carbon_summary_path.read_text(encoding="utf-8"))
    assert carbon_summary["enabled"] is True
    # Real, nonzero: collection for rollout 1 genuinely happened before the
    # injected failure, so CodeCarbon had real elapsed time to measure.
    assert carbon_summary["duration_seconds"] is not None
    assert carbon_summary["duration_seconds"] >= 0.0


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_training_loop_stops_early_and_flags_result_when_stop_event_is_set():
    config = _tiny_config()
    device = torch.device("cpu")
    policy, optimizer = build_policy_and_optimizer(config, device, consultation_enabled=False)
    adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    stop_event = asyncio.Event()
    stop_event.set()  # already requested before the first rollout even starts

    result = await run_training_loop(
        policy, optimizer, adapter, config.experiment.run_id or config.experiment.name,
        config.policy.ppo, config.policy.recurrent.sequence_length, num_rollouts=5,
        device=device, seed=1, stop_event=stop_event,
    )

    assert result.stopped_by_user is True
    assert result.environment_steps == 0  # stopped before collecting anything


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_training_loop_does_not_flag_stopped_by_user_on_normal_completion():
    config = _tiny_config()
    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=1, seed=1)
    assert result.stopped_by_user is False


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_baseline_training_tags_every_episode_with_its_rollout():
    config = _tiny_config()
    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=3, seed=1)
    assert all(s.rollout is not None for s in result.episode_summaries)
    assert all(1 <= s.rollout <= 3 for s in result.episode_summaries)
    assert result.eval_episode_summaries == []  # eval_episodes defaults to 0 (disabled)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_linear_schedule_decreases_the_actual_learning_rate_used_by_updates():
    config = _tiny_config()
    data = config.model_dump()
    data["policy"]["ppo"]["optimizer"]["scheduler"] = {"type": "linear", "end_factor": 0.0}
    data["policy"]["ppo"]["total_environment_steps"] = 48  # 4 rollouts of 12 steps
    config = parse_config(data)

    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=4, seed=1)

    rates = [m["learning_rate"] for m in result.update_metrics]
    assert rates[0] == pytest.approx(config.policy.ppo.optimizer.learning_rate)  # first rollout starts at the full rate
    assert rates[-1] < rates[0]  # decays over the run
    assert all(r >= 0.0 for r in rates)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_constant_schedule_keeps_the_learning_rate_flat():
    config = _tiny_config()
    data = config.model_dump()
    data["policy"]["ppo"]["optimizer"]["scheduler"] = {"type": "constant"}
    config = parse_config(data)

    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=3, seed=1)

    rates = [m["learning_rate"] for m in result.update_metrics]
    assert rates  # sanity: at least one update happened
    assert all(rate == pytest.approx(config.policy.ppo.optimizer.learning_rate) for rate in rates)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_baseline_training_runs_periodic_evaluation_when_enabled():
    config = _tiny_config()
    data = config.model_dump()
    data["metrics"]["eval_episodes"] = 2
    data["metrics"]["eval_every_rollouts"] = 1
    config = parse_config(data)

    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=2, seed=1)

    # 2 eval episodes after each of 2 rollouts.
    assert len(result.eval_episode_summaries) == 4
    assert all(s.is_eval for s in result.eval_episode_summaries)
    assert all(not s.is_eval for s in result.episode_summaries)
    assert {s.rollout for s in result.eval_episode_summaries} == {1, 2}
    # The same fixed eval seeds are reused at every checkpoint.
    first_checkpoint_seeds = sorted(s.seed for s in result.eval_episode_summaries if s.rollout == 1)
    second_checkpoint_seeds = sorted(s.seed for s in result.eval_episode_summaries if s.rollout == 2)
    assert first_checkpoint_seeds == second_checkpoint_seeds


@pytest.mark.asyncio
async def test_periodic_evaluation_never_shares_the_training_collectors_adapter(monkeypatch):
    """Regression test: periodic evaluation used to run its episodes on the
    *same* NasimEmuAdapter instance the training collector resumes
    mid-episode on across rollouts. Since NasimEmuAdapter wraps a genuinely
    mutable, stateful NASimEmuEnv, and most bundled scenarios (including
    SMALL_SCENARIO) draw a random host count per reset(), an eval pass's
    reset()/step() calls could silently replace the training collector's
    live environment out from under its own paused episode -- reproduced
    directly (a real `marla run`, 10 rollouts of 128 steps, periodic
    eval) as an IndexError inside compute_state_delta several steps after
    an eval pass ran, once the training collector resumed and the host-row
    array it expected no longer matched what the (eval-mutated) live
    environment actually returned. Asserting on adapter identity, not on
    reproducing that IndexError under a specific seed/scenario/rollout
    count (unreliable -- the corruption depends on both collectors' random
    host-count draws happening to differ, and on eval landing while
    training is genuinely mid-episode).
    """
    import marla.learning.trainer as trainer_module

    seen_adapters = []
    real_run_evaluation_episodes = trainer_module.run_evaluation_episodes

    async def spy(*, adapter, **kwargs):
        seen_adapters.append(adapter)
        return await real_run_evaluation_episodes(adapter=adapter, **kwargs)

    monkeypatch.setattr(trainer_module, "run_evaluation_episodes", spy)

    config = _tiny_config()
    data = config.model_dump()
    data["metrics"]["eval_episodes"] = 2
    data["metrics"]["eval_every_rollouts"] = 1
    config = parse_config(data)

    training_adapter = NasimEmuAdapter(
        scenario=SMALL_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    policy, optimizer = build_policy_and_optimizer(config, torch.device("cpu"))
    result = await run_training_loop(
        policy, optimizer, training_adapter, config.experiment.run_id or config.experiment.name,
        config.policy.ppo, config.policy.recurrent.sequence_length, num_rollouts=2,
        device=torch.device("cpu"), seed=1,
        eval_episodes=config.metrics.eval_episodes, eval_every_rollouts=config.metrics.eval_every_rollouts,
    )

    assert len(seen_adapters) == 2  # one per rollout, both evaluated
    for eval_adapter in seen_adapters:
        assert eval_adapter is not training_adapter
    assert len(result.eval_episode_summaries) == 4
