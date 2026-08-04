import asyncio
import copy
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
    data["policy"]["ppo"]["rollout_steps"] = 12
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    # examples/baseline.yaml enables periodic evaluation by default; tests
    # that want it explicitly override these two fields themselves, so the
    # shared tiny config stays eval-off regardless of that file's defaults.
    data["metrics"]["eval_episodes"] = 0
    return parse_config(data)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_baseline_smoke_run_cpu_small_scenario():
    config = _tiny_config()
    result = await run_baseline_training(config, SMALL_SCENARIO, num_rollouts=2, seed=1)

    assert result.environment_steps == 24
    assert len(result.episode_summaries) >= 1
    assert len(result.update_metrics) > 0
    for m in result.update_metrics:
        # mean_beta/mean_query_probability are intentionally None in baseline
        # mode (no consultations ever happen), and checkpoint_id is None
        # until checkpointing is wired into the run loop -- everything else
        # must be a finite number.
        for key, value in m.items():
            if value is None:
                assert key in {"mean_beta", "mean_query_probability", "checkpoint_id"}
                continue
            assert torch.isfinite(torch.tensor(value))


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
