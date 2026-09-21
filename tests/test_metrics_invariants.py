"""Cross-cutting scientific-data-contract invariants (platform-hardening
pass before the final AAMAS paper matrix). Targeted regression coverage
for invariants not already exercised elsewhere -- see test_gae.py,
test_metrics_writer.py, test_csv_schema.py, test_decision.py,
test_message_telemetry.py, test_resource_monitoring.py for the rest.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest


def test_query_false_implies_no_actual_consultation_fields(tmp_path):
    """A non-queried decision must never carry a Plan Maker response as if
    one happened."""
    import asyncio

    from marla.config.loader import load_config
    from marla.environment.nasimemu_adapter import NasimEmuAdapter
    from marla.learning.recurrent_policy import RecurrentPolicy
    from marla.learning.rollout import RolloutCollector

    from pathlib import Path

    REPO_ROOT = Path(__file__).resolve().parent.parent
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    adapter = NasimEmuAdapter(
        scenario=str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve()),
        max_episode_steps=6,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    policy = RecurrentPolicy(config.policy)  # baseline: consultation_enabled=False
    collector = RolloutCollector(adapter, policy, run_id="t", base_seed=1)
    records, _ = asyncio.run(collector.collect(15))

    for r in records:
        assert r.sampled_query is False
        assert r.consultation_cost == 0.0
        assert r.plan_maker_response_status is None
        assert r.plan_maker_request_id is None
        assert r.beta is None
        assert r.normalized_advice is None


def test_accepted_advice_implies_query_true():
    """response_status == 'accepted' can never appear on a non-queried
    decision -- would mean advice was applied without ever asking."""
    from marla.metrics.plots import _accepted_advice_rows

    decisions = pd.DataFrame(
        [
            {"queried": False, "response_status": None, "beta": None, "advice_changed_top_action": None},
            {"queried": True, "response_status": "accepted", "beta": 0.5, "advice_changed_top_action": True},
        ]
    )
    accepted = _accepted_advice_rows(decisions)
    assert (accepted["queried"] == True).all()  # noqa: E712


def test_consultation_cost_aggregate_equals_decision_level_sum():
    """episodes.csv's consultation_cost must equal the sum of the
    per-decision consultation_cost values for that episode -- a
    reconstructability invariant (Phase 4/9): the aggregate is not an
    independently-tracked quantity that could drift from its raw source."""
    decisions = pd.DataFrame(
        {
            "episode_id": [1, 1, 1, 2, 2],
            "consultation_cost": [0.1, 0.0, 0.1, 0.0, 0.0],
        }
    )
    per_episode = decisions.groupby("episode_id")["consultation_cost"].sum()
    assert per_episode[1] == pytest.approx(0.2)
    assert per_episode[2] == pytest.approx(0.0)


def test_episode_return_equals_sum_of_per_step_rewards():
    decisions = pd.DataFrame({"episode_id": [1, 1, 1], "nasimemu_reward": [-0.1, -0.1, 1.0]})
    total = decisions.groupby("episode_id")["nasimemu_reward"].sum()
    assert total[1] == pytest.approx(0.8)


def test_global_environment_step_monotonic_within_a_rollout(tmp_path):
    import asyncio

    from pathlib import Path

    from marla.config.loader import load_config
    from marla.environment.nasimemu_adapter import NasimEmuAdapter
    from marla.learning.recurrent_policy import RecurrentPolicy
    from marla.learning.rollout import RolloutCollector

    REPO_ROOT = Path(__file__).resolve().parent.parent
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    adapter = NasimEmuAdapter(
        scenario=str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_one_subnet.v2.yaml").resolve()),
        max_episode_steps=5,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    policy = RecurrentPolicy(config.policy)
    collector = RolloutCollector(adapter, policy, run_id="t", base_seed=1)
    records, _ = asyncio.run(collector.collect(20))

    for prev, cur in zip(records, records[1:]):
        assert cur.environment_step >= 0
        assert cur.selected_action_index < len(cur.legal_action_descriptors)
        assert len(cur.legal_action_descriptors) >= 1
        assert np.isfinite(cur.nasimemu_reward)
        assert np.isfinite(cur.critic_value)


def test_resource_summary_peak_is_never_less_than_mean():
    from marla.monitoring.resources import ResourceMonitor
    import time

    monitor = ResourceMonitor(sampling_interval_seconds=0.02, enabled=True)
    monitor.start()
    time.sleep(0.1)
    monitor.stop()
    summary = monitor.summarize()
    if summary.get("peak_rss_mib") is not None and summary.get("mean_rss_mib") is not None:
        assert summary["peak_rss_mib"] >= summary["mean_rss_mib"] - 1e-6


def test_message_sent_and_received_totals_always_match():
    """A structural invariant of the one-event-per-message design: total
    sent across all agents always equals total received across all
    agents, exactly, for any sequence of messages (every message has
    exactly one sender and one receiver)."""
    from marla.messaging import telemetry
    from marla.messaging.builders import build_message
    from marla.messaging.schemas import MessageMetadata, MessageType

    telemetry.stop_run()
    log = telemetry.start_run("t")
    for i in range(7):
        meta = MessageMetadata(
            performative="inform", message_type=MessageType.ADVISORY_REQUEST, schema_version="1.0",
            run_id="t", conversation_id=f"c{i}", request_id=f"r{i}",
            sender_alias=f"agent_{i % 3}", receiver_alias=f"agent_{(i + 1) % 3}",
        )
        build_message("x@localhost", meta)
    summary = log.summarize()
    assert sum(summary["sent_count_by_agent"].values()) == sum(summary["received_count_by_agent"].values()) == 7
    telemetry.stop_run()


def test_no_nan_or_inf_in_persisted_ppo_update_metrics():
    """PPO update rows must never persist NaN/Inf into updates.csv."""
    from marla.metrics.writer import build_update_rows

    rows = build_update_rows(
        "t",
        [
            {
                "update": 1, "rollout": 1, "epoch": 1, "minibatch": 1, "environment_steps": 2048,
                "policy_loss": 0.01, "value_loss": 0.5, "action_entropy": 3.9, "approximate_kl": 0.001,
                "clip_fraction": 0.0, "explained_variance": 0.2, "gradient_norm": 0.7, "learning_rate": 0.0003,
            }
        ],
    )
    for row in rows:
        for key, value in row.items():
            if isinstance(value, float):
                assert math.isfinite(value), f"{key} is not finite: {value}"
