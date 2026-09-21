"""Route-switch diagnostics and the PPO subnet-routing likelihood-ratio
audit (see ``marla.learning.decision``'s module docstring for the full
classification -- PIECEWISE_EXACT_SEMIGRADIENT).

Diagnostic-only seeds throughout (never 401/402/403). No Plan Maker call
ever happens during replay in any of these tests -- the scripted
``consult_fn`` is only ever invoked during COLLECTION.
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.consultation_scope import extract_action_subnet, select_consulted_subnet
from marla.environment.graph import GraphObservation
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.evaluation.overrides import EvaluationOverrides
from marla.learning.gae import compute_gae
from marla.learning.ppo import build_sequence_chunks, optimize, _replay_chunk
from marla.learning.recurrent_policy import RecurrentPolicy, RecurrentState
from marla.learning.rollout import ConsultationResult, RolloutCollector

REPO_ROOT = Path(__file__).resolve().parent.parent
TWO_SUBNET_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_two_subnets.v2.yaml").resolve())
DIAGNOSTIC_SEED = 777001  # arbitrary, local to this file, never 401/402/403


def _assisted_config():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 4
    data["policy"]["recurrent"]["sequence_length"] = 8
    data["consultation"] = {"mode": "learned", "cost": 0.1, "max_schema_revisions": 3}
    data["gatekeeper"] = {"alias": "gatekeeper", "jid": "gk@localhost"}
    data["agents"] = [
        {
            "alias": "plan_maker_1", "jid": "pm@localhost", "role": "plan_maker",
            "model": {"backend": "local", "name": "x"}, "prompt_version": "v1",
            "knowledge": {"path": "package://marla/knowledge/nasimemu_rules.yaml", "version": "v1"},
        }
    ]
    data["environment"]["scenario"] = TWO_SUBNET_SCENARIO
    return parse_config(data)


async def _scripted_consult(legal_actions, episode_id, step, source_observation_id, observation, consulted_subnet, global_candidate_action_count):
    return ConsultationResult(
        status="accepted",
        scores={a.action_id: (i + 1) / (len(legal_actions) + 1) for i, a in enumerate(legal_actions)},
        request_id=f"request-{episode_id}-{step}",
    )


async def _collect_all_queried(policy, adapter, num_steps=16) -> list:
    """Forces every step to query via EvaluationOverrides(query_mode="always")
    -- the gate's own query_probability is still the REAL, unforced value
    (only the sampling decision is overridden, per overrides.py's own
    docstring), which is essential here: forcing sampled_query via a
    monkeypatched compute_query_probability instead would bake a FAKE
    probability into old_joint_log_probability at collection time that
    replay's own (real, unpatched) recomputation could never reproduce --
    a test-methodology bug, not a production one, caught while writing
    this exact test."""
    collector = RolloutCollector(
        adapter, policy, run_id="route-switch-test", base_seed=DIAGNOSTIC_SEED,
        consultation_enabled=True, consultation_cost=0.1, consult_fn=_scripted_consult,
        overrides=EvaluationOverrides(query_mode="always"),
    )
    records, _summaries = await collector.collect(num_steps)
    queried = [r for r in records if r.sampled_query]
    assert queried, "expected every collected step to be queried"
    return records


# --- 8. Old-policy identity: the hard correctness gate ----------------------


@pytest.mark.asyncio
async def test_replay_at_theta_old_reproduces_old_log_prob_and_has_zero_route_switches():
    """Hard regression test (spec section 8): replaying a collected
    queried transition BEFORE any optimizer update must reproduce
    old_joint_log_probability within tight tolerance, INCLUDING
    subnet-scoped advice -- if this fails, it is a correctness bug, not a
    theory question. At theta_old, route-switch rate must also be exactly
    zero (the route is a deterministic function of base_logits, and
    nothing about the parameters has changed since collection)."""
    torch.manual_seed(DIAGNOSTIC_SEED)
    config = _assisted_config()
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    adapter = NasimEmuAdapter(
        scenario=TWO_SUBNET_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    records = await _collect_all_queried(policy, adapter)

    chunks = build_sequence_chunks(records, [0.0] * len(records), [0.0] * len(records), sequence_length=len(records))
    device = torch.device("cpu")

    all_switches: list[bool] = []
    for chunk in chunks:
        # theta has not moved at all since collection -- no optimizer.step()
        # has ever been called on `policy` at this point.
        replay = _replay_chunk(policy, chunk, device, consultation_cost=0.1)
        assert len(replay.new_joint_log_probs) == len(chunk.records)
        for i, record in enumerate(chunk.records):
            assert torch.isclose(
                replay.new_joint_log_probs[i], torch.tensor(record.old_joint_log_probability), atol=1e-5, rtol=1e-4
            ), f"replay log-prob diverged from old_joint_log_probability at theta_old for record {i}"
        all_switches.extend(replay.route_switches)

    assert all_switches, "expected at least one queried record to have a defined route-switch entry"
    assert not any(all_switches), "route-switch rate must be exactly zero when replaying at theta_old"


# --- 4/5. Deterministic route-switch mechanism check ------------------------


@pytest.mark.asyncio
async def test_route_switch_diagnostic_detects_an_engineered_route_change():
    """Proves the DETECTION MECHANISM itself is correct: given a chunk
    replayed once at theta_old (route_switches all False, per the test
    above) and once again after parameters have been deliberately pushed
    so the deterministic route for at least one transition now disagrees
    with the stored one, route_switches must report True for that
    transition -- without ever calling the Plan Maker a second time (the
    scripted consult_fn's call count is asserted unchanged)."""
    torch.manual_seed(DIAGNOSTIC_SEED)
    config = _assisted_config()
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    adapter = NasimEmuAdapter(
        scenario=TWO_SUBNET_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    call_count = 0

    async def counting_consult(legal_actions, episode_id, step, source_observation_id, observation, consulted_subnet, global_candidate_action_count):
        nonlocal call_count
        call_count += 1
        return await _scripted_consult(
            legal_actions, episode_id, step, source_observation_id, observation, consulted_subnet, global_candidate_action_count
        )

    collector = RolloutCollector(
        adapter, policy, run_id="route-switch-test-2", base_seed=DIAGNOSTIC_SEED,
        consultation_enabled=True, consultation_cost=0.1, consult_fn=counting_consult,
        overrides=EvaluationOverrides(query_mode="always"),
    )
    # A subnet only becomes visible after a SubnetScan reveals it -- most
    # early steps of an episode see only the DMZ entry subnet, which gives
    # compute_routing_margin nothing to switch to at all (spec: "None when
    # every non-FINISH candidate belongs to the same subnet"). Enough steps
    # for natural exploration (still diagnostic-seed, non-paper) to reach a
    # genuinely multi-subnet-visible state, then keep only the queried
    # records where a route switch is even POSSIBLE.
    records, _summaries = await collector.collect(400)
    all_queried = [r for r in records if r.sampled_query and r.consulted_subnet is not None]
    assert all_queried, "expected at least one queried, routed decision in 400 steps"
    calls_after_collection = call_count

    # Only records where MULTIPLE subnets were actually visible (a route
    # switch is structurally possible at all -- see compute_routing_margin)
    # are useful for engineering a switch; skip (rather than fail) if this
    # diagnostic seed's 400 collected steps never explored a second subnet,
    # since that is a property of the scenario/seed, not the mechanism
    # under test.
    def _visible_subnet_count(record) -> int:
        return len({
            int(a.target_key.split("-")[1])
            for a in record.legal_action_descriptors if not a.is_finish and a.target_key
        })

    queried = [r for r in all_queried if _visible_subnet_count(r) > 1]
    if not queried:
        pytest.skip("no queried decision in this diagnostic seed's 400 collected steps had >1 visible subnet")

    device = torch.device("cpu")
    chunks = build_sequence_chunks(records, [0.0] * len(records), [0.0] * len(records), sequence_length=len(records))

    def _fresh_base_logits(record):
        graph_obs = GraphObservation(data=record.graph_data.to(device), node_key_to_index=record.node_key_to_index)
        rstate = RecurrentState(
            z=record.initial_gru_hidden_state.to(device),
            previous_action_embedding=record.previous_action_embedding.to(device),
            previous_reward=record.previous_training_reward,
            previous_query=float(record.previous_query),
        )
        return policy.step(
            graph_obs, record.legal_action_descriptors, rstate,
            record.compatibility_features.to(device), record.visible_progress.to(device),
        ).base_logits

    # Deliberately push base_logits: for each queried record, find the
    # SPECIFIC cross-subnet competitor action compute_routing_margin
    # itself would identify (the highest-logit non-FINISH action from any
    # OTHER subnet than the current stored winner's), and maximize ITS
    # logit -- a large-LR, repeated gradient ascent engineered specifically
    # to flip routing on at least one transition. A real, mechanistically
    # faithful parameter update (not weight surgery), just an artificially
    # aggressive/targeted one -- diagnostic only, this policy/optimizer is
    # discarded after this test.
    optimizer = torch.optim.SGD(policy.parameters(), lr=5.0)
    for _ in range(20):
        loss = torch.zeros(())
        any_target = False
        for record in queried:
            base_logits = _fresh_base_logits(record)
            non_finish = [(i, a) for i, a in enumerate(record.legal_action_descriptors) if not a.is_finish]
            if not non_finish:
                continue
            winner_local = max(range(len(non_finish)), key=lambda j: base_logits[non_finish[j][0]].item())
            _winner_index, winner_action = non_finish[winner_local]
            winner_subnet = extract_action_subnet(winner_action)
            competitors = [
                i for i, a in non_finish if extract_action_subnet(a) != winner_subnet
            ]
            if not competitors:
                continue
            best_competitor = max(competitors, key=lambda i: base_logits[i].item())
            loss = loss - base_logits[best_competitor]
            any_target = True
        if not any_target:
            break
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    assert call_count == calls_after_collection, "the engineered parameter push must never call the Plan Maker again"

    # Re-replay with the now-perturbed parameters and check the mechanism
    # actually reports what genuinely changed.
    found_switch = False
    for chunk in chunks:
        replay = _replay_chunk(policy, chunk, device, consultation_cost=0.1)
        for record, switched in zip([r for r in chunk.records if r.sampled_query], replay.route_switches):
            if switched:
                found_switch = True
                # Cross-check: the diagnostic's own verdict must agree with
                # directly recomputing the counterfactual route.
                current_route = select_consulted_subnet(
                    record.legal_action_descriptors, _fresh_base_logits(record).detach()
                )
                assert current_route != record.consulted_subnet

    assert found_switch, (
        "expected the engineered parameter push to flip at least one transition's route -- "
        "if this ever fails, the perturbation itself (not the diagnostic) needs revisiting"
    )
    assert call_count == calls_after_collection, "no Plan Maker call may happen while checking for the switch either"


# --- 5/9. Route switching through real PPO epochs, vs. KL/clip/margin ------


@pytest.mark.asyncio
async def test_route_switch_fraction_starts_at_zero_and_stays_well_formed_across_real_ppo_updates(capsys):
    """A genuine (not engineered) short PPO training probe: real
    collection, real multi-epoch optimize(). Structural properties
    (spec sections 5/9), not tuned thresholds:

    - the FIRST minibatch replayed (still at theta_old, before its own
      optimizer.step()) has route_switch_fraction == 0.0 whenever it
      queried anything at all -- consistent with the hard identity proven
      above;
    - route_switch_fraction/mean_routing_margin/approximate_kl/clip_fraction
      are always well-formed (finite, fractions in [0, 1]) across every
      subsequent update, however training actually behaves;
    - route switching is empirically watched against approximate_kl/
      clip_fraction/routing margin for a qualitative report -- nothing here
      tunes a hyperparameter from what is observed.
    """
    torch.manual_seed(DIAGNOSTIC_SEED)
    config = _assisted_config()
    data = config.model_dump()
    data["policy"]["ppo"]["epochs"] = 4
    data["policy"]["ppo"]["steps_per_env"] = 400
    data["policy"]["ppo"]["minibatch_sequences"] = 8
    config = parse_config(data)

    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.policy.ppo.optimizer.learning_rate)
    adapter = NasimEmuAdapter(
        scenario=TWO_SUBNET_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )
    collector = RolloutCollector(
        adapter, policy, run_id="route-switch-probe", base_seed=DIAGNOSTIC_SEED,
        consultation_enabled=True, consultation_cost=config.consultation.cost, consult_fn=_scripted_consult,
        overrides=EvaluationOverrides(query_mode="always"),
    )
    # A longer collection window than a typical unit test, deliberately --
    # a subnet only becomes visible once a SubnetScan reveals it, and this
    # probe is specifically trying to observe route-switch behavior when
    # MULTIPLE subnets are actually in play, not just confirm the trivially
    # single-subnet case. Still a diagnostic seed, still no Plan Maker call
    # anywhere near replay, still no paper seed.
    records, _summaries = await collector.collect(config.policy.ppo.steps_per_env)

    rewards = [r.training_reward for r in records]
    values = [r.critic_value for r in records]
    terminated = [r.terminated for r in records]
    truncated = [r.truncated for r in records]
    bootstrap_values = [r.bootstrap_value for r in records]
    advantages, returns = compute_gae(
        rewards, values, terminated, truncated, bootstrap_values,
        config.policy.ppo.gamma, config.policy.ppo.gae_lambda,
    )

    all_metrics = optimize(
        policy, optimizer, records, advantages, returns, config.policy.ppo,
        config.policy.recurrent.sequence_length, config.policy.ppo.minibatch_sequences,
        torch.device("cpu"), random.Random(DIAGNOSTIC_SEED), consultation_cost=config.consultation.cost,
    )
    assert all_metrics, "expected at least one PPO update from this probe"

    first_queried_update = next(
        (m for m in all_metrics if m["queried_replay_transitions"] > 0), None
    )
    if first_queried_update is not None:
        assert first_queried_update["route_switch_fraction"] == 0.0, (
            "the very first replayed minibatch is still at theta_old (no optimizer.step() "
            "has touched it yet) -- its route_switch_fraction must be exactly 0.0"
        )

    report_lines = [
        f"{'update':>6} {'queried':>8} {'switches':>9} {'switch_frac':>12} "
        f"{'mean_margin':>12} {'approx_kl':>10} {'clip_frac':>10}"
    ]
    for i, m in enumerate(all_metrics):
        assert isinstance(m["queried_replay_transitions"], int) and m["queried_replay_transitions"] >= 0
        assert isinstance(m["route_switch_count"], int) and m["route_switch_count"] >= 0
        assert m["route_switch_count"] <= m["queried_replay_transitions"]
        if m["route_switch_fraction"] is not None:
            assert 0.0 <= m["route_switch_fraction"] <= 1.0
            assert torch.isfinite(torch.tensor(m["route_switch_fraction"]))
        if m["mean_routing_margin"] is not None:
            assert torch.isfinite(torch.tensor(m["mean_routing_margin"]))
        assert torch.isfinite(torch.tensor(m["approximate_kl"]))
        assert torch.isfinite(torch.tensor(m["clip_fraction"]))
        report_lines.append(
            f"{i:>6} {m['queried_replay_transitions']:>8} {m['route_switch_count']:>9} "
            f"{str(m['route_switch_fraction']):>12} {str(m['mean_routing_margin']):>12} "
            f"{m['approximate_kl']:>10.5f} {m['clip_fraction']:>10.4f}"
        )

    with capsys.disabled():
        print("\n--- route-switch vs. KL/clip diagnostic (diagnostic seed, informational only) ---")
        for line in report_lines:
            print(line)
