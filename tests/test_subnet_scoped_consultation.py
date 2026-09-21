"""Cross-cutting correctness tests for subnet-scoped Plan Maker consultation
-- the properties that only show up when advice.py/recurrent_policy.py/
decision.py/rollout.py/ppo.py/checkpoint.py/direct_consult.py/gatekeeper.py
interact, not any one module in isolation (see the per-module test files
for their own targeted coverage: test_consultation_scope.py,
test_decision.py, test_direct_consult.py).
"""

from __future__ import annotations

import random
from pathlib import Path

import pytest
import torch

from marla.config.loader import load_config, parse_config
from marla.environment.actions import ActionDescriptor, finish_descriptor, host_target_key
from marla.environment.consultation_scope import select_consulted_actions, select_consulted_subnet
from marla.environment.nasimemu_adapter import NasimEmuAdapter
from marla.evaluation.direct_consult import DirectConsultant
from marla.knowledge.retriever import KnowledgeBase
from marla.learning.advice import (
    clip_and_logit,
    compute_advice_summary,
    compute_agreement_features,
    normalize_advice,
    summary_to_tensor,
)
from marla.learning.checkpoint import ConsultationScopeMismatchError, load_checkpoint, save_checkpoint
from marla.learning.decision import compute_final_decision
from marla.learning.ppo import build_sequence_chunks, _replay_chunk
from marla.learning.recurrent_policy import PolicyStepOutput, RecurrentPolicy
from marla.learning.rollout import ConsultationResult, RolloutCollector
from marla.messaging.advisory_validation import AdvisoryValidationError, validate_advisory_response_body
from marla.messaging.schemas import AdvisoryActionDescriptor, AdvisoryObjective
from marla.models.plan_maker_backend import BackendResponse
from marla.models.prompt import build_prompt

REPO_ROOT = Path(__file__).resolve().parent.parent
TWO_SUBNET_SCENARIO = str((REPO_ROOT / "NASimEmu/scenarios/sm_entry_dmz_two_subnets.v2.yaml").resolve())


def _tiny_config():
    config = load_config(REPO_ROOT / "examples" / "baseline.yaml")
    data = config.model_dump()
    data["policy"]["ppo"]["steps_per_env"] = 16
    data["policy"]["ppo"]["epochs"] = 1
    data["policy"]["ppo"]["minibatch_sequences"] = 2
    data["policy"]["recurrent"]["sequence_length"] = 4
    return parse_config(data)


def make_assisted_policy():
    return RecurrentPolicy(_tiny_config().policy, consultation_enabled=True)


def _actions(subnet: int, count: int, start_host: int = 0) -> list[ActionDescriptor]:
    return [
        ActionDescriptor(
            action_id=f"service-scan:{host_target_key(subnet, start_host + i)}",
            action_type="service_scan",
            target_key=host_target_key(subnet, start_host + i),
        )
        for i in range(count)
    ]


def _recurrent_hidden_size(policy: RecurrentPolicy) -> int:
    # TrustHead.mlp[0] is nn.Linear(recurrent_hidden_size + SUMMARY_DIM + AGREEMENT_DIM, ...)
    from marla.learning.advice import TrustHead

    return policy.trust_head.mlp[0].in_features - TrustHead.SUMMARY_DIM - TrustHead.AGREEMENT_DIM


def _step_output(num_actions: int, hidden_size: int, action_hidden_size: int = 16) -> PolicyStepOutput:
    z = torch.randn(hidden_size)
    base_logits = torch.randn(num_actions)
    base_probs = torch.softmax(base_logits, dim=-1)
    action_embeddings = torch.randn(num_actions, action_hidden_size)
    value = torch.randn(())
    return PolicyStepOutput(z=z, base_logits=base_logits, base_probs=base_probs, action_embeddings=action_embeddings, value=value)


# --- 17. single-subnet equivalence ------------------------------------------


def test_single_subnet_equivalence_matches_the_dense_computation_exactly():
    """When every non-FINISH candidate belongs to one subnet, the new
    subnet-scoped apply_advice must be numerically IDENTICAL to computing
    the (old, pre-scoping) dense advice math directly over the full vector
    -- consulted == global in this case, so there is no scope-narrowing
    effect to introduce any numerical difference at all."""
    torch.manual_seed(0)
    policy = make_assisted_policy()
    actions = _actions(subnet=1, count=3) + [finish_descriptor()]
    step_out = _step_output(num_actions=4, hidden_size=_recurrent_hidden_size(policy))
    confidence = torch.tensor([0.9, 0.2, 0.6, 0.5])

    consulted_subnet = select_consulted_subnet(actions, step_out.base_logits)
    assert consulted_subnet == 1
    scope = select_consulted_actions(actions, consulted_subnet)
    assert len(scope.consulted_indices) == 4  # every action consulted -- single-subnet case
    consulted_indices = torch.tensor(scope.consulted_indices, dtype=torch.long)

    decision = compute_final_decision(policy, step_out, True, confidence, consulted_indices)

    # Manually reproduce the OLD dense computation over the FULL vector.
    log_odds = clip_and_logit(confidence)
    normalized = normalize_advice(log_odds)
    summary = summary_to_tensor(compute_advice_summary(log_odds))
    agreement = compute_agreement_features(step_out.base_logits, log_odds)
    expected_beta = policy.trust_head(step_out.z, summary, agreement)
    expected_alpha = policy.advice_scale()
    expected_final_logits = step_out.base_logits + expected_beta * expected_alpha * normalized

    assert torch.allclose(decision.beta, expected_beta)
    assert torch.allclose(decision.alpha, expected_alpha)
    assert torch.allclose(decision.normalized_advice, normalized)
    assert torch.allclose(decision.final_logits, expected_final_logits)


# --- 18. invariance to unrelated actions in other subnets -------------------


def test_invariance_to_unrelated_actions_in_other_subnets():
    """Case A: 5 subnet-1 actions + FINISH. Case B: same 6, PLUS 500 actions
    from other subnets. The consulted subset (the same 6) must produce an
    IDENTICAL sparse residual/beta/alpha in both cases, and every one of
    the 500 unrelated actions must have final_logit == base_logit exactly
    (spec section 18 -- one of the central scalability/correctness
    properties)."""
    torch.manual_seed(0)
    policy = make_assisted_policy()

    shared_actions = _actions(subnet=1, count=5) + [finish_descriptor()]
    unrelated_actions = _actions(subnet=2, count=500, start_host=100)

    actions_a = shared_actions
    actions_b = shared_actions + unrelated_actions

    hidden_size, action_hidden_size = _recurrent_hidden_size(policy), 16
    torch.manual_seed(1)
    z = torch.randn(hidden_size)
    # Shared actions' logits are offset well above the unrelated ones' range
    # so subnet 1 deterministically wins the routing argmax in BOTH cases --
    # invariance here means "the CONSULTED subset's local math is unaffected
    # by unrelated actions existing", not "routing itself ignores every
    # other action's logit" (routing IS a global argmax by design).
    base_logits_shared = torch.randn(6) + 100.0
    base_logits_unrelated = torch.randn(500)
    base_logits_a = base_logits_shared
    base_logits_b = torch.cat([base_logits_shared, base_logits_unrelated])
    embeddings_shared = torch.randn(6, action_hidden_size)
    embeddings_unrelated = torch.randn(500, action_hidden_size)

    step_out_a = PolicyStepOutput(
        z=z, base_logits=base_logits_a, base_probs=torch.softmax(base_logits_a, dim=-1),
        action_embeddings=embeddings_shared, value=torch.zeros(()),
    )
    step_out_b = PolicyStepOutput(
        z=z, base_logits=base_logits_b, base_probs=torch.softmax(base_logits_b, dim=-1),
        action_embeddings=torch.cat([embeddings_shared, embeddings_unrelated]), value=torch.zeros(()),
    )

    confidence = torch.tensor([0.9, 0.1, 0.5, 0.7, 0.3, 0.4])  # scores for the 6 shared/consulted actions

    consulted_subnet_a = select_consulted_subnet(actions_a, step_out_a.base_logits)
    scope_a = select_consulted_actions(actions_a, consulted_subnet_a)
    consulted_subnet_b = select_consulted_subnet(actions_b, step_out_b.base_logits)
    scope_b = select_consulted_actions(actions_b, consulted_subnet_b)

    # Both cases must route to the same subnet, and consult exactly the
    # same 6 actions at the same (leading) global positions.
    assert consulted_subnet_a == consulted_subnet_b
    assert scope_a.consulted_indices == [0, 1, 2, 3, 4, 5]
    assert scope_b.consulted_indices == [0, 1, 2, 3, 4, 5]

    indices_a = torch.tensor(scope_a.consulted_indices, dtype=torch.long)
    indices_b = torch.tensor(scope_b.consulted_indices, dtype=torch.long)

    decision_a = compute_final_decision(policy, step_out_a, True, confidence, indices_a)
    decision_b = compute_final_decision(policy, step_out_b, True, confidence, indices_b)

    # The consulted subset's advice/beta/alpha must be EXACTLY identical --
    # 500 unrelated actions elsewhere must not perturb local normalization,
    # TrustHead's local summary/agreement, or the resulting residual.
    assert torch.equal(decision_a.beta, decision_b.beta)
    assert torch.equal(decision_a.alpha, decision_b.alpha)
    assert torch.equal(decision_a.normalized_advice, decision_b.normalized_advice[:6])
    assert torch.equal(decision_a.final_logits, decision_b.final_logits[:6])

    # Every one of the 500 unrelated actions must be untouched EXACTLY.
    for i in range(6, 506):
        assert decision_b.final_logits[i].item() == step_out_b.base_logits[i].item()
        assert decision_b.normalized_advice[i].item() == 0.0


# --- Gatekeeper exact-coverage against a SCOPED request ---------------------


def test_gatekeeper_validation_accepts_exact_scoped_coverage_and_rejects_deviations():
    """validate_advisory_response_body's exact-set-equality rule (unchanged
    by this feature -- see agents/gatekeeper.py's own docstring on why it
    needs no code changes) applied against a SCOPED expected-ID set: this
    is what actually enforces spec section 20's Gatekeeper requirements."""
    scoped_ids = {"service-scan:host-1-0", "service-scan:host-1-1", "finish"}

    exact_body = {
        "schema_version": "1.1", "run_id": "r", "request_id": "req-1",
        "scores": {a: 0.5 for a in scoped_ids},
        "model_version": "m", "prompt_version": "p", "knowledge_version": "k",
        "inference_latency_ms": 1.0, "retrieved_rule_ids": [],
    }
    validated = validate_advisory_response_body(exact_body, "r", "req-1", scoped_ids)
    assert set(validated.scores.keys()) == scoped_ids

    missing_body = dict(exact_body, scores={"service-scan:host-1-0": 0.5, "finish": 0.5})
    with pytest.raises(AdvisoryValidationError):
        validate_advisory_response_body(missing_body, "r", "req-1", scoped_ids)

    invented_body = dict(
        exact_body,
        scores={**{a: 0.5 for a in scoped_ids}, "exploit:host-9-0:eternalblue": 0.9},
    )
    with pytest.raises(AdvisoryValidationError):
        validate_advisory_response_body(invented_body, "r", "req-1", scoped_ids)


# --- checkpoint semantic-version rejection (MARLA_FULL only) ---------------


def test_marla_full_checkpoint_rejects_pre_subnet_scoping_consultation_version(tmp_path):
    config = _tiny_config()
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, policy, optimizer, update_count=1, environment_steps=10, config_hash="h")

    # Simulate a checkpoint saved before subnet-scoped consultation existed
    # (no consultation_scope_version field at all).
    data = torch.load(path, map_location="cpu", weights_only=False)
    del data["consultation_scope_version"]
    torch.save(data, path)

    fresh_policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    with pytest.raises(ConsultationScopeMismatchError):
        load_checkpoint(path, fresh_policy)


def test_ppo_only_checkpoint_loading_is_unaffected_by_consultation_scope_version(tmp_path):
    """Invariant 11/12: a PPO_ONLY (consultation_enabled=False) policy never
    builds/uses TrustHead/advice, so a missing/mismatched
    consultation_scope_version must never be checked for it."""
    config = _tiny_config()
    policy = RecurrentPolicy(config.policy, consultation_enabled=False)
    optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)
    path = tmp_path / "checkpoint.pt"
    save_checkpoint(path, policy, optimizer, update_count=1, environment_steps=10, config_hash="h")

    data = torch.load(path, map_location="cpu", weights_only=False)
    del data["consultation_scope_version"]
    torch.save(data, path)

    fresh_policy = RecurrentPolicy(config.policy, consultation_enabled=False)
    metadata = load_checkpoint(path, fresh_policy)  # must not raise
    assert metadata.consultation_scope_version is None


# --- SPADE path / DirectConsultant prompt parity ----------------------------


class _CapturingBackend:
    def __init__(self, response_text: str):
        self._response_text = response_text
        self.captured_prompt: str | None = None

    async def generate(self, prompt: str, legal_action_ids: list[str]) -> BackendResponse:
        self.captured_prompt = prompt
        return BackendResponse(raw_text=self._response_text, latency_ms=1.0)


@pytest.mark.asyncio
async def test_direct_consultant_builds_the_identical_prompt_build_prompt_would():
    """DirectConsultant has no subnet-routing code of its own (spec section
    21) -- it must call build_prompt with exactly the scoped args it was
    given, producing a prompt byte-identical to calling build_prompt()
    directly with the same inputs. This is what "no duplicated
    subnet-routing code" actually proves: there is only one prompt-
    construction path, reused verbatim."""
    consulted_subnet = 3
    global_candidate_action_count = 40
    scoped_observation = {
        "global_progress": {
            "visible_sensitive_targets_total": 1, "visible_sensitive_targets_with_root": 0,
            "visible_sensitive_targets_remaining": 1, "known_subnets_count": 2,
            "successfully_scanned_subnets_count": 1, "known_unscanned_subnets_count": 1,
            "selected_subnet": consulted_subnet,
        },
        "local_hosts": [
            {
                "target": "host-3-0", "access": "none", "compromised": False, "reachable": True,
                "is_sensitive_target": True, "known_os": [], "known_services": [], "known_processes": [],
            }
        ],
    }
    legal_actions = [
        ActionDescriptor(action_id="service-scan:host-3-0", action_type="service_scan", target_key="host-3-0"),
        finish_descriptor(),
    ]
    objective = AdvisoryObjective(type="capture_target", description="test objective")

    backend = _CapturingBackend('{"service-scan:host-3-0": 0.7, "finish": 0.1}')
    consultant = DirectConsultant(
        backend=backend, knowledge_base=KnowledgeBase(version="v1", rules=()), model_version="m",
        objective=objective, run_id="r", max_schema_revisions=0,
    )
    await consultant(
        legal_actions, episode_id=1, step=0, source_observation_id="obs-1-0",
        observation=scoped_observation, consulted_subnet=consulted_subnet,
        global_candidate_action_count=global_candidate_action_count,
    )
    assert backend.captured_prompt is not None

    reference_prompt = build_prompt(
        retrieved_rules=[], objective=objective, observation=scoped_observation,
        candidate_actions=[
            AdvisoryActionDescriptor(action_id=a.action_id, type=a.action_type, target=a.target_key, parameters=a.parameters)
            for a in legal_actions
        ],
        consulted_subnet=consulted_subnet, global_candidate_action_count=global_candidate_action_count,
    )
    assert backend.captured_prompt == reference_prompt


# --- PPO replay reuses the stored consultation artifact exactly ------------


@pytest.mark.asyncio
async def test_ppo_replay_reuses_the_stored_consulted_indices_and_scores_verbatim(monkeypatch):
    """Spec sections 13/14: PPO replay must feed apply_advice EXACTLY the
    consulted_action_indices/plan_maker_scores_in_action_order stored at
    collection time -- never recomputed from the replay's own (possibly
    different, under updated parameters) base logits. Proven by spying on
    RecurrentPolicy.apply_advice during replay and comparing its actual
    call arguments against the StepRecord's own stored fields, not an
    artificial sentinel (the point is to confirm replay reuses what
    collection genuinely produced)."""
    config = _tiny_config()
    data = config.model_dump()
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
    config = parse_config(data)

    torch.manual_seed(0)
    policy = RecurrentPolicy(config.policy, consultation_enabled=True)
    adapter = NasimEmuAdapter(
        scenario=TWO_SUBNET_SCENARIO,
        max_episode_steps=config.environment.max_episode_steps,
        completion_reward=config.objective.completion_reward,
        premature_finish_penalty=config.objective.premature_finish_penalty,
    )

    call_count = 0

    async def scripted_consult(legal_actions, episode_id, step, source_observation_id, observation, consulted_subnet, global_candidate_action_count):
        nonlocal call_count
        call_count += 1
        # Varied, deterministic-but-non-uniform scores so the stored vector
        # is distinguishable from an accidental all-equal default.
        return ConsultationResult(
            status="accepted",
            scores={a.action_id: (i + 1) / (len(legal_actions) + 1) for i, a in enumerate(legal_actions)},
            request_id=f"request-{call_count}",
        )

    import unittest.mock

    with unittest.mock.patch("marla.learning.rollout.compute_query_probability", return_value=torch.tensor(1.0)):
        collector = RolloutCollector(
            adapter, policy, run_id="test-run", base_seed=1,
            consultation_enabled=True, consultation_cost=0.1, consult_fn=scripted_consult,
        )
        records, _summaries = await collector.collect(4)

    queried_accepted = [r for r in records if r.sampled_query and r.plan_maker_validation_status == "accepted"]
    assert queried_accepted, "expected at least one queried+accepted decision to replay"

    captured: list[tuple[torch.Tensor, torch.Tensor]] = []
    real_apply_advice = RecurrentPolicy.apply_advice

    def spy(self, z, base_logits, confidence_local, consulted_indices):
        captured.append((confidence_local.clone(), consulted_indices.clone()))
        return real_apply_advice(self, z, base_logits, confidence_local, consulted_indices)

    monkeypatch.setattr(RecurrentPolicy, "apply_advice", spy)

    chunks = build_sequence_chunks(records, [0.0] * len(records), [0.0] * len(records), sequence_length=len(records))
    calls_before_replay = call_count
    for chunk in chunks:
        _replay_chunk(policy, chunk, torch.device("cpu"), consultation_cost=0.1)

    assert call_count == calls_before_replay, "PPO replay must never call the Plan Maker again"
    assert len(captured) == len(queried_accepted)
    for (confidence_local, consulted_indices), record in zip(captured, queried_accepted):
        assert record.plan_maker_scores_in_action_order is not None
        assert record.consulted_action_indices is not None
        expected_confidence = torch.tensor(record.plan_maker_scores_in_action_order, dtype=torch.float32)
        expected_indices = torch.tensor(record.consulted_action_indices, dtype=torch.long)
        assert torch.equal(confidence_local, expected_confidence)
        assert torch.equal(consulted_indices, expected_indices)
