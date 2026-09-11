"""Actor-logit-level proof that the architecture can now represent formerly
forced-equal decisions (spec section 31) -- not just that the intermediate
embeddings differ (already covered by ``test_action_encoder.py``), but that
this actually reaches the base policy's logits/probabilities. Uses a fixed
seed for reproducible (not merely "eventually converged") weights -- this
establishes representational *capacity*, not that an untrained network
already prefers the correct action.
"""

from __future__ import annotations

import torch

from marla.config.models import (
    ActionEncoderConfig,
    ConstantSchedulerConfig,
    GraphEncoderConfig,
    OptimizerConfig,
    PolicyConfig,
    PPOConfig,
    RecurrentConfig,
)
from marla.environment.action_compatibility import compute_compatibility_matrix
from marla.environment.actions import ActionDescriptor
from marla.environment.graph import GraphObservation, NODE_FEATURE_DIM
from marla.environment.visible_facts import VisibleHostFacts, compute_visible_progress
from marla.learning.recurrent_policy import RecurrentPolicy
from nasimemu.nasim.envs.utils import AccessLevel
from torch_geometric.data import Data

TARGET = "host-2-0"


def _tiny_policy_config() -> PolicyConfig:
    return PolicyConfig(
        graph_encoder=GraphEncoderConfig(hidden_size=8, layers=1),
        action_encoder=ActionEncoderConfig(hidden_size=8, action_type_embedding_size=4),
        recurrent=RecurrentConfig(hidden_size=8, sequence_length=4),
        ppo=PPOConfig(
            total_environment_steps=1, steps_per_env=1, epochs=1, minibatch_sequences=1,
            gamma=0.99, gae_lambda=0.95, clip_epsilon=0.2, value_coefficient=0.5,
            query_entropy_coefficient=0.01, action_entropy_coefficient=0.01,
            max_grad_norm=0.5,
            optimizer=OptimizerConfig(learning_rate=0.0003, scheduler=ConstantSchedulerConfig()),
        ),
    )


def _single_host_graph() -> GraphObservation:
    x = torch.zeros((1, NODE_FEATURE_DIM), dtype=torch.float32)
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    return GraphObservation(data=Data(x=x, edge_index=edge_index), node_key_to_index={TARGET: 0})


def _exploit_pair() -> list[ActionDescriptor]:
    return [
        ActionDescriptor(f"exploit:{TARGET}:e_elasticsearch", "exploit", TARGET,
                          {"service": "9200_windows_elasticsearch", "os": "windows"}),
        ActionDescriptor(f"exploit:{TARGET}:e_wp_ninja", "exploit", TARGET,
                          {"service": "80_windows_wp_ninja", "os": "windows"}),
    ]


def _step(policy: RecurrentPolicy, descriptors, compatibility, facts_by_target=None):
    graph_obs = _single_host_graph()
    rstate = policy.initial_recurrent_state()
    visible_progress = compute_visible_progress(facts_by_target or {})
    return policy.step(graph_obs, descriptors, rstate, compatibility, visible_progress)


def test_formerly_aliased_actions_can_receive_different_logits_once_facts_differ():
    torch.manual_seed(0)
    policy = RecurrentPolicy(_tiny_policy_config())
    descriptors = _exploit_pair()

    # Before any fact is observed: both UNKNOWN, features identical ->
    # logits forced equal (not overcorrected, spec section 9).
    no_facts: dict[str, VisibleHostFacts] = {}
    zero_compat = compute_compatibility_matrix(descriptors, no_facts)
    out_unknown = _step(policy, descriptors, zero_compat, no_facts)
    assert torch.allclose(out_unknown.base_logits[0], out_unknown.base_logits[1])

    # After elasticsearch is confirmed present (Windows OS also observed):
    # the two actions' compatibility features now differ, and so must the
    # resulting logits -- this is the representational-capacity proof.
    facts = {
        TARGET: VisibleHostFacts(
            target_key=TARGET, reachable=True, compromised=False, access=AccessLevel.NONE,
            known_os=frozenset({"windows"}), known_services=frozenset({"9200_windows_elasticsearch"}),
            known_processes=frozenset(),
        )
    }
    real_compat = compute_compatibility_matrix(descriptors, facts)
    assert not torch.allclose(real_compat[0], real_compat[1])  # precondition

    out_observed = _step(policy, descriptors, real_compat, facts)
    assert not torch.allclose(out_observed.base_logits[0], out_observed.base_logits[1])
    assert not torch.allclose(out_observed.base_probs[0], out_observed.base_probs[1])


def test_windows_and_linux_privesc_logits_differ_once_os_is_observed():
    torch.manual_seed(0)
    policy = RecurrentPolicy(_tiny_policy_config())
    descriptors = [
        ActionDescriptor(f"privilege-escalation:{TARGET}:marla_repair_windows_root_privesc",
                          "privilege_escalation", TARGET, {"process": None, "os": "windows"}),
        ActionDescriptor(f"privilege-escalation:{TARGET}:pe_kernel",
                          "privilege_escalation", TARGET, {"process": None, "os": "linux"}),
    ]
    facts = {
        TARGET: VisibleHostFacts(
            target_key=TARGET, reachable=True, compromised=True, access=AccessLevel.USER,
            known_os=frozenset({"windows"}), known_services=frozenset(), known_processes=frozenset(),
        )
    }
    compatibility = compute_compatibility_matrix(descriptors, facts)
    out = _step(policy, descriptors, compatibility, facts)
    assert not torch.allclose(out.base_logits[0], out.base_logits[1])
