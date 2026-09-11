r"""Per-action embedding
:math:`e_{t,i} = f_{action}(E_{type}, h_{target}, p_{t,i}, c_{t,i})`.

Non-target actions (``finish``) use a learned no-target embedding in place
of a target-host embedding (spec section 12). Action-specific parameters
(exploit/privesc ``service``/``os``/``process`` names) are summarized two
ways, neither of which bakes in a per-scenario service/OS/process
vocabulary -- the model shape never depends on how many distinct
services/OSes/processes/exploits/privescs a scenario defines:

- ``p_{t,i}`` (:func:`parameter_features`, :data:`PARAMETER_FEATURE_DIM`):
  compact *presence* indicators -- does this action have a service/process
  parameter at all, does it have an OS parameter at all.
- ``c_{t,i}`` (:mod:`marla.environment.action_compatibility`,
  :data:`marla.environment.action_compatibility.COMPATIBILITY_FEATURE_DIM`):
  fixed-width *relational* features between the action's static
  requirements and the visible facts of its target -- "the service this
  action requires has been observed on this target", not "this action
  requires Elasticsearch". This is what lets two exploits/privescs with
  identical broad type and parameter-presence shape (e.g. ``e_elasticsearch``
  vs. ``e_wp_ninja`` on the same Windows host, or a Windows-only privesc vs.
  a Linux-only one) receive genuinely different embeddings once the
  relevant facts have been observed -- see that module's docstring for the
  full rationale and exactly which facts are (and are not) used.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

from marla.environment.action_compatibility import COMPATIBILITY_FEATURE_DIM
from marla.environment.actions import ActionDescriptor

ACTION_TYPES = (
    "service_scan",
    "os_scan",
    "subnet_scan",
    "process_scan",
    "exploit",
    "privilege_escalation",
    "finish",
)
ACTION_TYPE_TO_ID = {name: i for i, name in enumerate(ACTION_TYPES)}
PARAMETER_FEATURE_DIM = 2  # [has_service_or_process_param, has_os_param]

# Bumped whenever a change alters the trained policy's parameter shape --
# ActionEncoder's own input width, or (v3) RecurrentCore's -- so a
# checkpoint saved under an older representation fails loudly and
# specifically at load time rather than with a bare PyTorch shape-mismatch
# stack trace, or -- worse -- silently (spec section 44/45). Despite living
# in this module (where the concept originated), this is the ONE
# representation-version constant for the whole trained policy, imported
# generically by marla.learning.checkpoint.
# v1: [type_embedding, target_embedding, parameter_features] only.
# v2: adds c_{t,i}, the fixed-width action/target compatibility features
#     (see module docstring) -- input width grows by COMPATIBILITY_FEATURE_DIM;
#     output width (hidden_size) is unchanged.
# v3: RecurrentCore's x_t grows by VISIBLE_PROGRESS_DIM (FINISH-
#     learnability investigation -- see marla.environment.visible_facts
#     .compute_visible_progress and recurrent_core.RecurrentCore's own
#     docstring); ActionEncoder itself is unchanged from v2.
# v4: GraphSAGE's NODE_FEATURE_DIM grows 12 -> 13 (per-subnet-node
#     ``subnet_scan_completed`` feature, observable subnet-exploration
#     progress) and RecurrentCore's x_t gains an optional, independently-
#     ablatable exploration-progress component (VISIBLE_EXPLORATION_PROGRESS_DIM)
#     alongside the existing target-progress one -- see
#     marla.environment.visible_facts.VisibleNetworkExploration. Because
#     the two visible-progress components are now independently
#     switchable (RecurrentConfig.visible_target_progress/
#     visible_subnet_exploration -- the v2/v3-target/v3-full ablation),
#     this integer alone is NOT sufficient identity for RecurrentCore's
#     actual input width; marla.learning.checkpoint additionally records
#     and checks both flags directly (see CheckpointMetadata).
POLICY_REPRESENTATION_VERSION = 4


def action_type_id(action_type: str) -> int:
    try:
        return ACTION_TYPE_TO_ID[action_type]
    except KeyError as exc:
        raise ValueError(f"Unknown action_type: {action_type!r}") from exc


def parameter_features(parameters: dict[str, object]) -> Tensor:
    has_service_or_process = bool(parameters.get("service") or parameters.get("process"))
    has_os = bool(parameters.get("os"))
    return torch.tensor([float(has_service_or_process), float(has_os)], dtype=torch.float32)


class ActionEncoder(nn.Module):
    def __init__(self, node_embedding_size: int, action_type_embedding_size: int, hidden_size: int):
        super().__init__()
        self.node_embedding_size = node_embedding_size
        self.type_embedding = nn.Embedding(len(ACTION_TYPES), action_type_embedding_size)
        self.no_target_embedding = nn.Parameter(torch.zeros(node_embedding_size))
        nn.init.normal_(self.no_target_embedding, std=0.02)

        input_dim = (
            action_type_embedding_size + node_embedding_size + PARAMETER_FEATURE_DIM + COMPATIBILITY_FEATURE_DIM
        )
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.Tanh())

    def forward(
        self,
        type_ids: Tensor,
        target_embeddings: Tensor,
        has_target_mask: Tensor,
        parameter_feats: Tensor,
        compatibility_feats: Tensor,
    ) -> Tensor:
        """All inputs are batched over actions: shapes ``(N, ...)``.

        ``target_embeddings`` rows are ignored (replaced by the learned
        no-target embedding) wherever ``has_target_mask`` is ``False``.
        ``compatibility_feats`` must be exactly the tensor computed at
        rollout-collection time for this action set (spec section 11) --
        never recomputed from live simulator state during PPO replay.
        """
        type_emb = self.type_embedding(type_ids)
        no_target = self.no_target_embedding.unsqueeze(0).expand(target_embeddings.shape[0], -1)
        mask = has_target_mask.bool().unsqueeze(-1)
        target_emb = torch.where(mask, target_embeddings, no_target)
        x = torch.cat([type_emb, target_emb, parameter_feats, compatibility_feats], dim=-1)
        return self.mlp(x)

    def encode_descriptors(
        self,
        descriptors: list[ActionDescriptor],
        node_embeddings: Tensor,
        node_key_to_index: dict[str, int],
        device: torch.device,
        compatibility_matrix: Tensor,
    ) -> Tensor:
        """Convenience wrapper: build encoder inputs directly from descriptors.

        ``compatibility_matrix`` -- ``(len(descriptors), COMPATIBILITY_FEATURE_DIM)``,
        in the same order as ``descriptors`` -- is a required, explicit
        argument rather than computed internally, precisely so rollout
        collection and PPO replay are forced to supply (and can be proven
        to supply) the identical tensor; see
        :func:`marla.environment.action_compatibility.compute_compatibility_matrix`.
        """
        if compatibility_matrix.shape != (len(descriptors), COMPATIBILITY_FEATURE_DIM):
            raise ValueError(
                f"compatibility_matrix shape {tuple(compatibility_matrix.shape)} does not match "
                f"({len(descriptors)}, {COMPATIBILITY_FEATURE_DIM})"
            )
        type_ids = torch.tensor(
            [action_type_id(d.action_type) for d in descriptors], dtype=torch.long, device=device
        )
        has_target_mask = torch.tensor(
            [d.target_key is not None for d in descriptors], dtype=torch.bool, device=device
        )
        target_embeddings = torch.zeros(
            (len(descriptors), self.node_embedding_size), dtype=torch.float32, device=device
        )
        for i, descriptor in enumerate(descriptors):
            if descriptor.target_key is not None:
                node_index = node_key_to_index[descriptor.target_key]
                target_embeddings[i] = node_embeddings[node_index]
        param_feats = torch.stack([parameter_features(d.parameters) for d in descriptors]).to(device)

        return self.forward(
            type_ids, target_embeddings, has_target_mask, param_feats, compatibility_matrix.to(device)
        )
