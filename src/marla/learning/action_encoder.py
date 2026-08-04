"""Per-action embedding :math:`e_{t,i} = f_{action}(E_{type}, h_{target}, E_{parameters})`.

Non-target actions (``finish``) use a learned no-target embedding in place
of a target-host embedding (spec section 12). Action-specific parameters
(exploit/privesc ``service``/``os``/``process`` names) are summarized as a
small fixed-size presence indicator rather than embedded by name, so the
model does not bake in a per-scenario service/OS vocabulary -- scenarios can
introduce new service/OS/process names without changing the model shape.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

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

        input_dim = action_type_embedding_size + node_embedding_size + PARAMETER_FEATURE_DIM
        self.mlp = nn.Sequential(nn.Linear(input_dim, hidden_size), nn.Tanh())

    def forward(
        self,
        type_ids: Tensor,
        target_embeddings: Tensor,
        has_target_mask: Tensor,
        parameter_feats: Tensor,
    ) -> Tensor:
        """All inputs are batched over actions: shapes ``(N, ...)``.

        ``target_embeddings`` rows are ignored (replaced by the learned
        no-target embedding) wherever ``has_target_mask`` is ``False``.
        """
        type_emb = self.type_embedding(type_ids)
        no_target = self.no_target_embedding.unsqueeze(0).expand(target_embeddings.shape[0], -1)
        mask = has_target_mask.bool().unsqueeze(-1)
        target_emb = torch.where(mask, target_embeddings, no_target)
        x = torch.cat([type_emb, target_emb, parameter_feats], dim=-1)
        return self.mlp(x)

    def encode_descriptors(
        self,
        descriptors: list[ActionDescriptor],
        node_embeddings: Tensor,
        node_key_to_index: dict[str, int],
        device: torch.device,
    ) -> Tensor:
        """Convenience wrapper: build encoder inputs directly from descriptors."""
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

        return self.forward(type_ids, target_embeddings, has_target_mask, param_feats)
