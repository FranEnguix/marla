"""GRU recurrent state (spec section 13).

``z_t = GRUCell(x_t, z_{t-1})`` with
``x_t = [g_t, p_t^visible, E(a_{t-1}), r~_{t-1}, q_{t-1}]``. ``p_t^visible``
(spec: FINISH-learnability investigation, ``VISIBLE_PROGRESS_DIM``-wide --
see :func:`marla.environment.visible_facts.compute_visible_progress`) is a
compact, fixed-width, VISIBLE-ONLY summary of global objective progress:
GraphSAGE deliberately encodes only local/topological host features and
ActionEncoder only per-action/target relations, so neither carries any
signal for "how much of the objective is already done" -- this is the one
place that global summary enters the policy, and it is never derived from
hidden simulator state (see that function's own docstring for the exact
anti-leak argument).

At episode start, ``z`` is zero, the previous action is a learned start
token, and the previous reward/query are zero. All tensors are batch-first
(``B`` = 1 for single-instance live rollout collection, ``B`` > 1 for PPO
minibatch replay in Milestone 4) so the same module serves both.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RecurrentCore(nn.Module):
    def __init__(
        self, graph_embedding_size: int, action_embedding_size: int, hidden_size: int, visible_progress_dim: int
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.action_embedding_size = action_embedding_size
        self.visible_progress_dim = visible_progress_dim
        input_size = (
            graph_embedding_size + visible_progress_dim + action_embedding_size + 1 + 1
        )  # + reward + query
        self.gru_cell = nn.GRUCell(input_size, hidden_size)
        self.start_action_token = nn.Parameter(torch.zeros(action_embedding_size))
        nn.init.normal_(self.start_action_token, std=0.02)

    def initial_hidden_state(self, batch_size: int, device: torch.device) -> Tensor:
        return torch.zeros(batch_size, self.hidden_size, device=device)

    def initial_previous_action_embedding(self, batch_size: int, device: torch.device) -> Tensor:
        return self.start_action_token.to(device).unsqueeze(0).expand(batch_size, -1).clone()

    def forward(
        self,
        graph_embedding: Tensor,
        visible_progress: Tensor,
        previous_action_embedding: Tensor,
        previous_reward: Tensor,
        previous_query: Tensor,
        previous_hidden: Tensor,
    ) -> Tensor:
        if previous_reward.dim() == 1:
            previous_reward = previous_reward.unsqueeze(-1)
        if previous_query.dim() == 1:
            previous_query = previous_query.unsqueeze(-1)
        x = torch.cat(
            [graph_embedding, visible_progress, previous_action_embedding, previous_reward, previous_query],
            dim=-1,
        )
        return self.gru_cell(x, previous_hidden)
