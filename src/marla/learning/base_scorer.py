"""Dynamic candidate-action scorer (spec section 13).

``b_{t,i} = w^T tanh(W_z z_t + W_e e_{t,i})``. Candidate sets vary in size
per sample, so actions are passed flattened across the batch together with
an ``action_to_sample`` index vector -- the same pattern Torch Geometric
uses for variable-size graphs in a ``Batch`` (see ``data.batch``).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn


class BaseActionScorer(nn.Module):
    def __init__(self, recurrent_hidden_size: int, action_hidden_size: int, scorer_hidden_size: int):
        super().__init__()
        self.w_z = nn.Linear(recurrent_hidden_size, scorer_hidden_size, bias=True)
        self.w_e = nn.Linear(action_hidden_size, scorer_hidden_size, bias=False)
        self.w = nn.Linear(scorer_hidden_size, 1, bias=False)

    def forward(self, z: Tensor, action_embeddings: Tensor, action_to_sample: Tensor) -> Tensor:
        """Returns raw (unnormalized) logits, one per action row.

        ``z``: ``(B, recurrent_hidden_size)``.
        ``action_embeddings``: ``(N_total, action_hidden_size)``.
        ``action_to_sample``: ``(N_total,)`` int64, values in ``[0, B)``.
        """
        z_per_action = z[action_to_sample]
        hidden = torch.tanh(self.w_z(z_per_action) + self.w_e(action_embeddings))
        return self.w(hidden).squeeze(-1)
