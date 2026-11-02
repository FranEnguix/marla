"""Dynamic candidate-action scorer (spec section 13).

``b_{t,i} = w^T tanh(W_z z_t + W_e e_{t,i} + W_x x_{t,i})``. Candidate sets
vary in size per sample, so actions are passed flattened across the batch
together with an ``action_to_sample`` index vector -- the same pattern Torch
Geometric uses for variable-size graphs in a ``Batch`` (see ``data.batch``).

``x_{t,i}`` is the optional per-action context (evidence and/or host-action
history features, :mod:`marla.environment.action_context`); the ``W_x`` term
exists only when ``action_context_size > 0``. ``W_x`` starts at zero and is
created without drawing from the RNG, so a policy built with action context
has exactly the same initial weights and initial action distribution as one
built without it; only learning can make the context matter.
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class BaseActionScorer(nn.Module):
    def __init__(
        self, recurrent_hidden_size: int, action_hidden_size: int, scorer_hidden_size: int, action_context_size: int = 0
    ):
        super().__init__()
        self.w_z = nn.Linear(recurrent_hidden_size, scorer_hidden_size, bias=True)
        self.w_e = nn.Linear(action_hidden_size, scorer_hidden_size, bias=False)
        self.w = nn.Linear(scorer_hidden_size, 1, bias=False)
        self.action_context_size = action_context_size
        if action_context_size > 0:
            self.w_x = nn.Parameter(torch.zeros(scorer_hidden_size, action_context_size))

    def forward(
        self, z: Tensor, action_embeddings: Tensor, action_to_sample: Tensor, action_context: Tensor | None = None
    ) -> Tensor:
        """Returns raw (unnormalized) logits, one per action row.

        ``z``: ``(B, recurrent_hidden_size)``.
        ``action_embeddings``: ``(N_total, action_hidden_size)``.
        ``action_to_sample``: ``(N_total,)`` int64, values in ``[0, B)``.
        ``action_context``: ``(N_total, action_context_size)``; required iff
        ``action_context_size > 0``.
        """
        z_per_action = z[action_to_sample]
        pre_activation = self.w_z(z_per_action) + self.w_e(action_embeddings)
        if self.action_context_size > 0:
            if action_context is None or action_context.shape != (action_embeddings.shape[0], self.action_context_size):
                raise ValueError(
                    f"action_context must have shape ({action_embeddings.shape[0]}, {self.action_context_size}), "
                    f"got {None if action_context is None else tuple(action_context.shape)}"
                )
            pre_activation = pre_activation + F.linear(action_context, self.w_x)
        elif action_context is not None:
            raise ValueError("action_context given to a scorer built without action context")
        hidden = torch.tanh(pre_activation)
        return self.w(hidden).squeeze(-1)
