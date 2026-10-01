"""Evaluation policy modes and reproducible evaluation-time sampling.

:class:`EvaluationPolicyMode` selects how a frozen policy's decisions are
turned into actions during checkpoint evaluation
(:func:`marla.evaluation.checkpoint_eval.evaluate_checkpoint`):

``GREEDY`` (default, unchanged behavior)
    ``q_t = [p_t^q >= 0.5]``, ``a_t = argmax pi_t``. Evaluates the
    deterministic controller derived from the learned policy.

``STOCHASTIC_POLICY``
    ``q_t ~ Bernoulli(p_t^q)``, ``a_t ~ pi_t`` (for MARLA_FULL, ``pi_t`` is
    the final, post-advice distribution MARLA already builds). Estimates
    performance under the learned stochastic policy itself.

Neither mode is universally preferable; they answer different questions.

Reproducible sampling (``EVALUATION_RNG_SCHEME``)
-------------------------------------------------
``STOCHASTIC_POLICY`` never draws from torch's (or Python's/NumPy's) global
RNG. Each evaluation episode gets two independent streams, one per purpose
``"query"`` and ``"action"``::

    key  = "marla-stochastic-eval|{agent_seed}|{eval_seed}|{purpose}"
    seed = int.from_bytes(sha256(key.encode("utf-8")).digest()[:8], "little")
    rng  = numpy.random.default_rng(seed)          # PCG64

``agent_seed`` is the trained run's ``experiment.seed``; ``eval_seed`` is the
episode's environment seed (``NasimEmuAdapter.reset(seed=...)``). Each
stream is created lazily on its first draw and consumed in decision order
within the episode. One draw per sampled decision:

* query: ``u = rng.random()``; ``q_t = u < p_t^q`` (``p_t^q`` read as a
  Python float of the gate's probability).
* action: ``p`` = the categorical distribution's (normalized) probabilities
  in float64; ``u = rng.random() * p.sum()``;
  ``a_t = min(searchsorted(cumsum(p), u, side="right"), len(p) - 1)``.

Consequences: the same checkpoint + eval seed + mode reproduces the same
trajectory; query draws never shift action draws (separate streams); a
decision whose query is forced by an override (NO_QUERY/ALWAYS_QUERY), and
every PPO_ONLY decision, consumes no query draw; PLAN_MAKER_ONLY selects by
Plan Maker argmax and consumes no action draw; and logging or any other code
that touches a global RNG cannot change the trajectory.
"""

from __future__ import annotations

import hashlib
from enum import Enum

import numpy as np
from torch import Tensor

EVALUATION_RNG_SCHEME = "marla-eval-rng-v1:sha256(agent_seed|eval_seed|purpose)->pcg64"
QUERY_STREAM = "query"
ACTION_STREAM = "action"


class EvaluationPolicyMode(str, Enum):
    GREEDY = "GREEDY"
    STOCHASTIC_POLICY = "STOCHASTIC_POLICY"


def evaluation_stream_seed(agent_seed: int, eval_seed: int, purpose: str) -> int:
    """The 64-bit seed of one evaluation random stream (see module docstring)."""
    key = f"marla-stochastic-eval|{int(agent_seed)}|{int(eval_seed)}|{purpose}"
    return int.from_bytes(hashlib.sha256(key.encode("utf-8")).digest()[:8], "little")


class EvaluationSampler:
    """Per-episode, per-purpose evaluation random streams for one trained agent.

    ``RolloutCollector`` calls :meth:`start_episode` with each episode's
    environment seed, then :meth:`sample_query`/:meth:`sample_action` in
    decision order. ``draws`` counts draws per purpose across the whole
    evaluation (recorded in evaluation metadata).
    """

    def __init__(self, agent_seed: int) -> None:
        self.agent_seed = int(agent_seed)
        self.draws = {QUERY_STREAM: 0, ACTION_STREAM: 0}
        self._eval_seed: int | None = None
        self._streams: dict[str, np.random.Generator] = {}

    def start_episode(self, eval_seed: int) -> None:
        self._eval_seed = int(eval_seed)
        self._streams = {}

    def _next_uniform(self, purpose: str) -> float:
        if self._eval_seed is None:
            raise RuntimeError("EvaluationSampler used before start_episode()")
        stream = self._streams.get(purpose)
        if stream is None:
            stream = np.random.default_rng(evaluation_stream_seed(self.agent_seed, self._eval_seed, purpose))
            self._streams[purpose] = stream
        self.draws[purpose] += 1
        return float(stream.random())

    def sample_query(self, probability: Tensor) -> bool:
        """``q ~ Bernoulli(probability)`` from the episode's query stream."""
        if probability.dim() != 0:
            raise ValueError("sample_query expects a scalar probability")
        return self._next_uniform(QUERY_STREAM) < float(probability)

    def sample_action(self, probs: Tensor) -> int:
        """An index ``~ Categorical(probs)`` from the episode's action stream."""
        if probs.dim() != 1:
            raise ValueError("sample_action expects a 1-D probability vector")
        p = probs.detach().double().cpu().numpy()
        u = self._next_uniform(ACTION_STREAM) * p.sum()
        return min(int(np.searchsorted(np.cumsum(p), u, side="right")), len(p) - 1)
