"""Generalized Advantage Estimation with correct episode-boundary handling.

Three cases determine the "next value" used at each step:

- ``terminated``: a true terminal transition (FINISH). The next value is 0
  -- there is no future return to bootstrap.
- ``truncated``: the episode was cut short (``max_episode_steps``). The
  caller must supply a ``bootstrap_values[t]`` estimate of the state that
  followed, since the *next* buffer row (if any) belongs to a fresh episode
  with a reset GRU state and is not a valid continuation.
- Neither, but ``t`` is the last row of the rollout window: the PPO horizon
  ended mid-episode. The caller must again supply ``bootstrap_values[t]``
  (the value of continuing under the current policy).

In every other case, the next value is simply ``values[t + 1]`` -- the same
episode continues into the next buffer row.
"""

from __future__ import annotations

from collections.abc import Sequence


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    terminated: Sequence[bool],
    truncated: Sequence[bool],
    bootstrap_values: Sequence[float | None],
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    """Returns ``(advantages, returns)``, both length ``len(rewards)``."""
    length = len(rewards)
    if not (len(values) == len(terminated) == len(truncated) == len(bootstrap_values) == length):
        raise ValueError("all input sequences must have the same length")

    advantages = [0.0] * length
    last_gae = 0.0

    for t in reversed(range(length)):
        episode_ended = terminated[t] or truncated[t]

        if terminated[t]:
            next_value = 0.0
        elif truncated[t]:
            if bootstrap_values[t] is None:
                raise ValueError(f"bootstrap_values[{t}] is required when truncated[{t}] is True")
            next_value = bootstrap_values[t]
        elif t == length - 1:
            if bootstrap_values[t] is None:
                raise ValueError(
                    f"bootstrap_values[{t}] is required for the last row of a rollout "
                    "window unless it is terminated"
                )
            next_value = bootstrap_values[t]
        else:
            next_value = values[t + 1]

        delta = rewards[t] + gamma * next_value - values[t]
        last_gae = delta if episode_ended else delta + gamma * gae_lambda * last_gae
        advantages[t] = last_gae

    returns = [advantages[t] + values[t] for t in range(length)]
    return advantages, returns
