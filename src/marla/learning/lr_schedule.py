"""Learning-rate schedules for PPO training (spec section 12).

Progress is measured in *environment steps* actually completed against the
config's ``total_environment_steps`` budget, never in minibatch or epoch
count -- the number of minibatches a rollout happens to produce depends on
episode-boundary chunking (see learning/ppo.py) and would make the schedule
depend on incidental episode lengths rather than genuine training progress.
The schedule is evaluated once per rollout (see trainer.py), not once per
minibatch, which is coarser but avoids coupling the schedule to how many
minibatches a given rollout's chunking happens to generate.
"""

from __future__ import annotations

from typing import Literal


def compute_learning_rate(
    schedule: Literal["constant", "linear"],
    initial_learning_rate: float,
    completed_environment_steps: int,
    total_environment_steps: int,
) -> float:
    """Current learning rate for ``schedule``, never negative.

    ``progress = completed_environment_steps / total_environment_steps``,
    clamped to ``[0, 1]`` so a resumed run or an off-by-one at the exact
    budget boundary never produces a negative rate. ``constant`` ignores
    progress entirely (prior behavior, unchanged).
    """
    if schedule == "constant":
        return initial_learning_rate

    if total_environment_steps <= 0:
        raise ValueError("total_environment_steps must be > 0")

    progress = completed_environment_steps / total_environment_steps
    progress = min(1.0, max(0.0, progress))
    return initial_learning_rate * (1.0 - progress)
