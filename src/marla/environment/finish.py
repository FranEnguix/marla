"""``FINISH`` reward semantics.

``FINISH`` is a MARLA wrapper action, not a NASimEmu action. Selecting it
never calls into the underlying simulator: NASimEmu's own ``TerminalAction``
path unconditionally returns ``reward=0`` and triggers an internal
auto-reset (see ``NASimEmuEnv.step``), which is not what MARLA needs (a
configured completion reward or premature-finish penalty, and precise
control over when the *next* episode's scenario is generated). Since a
compound decision advances NASimEmu *at most* once and FINISH performs zero
NASimEmu actions, this trivially satisfies that invariant while avoiding a
redundant double-reset.
"""

from __future__ import annotations


def compute_finish_reward(
    objective_satisfied: bool,
    completion_reward: float,
    premature_finish_penalty: float,
) -> float:
    """Reward for a FINISH action, per spec section 7."""
    return completion_reward if objective_satisfied else premature_finish_penalty
