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
    premature_finish_penalty_per_remaining_target: float = 0.0,
    remaining_sensitive_targets: int = 0,
) -> float:
    """Reward for a FINISH action, per spec section 7.

    A premature FINISH (``objective_satisfied`` is False) costs a fixed
    ``premature_finish_penalty`` plus ``premature_finish_penalty_per_remaining_target``
    for each sensitive/value host not yet at ROOT access. The per-target
    term defaults to 0.0, so callers that never pass it (or pass the
    default) get exactly the old fixed-penalty behavior. ``remaining_sensitive_targets``
    is ignored -- and should be 0 -- when the objective is satisfied, since
    a satisfied objective means no targets remain.
    """
    if objective_satisfied:
        return completion_reward
    return premature_finish_penalty + premature_finish_penalty_per_remaining_target * remaining_sensitive_targets
