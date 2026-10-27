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

Default (target-dependent) reward, with ``N`` sensitive targets in the
scenario and ``M`` of them still lacking ROOT access when FINISH is chosen::

    R_FINISH = +50 * N     if M == 0   (successful FINISH)
             = -100 * M    otherwise   (premature FINISH)

This reward exists only on the FINISH transition: rooting the last target
with any other action earns only NASimEmu's own reward for that action.
"""

from __future__ import annotations

# Defaults live with ObjectiveConfig (a pydantic-only module, importable without NASimEmu); re-exported here.
from marla.config.models import (  # noqa: F401
    DEFAULT_COMPLETION_REWARD_PER_TARGET,
    DEFAULT_PREMATURE_FINISH_PENALTY,
    DEFAULT_PREMATURE_FINISH_PENALTY_PER_REMAINING_TARGET,
)


def compute_finish_reward(
    objective_satisfied: bool,
    completion_reward: float | None,
    premature_finish_penalty: float,
    premature_finish_penalty_per_remaining_target: float = 0.0,
    remaining_sensitive_targets: int = 0,
    completion_reward_per_target: float = 0.0,
    total_sensitive_targets: int = 0,
) -> float:
    """Reward for a FINISH action, per spec section 7.

    Successful FINISH (``objective_satisfied``): a fixed ``completion_reward``
    when one is given (legacy/explicit configuration), otherwise
    ``completion_reward_per_target * total_sensitive_targets``.

    Premature FINISH: ``premature_finish_penalty`` plus
    ``premature_finish_penalty_per_remaining_target`` for each sensitive/value
    host not yet at ROOT access (USER access does not count).
    ``remaining_sensitive_targets`` is ignored -- and should be 0 -- when the
    objective is satisfied, since a satisfied objective means no targets remain.
    """
    if objective_satisfied:
        if completion_reward is not None:
            return completion_reward
        return completion_reward_per_target * total_sensitive_targets
    return premature_finish_penalty + premature_finish_penalty_per_remaining_target * remaining_sensitive_targets
