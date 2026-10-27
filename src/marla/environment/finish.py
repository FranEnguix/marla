"""``FINISH`` reward semantics.

``FINISH`` is a MARLA wrapper action, not a NASimEmu action. Selecting it
never calls into the underlying simulator: NASimEmu's own ``TerminalAction``
path unconditionally returns ``reward=0`` and triggers an internal
auto-reset (see ``NASimEmuEnv.step``), which is not what MARLA needs (a
target-dependent reward, and precise control over when the *next* episode's
scenario is generated). Since a compound decision advances NASimEmu *at
most* once and FINISH performs zero NASimEmu actions, this trivially
satisfies that invariant while avoiding a redundant double-reset.

With ``N`` sensitive targets in the scenario and ``M`` of them still lacking
ROOT access when FINISH is selected (defaults shown)::

    R_FINISH = +1 * N    if M == 0   (successful FINISH)
             = -1 * M    otherwise   (premature FINISH)

This reward exists only on the FINISH transition: rooting the last target
with any other action earns only NASimEmu's own reward for that action, and
every non-FINISH action keeps NASimEmu's original reward unchanged.
"""

from __future__ import annotations

# Defaults live with ObjectiveConfig (a pydantic-only module, importable without NASimEmu); re-exported here.
from marla.config.models import (  # noqa: F401
    DEFAULT_PREMATURE_FINISH_REWARD_PER_REMAINING_TARGET,
    DEFAULT_SUCCESSFUL_FINISH_REWARD_PER_TARGET,
)


def compute_finish_reward(
    total_sensitive_targets: int,
    remaining_sensitive_targets: int,
    successful_finish_reward_per_target: float = DEFAULT_SUCCESSFUL_FINISH_REWARD_PER_TARGET,
    premature_finish_reward_per_remaining_target: float = DEFAULT_PREMATURE_FINISH_REWARD_PER_REMAINING_TARGET,
) -> float:
    """Reward for a FINISH action.

    ``remaining_sensitive_targets`` counts sensitive targets without ROOT
    access (USER access does not count; non-sensitive hosts never count).
    A scenario without sensitive targets has no well-defined objective -- a
    vacuous "all zero targets rooted" must never become a successful FINISH
    -- so ``total_sensitive_targets < 1`` is rejected, as is a remaining
    count outside ``[0, total]``.
    """
    if total_sensitive_targets < 1:
        raise ValueError(
            f"FINISH reward is undefined for a scenario with {total_sensitive_targets} sensitive targets"
        )
    if not 0 <= remaining_sensitive_targets <= total_sensitive_targets:
        raise ValueError(
            f"remaining_sensitive_targets={remaining_sensitive_targets} outside [0, {total_sensitive_targets}]"
        )
    if remaining_sensitive_targets == 0:
        return float(successful_finish_reward_per_target * total_sensitive_targets)
    return float(premature_finish_reward_per_remaining_target * remaining_sensitive_targets)
