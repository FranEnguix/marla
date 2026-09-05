"""Evaluation-time overrides consumed by ``RolloutCollector._decide()``.

Every field defaults to exactly today's behavior. ``RolloutCollector`` only
looks at this dataclass when explicitly given one (``overrides=...``);
``marla.learning.trainer`` never constructs or passes one, so the training
path is unaffected by this module's existence. See
``research/aamas2027/AUDIT.md`` section 4 and the plan's "Architecture"
section for the reasoning behind each field.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

QueryMode = Literal["learned", "never", "always"]
ActionSelection = Literal["policy", "plan_maker_argmax"]
AdviceTransformation = Literal["identity", "shuffled", "reversed"]


@dataclass(frozen=True)
class EvaluationOverrides:
    """One evaluation condition's deviation from MARLA_FULL's learned behavior.

    ``query_mode``:
      - ``"learned"`` -- unchanged: threshold the query gate's own output at
        >= 0.5 (deterministic eval already does this with no overrides at
        all; spelled out here so every ablation's ``query_mode`` is
        self-documenting).
      - ``"never"`` -- force ``sampled_query = False``; ``consult_fn`` is
        never called. (NO_QUERY.)
      - ``"always"`` -- force ``sampled_query = True``; ``consult_fn`` is
        always called. (ALWAYS_QUERY, and PLAN_MAKER_ONLY via
        ``action_selection``.)
      In every case the gate's own ``query_probability`` is still computed
      and recorded, for comparison -- only the *sampling* decision is
      overridden.

    ``beta_override``:
      When not ``None`` and a query happened and Plan Maker advice was
      accepted (``normalized_advice is not None``), replaces the trust
      head's learned ``beta`` with this fixed value before recomputing
      ``final_logits``. A no-op whenever advice wasn't obtained -- you
      cannot force trust in advice that doesn't exist, and that is the
      scientifically correct behavior for this override, not a gap.
      (BETA_ZERO: 0.0. BETA_ONE: 1.0.)

    ``action_selection``:
      - ``"policy"`` -- unchanged: sample/argmax over ``final_logits``.
      - ``"plan_maker_argmax"`` -- ignore the policy's distribution
        entirely; select ``argmax`` over the (validated) Plan Maker score
        vector. On ``schema_rejected``, falls back to a fixed, seeded
        uniform-random legal-action choice (``fallback_rng_seed``) --
        documented as exactly that, a fallback rule, never a re-use of any
        learned policy output, which would contradict "no learned RL
        action choice." (PLAN_MAKER_ONLY.)

    ``advice_transformation``:
      ``"identity"`` is the only mode implemented in this phase.
      ``"shuffled"``/``"reversed"`` are reserved for the postponed
      SHUFFLED_ADVICE/REVERSED_ADVICE ablations (see AUDIT.md) and raise
      ``NotImplementedError`` if selected, so the surface exists without
      pretending the behavior does.
    """

    query_mode: QueryMode = "learned"
    beta_override: float | None = None
    action_selection: ActionSelection = "policy"
    advice_transformation: AdviceTransformation = "identity"
    fallback_rng_seed: int = 0

    def __post_init__(self) -> None:
        if self.advice_transformation != "identity":
            raise NotImplementedError(
                f"advice_transformation={self.advice_transformation!r} is not implemented in this "
                "phase (see research/aamas2027/AUDIT.md) -- only 'identity' is supported."
            )


# The five Phase-4-equivalent ablations actually implemented in this phase,
# named to match research/aamas2027/manifest.yaml's `condition` values.
NORMAL = EvaluationOverrides()
NO_QUERY = EvaluationOverrides(query_mode="never")
ALWAYS_QUERY = EvaluationOverrides(query_mode="always")
BETA_ZERO = EvaluationOverrides(beta_override=0.0)
BETA_ONE = EvaluationOverrides(beta_override=1.0)
PLAN_MAKER_ONLY = EvaluationOverrides(query_mode="always", action_selection="plan_maker_argmax")
