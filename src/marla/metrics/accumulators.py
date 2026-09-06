"""Compact, rollout-by-rollout accumulators for summary.json.

Replace the old pattern of deriving summary.json's consultation/advice
statistics from a retained ``all_records: list[StepRecord]`` covering the
whole run (spec section 1). Each of these holds only scalars/small lists
whose size is proportional to the number of *consultations* (or, for
``action_type_counts``, the number of distinct action types) -- never to
the number of environment steps' worth of heavyweight per-step tensors.

Imported by both ``learning.trainer`` (which updates them once per
rollout, right before that rollout's ``StepRecord`` list goes out of scope)
and ``metrics.writer`` (which reads them at shutdown) -- kept in this
standalone module specifically so ``learning.trainer`` never has to import
``metrics.writer`` (which itself imports ``TrainingResult`` from
``learning.trainer``) just to update these.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field


@dataclass
class ConsultationStats:
    """Running totals over every consultation made so far this run."""

    consultations_total: int = 0
    schema_rejections_total: int = 0
    accepted_count: int = 0
    advice_changed_count: int = 0
    beta_sum: float = 0.0
    beta_count: int = 0
    # One float per consultation (not per environment step) -- bounded by
    # how often the query gate actually fires, which is always <= the
    # number of decisions and in practice a small fraction of it.
    latencies_ms: list[float] = field(default_factory=list)

    def update(self, decision_rows: list[dict]) -> None:
        """Folds one rollout's already-built decisions.csv rows into the running totals."""
        for row in decision_rows:
            if not row.get("queried"):
                continue
            self.consultations_total += 1
            if row.get("response_status") == "schema_rejected":
                self.schema_rejections_total += 1
            elif row.get("response_status") == "accepted":
                self.accepted_count += 1
                if row.get("advice_changed_top_action"):
                    self.advice_changed_count += 1
            if row.get("beta") is not None:
                self.beta_sum += row["beta"]
                self.beta_count += 1
            if row.get("response_latency_ms") is not None:
                self.latencies_ms.append(row["response_latency_ms"])

    @property
    def mean_beta(self) -> float | None:
        return self.beta_sum / self.beta_count if self.beta_count else None

    @property
    def advice_changed_top_action_rate(self) -> float | None:
        return self.advice_changed_count / self.accepted_count if self.accepted_count else None

    @property
    def schema_rejection_rate(self) -> float | None:
        return self.schema_rejections_total / self.consultations_total if self.consultations_total else None

    @property
    def advice_acceptance_rate(self) -> float | None:
        return self.accepted_count / self.consultations_total if self.consultations_total else None


def new_action_type_counter() -> Counter:
    return Counter()
