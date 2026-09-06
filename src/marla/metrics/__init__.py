"""Run metrics: writing (spec section 21) and summarizing (``marla summarize``).

Deliberately does not eagerly import ``marla.metrics.writer`` here:
``writer`` imports ``TrainingResult`` from ``marla.learning.trainer``, which
itself imports ``marla.metrics.accumulators`` -- an eager
``from marla.metrics.writer import ...`` at package-init time would make
that a circular import. Import ``marla.metrics.writer`` directly instead of
relying on this package re-exporting it.
"""
