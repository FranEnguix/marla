"""Evaluation-time-only tooling for checkpoint ablations and cross-scenario
generalization testing (research/aamas2027).

Everything in this package operates on a *frozen* :class:`RecurrentPolicy`:
no optimizer is ever constructed here, no gradient is ever taken, and
nothing here is imported by :mod:`marla.learning.trainer` or any real
``marla run`` code path. See ``research/aamas2027/AUDIT.md`` and
``VALIDATION.md`` for why this is a separate package rather than new
branches in the training loop itself.
"""
