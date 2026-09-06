"""Learning-rate schedule semantics (spec section 12)."""

import pytest

from marla.learning.lr_schedule import compute_learning_rate


def test_constant_schedule_ignores_progress():
    for step in (0, 1, 5000, 10000, 999999):
        assert compute_learning_rate("constant", 0.001, step, 10000) == 0.001


def test_linear_schedule_starts_at_the_initial_rate():
    assert compute_learning_rate("linear", 0.001, 0, 10000) == pytest.approx(0.001)


def test_linear_schedule_at_an_intermediate_point():
    assert compute_learning_rate("linear", 0.001, 5000, 10000) == pytest.approx(0.0005)
    assert compute_learning_rate("linear", 0.0004, 2500, 10000) == pytest.approx(0.0003)


def test_linear_schedule_reaches_zero_at_the_end_of_training():
    assert compute_learning_rate("linear", 0.001, 10000, 10000) == pytest.approx(0.0)


def test_linear_schedule_clamps_past_the_budget_rather_than_going_negative():
    assert compute_learning_rate("linear", 0.001, 15000, 10000) == pytest.approx(0.0)
    assert compute_learning_rate("linear", 0.001, 10_000_000, 10000) == pytest.approx(0.0)


def test_linear_schedule_never_negative_anywhere_in_range():
    for step in range(0, 11000, 500):
        assert compute_learning_rate("linear", 0.001, step, 10000) >= 0.0


def test_linear_schedule_requires_a_positive_budget():
    with pytest.raises(ValueError):
        compute_learning_rate("linear", 0.001, 0, 0)
