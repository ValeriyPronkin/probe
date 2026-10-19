"""The generated tables must stay valid distributions and keep their ordering."""

import pytest

from probe.inventory.constructs import LEVELS
from probe.inventory.indicators import INDICATORS_BY_ID
from probe.scenario.tasks import TASKS_BY_ID
from probe.score.cpt import (
    construct_given_disposition,
    graded_response,
    indicator_cpt,
    task_cpt,
)


def test_graded_response_is_a_distribution():
    for theta in (0.0, 0.25, 0.5, 0.75, 1.0):
        dist = graded_response(theta, 4.0, (0.4, 0.7))
        assert len(dist) == 3
        assert all(p >= 0 for p in dist)
        assert sum(dist) == pytest.approx(1.0)


def test_graded_response_is_monotone_in_ability():
    low = graded_response(0.2, 4.0, (0.4, 0.7))
    high = graded_response(0.9, 4.0, (0.4, 0.7))
    assert high[-1] > low[-1]
    assert high[0] < low[0]


def test_unordered_thresholds_are_rejected():
    """Reversed thresholds silently produce a negative probability for the
    middle state, so they must fail loudly."""
    with pytest.raises(ValueError, match="ascending"):
        graded_response(0.5, 4.0, (0.7, 0.4))


def test_extreme_ability_does_not_overflow():
    assert sum(graded_response(1.0, 1e6, (0.4, 0.7))) == pytest.approx(1.0)
    assert sum(graded_response(0.0, 1e6, (0.4, 0.7))) == pytest.approx(1.0)


def test_disposition_table_is_diagonally_dominant():
    table = construct_given_disposition(0.55)
    for i, level in enumerate(LEVELS):
        row = table[(level,)]
        assert sum(row) == pytest.approx(1.0)
        assert row[i] == max(row)


def test_negative_indicator_inverts_the_scale():
    """A negative indicator must be likelier at low ability, not high."""
    negative = indicator_cpt(INDICATORS_BY_ID["ag.parrot"])
    assert negative[("low",)][-1] > negative[("high",)][-1]
    positive = indicator_cpt(INDICATORS_BY_ID["ag.own"])
    assert positive[("high",)][-1] > positive[("low",)][-1]


def test_two_parent_task_combines_compensatorily():
    """One strong construct must partly cover a weak one — otherwise the task
    measures the product of two constructs and credits it to each."""
    table = task_cpt(TASKS_BY_ID["T1"])
    mixed = table[("high", "low")][-1]
    both_low = table[("low", "low")][-1]
    both_high = table[("high", "high")][-1]
    assert both_low < mixed < both_high


def test_every_task_and_indicator_table_is_valid():
    for task in TASKS_BY_ID.values():
        for row in task_cpt(task).values():
            assert sum(row) == pytest.approx(1.0) and all(p >= 0 for p in row)
    for indicator in INDICATORS_BY_ID.values():
        for row in indicator_cpt(indicator).values():
            assert sum(row) == pytest.approx(1.0) and all(p >= 0 for p in row)
