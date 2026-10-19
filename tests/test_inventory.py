"""Inventory invariants. These run at import, so the tests document them."""

import importlib

import pytest

from probe.inventory.constructs import CONSTRUCT_IDS
from probe.inventory.indicators import ACTIVE_INDICATORS, INDICATORS, indicators_for
from probe.scenario.tasks import TASKS, tasks_eliciting


def test_every_construct_has_both_polarities():
    """Without a negative indicator a construct can only ever go up."""
    for construct in CONSTRUCT_IDS:
        polarities = {i.polarity for i in indicators_for(construct)}
        assert polarities == {"positive", "negative"}, construct


def test_every_construct_has_at_least_one_task():
    """A construct with no task is observable only by luck and never produces
    negative evidence (design/00 §3)."""
    covered = {c for task in TASKS for c in task.constructs}
    assert set(CONSTRUCT_IDS) <= covered


def test_every_indicator_carries_both_examples():
    """The near-miss example is what stops the observer drifting; an indicator
    without one is not operationalised."""
    for indicator in INDICATORS:
        assert indicator.counts_ru.strip()
        assert indicator.not_counts_ru.strip()
        assert indicator.definition_ru.strip()


def test_indicator_ids_are_namespaced_by_construct():
    for indicator in INDICATORS:
        assert indicator.id.startswith(f"{indicator.construct}.")


def test_task_elicited_indicators_exist_and_point_somewhere_useful():
    for task in TASKS:
        assert task.elicits, task.id
        for indicator_id in task.elicits:
            assert any(i.id == indicator_id for i in ACTIVE_INDICATORS)


def test_at_least_one_task_elicits_each_negative_indicator_of_its_construct():
    """Negative evidence comes almost entirely from tasks, so a negative
    indicator no task can elicit is nearly unobservable."""
    orphans = [
        i.id
        for i in ACTIVE_INDICATORS
        if i.polarity == "negative" and not tasks_eliciting(i.id)
    ]
    assert orphans == [], f"negative indicators no task elicits: {orphans}"


def test_module_level_validation_rejects_a_broken_inventory(monkeypatch):
    """The import-time guards are the real protection; prove they bite."""
    import probe.inventory.indicators as module

    broken = module.Indicator(
        id="zz.bogus",
        construct="zz",
        polarity="positive",
        definition_ru="x",
        counts_ru="x",
        not_counts_ru="x",
    )
    monkeypatch.setattr(module, "INDICATORS", (*module.INDICATORS, broken))
    monkeypatch.setattr(module, "INDICATORS_BY_ID", {i.id: i for i in module.INDICATORS})
    monkeypatch.setattr(
        module, "ACTIVE_INDICATORS", tuple(i for i in module.INDICATORS if not i.deprecated)
    )
    with pytest.raises(ValueError, match="unknown construct"):
        module._validate()
