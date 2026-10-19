"""The two independence guards, and the one-observation-one-node rule."""

import pytest

from probe.score.aggregate import (
    MISSED_CONFIDENCE,
    Observation,
    build_evidence,
    dedupe_within_turn,
    rollup_indicators,
    rollup_tasks,
    to_evidence_map,
)


def obs(indicator, turn, strength="weak", confidence=0.7, task=None, quote="q"):
    return Observation(indicator, turn, strength, confidence, quote, task)


def test_unknown_indicator_is_rejected_at_construction():
    with pytest.raises(ValueError, match="unknown indicator"):
        obs("kn.does_not_exist", 1)


def test_repeated_manifestations_in_one_turn_collapse_to_one():
    kept = dedupe_within_turn([obs("kn.gap", 1), obs("kn.gap", 1), obs("kn.gap", 1)])
    assert len(kept) == 1


def test_strongest_wins_within_a_turn_then_confidence():
    kept = dedupe_within_turn(
        [obs("kn.gap", 1, "weak", 0.9), obs("kn.gap", 1, "strong", 0.6)]
    )
    assert kept[0].strength == "strong"
    kept = dedupe_within_turn(
        [obs("kn.gap", 1, "weak", 0.5), obs("kn.gap", 1, "weak", 0.8)]
    )
    assert kept[0].confidence == 0.8


def test_one_reply_exposing_two_tasks_keeps_both():
    """Found on the first live run: `strong` names both the missing denominator
    (T4) and the survivor sample (T1) in turn 1, both as kn.gap. Collapsing
    them scored T1 as missed."""
    kept = dedupe_within_turn(
        [obs("kn.gap", 1, "strong", 0.95, task="T4"), obs("kn.gap", 1, "strong", 0.9, task="T1")]
    )
    assert {o.task_id for o in kept} == {"T4", "T1"}
    evidence = build_evidence(list(kept), ["T1", "T4"])
    assert {e.node: e.state for e in evidence} == {"task:T1": "caught", "task:T4": "caught"}


def test_positive_and_negative_in_one_turn_both_survive():
    """Different nodes. A person can ask a sharp question and parrot in one breath."""
    kept = dedupe_within_turn([obs("ag.own", 1), obs("ag.parrot", 1)])
    assert {o.indicator_id for o in kept} == {"ag.own", "ag.parrot"}


def test_one_indicator_is_one_node_for_the_whole_session():
    evidence = rollup_indicators([obs("kn.gap", 1), obs("kn.gap", 4), obs("kn.gap", 9)])
    assert len(evidence) == 1
    assert evidence[0].state == "strong"  # fired across turns
    assert len(evidence[0].sources) == 3


def test_a_single_low_confidence_hit_stays_weak():
    evidence = rollup_indicators([obs("kn.gap", 1, "strong", 0.5)])
    assert evidence[0].state == "weak"


def test_one_decisive_hit_is_enough_for_strong():
    evidence = rollup_indicators([obs("kn.gap", 1, "strong", 0.9)])
    assert evidence[0].state == "strong"


def test_independent_turns_raise_confidence_but_never_to_certainty():
    one = rollup_indicators([obs("kn.gap", 1, "weak", 0.6)])[0].confidence
    two = rollup_indicators([obs("kn.gap", 1, "weak", 0.6), obs("kn.gap", 5, "weak", 0.6)])[
        0
    ].confidence
    assert one < two < 1.0


def test_absence_never_becomes_evidence():
    """No observation -> no node. 'none' is never submitted."""
    assert rollup_indicators([]) == ()
    assert all(e.state != "none" for e in rollup_indicators([obs("kn.gap", 1)]))


def test_surfaced_task_with_nothing_detected_is_missed():
    evidence = rollup_tasks([], surfaced=["T7"])
    assert evidence[0].state == "missed"
    assert evidence[0].confidence == MISSED_CONFIDENCE
    assert evidence[0].sources == ()


def test_unsurfaced_task_produces_no_evidence_at_all():
    assert rollup_tasks([], surfaced=[]) == ()


def test_task_outcome_grades_on_strength_and_count():
    assert rollup_tasks([obs("kn.gap", 1, "weak", 0.7, task="T1")], ["T1"])[0].state == "partial"
    assert rollup_tasks([obs("kn.gap", 1, "strong", 0.7, task="T1")], ["T1"])[0].state == "caught"
    two = [obs("kn.gap", 1, "weak", 0.7, task="T1"), obs("kn.scope", 2, "weak", 0.7, task="T1")]
    assert rollup_tasks(two, ["T1"])[0].state == "caught"


def test_a_negative_indicator_on_a_task_marks_it_missed():
    evidence = rollup_tasks([obs("da.surface", 3, "strong", 0.9, task="T5")], ["T5"])
    assert evidence[0].state == "missed"
    assert evidence[0].confidence > MISSED_CONFIDENCE


def test_mixed_signals_on_one_task_read_as_partial():
    mixed = [
        obs("da.evidence", 3, "strong", 0.8, task="T5"),
        obs("da.surface", 4, "weak", 0.6, task="T5"),
    ]
    assert rollup_tasks(mixed, ["T5"])[0].state == "partial"


def test_observation_referencing_a_task_implies_it_was_surfaced():
    """Trusting the scenario tracker over the evidence would drop real signal."""
    evidence = rollup_tasks([obs("kn.gap", 1, "strong", 0.8, task="T4")], surfaced=[])
    assert evidence[0].node == "task:T4"


def test_every_observation_feeds_exactly_one_node():
    """A task-bound observation must not also move its indicator node: both are
    children of the same construct, so it would count the behaviour twice."""
    observations = [
        obs("kn.gap", 1, "strong", 0.9, task="T1"),
        obs("ag.own", 2, "weak", 0.7),
    ]
    evidence = build_evidence(observations, surfaced=["T1"])
    nodes = {e.node for e in evidence}
    assert "kn.gap" not in nodes
    assert nodes == {"task:T1", "ag.own"}
    assert sum(len(e.sources) for e in evidence) == len(observations)


def test_evidence_map_shape_matches_node_states():
    evidence = build_evidence([obs("kn.gap", 1, "strong", 0.9, task="T1")], ["T1", "T7"])
    mapped = to_evidence_map(evidence)
    assert len(mapped["task:T1"]) == 3
    assert sum(mapped["task:T1"]) == pytest.approx(1.0)
