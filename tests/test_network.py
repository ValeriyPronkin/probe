"""Properties of the assembled scoring network."""

import pytest

from probe.inventory.constructs import CONSTRUCT_IDS, LEVEL_VALUES
from probe.inventory.indicators import ACTIVE_INDICATORS
from probe.scenario.tasks import TASKS
from probe.score.aggregate import Observation, build_evidence, to_evidence_map
from probe.score.bayes import expected_level, hard_evidence, posterior
from probe.score.network import default_network, task_node_name

TASK_STATES = ("missed", "partial", "caught")


def expected(evidence_map):
    marginals = posterior(default_network(), evidence_map)
    return {c: expected_level(marginals[c], LEVEL_VALUES) for c in CONSTRUCT_IDS}


def test_every_indicator_and_task_has_a_node():
    names = {n.name for n in default_network().nodes}
    assert {i.id for i in ACTIVE_INDICATORS} <= names
    assert {task_node_name(t.id) for t in TASKS} <= names


def test_an_empty_session_returns_the_prior_and_says_nothing():
    assert all(value == pytest.approx(0.5) for value in expected({}).values())


def test_catching_a_task_raises_its_constructs_most():
    scores = expected({task_node_name("T1"): hard_evidence(TASK_STATES, "caught")})
    assert scores["kn"] > scores["ss"]
    assert scores["mf"] > scores["ss"]


def test_cross_construct_bleed_stays_small():
    """One observation may pull unrelated constructs through the shared
    disposition, but not far — otherwise the instrument invents scores.
    See DISPOSITION_STRENGTH in inventory/constructs.py."""
    scores = expected({task_node_name("T1"): hard_evidence(TASK_STATES, "caught")})
    assert scores["ss"] - 0.5 < 0.08


def test_missing_a_surfaced_task_lowers_its_construct():
    scores = expected({task_node_name("T7"): hard_evidence(TASK_STATES, "missed")})
    assert scores["ss"] < 0.5


def test_a_task_never_surfaced_leaves_its_construct_untouched():
    """The whole point of the task model: silence is not a low score."""
    assert expected({})["ss"] == pytest.approx(0.5)


def test_negative_indicators_lower_their_construct():
    evidence = build_evidence([Observation("ag.parrot", 1, "strong", 0.9, "…")], surfaced=[])
    assert expected(to_evidence_map(evidence))["ag"] < 0.5


def test_more_consistent_evidence_moves_further():
    one = expected({task_node_name("T2"): hard_evidence(TASK_STATES, "caught")})["da"]
    two = expected(
        {
            task_node_name("T2"): hard_evidence(TASK_STATES, "caught"),
            task_node_name("T3"): hard_evidence(TASK_STATES, "caught"),
        }
    )["da"]
    assert two > one
