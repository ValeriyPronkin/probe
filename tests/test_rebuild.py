"""The rebuild invariant: a stored score is the fold of the stored rows.

Hold this and a parameter change means rescoring history; lose it and every past
session becomes a number nobody can check.
"""

import pytest

from probe.inventory.constructs import CONSTRUCT_IDS
from probe.score.aggregate import Observation, Thresholds
from probe.score.bayes import Network, Node
from probe.score.network import build_network, default_network
from probe.score.session import (
    parameters_fingerprint,
    rebuild_matches_stored,
    score_and_store,
    score_session,
)
from probe.store import repo


def session_with_evidence(connection, session_id="s1"):
    repo.create_session(
        connection,
        repo.SessionMeta(
            id=session_id,
            participant_label="p-01",
            source="persona",
            scenario_id="churn-v1",
            observer_model="observer-x",
            judge_model="judge-y",
            prompt_version="v1",
        ),
    )
    repo.append_turn(connection, session_id, 0, "assistant", "Отток вырос на 1240 человек.")
    repo.append_turn(connection, session_id, 1, "participant", "А из скольки? Считали по активным?")
    repo.append_turn(connection, session_id, 2, "assistant", "По отраслевому бенчмарку 23%.")
    repo.append_turn(connection, session_id, 3, "participant", "Откуда 23%? Источник назови.")
    repo.record_surfaced(connection, session_id, "T4", 0)
    repo.record_surfaced(connection, session_id, "T5", 2)
    repo.record_surfaced(connection, session_id, "T7", 2)
    repo.record_observation(
        connection,
        session_id,
        Observation("kn.gap", 1, "strong", 0.9, "А из скольки?", "T4"),
        observer_model="observer-x",
        prompt_version="v1",
        verdicts=[("grounding", True, ""), ("attribution", True, "")],
    )
    repo.record_observation(
        connection,
        session_id,
        Observation("da.evidence", 3, "strong", 0.85, "Откуда 23%?", "T5"),
        observer_model="observer-x",
        prompt_version="v1",
        verdicts=[("grounding", True, ""), ("attribution", True, "")],
    )
    return session_id


@pytest.fixture
def connection():
    with repo.connect() as connection:
        yield connection


def test_stored_score_equals_the_recomputed_fold(connection):
    session_id = session_with_evidence(connection)
    score_and_store(connection, session_id)
    assert rebuild_matches_stored(connection, session_id)


def test_a_session_with_no_stored_score_does_not_claim_to_match(connection):
    session_id = session_with_evidence(connection)
    assert not rebuild_matches_stored(connection, session_id)


def test_inadmissible_observations_never_enter_the_fold(connection):
    session_id = session_with_evidence(connection)
    before = score_session(connection, session_id).expected()
    repo.record_observation(
        connection,
        session_id,
        Observation("ag.parrot", 3, "strong", 0.95, "фраза из ответа ИИ"),
        observer_model="observer-x",
        prompt_version="v1",
        verdicts=[("grounding", True, ""), ("attribution", False, "6-gram overlap with turn 2")],
    )
    after = score_session(connection, session_id).expected()
    assert before == after


def test_rejected_observations_are_kept_with_their_reason(connection):
    """Drop rate is the instrument's health metric; deleting rejects would hide
    a degrading prompt."""
    session_id = session_with_evidence(connection)
    repo.record_observation(
        connection,
        session_id,
        Observation("ag.parrot", 3, "strong", 0.95, "фраза из ответа ИИ"),
        observer_model="observer-x",
        prompt_version="v1",
        verdicts=[("attribution", False, "6-gram overlap with turn 2")],
    )
    rejected = repo.rejected_observations(connection, session_id)
    assert len(rejected) == 1
    assert rejected[0]["check_name"] == "attribution"
    assert "overlap" in rejected[0]["detail"]


def test_changing_the_parameters_invalidates_the_stored_score(connection):
    """The point of the fingerprint: a stale score must be detectable, not
    quietly authoritative."""
    session_id = session_with_evidence(connection)
    score_and_store(connection, session_id)
    assert rebuild_matches_stored(connection, session_id)

    stored, fingerprint = repo.load_scores(connection, session_id)
    repo.save_scores(connection, session_id, stored, "not-the-real-fingerprint")
    assert not rebuild_matches_stored(connection, session_id)


def test_rescoring_under_new_parameters_needs_no_new_session(connection):
    """The demonstration this whole layer exists for."""
    session_id = session_with_evidence(connection)
    original = score_and_store(connection, session_id)

    harsher = Network(
        [
            node
            if node.name != "disposition"
            else Node("disposition", node.states, (), {(): (0.5, 0.4, 0.1)})
            for node in build_network().nodes
        ]
    )
    rescored = score_session(connection, session_id, harsher)

    assert rescored.fingerprint != original.fingerprint
    assert rescored.expected()["kn"] < original.expected()["kn"]
    # Same stored rows, different parameters: nobody was asked to repeat anything.
    assert len(repo.admitted_observations(connection, session_id)) == 2


def test_rollup_thresholds_are_part_of_the_fingerprint():
    """MISSED_CONFIDENCE moves every negative conclusion. A fingerprint that
    covered only the CPTs would let a score computed under another value pass
    as current."""
    network = default_network()
    assert parameters_fingerprint(network) != parameters_fingerprint(
        network, Thresholds(missed_confidence=0.6)
    )


def test_fingerprint_is_stable_across_rebuilds():
    assert parameters_fingerprint(build_network()) == parameters_fingerprint(default_network())


def test_scores_cover_every_construct(connection):
    session_id = session_with_evidence(connection)
    score = score_and_store(connection, session_id)
    assert set(score.posteriors) == set(CONSTRUCT_IDS)
    stored, _ = repo.load_scores(connection, session_id)
    assert set(stored) == set(CONSTRUCT_IDS)
