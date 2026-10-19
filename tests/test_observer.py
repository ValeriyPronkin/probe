"""The observer's post-hoc validation and its health telemetry."""

import dataclasses
import json

import pytest

from probe.config import Config, replay_only
from probe.inventory.render import render_inventory, valid_indicator_ids
from probe.llm import (
    CassetteMissing,
    IncompleteLabelling,
    Request,
    cassette_path,
    check_finished,
    complete_json,
)
from probe.models import OBSERVER
from probe.observe import observer as observer_module
from probe.observe.observer import Telemetry, build_prompt, observe_turn
from probe.transcript import Transcript, Turn


def transcript() -> Transcript:
    return Transcript(
        label="t",
        source="persona",
        scenario_id="churn-v1",
        turns=(
            Turn(0, "assistant", "Ушло 1240 человек."),
            Turn(1, "participant", "А из скольки это 1240?"),
        ),
        surfaced={"T4": 0},
    )


def respond_with(observations):
    def fake(request, config, cassette_dir=None):
        return {"observations": observations}

    return fake


def observation(indicator_id, task_id=""):
    return {
        "indicator_id": indicator_id,
        "quote": "А из скольки это 1240?",
        "strength": "strong",
        "confidence": 0.9,
        "task_id": task_id,
        "rationale": "",
    }


def test_the_prompt_carries_the_whole_labelled_inventory():
    """An observer without the inventory does not refuse — it invents ids, every
    one of them dies at validation, and a rich turn records nothing."""
    system, user = build_prompt(transcript(), 1)
    assert render_inventory() in system
    for indicator_id in valid_indicator_ids():
        assert indicator_id in system
    assert "А из скольки это 1240?" in user


def test_only_the_participant_turn_is_offered_for_tagging():
    _, user = build_prompt(transcript(), 1)
    assert "не размечать" in user


def test_an_invented_id_is_dropped_and_counted(monkeypatch):
    monkeypatch.setattr(
        observer_module, "complete_json", respond_with([observation("kn.punctuation")])
    )
    telemetry = Telemetry()
    kept = observe_turn(transcript(), 1, replay_only(), telemetry)
    assert kept == []
    assert telemetry.dropped_unknown_indicator == 1
    assert telemetry.unknown_ids == ["kn.punctuation"]
    assert telemetry.drop_rate == 1.0


def test_a_bad_task_link_drops_the_link_not_the_evidence(monkeypatch):
    monkeypatch.setattr(
        observer_module, "complete_json", respond_with([observation("kn.gap", task_id="T99")])
    )
    telemetry = Telemetry()
    kept = observe_turn(transcript(), 1, replay_only(), telemetry)
    assert len(kept) == 1
    assert kept[0].task_id == ""
    assert telemetry.dropped_unknown_task == 1


def test_a_malformed_observation_costs_itself_not_the_turn(monkeypatch):
    """GigaChat, 2026-10-19: strength "medium". Not coerced to a valid value."""
    medium = {**observation("kn.gap"), "strength": "medium"}
    monkeypatch.setattr(
        observer_module, "complete_json", respond_with([medium, observation("kn.gap")])
    )
    telemetry = Telemetry()
    kept = observe_turn(transcript(), 1, replay_only(), telemetry)
    assert [o.strength for o in kept] == ["strong"]
    assert telemetry.dropped_malformed == 1
    assert telemetry.emitted == 2


def test_a_bare_list_is_read_but_counted(monkeypatch):
    """GigaChat, 2026-10-19: `[]` instead of {"observations": []}."""
    monkeypatch.setattr(
        observer_module, "complete_json", lambda request, config, cassette_dir=None: []
    )
    telemetry = Telemetry()
    assert observe_turn(transcript(), 1, replay_only(), telemetry) == []
    assert telemetry.envelope_broken == 1
    assert telemetry.turns_empty == 1


def test_an_empty_answer_is_a_valid_answer(monkeypatch):
    monkeypatch.setattr(observer_module, "complete_json", respond_with([]))
    telemetry = Telemetry()
    assert observe_turn(transcript(), 1, replay_only(), telemetry) == []
    assert telemetry.turns_empty == 1
    assert telemetry.drop_rate == 0.0


# ── cassettes ────────────────────────────────────────────────────────────────

def request() -> Request:
    return Request(
        role=OBSERVER, system="s", user="u", schema_name="observations", schema={"type": "object"}
    )


def test_replay_without_a_cassette_fails_loudly_instead_of_calling_out(tmp_path):
    with pytest.raises(CassetteMissing, match="PROBE_RECORD"):
        complete_json(request(), replay_only(), cassette_dir=tmp_path)


def test_a_recorded_cassette_replays_exactly(tmp_path):
    path = cassette_path(request(), tmp_path)
    path.write_text(json.dumps({"response": {"observations": []}}), encoding="utf-8")
    assert complete_json(request(), replay_only(), cassette_dir=tmp_path) == {"observations": []}


def test_changing_the_prompt_changes_the_cassette_key():
    """A changed prompt must not silently reuse an old recording."""
    other = Request(
        role=OBSERVER, system="s2", user="u", schema_name="observations", schema={"type": "object"}
    )
    assert request().key() != other.key()


def test_tests_cannot_reach_a_provider():
    config = replay_only()
    assert config.mode == "replay"
    for provider in ("openrouter", "deepseek"):
        with pytest.raises(RuntimeError, match="PROBE_MODE=live"):
            Config(mode="live", db_path=":memory:", record=False).require_key(provider)


def test_the_same_model_behind_another_provider_is_another_cassette():
    moved = Request(
        role=dataclasses.replace(OBSERVER, provider="openrouter"),
        system="s",
        user="u",
        schema_name="observations",
        schema={"type": "object"},
    )
    assert request().key() != moved.key()


@pytest.mark.parametrize(
    "finish_reason, content",
    [("length", '{"observations": [{"indicator_id": "kn.ga'), ("blacklist", "Не люблю менять тему"),
     ("stop", "Вот разметка: ...")],
)
def test_an_unfinished_answer_is_an_error_not_an_empty_one(finish_reason, content):
    with pytest.raises(IncompleteLabelling):
        check_finished("observer", finish_reason, content)


def test_a_finished_answer_parses():
    assert check_finished("observer", "stop", '{"observations": []}') == {"observations": []}


def test_a_failed_turn_leaves_no_session_behind(monkeypatch):
    """Stored complete or not at all: no score built on part of the evidence."""
    from probe.run import run_transcript
    from probe.store import repo

    def fail(*args, **kwargs):
        raise IncompleteLabelling("observer: cut off")

    monkeypatch.setattr("probe.run.observe_turn", fail)
    with repo.connect() as connection:
        with pytest.raises(IncompleteLabelling):
            run_transcript(transcript(), replay_only(), connection)
        assert connection.execute("SELECT COUNT(*) FROM sessions").fetchone()[0] == 0
