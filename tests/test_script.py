"""The live scenario is the personas' scenario, and the interlocutor stays narrow.

The strongest check here is the replay one: a persona's answers driven through
the live state machine, with no reactions, must hit the persona's recorded
observer cassettes. A cassette key covers the exact prompt, so a hit proves the
live path asks the observer exactly what the persona path asks.
"""

import glob
from pathlib import Path

import pytest

from probe.config import replay_only
from probe.run import run_transcript
from probe.scenario import interlocutor
from probe.scenario.interlocutor import build_prompt, react
from probe.scenario.script import FACTS, STEPS, LiveSession, leaked_fact
from probe.store import repo
from probe.transcript import dump, load

PERSONAS = sorted(glob.glob("fixtures/personas/*.json"))


def drive(persona_path: str) -> LiveSession:
    persona = load(persona_path)
    session = LiveSession.start(persona.label)
    for turn in persona.participant_turns():
        session.answer(turn.content)
    return session


@pytest.mark.parametrize("path", PERSONAS)
def test_every_persona_follows_the_script(path):
    persona = load(path)
    assistant = [t.content for t in persona.turns if t.role == "assistant"]
    assert assistant == [s.line_ru for s in STEPS]
    session = drive(path)
    assert session.finished
    assert [(t.role, t.content) for t in session.turns] == [
        (t.role, t.content) for t in persona.turns
    ]
    assert session.surfaced == persona.surfaced


def test_a_live_session_asks_the_observer_what_the_persona_asked():
    live = drive("fixtures/personas/strong.json").to_transcript()
    assert live.source == "live"
    with repo.connect() as connection:
        live_run = run_transcript(live, replay_only(), connection)
    with repo.connect() as connection:
        persona_run = run_transcript(
            load("fixtures/personas/strong.json"), replay_only(), connection
        )
    assert live_run.score.expected() == persona_run.score.expected()


def test_the_reaction_joins_the_scripted_line_in_one_turn():
    session = LiveSession.start("t")
    session.answer("А из скольки это 1 240?", "Общего числа клиентов в выгрузке нет.")
    turn = session.turns[-1]
    assert turn.content == "Общего числа клиентов в выгрузке нет.\n\n" + STEPS[1].line_ru
    assert session.surfaced["T2"] == turn.idx


def test_a_reaction_with_a_closed_fact_is_dropped_and_kept_for_audit():
    session = LiveSession.start("t")
    session.answer("Что ещё менялось?", "Ещё отключили пуш-уведомления.")
    assert session.turns[-1].content == STEPS[1].line_ru
    assert len(session.dropped) == 1
    assert "push" in session.dropped[0].fact


def test_the_price_marker_does_not_fire_on_ordinary_words():
    assert leaked_fact("Оценка в процентах неточна, сценарий тот же.", answered=0) is None
    assert leaked_fact("Тогда же подняли цену.", answered=2) is not None


def test_a_traps_own_fact_is_confirmed_never_volunteered():
    """The author's first live session: the price rise, offered in reply to «что
    ещё могло случиться?», later scored T3 as caught. design/00 §3."""
    said = "Тогда же подняли цену подписки."
    assert leaked_fact(said, answered=5, prompt="Что ещё могло случиться?") is not None
    assert leaked_fact(said, answered=5, prompt="А цену тогда не меняли?") is None
    holiday = "Это праздничная неделя."
    assert leaked_fact(holiday, answered=1, prompt="В чём вопрос?") is not None
    assert leaked_fact(holiday, answered=1, prompt="Это же праздники?") is None


def test_confirm_only_facts_sit_in_their_own_section():
    session = LiveSession.start("t")
    for answer in ["a", "b", "c", "d", "e"]:
        session.answer(answer)
    system, _ = build_prompt(session, "f")
    tell, confirm = system.split("ФАКТЫ, КОТОРЫЕ МОЖНО ТОЛЬКО ПОДТВЕРДИТЬ")
    for fact in FACTS:
        assert (fact.text_ru in confirm) == fact.confirm_only
        assert (fact.text_ru in tell) != fact.confirm_only


def test_the_script_refuses_what_is_not_a_turn():
    session = LiveSession.start("t")
    with pytest.raises(ValueError):
        session.answer("   ")
    finished = drive("fixtures/personas/weak.json")
    with pytest.raises(RuntimeError):
        finished.answer("ещё")


def test_the_interlocutor_sees_only_facts_already_open():
    session = LiveSession.start("t")
    system, user = build_prompt(session, "Что ещё менялось?")
    assert "push" not in system and "цену" not in system
    assert STEPS[0].line_ru in user
    for answer in ["a", "b", "c", "d", "e"]:
        session.answer(answer)
    system, _ = build_prompt(session, "f")
    assert all(f.text_ru in system for f in FACTS)


def test_the_interlocutor_passes_its_reply_through(monkeypatch):
    monkeypatch.setattr(interlocutor, "complete_json", lambda r, c: {"reply": "  Данных нет. "})
    assert react(LiveSession.start("t"), "Из скольки?", replay_only()) == "Данных нет."


def test_nothing_follows_the_final_answer(monkeypatch):
    def unreachable(request, config):
        raise AssertionError("the last step needs no reaction")

    monkeypatch.setattr(interlocutor, "complete_json", unreachable)
    session = LiveSession.start("t")
    for answer in ["a", "b", "c", "d", "e", "f"]:
        session.answer(answer)
    assert react(session, "Нет.", replay_only()) == ""


def test_a_saved_live_session_loads_back(tmp_path: Path):
    session = drive("fixtures/personas/passive.json")
    path = tmp_path / "s.json"
    dump(session.to_transcript(), path, audit={"dropped_reactions": []})
    loaded = load(path)
    assert loaded.source == "live"
    assert loaded.turns == session.to_transcript().turns
    assert loaded.surfaced == session.surfaced


def test_a_provider_failure_costs_the_reaction_not_the_session():
    session = LiveSession.start("t")
    session.answer("Из скольки?", "этот текст не должен попасть", failure="non-JSON")
    assert session.turns[-1].content == STEPS[1].line_ru
    assert session.failures == ["ход 2: non-JSON"]


def test_a_live_session_replays_from_its_own_cassettes(tmp_path: Path, monkeypatch):
    """Recorded into the live folder when scored, replayed from it offline. The
    provider is a stand-in: what is tested is where the cassette lands and that
    replay finds it there, not what any model says."""
    import dataclasses

    from probe import llm
    from probe.observe.observer import observe_turn

    monkeypatch.setattr(
        llm, "_call_provider", lambda request, config: ({"observations": []}, "stub", "")
    )
    live = dataclasses.replace(replay_only(tmp_path), mode="live", record=True)
    transcript = drive("fixtures/personas/strong.json").to_transcript()
    observe_turn(transcript, 1, live)
    assert list(tmp_path.glob("observer-*.json"))
    assert observe_turn(transcript, 1, replay_only(tmp_path)) == []
