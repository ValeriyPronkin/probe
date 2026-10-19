"""The live app opens on the first scripted step without calling anything.

Only the opening render is tested: every later step calls the interlocutor, and
a test that reaches a provider is not a test. The state machine behind those
steps is covered in test_script.py.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from probe.scenario.script import STEPS  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "scripts" / "live.py"


def test_the_live_app_opens_on_the_first_step():
    app = AppTest.from_file(str(APP), default_timeout=30).run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.chat_message[0].markdown[0].value == STEPS[0].line_ru


def test_a_scored_live_session_shows_the_score_and_the_protocol():
    """The result screen, fed a session and a run built from cassettes."""
    from probe.config import replay_only
    from probe.run import run_transcript
    from probe.scenario.script import LiveSession
    from probe.store import repo
    from probe.transcript import load

    persona = load("fixtures/personas/strong.json")
    session = LiveSession.start("t")
    for turn in persona.participant_turns():
        session.answer(turn.content)
    with repo.connect() as connection:
        run = run_transcript(session.to_transcript(), replay_only(), connection)

    app = AppTest.from_file(str(APP), default_timeout=30)
    app.session_state["session"] = session
    app.session_state["run"] = run
    # Pretend it was saved: the test must not write into fixtures/transcripts.
    app.session_state["saved"] = APP.parent.parent / "fixtures" / "transcripts" / "unsaved.json"
    app.run()
    assert not app.exception, [e.value for e in app.exception]
    assert [t.label for t in app.tabs] == ["Оценка", "Протокол сессии", "Текстовый отчёт"]
    assert any("засчитано" in m.value for m in app.markdown)
