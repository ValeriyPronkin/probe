"""The demo app renders the roster and every participant from cassettes without raising.

Skipped when the optional `web` extra is not installed: the instrument does not
depend on its demo surface, and a plain `uv sync` does not pull Streamlit in.
"""

from pathlib import Path

import pytest

pytest.importorskip("streamlit")

from streamlit.testing.v1 import AppTest  # noqa: E402

from probe.config import LIVE_DIR  # noqa: E402

APP = Path(__file__).resolve().parent.parent / "scripts" / "app.py"
PERSONAS = ["strong", "polished", "contrarian", "gaming", "passive", "weak"]
PERSONA_TITLES = {
    "strong": "Сильный",
    "polished": "Гладкий",
    "contrarian": "Спорщик",
    "gaming": "Игрок",
    "passive": "Пассивный",
    "weak": "Слабый",
}
LIVE = "live-Valeriy-20261013-100541.json"
# Live sessions are gitignored (they hold a person's words), so a fresh clone has none.
needs_live = pytest.mark.skipif(
    not (LIVE_DIR / LIVE).exists(), reason="live session is local-only, not in git"
)


@pytest.fixture(scope="module")
def app():
    return AppTest.from_file(str(APP), default_timeout=60).run()


def open_participant(app, key):
    """Through the page's own picker, the way a person switches participants."""
    app.switch_page("app_pages/participant.py").run()
    return app.selectbox(key="pick").set_value(key).run()


@pytest.mark.parametrize("page", ["app_pages/about.py"])
def test_the_framing_pages_render(app, page):
    app.switch_page(page).run()
    assert not app.exception, [e.value for e in app.exception]


def test_a_new_measurement_opens_inside_the_demo(app):
    """Opening the page calls nothing: the interlocutor only runs on an answer."""
    from probe.scenario.script import STEPS

    app.switch_page("app_pages/live.py").run()
    assert not app.exception, [e.value for e in app.exception]
    assert app.chat_message[0].markdown[0].value == STEPS[0].line_ru


def test_the_roster_lists_every_participant(app):
    app.switch_page("app_pages/roster.py").run()
    assert not app.exception, [e.value for e in app.exception]
    keys = list(app.dataframe[0].value.index)
    assert keys[: len(PERSONAS)] == PERSONAS
    if (LIVE_DIR / LIVE).exists():
        assert LIVE in keys


@pytest.mark.parametrize("observer", ["model", "gold"])
def test_every_persona_opens(app, observer):
    app.radio(key="observer").set_value(observer).run()
    for persona in PERSONAS:
        open_participant(app, persona)
        assert not app.exception, (observer, persona, [e.value for e in app.exception])
        assert app.title[0].value.endswith(PERSONA_TITLES[persona])
        assert [t.label for t in app.tabs] == ["Оценка", "Диалог и доказательства", "Источники"]


@needs_live
def test_a_live_session_opens_offline(app):
    open_participant(app, LIVE)
    assert not app.exception, [e.value for e in app.exception]
    assert app.title[0].value == "Живая сессия: Valeriy"
    assert any("засчитано" in m.value for m in app.markdown)


def test_the_sensitivity_slider_matches_the_sweep_script(app):
    """At 0.5 the gold-label sweep in design/02-validation.md reads 0.64 / 0.16."""
    app.switch_page("app_pages/roster.py").run()
    app.radio(key="observer").set_value("gold").run()
    app.slider(key="missed").set_value(0.5).run()
    assert [m.value for m in app.metric][:2] == ["0.64", "0.16"]
