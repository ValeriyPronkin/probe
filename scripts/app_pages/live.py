"""A new measurement: a person goes through the scripted scenario, then gets scored.

The one page of the demo that reaches a provider. It reads .env, forces
PROBE_MODE=live and needs DEEPSEEK_API_KEY; every other page stays replay-only.
It always records cassettes, into fixtures/transcripts/cassettes/, so a scored
session then opens under «Испытуемые» like any other, offline.

The transcript is written to fixtures/transcripts/, which git ignores: these are
a person's own words. Rescore it later with

    PROBE_MODE=live uv run scripts/replay.py fixtures/transcripts/<file>.json --observer model

Also runs on its own as `scripts/live.py`.
"""

from __future__ import annotations

import dataclasses
import re
from datetime import datetime
from pathlib import Path

import streamlit as st

from probe import config as config_module
from probe.config import ROOT
from probe.models import INTERLOCUTOR, INTERLOCUTOR_PROMPT_VERSION, PROMPT_VERSION
from probe.report.build import render_text
from probe.run import run_transcript
from probe.scenario.interlocutor import react
from probe.scenario.script import CLOSING_RU, STEPS, LiveSession
from probe.store import repo
from probe.transcript import dump
from scripts.views import render_protocol, render_scores

TRANSCRIPTS = config_module.LIVE_DIR


def live_config() -> config_module.Config:
    """Always records, into the live sessions' own cassette folder: a scored live
    session can then be opened in the offline demo exactly as it was scored."""
    return dataclasses.replace(
        config_module.load(),
        mode="live",
        record=True,
        cassette_dir=config_module.LIVE_CASSETTE_DIR,
    )


def save(session: LiveSession) -> Path:
    slug = re.sub(r"\W+", "-", session.label).strip("-") or "participant"
    path = TRANSCRIPTS / f"live-{slug}-{datetime.now():%Y%m%d-%H%M%S}.json"
    dump(
        session.to_transcript(),
        path,
        audit={
            "interlocutor_model": INTERLOCUTOR.model,
            "prompt_version": PROMPT_VERSION,
            "interlocutor_prompt_version": INTERLOCUTOR_PROMPT_VERSION,
            "interlocutor_failures": session.failures,
            "dropped_reactions": [dataclasses.asdict(d) for d in session.dropped],
        },
    )
    return path



with st.sidebar:
    st.markdown("**Новый замер**")
    label = st.text_input("Псевдоним участника", value="я", key="label")
    if st.button("Начать заново", key="restart"):
        for key in ("session", "saved", "run"):
            st.session_state.pop(key, None)
        st.rerun()
    st.divider()
    st.caption(
        "Сценарий тот же, что у персон: семь шагов, один ваш ответ на шаг. "
        "Оценка считается только после конца разбора и в диалоге не видна."
    )

if "session" not in st.session_state:
    st.session_state.session = LiveSession.start(label)
session: LiveSession = st.session_state.session
session.label = label

st.title("Разбор кейса: отток после редизайна")
st.caption(f"Шаг {min(session.answered + 2, len(STEPS))} из {len(STEPS)}")

for turn in session.turns:
    with st.chat_message("assistant" if turn.role == "assistant" else "user"):
        st.markdown(turn.content)

config = live_config()

if not session.finished:
    if not config.deepseek_api_key:
        st.error("Нужен DEEPSEEK_API_KEY в .env: живая сессия вызывает модель.")
        st.stop()
    answer = st.chat_input("Ваш ответ", key="answer")
    if answer and answer.strip():
        reaction, failure = "", ""
        try:
            with st.spinner("Собеседник отвечает…"):
                reaction = react(session, answer, config)
        except Exception as exc:  # the step goes on with the scripted line alone
            failure = str(exc)[:200]
        session.answer(answer, reaction, failure)
        st.rerun()
else:
    with st.chat_message("assistant"):
        st.markdown(CLOSING_RU)
    if "saved" not in st.session_state:
        st.session_state.saved = save(session)
    st.success(f"Транскрипт сохранён: `{st.session_state.saved.relative_to(ROOT)}`")
    if session.failures:
        st.caption(
            f"Шагов без реакции из-за сбоя провайдера: {len(session.failures)}. "
            "Сессия продолжилась репликой сценария; сбои записаны в audit."
        )
    if session.dropped:
        st.caption(
            f"Отброшено реакций собеседника с ещё закрытым фактом: {len(session.dropped)}. "
            "Они записаны в транскрипт в раздел audit."
        )
    if "run" not in st.session_state and st.button("Оценить сессию", key="score"):
        with st.spinner("Наблюдатель и судья размечают ходы…"):
            with repo.connect(config.db_path) as connection:
                st.session_state.run = run_transcript(session.to_transcript(), config, connection)
        st.rerun()
    if "run" in st.session_state:
        run = st.session_state.run
        st.divider()
        st.header("Результат")
        for note in run.report.notes_ru:
            st.warning(note)
        tab_scores, tab_protocol, tab_text = st.tabs(
            ["Оценка", "Протокол сессии", "Текстовый отчёт"]
        )
        with tab_scores:
            render_scores(run.report)
        with tab_protocol:
            render_protocol(session.to_transcript(), run)
        with tab_text:
            st.text(render_text(run.report))
