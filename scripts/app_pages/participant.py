"""One participant: the score, the dialogue under it, and where both came from."""

from __future__ import annotations

import streamlit as st

from scripts.app_data import (
    SOURCES_RU,
    is_live,
    observer_for,
    participants,
    scored,
    title,
)
from scripts.views import render_protocol, render_scores
from probe.llm import CassetteMissing
from probe.models import ADJUDICATOR, OBSERVER, judge_shares_observer_family
from probe.scenario.tasks import TASKS

everyone = participants()
# The URL wins, so a copied link opens the same person; then the last one picked.
key = st.query_params.get("id") or st.session_state.get("participant") or everyone[0]
if key not in everyone:
    key = everyone[0]
st.session_state["participant"] = key
st.query_params["id"] = key


def pick() -> None:
    st.session_state["participant"] = st.session_state["pick"]
    st.query_params["id"] = st.session_state["pick"]


st.page_link("app_pages/roster.py", label="Все испытуемые", icon=":material/arrow_back:")
st.session_state["pick"] = key
st.selectbox(
    "Испытуемый",
    everyone,
    format_func=lambda k: ("Живая: " if is_live(k) else "") + title(k),
    key="pick",
    on_change=pick,
)

live = is_live(key)
observer = observer_for(key, st.session_state.get("observer", "model"))

try:
    transcript, run = scored(key, observer)
except CassetteMissing:
    st.info(
        "Оценка не сохранилась: сессия прошла до того, как прибор начал записывать "
        "ответы моделей."
    )
    st.stop()
report = run.report

if live:
    st.title(f"Живая сессия: {transcript.label}")
    st.markdown(
        "_Сценарий проходил человек, а не персона. Ответы наблюдателя и судьи записаны "
        "при оценке и воспроизводятся без сети: это та же оценка, а не новая. "
        "Эталонной разметки нет, поэтому размечала модель._"
    )
else:
    st.title(f"Персона: {title(key)}")
    st.markdown(f"_{transcript.design_intent_ru}_")
for note in report.notes_ru:
    st.warning(note)

tab_scores, tab_dialog, tab_sources = st.tabs(
    ["Оценка", "Диалог и доказательства", "Источники"]
)

with tab_scores:
    render_scores(report)

with tab_dialog:
    render_protocol(transcript, run)

with tab_sources:
    same_family = judge_shares_observer_family()
    st.markdown(
        f"- **Разметка:** {SOURCES_RU[observer]}\n"
        f"- **Наблюдатель:** `{OBSERVER.model}` (DeepSeek-V4.1-Flash на дату записи)\n"
        f"- **Судья:** `{ADJUDICATOR.model}`"
        + (" — того же семейства, согласие не независимо" if same_family else "")
        + "\n"
        + (
            "- **Запись ответов моделей:** при оценке сессии, `fixtures/transcripts/cassettes/`\n"
            if live
            else "- **Запись ответов моделей:** 2026-10-11, `fixtures/cassettes/`\n"
        )
        + f"- **Отпечаток параметров:** `{report.fingerprint}` — меняется при любой "
        "правке таблиц или порогов\n"
        f"- **Сценарий:** `{transcript.scenario_id}`, предъявлено "
        f"{len(report.surfaced)} из {len(TASKS)} заданий"
    )
    telemetry = report.telemetry
    if telemetry and telemetry.turns_observed:
        st.subheader("Здоровье прибора")
        c1, c2, c3 = st.columns(3)
        c1.metric("Ходов размечено", telemetry.turns_observed)
        c2.metric("Наблюдений", telemetry.emitted)
        c3.metric("Выдуманных id", f"{telemetry.dropped_unknown_indicator} ({telemetry.drop_rate:.0%})")
