"""Who the demo can show, and their scored runs, cached across both pages.

Every run is built with `replay_only()`: the demo never reads .env, so it cannot
reach a provider. Live sessions replay from the cassettes recorded when they
were scored; personas replay from the cassettes recorded on 2026-10-11.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from probe.config import LIVE_CASSETTE_DIR, LIVE_DIR, ROOT, replay_only
from probe.inventory.constructs import CONSTRUCTS
from probe.llm import CassetteMissing
from probe.run import Run, run_transcript
from probe.score.aggregate import Thresholds
from probe.score.session import score_session
from probe.store import repo
from probe.transcript import Transcript, load

PERSONAS = ROOT / "fixtures" / "personas"

PERSONA_NAMES_RU = {
    "strong": "Сильный",
    "weak": "Слабый",
    "polished": "Гладкий",
    "passive": "Пассивный",
    "contrarian": "Спорщик",
    "gaming": "Игрок",
}
PERSONA_ORDER = ["strong", "polished", "contrarian", "gaming", "passive", "weak"]

SOURCES_RU = {
    "model": "Модель (записанный прогон DeepSeek)",
    "gold": "Эталонная разметка автора",
}


@st.cache_resource(show_spinner=False)
def run_persona(label: str, observer: str, adjudicator: bool) -> tuple[Transcript, Run]:
    transcript = load(PERSONAS / f"{label}.json")
    with repo.connect() as connection:
        run = run_transcript(
            transcript, replay_only(), connection, observer=observer, adjudicator=adjudicator
        )
    return transcript, run


@st.cache_resource(show_spinner=False)
def run_saved(name: str) -> tuple[Transcript, Run]:
    """A live session, replayed from the cassettes recorded when it was scored.
    Observer only: a live session has no hand-written labels to stand in for it."""
    transcript = load(LIVE_DIR / name)
    with repo.connect() as connection:
        run = run_transcript(transcript, replay_only(LIVE_CASSETTE_DIR), connection)
    return transcript, run


def saved_sessions() -> list[str]:
    """Newest first. The file name carries the pseudonym and the time it was saved."""
    return sorted((p.name for p in LIVE_DIR.glob("live-*.json")), reverse=True)


def participants() -> list[str]:
    return PERSONA_ORDER + saved_sessions()


def is_live(key: str) -> bool:
    return key not in PERSONA_NAMES_RU


def title(key: str) -> str:
    if not is_live(key):
        return PERSONA_NAMES_RU[key]
    stem = key.removeprefix("live-").removesuffix(".json")
    label, _, stamp = stem.rpartition("-")
    label, _, day = label.rpartition("-")
    return f"{label} · {day[6:8]}.{day[4:6]} {stamp[:2]}:{stamp[2:4]}"


def observer_for(key: str, chosen: str) -> str:
    """A live session has no gold labels: whatever is chosen, the model labelled it."""
    return "model" if is_live(key) else chosen


def scored(key: str, observer: str) -> tuple[Transcript, Run]:
    """Raises CassetteMissing for a live session scored before cassettes were kept."""
    if is_live(key):
        return run_saved(key)
    return run_persona(key, observer, observer == "model")


def construct_titles() -> dict[str, str]:
    return {c.id: c.title_ru for c in CONSTRUCTS}


def roster(observer: str) -> pd.DataFrame:
    """One row per participant: who they are and the expectation per construct,
    read straight off each report. A session that cannot be replayed offline
    keeps its row, with empty scores, so nobody silently drops off the list."""
    titles = construct_titles()
    rows = []
    for key in participants():
        used = observer_for(key, observer)
        row = {
            "key": key,
            "испытуемый": title(key),
            "кто": "живая сессия" if is_live(key) else "персона автора",
            "разметка": "модель" if used == "model" else "эталон",
        }
        try:
            _, run = scored(key, used)
        except CassetteMissing:
            row["разметка"] = "нет записи ответов моделей"
        else:
            row.update({titles[c.id]: c.expected for c in run.report.constructs})
        rows.append(row)
    return pd.DataFrame(rows).set_index("key")


@st.cache_data(show_spinner=False)
def rescored(observer: str, adjudicator: bool, missed: float) -> pd.DataFrame:
    """Every persona rescored from its stored rows under another MISSED_CONFIDENCE.
    Nothing is re-observed; only the rollup parameter changes."""
    rows = []
    thresholds = Thresholds(missed_confidence=missed)
    with repo.connect() as connection:
        for label in PERSONA_ORDER:
            session_id = run_transcript(
                load(PERSONAS / f"{label}.json"),
                replay_only(),
                connection,
                observer=observer,
                adjudicator=adjudicator,
            ).session_id
            expected = score_session(connection, session_id, thresholds=thresholds).expected()
            rows.append({"персона": PERSONA_NAMES_RU[label], **expected})
    return pd.DataFrame(rows).set_index("персона")
