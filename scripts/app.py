"""The demo surface: everyone who was measured, then one of them, traceable from
score down to their own words.

    uv run --extra web streamlit run scripts/app.py
    uv run --offline --extra web streamlit run scripts/app.py   # no network at all

Pages, in the order of a showing: `about.py` (terms and the pipeline),
`roster.py` (every participant on one scale), `participant.py` (one of them),
`live.py` (a new measurement), all under `app_pages/`. A participant travels between them as `?id=<key>` in the URL and as `participant` in session state.

Replay only, except one page. Every run the other pages show is built with
`replay_only()` and none of them reads .env, so they cannot reach a provider or
spend money, and what they show is exactly what the recorded cassettes and the
tests pin. «Новый замер» (`app_pages/live.py`) is the single page that calls the
provider; it records cassettes, so its session then replays like any other.

It renders the same objects the text report is built from (`Run`, `Report`,
`Reviewed`) and computes nothing of its own beyond layout. A number on screen
that the report could not reproduce would be a second instrument.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

from scripts.app_data import SOURCES_RU  # noqa: E402

st.set_page_config(page_title="probe — прибор", layout="wide")

pages = st.navigation(
    [
        st.Page("app_pages/about.py", title="О приборе", default=True),
        st.Page("app_pages/roster.py", title="Испытуемые"),
        # Reached only from a row of the roster, never from the menu.
        st.Page("app_pages/participant.py", title="Испытуемый", visibility="hidden"),
        st.Page("app_pages/live.py", title="Новый замер"),
    ]
)

with st.sidebar:
    st.markdown("### probe")
    st.caption("Измерение критического мышления в диалоге с ИИ")
    st.radio(
        "Кто размечал ходы",
        ["model", "gold"],
        format_func=lambda k: SOURCES_RU[k],
        key="observer",
        help="Касается персон. У живых сессий эталонной разметки нет: только модель.",
    )
    st.divider()
    st.caption(
        "Всё, кроме «Нового замера», работает без сети на записанных ответах моделей "
        "и вызвать модель не может."
    )

pages.run()
