"""A live session on its own, without the rest of the demo.

    uv run --extra web streamlit run scripts/live.py

The same page the demo shows as «Новый замер» (`app_pages/live.py`); this entry
only gives it a window of its own.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import streamlit as st  # noqa: E402

st.set_page_config(page_title="probe — живая сессия", layout="wide")
st.navigation([st.Page("app_pages/live.py", title="Новый замер")], position="hidden").run()
