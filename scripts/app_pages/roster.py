"""Everyone who was measured, on one scale. A row opens that participant."""

from __future__ import annotations

import altair as alt
import streamlit as st

from scripts.app_data import (
    PERSONA_NAMES_RU,
    PERSONA_ORDER,
    construct_titles,
    rescored,
    roster,
)
from probe.score.aggregate import DEFAULT_THRESHOLDS

observer = st.session_state.get("observer", "model")
adjudicator = observer == "model"

st.title("Испытуемые")
st.caption(
    "Все, кого измерил прибор, на одной шкале: ожидание от 0 до 1 по каждой из пяти "
    "способностей, 0.5 — «ничего не известно». Персоны написаны автором прибора, поэтому "
    "их сравнение — проверка согласованности, а не валидности. Живые сессии — люди, "
    "прошедшие тот же сценарий."
)

table = roster(observer)
scores = list(construct_titles().values())
picked = st.dataframe(
    table,
    width="stretch",
    hide_index=True,
    on_select="rerun",
    selection_mode="single-row",
    key="roster",
    column_config={
        title: st.column_config.ProgressColumn(title, min_value=0.0, max_value=1.0, format="%.2f")
        for title in scores
    },
)
st.caption("Выберите строку, чтобы открыть оценку, диалог и доказательства испытуемого.")

rows = picked.selection.rows
if rows:
    key = table.index[rows[0]]
    st.session_state["participant"] = key
    st.switch_page("app_pages/participant.py", query_params={"id": key})

tab_sensitivity, tab_limits = st.tabs(["Чувствительность", "Чего прибор не доказывает"])

with tab_sensitivity:
    st.caption(
        "Если наблюдатель не нашёл реакции на задание, прибор считает его пропущенным — "
        "но наблюдатель мог и не заметить. Насколько этому верить, автор выбрал без "
        "данных. Ползунок меняет только это число; наблюдения те же."
    )
    missed = st.slider(
        "Насколько верить, что «не нашли» значит «пропустил»",
        min_value=0.5,
        max_value=0.95,
        value=DEFAULT_THRESHOLDS.missed_confidence,
        step=0.05,
        key="missed",
    )
    st.caption("0.75 — сейчас в приборе.")
    swept = rescored(observer, adjudicator, missed)
    means = swept.mean(axis=1).rename("среднее").reset_index()
    bars = (
        alt.Chart(means)
        .mark_bar(color="#2f6b4f")
        .encode(
            x=alt.X("среднее:Q", scale=alt.Scale(domain=[0, 1]), title="среднее по пяти способностям"),
            y=alt.Y("персона:N", sort=[PERSONA_NAMES_RU[k] for k in PERSONA_ORDER], title=None),
            tooltip=["персона", alt.Tooltip("среднее:Q", format=".2f")],
        )
        .properties(height=220)
    )
    st.altair_chart(bars, width="stretch")
    others = means[means["персона"] != PERSONA_NAMES_RU["strong"]]["среднее"]
    col1, col2 = st.columns(2)
    col1.metric("Отрыв сильного от лучшего из остальных", f"{means['среднее'].max() - others.max():.2f}")
    col2.metric("Различие между остальными пятью", f"{others.max() - others.min():.2f}")
    st.caption(
        "Сильный отделён от остальных при любом значении: он ничего не пропустил. "
        "Число влияет только на то, насколько различаются слабые между собой."
    )

with tab_limits:
    st.markdown(
        "- **Контур замкнут.** Индикаторы, вероятностные таблицы и персоны написаны "
        "одним человеком. Совпадение с замыслом — согласованность, а не валидность.\n"
        "- **Параметры не калиброваны.** Все пороги и таблицы авторские.\n"
        "- **Атрибуция ловит только дословное.** Пересказ слов ИИ своими словами "
        "проходит незамеченным.\n"
        "- **Один сценарий.** Сопоставимость между кейсами не проверялась.\n"
        "- **Смещения не проверялись.** Нет данных, чтобы сравнить группы при равном навыке.\n"
        "- **Судья строже к отрицательным свидетельствам:** отклоняет их почти вдвое "
        "чаще положительных. Это может завышать баллы слабым участникам.\n"
        "- **Это авторская модель.** Рабочая операционализация пяти способностей, "
        "а не реконструкция чужой методики."
    )
    st.caption("Полный разбор — в design/02-validation.md.")
