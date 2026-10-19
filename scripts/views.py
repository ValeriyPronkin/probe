"""The two screens a session is read through: the score, and the protocol.

Shared by the demo (`app.py`, replay only) and the live session (`live.py`), so
a live participant's result is read on exactly the screens the personas are.
Like the app, they render the objects the text report is built from and compute
nothing of their own beyond layout.
"""

from __future__ import annotations

import altair as alt
import pandas as pd
import streamlit as st

from probe.inventory.constructs import LEVEL_VALUES
from probe.inventory.indicators import INDICATORS_BY_ID
from probe.report.build import LEVEL_LABELS_RU, Report
from probe.run import Run
from probe.scenario.tasks import TASKS_BY_ID
from probe.transcript import Transcript

CHECK_NAMES_RU = {
    "meta_turn": "мета-ход",
    "grounding": "заземление",
    "attribution": "атрибуция",
    "adjudication": "судья",
}
CHECK_HELP_RU = {
    "meta_turn": "ход не о самой оценке и не о сбое («как ты меня оцениваешь?», «страница зависла»)",
    "grounding": "цитата дословно есть в сообщении участника",
    "attribution": "цитата не списана с реплик ИИ: дословных совпадений меньше половины",
    "adjudication": "другая модель пыталась опровергнуть наблюдение и не смогла",
}
TASK_STATES_RU = {"caught": "вскрыл", "partial": "частично", "missed": "пропустил"}


def badge(text: str, color: str) -> str:
    return f":{color}-background[{text}]"


def render_scores(report: Report) -> None:
    with st.expander("Как читать оценку", expanded=False):
        st.markdown(
            "**Полоса — уверенность прибора, а не процент правильных ответов.** Каждая полоса "
            "делится на три части: насколько вероятно, что уровень способности у участника "
            "низкий, средний или высокий. Тёмная часть занимает 79% — значит, прибор на 79% "
            "уверен, что уровень высокий.\n\n"
            "**Ожидание — та же полоса одним числом** от 0 до 1: низкий уровень считается за 0, "
            "средний за 0.5, высокий за 1. Например, низкий 1%, средний 20%, высокий 79% дают "
            "0.20 × 0.5 + 0.79 × 1 ≈ 0.89. Удобно для сравнения, но теряет главное — насколько "
            "прибор уверен.\n\n"
            "**0.5 — это «ничего не известно»**, с этого прибор начинает. Ближе к 1 — "
            "свидетельства в пользу высокого уровня, ближе к 0 — в пользу низкого.\n\n"
            "**Мелкие различия не значимы.** 0.12 и 0.15 — один и тот же низкий уровень: "
            "параметры прибора авторские и не калиброваны, точность в сотых он не обещает.\n\n"
            "**Почему именно такая оценка** — раскройте строку способности ниже: там задания "
            "(вскрыл / частично / пропустил) и дословные цитаты участника. Зелёное говорит в "
            "пользу высокого уровня, красное — против."
        )
    levels = list(LEVEL_VALUES)
    rows = []
    for construct in report.constructs:
        for level in levels:
            rows.append(
                {
                    "способность": construct.title_ru,
                    "уровень": LEVEL_LABELS_RU[level],
                    "порядок": levels.index(level),
                    "вероятность": construct.posterior[level],
                }
            )
    chart = (
        alt.Chart(pd.DataFrame(rows))
        .mark_bar()
        .encode(
            y=alt.Y(
                "способность:N",
                sort=[c.title_ru for c in report.constructs],
                title=None,
                axis=alt.Axis(labelLimit=220),
            ),
            x=alt.X("вероятность:Q", stack="normalize", axis=alt.Axis(format="%"), title=None),
            color=alt.Color(
                "уровень:N",
                sort=[LEVEL_LABELS_RU[lv] for lv in levels],
                scale=alt.Scale(
                    domain=[LEVEL_LABELS_RU[lv] for lv in levels],
                    range=["#c9d3cc", "#8fb8a0", "#2f6b4f"],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            order=alt.Order("порядок:Q"),
            tooltip=["способность", "уровень", alt.Tooltip("вероятность:Q", format=".0%")],
        )
        .properties(height=230)
    )
    st.altair_chart(chart, width="stretch")

    for construct in report.constructs:
        spread = " · ".join(
            f"{LEVEL_LABELS_RU[lv]} {construct.posterior[lv]:.0%}" for lv in levels
        )
        header = f"**{construct.title_ru}** — ожидание {construct.expected:.2f}  ({spread})"
        if not construct.observed:
            header += " · свидетельств нет, показан априор"
        with st.expander(header):
            for evidence in construct.evidence:
                if evidence.kind == "task":
                    task = TASKS_BY_ID[evidence.node.removeprefix("task:")]
                    color = {"caught": "green", "partial": "orange", "missed": "red"}[evidence.state]
                    st.markdown(
                        f"{badge(task.id + ' · ' + TASK_STATES_RU[evidence.state], color)} "
                        f"{task.summary_ru}"
                    )
                else:
                    indicator = INDICATORS_BY_ID[evidence.node]
                    color = "green" if indicator.polarity == "positive" else "red"
                    st.markdown(
                        f"{badge(indicator.id, color)} {indicator.definition_ru}"
                    )
                seen: set[tuple[int, str]] = set()
                for source in evidence.sources:
                    key = (source.turn_index, source.quote)
                    if key in seen:
                        continue
                    seen.add(key)
                    st.markdown(f"> ход {source.turn_index}: «{source.quote}»")
                if evidence.kind == "task" and not evidence.sources:
                    st.caption("Задание предъявлено, реакции участника не было.")

    if report.not_surfaced:
        st.subheader("Не наблюдалось")
        st.caption("Эти задания сценарий не предъявил. Молчание участника по ним ничего не значит.")
        for task_id in report.not_surfaced:
            st.markdown(f"- {task_id} — {TASKS_BY_ID[task_id].summary_ru}")


def render_protocol(transcript: Transcript, run: Run) -> None:
    st.caption(
        "Весь разговор. Под каждым ответом участника — что в нём отметил наблюдатель "
        "и что решили четыре проверки. Отклонённое — не низкий балл, а материал, "
        "который прибор отказался засчитывать."
    )
    surfaced_at: dict[int, list[str]] = {}
    for task_id, turn_idx in transcript.surfaced.items():
        surfaced_at.setdefault(turn_idx, []).append(task_id)
    by_turn: dict[int, list] = {}
    for item in run.reviewed:
        by_turn.setdefault(item.observation.turn_index, []).append(item)

    for turn in transcript.turns:
        role = "assistant" if turn.role == "assistant" else "user"
        with st.chat_message(role):
            st.markdown(turn.content)
            for task_id in sorted(surfaced_at.get(turn.idx, [])):
                st.markdown(
                    f"{badge('задание ' + task_id, 'violet')} {TASKS_BY_ID[task_id].summary_ru}"
                )
            items = by_turn.get(turn.idx, [])
            if role == "user" and not items:
                st.caption("Наблюдатель ничего не отметил.")
            for item in items:
                obs = item.observation
                indicator = INDICATORS_BY_ID[obs.indicator_id]
                task = f" · {obs.task_id}" if obs.task_id else ""
                if item.admitted:
                    st.markdown(
                        f"{badge('засчитано', 'green')} `{obs.indicator_id}`{task} "
                        f"({'+' if indicator.polarity == 'positive' else '−'}) — «{obs.quote}»"
                    )
                else:
                    reason = next((d for _, ok, d in item.verdicts if not ok), "")
                    st.markdown(
                        f"{badge('отклонено: ' + CHECK_NAMES_RU[item.failed_check], 'red')} "
                        f"`{obs.indicator_id}`{task} — «{obs.quote}»  \n_{reason}_"
                    )
                with st.popover("проверки"):
                    for name, ok, detail in item.verdicts:
                        mark = "✓" if ok else "✗"
                        st.markdown(f"{mark} **{CHECK_NAMES_RU[name]}** {detail}")
                        st.caption(CHECK_HELP_RU[name])
