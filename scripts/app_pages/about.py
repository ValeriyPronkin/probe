"""What the instrument is, in its own terms, and how a turn becomes a score.

Constructs and tasks are read from the inventories, so this page cannot drift
from what the instrument actually measures.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from probe.inventory.constructs import CONSTRUCTS
from probe.inventory.indicators import INDICATORS
from probe.scenario.script import STEPS
from probe.scenario.tasks import TASKS
from probe.score.aggregate import DEFAULT_THRESHOLDS

MODEL = 'shape=box, style="rounded,filled", fillcolor="#e8eef9", color="#5b7bb5"'
CODE = 'shape=box, style="rounded,filled", fillcolor="#e3efe7", color="#2f6b4f"'

NOW = f"""
digraph {{
  rankdir=LR; bgcolor="transparent";
  node [fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
  edge [fontname="Helvetica", fontsize=9, color="#777777"];
  P  [label="Участник", shape=ellipse];
  I  [label="Собеседник-ИИ\\nведёт кейс,\\nпредъявляет задания", {MODEL}];
  O  [label="Наблюдатель\\nотмечает индикаторы\\nс цитатой", {MODEL}];
  A  [label="Проверки\\nтри фильтра кода\\n+ судья-модель", {CODE}];
  S  [label="Байесовская сеть\\nсчитает оценку,\\nмоделей не вызывает", {CODE}];
  R  [label="Отчёт\\nоценка · цитаты ·\\nотклонённое", {CODE}];
  P -> I [dir=both];
  I -> O [label="ход участника"];
  O -> A [label="наблюдения"];
  A -> S [label="засчитанные"];
  A -> R [label="отклонённые\\nс причиной", style=dashed];
  S -> R;
}}
"""

FUTURE = f"""
digraph {{
  rankdir=LR; bgcolor="transparent";
  node [fontname="Helvetica", fontsize=11, margin="0.15,0.08"];
  edge [fontname="Helvetica", fontsize=9, color="#777777"];
  P  [label="Участник", shape=ellipse];
  I  [label="Собеседник-ИИ\\nпредъявляет задание,\\nкоторое выбрал ведущий", {MODEL}];
  O  [label="Наблюдатель и проверки\\nиндикаторы с цитатой;\\nплюс тайминги и правки", {MODEL}];
  S  [label="Динамическая байесовская сеть\\nпосле каждого хода обновляет\\nP(уровень способности) на шаге t", {CODE}];
  L  [label="Ведущий\\nвыбирает следующее задание\\nтам, где прибор меньше всего уверен", {CODE}];
  R  [label="Отчёт\\nтраектория оценки\\nпо ходу диалога", {CODE}];
  P -> I [dir=both, label="(1) диалог"];
  I -> O [label="(2) ход участника"];
  O -> S [label="(3) засчитанные\\nсвидетельства"];
  S -> L [label="(4) вероятности"];
  L -> I [label="(5) следующее\\nзадание", constraint=false];
  S -> R;
}}
"""

PREFIX_EN = {
    "kn": "knowledge",
    "mf": "mental focus",
    "ag": "agency",
    "da": "deep analysis",
    "ss": "situational synthesis",
}
SUFFIX_RU = {
    "kn.gap": "gap — пробел",
    "kn.assumption": "assumption — предпосылка",
    "kn.scope": "scope — границы",
    "kn.accept_silent": "accept silent — принял молча",
    "mf.question": "question — вопрос",
    "mf.priority": "priority — приоритет",
    "mf.operational": "operational — исполнимая проверка",
    "mf.scatter": "scatter — разброс",
    "ag.own": "own — своя версия",
    "ag.attribute": "attribute — разделяет своё и ИИ",
    "ag.reject": "reject — отклоняет",
    "ag.parrot": "parrot — повторяет за ИИ",
    "ag.delegate": "delegate — передаёт решение",
    "da.evidence": "evidence — основания",
    "da.mechanism": "mechanism — механизм",
    "da.counter": "counter — опровержение",
    "da.surface": "surface — поверхностное принятие",
    "ss.update": "update — обновляет позицию",
    "ss.integrate": "integrate — соединяет",
    "ss.anchor": "anchor — держится прежнего",
    "ss.flip": "flip — переворачивается за ИИ",
}
"""Display glosses only. Kept out of the inventory on purpose: the inventory is
rendered into the observer's prompt, and a new field there would change every
cassette key. A new indicator without a gloss here fails the app test."""

st.title("probe")
st.markdown(
    "**Прибор, который измеряет латентные конструкты по поведению человека в диалоге с ИИ.** "
    "Первая операционализация — критическое мышление: пять способностей. "
    "Модель только отмечает поведение с цитатой, оценку считает код. "
    "Каждый балл прослеживается до слов участника."
)
st.caption(
    "Конструкты, индикаторы и задания приходят в конвейер как данные: критическое мышление — "
    "наполнение, а не устройство прибора. Что движок годится для другого конструкта, пока "
    "не проверено — наполнение одно."
)

st.subheader("Как устроено сейчас")
st.graphviz_chart(NOW, width="stretch")
st.caption("Синее — модели, зелёное — код.")

with st.expander("Как считается", expanded=False):
    st.markdown(
        "**1. Сеть.** Общая установка $D$ влияет на пять способностей $C_j$, "
        "способности — на задания и индикаторы $X_k$. `probe/score/network.py`"
    )
    st.latex(r"P(D, C, X) = P(D)\,\prod_{j=1}^{5} P(C_j \mid D)\,\prod_k P(X_k \mid C_{\text{род}(k)})")
    st.markdown(
        "**2. Связь способности и поведения.** Чем выше уровень $\\theta$, тем вероятнее "
        "вскрыть задание; $b$ — сложность, $a$ — различающая сила. Если у задания две "
        "способности, $\\theta$ — их среднее. `probe/score/cpt.py`"
    )
    st.latex(r"P(X \ge k \mid \theta) = \sigma\big(a(\theta - b_k)\big), \quad \theta \in \{0,\ 0.5,\ 1\}")
    st.markdown(
        "**3. Наблюдения в свидетельство.** Несколько наблюдений одного узла "
        "складываются, но не до полной уверенности; задание без реакции — "
        f"«пропустил» с уверенностью {DEFAULT_THRESHOLDS.missed_confidence}. "
        "`probe/score/aggregate.py`"
    )
    st.latex(
        rf"c = \min\Big(1 - \prod_i (1 - c_i),\ {DEFAULT_THRESHOLDS.max_confidence}\Big), \qquad"
        r" \lambda(x) = \begin{cases} c, & x = \text{наблюдаемое} \\ \frac{1-c}{2}, & \text{иначе} \end{cases}"
    )
    st.markdown(
        "**4. Вывод.** Точный перебор всех $3^6 = 729$ сочетаний уровней. "
        "`probe/score/bayes.py`"
    )
    st.latex(r"P(C_j \mid E) \propto \sum_{D,\,C_{-j}} P(D, C)\,\prod_{e \in E} \sum_x P(x \mid \text{род}_e)\,\lambda_e(x)")
    st.markdown("**5. Ожидание** — то же одним числом.")
    st.latex(r"\mathbb{E}[C_j] = 0 \cdot P(\text{низкий}) + 0.5 \cdot P(\text{средний}) + 1 \cdot P(\text{высокий})")

st.subheader("Основные понятия")
left, right = st.columns(2)
with left:
    st.markdown(
        f"- **Испытуемый** — персона автора или человек, прошедший сценарий.\n"
        f"- **Сценарий** — кейс «отток вырос после редизайна», одинаковый для всех.\n"
        f"- **Задание** — ловушка или проба внутри кейса, всего {len(TASKS)}: "
        "вскрыл / частично / пропустил.\n"
        f"- **Индикатор** — наблюдаемое поведение из закрытого списка ({len(INDICATORS)}).\n"
        "- **Наблюдатель** — модель, отмечает индикаторы в ходе участника с цитатой.\n"
        "- **Проверки** — решают, засчитать ли наблюдение; отклонённое видно в протоколе:\n"
        "    1. **Мета-ход** (код). Отсекает организационные и технические реплики "
        "(«завис экран», «как ты меня оцениваешь?»). Оставляет только содержательную "
        "работу с кейсом.\n"
        "    2. **Заземление** (код). Проверяет дословное наличие цитаты в логе. Если "
        "наблюдатель перефразировал слова человека — наблюдение отбрасывается.\n"
        "    3. **Атрибуция** (код). Следит за тем, чтобы человеку не приписывался текст ИИ "
        "(проверка на совпадение пассажей от 5 слов).\n"
        "    4. **Судья** (модель-арбитр). Старшая модель пытается опровергнуть гипотезу "
        "наблюдателя. Проходят только устойчивые выводы."
    )
with right:
    st.markdown(
        "- **Оценка** — уверенность «низкий / средний / высокий», а не процент.\n"
        "- **Ожидание** — та же оценка одним числом от 0 до 1; 0.5 — «ничего не известно».\n"
        "- **Запись** — сохранённые ответы моделей: та же запись даёт те же числа.\n"
        "- **Отпечаток параметров** — меняется при любой правке таблиц и порогов."
    )

st.markdown("**Пять способностей**")
st.markdown("\n".join(f"- **{c.title_ru}** — {c.definition_ru}" for c in CONSTRUCTS))

st.subheader("Что чем измеряется: Q-матрица")
st.markdown(
    "Каркас — **Evidence-Centered Design** (дизайн, ориентированный на доказательства): "
    "способности — что измеряем, индикаторы — чем это проявляется, задания — в каких ситуациях."
)
st.caption(
    "● — для задания нужна эта способность. Базовый вариант, собранный вручную "
    "по определениям способностей и классическим ошибкам рассуждения, — чтобы было от чего "
    "оттолкнуться. Сила каждой связи задаётся отдельно, параметрами задания (шаг 2 "
    "«Как считается»)."
)
KINDS_RU = {"flaw": "ловушка", "probe": "проба"}
q = pd.DataFrame(
    [
        {
            "задание": t.id,
            "тип": KINDS_RU[t.kind],
            **{c.title_ru: "●" if c.id in t.constructs else "" for c in CONSTRUCTS},
            "что предъявлено": t.summary_ru,
        }
        for t in TASKS
    ]
).set_index("задание")
st.dataframe(q, width="stretch")
per_construct = {c.title_ru: sum(c.id in t.constructs for t in TASKS) for c in CONSTRUCTS}
st.caption(
    "Заданий на способность: "
    + " · ".join(f"{title} {n}" for title, n in per_construct.items())
    + f". Индикаторов {len(INDICATORS)}, каждый относится к одной способности."
)

with st.expander("Задания: откуда взяты, где стоят, как связаны с индикаторами", expanded=False):
    st.markdown(
        "**Откуда взяты.** Ловушки T1–T4 — классические ошибки рассуждения с данными: "
        "ошибка выжившего, не та база сравнения, корреляция вместо причины, число без "
        "знаменателя. T5 — ошибка, характерная именно для ИИ: уверенный тон вместо источника. "
        "Пробы T6–T8 добавлены, потому что у трёх способностей не было ни одной ситуации, "
        "где они обязаны проявиться. Требования к заданию: правдоподобно звучит в кейсе, "
        "вскрывается одним вопросом участника, собеседник ответ не подсказывает.\n\n"
        "**Где стоят.** Сценарий — семь шагов с репликами ИИ, одинаковыми для всех. "
        "Задание предъявляет реплика сценария, а не модель: перефразированная ловушка — "
        "уже другое задание с неизвестной сложностью. Между шагами собеседник-ИИ добавляет "
        "только короткую реакцию по закрытому списку фактов. Факт, который вскрывает ловушку "
        "(праздничная неделя, рост цены), он может лишь подтвердить, если участник назвал его сам.\n\n"
        "**Связь с индикаторами.** Ответ участника после реплики относится к её заданию. "
        "Сильный положительный индикатор или два — «вскрыл»; один слабый или вперемешку с "
        "отрицательным — «частично»; только отрицательный или ничего — «пропустил». "
        "Задание, которое не предъявлено, не даёт свидетельств совсем."
    )
    step_of = {t: n for n, step in enumerate(STEPS, start=1) for t in step.tasks}
    line_of = {t: step.line_ru for step in STEPS for t in step.tasks}
    st.table(
        pd.DataFrame(
            [
                {
                    "задание": t.id,
                    "шаг": step_of[t.id],
                    "реплика ИИ (дословно)": line_of[t.id],
                    "через какие индикаторы вскрывается": ", ".join(t.elicits),
                }
                for t in TASKS
            ]
        )
        .sort_values(["шаг", "задание"])
        .set_index("задание")
    )

with st.expander(f"Индикаторы: все {len(INDICATORS)}, с примерами", expanded=False):
    st.caption(
        "Закрытый список: наблюдатель отмечает только эти, ничего не придумывает. "
        "Пара примеров передаётся ему дословно. + — говорит в пользу высокого уровня, "
        "− — против."
    )
    st.markdown(
        "**Как читать код.** `способность.поведение`: первая часть — способность, вторая — "
        "короткое английское слово о поведении. Код постоянный: на нём держится пересчёт "
        "прошлых сессий, поэтому определение можно уточнять, а код не меняется."
    )
    st.table(
        pd.DataFrame(
            [{"код": c.id, "способность": c.title_ru, "по-английски": PREFIX_EN[c.id]} for c in CONSTRUCTS]
        ).set_index("код")
    )
    POLARITY_RU = {"positive": "+", "negative": "−"}
    for tab, c in zip(st.tabs([c.title_ru for c in CONSTRUCTS]), CONSTRUCTS):
        with tab:
            st.table(
                pd.DataFrame(
                    [
                        {
                            "индикатор": i.id,
                            "код читается": SUFFIX_RU[i.id],
                            "знак": POLARITY_RU[i.polarity],
                            "что это": i.definition_ru,
                            "засчитывается": i.counts_ru,
                            "не засчитывается": i.not_counts_ru,
                        }
                        for i in INDICATORS
                        if i.construct == c.id and not i.deprecated
                    ]
                ).set_index("индикатор")
            )

with st.expander("Почему байесовская сеть, а не IRT", expanded=False):
    st.markdown(
        "Байесовская сеть — один из способов оценивать латентные конструкты, наряду с "
        "классической теорией тестов, IRT и когнитивно-диагностическими моделями (DINA, G-DINA).\n\n"
        "- **Данных для калибровки нет** — сеть работает на экспертных параметрах.\n"
        "- **Пропуски допустимы** — не каждое задание проявляется в каждом диалоге.\n"
        "- **Прослеживаемость** — каждый балл раскладывается до цитаты участника.\n\n"
        "Слабое место: таблицы вероятностей авторские. С данными — оценить их по данным (EM) "
        "и подогнать G-DINA или многомерную IRT на той же Q-матрице, затем сравнить модели."
    )

st.subheader("Куда развивать")
st.caption("Перспектива, не текущее состояние.")
st.graphviz_chart(FUTURE, width="stretch")
st.markdown(
    "- **Оценка по ходу диалога**, а не в конце: сеть становится динамической.\n"
    "- **Адаптивный сценарий:** следующее задание выбирается там, где прибор меньше всего уверен.\n"
    "- **Свидетельства шире текста:** тайминги ответов, правки.\n"
    "- **Не меняется:** модель балл не ставит, его считает код."
)

with st.expander("Как считалось бы", expanded=False):
    st.markdown(
        "**1. Оценка после каждого хода** (фильтрация). $S_t$ — уровни способностей "
        "на шаге $t$, $E_t$ — свидетельства этого хода. Новое по сравнению с сегодняшним — "
        "переход $P(S_t \\mid S_{t-1})$: может ли способность меняться внутри диалога."
    )
    st.latex(r"P(S_t \mid E_{1:t}) \propto P(E_t \mid S_t) \sum_{s} P(S_t \mid S_{t-1} = s)\, P(S_{t-1} = s \mid E_{1:t-1})")
    st.markdown(
        "**2. Выбор следующего задания.** Ведущий берёт задание, ответ на которое "
        "сильнее всего уменьшит неопределённость $H$."
    )
    st.latex(r"T_{t+1} = \arg\max_T \Big[ H(S_t) - \mathbb{E}_{x \sim P(X_T \mid E_{1:t})}\, H(S_t \mid E_{1:t}, X_T = x) \Big]")
    st.caption("Не реализовано: это направление, а не код.")
