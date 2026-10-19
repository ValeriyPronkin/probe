"""The scripted scenario `churn-v1` as a state machine. design/00 §3, live mode.

The personas were written against this script before it existed as code: every
one of them carries the same seven assistant lines at the same turns. Pulling the
lines out here makes that implicit script the single source, and a test holds
every persona to it — a live session and a persona go through the same scenario,
not a similar one.

What the machine fixes, and why each is not the model's call:

  the lines that surface tasks — task thresholds are calibrated to this wording;
      a paraphrased trap is a different item of unknown difficulty
  the order and the pace — one participant turn per step, as in the personas;
      a model that chose when to move on would choose who gets more chances
  what the interlocutor may know — each case fact opens at a step, so a trap
      cannot be revealed before it is presented

No I/O and no model calls here. The interlocutor that writes the short reaction
between steps lives in `interlocutor.py`.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from probe.scenario.tasks import TASKS_BY_ID
from probe.transcript import Transcript, Turn

SCENARIO_ID = "churn-v1"


@dataclass(frozen=True)
class Step:
    tasks: tuple[str, ...]
    line_ru: str


STEPS: tuple[Step, ...] = (
    Step(
        tasks=("T1", "T4"),
        line_ru=(
            "Разобрал данные по оттоку. За неделю после редизайна ушло 1 240 человек против 890 "
            "неделей раньше. Считал по клиентам, которые оставались активными на начало периода. "
            "Вывод: виноват редизайн, он единственное крупное изменение."
        ),
    ),
    Step(
        tasks=("T2",),
        line_ru=(
            "Сравнивал с неделей 1–7 января. Это ближайший полный недельный период, по которому "
            "есть сопоставимые данные."
        ),
    ),
    Step(
        tasks=("T5",),
        line_ru=(
            "Кстати, по отраслевому бенчмарку нормальный недельный отток — 23%. У нас выше, так "
            "что проблема подтверждается и внешними данными."
        ),
    ),
    Step(
        tasks=("T3",),
        line_ru=(
            "Редизайн выкатили 15-го, отток пошёл вверх с 16-го. Причинно-следственная связь "
            "здесь очевидна."
        ),
    ),
    Step(
        tasks=("T6",),
        line_ru=(
            "Что проверим дальше? 1) цвет основной кнопки 2) базу сравнения 3) шрифты в новом "
            "макете 4) отзывы в сторе."
        ),
    ),
    Step(
        tasks=("T7",),
        line_ru="Кстати, в тот же период отключили push-уведомления о незавершённых действиях.",
    ),
    Step(
        tasks=("T8",),
        line_ru="Мой вывод: редизайн надо откатывать. Согласны?",
    ),
)

CLOSING_RU = "Спасибо, на этом разбор закончен."
"""Shown after the last answer, never stored: it is not a turn anyone responds to."""


@dataclass(frozen=True)
class Fact:
    text_ru: str
    opens_after: int
    """The step the participant must already have answered before the interlocutor
    may say this. A fact that is a task's own content opens only once that task
    has been put in front of the participant."""
    markers: tuple[str, ...] = ()
    """Word-start patterns that betray the fact. Used to drop a reaction that
    mentions a fact it may not: a prompt asks, this guarantees."""
    confirm_only: bool = False
    """The fact is a trap's own answer. Even once open, the interlocutor may only
    confirm it after the participant named it; volunteered, it would hand the
    participant the catch and the instrument would credit them with the AI's
    idea. Found on the author's first full live session (2026-10-12): the price
    rise, offered in reply to «что ещё могло случиться?», scored T3 as caught
    after the participant had answered T3 with «согласен»."""


FACTS: tuple[Fact, ...] = (
    Fact(
        "Отток посчитан только по клиентам, активным на начало периода; клиенты, ушедшие "
        "раньше, в выборку не попали.",
        opens_after=0,
    ),
    Fact(
        "890 — отток за предыдущую неделю, посчитанный тем же способом.",
        opens_after=0,
    ),
    Fact(
        "Общего числа клиентов в выгрузке нет, поэтому долю оттока по ней посчитать нельзя.",
        opens_after=0,
    ),
    Fact(
        "Разбивки оттока по шагам воронки в выгрузке нет.",
        opens_after=0,
    ),
    Fact(
        "Неделя 1–7 января праздничная.",
        opens_after=1,
        markers=(r"праздн", r"новогод", r"рождеств", r"каникул"),
        confirm_only=True,
    ),
    Fact(
        "Данных за ту же неделю прошлого года в выгрузке нет, их можно запросить у аналитиков.",
        opens_after=1,
    ),
    Fact(
        "Назвать источник бенчмарка 23% ты не можешь.",
        opens_after=2,
    ),
    Fact(
        "15-го, одновременно с редизайном, подняли цену подписки.",
        opens_after=3,
        markers=(r"цен", r"подорож", r"тариф", r"стоимост", r"прайс"),
        confirm_only=True,
    ),
    Fact(
        "В тот же период отключили push-уведомления о незавершённых действиях.",
        opens_after=5,
        markers=(r"push", r"пуш", r"уведомлен"),
    ),
)


def open_facts(answered: int) -> tuple[Fact, ...]:
    return tuple(f for f in FACTS if f.opens_after <= answered)


def _mentions(fact: Fact, text: str) -> bool:
    return any(re.search(rf"(?<!\w){m}", text, flags=re.IGNORECASE) for m in fact.markers)


def leaked_fact(text: str, answered: int, prompt: str = "") -> Fact | None:
    """The first fact `text` may not mention: one still closed, or a confirm-only
    fact that `prompt` (the participant's answer) did not name first. Word-start
    matching, so «цен» catches «цена» and «ценой» but not «оценка» or «процент»."""
    for fact in FACTS:
        if not _mentions(fact, text):
            continue
        if fact.opens_after > answered:
            return fact
        if fact.confirm_only and not _mentions(fact, prompt):
            return fact
    return None


@dataclass(frozen=True)
class DroppedReaction:
    turn: int
    text: str
    fact: str


@dataclass
class LiveSession:
    """One participant going through the script. Mutable on purpose: it is the
    state a chat surface holds between messages, and nothing else reads it until
    `to_transcript` freezes it."""

    label: str
    turns: list[Turn] = field(default_factory=list)
    surfaced: dict[str, int] = field(default_factory=dict)
    answered: int = -1
    """Index of the last step the participant has answered; -1 before the first."""
    dropped: list[DroppedReaction] = field(default_factory=list)
    failures: list[str] = field(default_factory=list)
    """Provider errors on the interlocutor. The step went on without a reaction:
    a session is never lost to the provider, but the gap is recorded."""

    @classmethod
    def start(cls, label: str) -> LiveSession:
        session = cls(label=label)
        session._present(0, "")
        return session

    @property
    def finished(self) -> bool:
        return self.answered == len(STEPS) - 1

    @property
    def last_assistant(self) -> str:
        return next(t.content for t in reversed(self.turns) if t.role == "assistant")

    def answer(self, text: str, reaction: str = "", failure: str = "") -> None:
        """Record the participant's answer to the current step, then present the
        next one, prefixed by the interlocutor's reaction. A reaction that mentions
        a closed fact is dropped and kept for audit; the step goes on without it."""
        if self.finished:
            raise RuntimeError("the scenario is over")
        if not text.strip():
            raise ValueError("an empty answer is not a turn")
        self.turns.append(Turn(len(self.turns), "participant", text.strip()))
        if failure:
            self.failures.append(f"ход {len(self.turns)}: {failure}")
            reaction = ""
        self.answered += 1
        if self.finished:
            return
        reaction = reaction.strip()
        leak = leaked_fact(reaction, self.answered, text) if reaction else None
        if leak is not None:
            self.dropped.append(DroppedReaction(len(self.turns), reaction, leak.text_ru))
            reaction = ""
        self._present(self.answered + 1, reaction)

    def _present(self, step_index: int, reaction: str) -> None:
        step = STEPS[step_index]
        content = f"{reaction}\n\n{step.line_ru}" if reaction else step.line_ru
        idx = len(self.turns)
        self.turns.append(Turn(idx, "assistant", content))
        for task_id in step.tasks:
            self.surfaced[task_id] = idx

    def to_transcript(self) -> Transcript:
        return Transcript(
            label=self.label,
            source="live",
            scenario_id=SCENARIO_ID,
            turns=tuple(self.turns),
            surfaced=dict(self.surfaced),
        )


def _validate() -> None:
    seen: list[str] = []
    for step in STEPS:
        for task_id in step.tasks:
            if task_id not in TASKS_BY_ID:
                raise ValueError(f"script surfaces unknown task {task_id!r}")
            seen.append(task_id)
    if len(seen) != len(set(seen)):
        raise ValueError("a task is surfaced twice")
    if set(seen) != set(TASKS_BY_ID):
        # A task missing from the live script would be scored as never surfaced:
        # quiet, but the live session would measure less than the personas do.
        raise ValueError(f"tasks the script never surfaces: {sorted(set(TASKS_BY_ID) - set(seen))}")
    for fact in FACTS:
        if not 0 <= fact.opens_after < len(STEPS):
            raise ValueError(f"fact opens after a step that does not exist: {fact.text_ru}")


_validate()
