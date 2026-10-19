"""The task model: the eight seeded situations that make constructs observable.

A task is a psychometric item, not scenery. The distinction it buys is the one
that keeps the instrument honest: a task the dialogue never surfaced yields no
evidence at all, so a participant is never penalised for a situation they were
not placed in.

Two kinds:
  flaw  — a defect in the AI's reasoning or data that can be caught or missed
  probe — a situation the scenario creates on purpose to elicit a construct
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from probe.inventory.constructs import CONSTRUCTS_BY_ID
from probe.inventory.indicators import INDICATORS_BY_ID

TaskKind = Literal["flaw", "probe"]

# Ordered worst-to-best, like every other scale here.
TASK_STATES: tuple[str, ...] = ("missed", "partial", "caught")


@dataclass(frozen=True)
class Task:
    id: str
    kind: TaskKind
    summary_ru: str
    constructs: tuple[str, ...]
    """Parents in the network. Two constructs combine compensatorily (see cpt.py)."""
    elicits: tuple[str, ...]
    """Indicators this task makes observable. Catching it means showing one of these."""
    partial_threshold: float = 0.4
    """Ability at which 'partial' becomes as likely as 'missed'."""
    caught_threshold: float = 0.7
    """Ability at which 'caught' becomes as likely as everything below it."""
    discrimination: float = 5.0
    note_ru: str = ""


TASKS: tuple[Task, ...] = (
    Task(
        id="T1",
        kind="flaw",
        summary_ru="Выжившие: отток посчитан только по клиентам, оставшимся активными.",
        constructs=("kn", "mf"),
        elicits=("kn.gap", "kn.scope", "mf.question", "kn.accept_silent"),
        partial_threshold=0.35,
        caught_threshold=0.65,
        note_ru=(
            "Компенсаторная связь: заметить пропущенную группу можно и от «знания», "
            "и от привычки спрашивать, как считали. Требовать оба конструкта сразу "
            "означало бы мерить их произведение, а не каждый."
        ),
    ),
    Task(
        id="T2",
        kind="flaw",
        summary_ru="Подменённая база сравнения: неделя сопоставлена с праздничной.",
        constructs=("da", "mf"),
        elicits=("da.evidence", "mf.operational", "kn.scope"),
        partial_threshold=0.45,
        caught_threshold=0.75,
        note_ru=(
            "Тоже компенсаторная, хотя соблазн конъюнктивной велик: кажется, что нужно "
            "и усомниться, и уметь сформулировать проверку. Но участник, который просто "
            "спросил «с чем сравнивали», ловушку уже вскрыл — операционализация улучшает "
            "ответ, а не создаёт его."
        ),
    ),
    Task(
        id="T3",
        kind="flaw",
        summary_ru="Корреляция выдана за причинность: одновременно с редизайном подняли цену.",
        constructs=("da",),
        elicits=("da.mechanism", "da.counter", "kn.assumption"),
        partial_threshold=0.4,
        caught_threshold=0.7,
    ),
    Task(
        id="T4",
        kind="flaw",
        summary_ru="Абсолютные числа без знаменателя: «ушло 1 240 человек».",
        constructs=("kn",),
        elicits=("kn.gap", "mf.question", "kn.accept_silent"),
        partial_threshold=0.3,
        caught_threshold=0.6,
        note_ru="Самая лёгкая ловушка набора — порог низкий намеренно.",
    ),
    Task(
        id="T5",
        kind="flaw",
        summary_ru="Уверенная галлюцинация: несуществующий «отраслевой бенчмарк оттока 23%».",
        constructs=("da", "ag"),
        elicits=("da.evidence", "ag.reject", "da.surface"),
        partial_threshold=0.45,
        caught_threshold=0.7,
        note_ru=(
            "Единственная ловушка, где пропуск почти всегда означает da.surface: "
            "участник принял уверенный тон за основание."
        ),
    ),
    Task(
        id="T6",
        kind="probe",
        summary_ru="Меню из четырёх продолжений, три из которых не влияют на решение.",
        constructs=("mf",),
        elicits=("mf.priority", "mf.operational", "mf.scatter"),
        partial_threshold=0.4,
        caught_threshold=0.7,
    ),
    Task(
        id="T7",
        kind="probe",
        summary_ru="Новый факт в середине: в тот же период отключили push-уведомления.",
        constructs=("ss",),
        elicits=("ss.update", "ss.integrate", "ss.anchor", "ss.flip"),
        partial_threshold=0.35,
        caught_threshold=0.75,
        note_ru=(
            "Единственная проба, в которой наблюдаемы оба отрицательных индикатора "
            "конструкта: удержать позицию и перевернуть её можно только здесь."
        ),
    ),
    Task(
        id="T8",
        kind="probe",
        summary_ru="В конце разбора ИИ выдаёт готовое решение и спрашивает «согласны?».",
        constructs=("ag",),
        elicits=("ag.own", "ag.attribute", "ag.parrot", "ag.delegate"),
        partial_threshold=0.4,
        caught_threshold=0.7,
    ),
)

TASKS_BY_ID: dict[str, Task] = {t.id: t for t in TASKS}


def tasks_eliciting(indicator_id: str) -> tuple[Task, ...]:
    return tuple(t for t in TASKS if indicator_id in t.elicits)


def _validate() -> None:
    if len(TASKS_BY_ID) != len(TASKS):
        raise ValueError("duplicate task id")
    covered: set[str] = set()
    for task in TASKS:
        if not task.constructs:
            raise ValueError(f"{task.id}: a task with no construct scores nothing")
        for construct in task.constructs:
            if construct not in CONSTRUCTS_BY_ID:
                raise ValueError(f"{task.id}: unknown construct {construct!r}")
        for indicator in task.elicits:
            if indicator not in INDICATORS_BY_ID:
                raise ValueError(f"{task.id}: unknown indicator {indicator!r}")
        if not 0.0 <= task.partial_threshold < task.caught_threshold <= 1.0:
            # Out of order thresholds produce negative probabilities for 'partial'.
            raise ValueError(f"{task.id}: thresholds must satisfy 0 <= partial < caught <= 1")
        covered.update(task.constructs)
    missing = set(CONSTRUCTS_BY_ID) - covered
    if missing:
        # Without a task, a construct can only be observed by luck, and never
        # produces negative evidence. design/00 §3.
        raise ValueError(f"constructs with no task: {sorted(missing)}")

    # Negative evidence comes almost entirely from tasks, so a negative indicator
    # that no task elicits is nearly unobservable — it would sit in the inventory
    # looking like coverage while never firing.
    elicited = {indicator for task in TASKS for indicator in task.elicits}
    orphans = sorted(
        i.id
        for i in INDICATORS_BY_ID.values()
        if i.polarity == "negative" and not i.deprecated and i.id not in elicited
    )
    if orphans:
        raise ValueError(f"negative indicators no task elicits: {orphans}")


_validate()
