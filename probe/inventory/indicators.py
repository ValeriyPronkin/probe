"""The evidence model: a closed inventory of observable behaviours.

A construct is a frame, not something a model can tag against. This inventory is
what the observer is actually allowed to emit, and the rule is absolute: never
invent an indicator. Nothing fits -> emit nothing.

Ids are permanent. An indicator that stops earning its place gets `deprecated`,
never deletion: removing it would strand every session already scored with it and
break the rebuild invariant the whole instrument rests on.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from probe.inventory.constructs import CONSTRUCTS_BY_ID

Polarity = Literal["positive", "negative"]

# Ordered worst-to-best, like LEVELS. "none" is never submitted as evidence —
# a silent detector is not an absent behaviour (design/00 §4).
INDICATOR_STATES: tuple[str, ...] = ("none", "weak", "strong")


@dataclass(frozen=True)
class Indicator:
    id: str
    construct: str
    polarity: Polarity
    definition_ru: str
    counts_ru: str
    """An example that DOES count — shown to the observer verbatim."""
    not_counts_ru: str
    """The near-miss that does NOT count. Without this pair the observer drifts."""
    difficulty: float = 0.5
    """Ability at which a participant is as likely as not to show this. Negative
    indicators read inverted: a high value means even strong participants slip."""
    discrimination: float = 4.0
    """How sharply the indicator separates levels. Low = noisy, fires at any level."""
    deprecated: bool = False


INDICATORS: tuple[Indicator, ...] = (
    # ── Знание ────────────────────────────────────────────────────────────
    Indicator(
        id="kn.gap",
        construct="kn",
        polarity="positive",
        definition_ru="Называет конкретный отсутствующий факт, без которого вывод не держится.",
        counts_ru="«А отток считали по всем клиентам или только по тем, кто остался активен?»",
        not_counts_ru="«Хотелось бы больше данных» — общее пожелание, не названный факт.",
        difficulty=0.45,
    ),
    Indicator(
        id="kn.assumption",
        construct="kn",
        polarity="positive",
        definition_ru="Вслух формулирует неявную предпосылку постановки или вывода.",
        counts_ru="«Вы исходите из того, что до редизайна поведение было стабильным».",
        not_counts_ru="«Это предположение» — без указания, какое именно.",
        difficulty=0.6,
    ),
    Indicator(
        id="kn.scope",
        construct="kn",
        polarity="positive",
        definition_ru="Отмечает границы применимости: на какой выборке, за какой период это верно.",
        counts_ru="«Это верно для недельного окна, а на месячном эффект может исчезнуть».",
        not_counts_ru="«Данных мало» — без указания границы.",
        difficulty=0.65,
    ),
    Indicator(
        id="kn.accept_silent",
        construct="kn",
        polarity="negative",
        definition_ru=(
            "Принимает вывод как основание для решения, не назвав ни одного "
            "недостающего условия."
        ),
        counts_ru="«Понятно, значит откатываем редизайн».",
        not_counts_ru="«Понятно» как реплика поддержки диалога, без перехода к решению.",
        difficulty=0.4,
    ),
    # ── Ментальный фокус ──────────────────────────────────────────────────
    Indicator(
        id="mf.question",
        construct="mf",
        polarity="positive",
        definition_ru="Задаёт вопрос, ответ на который способен изменить вывод.",
        counts_ru="«Если цену не поднимали — вывод про редизайн остаётся в силе?»",
        not_counts_ru="«А что ещё интересного в данных?» — ответ не меняет решения.",
        difficulty=0.4,
    ),
    Indicator(
        id="mf.priority",
        construct="mf",
        polarity="positive",
        definition_ru="Выбирает, что проверять первым, и называет основание выбора.",
        counts_ru="«Сначала база сравнения: если она неверна, остальное считать бессмысленно».",
        not_counts_ru="«Давайте начнём с первого» — выбор без основания.",
        difficulty=0.65,
    ),
    Indicator(
        id="mf.operational",
        construct="mf",
        polarity="positive",
        definition_ru=(
            "Формулирует проверку исполнимо: какие данные, какой разрез, что будет опровержением."
        ),
        counts_ru="«Возьмите когорты по дате регистрации и сравните с той же неделей год назад».",
        not_counts_ru="«Надо проверить данные» — без того, что именно делать.",
        difficulty=0.7,
    ),
    Indicator(
        id="mf.scatter",
        construct="mf",
        polarity="negative",
        definition_ru=(
            "Задаёт вопросы, не связанные с решением, или просит ИИ выбрать за него, "
            "что проверять."
        ),
        counts_ru="«С чего бы ты начал на моём месте?»",
        not_counts_ru="«Что ещё стоит учесть?» после собственного разбора — это расширение, не делегирование.",
        difficulty=0.45,
    ),
    # ── Субъектность ──────────────────────────────────────────────────────
    Indicator(
        id="ag.own",
        construct="ag",
        polarity="positive",
        definition_ru="Выдвигает собственную версию до или независимо от предложенной ИИ.",
        counts_ru="«Моя версия — дело не в редизайне, а в цене. Проверим?»",
        not_counts_ru="«Согласен, дело в редизайне» — присоединение к чужой версии.",
        difficulty=0.5,
    ),
    Indicator(
        id="ag.attribute",
        construct="ag",
        polarity="positive",
        definition_ru="Явно разделяет, что предложил ИИ и что думает он сам.",
        counts_ru="«Ты предложил смотреть на когорты; я бы сначала проверил базу сравнения».",
        not_counts_ru="«Мы решили смотреть когорты» — авторство стёрто.",
        difficulty=0.75,
    ),
    Indicator(
        id="ag.reject",
        construct="ag",
        polarity="positive",
        definition_ru="Отклоняет предложение ИИ с названным основанием.",
        counts_ru="«Бенчмарк 23% не возьму: источник не назван».",
        not_counts_ru="«Не согласен» — отклонение без основания.",
        difficulty=0.55,
    ),
    Indicator(
        id="ag.parrot",
        construct="ag",
        polarity="negative",
        definition_ru="Воспроизводит формулировку или вывод ИИ как собственный.",
        counts_ru="Дословный повтор фразы ИИ в роли собственного вывода.",
        not_counts_ru="Цитирование ИИ с указанием, что это его слова.",
        difficulty=0.35,
    ),
    Indicator(
        id="ag.delegate",
        construct="ag",
        polarity="negative",
        definition_ru="Передаёт ИИ само решение, а не работу по его подготовке.",
        counts_ru="«Реши за меня: откатывать редизайн или нет?»",
        not_counts_ru="«Посчитай отток по когортам» — делегирование работы, не решения.",
        difficulty=0.4,
    ),
    # ── Глубокий анализ ───────────────────────────────────────────────────
    Indicator(
        id="da.evidence",
        construct="da",
        polarity="positive",
        definition_ru="Спрашивает, на чём основан вывод и чем он подтверждается.",
        counts_ru="«Откуда цифра 23%? Какой источник и за какой год?»",
        not_counts_ru="«Точно?» — сомнение без запроса оснований.",
        difficulty=0.4,
    ),
    Indicator(
        id="da.mechanism",
        construct="da",
        polarity="positive",
        definition_ru="Проверяет причинный механизм, а не совпадение во времени.",
        counts_ru="«Через что именно редизайн приводит к уходу? Где в воронке просадка?»",
        not_counts_ru="«Это может быть корреляция» — названо, но не проверено.",
        difficulty=0.7,
    ),
    Indicator(
        id="da.counter",
        construct="da",
        polarity="positive",
        definition_ru="Называет, что опровергло бы вывод, или альтернативное объяснение.",
        counts_ru="«Если у не затронутых редизайном отток такой же — версия не работает».",
        not_counts_ru="«Возможны и другие причины» — без называния.",
        difficulty=0.75,
    ),
    Indicator(
        id="da.surface",
        construct="da",
        polarity="negative",
        definition_ru="Принимает уверенность, структурность или объём ответа за обоснование.",
        counts_ru="«Раз есть бенчмарк — значит, всё подтверждается».",
        not_counts_ru="Согласие с выводом после самостоятельной проверки оснований.",
        difficulty=0.4,
    ),
    # ── Ситуативный синтез ────────────────────────────────────────────────
    Indicator(
        id="ss.update",
        construct="ss",
        polarity="positive",
        definition_ru="Меняет позицию после нового факта и называет, что именно её изменило.",
        counts_ru="«Push отключили в тот же период — тогда моя версия про редизайн слабеет».",
        not_counts_ru="«Хорошо, пусть будет так» — смена позиции без названного основания.",
        difficulty=0.5,
    ),
    Indicator(
        id="ss.integrate",
        construct="ss",
        polarity="positive",
        definition_ru="Соединяет новый факт с прежним выводом, а не заменяет один другим.",
        counts_ru="«Тогда это два разных эффекта, и их надо развести по когортам».",
        not_counts_ru="«Значит, дело в push» — полная замена версии.",
        difficulty=0.8,
    ),
    Indicator(
        id="ss.anchor",
        construct="ss",
        polarity="negative",
        definition_ru="Держится прежней позиции после факта, который её опровергает.",
        counts_ru="«Всё равно считаю, что дело в редизайне» — без разбора нового факта.",
        not_counts_ru="Удержание позиции с объяснением, почему новый факт её не затрагивает.",
        difficulty=0.45,
    ),
    Indicator(
        id="ss.flip",
        construct="ss",
        polarity="negative",
        definition_ru="Меняет позицию вслед за ИИ, не называя основания.",
        counts_ru="«Да, наверное, ты прав» сразу после возражения ИИ.",
        not_counts_ru="Смена позиции с названным основанием — это ss.update.",
        difficulty=0.4,
    ),
)

INDICATORS_BY_ID: dict[str, Indicator] = {i.id: i for i in INDICATORS}
ACTIVE_INDICATORS: tuple[Indicator, ...] = tuple(i for i in INDICATORS if not i.deprecated)


def indicators_for(construct: str) -> tuple[Indicator, ...]:
    return tuple(i for i in ACTIVE_INDICATORS if i.construct == construct)


def _validate() -> None:
    if len(INDICATORS_BY_ID) != len(INDICATORS):
        raise ValueError("duplicate indicator id")
    for ind in INDICATORS:
        if ind.construct not in CONSTRUCTS_BY_ID:
            raise ValueError(f"{ind.id}: unknown construct {ind.construct!r}")
        # The id prefix is not decoration: reports group by it, and a mismatch
        # would show an indicator under a construct it does not score.
        if not ind.id.startswith(f"{ind.construct}."):
            raise ValueError(f"{ind.id}: id must be prefixed with its construct")
        if not 0.0 <= ind.difficulty <= 1.0:
            raise ValueError(f"{ind.id}: difficulty out of range")
        if ind.discrimination <= 0:
            raise ValueError(f"{ind.id}: discrimination must be positive")
    for construct in CONSTRUCTS_BY_ID:
        found = indicators_for(construct)
        if not any(i.polarity == "positive" for i in found):
            raise ValueError(f"{construct}: needs at least one positive indicator")
        if not any(i.polarity == "negative" for i in found):
            raise ValueError(f"{construct}: needs at least one negative indicator")


_validate()
