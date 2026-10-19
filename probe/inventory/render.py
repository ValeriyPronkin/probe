"""Renders the inventory into the prompt preamble.

This block goes into *every* prompt that tags. A tagger without the labelled
inventory in front of it does not refuse — it invents plausible-looking ids, and
every one of them dies at validation, so a reply full of real behaviour records
nothing at all. The failure is silent, which is what makes it dangerous.

The text is deterministic and identical across calls, which also makes it the
cacheable prefix of the request.
"""

from __future__ import annotations

from probe.inventory.constructs import CONSTRUCTS
from probe.inventory.indicators import indicators_for
from probe.scenario.tasks import TASKS

PRINCIPLE_RU = """\
ПРАВИЛО РАЗМЕТКИ, ВАЖНЕЕ ОСТАЛЬНЫХ

Используй только идентификаторы из списка ниже, скопированные дословно. Никогда не придумывай
новый идентификатор и не переводи существующий. Если ни один индикатор не подходит — верни пустой
список наблюдений. Пустой список не является ошибкой; выдуманный идентификатор является.

Каждое наблюдение обязано нести дословную цитату из сообщения участника. Цитата должна
встречаться в его сообщении буквально, символ в символ. Не перефразируй, не исправляй опечатки,
не склеивай куски из разных мест.

Размечай только то, что участник сделал сам. Пересказ слов ИИ — не наблюдение о человеке.
"""


def render_constructs() -> str:
    lines = ["КОНСТРУКТЫ"]
    for construct in CONSTRUCTS:
        lines.append(f"  {construct.id} — {construct.title_ru}: {construct.definition_ru}")
    return "\n".join(lines)


def render_indicators() -> str:
    lines = ["ИНДИКАТОРЫ (закрытый список)"]
    for construct in CONSTRUCTS:
        lines.append(f"\n  {construct.id} — {construct.title_ru}")
        for indicator in indicators_for(construct.id):
            sign = "+" if indicator.polarity == "positive" else "−"
            lines.append(f"    {indicator.id}  [{sign}]  {indicator.definition_ru}")
            lines.append(f"        считается:    {indicator.counts_ru}")
            lines.append(f"        не считается: {indicator.not_counts_ru}")
    return "\n".join(lines)


def render_tasks() -> str:
    lines = [
        "ЗАДАНИЯ СЦЕНАРИЯ",
        "  Если наблюдение относится к одному из них — укажи его идентификатор.",
        "  Если ни к одному, оставь поле пустым. Не угадывай.",
    ]
    for task in TASKS:
        lines.append(f"    {task.id} — {task.summary_ru}")
    return "\n".join(lines)


def render_inventory() -> str:
    return "\n\n".join([PRINCIPLE_RU.rstrip(), render_constructs(), render_indicators(), render_tasks()])


def valid_indicator_ids() -> frozenset[str]:
    """What post-hoc validation checks against. The request schema keeps the id a
    plain string on purpose: a 21-value enum bloats every request, and an invalid
    id has to be *counted* (it is the drop-rate metric) rather than made
    impossible to express."""
    from probe.inventory.indicators import ACTIVE_INDICATORS

    return frozenset(i.id for i in ACTIVE_INDICATORS)
