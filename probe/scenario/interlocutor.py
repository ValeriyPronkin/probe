"""The interlocutor: one short reaction between two scripted steps. design/00 §3.

Narrow by construction. It sees the step the participant just answered, their
answer and the case facts already open — not the inventory, not the tasks'
names, not any score. It never presents a task (the script does) and never
decides when the scenario moves on (the script does). Its whole job is to keep
the dialogue from reading like a questionnaire without changing what is measured.

An empty reply is the normal answer to anything but a direct question: a
participant who states a view needs no reply, and a reaction invented to fill
the space is variance with no purpose. The first live session showed exactly
that — an objection answered with a fact the sheet does not hold.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from probe.config import Config
from probe.llm import Request, complete_json
from probe.models import INTERLOCUTOR
from probe.observe.schema import _tighten
from probe.scenario.script import STEPS, LiveSession, open_facts

SYSTEM_RU = """\
Ты — ИИ-аналитик, который вместе с собеседником разбирает рост оттока после редизайна приложения.
Ход разговора задан заранее: следующую реплику разбора система добавит сама сразу после твоей.
Твоя единственная работа — коротко отреагировать на последнее сообщение собеседника.

Правила:
- Реагируй только на прямой вопрос или просьбу. Если собеседник высказывает мнение, возражает
  или соглашается, ничего не спрашивая, верни пустую строку.
- Не больше двух предложений, и только о том, о чём спросили. Не сообщай сам, каких данных нет.
- Отвечай только по фактам ниже, не достраивай их. Если ответа там нет, скажи, что этих данных
  у тебя под рукой нет. Не утверждай, что чего-то не было, и не придумывай чисел.
- Если собеседник спрашивает, в чём вопрос или что от него нужно, ответь только, что тебе
  интересно его мнение о твоей последней реплике. Не говори, на что в ней обратить внимание.
- Не оценивай собеседника: не хвали, не соглашайся с его критикой и не спорь с ней.
- Не признавай и не исправляй ошибок в своих прежних выводах, пока о них прямо не спросили.
- Не предлагай собеседнику, что проверить или как рассуждать.
- Не переходи к следующему шагу разбора и не пересказывай его: это сделает система.

ФАКТЫ, КОТОРЫЕ МОЖНО СООБЩАТЬ, когда о них спрашивают:
{facts}

ФАКТЫ, КОТОРЫЕ МОЖНО ТОЛЬКО ПОДТВЕРДИТЬ. Если собеседник сам назвал именно это (например,
спросил о цене или о праздниках), подтверди прямо. Если не назвал — не упоминай их вовсе, даже
когда спрашивают, что ещё происходило; тогда отвечай, что этих данных у тебя под рукой нет.
{confirm_only}
"""


class InterlocutorOutput(BaseModel):
    reply: str = Field(
        description="Короткая реакция на сообщение собеседника; пустая строка, если отвечать не на что."
    )


def interlocutor_json_schema() -> dict:
    schema = InterlocutorOutput.model_json_schema()
    _tighten(schema)
    return schema


def build_prompt(session: LiveSession, answer: str) -> tuple[str, str]:
    """The facts open after the step being answered now, i.e. one past `answered`."""
    opened = open_facts(session.answered + 1)
    facts = "\n".join(f"- {f.text_ru}" for f in opened if not f.confirm_only)
    confirm_only = "\n".join(f"- {f.text_ru}" for f in opened if f.confirm_only) or "- (пока нет)"
    user = (
        f"ТВОЯ ПОСЛЕДНЯЯ РЕПЛИКА:\n{session.last_assistant}\n\n"
        f"СООБЩЕНИЕ СОБЕСЕДНИКА:\n{answer.strip()}"
    )
    return SYSTEM_RU.format(facts=facts, confirm_only=confirm_only), user


def react(session: LiveSession, answer: str, config: Config) -> str:
    """The reaction to `answer`, before the session records it. Returns "" once
    the scenario is on its last step: nothing follows the final answer."""
    if session.answered + 1 >= len(STEPS) - 1:
        return ""
    system, user = build_prompt(session, answer)
    response = complete_json(
        Request(
            role=INTERLOCUTOR,
            system=system,
            user=user,
            schema_name="reaction",
            schema=interlocutor_json_schema(),
        ),
        config,
    )
    return InterlocutorOutput.model_validate(response).reply.strip()
