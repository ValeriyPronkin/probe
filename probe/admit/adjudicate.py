"""Check 4: an independent judge, prompted to refute.

Two design choices carry the weight here.

A different model family from the observer. Same-family judges share blind spots,
so their agreement says something about the pair of models rather than about the
participant, and the agreement rate stops being a usable metric. (The demo
prototype breaks this knowingly — see probe/models.py — and the report says so.)

Adversarial framing. The judge is asked to knock the observation down, not to
confirm it, and told to refute when unsure. A judge asked "is this right?"
agrees with almost anything that looks plausible.

Disagreement between observer and judge is not a bug to tune away. It is the
number the validation section reports.

Two things the first live run showed the judge needs. The AI's preceding reply:
without it "repeats the AI's wording" or "flips after the AI objects" cannot be
checked at all, so a judge told to refute when unsure refuted every one. And
polarity-aware criteria: "the participant only agrees, with no action of their
own" is a reason to refute a positive indicator and the very behaviour a
negative one describes.
"""

from __future__ import annotations

from dataclasses import dataclass

from pydantic import BaseModel, Field

from probe.config import Config
from probe.inventory.indicators import INDICATORS_BY_ID
from probe.llm import Request, complete_json
from probe.models import ADJUDICATOR
from probe.observe.schema import _tighten

SYSTEM_RU = """\
Ты — независимый проверяющий в измерительной системе. Твоя задача — **опровергнуть** разметку,
а не подтвердить её.

Тебе дают: определение индикатора и его полярность, пример того, что считается, пример того,
что не считается, предыдущую реплику ИИ, цитату из сообщения участника и само сообщение целиком.

Полярность важна. Положительный индикатор описывает сильное действие участника. Отрицательный
описывает слабость: согласие без проверки, повтор слов ИИ, передачу решения ИИ. Для
отрицательного индикатора то, что участник лишь соглашается или повторяет, — не повод
опровергать, а само проявление.

Опровергай, если верно хотя бы одно:
— цитата показывает поведение, но не это, а другое (или вообще не из инвентаря);
— цитата слишком коротка или вырвана так, что поведение из неё не следует;
— это ближе к примеру «не считается», чем к примеру «считается»;
— для положительного индикатора: участник лишь повторяет или соглашается, своего действия нет;
— для индикатора, который описывает реакцию на слова ИИ: в предыдущей реплике ИИ этого нет.

Если сомневаешься — опровергай. Ошибка в сторону пропуска доказательства дешевле, чем
ошибка в сторону приписывания человеку того, чего он не делал.
"""


class Verdict(BaseModel):
    refuted: bool = Field(description="true — разметку следует отклонить")
    reason: str = Field(description="Одно предложение. На русском.")


def _schema() -> dict:
    schema = Verdict.model_json_schema()
    _tighten(schema)
    return schema


@dataclass(frozen=True)
class Adjudication:
    passed: bool
    detail: str


def adjudicate(
    indicator_id: str,
    quote: str,
    participant_text: str,
    previous_ai_text: str,
    config: Config,
) -> Adjudication:
    indicator = INDICATORS_BY_ID[indicator_id]
    polarity = "положительный" if indicator.polarity == "positive" else "отрицательный"
    user = (
        f"ИНДИКАТОР: {indicator.id}\n"
        f"ПОЛЯРНОСТЬ: {polarity}\n"
        f"ОПРЕДЕЛЕНИЕ: {indicator.definition_ru}\n"
        f"СЧИТАЕТСЯ: {indicator.counts_ru}\n"
        f"НЕ СЧИТАЕТСЯ: {indicator.not_counts_ru}\n\n"
        f"ПРЕДЫДУЩАЯ РЕПЛИКА ИИ:\n{previous_ai_text}\n\n"
        f"ЦИТАТА: {quote}\n\n"
        f"СООБЩЕНИЕ УЧАСТНИКА ЦЕЛИКОМ:\n{participant_text}"
    )
    response = complete_json(
        Request(
            role=ADJUDICATOR,
            system=SYSTEM_RU,
            user=user,
            schema_name="verdict",
            schema=_schema(),
        ),
        config,
    )
    verdict = Verdict.model_validate(response)
    return Adjudication(passed=not verdict.refuted, detail=verdict.reason if verdict.refuted else "")
