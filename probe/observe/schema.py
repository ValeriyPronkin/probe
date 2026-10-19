"""The observer's output contract.

Ids stay plain strings rather than enums on purpose. A 21-value enum in every
request costs tokens on every call and, more importantly, makes an invented id
*impossible to express* — which means impossible to count. The drop rate is the
instrument's health metric, so the id has to be free to be wrong and then
validated after the fact.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class RawObservation(BaseModel):
    indicator_id: str = Field(
        description="Идентификатор индикатора, скопированный дословно из инвентаря."
    )
    quote: str = Field(
        description="Дословная цитата из сообщения участника, символ в символ."
    )
    strength: Literal["weak", "strong"] = Field(
        description="weak — проявление есть; strong — проявление отчётливое и развёрнутое."
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Насколько уверенно это именно данный индикатор."
    )
    task_id: str = Field(
        default="",
        description="Идентификатор задания, если наблюдение относится к нему. Иначе пустая строка.",
    )
    rationale: str = Field(
        default="",
        description="Одно предложение: почему это именно этот индикатор. Для аудита.",
    )


class ObserverOutput(BaseModel):
    observations: list[RawObservation] = Field(
        default_factory=list,
        description="Пустой список — нормальный ответ, если ничего из инвентаря не подошло.",
    )


def observer_json_schema() -> dict:
    """OpenAI strict json_schema needs every property required and no extras."""
    schema = ObserverOutput.model_json_schema()
    _tighten(schema)
    return schema


def _tighten(node: dict) -> None:
    if node.get("type") == "object":
        node["additionalProperties"] = False
        properties = node.get("properties", {})
        node["required"] = list(properties)
        for child in properties.values():
            _tighten(child)
    for child in node.get("$defs", {}).values():
        _tighten(child)
    if "items" in node and isinstance(node["items"], dict):
        _tighten(node["items"])
