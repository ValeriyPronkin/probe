"""The student model: five latent constructs under one shared parent.

Definitions live in design/00-measurement-model.md. This module is the single
place that knows construct ids exist; everything else imports from here so a
typo fails at import rather than silently scoring nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

# Ordered worst-to-best. The order is load-bearing: the graded-response link in
# cpt.py maps position to ability, so reordering silently inverts every table.
LEVELS: tuple[str, ...] = ("low", "mid", "high")

# Ability value each level stands for, on the [0, 1] scale the CPT builders use.
LEVEL_VALUES: dict[str, float] = {"low": 0.0, "mid": 0.5, "high": 1.0}

DISPOSITION = "disposition"

# Neutral start, not a population estimate: the instrument must be moved by
# evidence rather than arrive with an opinion.
DISPOSITION_PRIOR: tuple[float, ...] = (0.25, 0.50, 0.25)

# How strongly the shared disposition pulls each construct: the probability a
# construct sits at its disposition's level. One number with a meaning a
# methodologist can argue with, instead of nine hand-typed cells.
#
# Chosen by behaviour, not taste. The quantity that matters is cross-construct
# bleed — how far one observation drags the four constructs it says nothing
# about. Catching T1 alone moves its own constructs to ~0.69 on the ability
# scale at every setting tried; the others move:
#     0.45 -> 0.54    0.55 -> 0.56    0.70 -> 0.62    0.80 -> 0.66
# At 0.70 a single caught flaw adds +0.12 to constructs with no evidence at all,
# which reads as the instrument inventing scores. At 0.55 the pull is real but
# small, and direct evidence still dominates. Lower still would make the shared
# parent decorative.
DISPOSITION_STRENGTH = 0.55


@dataclass(frozen=True)
class Construct:
    id: str
    title_ru: str
    definition_ru: str


CONSTRUCTS: tuple[Construct, ...] = (
    Construct(
        id="kn",
        title_ru="Знание",
        definition_ru=(
            "Замечает существенное неизвестное: называет факт, которого нет "
            "в данных, и без которого вывод не держится."
        ),
    ),
    Construct(
        id="mf",
        title_ru="Ментальный фокус",
        definition_ru=(
            "Определяет, что выяснить или проверить, и формулирует проверку исполнимо."
        ),
    ),
    Construct(
        id="ag",
        title_ru="Субъектность",
        definition_ru=(
            "Отделяет собственные идеи от вклада ИИ и сохраняет авторство решения."
        ),
    ),
    Construct(
        id="da",
        title_ru="Глубокий анализ",
        definition_ru="Различает ответ и достаточность оснований для него.",
    ),
    Construct(
        id="ss",
        title_ru="Ситуативный синтез",
        definition_ru="Пересматривает мнение при новой информации — и только при ней.",
    ),
)

CONSTRUCT_IDS: tuple[str, ...] = tuple(c.id for c in CONSTRUCTS)
CONSTRUCTS_BY_ID: dict[str, Construct] = {c.id: c for c in CONSTRUCTS}


def _validate() -> None:
    if len(CONSTRUCTS_BY_ID) != len(CONSTRUCTS):
        raise ValueError("duplicate construct id")
    if len(DISPOSITION_PRIOR) != len(LEVELS):
        raise ValueError("disposition prior must cover every level")
    if abs(sum(DISPOSITION_PRIOR) - 1.0) > 1e-9:
        raise ValueError("disposition prior must sum to 1")
    if set(LEVEL_VALUES) != set(LEVELS):
        raise ValueError("LEVEL_VALUES must cover exactly LEVELS")


_validate()
