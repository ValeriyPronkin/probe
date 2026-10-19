"""Check 3: turns about the assessment, and technical failures, score nothing.

Their principle, written as a condition rather than repeated as a value: asking
"как ты меня оцениваешь?" is not a measurement occasion, and neither is the
participant reporting that the page froze.

The patterns are narrow on purpose. "Здесь ошибка" is a participant catching a
flaw — the single most valuable thing they can do — so the failure patterns match
only reports about *the system*, never the word on its own. A wide filter here
would silently delete the best evidence the instrument collects.
"""

from __future__ import annotations

import re

ABOUT_THE_ASSESSMENT = (
    r"как ты меня (оцениваешь|оценива)",
    r"(это|у нас тут) (какой-то )?(тест|экзамен|проверка)",
    r"(какой|сколько) у меня (балл|оценка|результат)",
    r"ты меня (проверяешь|тестируешь|оцениваешь)",
    r"что ты (обо мне )?(записываешь|фиксируешь)",
    r"меня (сейчас )?(оценивают|тестируют)",
)

ABOUT_A_FAILURE = (
    r"(ничего )?не (работает|грузится|отправляется|открывается)",
    r"(страница|приложение|сайт|чат|бот) (завис|не отвеча|сломал)",
    r"(произошла|какая-то) ошибка (сервера|системы|приложения)",
    r"(перезагруз|обнови) (страницу|чат)",
)

_ASSESSMENT = [re.compile(p, re.IGNORECASE) for p in ABOUT_THE_ASSESSMENT]
_FAILURE = [re.compile(p, re.IGNORECASE) for p in ABOUT_A_FAILURE]


def classify_turn(text: str) -> str:
    """'substantive' | 'about_assessment' | 'about_failure'."""
    if any(pattern.search(text) for pattern in _ASSESSMENT):
        return "about_assessment"
    if any(pattern.search(text) for pattern in _FAILURE):
        return "about_failure"
    return "substantive"


def check_meta_turn(participant_text: str) -> tuple[bool, str]:
    kind = classify_turn(participant_text)
    if kind == "substantive":
        return True, ""
    return False, f"ход не является измерительным: {kind}"
