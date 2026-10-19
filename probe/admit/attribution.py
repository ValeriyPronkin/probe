"""Check 2: the participant's words must not be the AI's words.

This is the instrument's reading of "вклад ИИ не приписывается человеку", and it
is honest only about the literal half of that principle. Verbatim reuse is
caught; a participant who restates the model's idea in their own words is not.
That gap is stated plainly in the README because it is the weakest point of the
whole instrument, and pretending otherwise would be the one failure that matters.

What is measured: how much of the quote is covered by word n-grams that already
appeared in an earlier assistant message. Coverage, not a single hit, because one
shared phrase in a long original sentence is normal Russian, while half a
sentence lifted whole is not.
"""

from __future__ import annotations

import re

TOKEN = re.compile(r"\w+", re.UNICODE)

NGRAM_SIZE = 5
"""Shorter catches ordinary shared phrasing; longer misses real copying. Five
words is the point where a match stops being a coincidence of Russian syntax."""

MAX_COVERAGE = 0.5
"""Half the quote lifted verbatim is no longer the participant's sentence."""

MIN_TOKENS = 5
"""Below this the check is not applicable, and saying so is the point.

Found by the `gaming` persona: answering the menu probe with "Базу сравнения."
is exactly right, and necessarily repeats the option's own words. Scoring that
as borrowed would penalise every terse-but-correct answer — a systematic bias
against concise participants, dressed up as an integrity check. A quote this
short carries little evidence either way; the check passes and records that it
could not be applied."""


def tokens(text: str) -> list[str]:
    return [t.casefold() for t in TOKEN.findall(text)]


def ngrams(words: list[str], size: int) -> set[tuple[str, ...]]:
    if len(words) < size:
        return set()
    return {tuple(words[i : i + size]) for i in range(len(words) - size + 1)}


def coverage(quote: str, prior_assistant_text: str, size: int = NGRAM_SIZE) -> float:
    """Fraction of the quote's tokens sitting inside a borrowed n-gram."""
    quote_tokens = tokens(quote)
    if not quote_tokens:
        return 0.0

    if len(quote_tokens) < MIN_TOKENS:
        return 0.0

    source = ngrams(tokens(prior_assistant_text), size)
    borrowed: set[int] = set()
    for i in range(len(quote_tokens) - size + 1):
        if tuple(quote_tokens[i : i + size]) in source:
            borrowed.update(range(i, i + size))
    return len(borrowed) / len(quote_tokens)


def check_attribution(
    quote: str, prior_assistant_messages: list[str], size: int = NGRAM_SIZE
) -> tuple[bool, str]:
    if len(tokens(quote)) < MIN_TOKENS:
        return True, f"неприменимо: цитата короче {MIN_TOKENS} слов"
    joined = "\n".join(prior_assistant_messages)
    ratio = coverage(quote, joined, size)
    if ratio > MAX_COVERAGE:
        return False, f"{ratio:.0%} цитаты дословно совпадает с более ранними репликами ИИ"
    return True, ""
