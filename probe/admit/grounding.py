"""Check 1: the quote must really be in the participant's message.

Mechanical, not a prompt instruction. A model asked nicely to quote verbatim
will still paraphrase sometimes, and a paraphrased quote is an observation about
text nobody wrote. This check costs nothing and is the floor under every claim
the report makes.

Normalisation is deliberately shallow: unicode form, whitespace and case. Not
punctuation — dropping it is where a paraphrase starts to slip through.
"""

from __future__ import annotations

import re
import unicodedata

WHITESPACE = re.compile(r"\s+")


def normalise(text: str) -> str:
    return WHITESPACE.sub(" ", unicodedata.normalize("NFC", text)).strip().casefold()


def check_grounding(quote: str, participant_text: str) -> tuple[bool, str]:
    if not quote.strip():
        return False, "пустая цитата"
    if normalise(quote) in normalise(participant_text):
        return True, ""
    return False, f"цитаты нет в сообщении участника: {quote[:60]!r}"
