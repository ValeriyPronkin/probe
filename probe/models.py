"""Model roles in one place.

Two rules worth stating, because both are easy to lose in a refactor:

1. The observer and the adjudicator must not be the same model family. A judge
   that shares the observer's blind spots agrees with it for reasons that have
   nothing to do with the participant, and the agreement rate stops being
   evidence of anything.
2. Temperature is zero everywhere. The instrument's reliability check measures
   how stable a score is across runs; sampling noise would be measuring the
   provider, not the instrument.
"""

from __future__ import annotations

from dataclasses import dataclass

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
DEEPSEEK_BASE_URL = "https://api.deepseek.com"

APP_HEADERS = {
    "HTTP-Referer": "https://github.com/probe",
    "X-Title": "probe — critical thinking instrument",
}


@dataclass(frozen=True)
class ModelRole:
    name: str
    model: str
    provider: str = "openrouter"
    """'openrouter' or 'deepseek'. Part of the cassette key: the same model id
    behind two providers is not guaranteed to be the same model."""
    family: str = ""
    """Who trained it. Rule 1 is about this, not about the model id."""
    temperature: float = 0.0


# A small, cheap model family, so the comparison is on method rather than on
# model tier. Called directly at DeepSeek for the demo prototype.
# `deepseek-flash` is a floating alias — on 2026-10-11 it served
# DeepSeek-V4.1-Flash, not the V4 Flash it was planned for — so every live
# response records the model that actually answered.
OBSERVER = ModelRole(
    name="observer", model="deepseek-flash", provider="deepseek", family="deepseek"
)

# A known breach of rule 1, accepted for the demo prototype: a different and
# stronger model, but the same family, because only a DeepSeek key is at hand.
# The report says so on every run. The intended judge is one line away:
#   ModelRole(name="adjudicator", model="google/gemini-3.8-flash", family="google")
ADJUDICATOR = ModelRole(
    name="adjudicator", model="deepseek-v4-pro", provider="deepseek", family="deepseek"
)


def judge_shares_observer_family() -> bool:
    return ADJUDICATOR.family == OBSERVER.family

# The scenario's conversational surface. Narrow by construction: it presents a
# scripted state machine's current step and never evaluates anything.
INTERLOCUTOR = ModelRole(
    name="interlocutor", model="deepseek-flash", provider="deepseek", family="deepseek"
)

PROMPT_VERSION = "v3"
"""Bumped whenever any prompt text changes. Stored on every observation so a
score can be traced to the wording that produced it."""

INTERLOCUTOR_PROMPT_VERSION = "i4"
"""The interlocutor's own counter, kept apart from PROMPT_VERSION: bumping that
one re-keys every observer cassette, and the interlocutor never feeds a score.
Stored in each live transcript's audit. i1 -> i2 (2026-10-12): the first live
session showed the interlocutor answering a statement nobody asked about and
inventing how the 890 was counted; a statement now gets no reaction. i2 -> i3:
a trap's own facts (the holiday base, the price rise) are confirm-only. i3 -> i4:
confirm-only facts get their own section (an inline note made the model refuse
even a direct question), and «в чём вопрос?» gets no hint at the trap."""
