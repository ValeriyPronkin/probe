"""Stage 3: run every admissibility check and record each verdict separately.

An observation is admitted only if no check failed. Verdicts are stored per
check, not collapsed into a boolean, because the report has to be able to say
*why* something was excluded — "evidence was dropped as borrowed from the AI" is
a different statement to the participant than a low score, and conflating them is
how a measurement instrument turns into an opaque one.

Order is cheapest-first, and the adjudicator runs last because it costs a call.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from probe.admit.adjudicate import adjudicate
from probe.admit.attribution import check_attribution
from probe.admit.grounding import check_grounding
from probe.admit.meta_turns import check_meta_turn
from probe.config import Config
from probe.inventory.indicators import INDICATORS_BY_ID
from probe.observe.schema import RawObservation
from probe.score.aggregate import Observation
from probe.transcript import Transcript

Verdict = tuple[str, bool, str]


@dataclass(frozen=True)
class Reviewed:
    observation: Observation
    verdicts: tuple[Verdict, ...]

    @property
    def admitted(self) -> bool:
        return all(passed for _, passed, _ in self.verdicts)

    @property
    def failed_check(self) -> str:
        for name, passed, _ in self.verdicts:
            if not passed:
                return name
        return ""


def review_turn(
    transcript: Transcript,
    turn_idx: int,
    raw: Sequence[RawObservation],
    config: Config,
    *,
    adjudicator: bool = True,
) -> list[Reviewed]:
    turn = transcript.turn(turn_idx)
    prior_ai = transcript.assistant_text_before(turn_idx)
    meta_passed, meta_detail = check_meta_turn(turn.content)

    reviewed: list[Reviewed] = []
    for item in raw:
        verdicts: list[Verdict] = [("meta_turn", meta_passed, meta_detail)]

        grounded, detail = check_grounding(item.quote, turn.content)
        verdicts.append(("grounding", grounded, detail))

        if grounded and INDICATORS_BY_ID[item.indicator_id].polarity == "negative":
            # Attribution keeps the AI's contribution from being credited to the
            # participant. A negative indicator credits nothing, and for ag.parrot
            # the overlap with the AI's words is the behaviour itself: filtering
            # it made the indicator inadmissible by construction.
            attributed = True
            verdicts.append(
                ("attribution", True, "неприменимо: отрицательный индикатор не приписывает заслугу")
            )
        elif grounded:
            attributed, detail = check_attribution(item.quote, prior_ai)
            verdicts.append(("attribution", attributed, detail))
        else:
            # Attribution compares the participant's words to the AI's. With no
            # real quote there is nothing to compare, and a cheerful "passed"
            # would read as "we checked" when we did not.
            attributed = False

        if adjudicator and meta_passed and grounded and attributed:
            prior = transcript.assistant_text_before(turn.idx)
            adjudication = adjudicate(
                item.indicator_id,
                item.quote,
                turn.content,
                prior[-1] if prior else "(диалог ещё не начался)",
                config,
            )
            verdicts.append(("adjudication", adjudication.passed, adjudication.detail))

        reviewed.append(
            Reviewed(
                observation=Observation(
                    indicator_id=item.indicator_id,
                    turn_index=turn_idx,
                    strength=item.strength,
                    confidence=item.confidence,
                    quote=item.quote,
                    task_id=item.task_id or None,
                ),
                verdicts=tuple(verdicts),
            )
        )
    return reviewed


def admitted(reviewed: Sequence[Reviewed]) -> tuple[Observation, ...]:
    return tuple(r.observation for r in reviewed if r.admitted)


def rejected(reviewed: Sequence[Reviewed]) -> tuple[Reviewed, ...]:
    return tuple(r for r in reviewed if not r.admitted)
