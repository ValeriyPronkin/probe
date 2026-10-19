"""End to end: a transcript in, a stored session and a report out.

Two ways to produce observations, and the difference matters enough that the
report states which one ran:

  "model" — the observer actually tags the turns. What the instrument does.
  "gold"  — the persona's hand-written labels stand in for the observer. This
            tests the scoring model, NOT the observer, and a report built this
            way is evidence about the arithmetic and nothing else.

Keeping the stub inside the product rather than in the test harness is
deliberate: it is the only way the report can admit to having used it.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from typing import Literal

from probe.admit.pipeline import Reviewed, review_turn
from probe.config import Config
from probe.models import ModelRole
from probe.observe.observer import Telemetry, observe_turn
from probe.observe.schema import RawObservation
from probe.report.build import Report, build_report
from probe.score.session import SessionScore, score_and_store
from probe.store import repo
from probe.transcript import Transcript

ObserverSource = Literal["model", "gold"]


@dataclass(frozen=True)
class Run:
    session_id: str
    score: SessionScore
    report: Report
    reviewed: tuple[Reviewed, ...]
    telemetry: Telemetry


def _gold_observations(transcript: Transcript) -> dict[int, list[RawObservation]]:
    by_turn: dict[int, list[RawObservation]] = {}
    for label in transcript.gold_labels:
        by_turn.setdefault(int(label["turn"]), []).append(
            RawObservation(
                indicator_id=label["indicator_id"],
                quote=_quote_for(transcript, int(label["turn"]), label),
                strength=label["strength"],
                confidence=float(label["confidence"]),
                task_id=label.get("task_id", ""),
                rationale="gold label",
            )
        )
    return by_turn


def _quote_for(transcript: Transcript, turn_idx: int, label: dict) -> str:
    """Gold labels carry an explicit quote when the distinction matters; otherwise
    the whole turn stands in. The grounding check still runs on it — a gold label
    pointing at text the participant never wrote is a fixture bug worth catching."""
    return label.get("quote") or transcript.turn(turn_idx).content


def run_transcript(
    transcript: Transcript,
    config: Config,
    connection,
    *,
    observer: ObserverSource = "model",
    adjudicator: bool = True,
    observer_role: ModelRole | None = None,
) -> Run:
    """`observer_role` picks the model behind observer="model"; the default is
    models.OBSERVER, the one every recorded session was made with."""
    session_id = str(uuid.uuid4())
    from probe.models import ADJUDICATOR, OBSERVER, PROMPT_VERSION, judge_shares_observer_family

    role = observer_role or OBSERVER

    # Every turn is labelled and reviewed before anything is stored. A provider
    # failure on any turn (IncompleteLabelling, an exhausted retry) propagates
    # from here, so a session is stored complete or not at all: no participant
    # gets a score that silently lacks one turn's evidence.
    telemetry = Telemetry()
    gold = _gold_observations(transcript) if observer == "gold" else {}

    reviewed: list[Reviewed] = []
    for turn in transcript.participant_turns():
        if observer == "model":
            raw = observe_turn(transcript, turn.idx, config, telemetry, role)
        else:
            raw = gold.get(turn.idx, [])
        if not raw:
            continue
        reviewed.extend(
            review_turn(transcript, turn.idx, raw, config, adjudicator=adjudicator)
        )

    repo.create_session(
        connection,
        repo.SessionMeta(
            id=session_id,
            participant_label=transcript.label,
            source=transcript.source,
            scenario_id=transcript.scenario_id,
            observer_model=role.model if observer == "model" else "gold-labels",
            judge_model=ADJUDICATOR.model if adjudicator else "none",
            prompt_version=PROMPT_VERSION,
        ),
    )
    for turn in transcript.turns:
        repo.append_turn(connection, session_id, turn.idx, turn.role, turn.content)
    for task_id, turn_idx in transcript.surfaced.items():
        repo.record_surfaced(connection, session_id, task_id, turn_idx)

    for item in reviewed:
        repo.record_observation(
            connection,
            session_id,
            item.observation,
            observer_model=role.model if observer == "model" else "gold-labels",
            prompt_version=PROMPT_VERSION,
            verdicts=item.verdicts,
        )

    score = score_and_store(connection, session_id)
    report = build_report(transcript, score, reviewed, telemetry if observer == "model" else None)
    if observer == "gold":
        report.notes_ru.insert(
            0,
            "Наблюдения взяты из эталонной разметки, модель не вызывалась: "
            "этот отчёт проверяет арифметику прибора, а не работу наблюдателя.",
        )
    if not adjudicator:
        report.notes_ru.append("Второй судья отключён — проверка допустимости неполна.")
    elif observer == "model" and judge_shares_observer_family(role):
        report.notes_ru.append(
            f"Судья ({ADJUDICATOR.model}) — того же семейства, что и наблюдатель "
            f"({role.model}): он отсекает часть ошибок разметки, но его согласие с "
            "наблюдателем не является независимой проверкой."
        )

    return Run(
        session_id=session_id,
        score=score,
        report=report,
        reviewed=tuple(reviewed),
        telemetry=telemetry,
    )
