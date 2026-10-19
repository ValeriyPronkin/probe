"""Stage 5: the report.

Three things it must do, and the third is the one usually missing.

1. Show the posterior, not a grade. A distribution over three levels says what a
   single number cannot: how sure the instrument is.
2. Put the participant's own words under every construct. A score without the
   quote that produced it is unfalsifiable.
3. Name what was never observed. A task the scenario did not surface, and a
   construct with no evidence, are reported as gaps — not as low scores. This is
   the whole reason the task model exists, and a report that quietly prints 0.5
   for an unobserved construct throws it away.

Rejected evidence gets its own section for the same reason: "we dropped this as
borrowed from the AI" is a different statement than a low score, and the person
reading the report is entitled to the difference.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from probe.admit.pipeline import Reviewed
from probe.inventory.constructs import CONSTRUCTS, CONSTRUCTS_BY_ID, LEVEL_VALUES
from probe.inventory.indicators import INDICATORS_BY_ID
from probe.observe.observer import Telemetry
from probe.scenario.tasks import TASKS, TASKS_BY_ID
from probe.score.aggregate import NodeEvidence
from probe.score.bayes import expected_level
from probe.score.session import SessionScore
from probe.transcript import Transcript

LEVEL_LABELS_RU = {"low": "низкий", "mid": "средний", "high": "высокий"}


@dataclass(frozen=True)
class ConstructReport:
    id: str
    title_ru: str
    posterior: dict[str, float]
    expected: float
    evidence: tuple[NodeEvidence, ...]
    observed: bool
    """False when nothing at all bore on this construct — the number is the prior."""


@dataclass(frozen=True)
class Report:
    label: str
    source: str
    scenario_id: str
    fingerprint: str
    constructs: tuple[ConstructReport, ...]
    surfaced: tuple[str, ...]
    not_surfaced: tuple[str, ...]
    rejected: tuple[Reviewed, ...]
    telemetry: Telemetry | None = None
    notes_ru: list[str] = field(default_factory=list)


def _bearing_constructs(evidence: NodeEvidence) -> tuple[str, ...]:
    if evidence.kind == "task":
        return TASKS_BY_ID[evidence.node.removeprefix("task:")].constructs
    return (INDICATORS_BY_ID[evidence.node].construct,)


def build_report(
    transcript: Transcript,
    score: SessionScore,
    reviewed: Sequence[Reviewed],
    telemetry: Telemetry | None = None,
) -> Report:
    surfaced = tuple(sorted(transcript.surfaced))
    surfaced_from_evidence = {
        e.node.removeprefix("task:") for e in score.evidence if e.kind == "task"
    }
    all_surfaced = tuple(sorted(set(surfaced) | surfaced_from_evidence))
    not_surfaced = tuple(t.id for t in TASKS if t.id not in all_surfaced)

    constructs: list[ConstructReport] = []
    for construct in CONSTRUCTS:
        bearing = tuple(e for e in score.evidence if construct.id in _bearing_constructs(e))
        constructs.append(
            ConstructReport(
                id=construct.id,
                title_ru=construct.title_ru,
                posterior=score.posteriors[construct.id],
                expected=expected_level(score.posteriors[construct.id], LEVEL_VALUES),
                evidence=bearing,
                observed=bool(bearing),
            )
        )

    notes: list[str] = []
    if transcript.source == "persona":
        notes.append(
            "Источник — синтетическая персона, написанная автором прибора. "
            "Это проверка согласованности, а не измерение человека."
        )
    if not_surfaced:
        notes.append(
            f"Сценарий не предъявил {len(not_surfaced)} из {len(TASKS)} заданий — "
            "по ним выводов нет и быть не может."
        )

    return Report(
        label=transcript.label,
        source=transcript.source,
        scenario_id=transcript.scenario_id,
        fingerprint=score.fingerprint,
        constructs=tuple(constructs),
        surfaced=all_surfaced,
        not_surfaced=not_surfaced,
        rejected=tuple(r for r in reviewed if not r.admitted),
        telemetry=telemetry,
        notes_ru=notes,
    )


def _bar(value: float, width: int = 24) -> str:
    filled = round(value * width)
    return "█" * filled + "·" * (width - filled)


def render_text(report: Report) -> str:
    out: list[str] = []
    out.append(f"СЕССИЯ {report.label}  ·  источник: {report.source}  ·  сценарий: {report.scenario_id}")
    out.append(f"параметры прибора: {report.fingerprint}")
    for note in report.notes_ru:
        out.append(f"  ! {note}")
    out.append("")

    out.append("ОЦЕНКА ПО КОНСТРУКТАМ")
    for construct in report.constructs:
        out.append("")
        marker = "" if construct.observed else "   (свидетельств нет — показан априор)"
        out.append(f"  {construct.title_ru} [{construct.id}]{marker}")
        out.append(f"    {_bar(construct.expected)}  ожидание {construct.expected:.2f}")
        levels = "  ".join(
            f"{LEVEL_LABELS_RU[level]} {probability:.0%}"
            for level, probability in construct.posterior.items()
        )
        out.append(f"    {levels}")
        for evidence in construct.evidence:
            if evidence.kind == "task":
                task = TASKS_BY_ID[evidence.node.removeprefix("task:")]
                state = {"caught": "вскрыл", "partial": "частично", "missed": "пропустил"}[
                    evidence.state
                ]
                out.append(f"    · {task.id} {state} — {task.summary_ru}")
            else:
                indicator = INDICATORS_BY_ID[evidence.node]
                out.append(f"    · {indicator.id} ({evidence.state}) — {indicator.definition_ru}")
            # Two indicators can rest on the same sentence; printing it twice
            # reads as two pieces of evidence when there is one.
            seen: set[tuple[int, str]] = set()
            for source in evidence.sources:
                key = (source.turn_index, source.quote)
                if key in seen:
                    continue
                seen.add(key)
                out.append(f'        ход {source.turn_index}: «{source.quote}»')
            if evidence.kind == "task" and not evidence.sources:
                out.append("        задание предъявлено, реакции не было")

    if report.not_surfaced:
        out.append("")
        out.append("НЕ НАБЛЮДАЛОСЬ")
        out.append("  Эти задания сценарий не предъявил. Молчание участника по ним ничего не значит.")
        for task_id in report.not_surfaced:
            out.append(f"    {task_id} — {TASKS_BY_ID[task_id].summary_ru}")

    if report.rejected:
        out.append("")
        out.append("ОТКЛОНЁННЫЕ ДОКАЗАТЕЛЬСТВА")
        out.append("  Это не низкий балл. Это материал, который прибор отказался засчитывать.")
        for item in report.rejected:
            reason = next((d for _, passed, d in item.verdicts if not passed), "")
            out.append(
                f"    ход {item.observation.turn_index}  {item.observation.indicator_id}"
                f"  [{item.failed_check}] {reason}"
            )
            out.append(f'        «{item.observation.quote}»')

    if report.telemetry and report.telemetry.turns_observed:
        t = report.telemetry
        out.append("")
        out.append("ЗДОРОВЬЕ ПРИБОРА")
        out.append(
            f"  ходов размечено: {t.turns_observed}   из них пустых: {t.turns_empty}   "
            f"наблюдений: {t.emitted}"
        )
        out.append(
            f"  отброшено по неизвестному id: {t.dropped_unknown_indicator} "
            f"({t.drop_rate:.0%})   по неизвестному заданию: {t.dropped_unknown_task}   "
            f"с нарушением схемы: {t.dropped_malformed}   ответов без обёртки: {t.envelope_broken}"
        )
        if t.unknown_ids:
            out.append(f"  выдуманные id: {', '.join(sorted(set(t.unknown_ids)))}")

    return "\n".join(out)
