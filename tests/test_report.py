"""The report must say what it does not know as plainly as what it does."""

import re
from pathlib import Path

import pytest

from probe.config import replay_only
from probe.report.build import build_report, render_text
from probe.run import run_transcript
from probe.scenario.tasks import TASKS
from probe.store import repo
from probe.transcript import Transcript, Turn, load


def run(label: str, **kwargs):
    transcript = load(Path("fixtures/personas") / f"{label}.json")
    with repo.connect() as connection:
        return run_transcript(
            transcript, replay_only(), connection, observer="gold", adjudicator=False, **kwargs
        )


def test_every_score_carries_the_participants_own_words():
    text = render_text(run("strong").report)
    assert "Подожди. 1 240 — это из скольки?" in text


def test_a_quote_is_not_repeated_inside_one_evidence_block():
    """Two indicators can rest on one sentence; listing it twice under the same
    task reads as two pieces of evidence when there is one.

    Across blocks, repetition is meaningful and kept: one sentence really can
    catch two different flaws, and hiding that would understate the participant.
    """
    text = render_text(run("strong").report)
    blocks: list[list[str]] = []
    for line in text.splitlines():
        if line.strip().startswith("· "):
            blocks.append([])
        elif blocks and line.strip().startswith("ход "):
            blocks[-1].append(line.strip())

    assert blocks, "no evidence blocks found in the report"
    for quotes in blocks:
        assert len(quotes) == len(set(quotes)), quotes


def test_one_sentence_catching_two_flaws_is_credited_to_both():
    text = render_text(run("strong").report)
    section = text.split("Знание [kn]")[1].split("Ментальный фокус")[0]
    assert "T1 вскрыл" in section
    assert "T4 вскрыл" in section


def test_unsurfaced_tasks_are_reported_as_gaps_not_as_low_scores():
    short = Transcript(
        label="short",
        source="persona",
        scenario_id="churn-v1",
        turns=(Turn(0, "assistant", "Ушло 1240 человек."), Turn(1, "participant", "Ага.")),
        surfaced={"T4": 0},
    )
    with repo.connect() as connection:
        result = run_transcript(
            short, replay_only(), connection, observer="gold", adjudicator=False
        )
    text = render_text(result.report)
    assert "НЕ НАБЛЮДАЛОСЬ" in text
    assert len(result.report.not_surfaced) == len(TASKS) - 1
    assert "Молчание участника по ним ничего не значит" in text


def test_a_construct_with_no_evidence_is_labelled_as_such():
    short = Transcript(
        label="short",
        source="persona",
        scenario_id="churn-v1",
        turns=(Turn(0, "assistant", "Ушло 1240 человек."), Turn(1, "participant", "Ага.")),
        surfaced={},
    )
    with repo.connect() as connection:
        result = run_transcript(
            short, replay_only(), connection, observer="gold", adjudicator=False
        )
    text = render_text(result.report)
    assert "свидетельств нет — показан априор" in text
    assert all(not c.observed for c in result.report.constructs)
    assert all(c.expected == pytest.approx(0.5) for c in result.report.constructs)


def test_rejected_evidence_is_shown_separately_from_a_low_score():
    text = render_text(run("gaming").report)
    assert "ОТКЛОНЁННЫЕ ДОКАЗАТЕЛЬСТВА" in text
    assert "Это не низкий балл" in text
    assert "attribution" in text


def test_a_persona_report_says_it_is_a_persona():
    text = render_text(run("weak").report)
    assert "синтетическая персона" in text
    assert "проверка согласованности" in text


def test_a_gold_label_run_admits_the_model_never_ran():
    text = render_text(run("weak").report)
    assert "модель не вызывалась" in text


def test_disabling_the_adjudicator_is_disclosed():
    text = render_text(run("weak").report)
    assert "Второй судья отключён" in text


def test_the_report_carries_the_parameter_fingerprint():
    """A report without it cannot be told apart from one scored under different
    parameters."""
    result = run("strong")
    assert result.score.fingerprint in render_text(result.report)
