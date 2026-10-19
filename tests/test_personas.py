"""Designed contrasts across the synthetic personas.

WHAT THIS TESTS, PRECISELY: that the scoring model orders behaviours the way the
author designed them, when fed the author's own hand-written labels. It is a
consistency check on the arithmetic. It is **not** evidence that the instrument
measures critical thinking, and it cannot be: the personas, the indicators and
the probability tables all come from one person. See design/02-validation.md.

It also does not test the observer — the gold labels stand in for it here.

The assertions are contrasts rather than a total order. A total order over six
synthetic transcripts would be fitting noise with authority.
"""

import glob
from pathlib import Path

import pytest

from probe.config import replay_only
from probe.inventory.indicators import INDICATORS_BY_ID
from probe.models import GIGACHAT_OBSERVER
from probe.run import run_transcript
from probe.score.aggregate import Thresholds
from probe.score.session import score_session
from probe.store import repo
from probe.transcript import load

PERSONAS = Path("fixtures/personas")


def score(label: str):
    transcript = load(PERSONAS / f"{label}.json")
    with repo.connect() as connection:
        run = run_transcript(
            transcript, replay_only(), connection, observer="gold", adjudicator=False
        )
        return run.score.expected(), run


@pytest.fixture(scope="module")
def scores():
    return {
        Path(path).stem: score(Path(path).stem)
        for path in sorted(glob.glob(str(PERSONAS / "*.json")))
    }


def test_every_persona_declares_what_it_was_written_to_represent():
    """Without the stated intent the circularity is implied rather than visible."""
    for path in glob.glob(str(PERSONAS / "*.json")):
        assert load(path).design_intent_ru.strip(), path


def test_the_strong_participant_leads_on_every_construct(scores):
    strong, _ = scores["strong"]
    for label, (other, _) in scores.items():
        if label == "strong":
            continue
        for construct, value in strong.items():
            assert value > other[construct], f"{label} beat strong on {construct}"


def test_nobody_but_the_strong_participant_reads_as_capable(scores):
    for label, (values, _) in scores.items():
        if label == "strong":
            continue
        assert max(values.values()) < 0.5, f"{label} scored high somewhere"


def test_fluent_language_does_not_buy_a_score(scores):
    """The `polished` persona argues smoothly and checks nothing. If form moved
    the needle, this is where it would show."""
    polished, _ = scores["polished"]
    weak, _ = scores["weak"]
    assert polished["da"] < 0.25
    assert abs(polished["da"] - weak["da"]) < 0.15


def test_disagreeing_is_not_critical_thinking(scores):
    """`contrarian` objects to everything without naming a ground, then flips."""
    contrarian, _ = scores["contrarian"]
    assert contrarian["ss"] < 0.3
    assert contrarian["da"] < 0.3


def test_delegating_the_decision_shows_up_as_low_agency(scores):
    weak, _ = scores["weak"]
    strong, _ = scores["strong"]
    assert weak["ag"] < 0.25 < strong["ag"]


def test_borrowed_evidence_is_dropped_rather_than_scored(scores):
    """The `gaming` persona returns the AI's own sentences as analysis. Those
    observations look strong and must not survive admission."""
    _, run = scores["gaming"]
    rejected = [r for r in run.reviewed if not r.admitted]
    assert len(rejected) >= 3
    assert all(r.failed_check == "attribution" for r in rejected)


def test_gaming_is_not_rewarded_but_is_not_punished_either(scores):
    """Attribution withholds credit; it does not invent a penalty. A participant
    who parrots ends up where a participant who said nothing ends up."""
    gaming, _ = scores["gaming"]
    strong, _ = scores["strong"]
    passive, _ = scores["passive"]
    average = lambda values: sum(values.values()) / len(values)
    assert average(gaming) < average(strong) / 2
    assert abs(average(gaming) - average(passive)) < 0.15


def test_a_terse_but_correct_answer_still_counts(scores):
    """`gaming` answers the menu probe with the option's own name — right, and
    necessarily echoing the AI. The short-quote rule exists so this survives."""
    _, run = scores["gaming"]
    admitted = [r for r in run.reviewed if r.admitted]
    assert any(r.observation.indicator_id == "mf.priority" for r in admitted)


def test_all_eight_tasks_are_surfaced_for_every_persona():
    """Different participants must face the same instrument; otherwise the
    comparison between them is not a comparison."""
    first = None
    for path in sorted(glob.glob(str(PERSONAS / "*.json"))):
        transcript = load(path)
        assert len(transcript.surfaced) == 8, path
        if first is None:
            first = transcript.surfaced
        assert transcript.surfaced == first, path


def test_the_shared_assistant_script_is_identical_across_personas():
    scripts = set()
    for path in sorted(glob.glob(str(PERSONAS / "*.json"))):
        transcript = load(path)
        scripts.add(tuple(t.content for t in transcript.turns if t.role == "assistant"))
    assert len(scripts) == 1


@pytest.mark.parametrize("missed", [0.6, 0.7, 0.8, 0.9])
def test_the_headline_contrast_does_not_hinge_on_missed_confidence(missed):
    """MISSED_CONFIDENCE is an authored number with no data behind it. The claim
    the instrument actually makes — the strong participant leads everywhere and
    nobody else reads as capable — has to survive any plausible value of it, or
    the claim is about the parameter. Rescores stored rows; nothing is re-observed."""
    with repo.connect() as connection:
        expected = {}
        for path in sorted(glob.glob(str(PERSONAS / "*.json"))):
            session_id = run_transcript(
                load(path), replay_only(), connection, observer="gold", adjudicator=False
            ).session_id
            expected[Path(path).stem] = score_session(
                connection, session_id, thresholds=Thresholds(missed_confidence=missed)
            ).expected()

    strong = expected.pop("strong")
    for label, values in expected.items():
        assert max(values.values()) < 0.5, f"{label} reads as capable at {missed}"
        for construct, value in values.items():
            assert strong[construct] > value, f"{label} beat strong on {construct} at {missed}"


@pytest.fixture(scope="module")
def observed():
    """The real observer's and judge's recorded output (DeepSeek, 2026-10-11),
    replayed from fixtures/cassettes. Unlike every other test here, this one does
    not take the author's labels as given — it checks the headline survives a
    model doing the tagging. The judge is the same family as the observer; see
    probe/models.py."""
    results = {}
    for path in sorted(glob.glob(str(PERSONAS / "*.json"))):
        with repo.connect() as connection:
            run = run_transcript(
                load(path), replay_only(), connection, observer="model", adjudicator=True
            )
        results[Path(path).stem] = run
    return results


def test_the_live_observer_invents_no_indicator_ids(observed):
    for label, run in observed.items():
        assert run.telemetry.dropped_unknown_indicator == 0, label


def test_the_headline_contrast_survives_a_real_observer(observed):
    strong = observed["strong"].score.expected()
    for label, run in observed.items():
        if label == "strong":
            continue
        values = run.score.expected()
        assert max(values.values()) < 0.5, f"{label} reads as capable"
        for construct, value in values.items():
            assert strong[construct] > value, f"{label} beat strong on {construct}"


@pytest.fixture(scope="module")
def observed_gigachat():
    """The same personas tagged by GigaChat-2-Max (2026-10-19), judged by the same
    DeepSeek judge, replayed from fixtures/cassettes."""
    results = {}
    for path in sorted(glob.glob(str(PERSONAS / "*.json"))):
        with repo.connect() as connection:
            run = run_transcript(
                load(path), replay_only(), connection, observer_role=GIGACHAT_OBSERVER
            )
        results[Path(path).stem] = run
    return results


def test_the_headline_contrast_survives_a_second_model_family(observed_gigachat):
    test_the_headline_contrast_survives_a_real_observer(observed_gigachat)
    for label, run in observed_gigachat.items():
        assert run.telemetry.dropped_unknown_indicator == 0, label


def test_parroting_is_still_caught_when_a_model_does_the_tagging(observed):
    """The model reads `gaming` as parroting rather than as borrowed analysis.
    Either reading must end the same way: no credit for the AI's words, and
    parroting itself counted against agency."""
    reviewed = observed["gaming"].reviewed
    parroting = [r for r in reviewed if r.observation.indicator_id == "ag.parrot" and r.admitted]
    assert len(parroting) >= 3
    for r in reviewed:
        if r.admitted and INDICATORS_BY_ID[r.observation.indicator_id].polarity == "positive":
            assert "совпадает" not in " ".join(detail for _, _, detail in r.verdicts)
    assert observed["gaming"].score.expected()["ag"] < observed["passive"].score.expected()["ag"] + 0.05


def test_the_final_agree_question_is_where_t8_surfaces():
    """T8 was described as 'immediately' while the script asks only at the end;
    a model reading the description literally could never link the answer."""
    transcript = load(PERSONAS / "strong.json")
    assert transcript.tasks_surfaced_just_before(13) == ["T8"]
    assert "T8" not in transcript.tasks_surfaced_just_before(1)
