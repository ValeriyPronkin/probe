"""The four admissibility checks. Three are mechanical and tested exactly."""

import pytest

from probe.admit.attribution import MIN_TOKENS, check_attribution, coverage
from probe.admit.grounding import check_grounding, normalise
from probe.admit.meta_turns import check_meta_turn, classify_turn
from probe.admit.pipeline import review_turn
from probe.config import replay_only
from probe.observe.schema import RawObservation
from probe.transcript import Transcript, Turn

AI = "Отток вырос на 1240 человек за неделю после редизайна приложения, это заметное изменение."


# ── grounding ────────────────────────────────────────────────────────────────

def test_quote_present_verbatim_passes():
    assert check_grounding("А из скольки?", "Подожди. А из скольки? И по активным?")[0]


def test_paraphrased_quote_is_rejected():
    passed, detail = check_grounding("а из какого числа", "А из скольки?")
    assert not passed
    assert "нет в сообщении" in detail


def test_case_and_whitespace_are_normalised_but_punctuation_is_not():
    assert check_grounding("а   ИЗ скольки?", "А из скольки?")[0]
    assert not check_grounding("А из скольки", "А, из, скольки")[0]


def test_empty_quote_is_rejected():
    assert not check_grounding("   ", "что угодно")[0]


def test_normalise_is_idempotent():
    assert normalise(normalise("  А  из\nскольки ")) == normalise("  А  из\nскольки ")


# ── attribution ──────────────────────────────────────────────────────────────

def test_verbatim_reuse_of_the_ai_is_rejected():
    passed, detail = check_attribution("Отток вырос на 1240 человек за неделю после редизайна", [AI])
    assert not passed
    assert "%" in detail


def test_the_participants_own_sentence_passes():
    assert check_attribution("А из скольки это 1240? Нужен знаменатель", [AI])[0]


def test_a_short_quote_is_not_assessable_and_must_not_be_penalised():
    """Answering a menu probe with the option's own name is correct behaviour.
    Scoring it as borrowed penalises every terse participant."""
    passed, detail = check_attribution("Базу сравнения.", ["1) цвет кнопки 2) базу сравнения"])
    assert passed
    assert "неприменимо" in detail


def test_coverage_is_zero_below_the_minimum_token_count():
    assert coverage("базу сравнения", "возьмём базу сравнения потом") == 0.0
    assert len("базу сравнения".split()) < MIN_TOKENS


def test_partial_overlap_below_the_threshold_survives():
    quote = "Ты говоришь про редизайн приложения, но я думаю про цену и про отток в когортах"
    assert check_attribution(quote, [AI])[0]


def test_no_prior_ai_text_means_nothing_to_borrow():
    assert check_attribution("любая фраза участника здесь", [])[0]


# ── meta turns ───────────────────────────────────────────────────────────────

def test_questions_about_the_assessment_score_nothing():
    assert not check_meta_turn("а как ты меня оцениваешь?")[0]
    assert not check_meta_turn("это у нас тут тест какой-то?")[0]


def test_reports_of_a_system_failure_score_nothing():
    assert not check_meta_turn("страница зависла, перезагрузи чат")[0]


def test_the_word_error_about_the_reasoning_stays_substantive():
    """The single most valuable thing a participant can say must not be filtered."""
    assert classify_turn("Здесь ошибка в рассуждении: база сравнения другая") == "substantive"
    assert classify_turn("Это неверный вывод, данные не те") == "substantive"
    assert check_meta_turn("Откуда цифра 23%? Это ошибка.")[0]


# ── pipeline ─────────────────────────────────────────────────────────────────

def transcript() -> Transcript:
    return Transcript(
        label="t",
        source="persona",
        scenario_id="churn-v1",
        turns=(
            Turn(0, "assistant", AI),
            Turn(1, "participant", "А из скольки это 1240? Нужен знаменатель."),
            Turn(2, "participant", "Отток вырос на 1240 человек за неделю после редизайна."),
            Turn(3, "participant", "а как ты меня оцениваешь?"),
        ),
        surfaced={"T4": 0},
    )


def raw(indicator, quote):
    return [RawObservation(indicator_id=indicator, quote=quote, strength="strong", confidence=0.9)]


def test_every_check_records_a_verdict_even_when_it_passes():
    reviewed = review_turn(
        transcript(), 1, raw("kn.gap", "А из скольки это 1240?"), replay_only(), adjudicator=False
    )
    names = [name for name, _, _ in reviewed[0].verdicts]
    assert names == ["meta_turn", "grounding", "attribution"]
    assert reviewed[0].admitted


def test_borrowed_evidence_is_rejected_and_names_the_check():
    reviewed = review_turn(
        transcript(),
        2,
        raw("da.mechanism", "Отток вырос на 1240 человек за неделю после редизайна."),
        replay_only(),
        adjudicator=False,
    )
    assert not reviewed[0].admitted
    assert reviewed[0].failed_check == "attribution"


def test_parroting_is_admitted_as_parroting():
    """Found on the live run. For ag.parrot the overlap with the AI's words is the
    behaviour itself; running attribution on it made the indicator inadmissible."""
    reviewed = review_turn(
        transcript(),
        2,
        raw("ag.parrot", "Отток вырос на 1240 человек за неделю после редизайна."),
        replay_only(),
        adjudicator=False,
    )
    assert reviewed[0].admitted
    attribution = [v for v in reviewed[0].verdicts if v[0] == "attribution"][0]
    assert attribution[1] and "неприменимо" in attribution[2]


def test_a_meta_turn_admits_nothing_however_good_the_quote_looks():
    reviewed = review_turn(
        transcript(), 3, raw("da.evidence", "а как ты меня оцениваешь?"), replay_only(),
        adjudicator=False,
    )
    assert not reviewed[0].admitted
    assert reviewed[0].failed_check == "meta_turn"


def test_attribution_is_not_claimed_to_pass_when_grounding_failed():
    """A cheerful 'passed' on a check that never ran would read as 'we checked'."""
    reviewed = review_turn(
        transcript(), 1, raw("kn.gap", "текста такого нет"), replay_only(), adjudicator=False
    )
    verdicts = dict((name, passed) for name, passed, _ in reviewed[0].verdicts)
    assert verdicts["grounding"] is False
    assert "attribution" not in verdicts
