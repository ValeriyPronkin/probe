"""Conditional probability tables, generated from interpretable parameters.

Nothing here is a hand-typed 9-row table. Every table comes out of two or three
numbers a methodologist can argue about — "how hard is it to notice a swapped
comparison base" is answerable; "what is P(caught | kn=mid, mf=high)" is not.

That is also why calibration later is feasible: there are a few dozen parameters
to fit, not a few thousand cells.
"""

from __future__ import annotations

from collections.abc import Sequence
from itertools import product
from math import exp

from probe.inventory.constructs import (
    DISPOSITION_STRENGTH,
    LEVEL_VALUES,
    LEVELS,
)
from probe.inventory.indicators import INDICATOR_STATES, Indicator
from probe.scenario.tasks import TASK_STATES, Task

from probe.score.bayes import Distribution, ParentKey

# Gap between "shows this at all" and "shows it repeatedly or decisively".
STRONG_GAP = 0.25


def _sigmoid(x: float) -> float:
    # Branchless overflow guard: exp(709) is near the float64 ceiling.
    if x >= 0:
        return 1.0 / (1.0 + exp(-min(x, 700.0)))
    e = exp(max(x, -700.0))
    return e / (1.0 + e)


def graded_response(
    theta: float, discrimination: float, thresholds: Sequence[float]
) -> Distribution:
    """Samejima's graded response: P(X >= k) = sigmoid(a * (theta - b_k)).

    `thresholds` must be ascending; otherwise an intermediate state gets negative
    probability, which is a parameter bug worth failing on rather than clamping.
    """
    if any(b >= a for b, a in zip(thresholds, thresholds[1:], strict=False)):
        raise ValueError(f"thresholds must be strictly ascending: {tuple(thresholds)}")
    cumulative = [_sigmoid(discrimination * (theta - b)) for b in thresholds]
    dist = []
    previous = 1.0
    for c in cumulative:
        dist.append(previous - c)
        previous = c
    dist.append(previous)
    return tuple(dist)


def construct_given_disposition(strength: float = DISPOSITION_STRENGTH) -> dict[ParentKey, Distribution]:
    """P(construct level | disposition level).

    `strength` is the probability a construct sits at its disposition's level;
    the rest spreads to other levels by inverse-square distance, so "low
    disposition but high on one construct" stays possible and merely unlikely.
    """
    if not 0.0 < strength < 1.0:
        raise ValueError("strength must be in (0, 1)")
    table: dict[ParentKey, Distribution] = {}
    for i, parent_level in enumerate(LEVELS):
        weights = [0.0 if j == i else 1.0 / (abs(i - j) ** 2) for j in range(len(LEVELS))]
        total = sum(weights)
        row = tuple(
            strength if j == i else (1.0 - strength) * weights[j] / total
            for j in range(len(LEVELS))
        )
        table[(parent_level,)] = row
    return table


def indicator_cpt(indicator: Indicator) -> dict[ParentKey, Distribution]:
    """P(none / weak / strong | construct level) for one indicator.

    A negative indicator reads the construct scale inverted: `difficulty` keeps
    its meaning ("the ability at which this is as likely as not to appear"), so
    a high value on a negative indicator means even strong participants slip.
    """
    weak_threshold = indicator.difficulty
    strong_threshold = min(indicator.difficulty + STRONG_GAP, 0.999)
    if strong_threshold <= weak_threshold:  # pragma: no cover - difficulty is bounded above
        raise ValueError(f"{indicator.id}: difficulty leaves no room for a strong state")

    table: dict[ParentKey, Distribution] = {}
    for level in LEVELS:
        theta = LEVEL_VALUES[level]
        if indicator.polarity == "negative":
            theta = 1.0 - theta
        table[(level,)] = graded_response(
            theta, indicator.discrimination, (weak_threshold, strong_threshold)
        )
    if len(next(iter(table.values()))) != len(INDICATOR_STATES):
        raise ValueError(f"{indicator.id}: CPT width does not match INDICATOR_STATES")
    return table


def task_cpt(task: Task) -> dict[ParentKey, Distribution]:
    """P(missed / partial / caught | construct levels) for one task.

    Two parents combine **compensatorily**: theta is their mean, so strength in
    one construct partly covers weakness in the other. The alternative —
    conjunctive, theta = min — would measure the product of two constructs and
    report it as evidence for each. Per-task reasoning is in tasks.py `note_ru`.
    """
    table: dict[ParentKey, Distribution] = {}
    for combo in product(*([LEVELS] * len(task.constructs))):
        theta = sum(LEVEL_VALUES[level] for level in combo) / len(combo)
        table[tuple(combo)] = graded_response(
            theta, task.discrimination, (task.partial_threshold, task.caught_threshold)
        )
    if len(next(iter(table.values()))) != len(TASK_STATES):
        raise ValueError(f"{task.id}: CPT width does not match TASK_STATES")
    return table
