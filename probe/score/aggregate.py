"""Turning admitted observations into network evidence.

Two independence violations sit between raw observations and a Bayesian network,
and both have to be closed or the instrument reports confidence it has not
earned (design/00 §5).

1. Within a turn. Three manifestations of one indicator in a single reply are
   one piece of evidence, not three.
2. Across the session. One indicator is ONE node for the whole session. Without
   this, a long conversation scores more confidently than a short one at
   identical behaviour — an artefact of transcript length, not of the person.

A third rule is enforced here rather than documented and hoped for: **every
admitted observation contributes to exactly one node.** An observation tied to a
surfaced task is consumed by that task's node; only free-dialogue observations
reach indicator nodes. Feeding both would move the construct twice for one
behaviour, since task and indicator nodes are both its children.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from probe.inventory.indicators import INDICATOR_STATES, INDICATORS_BY_ID
from probe.scenario.tasks import TASK_STATES, TASKS_BY_ID

from probe.score.bayes import Evidence, soft_evidence
from probe.score.network import task_node_name

# One high-confidence manifestation is enough to call an indicator `strong`.
STRONG_CONFIDENCE = 0.8

# How sure we are that a surfaced task with no positive detection was really
# missed. This is a property of detector recall, not of any model's self-report,
# so it is a constant we own rather than a number the grader hands us. Lower it
# and negative evidence weakens across the board.
MISSED_CONFIDENCE = 0.75

# Noisy-OR saturates at 1.0; certainty is never warranted here.
MAX_CONFIDENCE = 0.97


@dataclass(frozen=True)
class Thresholds:
    """The rollup's own parameters, as one value that scoring can be handed.

    They move posteriors as surely as any CPT, so they travel with the network
    into the parameter fingerprint, and a sensitivity sweep rescores stored
    sessions under alternatives instead of patching module globals.
    """

    strong_confidence: float = STRONG_CONFIDENCE
    missed_confidence: float = MISSED_CONFIDENCE
    max_confidence: float = MAX_CONFIDENCE

    def __post_init__(self) -> None:
        for name in ("strong_confidence", "missed_confidence", "max_confidence"):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"{name} out of range")


DEFAULT_THRESHOLDS = Thresholds()


@dataclass(frozen=True)
class Observation:
    """One admitted observation. Produced by probe/admit/, consumed only here."""

    indicator_id: str
    turn_index: int
    strength: str
    """'weak' or 'strong' — INDICATOR_STATES minus 'none', which is never observed."""
    confidence: float
    quote: str
    task_id: str | None = None

    def __post_init__(self) -> None:
        if self.indicator_id not in INDICATORS_BY_ID:
            raise ValueError(f"unknown indicator {self.indicator_id!r}")
        if self.strength not in ("weak", "strong"):
            raise ValueError(f"{self.indicator_id}: strength must be weak or strong")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"{self.indicator_id}: confidence out of range")
        if self.task_id is not None and self.task_id not in TASKS_BY_ID:
            raise ValueError(f"unknown task {self.task_id!r}")


def _noisy_or(confidences: Iterable[float], cap: float = MAX_CONFIDENCE) -> float:
    """Independent partial evidence combines; it does not average.

    Two separate turns at 0.6 should leave us more sure than one, but never
    certain — hence the cap.
    """
    miss = 1.0
    for c in confidences:
        miss *= 1.0 - c
    return min(1.0 - miss, cap)


def dedupe_within_turn(observations: Sequence[Observation]) -> tuple[Observation, ...]:
    """Keep the strongest observation per (turn, indicator, task); ties go to confidence.

    Polarity plays no part: within one indicator it is fixed, and a positive and
    a negative indicator firing in the same turn are different nodes and both
    survive — a person really can ask a sharp question and parrot in one breath.

    The task is part of the key. One reply that exposes two different seeded
    flaws through the same indicator is two behaviours feeding two task nodes;
    collapsing them would score one of the two as missed.
    """
    best: dict[tuple[int, str, str | None], Observation] = {}
    for observation in observations:
        key = (observation.turn_index, observation.indicator_id, observation.task_id)
        current = best.get(key)
        if current is None:
            best[key] = observation
            continue
        rank = ("weak", "strong")
        if (rank.index(observation.strength), observation.confidence) > (
            rank.index(current.strength),
            current.confidence,
        ):
            best[key] = observation
    return tuple(
        sorted(best.values(), key=lambda o: (o.turn_index, o.indicator_id, o.task_id or ""))
    )


@dataclass(frozen=True)
class NodeEvidence:
    """What one network node ends up being told, and by which observations."""

    node: str
    state: str
    confidence: float
    sources: tuple[Observation, ...]
    kind: str
    """'indicator' or 'task' — report sections and admissibility audits key on it."""


def rollup_indicators(
    observations: Sequence[Observation], thresholds: Thresholds = DEFAULT_THRESHOLDS
) -> tuple[NodeEvidence, ...]:
    """One node per indicator for the whole session.

    `strong` when the indicator fired in two or more turns, or once with high
    confidence. `none` is never produced: a detector that stayed quiet is not a
    behaviour that was absent, so the node is simply left unobserved.
    """
    by_indicator: dict[str, list[Observation]] = {}
    for observation in dedupe_within_turn(observations):
        by_indicator.setdefault(observation.indicator_id, []).append(observation)

    evidence: list[NodeEvidence] = []
    for indicator_id, group in sorted(by_indicator.items()):
        turns = {o.turn_index for o in group}
        decisive = any(
            o.strength == "strong" and o.confidence >= thresholds.strong_confidence
            for o in group
        )
        state = "strong" if len(turns) >= 2 or decisive else "weak"
        evidence.append(
            NodeEvidence(
                node=indicator_id,
                state=state,
                confidence=_noisy_or(
                    (o.confidence for o in group), thresholds.max_confidence
                ),
                sources=tuple(group),
                kind="indicator",
            )
        )
    return tuple(evidence)


def rollup_tasks(
    observations: Sequence[Observation],
    surfaced: Iterable[str],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[NodeEvidence, ...]:
    """Score every task the scenario actually put in front of the participant.

    A task that never surfaced produces nothing — no node, no evidence, and a
    line in the report's "не наблюдалось" section. That is the whole reason the
    task model exists.
    """
    surfaced_ids = set(surfaced)
    # An observation referencing a task is itself proof the task was surfaced;
    # trusting the tracker over the evidence would silently drop real signal.
    surfaced_ids.update(o.task_id for o in observations if o.task_id is not None)

    unknown = surfaced_ids - set(TASKS_BY_ID)
    if unknown:
        raise ValueError(f"unknown task ids: {sorted(unknown)}")

    deduped = dedupe_within_turn(observations)
    evidence: list[NodeEvidence] = []

    for task_id in sorted(surfaced_ids):
        task = TASKS_BY_ID[task_id]
        group = [o for o in deduped if o.task_id == task_id]
        positives = [o for o in group if INDICATORS_BY_ID[o.indicator_id].polarity == "positive"]
        negatives = [o for o in group if INDICATORS_BY_ID[o.indicator_id].polarity == "negative"]

        if positives and not negatives:
            decisive = any(o.strength == "strong" for o in positives) or len(positives) >= 2
            state = "caught" if decisive else "partial"
            confidence = _noisy_or((o.confidence for o in positives), thresholds.max_confidence)
        elif positives and negatives:
            # Mixed: the participant engaged the flaw but also showed the failure
            # it was built to detect. Partial is the honest reading.
            state = "partial"
            confidence = _noisy_or((o.confidence for o in group), thresholds.max_confidence)
        elif negatives:
            state = "missed"
            confidence = _noisy_or(
                [thresholds.missed_confidence, *(o.confidence for o in negatives)],
                thresholds.max_confidence,
            )
        else:
            state = "missed"
            confidence = thresholds.missed_confidence

        evidence.append(
            NodeEvidence(
                node=task_node_name(task.id),
                state=state,
                confidence=confidence,
                sources=tuple(group),
                kind="task",
            )
        )
    return tuple(evidence)


def build_evidence(
    observations: Sequence[Observation],
    surfaced: Iterable[str],
    thresholds: Thresholds = DEFAULT_THRESHOLDS,
) -> tuple[NodeEvidence, ...]:
    """The full session rollup: task-bound observations to task nodes, the rest
    to indicator nodes, each observation counted exactly once."""
    surfaced_ids = set(surfaced)
    surfaced_ids.update(o.task_id for o in observations if o.task_id is not None)

    task_bound = [o for o in observations if o.task_id in surfaced_ids and o.task_id is not None]
    free = [o for o in observations if o not in task_bound]

    return tuple(
        sorted(
            (
                *rollup_tasks(task_bound, surfaced_ids, thresholds),
                *rollup_indicators(free, thresholds),
            ),
            key=lambda e: (e.kind, e.node),
        )
    )


def to_evidence_map(node_evidence: Sequence[NodeEvidence]) -> Evidence:
    """NodeEvidence -> the likelihood vectors bayes.posterior consumes."""
    evidence: dict[str, tuple[float, ...]] = {}
    for item in node_evidence:
        states = TASK_STATES if item.kind == "task" else INDICATOR_STATES
        evidence[item.node] = soft_evidence(states, item.state, item.confidence)
    return evidence
