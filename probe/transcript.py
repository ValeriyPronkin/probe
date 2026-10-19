"""A session as data: turns plus which tasks the scenario actually surfaced.

The `surfaced` map is not bookkeeping. It is the difference between "did not
show the behaviour" and "was never put in a situation where it could show", and
without it every quiet participant looks weak.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from probe.scenario.tasks import TASKS_BY_ID

Role = Literal["participant", "assistant", "system"]


@dataclass(frozen=True)
class Turn:
    idx: int
    role: Role
    content: str


@dataclass(frozen=True)
class Transcript:
    label: str
    source: str
    """'live' | 'persona' | 'transcript' — what kind of data this is, and the
    report says so, because a synthetic persona is not evidence about a person."""
    scenario_id: str
    turns: tuple[Turn, ...]
    surfaced: dict[str, int] = field(default_factory=dict)
    design_intent_ru: str = ""
    """Personas only: what this transcript was written to represent. Keeping it
    in the file is what makes the circularity visible instead of implied."""
    gold_labels: tuple[dict, ...] = ()
    """Personas only: hand-written indicator labels per turn. Used as the stub
    observer in scoring tests and as the reference set for the method check."""

    def participant_turns(self) -> tuple[Turn, ...]:
        return tuple(t for t in self.turns if t.role == "participant")

    def assistant_text_before(self, idx: int) -> list[str]:
        return [t.content for t in self.turns if t.role == "assistant" and t.idx < idx]

    def tasks_surfaced_just_before(self, idx: int) -> list[str]:
        """Tasks the scenario put in the assistant reply this turn answers. The
        scenario knows this for certain; making the observer guess it from a
        task list throws away information the instrument already has."""
        prior = [t.idx for t in self.turns if t.role == "assistant" and t.idx < idx]
        if not prior:
            return []
        return sorted(task for task, at in self.surfaced.items() if at == prior[-1])

    def turn(self, idx: int) -> Turn:
        for t in self.turns:
            if t.idx == idx:
                return t
        raise KeyError(f"no turn {idx}")


def dump(transcript: Transcript, path: str | Path, audit: dict | None = None) -> None:
    """The persona file format, so a live session loads, replays and rescores
    exactly like a persona. `audit` is kept beside the transcript and ignored by
    `load`: what produced it, not what it says."""
    data = {
        "label": transcript.label,
        "source": transcript.source,
        "scenario_id": transcript.scenario_id,
        "surfaced": transcript.surfaced,
        "turns": [{"role": t.role, "content": t.content} for t in transcript.turns],
    }
    if audit:
        data["audit"] = audit
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load(path: str | Path) -> Transcript:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    turns = tuple(
        Turn(idx=i, role=t["role"], content=t["content"]) for i, t in enumerate(data["turns"])
    )
    surfaced = {str(k): int(v) for k, v in data.get("surfaced", {}).items()}
    unknown = set(surfaced) - set(TASKS_BY_ID)
    if unknown:
        raise ValueError(f"{path}: unknown task ids {sorted(unknown)}")
    for task_id, turn_idx in surfaced.items():
        if not any(t.idx == turn_idx for t in turns):
            raise ValueError(f"{path}: task {task_id} surfaced at missing turn {turn_idx}")
    return Transcript(
        label=data["label"],
        source=data.get("source", "persona"),
        scenario_id=data.get("scenario_id", "churn-v1"),
        turns=turns,
        surfaced=surfaced,
        design_intent_ru=data.get("design_intent_ru", ""),
        gold_labels=tuple(data.get("gold_labels", ())),
    )
