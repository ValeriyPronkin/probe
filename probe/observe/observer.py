"""Stage 2: one model call per participant turn, emitting tagged observations.

Validation is post-hoc, by design (see observe/schema.py). An invented id is
dropped and counted; the drop rate and the empty-output rate are what tell you a
prompt has started to rot, and neither is visible if the schema makes a bad id
impossible to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from pydantic import ValidationError

from probe.config import Config
from probe.inventory.render import render_inventory, valid_indicator_ids
from probe.llm import Request, complete_json
from probe.models import OBSERVER, ModelRole
from probe.observe.schema import RawObservation, observer_json_schema
from probe.scenario.tasks import TASKS_BY_ID
from probe.transcript import Transcript

SYSTEM_RU = """\
Ты — наблюдатель в измерительной системе. Ты не разговариваешь с участником и не оцениваешь его.
Твоя единственная работа — отметить, какие из перечисленных ниже поведений участник проявил
в одном своём сообщении, и привести дословную цитату на каждое.

Ты видишь предыдущую реплику ИИ только для контекста. Разметке подлежит исключительно сообщение
участника.

{inventory}
"""


@dataclass
class Telemetry:
    """Instrument health, not model output. Rising numbers here mean the prompt
    or the inventory needs work — they are not noise to be filtered away."""

    turns_observed: int = 0
    turns_empty: int = 0
    emitted: int = 0
    dropped_unknown_indicator: int = 0
    dropped_unknown_task: int = 0
    dropped_malformed: int = 0
    """Observations that broke the output contract (e.g. strength "medium"). Same
    policy as an invented id: dropped and counted, never coerced into a valid one."""
    envelope_broken: int = 0
    """Replies that were a bare list instead of {"observations": [...]}. The list
    is unambiguous, so it is read as the observations; the break is counted."""
    unknown_ids: list[str] = field(default_factory=list)

    @property
    def drop_rate(self) -> float:
        return (self.dropped_unknown_indicator / self.emitted) if self.emitted else 0.0


def build_prompt(transcript: Transcript, turn_idx: int) -> tuple[str, str]:
    turn = transcript.turn(turn_idx)
    prior = transcript.assistant_text_before(turn_idx)
    last_ai = prior[-1] if prior else "(диалог ещё не начался)"
    just_surfaced = transcript.tasks_surfaced_just_before(turn_idx)
    tasks_line = (
        "ЗАДАНИЯ, ПРЕДЪЯВЛЕННЫЕ В ЭТОЙ РЕПЛИКЕ ИИ: "
        + ", ".join(f"{t} — {TASKS_BY_ID[t].summary_ru}" for t in just_surfaced)
        + "\nЕсли наблюдение — реакция на одно из них, укажи его task_id. Реакция может "
        "относиться и к заданию, предъявленному раньше."
        if just_surfaced
        else "В этой реплике ИИ новых заданий не предъявлено; task_id ставь, только если "
        "участник реагирует на задание, предъявленное раньше."
    )
    user = (
        f"ПРЕДЫДУЩАЯ РЕПЛИКА ИИ (только контекст, не размечать):\n{last_ai}\n\n"
        f"{tasks_line}\n\n"
        f"СООБЩЕНИЕ УЧАСТНИКА (размечать только это):\n{turn.content}"
    )
    return SYSTEM_RU.format(inventory=render_inventory()), user


def observe_turn(
    transcript: Transcript,
    turn_idx: int,
    config: Config,
    telemetry: Telemetry | None = None,
    role: ModelRole = OBSERVER,
) -> list[RawObservation]:
    telemetry = telemetry if telemetry is not None else Telemetry()
    system, user = build_prompt(transcript, turn_idx)
    response = complete_json(
        Request(
            role=role,
            system=system,
            user=user,
            schema_name="observations",
            schema=observer_json_schema(),
        ),
        config,
    )
    # The envelope must hold; a reply that is not {"observations": [...]} is a
    # provider failure, not a labelling error. Each observation is then checked
    # on its own, so one malformed item costs that item rather than the turn.
    if isinstance(response, list):
        telemetry.envelope_broken += 1
        items = response
    else:
        items = response.get("observations") if isinstance(response, dict) else None
    if not isinstance(items, list):
        raise ValueError(f"observer returned observations={items!r}")
    telemetry.turns_observed += 1
    telemetry.emitted += len(items)
    if not items:
        telemetry.turns_empty += 1

    parsed: list[RawObservation] = []
    for item in items:
        try:
            parsed.append(RawObservation.model_validate(item))
        except ValidationError:
            telemetry.dropped_malformed += 1

    valid = valid_indicator_ids()
    kept: list[RawObservation] = []
    for observation in parsed:
        if observation.indicator_id not in valid:
            telemetry.dropped_unknown_indicator += 1
            telemetry.unknown_ids.append(observation.indicator_id)
            continue
        if observation.task_id and observation.task_id not in TASKS_BY_ID:
            # The observation itself may still be sound; only the task link is
            # wrong, so drop the link rather than the evidence.
            telemetry.dropped_unknown_task += 1
            observation = observation.model_copy(update={"task_id": ""})
        kept.append(observation)
    return kept
