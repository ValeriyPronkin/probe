"""Stage 2: one model call per participant turn, emitting tagged observations.

Validation is post-hoc, by design (see observe/schema.py). An invented id is
dropped and counted; the drop rate and the empty-output rate are what tell you a
prompt has started to rot, and neither is visible if the schema makes a bad id
impossible to say.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from probe.config import Config
from probe.inventory.render import render_inventory, valid_indicator_ids
from probe.llm import Request, complete_json
from probe.models import OBSERVER
from probe.observe.schema import ObserverOutput, RawObservation, observer_json_schema
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
    transcript: Transcript, turn_idx: int, config: Config, telemetry: Telemetry | None = None
) -> list[RawObservation]:
    telemetry = telemetry if telemetry is not None else Telemetry()
    system, user = build_prompt(transcript, turn_idx)
    response = complete_json(
        Request(
            role=OBSERVER,
            system=system,
            user=user,
            schema_name="observations",
            schema=observer_json_schema(),
        ),
        config,
    )
    output = ObserverOutput.model_validate(response)
    telemetry.turns_observed += 1
    telemetry.emitted += len(output.observations)
    if not output.observations:
        telemetry.turns_empty += 1

    valid = valid_indicator_ids()
    kept: list[RawObservation] = []
    for observation in output.observations:
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
