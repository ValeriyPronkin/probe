#!/usr/bin/env python
"""Two observers on the same six personas, each against the author's labels.

The judge, prompts, inventory and scoring are identical; only the model behind
the observer changes. DeepSeek replays from its 2026-10-11 cassettes. GigaChat
replays when its cassettes are there and otherwise calls the provider and
records, one persona at a time, so an interrupted run resumes where it stopped.

    uv run scripts/compare_observers.py              # replay only, never calls out
    uv run scripts/compare_observers.py --record     # record what is missing

Agreement is Cohen's kappa over units of (participant turn, indicator): did this
rater tag this indicator in this turn. It is taken on what the observer emitted,
before the admission checks, because it measures the labelling itself; the
scores in the table are after admission. Model against model is the panel's
agreement: where two families both disagree with the gold labels, the indicator's
definition is the first suspect.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe import config as config_module  # noqa: E402
from probe.config import CASSETTE_DIR, replay_only  # noqa: E402
from probe.inventory.render import valid_indicator_ids  # noqa: E402
from probe.llm import CassetteMissing  # noqa: E402
from probe.models import OBSERVERS  # noqa: E402
from probe.run import Run, run_transcript  # noqa: E402
from probe.store import repo  # noqa: E402
from probe.transcript import load  # noqa: E402

PERSONAS = Path(__file__).resolve().parent.parent / "fixtures" / "personas"
ORDER = ["strong", "weak", "polished", "passive", "contrarian", "gaming"]


def mean_score(run: Run) -> float:
    values = run.score.expected().values()
    return sum(values) / len(values)


def task_states(run: Run) -> dict[str, str]:
    return {e.node: e.state for e in run.score.evidence if e.kind == "task"}


def run_one(path: Path, config, **kwargs) -> Run:
    with repo.connect() as connection:
        return run_transcript(load(path), config, connection, **kwargs)


def observer_run(path: Path, name: str, record: bool) -> Run:
    role = OBSERVERS[name]
    try:
        return run_one(path, replay_only(), observer_role=role)
    except CassetteMissing:
        if not record:
            raise
    live = dataclasses.replace(
        config_module.load(), mode="live", record=True, reuse_recorded=True
    )
    started = time.time()
    run = run_one(path, live, observer_role=role)
    print(f"  {path.stem}: recorded in {time.time() - started:.0f} s", file=sys.stderr)
    return run


Labels = set[tuple[str, int, str]]
"""(persona, turn, indicator) — the units a rater said yes to."""


def gold_labels(label: str, path: Path) -> Labels:
    return {(label, int(g["turn"]), g["indicator_id"]) for g in load(path).gold_labels}


def model_labels(label: str, run: Run) -> Labels:
    return {(label, r.observation.turn_index, r.observation.indicator_id) for r in run.reviewed}


def kappa(a: Labels, b: Labels, units: list[tuple[str, int, str]]) -> float | None:
    n = len(units)
    both = sum(u in a and u in b for u in units)
    neither = sum(u not in a and u not in b for u in units)
    observed = (both + neither) / n
    pa, pb = sum(u in a for u in units) / n, sum(u in b for u in units) / n
    expected = pa * pb + (1 - pa) * (1 - pb)
    return None if expected == 1 else (observed - expected) / (1 - expected)


def served_versions(provider: str) -> list[str]:
    served = set()
    for path in CASSETTE_DIR.glob("observer-*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("provider") == provider:
            served.add(data.get("served_model", "?"))
    return sorted(served)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record", action="store_true", help="call GigaChat for what is missing")
    args = parser.parse_args()

    rows = []
    raters: dict[str, Labels] = {"gold": set(), **{name: set() for name in OBSERVERS}}
    turns: list[tuple[str, int]] = []
    disagreements = {name: 0 for name in OBSERVERS}
    tasks_total = 0
    for label in ORDER:
        path = PERSONAS / f"{label}.json"
        gold = run_one(path, replay_only(), observer="gold", adjudicator=False)
        runs = {name: observer_run(path, name, args.record) for name in OBSERVERS}
        raters["gold"] |= gold_labels(label, path)
        for name, run in runs.items():
            raters[name] |= model_labels(label, run)
        turns += [(label, t.idx) for t in load(path).participant_turns()]
        gold_tasks = task_states(gold)
        tasks_total += len(gold_tasks)
        row = {"label": label, "gold": mean_score(gold)}
        for name, run in runs.items():
            states = task_states(run)
            disagreements[name] += sum(states.get(t) != s for t, s in gold_tasks.items())
            t = run.telemetry
            row[name] = (
                mean_score(run),
                t.emitted,
                f"{t.turns_empty} / {t.turns_observed}",
                sum(not r.admitted for r in run.reviewed),
                t.dropped_unknown_indicator + t.dropped_unknown_task,
                f"{t.dropped_malformed} / {t.envelope_broken}",
            )
        rows.append(row)

    names = list(OBSERVERS)
    print("| Персона | Эталон, ср. | " + " | ".join(
        f"{n}: ср. | набл. | пустых | откл. | выдум. | схема: набл. / ответ" for n in names) + " |")
    print("| --- | --- | " + " | ".join("--- | --- | --- | --- | --- | ---" for _ in names) + " |")
    for row in rows:
        cells = [f"`{row['label']}`", f"{row['gold']:.2f}"]
        for n in names:
            score, emitted, empty, rejected, invented, malformed = row[n]
            cells += [
                f"{score:.2f}", str(emitted), empty, str(rejected), str(invented), malformed
            ]
        print("| " + " | ".join(cells) + " |")
    print()
    for n in names:
        print(f"{n}: расхождений с эталоном по заданиям — {disagreements[n]} из {tasks_total}")
    indicators = sorted(valid_indicator_ids())
    units = [(label, turn, ind) for label, turn in turns for ind in indicators]
    pairs = [("gold", n) for n in names] + [(names[0], names[1])]
    print()
    print(f"Каппа Коэна, все индикаторы ({len(turns)} ходов × {len(indicators)} = {len(units)} единиц):")
    for a, b in pairs:
        print(f"  {a} — {b}: {kappa(raters[a], raters[b], units):.2f}")
    print()
    print("| Индикатор | " + " | ".join(f"{a}—{b}" for a, b in pairs) + " | "
          + " | ".join(f"да: {r}" for r in raters) + " |")
    print("| --- | " + " | ".join("---" for _ in pairs) + " | " + " | ".join("---" for _ in raters) + " |")
    for ind in indicators:
        sub_units = [u for u in units if u[2] == ind]
        yes = [sum(u in raters[r] for u in sub_units) for r in raters]
        if not any(yes):
            continue
        ks = [kappa(raters[a], raters[b], sub_units) for a, b in pairs]
        cells = ["—" if k is None else f"{k:.2f}" for k in ks]
        print(f"| `{ind}` | " + " | ".join(cells) + " | " + " | ".join(map(str, yes)) + " |")
    print()
    print(f"gigachat, версии модели в ответах: {', '.join(served_versions('gigachat')) or '—'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
