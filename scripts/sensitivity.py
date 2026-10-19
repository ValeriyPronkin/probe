#!/usr/bin/env python
"""How much the persona results owe to the rollup thresholds.

Each persona is run once, through gold labels, into a stored session; every
alternative value is then a rescore of those same stored rows. No observation is
redone, so any movement in the table is the parameter and nothing else.

    uv run scripts/sensitivity.py
    uv run scripts/sensitivity.py --param strong_confidence --values 0.6 0.7 0.8 0.9
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe.config import replay_only  # noqa: E402
from probe.inventory.constructs import CONSTRUCT_IDS  # noqa: E402
from probe.run import run_transcript  # noqa: E402
from probe.score.aggregate import DEFAULT_THRESHOLDS, Thresholds  # noqa: E402
from probe.score.session import score_session  # noqa: E402
from probe.store import repo  # noqa: E402
from probe.transcript import load  # noqa: E402

PERSONAS = Path(__file__).resolve().parent.parent / "fixtures" / "personas"


def sweep(param: str, values: list[float]) -> dict[float, dict[str, dict[str, float]]]:
    """{value: {persona: {construct: expected level}}}"""
    results: dict[float, dict[str, dict[str, float]]] = {v: {} for v in values}
    with repo.connect() as connection:
        sessions = {
            path.stem: run_transcript(
                load(path), replay_only(), connection, observer="gold", adjudicator=False
            ).session_id
            for path in sorted(PERSONAS.glob("*.json"))
        }
        for value in values:
            thresholds = dataclasses.replace(DEFAULT_THRESHOLDS, **{param: value})
            for label, session_id in sessions.items():
                results[value][label] = score_session(
                    connection, session_id, thresholds=thresholds
                ).expected()
    return results


def mean(values: dict[str, float]) -> float:
    return sum(values.values()) / len(values)


def render(param: str, results: dict[float, dict[str, dict[str, float]]]) -> str:
    labels = sorted(next(iter(results.values())))
    default = getattr(DEFAULT_THRESHOLDS, param)
    out = [f"{param}: среднее по конструктам (по умолчанию {default})", ""]
    out.append("значение  " + "".join(f"{label:>11}" for label in labels) + "   разрыв  полоса")
    for value, by_label in results.items():
        means = {label: mean(by_label[label]) for label in labels}
        others = [m for label, m in means.items() if label != "strong"]
        gap = means["strong"] - max(others)
        band = max(others) - min(others)
        marker = " *" if value == default else "  "
        out.append(
            f"{value:<6.2f}{marker}  "
            + "".join(f"{means[label]:>11.2f}" for label in labels)
            + f"   {gap:>6.2f}  {band:>6.2f}"
        )
    out.append("")
    out.append("разрыв — strong минус лучшая из остальных; полоса — размах остальных пяти.")
    out.append("")
    out.append("по конструктам, strong / лучшая из остальных:")
    out.append("значение  " + "".join(f"{c:>12}" for c in CONSTRUCT_IDS))
    for value, by_label in results.items():
        cells = []
        for construct in CONSTRUCT_IDS:
            best_other = max(
                by_label[label][construct] for label in labels if label != "strong"
            )
            cells.append(f"{by_label['strong'][construct]:.2f}/{best_other:.2f}")
        out.append(f"{value:<6.2f}    " + "".join(f"{cell:>12}" for cell in cells))
    return "\n".join(out)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--param",
        choices=[f.name for f in dataclasses.fields(Thresholds)],
        default="missed_confidence",
    )
    parser.add_argument(
        "--values", type=float, nargs="+", default=[0.5, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9]
    )
    args = parser.parse_args()
    print(render(args.param, sweep(args.param, args.values)))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
