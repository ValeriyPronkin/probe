#!/usr/bin/env python
"""Score a transcript and print the report.

    uv run scripts/replay.py fixtures/personas/strong.json
    uv run scripts/replay.py fixtures/personas/gaming.json --observer gold
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from probe import config as config_module  # noqa: E402
from probe.report.build import render_text  # noqa: E402
from probe.run import run_transcript  # noqa: E402
from probe.score.session import rebuild_matches_stored  # noqa: E402
from probe.store import repo  # noqa: E402
from probe.transcript import load  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("transcript", type=Path)
    parser.add_argument(
        "--observer",
        choices=("model", "gold"),
        default="gold",
        help="gold: use the fixture's hand-written labels (tests the arithmetic only)",
    )
    parser.add_argument("--no-adjudicator", action="store_true")
    parser.add_argument("--db", default=":memory:")
    parser.add_argument("--verify", action="store_true", help="check the rebuild invariant")
    args = parser.parse_args()

    transcript = load(args.transcript)
    cfg = config_module.load()
    with repo.connect(args.db) as connection:
        run = run_transcript(
            transcript,
            cfg,
            connection,
            observer=args.observer,
            adjudicator=not args.no_adjudicator,
        )
        print(render_text(run.report))
        if args.verify:
            ok = rebuild_matches_stored(connection, run.session_id)
            print()
            print(f"инвариант пересборки: {'держится' if ok else 'НАРУШЕН'}")
            return 0 if ok else 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
