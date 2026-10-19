"""SQLite access, written as explicit SQL.

An ORM would hide the one property that matters here: what exactly is stored
alongside each piece of evidence. For a measurement instrument the schema is
part of the argument, so it stays readable.
"""

from __future__ import annotations

import sqlite3
import uuid
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from probe.score.aggregate import Observation

SCHEMA_PATH = Path(__file__).with_name("schema.sql")


def _now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def connect(path: str | Path = ":memory:") -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.row_factory = sqlite3.Row
    connection.executescript(SCHEMA_PATH.read_text(encoding="utf-8"))
    return connection


@dataclass(frozen=True)
class SessionMeta:
    id: str
    participant_label: str
    source: str
    scenario_id: str
    observer_model: str
    judge_model: str
    prompt_version: str


def create_session(connection: sqlite3.Connection, meta: SessionMeta) -> str:
    connection.execute(
        """INSERT INTO sessions
             (id, created_at, participant_label, source, scenario_id,
              observer_model, judge_model, prompt_version)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            meta.id,
            _now(),
            meta.participant_label,
            meta.source,
            meta.scenario_id,
            meta.observer_model,
            meta.judge_model,
            meta.prompt_version,
        ),
    )
    return meta.id


def append_turn(
    connection: sqlite3.Connection, session_id: str, idx: int, role: str, content: str
) -> None:
    connection.execute(
        "INSERT INTO turns (session_id, idx, role, content) VALUES (?, ?, ?, ?)",
        (session_id, idx, role, content),
    )


def record_surfaced(
    connection: sqlite3.Connection, session_id: str, task_id: str, turn_idx: int
) -> None:
    connection.execute(
        """INSERT OR IGNORE INTO surfaced_tasks (session_id, task_id, turn_idx)
           VALUES (?, ?, ?)""",
        (session_id, task_id, turn_idx),
    )


def surfaced_tasks(connection: sqlite3.Connection, session_id: str) -> tuple[str, ...]:
    rows = connection.execute(
        "SELECT task_id FROM surfaced_tasks WHERE session_id = ? ORDER BY task_id",
        (session_id,),
    ).fetchall()
    return tuple(row["task_id"] for row in rows)


def record_observation(
    connection: sqlite3.Connection,
    session_id: str,
    observation: Observation,
    *,
    observer_model: str,
    prompt_version: str,
    verdicts: Iterable[tuple[str, bool, str]] = (),
) -> str:
    """Store one observation and the verdict of every admissibility check run on it.

    Inadmissible observations are stored too — see schema.sql.
    """
    observation_id = str(uuid.uuid4())
    connection.execute(
        """INSERT INTO observations
             (id, session_id, turn_idx, indicator_id, strength, confidence, quote,
              task_id, observer_model, prompt_version, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            observation_id,
            session_id,
            observation.turn_index,
            observation.indicator_id,
            observation.strength,
            observation.confidence,
            observation.quote,
            observation.task_id,
            observer_model,
            prompt_version,
            _now(),
        ),
    )
    connection.executemany(
        """INSERT INTO admissibility (observation_id, check_name, passed, detail)
           VALUES (?, ?, ?, ?)""",
        [(observation_id, name, int(passed), detail) for name, passed, detail in verdicts],
    )
    return observation_id


def admitted_observations(
    connection: sqlite3.Connection, session_id: str
) -> tuple[Observation, ...]:
    """Observations with no failing admissibility check, in turn order."""
    rows = connection.execute(
        """SELECT o.turn_idx, o.indicator_id, o.strength, o.confidence, o.quote, o.task_id
             FROM observations o
            WHERE o.session_id = ?
              AND NOT EXISTS (
                    SELECT 1 FROM admissibility a
                     WHERE a.observation_id = o.id AND a.passed = 0
              )
            ORDER BY o.turn_idx, o.indicator_id""",
        (session_id,),
    ).fetchall()
    return tuple(
        Observation(
            indicator_id=row["indicator_id"],
            turn_index=row["turn_idx"],
            strength=row["strength"],
            confidence=row["confidence"],
            quote=row["quote"],
            task_id=row["task_id"],
        )
        for row in rows
    )


def rejected_observations(connection: sqlite3.Connection, session_id: str) -> tuple[sqlite3.Row, ...]:
    """Observations excluded by at least one check, with the reason. The report
    shows these: "evidence was dropped" is a different statement from "scored low"."""
    return tuple(
        connection.execute(
            """SELECT o.id, o.turn_idx, o.indicator_id, o.quote, a.check_name, a.detail
                 FROM observations o
                 JOIN admissibility a ON a.observation_id = o.id AND a.passed = 0
                WHERE o.session_id = ?
                ORDER BY o.turn_idx, o.indicator_id""",
            (session_id,),
        ).fetchall()
    )


def save_scores(
    connection: sqlite3.Connection,
    session_id: str,
    posteriors: dict[str, dict[str, float]],
    fingerprint: str,
) -> None:
    computed_at = _now()
    connection.executemany(
        """INSERT OR REPLACE INTO scores
             (session_id, construct, level, probability, parameters_fingerprint, computed_at)
           VALUES (?, ?, ?, ?, ?, ?)""",
        [
            (session_id, construct, level, probability, fingerprint, computed_at)
            for construct, marginal in posteriors.items()
            for level, probability in marginal.items()
        ],
    )


def load_scores(
    connection: sqlite3.Connection, session_id: str
) -> tuple[dict[str, dict[str, float]], str | None]:
    rows = connection.execute(
        """SELECT construct, level, probability, parameters_fingerprint
             FROM scores WHERE session_id = ?""",
        (session_id,),
    ).fetchall()
    stored: dict[str, dict[str, float]] = {}
    fingerprint: str | None = None
    for row in rows:
        stored.setdefault(row["construct"], {})[row["level"]] = row["probability"]
        fingerprint = row["parameters_fingerprint"]
    return stored, fingerprint


def sessions(connection: sqlite3.Connection) -> Sequence[sqlite3.Row]:
    return connection.execute("SELECT * FROM sessions ORDER BY created_at").fetchall()
