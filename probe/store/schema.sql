-- Append-only evidence store.
--
-- The rule the whole instrument rests on: every row that the fold consumes
-- carries a snapshot of what it consumed. Re-running the fold over the stored
-- rows must reproduce the stored score exactly. That is what makes it possible
-- to change the parameters and rescore every past session without putting a
-- single person back through the scenario.
--
-- Nothing here is ever UPDATEd or DELETEd except `scores`, which is a cache of
-- a derivable value and records the parameter fingerprint it was derived under.

PRAGMA foreign_keys = ON;

CREATE TABLE IF NOT EXISTS sessions (
  id                TEXT PRIMARY KEY,
  created_at        TEXT NOT NULL,
  participant_label TEXT NOT NULL,     -- pseudonymous; no names in the store
  source            TEXT NOT NULL,     -- 'live' | 'persona' | 'transcript'
  scenario_id       TEXT NOT NULL,
  observer_model    TEXT NOT NULL,
  judge_model       TEXT NOT NULL,
  prompt_version    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS turns (
  session_id TEXT NOT NULL REFERENCES sessions(id),
  idx        INTEGER NOT NULL,
  role       TEXT NOT NULL CHECK (role IN ('participant', 'assistant', 'system')),
  content    TEXT NOT NULL,
  PRIMARY KEY (session_id, idx)
);

-- A task the scenario actually put in front of the participant. Absence here is
-- the difference between "did not show the behaviour" and "was never asked to".
CREATE TABLE IF NOT EXISTS surfaced_tasks (
  session_id TEXT NOT NULL REFERENCES sessions(id),
  task_id    TEXT NOT NULL,
  turn_idx   INTEGER NOT NULL,
  PRIMARY KEY (session_id, task_id)
);

-- Every observation the observer emitted, admissible or not. Inadmissible rows
-- are kept on purpose: the drop rate is the instrument's health metric, and
-- deleting them would hide a degrading prompt.
CREATE TABLE IF NOT EXISTS observations (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL REFERENCES sessions(id),
  turn_idx     INTEGER NOT NULL,
  indicator_id TEXT NOT NULL,
  strength     TEXT NOT NULL CHECK (strength IN ('weak', 'strong')),
  confidence   REAL NOT NULL CHECK (confidence BETWEEN 0 AND 1),
  quote        TEXT NOT NULL,
  task_id      TEXT,
  observer_model TEXT NOT NULL,
  prompt_version TEXT NOT NULL,
  created_at   TEXT NOT NULL,
  FOREIGN KEY (session_id, turn_idx) REFERENCES turns(session_id, idx)
);

CREATE INDEX IF NOT EXISTS observations_by_session ON observations(session_id);

-- One row per admissibility check per observation. An observation is admitted
-- only if it has no failing check; storing each verdict separately is what lets
-- the report say *why* something was excluded instead of showing a low score.
CREATE TABLE IF NOT EXISTS admissibility (
  observation_id TEXT NOT NULL REFERENCES observations(id),
  check_name     TEXT NOT NULL,
  passed         INTEGER NOT NULL CHECK (passed IN (0, 1)),
  detail         TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (observation_id, check_name)
);

-- A cache of the fold. `parameters_fingerprint` records which parameterisation
-- produced it, so a rescore after a CPT change is detectable rather than silent.
CREATE TABLE IF NOT EXISTS scores (
  session_id             TEXT NOT NULL REFERENCES sessions(id),
  construct              TEXT NOT NULL,
  level                  TEXT NOT NULL,
  probability            REAL NOT NULL,
  parameters_fingerprint TEXT NOT NULL,
  computed_at            TEXT NOT NULL,
  PRIMARY KEY (session_id, construct, level)
);
