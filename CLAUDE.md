# probe

A research instrument that measures critical thinking in human–AI dialogue. The core is a
measurement system, not a chat: evidence is collected, admitted or rejected, and folded into a
small Bayesian network whose every score traces back to the participant's own words.

**`design/` is the source of truth**, not this file. `design/00-measurement-model.md` defines the
constructs, indicators, tasks and parameters; `design/01-architecture.md` describes the five
stages; `design/02-validation.md` records what has been checked and what that does not prove.

## Rules

- A new indicator, task or parameter changes `design/00-measurement-model.md` **first**, then the
  code. The inventories in `probe/inventory/` and `probe/scenario/tasks.py` are the executable
  form of that document and validate themselves at import.
- Indicator and task ids are permanent. Deprecate, never delete: removing one strands every
  session already scored with it and breaks the rebuild invariant.
- Never widen what the observer may emit. If no indicator fits, it emits nothing.
- `probe/score/bayes.py` stays free of I/O and of imports from the rest of the package.
- Tests run in replay mode and must never call a provider.

## Commands

```
uv sync                                 # environment from uv.lock, exactly
uv run pytest                           # all tests; also runs on the Stop hook
uv run scripts/replay.py <transcript>   # score a transcript, print the report
uv run --extra web streamlit run scripts/live.py   # live session; calls the provider
```

Dependencies are locked in `uv.lock` and the lock is committed. An instrument that
claims "same evidence, same numbers" cannot float its own dependencies, so add
packages with `uv add` rather than editing `pyproject.toml` by hand, and commit the
resulting lock change with the code that needed it.

Docs, prompts and reports are written in Russian — they are shown to Russian-speaking readers.
Code and comments are in English.
