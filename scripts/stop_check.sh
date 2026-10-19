#!/bin/sh
# Stop hook: the measurement core must stay green.
#
# Three exit paths, each deliberate:
#   0 — tests passed, or pytest found no tests yet (code 5, normal early in the build)
#   0 — this hook already blocked once in this stop sequence (`stop_hook_active`), so we
#       let the agent stop and report instead of looping on a failure it cannot fix
#   2 — tests failed; stderr goes back to the agent and the turn does not end
set -u

payload=$(cat)
already_blocked=$(printf '%s' "$payload" | /usr/bin/python3 -c \
  'import json,sys;print("1" if json.load(sys.stdin).get("stop_hook_active") else "0")' \
  2>/dev/null || echo 0)

if [ "$already_blocked" = "1" ]; then
  exit 0
fi

dir="${CLAUDE_PROJECT_DIR:-$(dirname "$0")/..}"
cd "$dir" || exit 0

# uv keeps the environment reproducible, but the hook must not depend on it:
# a machine without uv still has to be able to run the tests.
if command -v uv >/dev/null 2>&1 && [ -f uv.lock ]; then
  out=$(uv run --frozen python -m pytest -q 2>&1)
elif [ -x .venv/bin/python ]; then
  out=$(.venv/bin/python -m pytest -q 2>&1)
else
  out=$(python3 -m pytest -q 2>&1)
fi
code=$?

# 5 = "no tests collected". Expected until the first test file lands, and not a failure.
if [ "$code" -eq 0 ] || [ "$code" -eq 5 ]; then
  exit 0
fi

printf '%s\n' "$out" >&2
printf '\npytest exited %s. Fix the failures, or stop and ask the user if the cause is outside your control.\n' "$code" >&2
exit 2
