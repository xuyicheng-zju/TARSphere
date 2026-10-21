#!/usr/bin/env bash
# Public-demo entry point. Local plans never activate Isaac or contact models.
set -euo pipefail
DEMO_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$DEMO_REPO_ROOT"
if [[ "${1:-}" == teleop ]]; then
  shift
  exec "$DEMO_REPO_ROOT/scripts/teleop.sh" "$@"
fi
DEMO_RUNTIME_REQUIRED=0
if [[ "${1:-}" == record ]]; then
  DEMO_RUNTIME_REQUIRED=1
fi
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    DEMO_RUNTIME_REQUIRED=0
  fi
done
if [[ "$DEMO_RUNTIME_REQUIRED" == 1 ]]; then
  if [[ -f host_env.sh ]]; then
    set +u
    source host_env.sh >/dev/null
    set -u
  fi
  : "${TARSPHERE_PYTHON:?Set TARSPHERE_PYTHON to the configured Isaac Python executable.}"
  set +u
  source scripts/activate.sh
  set -u
fi
exec "${TARSPHERE_PYTHON:-python3}" tarsphere/cli/demo.py "$@"
