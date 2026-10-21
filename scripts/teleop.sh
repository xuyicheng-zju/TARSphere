#!/usr/bin/env bash
# Interactive robot and animated NPC control in one Isaac Sim scene.
set -euo pipefail
TELEOP_REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TELEOP_REPO_ROOT"
TELEOP_RUNTIME_REQUIRED=1
for argument in "$@"; do
  if [[ "$argument" == --help || "$argument" == -h ]]; then
    TELEOP_RUNTIME_REQUIRED=0
  fi
done
if [[ "$TELEOP_RUNTIME_REQUIRED" == 1 ]]; then
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
exec "${TARSPHERE_PYTHON:-python3}" -m tarsphere.cli.teleop "$@"
