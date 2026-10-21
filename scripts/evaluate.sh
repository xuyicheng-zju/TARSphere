#!/usr/bin/env bash
# Default-mechanism evaluation entry point; one worker on a single GPU is supported.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

PLAN_ONLY=0
SESSION_MODE=start
PHASE=physical
FOREGROUND=0
FORWARD=()
for argument in "$@"; do
  case "${argument}" in
    --plan) PLAN_ONLY=1 ;;
    -h|--help) PLAN_ONLY=1; FORWARD+=("${argument}") ;;
    --supervise) SESSION_MODE=supervise ;;
    --status) SESSION_MODE=status ;;
    --stop) SESSION_MODE=stop ;;
    --foreground) FOREGROUND=1 ;;
    --score) PHASE=scoring ;;
    *) FORWARD+=("${argument}") ;;
  esac
done

if [[ -f host_env.sh ]]; then
  set +u
  # shellcheck source=/dev/null
  source host_env.sh >/dev/null
  set -u
elif [[ "${PLAN_ONLY}" != 1 ]]; then
  echo 'Missing host_env.sh. Copy scripts/host_env.example.sh and fill paths and endpoints.' >&2
  exit 3
fi

if [[ "${PLAN_ONLY}" == 1 ]]; then
  PYTHON_BIN="${TARSPHERE_PYTHON:-$(command -v python3)}"
  if [[ ! -x "${PYTHON_BIN}" ]]; then
    echo "Planning Python is not executable: ${PYTHON_BIN}" >&2
    exit 3
  fi
  exec "${PYTHON_BIN}" tarsphere/cli/evaluate.py ${FORWARD[@]+"${FORWARD[@]}"}
fi
: "${TARSPHERE_PYTHON:?Set TARSPHERE_PYTHON in host_env.sh.}"
[[ -x "${TARSPHERE_PYTHON}" ]] || {
  echo "TARSPHERE_PYTHON is not executable: ${TARSPHERE_PYTHON}" >&2
  exit 3
}

if [[ "${FOREGROUND}" == 0 ]]; then
  exec "${TARSPHERE_PYTHON}" tarsphere/cli/session.py "${SESSION_MODE}" --phase "${PHASE}" ${FORWARD[@]+"${FORWARD[@]}"}
fi
[[ "${TARSPHERE_EVALUATION_SUPERVISED:-}" == 1 ]] || {
  echo '--foreground is reserved for the server-side supervisor.' >&2
  exit 78
}

set +u
# shellcheck source=scripts/activate.sh
source scripts/activate.sh >/dev/null || exit 78
set -u
# Local model ports must never travel through a login shell's HTTP proxy.
export NO_PROXY="127.0.0.1,localhost,::1${NO_PROXY:+,${NO_PROXY}}"
export no_proxy="${NO_PROXY}"

if [[ "${PHASE}" == scoring ]]; then
  exec "${TARSPHERE_PYTHON}" tarsphere/cli/score.py --execute ${FORWARD[@]+"${FORWARD[@]}"}
fi

# Check the selected episodes before starting support services.
PREFLIGHT=()
while IFS= read -r argument; do
  PREFLIGHT+=("${argument}")
done < <("${TARSPHERE_PYTHON}" -c 'import sys; from tarsphere.cli.evaluate import build_parser; a=build_parser().parse_args(sys.argv[1:]); print("--episodes", a.episodes, sep="\n"); [print("--episode-id", e, sep="\n") for e in a.episode_id or []]' ${FORWARD[@]+"${FORWARD[@]}"})
"${TARSPHERE_PYTHON}" tarsphere/cli/check_environment.py "${PREFLIGHT[@]}" || exit 78
TARSPHERE_PYTHON="${TARSPHERE_PYTHON}" tarsphere/infra/services/start.sh || exit 75

exec "${TARSPHERE_PYTHON}" tarsphere/cli/evaluate.py \
  --launch ${FORWARD[@]+"${FORWARD[@]}"}
