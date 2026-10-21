#!/usr/bin/env bash
# Start a scene-owned policy server; explicit endpoints permit sequential reuse.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${TARSPHERE_PYTHON:?Set TARSPHERE_PYTHON in the host environment.}"
PYTHON_BIN="${TARSPHERE_PYTHON}"
XNAV_ROOT="${TARSPHERE_XNAVDP_ROOT:-${ROOT}/third_party/NavDP/baselines/x-navdp}"
XNAV_CHECKPOINT="${TARSPHERE_XNAVDP_CHECKPOINT:-${ROOT}/assets/models/x-navdp/x-navdp_posttrain.ckpt}"
XNAV_PORT="${TARSPHERE_XNAVDP_PORT:?Set the scene-owned TARSPHERE_XNAVDP_PORT explicitly.}"
NEW_SERVER=0
if [[ $# == 1 && "$1" == --new-server ]]; then
  NEW_SERVER=1
elif [[ $# == 1 && "$1" == --reuse-existing ]]; then
  NEW_SERVER=0
elif [[ $# != 0 ]]; then
  echo 'Usage: start_x_navdp_server.sh [--new-server|--reuse-existing]' >&2
  exit 3
fi
if [[ ! "$XNAV_PORT" =~ ^[0-9]{1,5}$ ]] \
  || (( 10#$XNAV_PORT < 1 || 10#$XNAV_PORT > 65535 )); then
  echo 'TARSPHERE_XNAVDP_PORT must be an integer in 1..65535.' >&2
  exit 3
fi
XNAV_PORT=$((10#$XNAV_PORT))
MODEL_GPU="${TARSPHERE_MODEL_GPU:-0}"
HOST_ID="$(hostname -s | tr -cd '[:alnum:]_.-')"
STATE_DIR="${TARSPHERE_XNAVDP_STATE_DIR:-${HOME}/.cache/tarsphere/${HOST_ID}/x_navdp}"
LOG_DIR="${TARSPHERE_XNAVDP_LOG_DIR:-${STATE_DIR}/logs}"
PID_FILE="${STATE_DIR}/x-navdp-${XNAV_PORT}.pid"
LOG_FILE="${LOG_DIR}/x-navdp-${XNAV_PORT}.log"

[[ -x "${PYTHON_BIN}" ]] || { echo "Missing runtime Python: ${PYTHON_BIN}" >&2; exit 3; }
[[ -f "${XNAV_ROOT}/eval/src/policy_server.py" ]] || { echo "Missing official X-NavDP checkout: ${XNAV_ROOT}" >&2; exit 3; }
[[ -f "${XNAV_CHECKPOINT}" ]] || { echo "Missing X-NavDP checkpoint: ${XNAV_CHECKPOINT}" >&2; exit 3; }

port_ready() {
  XNAV_PORT="${XNAV_PORT}" "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import os
import socket
with socket.create_connection(('127.0.0.1', int(os.environ['XNAV_PORT'])), timeout=1.0):
    pass
PY
}

if port_ready; then
  if [[ "$NEW_SERVER" == 1 ]]; then
    echo "Derived X-NavDP port ${XNAV_PORT} is occupied; refusing implicit shared history. Set an explicit unused endpoint, or explicitly reuse it only after its previous scene stops." >&2
    exit 4
  fi
  if pgrep -af 'eval.src.policy_server' | grep -Eq -- "--port ${XNAV_PORT}([[:space:]]|$)"; then
    echo "[XNAVDP] Explicitly reusing 127.0.0.1:${XNAV_PORT}; the caller must ensure no concurrent scene uses it."
    exit 0
  fi
  echo "Port ${XNAV_PORT} is occupied by a process that is not X-NavDP." >&2
  exit 4
fi

mkdir -p "${STATE_DIR}" "${LOG_DIR}"
: > "${LOG_FILE}"
pushd "${XNAV_ROOT}" >/dev/null
nohup env CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
  "${PYTHON_BIN}" -m eval.src.policy_server \
    --port "${XNAV_PORT}" --embodiment humanoid \
    --checkpoint "${XNAV_CHECKPOINT}" --device cuda:0 --no-visualization \
    > "${LOG_FILE}" 2>&1 &
server_pid=$!
popd >/dev/null
echo "${server_pid}" > "${PID_FILE}"

for _ in $(seq 1 120); do
  if ! kill -0 "${server_pid}" 2>/dev/null; then
    echo "X-NavDP exited during startup:" >&2
    tail -n 80 "${LOG_FILE}" >&2 || true
    exit 4
  fi
  if port_ready; then
    echo "[XNAVDP] Listening on 127.0.0.1:${XNAV_PORT}; pid=${server_pid}; gpu=${MODEL_GPU}."
    exit 0
  fi
  sleep 2
done

echo "X-NavDP did not listen within 240 seconds." >&2
tail -n 80 "${LOG_FILE}" >&2 || true
exit 4
