#!/usr/bin/env bash
# Start one self-healing SSH forward to the four open-model services.

set -euo pipefail

OPEN_MODEL_HOST="${TARSPHERE_OPEN_MODEL_HOST:-127.0.0.1}"
if [[ "${OPEN_MODEL_HOST}" != "127.0.0.1" && "${OPEN_MODEL_HOST}" != "localhost" ]]; then
  echo "[MODEL-TUNNEL] Direct open-model host configured: ${OPEN_MODEL_HOST}."
  exit 0
fi

: "${TARSPHERE_OPEN_MODEL_SSH_TARGET:?Set TARSPHERE_OPEN_MODEL_SSH_TARGET in host_env.sh.}"
: "${TARSPHERE_OPEN_MODEL_SSH_PORT:?Set TARSPHERE_OPEN_MODEL_SSH_PORT in host_env.sh.}"
SSH_TARGET="${TARSPHERE_OPEN_MODEL_SSH_TARGET}"
SSH_PORT="${TARSPHERE_OPEN_MODEL_SSH_PORT}"
if [[ ! "${SSH_PORT}" =~ ^[0-9]{1,5}$ ]] \
  || (( 10#${SSH_PORT} < 1 || 10#${SSH_PORT} > 65535 )); then
  echo 'TARSPHERE_OPEN_MODEL_SSH_PORT must be an integer in 1..65535.' >&2
  exit 3
fi
SSH_PORT=$((10#${SSH_PORT}))
HOST_ID="$(hostname -s | tr -cd '[:alnum:]_.-')"
STATE_DIR="${TARSPHERE_MODEL_TUNNEL_STATE_DIR:-${HOME}/.cache/tarsphere/${HOST_ID}/open_model_tunnel}"
PID_FILE="${STATE_DIR}/supervisor.pid"
LOG_FILE="${STATE_DIR}/supervisor.log"
mkdir -p "${STATE_DIR}"

forward_command=(
  ssh -p "${SSH_PORT}" -N -T
  -o BatchMode=yes
  -o StrictHostKeyChecking=accept-new
  -o ExitOnForwardFailure=yes
  -o TCPKeepAlive=yes
  -o ServerAliveInterval=15
  -o ServerAliveCountMax=3
  -o ConnectTimeout=15
)
for port in 8001 8002 8003 8004; do
  forward_command+=(
    -L "127.0.0.1:${port}:127.0.0.1:${port}"
  )
done
forward_command+=("${SSH_TARGET}")

if [[ "${1:-}" == "--supervise" ]]; then
  # flock is inherited by ssh; stale PID files cannot create parallel tunnels.
  exec 9>"${STATE_DIR}/tunnel.lock"
  flock -n 9 || exit 0
  child_pid=''
  stop_tunnel() {
    [[ -z "${child_pid}" ]] || kill "${child_pid}" 2>/dev/null || true
    exit 0
  }
  trap stop_tunnel TERM INT
  delay=2
  while true; do
    printf '[%s] opening model tunnel to %s:%s\n' \
      "$(date -Is)" "${SSH_TARGET}" "${SSH_PORT}"
    set +e
    "${forward_command[@]}" &
    child_pid=$!
    started=${SECONDS}
    wait "${child_pid}"
    status=$?
    set -e
    child_pid=''
    (( SECONDS - started < 60 )) || delay=2
    printf '[%s] tunnel exited status=%s; retrying in %ss\n' \
      "$(date -Is)" "${status}" "${delay}"
    sleep "${delay}"
    delay=$(( delay < 60 ? delay * 2 : 60 ))
    (( delay <= 60 )) || delay=60
  done
fi

exec 8>"${STATE_DIR}/startup.lock"
flock -x 8
supervisor_pid=""
if [[ -s "${PID_FILE}" ]]; then
  supervisor_pid="$(tr -dc '0-9' < "${PID_FILE}")"
fi
if [[ -z "${supervisor_pid}" ]] || ! kill -0 "${supervisor_pid}" 2>/dev/null \
  || ! tr '\0' ' ' < "/proc/${supervisor_pid}/cmdline" \
    | grep -q 'start_open_model_tunnel.sh --supervise'; then
  rm -f "${PID_FILE}"
  nohup setsid "$0" --supervise > "${LOG_FILE}" 2>&1 < /dev/null 8>&- &
  supervisor_pid=$!
  printf '%s\n' "${supervisor_pid}" > "${PID_FILE}"
fi

sleep 1
if ! kill -0 "${supervisor_pid}" 2>/dev/null; then
  echo "Tunnel supervisor exited; inspect ${LOG_FILE}." >&2
  exit 1
fi
echo "[MODEL-TUNNEL] Server-owned supervisor active: pid=${supervisor_pid}."
echo '[MODEL-TUNNEL] Scheduler probes all four exact model IDs and inference before recovered lanes launch.'
