#!/usr/bin/env bash
# Start the one shared SAM3 instance and one shared GraspGenX instance.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
: "${TARSPHERE_PYTHON:?Set TARSPHERE_PYTHON in the host environment.}"
PYTHON_BIN="${TARSPHERE_PYTHON}"
GRASP_PYTHON="${TARSPHERE_GRASPGENX_PYTHON:-${PYTHON_BIN}}"
MODEL_GPU="${TARSPHERE_MODEL_SERVICE_GPU:-0}"
SAM3_HOST="${TARSPHERE_SAM3_HOST:-127.0.0.1}"
SAM3_PORT="${TARSPHERE_SAM3_PORT:-5558}"
GRASP_HOST="${TARSPHERE_GRASPGENX_HOST:-127.0.0.1}"
GRASP_PORT="${TARSPHERE_GRASPGENX_PORT:-5557}"
HOST_ID="$(hostname -s | tr -cd '[:alnum:]_.-')"
STATE_ROOT="${TARSPHERE_MODEL_SERVICE_STATE_DIR:-${HOME}/.cache/tarsphere/${HOST_ID}/model_services}"
CHECK_ONLY=0
DRY_RUN=0

usage() {
  cat <<'EOF'
Usage: tarsphere/infra/services/start.sh [--check-only] [--dry-run]

Environment:
  TARSPHERE_PYTHON                      Python from tarsphere-isaac51
  TARSPHERE_GRASPGENX_PYTHON        Isolated GraspGenX service Python
  TARSPHERE_MODEL_SERVICE_GPU       Physical GPU index (default: 0)
  TARSPHERE_SAM3_CHECKPOINT         Required SAM3.1 checkpoint
  TARSPHERE_SAM3_HOST/PORT          Shared SAM3 endpoint (127.0.0.1:5558)
  TARSPHERE_GRASPGENX_HOST/PORT     Shared GraspGenX endpoint (127.0.0.1:5557)
  TARSPHERE_MODEL_SERVICE_STATE_DIR Host-local PID and log directory
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only) CHECK_ONLY=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage >&2; exit 2 ;;
  esac
done

[[ -x "${PYTHON_BIN}" ]] || {
  echo "Runtime Python is not executable: ${PYTHON_BIN}" >&2
  exit 3
}
[[ -x "${GRASP_PYTHON}" ]] || {
  echo "GraspGenX Python is not executable: ${GRASP_PYTHON}" >&2
  exit 3
}
[[ "${MODEL_GPU}" =~ ^[0-9]+$ ]] || {
  echo "TARSPHERE_MODEL_SERVICE_GPU must be a non-negative GPU index." >&2
  exit 3
}
# Services and Isaac workers share the caller's permitted physical GPU set.
if [[ -n "${CUDA_VISIBLE_DEVICES+x}" ]]; then
  MODEL_GPU="$(PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" "${PYTHON_BIN}" -m tarsphere.cli.gpu_selection)"
fi
: "${TARSPHERE_SAM3_CHECKPOINT:?Set TARSPHERE_SAM3_CHECKPOINT.}"
[[ -f "${TARSPHERE_SAM3_CHECKPOINT}" ]] || {
  echo "SAM3 checkpoint is missing: ${TARSPHERE_SAM3_CHECKPOINT}" >&2
  exit 3
}

mkdir -p "${STATE_ROOT}"
SAM3_PID_FILE="${STATE_ROOT}/sam3.pid"
SAM3_LOG="${STATE_ROOT}/sam3.log"
GRASP_PID_FILE="${STATE_ROOT}/graspgenx.pid"
GRASP_LOG="${STATE_ROOT}/graspgenx.log"

sam3_healthy() {
  TARSPHERE_SAM3_HOST="${SAM3_HOST}" \
  TARSPHERE_SAM3_PORT="${SAM3_PORT}" \
  TARSPHERE_SAM3_REQUEST_TIMEOUT_SECONDS=5 \
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
from tarsphere.harness.perception.remote_backends import SAM3ServiceClient
SAM3ServiceClient().request({'cmd': 'sam3_require_ready'})
PY
}

grasp_healthy() {
  TARSPHERE_GRASPGENX_HOST="${GRASP_HOST}" \
  TARSPHERE_GRASPGENX_PORT="${GRASP_PORT}" \
  TARSPHERE_GRASPGENX_HEALTH_TIMEOUT_MS=5000 \
  PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import os

from tarsphere.infra.manipulation.graspgenx import (
    GraspGenXClient,
    GraspGenXSettings,
)

client = GraspGenXClient(settings=GraspGenXSettings(
    host=os.environ['TARSPHERE_GRASPGENX_HOST'],
    port=int(os.environ['TARSPHERE_GRASPGENX_PORT']),
    request_timeout_ms=int(os.environ['TARSPHERE_GRASPGENX_HEALTH_TIMEOUT_MS']),
))
try:
    client.require_ready()
finally:
    client.close()
PY
}

if [[ "${CHECK_ONLY}" == 1 ]]; then
  sam3_healthy || { echo "SAM3 unhealthy at ${SAM3_HOST}:${SAM3_PORT}" >&2; exit 4; }
  grasp_healthy || { echo "GraspGenX unhealthy at ${GRASP_HOST}:${GRASP_PORT}" >&2; exit 4; }
  echo "[MODELS] SAM3 and GraspGenX are healthy."
  exit 0
fi

if [[ "${DRY_RUN}" == 1 ]]; then
  echo "[DRY-RUN] GPU ${MODEL_GPU}: SAM3 ${SAM3_HOST}:${SAM3_PORT}"
  echo "[DRY-RUN] GPU ${MODEL_GPU}: GraspGenX ${GRASP_HOST}:${GRASP_PORT}"
  echo "[DRY-RUN] state=${STATE_ROOT}"
  exit 0
fi

cd "${ROOT}"
if sam3_healthy; then
  echo "[MODELS] Reusing healthy SAM3 at ${SAM3_HOST}:${SAM3_PORT}."
else
  rm -f "${SAM3_PID_FILE}"
  : > "${SAM3_LOG}"
  nohup setsid env CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
    PYTHONPATH="${ROOT}${PYTHONPATH:+:${PYTHONPATH}}" \
    "${PYTHON_BIN}" -m tarsphere.infra.services.sam3 \
      --host "${SAM3_HOST}" --port "${SAM3_PORT}" \
      --checkpoint "${TARSPHERE_SAM3_CHECKPOINT}" \
      > "${SAM3_LOG}" 2>&1 < /dev/null &
  echo "$!" > "${SAM3_PID_FILE}"
  echo "[MODELS] Starting SAM3 on GPU ${MODEL_GPU}; pid=$!."
fi

if grasp_healthy; then
  echo "[MODELS] Reusing healthy GraspGenX at ${GRASP_HOST}:${GRASP_PORT}."
else
  rm -f "${GRASP_PID_FILE}"
  : > "${GRASP_LOG}"
  nohup setsid env -u PYTHONPATH CUDA_VISIBLE_DEVICES="${MODEL_GPU}" \
    "${GRASP_PYTHON}" "${ROOT}/tarsphere/infra/services/graspgenx.py" \
      --host "${GRASP_HOST}" --port "${GRASP_PORT}" \
      > "${GRASP_LOG}" 2>&1 < /dev/null &
  echo "$!" > "${GRASP_PID_FILE}"
  echo "[MODELS] Starting GraspGenX on GPU ${MODEL_GPU}; pid=$!."
fi

for _ in $(seq 1 180); do
  if sam3_healthy && grasp_healthy; then
    echo "[MODELS] Ready: SAM3=${SAM3_HOST}:${SAM3_PORT} GraspGenX=${GRASP_HOST}:${GRASP_PORT}."
    exit 0
  fi
  for spec in "${SAM3_PID_FILE}:${SAM3_LOG}:SAM3" "${GRASP_PID_FILE}:${GRASP_LOG}:GraspGenX"; do
    IFS=: read -r pid_file log_file name <<<"${spec}"
    if [[ -s "${pid_file}" ]]; then
      pid="$(tr -dc '0-9' < "${pid_file}")"
      if [[ -n "${pid}" ]] && ! kill -0 "${pid}" 2>/dev/null; then
        echo "${name} exited during startup:" >&2
        tail -n 100 "${log_file}" >&2 || true
        exit 4
      fi
    fi
  done
  sleep 2
done

echo "Model services did not become ready within 360 seconds." >&2
tail -n 100 "${SAM3_LOG}" >&2 || true
tail -n 100 "${GRASP_LOG}" >&2 || true
exit 4
