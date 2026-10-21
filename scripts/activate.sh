#!/usr/bin/env bash
# Configure the built-in TARSphere simulator and Isaac Sim.

if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
  echo "Source this file instead of executing it:" >&2
  echo "  source scripts/activate.sh" >&2
  exit 1
fi

TARSPHERE_REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export TARSPHERE_REPOSITORY_ROOT

# stage_runtime_local.sh writes these overrides after a successful NFS mirror.
# Loading them here makes every later shell and launch use the local trees
# without requiring hand-edits to host_env.sh.
if [[ -n "${TARSPHERE_LOCAL_SCRATCH:-}" ]] \
  && [[ -f "${TARSPHERE_LOCAL_SCRATCH}/runtime_env.sh" ]]; then
  # shellcheck source=/dev/null
  source "${TARSPHERE_LOCAL_SCRATCH}/runtime_env.sh"
fi

: "${ISAAC_SIM_PATH:?Set ISAAC_SIM_PATH to the Isaac Sim 5.1 standalone directory.}"
: "${TARSPHERE_PYTHON:?Set TARSPHERE_PYTHON to the Isaac evaluation environment Python.}"

if [[ ! -x "${TARSPHERE_PYTHON}" ]]; then
  echo "TARSphere Python is not executable: ${TARSPHERE_PYTHON}" >&2
  return 1
fi
TARSPHERE_ENV_PREFIX="$(cd "$(dirname "$(dirname "${TARSPHERE_PYTHON}")")" && pwd -P)"
_declared_python="${TARSPHERE_ENV_PREFIX}/bin/$(basename "${TARSPHERE_PYTHON}")"
_active_python="$(command -v python 2>/dev/null || true)"
_active_prefix="${CONDA_PREFIX:-}"
if [[ -n "${_active_prefix}" && -d "${_active_prefix}" ]]; then
  _active_prefix="$(cd "${_active_prefix}" && pwd -P)"
fi
if [[ "${_active_prefix}" != "${TARSPHERE_ENV_PREFIX}" ]] \
  || [[ "${_active_python}" != "${_declared_python}" ]]; then
  if ! command -v conda >/dev/null; then
    echo "conda is required to activate ${TARSPHERE_ENV_PREFIX}." >&2
    return 1
  fi
  eval "$(conda shell.bash hook)"
  conda activate "${TARSPHERE_ENV_PREFIX}" || return 1
fi
_active_python="$(command -v python 2>/dev/null || true)"
_active_prefix="${CONDA_PREFIX:-}"
if [[ -n "${_active_prefix}" && -d "${_active_prefix}" ]]; then
  _active_prefix="$(cd "${_active_prefix}" && pwd -P)"
fi
if [[ "${_active_prefix}" != "${TARSPHERE_ENV_PREFIX}" ]] \
  || [[ "${_active_python}" != "${_declared_python}" ]]; then
  echo "Activated environment ${_active_prefix:-missing} with Python ${_active_python:-missing}; expected ${TARSPHERE_ENV_PREFIX} with ${_declared_python}." >&2
  return 1
fi
unset _active_prefix _active_python _declared_python
export TARSPHERE_ENV_PREFIX

if [[ ! -f "$ISAAC_SIM_PATH/setup_conda_env.sh" ]]; then
  echo "Isaac Sim setup script is missing: $ISAAC_SIM_PATH/setup_conda_env.sh" >&2
  return 1
fi
if [[ ! -d "$TARSPHERE_REPOSITORY_ROOT/tarsphere/simulator/core" ]] \
  || [[ ! -d "$TARSPHERE_REPOSITORY_ROOT/tarsphere/simulator/extensions" ]]; then
  echo "Built-in simulator is incomplete: $TARSPHERE_REPOSITORY_ROOT/tarsphere/simulator" >&2
  return 1
fi

export TARSPHERE_ASSETS_PATH="${TARSPHERE_ASSETS_PATH:-$TARSPHERE_REPOSITORY_ROOT/assets}"
if [[ ! -d "$TARSPHERE_ASSETS_PATH" ]]; then
  echo "TARSphere asset root is missing: $TARSPHERE_ASSETS_PATH" >&2
  return 1
fi

# Isaac's script adds its Python and shared-library directories to the active
# conda environment. The repository owns the simulation source.
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}"
source "$ISAAC_SIM_PATH/setup_conda_env.sh"

# Isaac extensions import the torch build shipped in omni.isaac.ml_archive.
# Its CUDA 12.8 cuSPARSE must resolve the matching nvJitLink, not an older
# nvidia wheel left in the active conda environment (for example CUDA 12.6).
# Otherwise SimulationApp fails during extension startup with a missing
# __nvJitLinkCreate_12_8 symbol before any scene is opened.
ISAAC_ML_NVIDIA_ROOT="$ISAAC_SIM_PATH/exts/omni.isaac.ml_archive/pip_prebundle/nvidia"
ISAAC_ML_NVJITLINK_LIB="$ISAAC_ML_NVIDIA_ROOT/nvjitlink/lib"
if [[ ! -f "$ISAAC_ML_NVJITLINK_LIB/libnvJitLink.so.12" ]]; then
  echo "Isaac's bundled CUDA nvJitLink library is missing: $ISAAC_ML_NVJITLINK_LIB" >&2
  return 1
fi
export ISAAC_ML_NVIDIA_ROOT ISAAC_ML_NVJITLINK_LIB
export LD_LIBRARY_PATH="$ISAAC_ML_NVJITLINK_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
# Kit can load a same-SONAME CUDA library before Python imports torch, at which
# point LD_LIBRARY_PATH can no longer replace it. Preload the matched library
# at process creation; the 12.8 build also exports the older 12.6 symbols used
# by conda-side tools in this runtime.
export LD_PRELOAD="$ISAAC_ML_NVJITLINK_LIB/libnvJitLink.so.12${LD_PRELOAD:+:$LD_PRELOAD}"

# Isaac's conda setup intentionally omits extension-managed OpenUSD modules.
# TARSphere scene grounding imports pxr before SimulationApp starts, so expose the
# one USD distribution owned by this exact Isaac installation together with
# its shared libraries. Do not rely on a separately installed usd-core wheel:
# mixing USD builds in one runtime is ABI-unsafe.
_isaac_usd_roots=()
while IFS= read -r _isaac_usd_root; do
  _isaac_usd_roots+=("$_isaac_usd_root")
done < <(
  find "$ISAAC_SIM_PATH/extscache" -mindepth 1 -maxdepth 1 -type d \
    -name 'omni.usd.libs-*' -print
)
unset _isaac_usd_root
if [[ "${#_isaac_usd_roots[@]}" -ne 1 ]]; then
  echo "Expected exactly one Isaac OpenUSD distribution; found ${#_isaac_usd_roots[@]}." >&2
  return 1
fi
ISAAC_USD_ROOT="${_isaac_usd_roots[0]}"
export ISAAC_USD_ROOT
export LD_LIBRARY_PATH="$ISAAC_USD_ROOT/bin${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="$TARSPHERE_REPOSITORY_ROOT:$ISAAC_USD_ROOT${PYTHONPATH:+:$PYTHONPATH}"

if [[ -z "${VK_ICD_FILENAMES:-}" ]] \
  && [[ -f /usr/share/vulkan/icd.d/nvidia_icd.json ]]; then
  export VK_ICD_FILENAMES=/usr/share/vulkan/icd.d/nvidia_icd.json
fi

# Kit writes its shader, OGN and UJITSO derived-data caches constantly during
# scene load. A cache on a network mount costs as much as a remote asset tree,
# so pin them next to the staged runtime whenever a local scratch root exists.
if [[ -n "${TARSPHERE_LOCAL_SCRATCH:-}" ]]; then
  if [[ -d "$TARSPHERE_LOCAL_SCRATCH" ]]; then
    export XDG_CACHE_HOME="${XDG_CACHE_HOME:-$TARSPHERE_LOCAL_SCRATCH/cache/xdg}"
    export OMNI_USER_CACHE="${OMNI_USER_CACHE:-$TARSPHERE_LOCAL_SCRATCH/cache/ov}"
    mkdir -p "$XDG_CACHE_HOME" "$OMNI_USER_CACHE"
  else
    echo "TARSPHERE_LOCAL_SCRATCH is set but missing: $TARSPHERE_LOCAL_SCRATCH" >&2
    echo "Run scripts/stage.sh or unset the variable." >&2
    return 1
  fi
fi

echo "TARSphere runtime configured:"
echo "  repository:   $TARSPHERE_REPOSITORY_ROOT"
echo "  Isaac Sim:    $ISAAC_SIM_PATH"
echo "  simulator:    $TARSPHERE_REPOSITORY_ROOT/tarsphere/simulator"
echo "  assets:       $TARSPHERE_ASSETS_PATH"
echo "  OpenUSD:      $ISAAC_USD_ROOT"

# Report remote roots before scene loading. A missing runtime dependency
# should not block shell activation.
if _fs_report="$("${TARSPHERE_PYTHON}" -m tarsphere.infra.simulation.local_stage 2>/dev/null)"; then
  echo "$_fs_report"
else
  _fs_status=$?
  if [[ -n "${_fs_report:-}" ]]; then
    echo "$_fs_report"
    if [[ "$_fs_status" -eq 1 ]]; then
      echo "WARNING: runtime roots above are on a network filesystem." >&2
      echo "  Network metadata latency can slow MDL material loading." >&2
      echo "  Mirror them locally: scripts/stage.sh" >&2
    fi
  fi
fi
unset _fs_report _fs_status
