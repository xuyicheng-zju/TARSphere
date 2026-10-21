#!/usr/bin/env bash
# Build the single supported Isaac evaluation environment on a new host.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${ROOT}"

if [[ $# -ne 0 ]]; then
  echo 'Usage: scripts/setup.sh' >&2
  exit 2
fi

if [[ ! -f host_env.sh ]]; then
  echo "Missing host_env.sh." >&2
  echo "Copy scripts/host_env.example.sh and fill every path first." >&2
  exit 3
fi
set +u
# shellcheck source=/dev/null
source host_env.sh
set -u

: "${ISAAC_SIM_PATH:?Set ISAAC_SIM_PATH to Isaac Sim 5.1 standalone.}"
: "${TARSPHERE_PYTHON:?Set TARSPHERE_PYTHON to the Isaac evaluation environment Python.}"
: "${TARSPHERE_GRASPGENX_PYTHON:?Set TARSPHERE_GRASPGENX_PYTHON to the isolated GraspGenX Python.}"

TARSPHERE_ENV_PREFIX="$(dirname "$(dirname "${TARSPHERE_PYTHON}")")"
GRASPGENX_ENV_PREFIX="$(dirname "$(dirname "${TARSPHERE_GRASPGENX_PYTHON}")")"
[[ "${TARSPHERE_ENV_PREFIX}" != "${GRASPGENX_ENV_PREFIX}" ]] || {
  echo 'Isaac and GraspGenX require different environment prefixes.' >&2
  exit 3
}

command -v conda >/dev/null || {
  echo "conda is required to create the evaluation environment." >&2
  exit 3
}
eval "$(conda shell.bash hook)"
if [[ -x "${TARSPHERE_PYTHON}" ]]; then
  conda env update --prefix "${TARSPHERE_ENV_PREFIX}" -f environment.yml
else
  mkdir -p "$(dirname "${TARSPHERE_ENV_PREFIX}")"
  conda env create --prefix "${TARSPHERE_ENV_PREFIX}" -f environment.yml
fi
conda activate "${TARSPHERE_ENV_PREFIX}"
if [[ "$(command -v python)" != "${TARSPHERE_PYTHON}" ]]; then
  echo "Conda activated $(command -v python), expected TARSPHERE_PYTHON=${TARSPHERE_PYTHON}." >&2
  exit 3
fi

# Dependencies below are vendored source, not Git submodule links. Never pull
# mutable upstream code during environment construction.
for required_source in third_party/GraspGenX/pyproject.toml third_party/curobo/pyproject.toml ; do
  [[ -f "${required_source}" ]] || {
    echo "Missing frozen source: ${required_source}. Restore the release snapshot." >&2
    exit 3
  }
done
if [[ -n "${TARSPHERE_LOCAL_SCRATCH:-}" ]]; then
  mkdir -p "${TARSPHERE_LOCAL_SCRATCH}"
fi
# Install into the new Conda environments before adding Isaac's bundled Python
# packages. Otherwise pip may accept SDK archives instead of installing the
# environment's declared dependencies.
unset PYTHONPATH

# SAM3 still imports pkg_resources; setuptools 81+ removed that module.
# Apply one stack throughout installation so later transitive dependencies
# cannot upgrade NumPy or replace only one half of the torch/torchvision pair.
export PIP_CONSTRAINT="${ROOT}/scripts/runtime-constraints.txt"
export PIP_EXTRA_INDEX_URL="https://download.pytorch.org/whl/cu126"
python -m pip install --upgrade pip 'setuptools==80.9.0' wheel
python -m pip install -e '.[sim,eval,grasp,dev]'
python -m tarsphere.cli.runtime
python -m pip install 'hdbscan>=0.8.40' 'ml-dtypes==0.5.4' \
  'opencv-python==4.11.0.86' 'einops==0.8.2' 'imageio-ffmpeg==0.6.0' 'psutil>=6.0'

# CuRobo runs in an isolated subprocess under this same TARSPHERE_PYTHON. Its
# optional CUDA/Torch dependencies are not installed by the base project or
# by curobo's default extra; omitting them fails only when a fetch reaches IK.
python -m pip install 'torch==2.6.0+cu126' 'torchvision==0.21.0+cu126' \
  --index-url https://download.pytorch.org/whl/cu126
python -m pip install -e third_party/sam3
python -m pip install 'warp-lang==1.16.0' 'cuda-core[cu12]==1.1.1' \
  'setuptools-scm==10.2.3'
SETUPTOOLS_SCM_PRETEND_VERSION=0.8.0 python -m pip install -e third_party/curobo
python - <<'PY'
import torch
import warp
import cuda.core
from tarsphere.infra.manipulation.g1_grasp import _require_curobo_v2

_require_curobo_v2()
print(f'CuRobo environment ready: torch={torch.__version__}, warp={warp.__version__}')
PY

# GraspGenX pins a different ML dependency stack. Keep its model server out of
# Isaac's Python while retaining only the lightweight ZMQ client dependencies
# in tarsphere-isaac51 through the repository's `grasp` extra.
unset PIP_CONSTRAINT PIP_EXTRA_INDEX_URL
if [[ ! -x "${TARSPHERE_GRASPGENX_PYTHON}" ]]; then
  mkdir -p "$(dirname "${GRASPGENX_ENV_PREFIX}")"
  conda create -y --prefix "${GRASPGENX_ENV_PREFIX}" \
    python=3.11 pip setuptools wheel
fi
conda run --prefix "${GRASPGENX_ENV_PREFIX}" \
  python -m pip install --upgrade pip setuptools wheel
conda run --prefix "${GRASPGENX_ENV_PREFIX}" python -m pip install \
  -r scripts/graspgenx-requirements.txt

python -m pip check
"${TARSPHERE_GRASPGENX_PYTHON}" -m pip check
# Import the real service modules before any GPU processes are launched.
python - <<'PY_MAIN'
import sys
from pathlib import Path
import cv2
import hdbscan
from sam3.model_builder import build_sam3_image_model
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

sys.path.insert(0, str(Path("third_party/NavDP/baselines/x-navdp").resolve()))
from eval.src.policy_backbone import NavDP_RGBD_Backbone
from eval.src.policy_server import app
print("SAM3, X-NavDP, FrontierNet and image-processing dependencies import successfully.")
PY_MAIN
"${TARSPHERE_GRASPGENX_PYTHON}" - <<'PY_GRASP'
import sys
from pathlib import Path
sys.path.insert(0, str(Path("third_party/GraspGenX").resolve()))
from graspgenx.serving.zmq_server import GraspGenXZMQServer
print("GraspGenX service dependencies import successfully.")
PY_GRASP
set +u
# shellcheck source=scripts/activate.sh
source scripts/activate.sh >/dev/null
set -u
python tarsphere/cli/check_environment.py
echo "Environment ready. Run scripts/evaluate.sh --plan, then scripts/evaluate.sh for the example episode."
