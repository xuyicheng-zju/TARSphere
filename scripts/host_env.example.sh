#!/usr/bin/env bash
# Copy to repository-root host_env.sh; fill local paths and endpoint credentials.
# Do not commit the filled file. The default uses one GPU and remote Agent/support APIs.

export TARSPHERE_PYTHON="$HOME/miniforge3/envs/tarsphere-isaac51/bin/python"
export TARSPHERE_GRASPGENX_PYTHON="$HOME/miniforge3/envs/tarsphere-graspgenx/bin/python"
export ISAAC_SIM_PATH="$HOME/isaacsim"
TARSPHERE_REPOSITORY_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export TARSPHERE_ASSETS_PATH="$HOME/data/tarsphere/assets"
export TARSPHERE_LOCAL_SCRATCH="/tmp/${USER}_tarsphere_runtime"

export TARSPHERE_SAM3_CHECKPOINT="$HOME/models/sam3.1_multiplex.pt"
export TARSPHERE_XNAVDP_ROOT="${TARSPHERE_REPOSITORY_ROOT}/third_party/NavDP/baselines/x-navdp"
export TARSPHERE_XNAVDP_CHECKPOINT="$HOME/models/x-navdp_posttrain.ckpt"
export TARSPHERE_FRONTIERNET_CHECKPOINT="$HOME/models/rgbd_11cls.pth"
export TARSPHERE_MODEL_SERVICE_GPU=0
export TARSPHERE_SAM3_HOST=127.0.0.1
export TARSPHERE_SAM3_PORT=5558
export TARSPHERE_GRASPGENX_HOST=127.0.0.1
export TARSPHERE_GRASPGENX_PORT=5557

# Default single-server Agent: provider must support image inputs and tool calls.
export TARSPHERE_AGENT_MODEL="replace-with-vision-model-id"
export TARSPHERE_AGENT_API_BASE="https://model-provider.example/v1"
export TARSPHERE_AGENT_API_KEY=""

# One support endpoint is shared by Human/NPC routing and semantic scoring.
export TARSPHERE_EVALUATION_SUPPORT_MODEL="replace-with-support-model-id"
export TARSPHERE_EVALUATION_SUPPORT_API_BASE="https://model-provider.example/v1"
export TARSPHERE_EVALUATION_SUPPORT_API_KEY=""
export TARSPHERE_EVALUATION_SUPPORT_TIMEOUT_SECONDS=180
