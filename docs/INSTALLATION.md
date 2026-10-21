# Installation

## Choose an environment

| Purpose | Requirements |
| --- | --- |
| Run physical episodes | Ubuntu 22.04/24.04 x86-64, NVIDIA RTX GPU/driver and Vulkan, Conda, Isaac Sim 5.1 standalone, assets and model APIs. |
| Single-GPU deployment | One RTX 4090 (24 GiB), one Isaac worker, remote agent/support models. Scene and service memory must fit together. |

Isaac Sim 5.1 uses Python 3.11. Install its standalone distribution, which provides `setup_conda_env.sh`, before running project setup. Follow NVIDIA's [5.1 requirements](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/requirements.html) and [workstation installation](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/installation/install_workstation.html) to prepare the driver, Vulkan, and SDK. Physical deployment uses Linux.

Use a complete source checkout or extract the source archive for the commands below. The wheel contains the importable library and package resources; deployment scripts, episode examples, and bundled third-party services are provided in the source release.

## Prepare the physical runtime

1. Install Isaac Sim 5.1 standalone and Conda. Install `tmux`, `rsync`, `git`, `git-lfs`, `ffmpeg`, `util-linux`, and `procps` for managed sessions, process control, and video capture. CUDA extension builds also need a compiler and CUDA toolkit compatible with Torch 2.6 / CUDA 12.6.
2. Obtain the scenes, robot packages, native animations, and checkpoints listed in [assets](../assets/README.md). Preserve relative USD material/model references and the documented asset layout. Keep large assets outside the source checkout and point `TARSPHERE_ASSETS_PATH` to that tree.
3. Configure a vision-capable OpenAI-compatible agent endpoint and a support endpoint for interaction and semantic scoring. Use model IDs supported by the provider. Serve these models remotely for the single-card deployment.
4. Copy the host template to the repository root:

```bash
cp scripts/host_env.example.sh host_env.sh
```

Edit `ISAAC_SIM_PATH`, `TARSPHERE_PYTHON`, `TARSPHERE_GRASPGENX_PYTHON`, `TARSPHERE_ASSETS_PATH`, `TARSPHERE_SAM3_CHECKPOINT`, `TARSPHERE_XNAVDP_CHECKPOINT`, and `TARSPHERE_FRONTIERNET_CHECKPOINT`. Set `TARSPHERE_AGENT_MODEL`, `TARSPHERE_AGENT_API_BASE`, `TARSPHERE_AGENT_API_KEY`, and the `TARSPHERE_EVALUATION_SUPPORT_*` values. The template uses placeholders; `host_env.sh` is local and ignored by Git.

Install the two environments and check the selected episode:

```bash
scripts/setup.sh
source host_env.sh
source scripts/activate.sh
"$TARSPHERE_PYTHON" -m tarsphere.cli.check_environment \
  --episodes examples/episodes/navigation.yaml
```

Setup uses [runtime constraints](../scripts/runtime-constraints.txt) for the simulator stack and [separate requirements](../scripts/graspgenx-requirements.txt) for GraspGenX. It does not download Isaac Sim, scenes, robot/animation assets, or model checkpoints. Resolve missing resources and unavailable generation endpoints before launching.

## Run an episode

```bash
scripts/evaluate.sh --plan
scripts/evaluate.sh
scripts/evaluate.sh --status
scripts/evaluate.sh --stop
```

The default input is [the navigation example](../examples/episodes/navigation.yaml). Supply `--episodes /path/to/episodes` to run your own YAML files or an episode index; see [evaluation](EVALUATION.md). Runs are written to `~/.local/share/tarsphere/runs` by default. Use `--output-dir` to select another directory.

The scheduler defaults to one worker per GPU. On a single GPU, Isaac, SAM3, GraspGenX, FrontierNet, and X-NavDP share that card. Available memory depends on the loaded scene and services; start with the supplied navigation example before adding concurrency.

## Troubleshooting

| Symptom | Check |
| --- | --- |
| Imports work but simulation fails | Standalone SDK, driver/Vulkan, asset roots and checkpoints. |
| GPU out of memory | Running processes, loaded scene and combined service memory. Use one Isaac worker on a 24 GiB card. |
| Scene load is slow | Storage and missing material dependencies; `scripts/stage.sh` supports local staging. |
| Perception fails on first request | SAM3.1 checkpoint path, bundled tokenizer assets and checkpoint/model version. |
| Endpoint responds but execution fails | Vision model ID, credentials and an actual generation request. |
| Scoring is incomplete | Preserve the trajectory and use offline scoring. |
