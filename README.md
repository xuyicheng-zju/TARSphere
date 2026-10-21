<div align="center">

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="docs/assets/tarsphere-header-dark.svg">
  <img src="docs/assets/tarsphere-header.svg" alt="TARSphere: A Robot-Oriented Full-Stack Framework for Long-Horizon Embodied Agents in Evolving Physical and Social Environments" width="100%">
</picture>

<p>
<a href="https://github.com/xuyicheng-zju">Yicheng&nbsp;Xu</a><sup>1</sup> &nbsp;·&nbsp; <a href="https://github.com/ZhikaiXu24">Zhikai&nbsp;Xu</a><sup>1</sup> &nbsp;·&nbsp; <a href="https://openreview.net/profile?id=~Ziyuan_Chen9">Ziyuan&nbsp;Chen</a><sup>2</sup> &nbsp;·&nbsp; <a href="https://openreview.net/profile?id=~Yuhang_Dong4">Yuhang&nbsp;Dong</a><sup>1</sup> &nbsp;·&nbsp; <a href="https://openreview.net/profile?id=~Tao_Zhang100">Tao&nbsp;Zhang</a><sup>1</sup> &nbsp;·&nbsp; <a href="https://github.com/xzc-zju">Zhucun&nbsp;Xue</a><sup>1</sup> &nbsp;·&nbsp; <a href="https://openreview.net/profile?id=~Yong_Liu11">Yong&nbsp;Liu</a><sup>1</sup> &nbsp;·&nbsp; <a href="https://github.com/zhangzjn">Jiangning&nbsp;Zhang</a><sup>1</sup>
</p>

<sup>1</sup> Zhejiang University &nbsp;&nbsp; <sup>2</sup> Fudan University

<a href="https://github.com/xuyicheng-zju/TARSphere"><img src="https://img.shields.io/badge/Code-181717?style=for-the-badge&logo=github&logoColor=white" alt="Code"></a>
<a href="https://xuyicheng-zju.github.io/TARSphere/"><img src="https://img.shields.io/badge/Project-76549C?style=for-the-badge&logo=googlechrome&logoColor=white" alt="Project"></a>

<a href="#quick-start"><img src="docs/assets/icons/rocket.svg" width="16" height="16" alt=""> Quick Start</a>&nbsp;&nbsp; · &nbsp;&nbsp;<a href="#documentation"><img src="docs/assets/icons/book.svg" width="16" height="16" alt=""> Documentation</a>&nbsp;&nbsp; · &nbsp;&nbsp;<a href="README.zh-CN.md">中文</a>

<br>

<a href="docs/assets/teaser.png"><img src="docs/assets/teaser.png" alt="TARSphere: socially grounded interaction, policy-driven grasping, evolving tasks, and physical execution" width="100%"></a>

<sub>Socially grounded interaction · Policy-driven grasping · Evolving tasks · Physical execution</sub>

<img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python 3.11">
<img src="https://img.shields.io/badge/Isaac_Sim-5.1-76B900?logo=nvidia&logoColor=white" alt="Isaac Sim 5.1">
<a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-76549C" alt="Apache-2.0"></a>

</div>

---

<h2 id="overview"><img src="docs/assets/icons/stack-2.svg" width="24" height="24" alt=""> Meet TARSphere</h2>

Connect multimodal reasoning to physical robot execution, embodied human interaction, and changing task requirements.

| Component | Capabilities |
| --- | --- |
| **TARSphere-Infra** | Robot execution, embodied NPCs, dialogue, and event-driven task transitions. |
| **TARSphere-Harness** | Observations, hierarchical memory, executable skills, and feedback for a frozen MLLM. |
| **TARSphere-Bench** | Task authoring, paired condition examples, and causal evaluation for navigation and delivery. |

<h2 id="todo"><img src="docs/assets/icons/list-check.svg" width="24" height="24" alt=""> TODO</h2>

<p>
<img src="docs/assets/icons/checkbox-green.svg" width="18" height="18" alt="[x]"> Infra Harness<br>
<img src="docs/assets/icons/checkbox-green.svg" width="18" height="18" alt="[x]"> Data Curation<br>
<img src="docs/assets/icons/checkbox-green.svg" width="18" height="18" alt="[x]"> Benchmark Evaluation<br>
<img src="docs/assets/icons/checkbox-empty.svg" width="18" height="18" alt="[ ]"> Real-Robot
</p>

<h2 id="quick-start"><img src="docs/assets/icons/rocket.svg" width="24" height="24" alt=""> Quick Start</h2>

**Single RTX 4090 · Ubuntu 22.04 / 24.04 · Isaac Sim 5.1 · Python 3.11**

The default deployment uses one GPU and one Isaac worker. Agent and support language models run through remote OpenAI-compatible APIs; perception, navigation, and grasp services run locally.

**1. Prepare.** Install Isaac Sim and the [assets and checkpoints](assets/README.md), then fill the host template with your paths and model endpoints. Follow the [installation guide](docs/INSTALLATION.md) for Conda, system dependencies, and resource layout.

```bash
cp scripts/host_env.example.sh host_env.sh
# Fill local paths and agent/support API credentials in host_env.sh.
```

**2. Install and launch.** The default input is [a navigation example](examples/episodes/navigation.yaml).

```bash
scripts/setup.sh
scripts/evaluate.sh --plan
scripts/evaluate.sh
```

**3. Manage the run or supply your own tasks.**

```bash
scripts/evaluate.sh --status
scripts/evaluate.sh --stop
scripts/evaluate.sh --episodes /path/to/episodes
```

`--episodes` accepts YAML, a directory, or a JSON path index. `--plan` previews the launch configuration. Runtime files go to `~/.local/share/tarsphere/runs`; use `--output-dir` to choose another directory. Public execution uses the default mechanisms; ordinary history, action, camera, and mapping parameters remain configurable.

<h2 id="documentation"><img src="docs/assets/icons/book.svg" width="24" height="24" alt=""> Documentation</h2>

| Guide | Start here for |
| --- | --- |
| <img src="docs/assets/icons/settings.svg" width="18" height="18" alt=""> [Installation](docs/INSTALLATION.md) | Environment setup, single-GPU deployment, and troubleshooting. |
| <img src="docs/assets/icons/download.svg" width="18" height="18" alt=""> [Assets & checkpoints](assets/README.md) | Upstream downloads and the required directory layout. |
| <img src="docs/assets/icons/list-check.svg" width="18" height="18" alt=""> [Evaluation](docs/EVALUATION.md) | Task inputs, execution, and scoring saved trajectories. |
| <img src="docs/assets/icons/file-text.svg" width="18" height="18" alt=""> [Task authoring](docs/BENCHMARK_AUTHORING.md) | Episode schemas, conditions, and physical validation. |
| <img src="docs/assets/icons/device-gamepad.svg" width="18" height="18" alt=""> [Teleoperation](docs/TELEOPERATION.md) | Interactive robot and animated NPC control. |

<h2 id="tasks"><img src="docs/assets/icons/route.svg" width="24" height="24" alt=""> Tasks & Conditions</h2>

**Navigate-and-report** and **fetch-and-deliver** share a five-stage causal evaluation. The [source examples](episodes/README.md) cover six task conditions:

| | | |
| --- | --- | --- |
| **Base**<br>Default task | **PR** · Perceptual robustness<br>Changes in sensing | **DA** · Disruption adaptation<br>Changes in execution availability |
| **TU** · Task update<br>Revised requirements | **HC** · Human coordination<br>Onsite collaboration | **IU** · Intent understanding<br>Clarification and contextual intent |

Evaluation includes Success, Progress, path efficiency, safety, and inference cost. Provide your task collection through `--episodes`.

<h2 id="repository"><img src="docs/assets/icons/folders.svg" width="24" height="24" alt=""> Repository</h2>

```text
tarsphere/
  simulator/       Simulation core and extensions
  embodiment/      Robot profiles and controllers
  infra/           Physical execution and simulator integration
  harness/         Model interface, memory, and skills
  bench/           Episode schemas and task authoring
  evaluation/      Task-state evaluation and scoring
scripts/           Setup, execution, scoring, and rendering
episodes/          Task and condition examples
examples/          Small runnable examples
third_party/       Bundled upstream source and notices
```

Assets and checkpoints are installed separately. Keep credentials and generated runtime files outside the distributed source.

<h2 id="acknowledgments"><img src="docs/assets/icons/heart-handshake.svg" width="24" height="24" alt=""> Acknowledgments</h2>

We thank the authors and maintainers of the works that support TARSphere:

- [GRScenes](https://huggingface.co/datasets/InternRobotics/GRScenes) for indoor scene assets.
- [OpenFrontier](https://github.com/cvg/OpenFrontier) for the visual frontier exploration approach, [FrontierNet](https://github.com/cvg/FrontierNet) for frontier prediction, and [NavDP / X-NavDP](https://github.com/InternRobotics/NavDP) and [InternNav](https://github.com/InternRobotics/InternNav) for navigation components.
- [SAM3](https://github.com/facebookresearch/sam3), [GraspGenX](https://github.com/NVlabs/GraspGenX), and [cuRobo](https://github.com/NVlabs/curobo) for perception, grasp generation, and motion planning.
- [NVIDIA Isaac Sim](https://docs.isaacsim.omniverse.nvidia.com/5.1.0/index.html) for the underlying simulation engine.

Component-specific upstream links, citations, licenses, and local adaptations are documented in [third_party](third_party/README.md). Scene packages, robot assets, Isaac Sim, native character animations, and model weights retain their own terms and are installed separately; see [assets](assets/README.md).

<h2 id="license"><img src="docs/assets/icons/scale.svg" width="24" height="24" alt=""> License</h2>

Copyright 2026 TARSphere contributors. Original project code and documentation are licensed under [Apache-2.0](LICENSE).
