# Bundled source

These source snapshots are included directly in TARSphere, with no Git submodules or additional source checkout required. We thank the authors and maintainers of the following projects. Model weights and simulation assets are installed separately and retain their own terms.

| Bundled component | Upstream project | Associated paper | Local license and source notes |
| --- | --- | --- | --- |
| GraspGenX | [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX) | [GraspGen-X: Cross-Embodiment 6-DOF Diffusion-based Grasping](https://arxiv.org/abs/2606.00998) | [Apache-2.0](GraspGenX/LICENSE) · [source notes](GraspGenX/TARSPHERE_SOURCE.md) |
| Hy-Embodied-VLM-1.0 | [Tencent-Hunyuan/HY-Embodied](https://github.com/Tencent-Hunyuan/HY-Embodied) | [Hy-Embodied-VLM-1.0: Efficient Physical-World Agents](https://arxiv.org/abs/2607.12894) | [Apache-2.0](HY-Embodied/LICENSE) · [source notes](HY-Embodied/TARSPHERE_SOURCE.md) |
| InternNav / InternVLA-N1 | [InternRobotics/InternNav](https://github.com/InternRobotics/InternNav) | [Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation](https://openreview.net/forum?id=GK4rznYwhn) | [MIT](InternNav/LICENSE) · [source notes](InternNav/TARSPHERE_SOURCE.md) |
| Long-CLIP, nested in InternNav | [beichenzbc/Long-CLIP](https://github.com/beichenzbc/Long-CLIP) | [Long-CLIP: Unlocking the Long-Text Capability of CLIP](https://arxiv.org/abs/2403.15378) | [Apache-2.0](InternNav/internnav/model/basemodel/LongCLIP/LICENSE) · [source notes](InternNav/internnav/model/basemodel/LongCLIP/TARSPHERE_SOURCE.md) |
| MinkowskiEngine | [NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) | [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755) | [MIT](MinkowskiEngine/LICENSE) · [source notes](MinkowskiEngine/TARSPHERE_SOURCE.md) |
| X-NavDP, a NavDP subset | [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP/tree/main/baselines/x-navdp) | [X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching](https://arxiv.org/abs/2607.28560) | [MIT](NavDP/baselines/x-navdp/LICENSE) · [source notes](NavDP/TARSPHERE_SOURCE.md) |
| cuRobo | [NVlabs/curobo](https://github.com/NVlabs/curobo) | [cuRoboV2: Dynamics-Aware Motion Generation with Depth-Fused Distance Fields for High-DoF Robots](https://arxiv.org/abs/2603.05493) | [Apache-2.0](curobo/LICENSE) · [asset terms](curobo/LICENSE_ASSETS) · [source notes](curobo/TARSPHERE_SOURCE.md) |
| SAM3 | [facebookresearch/sam3](https://github.com/facebookresearch/sam3) | [SAM 3: Segment Anything with Concepts](https://arxiv.org/abs/2511.16719) | [SAM License](sam3/LICENSE) · [source notes](sam3/TARSPHERE_SOURCE.md) |

Long-CLIP records upstream revision `3966af9ae9331666309a22128468b734db4672a7`. The other bundled snapshots do not record an upstream commit; their directory names and upstream release announcements do not establish a revision pin. Original per-file notices and nested dependency licenses remain applicable. See each upstream README for its full citation and dependency acknowledgments.

Nested source includes [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) ([paper](https://arxiv.org/abs/2303.04137), [local MIT license](InternNav/internnav/model/encoder/diffusion_policy/LICENSE), [source notes](InternNav/internnav/model/encoder/diffusion_policy/TARSPHERE_SOURCE.md)), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) ([paper](https://arxiv.org/abs/2406.09414)), and [DINOv2](https://github.com/facebookresearch/dinov2) ([paper](https://arxiv.org/abs/2304.07193)). The latter two are incorporated in X-NavDP with a local [Apache-2.0 license copy](NavDP/baselines/x-navdp/third_party/LICENSES/Apache-2.0.txt) and retained per-file notices. Their model weights have separate upstream terms.

## Local adaptations

The patches in `patches/` describe adaptations already included in the source. Do not apply them again:

- [SAM3 checkpoint loading on the configured device and detector checkpoint validation](patches/0001-sam3-load-checkpoint-on-target-device.patch).
- [InternNav G1 service adapter and inference error propagation](patches/0002-internnav-g1-agent-server-adapter.patch).
- [X-NavDP operation with visualization disabled](patches/0003-x-navdp-honor-no-visualization.patch).

The Long-CLIP model implementation and tokenizer data used by InternNav are also bundled locally; model weights remain separate.

TARSphere-Infra implements physical execution, NPC interaction, dialogue, and task integration on NVIDIA Isaac Sim. InternNav's navigation adapter connects to the local `tarsphere.simulator` interface through `internnav.env.utils.tarsphere_extension`. Simulator interfaces and robot extensions live in `tarsphere/simulator`; embodiment profiles and controller integration live in `tarsphere/embodiment`.

Run `python -m tarsphere.cli.runtime` to check the bundled runtime source. GPU and asset readiness are checked separately by `python -m tarsphere.cli.check_environment`; follow the root [installation instructions](../README.md).
