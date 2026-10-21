# X-NavDP source

- Upstream: [InternRobotics/NavDP](https://github.com/InternRobotics/NavDP), specifically [baselines/x-navdp](https://github.com/InternRobotics/NavDP/tree/main/baselines/x-navdp).
- Paper: [X-NavDP: Generalizing Navigation Diffusion Policy to Novel Behavior and Embodiments with Group Q-score Reweighted Matching](https://arxiv.org/abs/2607.28560).
- Source: only the X-NavDP baseline subtree is bundled; its upstream commit is not recorded.
- License: [MIT](baselines/x-navdp/LICENSE). See its [README](baselines/x-navdp/README.md) for nested dependency notices.

The local [visualization patch](../patches/0003-x-navdp-honor-no-visualization.patch) makes policy inference honor disabled debug visualization. It is already applied. TARSphere's navigation service and simulation adapters live outside this directory. Model weights, robot assets, and scenes are installed separately. Original copyright and citation notices are retained.

Nested source under `baselines/x-navdp/third_party/depth_anything` incorporates [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2) ([paper](https://arxiv.org/abs/2406.09414)) and [DINOv2](https://github.com/facebookresearch/dinov2) ([paper](https://arxiv.org/abs/2304.07193)). Their source notices and [Apache-2.0 license text](baselines/x-navdp/third_party/LICENSES/Apache-2.0.txt) are retained; their model weights are subject to the respective upstream terms. No separate upstream revision or TARSphere patch is recorded for these nested copies.
