# MinkowskiEngine source

- Upstream: [NVIDIA/MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine).
- Paper: [4D Spatio-Temporal ConvNets: Minkowski Convolutional Neural Networks](https://arxiv.org/abs/1904.08755).
- Source: a bundled source snapshot; its upstream commit is not recorded.
- License: [MIT](LICENSE).

TARSphere uses MinkowskiEngine for the GraspGenX sparse network. The local build configuration includes the CUDA toolkit's `include/cccl` directory when present. Native extensions are compiled on the target machine rather than distributed as prebuilt binaries. Original copyright and dependency notices are retained.
