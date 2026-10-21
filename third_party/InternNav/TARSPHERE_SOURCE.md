# InternNav source

- Upstream: [InternRobotics/InternNav](https://github.com/InternRobotics/InternNav).
- Paper: [Ground Slow, Move Fast: A Dual-System Foundation Model for Generalizable Vision-Language Navigation](https://openreview.net/forum?id=GK4rznYwhn).
- Source: a bundled source snapshot; its upstream commit is not recorded.
- License: [MIT](LICENSE), with additional notices for nested dependencies.

Local adaptations replace the simulator-facing imports and environment configuration with `tarsphere.simulator`, `TARSphereEnv`, and `internnav.env.utils.tarsphere_extension`. The [G1 service adapter patch](../patches/0002-internnav-g1-agent-server-adapter.patch) records inference error propagation and camera action integration; these changes are already applied.

The nested [Long-CLIP inference source](internnav/model/basemodel/LongCLIP/TARSPHERE_SOURCE.md) and [Diffusion Policy source](internnav/model/encoder/diffusion_policy/TARSPHERE_SOURCE.md) have their own source and license notes. Model weights and simulation assets are installed separately. Original copyright, citation, and dependency notices are retained.

Archived captured observations and real-world sample archives are omitted from the source distribution. The competition SDK test accepts user-provided RGB/depth metadata via `--observation-meta` and an `--instruction`.
