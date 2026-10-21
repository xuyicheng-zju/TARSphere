# GraspGenX source

- Upstream: [NVlabs/GraspGenX](https://github.com/NVlabs/GraspGenX).
- Paper: [GraspGen-X: Cross-Embodiment 6-DOF Diffusion-based Grasping](https://arxiv.org/abs/2606.00998).
- Source: a bundled source snapshot; its upstream commit is not recorded.
- License: [Apache-2.0](LICENSE). Checkpoints have separate terms described in the upstream [README](README.md#license).

TARSphere integrates inference through its own service and manipulation adapters. No separate TARSphere patch is recorded for this source directory. Checkpoints and gripper descriptions are installed separately. The TARSphere service resolves its default `unitree_g1` description from the external `gripper_descriptions` checkout under `ext/`; it does not use GraspGenX's procedural training grippers.

The upstream `assets/proc_grippers` meshes and `assets/sample_data` payloads are Git-LFS resources and are omitted from this source snapshot. Install those resources from their upstream distribution only when running the corresponding upstream demos or procedural-gripper tools. They are not required by TARSphere's default `unitree_g1` service path. Original copyright and dependency notices are retained.
