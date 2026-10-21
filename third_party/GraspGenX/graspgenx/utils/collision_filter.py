# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Ported from grasp_gen/utils/point_cloud_utils.py: filter grasps by
# nearest-neighbor distance from sampled gripper-mesh surface points to a
# scene point cloud.

from __future__ import annotations

import numpy as np
import torch
import trimesh

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


def filter_colliding_grasps(
    scene_pc: np.ndarray,
    grasp_poses: np.ndarray,
    gripper_collision_mesh: trimesh.Trimesh | None = None,
    collision_threshold: float = 0.02,
    num_collision_samples: int = 2000,
    batch_size: int = 16,
    gripper_surface_points: np.ndarray | None = None,
    device: str | torch.device | None = None,
) -> np.ndarray:
    """Return a boolean mask (len = K) where True = collision-free.

    Algorithm:
      1. Uniformly sample ``num_collision_samples`` points (= M) on the
         gripper collision mesh surface (gripper-local frame). Skipped when
         ``gripper_surface_points`` is supplied.
      2. Vectorized over grasps in chunks of ``batch_size`` poses:
         transform the M gripper samples into world frame via the chunk's
         rotations + translations, then run a single ``torch.cdist``
         against the scene point cloud on GPU. The cdist intermediate is
         ``(batch_size * M, N)`` floats — chunk size bounds memory.
      3. A grasp is collision-free iff every one of its M samples'
         nearest-scene-point distance is ≥ ``collision_threshold``.

    Args:
        scene_pc: (N, 3) scene point cloud (target object's points should
                  already be removed by the caller).
        grasp_poses: (K, 4, 4) grasp poses in the same frame as scene_pc.
        gripper_collision_mesh: trimesh.Trimesh of the gripper geometry.
                                Required only if ``gripper_surface_points``
                                is not provided.
        collision_threshold: meters; gripper samples within this distance of
                             any scene point count as a collision.
        num_collision_samples: number of points sampled on the gripper. Used
                               only when ``gripper_surface_points`` is None.
        batch_size: grasps per vectorized cdist call. The per-call distance
                    matrix is ``batch_size * M * N`` fp32 entries — keep
                    ``batch_size * M * N * 4 bytes`` under available GPU
                    memory. Default 16 ≈ 1 GB at M=2000, N=8192.
        gripper_surface_points: optional (M, 3) array of pre-sampled gripper
                                surface points in gripper-local frame. When
                                supplied, the per-call
                                ``trimesh.sample.sample_surface`` is skipped
                                — sample once outside the hot loop and
                                reuse.
        device: torch device for the cdist work. Defaults to "cuda" if
                available, else "cpu". CPU is dramatically slower for
                typical scene sizes — let it auto-pick CUDA when possible.

    Returns:
        (K,) bool ndarray; True == collision-free.
    """
    K = len(grasp_poses)
    if K == 0:
        return np.zeros((0,), dtype=bool)
    if len(scene_pc) == 0:
        logger.info("[collision] scene_pc empty → all grasps marked collision-free.")
        return np.ones((K,), dtype=bool)

    if gripper_surface_points is None:
        if gripper_collision_mesh is None:
            raise ValueError(
                "filter_colliding_grasps: must provide gripper_collision_mesh "
                "or pre-sampled gripper_surface_points"
            )
        sampled, _ = trimesh.sample.sample_surface(
            gripper_collision_mesh, num_collision_samples
        )
        gripper_surface_points = np.asarray(sampled, dtype=np.float32)
    else:
        gripper_surface_points = np.asarray(gripper_surface_points, dtype=np.float32)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device)

    scene_t = torch.as_tensor(scene_pc, dtype=torch.float32, device=device)  # (N, 3)
    pts_local = torch.as_tensor(
        gripper_surface_points, dtype=torch.float32, device=device
    )  # (M, 3)
    poses = torch.as_tensor(
        np.asarray(grasp_poses, dtype=np.float32), dtype=torch.float32, device=device
    )  # (K, 4, 4)

    R = poses[:, :3, :3]  # (K, 3, 3)
    t = poses[:, :3, 3]  # (K, 3)
    M = pts_local.shape[0]

    logger.info(
        f"[collision] checking {K} grasps against {len(scene_pc)} scene points "
        f"(thr={collision_threshold:.3f}m, samples={M}, device={device}, "
        f"chunk={batch_size})"
    )

    collision_free = torch.empty((K,), dtype=torch.bool, device=device)
    for s in range(0, K, batch_size):
        e = min(s + batch_size, K)
        Kc = e - s
        # pts_world[k, m, :] = R[k] @ pts_local[m] + t[k]
        pts_world = torch.einsum("kij,mj->kmi", R[s:e], pts_local) + t[s:e].unsqueeze(1)
        # (Kc*M, N) cdist, then reduce.
        flat = pts_world.reshape(Kc * M, 3)
        d = torch.cdist(flat, scene_t, p=2)
        min_d = d.amin(dim=1).view(Kc, M)
        collision_free[s:e] = ~torch.any(min_d < collision_threshold, dim=1)

    out = collision_free.detach().cpu().numpy()
    logger.info(f"[collision] {int(out.sum())}/{K} grasps collision-free")
    return out
