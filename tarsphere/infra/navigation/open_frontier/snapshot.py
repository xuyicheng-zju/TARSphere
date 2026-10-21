"""Strict input contract for deterministic OpenFrontier snapshot replay."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from tarsphere.infra.navigation.open_frontier.contracts import CameraIntrinsics


@dataclass(frozen=True, slots=True)
class FrontierObservationSnapshot:
    """One synchronized RGB-D observation and its measured world pose."""

    rgb: np.ndarray
    depth_m: np.ndarray
    intrinsics: CameraIntrinsics
    world_T_camera: np.ndarray
    robot_world_xy: np.ndarray
    observation_step: int = 0

    @classmethod
    def from_npz(cls, path: str | Path) -> "FrontierObservationSnapshot":
        snapshot_path = Path(path).expanduser().resolve()
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"OpenFrontier observation snapshot not found: {snapshot_path}"
            )
        required = {
            "rgb",
            "depth_m",
            "camera_intrinsics",
            "world_T_camera",
            "robot_world_xy",
        }
        with np.load(snapshot_path, allow_pickle=False) as snapshot:
            missing = sorted(required - set(snapshot.files))
            if missing:
                raise ValueError(
                    f"OpenFrontier observation snapshot is missing fields: {missing}"
                )
            rgb = np.asarray(snapshot["rgb"])
            depth = np.asarray(snapshot["depth_m"], dtype=np.float32)
            intrinsic = np.asarray(snapshot["camera_intrinsics"], dtype=np.float64)
            transform = np.asarray(snapshot["world_T_camera"], dtype=np.float64)
            robot_xy = np.asarray(snapshot["robot_world_xy"], dtype=np.float64).reshape(
                -1
            )
            step = (
                int(np.asarray(snapshot["observation_step"]).item())
                if "observation_step" in snapshot.files
                else 0
            )
        if rgb.ndim != 3 or rgb.shape[2] < 3:
            raise ValueError(
                f"observation RGB must have shape HxWx3; got {rgb.shape!r}"
            )
        if depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"observation RGB/depth sizes differ: {rgb.shape[:2]} vs {depth.shape}"
            )
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("world_T_camera must be one finite 4x4 matrix")
        if not np.allclose(transform[3], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
            raise ValueError("world_T_camera must have homogeneous final row [0,0,0,1]")
        if robot_xy.size != 2 or not np.all(np.isfinite(robot_xy)):
            raise ValueError("robot_world_xy must contain two finite values")
        return cls(
            rgb=np.asarray(rgb[:, :, :3], dtype=np.uint8),
            depth_m=depth,
            intrinsics=CameraIntrinsics.from_matrix(
                intrinsic,
                width=int(rgb.shape[1]),
                height=int(rgb.shape[0]),
            ),
            world_T_camera=transform,
            robot_world_xy=robot_xy,
            observation_step=step,
        )


__all__ = ["FrontierObservationSnapshot"]
