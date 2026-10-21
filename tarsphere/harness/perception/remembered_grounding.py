"""Re-express agent-established grounding into the current ego camera frame.

Remembered geometry comes only from prior ego RGB, synchronized depth, SAM3,
and robot localisation. It never reads simulator ground truth.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from tarsphere.harness.memory.episodic import GroundingEntry
from tarsphere.geometry import fov_to_intrinsics
from tarsphere.infra.manipulation.agent_grasp import (
    GRASP_REMEMBERED_GROUNDING_MIN_DISK_RADIUS_PX,
)


class RememberedGroundingUnavailableError(RuntimeError):
    """The stored entry cannot be expressed in the current ego view."""


@dataclass(frozen=True)
class RememberedObjectGrounding:
    """One remembered target re-projected into the current synchronized view."""

    label: str
    remembered_action_step: int
    depth_m: np.ndarray
    target_mask: np.ndarray
    camera_intrinsics: np.ndarray
    world_T_camera: np.ndarray
    target_center_camera_m: tuple[float, float, float]
    target_center_world_m: tuple[float, float, float]
    bbox_2d: tuple[float, float, float, float]
    semantic_score: float


def world_point_to_camera_m(
    world_xyz: Sequence[float],
    *,
    world_T_camera: np.ndarray,
) -> np.ndarray:
    """Map one world point into the optical camera frame (right, down, forward)."""

    point = np.asarray(world_xyz, dtype=np.float64).reshape(3)
    transform = np.asarray(world_T_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world_T_camera must be a finite 4x4 transform")
    if not np.all(np.isfinite(point)):
        raise ValueError("world_xyz must contain three finite coordinates")
    camera_T_world = np.linalg.inv(transform)
    homogeneous = np.append(point, 1.0)
    camera = camera_T_world @ homogeneous
    return camera[:3]


def project_camera_point_to_pixel(
    target_center_camera_m: Sequence[float],
    *,
    intrinsics: np.ndarray,
    image_size: tuple[int, int],
) -> tuple[float, float]:
    """Project one optical-frame point to pixel coordinates."""

    center = np.asarray(target_center_camera_m, dtype=np.float64).reshape(3)
    if center[2] <= 0.0:
        raise RememberedGroundingUnavailableError(
            "the remembered target lies behind the current ego camera"
        )
    matrix = np.asarray(intrinsics, dtype=np.float64)
    projected = matrix @ center
    u = float(projected[0] / center[2])
    v = float(projected[1] / center[2])
    height, width = image_size
    if u < 0.0 or u >= width or v < 0.0 or v >= height:
        raise RememberedGroundingUnavailableError(
            "the remembered target projects outside the current ego frame"
        )
    return u, v


def remembered_disk_radius_px(
    entry: GroundingEntry,
    *,
    center_camera_m: Sequence[float],
    intrinsics: np.ndarray,
) -> int:
    """Re-project the remembered bbox extent to a disk radius at the current range.

    The entry's ``bbox_2d`` and ``grounding_depth_m`` record the observed pixel
    size and forward depth from the original ego capture. Pinhole scaling maps
    that footprint to the current forward depth without inventing a segmentation
    mask the agent never saw.
    """

    bbox = entry.bbox_2d
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise RememberedGroundingUnavailableError(
            "remembered grounding requires a stored bbox_2d box_xyxy list"
        )
    z0 = float(entry.grounding_depth_m)
    if not math.isfinite(z0) or z0 <= 0.0:
        raise RememberedGroundingUnavailableError(
            "remembered grounding requires a positive grounding_depth_m from "
            "the original synchronized depth capture"
        )
    center = np.asarray(center_camera_m, dtype=np.float64).reshape(3)
    z1 = float(center[2])
    if not math.isfinite(z1) or z1 <= 0.0:
        raise RememberedGroundingUnavailableError(
            "the remembered target must lie in front of the current ego camera"
        )
    matrix = np.asarray(intrinsics, dtype=np.float64)
    fx = float(matrix[0, 0])
    fy = float(matrix[1, 1])
    if not math.isfinite(fx) or fx <= 0.0 or not math.isfinite(fy) or fy <= 0.0:
        raise RememberedGroundingUnavailableError(
            "remembered grounding requires finite positive camera intrinsics"
        )
    x1, y1, x2, y2 = (float(value) for value in bbox)
    width_px = max(x2 - x1, 0.0)
    height_px = max(y2 - y1, 0.0)
    if width_px <= 0.0 or height_px <= 0.0:
        raise RememberedGroundingUnavailableError(
            "remembered grounding requires a non-degenerate stored bbox_2d"
        )
    half_w_m = 0.5 * (width_px / fx) * z0
    half_h_m = 0.5 * (height_px / fy) * z0
    half_w_px = (half_w_m / z1) * fx
    half_h_px = (half_h_m / z1) * fy
    radius = int(math.ceil(max(half_w_px, half_h_px)))
    return max(radius, int(GRASP_REMEMBERED_GROUNDING_MIN_DISK_RADIUS_PX))


def disk_mask_from_projection(
    *,
    center_uv: tuple[float, float],
    image_size: tuple[int, int],
    radius_px: int,
) -> np.ndarray:
    """Build a boolean disk mask around one projected target centre."""

    height, width = image_size
    radius = int(radius_px)
    if radius <= 0:
        raise ValueError("the remembered-grounding disk radius must be positive")
    u0, v0 = center_uv
    rows, cols = np.ogrid[:height, :width]
    squared = (cols - u0) ** 2 + (rows - v0) ** 2
    return squared <= float(radius * radius)


def grounding_from_remembered_entry(
    *,
    entry: GroundingEntry,
    observation: Any,
) -> RememberedObjectGrounding:
    """Re-express one valid remembered entry into the current ego RGB-D frame."""

    if not entry.geometry_valid:
        raise RememberedGroundingUnavailableError(
            f'grounding geometry for "{entry.label}" was invalidated and must '
            "be reacquired"
        )
    ego = observation.views.get("ego") if observation is not None else None
    if ego is None or ego.rgb is None:
        raise RememberedGroundingUnavailableError(
            "the current decision has no ego RGB frame"
        )
    depth = np.asarray(ego.depth) if ego.depth is not None else None
    if depth is None or depth.ndim != 2:
        raise RememberedGroundingUnavailableError(
            "synchronized metric depth is unavailable for remembered grounding"
        )
    camera = dict(ego.camera_params or {})
    try:
        depth_capture = int(camera["depth_capture_id"])
        extrinsics_capture = int(camera["capture_id"])
    except (KeyError, TypeError, ValueError) as exc:
        raise RememberedGroundingUnavailableError(
            f"the current ego capture ids are incomplete: {exc}"
        ) from exc
    if depth_capture != extrinsics_capture:
        raise RememberedGroundingUnavailableError(
            "the current ego depth and world_T_cam come from different captures"
        )
    try:
        intrinsics = fov_to_intrinsics(
            float(camera["fov_deg"]),
            int(camera.get("width") or depth.shape[1]),
            int(camera.get("height") or depth.shape[0]),
        )
        world_T_camera = np.asarray(camera["world_T_cam"], dtype=np.float64)
    except (KeyError, TypeError, ValueError) as exc:
        raise RememberedGroundingUnavailableError(
            f"the current ego camera calibration is incomplete: {exc}"
        ) from exc
    if world_T_camera.shape != (4, 4) or not np.all(np.isfinite(world_T_camera)):
        raise RememberedGroundingUnavailableError(
            "world_T_cam must be a finite 4x4 transform"
        )

    world_xyz = entry.world_xyz
    center_camera = world_point_to_camera_m(
        world_xyz,
        world_T_camera=world_T_camera,
    )
    center_uv = project_camera_point_to_pixel(
        center_camera,
        intrinsics=intrinsics,
        image_size=depth.shape,
    )
    disk_radius_px = remembered_disk_radius_px(
        entry,
        center_camera_m=center_camera,
        intrinsics=intrinsics,
    )
    mask = disk_mask_from_projection(
        center_uv=center_uv,
        image_size=depth.shape,
        radius_px=disk_radius_px,
    )
    if int(np.count_nonzero(mask)) < 32:
        raise RememberedGroundingUnavailableError(
            "the remembered target disk is too small in the current ego frame"
        )
    radius = float(disk_radius_px)
    bbox = (
        max(0.0, center_uv[0] - radius),
        max(0.0, center_uv[1] - radius),
        min(float(depth.shape[1] - 1), center_uv[0] + radius),
        min(float(depth.shape[0] - 1), center_uv[1] + radius),
    )
    return RememberedObjectGrounding(
        label=str(entry.label),
        remembered_action_step=int(entry.action_step),
        depth_m=np.asarray(depth, dtype=np.float32),
        target_mask=mask,
        camera_intrinsics=intrinsics,
        world_T_camera=world_T_camera,
        target_center_camera_m=tuple(float(value) for value in center_camera),
        target_center_world_m=tuple(float(value) for value in world_xyz),
        bbox_2d=bbox,
        semantic_score=float(entry.score),
    )


__all__ = [
    "RememberedGroundingUnavailableError",
    "RememberedObjectGrounding",
    "disk_mask_from_projection",
    "grounding_from_remembered_entry",
    "project_camera_point_to_pixel",
    "remembered_disk_radius_px",
    "world_point_to_camera_m",
]
