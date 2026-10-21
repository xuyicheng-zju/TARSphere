"""Pure-numpy geometry for BEV mapping.

Coordinate conventions
----------------------
World frame: metres, ``X``/``Y`` horizontal, ``Z`` up.  Robot yaw is a rotation
about ``+Z``; ``yaw=0`` points the camera forward along ``+X``.

Camera frame (pinhole, standard OpenCV layout): ``+X`` right, ``+Y`` down,
``+Z`` forward.  Intrinsics are ``fx, fy, cx, cy`` in pixels.

The camera is treated as mounted at ``position`` (the robot base / mount) plus
an optional ``camera_height_m`` above it, and tilted up by an optional
``camera_pitch_deg`` (positive = look up).  Robot ego cameras normally use the
asset-authored orientation, so TARSphere passes ``0.0`` unless a future calibrated
sensor profile explicitly says otherwise.
"""

from __future__ import annotations

import math
from typing import Any, Optional, Sequence

import numpy as np


def intrinsics_from_fov(
    horizontal_fov_deg: float,
    width: int,
    height: int,
) -> tuple[float, float, float, float]:
    """Derive pinhole intrinsics from a horizontal FOV assuming square pixels."""
    fov_rad = math.radians(float(horizontal_fov_deg))
    fx = float(width) / (2.0 * math.tan(fov_rad / 2.0))
    fy = fx
    cx = (float(width) - 1.0) / 2.0
    cy = (float(height) - 1.0) / 2.0
    return fx, fy, cx, cy


def resolve_intrinsics(
    camera_params: Optional[dict[str, Any]],
    image_shape: tuple[int, int],
    default_horizontal_fov_deg: float = 90.0,
) -> tuple[float, float, float, float]:
    """Resolve (fx, fy, cx, cy) from observation camera_params or sane defaults.

    Accepts an explicit ``fx/fy/cx/cy`` dict, an explicit ``fov_deg``, or falls
    back to ``default_horizontal_fov_deg`` over the given image (height, width).
    """
    if camera_params:
        if all(key in camera_params for key in ("fx", "fy", "cx", "cy")):
            return (
                float(camera_params["fx"]),
                float(camera_params["fy"]),
                float(camera_params["cx"]),
                float(camera_params["cy"]),
            )
        fov = camera_params.get("fov_deg") or camera_params.get("horizontal_fov_deg")
        if fov is not None:
            width = int(camera_params.get("width", image_shape[1]))
            height = int(camera_params.get("height", image_shape[0]))
            return intrinsics_from_fov(float(fov), width, height)

    height, width = int(image_shape[0]), int(image_shape[1])
    return intrinsics_from_fov(default_horizontal_fov_deg, width, height)


def depth_to_camera_points(
    depth: np.ndarray,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
    *,
    max_depth_m: Optional[float] = None,
) -> np.ndarray:
    """Back-project a depth image to camera-frame points (N, 3).

    Returns ``x`` right, ``y`` down, ``z`` forward in the camera frame.  Invalid
    (non-finite or non-positive) pixels and anything beyond ``max_depth_m`` are
    dropped.
    """
    depth_array = np.asarray(depth, dtype=float)
    if depth_array.ndim != 2:
        raise ValueError(f"depth must be 2D (H, W); got shape {depth_array.shape}")

    valid = np.isfinite(depth_array) & (depth_array > 0)
    if max_depth_m is not None:
        valid &= depth_array <= float(max_depth_m)
    rows, cols = np.nonzero(valid)
    if rows.size == 0:
        return np.zeros((0, 3), dtype=float)

    z = depth_array[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    return np.stack([x, y, z], axis=1)


def camera_to_world(
    points_cam: np.ndarray,
    position: Sequence[float],
    yaw_rad: float,
    *,
    camera_height_m: float = 0.0,
    camera_pitch_deg: float = 0.0,
) -> np.ndarray:
    """Transform camera-frame points to the world frame (N, 3).

    See module docstring for the axis convention.  ``position`` is the camera
    mount ``[x, y, z]``; ``camera_height_m`` lifts the optical centre in ``Z``.
    """
    pts = np.asarray(points_cam, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points_cam must be (N, 3); got shape {pts.shape}")
    if pts.shape[0] == 0:
        return np.zeros((0, 3), dtype=float)

    pitch = math.radians(float(camera_pitch_deg))
    cos_p, sin_p = math.cos(pitch), math.sin(pitch)
    x = pts[:, 0]
    y = pts[:, 1]
    z = pts[:, 2]
    # Rotate about the camera right axis (+X): tilt the forward axis up/down.
    y_pitched = y * cos_p - z * sin_p
    z_pitched = y * sin_p + z * cos_p

    yaw = float(yaw_rad)
    cos_y, sin_y = math.cos(yaw), math.sin(yaw)
    # Camera +Z (forward) -> world heading (cos yaw, sin yaw, 0);
    # camera +X (right)   -> world (-sin yaw, cos yaw, 0);
    # camera +Y (down)    -> world (0, 0, -1).
    world_x = -x * sin_y + z_pitched * cos_y
    world_y = x * cos_y + z_pitched * sin_y
    world_z = -y_pitched

    base = np.asarray(position, dtype=float).reshape(3)
    world_x += base[0]
    world_y += base[1]
    world_z += base[2] + float(camera_height_m)
    return np.stack([world_x, world_y, world_z], axis=1)
