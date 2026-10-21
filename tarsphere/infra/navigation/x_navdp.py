"""Thin adapter around the official X-NavDP PointGoal client.

X-NavDP is a local policy, not a global planner.  The evaluated MLLM owns the
choice of a body-relative PointGoal.  This adapter sends that goal plus the
current synchronized ego RGB-D frame to the official server and returns one
bounded signed body-frame waypoint.  It never consumes occupancy, episode
routes, task targets, or evaluator state.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import os
from pathlib import Path
import sys
import time
from collections.abc import Mapping
from typing import Any, Sequence

import numpy as np

from tarsphere.infra.control.physical_contracts import (
    LOCOMOTION_MAX_TURN_DEG,
    NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD,
)


@dataclass(frozen=True)
class XNavDPSettings:
    """Deployment and local-prefix settings for the official server."""

    root: str
    port: int = 19999
    prefix_distance_m: float = 0.50
    max_command_m: float = 0.55

    def __post_init__(self) -> None:
        root = Path(self.root).expanduser()
        client = root / "eval/src/client_utils.py"
        if not client.is_file():
            raise FileNotFoundError(f"Official X-NavDP client missing: {client}")
        if int(self.port) <= 0:
            raise ValueError("X-NavDP port must be positive.")
        if float(self.prefix_distance_m) <= 0.0:
            raise ValueError("X-NavDP prefix_distance_m must be positive.")
        if float(self.max_command_m) <= 0.0:
            raise ValueError("X-NavDP max_command_m must be positive.")

    @classmethod
    def from_environment(cls) -> "XNavDPSettings":
        root = (os.environ.get("TARSPHERE_XNAVDP_ROOT") or "").strip()
        if not root:
            raise RuntimeError(
                "The configured X-NavDP navigation backend requires "
                "TARSPHERE_XNAVDP_ROOT to name the official checkout."
            )
        return cls(
            root=root,
            port=int(os.environ.get("TARSPHERE_XNAVDP_PORT", "19999")),
            prefix_distance_m=float(
                os.environ.get("TARSPHERE_XNAVDP_PREFIX_M", "0.50")
            ),
            max_command_m=float(
                os.environ.get("TARSPHERE_XNAVDP_MAX_COMMAND_M", "0.55")
            ),
        )


@dataclass(frozen=True)
class XNavDPResult:
    """One selected local trajectory and its bounded executable prefix."""

    command_body_xy: tuple[float, float]
    selected_trajectory_body_xy: tuple[tuple[float, float], ...]
    prefix_index: int
    inference_seconds: float
    model_algorithm: str


class XNavDPNoMotionError(RuntimeError):
    """The predicted trajectory contains no executable local displacement."""


def prefix_body_delta(command_body_xy: Sequence[float]) -> list[float]:
    """Face a predicted waypoint before translating, using the A* entry rule."""

    x, y = (float(value) for value in command_body_xy)
    if not all(math.isfinite(value) for value in (x, y)):
        raise ValueError("X-NavDP motion must be finite.")
    heading = math.atan2(y, x)
    if abs(heading) > NAVIGATION_PATH_ENTRY_YAW_TOLERANCE_RAD:
        return [
            0.0,
            0.0,
            float(
                np.clip(
                    math.degrees(heading),
                    -LOCOMOTION_MAX_TURN_DEG,
                    LOCOMOTION_MAX_TURN_DEG,
                )
            ),
        ]
    return [x, y, 0.0]


def selected_trajectory_xy(raw: Any) -> np.ndarray:
    """Normalize the official selected-trajectory response to finite Nx2."""

    values = np.asarray(raw, dtype=np.float64)
    while values.ndim > 2 and values.shape[0] == 1:
        values = values[0]
    if values.ndim == 3:
        values = values[0]
    if values.ndim != 2 or values.shape[0] < 1 or values.shape[1] < 2:
        raise RuntimeError(
            f"X-NavDP returned invalid trajectory shape {values.shape!r}."
        )
    values = values[:, :2]
    if not np.all(np.isfinite(values)):
        raise RuntimeError("X-NavDP returned a non-finite trajectory.")
    return values


def select_prefix_waypoint(
    trajectory_xy: np.ndarray,
    *,
    prefix_distance_m: float,
    max_command_m: float,
) -> tuple[int, np.ndarray]:
    """Select and bound one signed waypoint without changing its direction."""

    points = np.asarray(trajectory_xy, dtype=np.float64)
    distances = np.linalg.norm(points, axis=1)
    candidates = np.flatnonzero(distances >= float(prefix_distance_m))
    index = int(candidates[0]) if candidates.size else int(len(points) - 1)
    waypoint = points[index].copy()
    distance = float(np.linalg.norm(waypoint))
    if distance < 0.08:
        raise XNavDPNoMotionError(
            "X-NavDP selected trajectory has no executable waypoint beyond 0.08 m."
        )
    if distance > float(max_command_m):
        waypoint *= float(max_command_m) / distance
    return index, waypoint


class XNavDPClient:
    """Episode-scoped official X-NavDP client with lazy camera reset."""

    def __init__(self, settings: XNavDPSettings) -> None:
        self.settings = settings
        self._scene_name = "tarsphere_episode"
        self._reset_required = True
        self._model_algorithm = ""

    def reset(self) -> None:
        """Clear only local client state; server reset occurs on next frame."""

        self._reset_required = True
        self._model_algorithm = ""

    def configure_episode(self, episode: Any) -> None:
        metadata = getattr(episode, "metadata", None)
        if isinstance(metadata, Mapping):
            scene_id = str(metadata.get("scene_id") or "").strip()
            episode_id = str(metadata.get("episode_id") or "").strip()
        else:
            scene_id = str(getattr(metadata, "scene_id", "") or "").strip()
            episode_id = str(getattr(metadata, "episode_id", "") or "").strip()
        self._scene_name = (
            ":".join(part for part in (scene_id, episode_id) if part)
            or "tarsphere_episode"
        )
        self.reset()

    def infer(
        self,
        *,
        observation: Any,
        pointgoal_body_xy: Sequence[float],
    ) -> XNavDPResult:
        """Infer one local signed waypoint from current RGB-D and PointGoal."""

        ego = getattr(observation, "views", {}).get("ego")
        pose = getattr(observation, "pose", None)
        if ego is None or ego.rgb is None or ego.depth is None:
            raise RuntimeError(
                "X-NavDP requires synchronized current ego RGB and metric depth."
            )
        if not isinstance(pose, dict) or pose.get("position") is None:
            raise RuntimeError("X-NavDP requires the current robot pose.")
        rgb = np.asarray(ego.rgb, dtype=np.uint8)
        depth = np.asarray(ego.depth, dtype=np.float32)
        if depth.ndim == 3 and depth.shape[-1] == 1:
            depth = depth[:, :, 0]
        if rgb.ndim != 3 or rgb.shape[2] < 3 or depth.ndim != 2:
            raise RuntimeError(
                f"Invalid X-NavDP RGB-D shapes: rgb={rgb.shape}, depth={depth.shape}."
            )
        if rgb.shape[:2] != depth.shape:
            raise RuntimeError("X-NavDP RGB and depth resolutions do not match.")
        position = np.asarray(pose["position"], dtype=np.float32).reshape(-1)
        if position.size < 3 or not np.all(np.isfinite(position[:3])):
            raise RuntimeError("X-NavDP robot position is invalid.")
        yaw = float(pose.get("yaw_rad", 0.0))
        if not math.isfinite(yaw):
            raise RuntimeError("X-NavDP robot yaw is invalid.")

        pointgoal = np.asarray(pointgoal_body_xy, dtype=np.float32).reshape(-1)
        if pointgoal.size != 2 or not np.all(np.isfinite(pointgoal)):
            raise RuntimeError("X-NavDP PointGoal must be two finite metres.")
        pointgoal_step, navigator_reset = self._official_calls()
        if self._reset_required:
            height, width = depth.shape
            params = ego.camera_params if isinstance(ego.camera_params, dict) else {}
            fov_deg = float(params.get("fov_deg", 100.0))
            focal = (width / 2.0) / math.tan(math.radians(fov_deg / 2.0))
            intrinsic = np.asarray(
                [
                    [focal, 0.0, width / 2.0],
                    [0.0, focal, height / 2.0],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            self._model_algorithm = str(
                navigator_reset(
                    intrinsic=intrinsic,
                    batch_size=1,
                    port=int(self.settings.port),
                    scene_name=self._scene_name,
                )
            )
            self._reset_required = False

        quaternion_xyzw = np.asarray(
            [
                [
                    0.0,
                    0.0,
                    math.sin(yaw / 2.0),
                    math.cos(yaw / 2.0),
                ]
            ],
            dtype=np.float32,
        )
        started = time.monotonic()
        selected, _all_trajectories, _values = pointgoal_step(
            pointgoal[None, :],
            rgb[None, :, :, :3],
            depth[None, :, :, None],
            port=int(self.settings.port),
            robot_pos=position[:3][None, :],
            robot_quat=quaternion_xyzw,
        )
        elapsed = time.monotonic() - started
        trajectory = selected_trajectory_xy(selected)
        prefix_index, command = select_prefix_waypoint(
            trajectory,
            prefix_distance_m=self.settings.prefix_distance_m,
            max_command_m=self.settings.max_command_m,
        )
        return XNavDPResult(
            command_body_xy=(float(command[0]), float(command[1])),
            selected_trajectory_body_xy=tuple(
                (float(point[0]), float(point[1])) for point in trajectory
            ),
            prefix_index=prefix_index,
            inference_seconds=float(elapsed),
            model_algorithm=self._model_algorithm,
        )

    def _official_calls(self) -> tuple[Any, Any]:
        root = str(Path(self.settings.root).expanduser().resolve())
        if root not in sys.path:
            sys.path.insert(0, root)
        from eval.src.client_utils import navigator_reset, pointgoal_step

        return pointgoal_step, navigator_reset


__all__ = [
    "XNavDPClient",
    "XNavDPResult",
    "XNavDPSettings",
    "select_prefix_waypoint",
    "selected_trajectory_xy",
]
