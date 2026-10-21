"""Typed contracts for visual-frontier detection and tracking.

Coordinates are explicit throughout this package:

* image points are ``(x, y)`` pixels in the FrontierNet crop;
* camera points use the optical convention ``+X right, +Y down, +Z forward``;
* world points are Isaac metres in the supplied ``world_T_camera`` frame;
* navigation consumes only world ``(x, y)`` and an online occupancy grid.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import math
from typing import Any, Mapping

import numpy as np


def _finite_tuple(values: Any, size: int, *, field_name: str) -> tuple[float, ...]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size != size or not np.all(np.isfinite(array)):
        raise ValueError(f"{field_name} must contain {size} finite values.")
    return tuple(float(value) for value in array)


@dataclass(frozen=True, slots=True)
class CameraIntrinsics:
    """Pinhole intrinsics for one aligned RGB-D image."""

    width: int
    height: int
    fx: float
    fy: float
    cx: float
    cy: float

    def __post_init__(self) -> None:
        if int(self.width) <= 1 or int(self.height) <= 1:
            raise ValueError("Camera width and height must exceed one pixel.")
        values = (self.fx, self.fy, self.cx, self.cy)
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Camera intrinsics must be finite.")
        if float(self.fx) <= 0.0 or float(self.fy) <= 0.0:
            raise ValueError("Camera focal lengths must be positive.")

    @classmethod
    def from_matrix(
        cls,
        matrix: Any,
        *,
        width: int,
        height: int,
    ) -> "CameraIntrinsics":
        intrinsic = np.asarray(matrix, dtype=np.float64)
        if intrinsic.shape != (3, 3) or not np.all(np.isfinite(intrinsic)):
            raise ValueError("Camera intrinsic matrix must be finite 3x3.")
        return cls(
            width=int(width),
            height=int(height),
            fx=float(intrinsic[0, 0]),
            fy=float(intrinsic[1, 1]),
            cx=float(intrinsic[0, 2]),
            cy=float(intrinsic[1, 2]),
        )

    def matrix(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )


@dataclass(frozen=True, slots=True)
class FrontierNetPrediction:
    """Dense outputs aligned with the preprocessed RGB-D crop."""

    normalized_distance_field: np.ndarray
    distance_field_px: np.ndarray
    class_indices: np.ndarray
    frontier_mask: np.ndarray
    information_gain: np.ndarray

    def __post_init__(self) -> None:
        arrays = (
            np.asarray(self.normalized_distance_field),
            np.asarray(self.distance_field_px),
            np.asarray(self.class_indices),
            np.asarray(self.frontier_mask),
            np.asarray(self.information_gain),
        )
        shape = arrays[0].shape
        if len(shape) != 2 or any(array.shape != shape for array in arrays[1:]):
            raise ValueError("Every FrontierNet output must be one aligned 2-D array.")
        if not np.all(np.isfinite(arrays[0])):
            raise ValueError(
                "FrontierNet distance prediction contains non-finite values."
            )
        if not np.all(np.isfinite(arrays[1])):
            raise ValueError("FrontierNet decoded distance contains non-finite values.")
        if not np.all(np.isfinite(arrays[4])):
            raise ValueError("FrontierNet information gain contains non-finite values.")


@dataclass(frozen=True, slots=True)
class ObservedFrontier:
    """One FrontierNet region anchored into the current world frame."""

    observation_index: int
    pixel_xy: tuple[float, float]
    world_xyz: tuple[float, float, float]
    view_direction_world: tuple[float, float, float]
    information_gain_m3: float
    support_pixel_count: int

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "pixel_xy",
            _finite_tuple(self.pixel_xy, 2, field_name="pixel_xy"),
        )
        object.__setattr__(
            self,
            "world_xyz",
            _finite_tuple(self.world_xyz, 3, field_name="world_xyz"),
        )
        direction = np.asarray(
            _finite_tuple(
                self.view_direction_world,
                3,
                field_name="view_direction_world",
            ),
            dtype=np.float64,
        )
        norm = float(np.linalg.norm(direction))
        if norm <= 1e-8:
            raise ValueError("view_direction_world must be non-zero.")
        object.__setattr__(
            self,
            "view_direction_world",
            tuple(float(value) for value in direction / norm),
        )
        if not math.isfinite(float(self.information_gain_m3)):
            raise ValueError("information_gain_m3 must be finite.")
        if float(self.information_gain_m3) < 0.0:
            raise ValueError("information_gain_m3 cannot be negative.")
        if int(self.support_pixel_count) <= 0:
            raise ValueError("support_pixel_count must be positive.")


class FrontierState(str, Enum):
    ACTIVE = "active"
    VISITED = "visited"


@dataclass(slots=True)
class TrackedFrontier:
    """A stable frontier landmark accumulated across observations."""

    frontier_id: str
    world_xyz: tuple[float, float, float]
    view_direction_world: tuple[float, float, float]
    base_information_gain_m3: float
    updated_information_gain_m3: float
    first_observation_step: int
    last_observation_step: int
    observation_count: int
    parent_pose_ids: list[str] = field(default_factory=list)
    state: FrontierState = FrontierState.ACTIVE

    def as_dict(self) -> dict[str, Any]:
        return {
            "frontier_id": self.frontier_id,
            "world_xyz": list(self.world_xyz),
            "view_direction_world": list(self.view_direction_world),
            "base_information_gain_m3": self.base_information_gain_m3,
            "updated_information_gain_m3": self.updated_information_gain_m3,
            "first_observation_step": self.first_observation_step,
            "last_observation_step": self.last_observation_step,
            "observation_count": self.observation_count,
            "parent_pose_ids": list(self.parent_pose_ids),
            "state": self.state.value,
        }


@dataclass(frozen=True, slots=True)
class DisplayFrontier:
    """A current-image A/B/C mark ranked independently of route planning."""

    label: str
    frontier_id: str
    pixel_xy: tuple[float, float]
    world_xyz: tuple[float, float, float]
    information_gain_m3: float

    def __post_init__(self) -> None:
        if len(self.label) != 1 or not self.label.isalpha():
            raise ValueError("DisplayFrontier.label must be one alphabetic character.")
        object.__setattr__(
            self,
            "pixel_xy",
            _finite_tuple(self.pixel_xy, 2, field_name="pixel_xy"),
        )
        object.__setattr__(
            self,
            "world_xyz",
            _finite_tuple(self.world_xyz, 3, field_name="world_xyz"),
        )
        if not math.isfinite(float(self.information_gain_m3)):
            raise ValueError("information_gain_m3 must be finite.")
        if float(self.information_gain_m3) < 0.0:
            raise ValueError("information_gain_m3 cannot be negative.")

    def as_dict(self) -> dict[str, Any]:
        return {
            "label": self.label,
            "frontier_id": self.frontier_id,
            "pixel_xy": list(self.pixel_xy),
            "world_xyz": list(self.world_xyz),
            "information_gain_m3": self.information_gain_m3,
        }


@dataclass(frozen=True, slots=True)
class FrontierFrame:
    """Complete result for one RGB-D observation."""

    observation_step: int
    processed_rgb: np.ndarray
    processed_depth_m: np.ndarray
    processed_intrinsics: CameraIntrinsics
    prediction: FrontierNetPrediction
    observed_frontiers: tuple[ObservedFrontier, ...]
    displayed_frontiers: tuple[DisplayFrontier, ...]
    overlay_rgb: np.ndarray
    graph_snapshot: Mapping[str, Any]

    def frontier_for_label(self, label: str) -> DisplayFrontier:
        matches = [item for item in self.displayed_frontiers if item.label == label]
        if len(matches) != 1:
            available = ", ".join(item.label for item in self.displayed_frontiers)
            raise KeyError(
                f"Frontier label {label!r} is not active in observation "
                f"{self.observation_step}; available labels: {available or 'none'}."
            )
        return matches[0]


__all__ = [
    "CameraIntrinsics",
    "DisplayFrontier",
    "FrontierFrame",
    "FrontierNetPrediction",
    "FrontierState",
    "ObservedFrontier",
    "TrackedFrontier",
]
