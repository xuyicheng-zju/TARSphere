"""Deterministic Human waypoint motion on the simulator's physics clock.

This module owns trajectories, not rendering, collision geometry or planning.
Callers supply an already validated route and apply each returned pose to the
visible character and its collision body before advancing PhysX.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass


def _point(value: Sequence[float]) -> tuple[float, float, float]:
    if len(value) != 3:
        raise ValueError("Human route points must contain world XYZ metres")
    point = tuple(float(coordinate) for coordinate in value)
    if not all(math.isfinite(coordinate) for coordinate in point):
        raise ValueError("Human route points must be finite")
    return point


@dataclass(frozen=True)
class HumanMotionPose:
    """World pose at a physics-clock instant; heading is +X-forward yaw."""

    position: tuple[float, float, float]
    yaw_rad: float
    completed: bool
    distance_m: float


class KinematicHumanMotion:
    """Sample a polyline by elapsed simulation time without timestep drift."""

    def __init__(
        self,
        *,
        start: Sequence[float],
        waypoints: Sequence[Sequence[float]],
        speed_mps: float,
        initial_yaw_rad: float,
        final_yaw_rad: float | None = None,
    ) -> None:
        self.speed_mps = float(speed_mps)
        self.initial_yaw_rad = float(initial_yaw_rad)
        self.final_yaw_rad = None if final_yaw_rad is None else float(final_yaw_rad)
        if not math.isfinite(self.speed_mps) or self.speed_mps <= 0:
            raise ValueError("Human speed_mps must be finite and positive")
        if not math.isfinite(self.initial_yaw_rad) or (
            self.final_yaw_rad is not None and not math.isfinite(self.final_yaw_rad)
        ):
            raise ValueError("Human headings must be finite radians")
        if not waypoints:
            raise ValueError("Human goto requires a nonempty validated route")
        self.points = (_point(start), *(_point(point) for point in waypoints))
        self.lengths = tuple(
            math.dist(first, second)
            for first, second in zip(self.points, self.points[1:])
        )
        self.length_m = math.fsum(self.lengths)

    def sample(self, elapsed_s: float) -> HumanMotionPose:
        """Sample absolute elapsed time; repeated observations cannot move it."""
        elapsed_s = float(elapsed_s)
        if not math.isfinite(elapsed_s) or elapsed_s < 0:
            raise ValueError(
                "Human elapsed simulation time must be finite and nonnegative"
            )
        requested_distance = elapsed_s * self.speed_mps
        route_complete = requested_distance >= self.length_m
        distance = min(requested_distance, self.length_m)
        remaining = distance
        yaw = self.initial_yaw_rad
        for first, second, length in zip(self.points, self.points[1:], self.lengths):
            if length == 0:
                continue
            dx, dy = second[0] - first[0], second[1] - first[1]
            if dx != 0 or dy != 0:
                yaw = math.atan2(dy, dx)
            if remaining < length and not route_complete:
                fraction = remaining / length
                return HumanMotionPose(
                    tuple(a + (b - a) * fraction for a, b in zip(first, second)),
                    yaw,
                    False,
                    distance,
                )
            remaining = max(0.0, remaining - length)
        return HumanMotionPose(
            self.points[-1],
            yaw if self.final_yaw_rad is None else self.final_yaw_rad,
            True,
            distance,
        )
