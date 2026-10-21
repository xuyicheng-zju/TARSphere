"""Sparse, episode-scoped graph of robot poses and visual frontiers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

import numpy as np

from tarsphere.infra.navigation.open_frontier.contracts import (
    FrontierState,
    ObservedFrontier,
    TrackedFrontier,
)


@dataclass(frozen=True, slots=True)
class FrontierGraphConfig:
    merge_position_distance_m: float = 1.8
    merge_direction_weight: float = 2.0
    maximum_merge_cost: float = 1.8
    visited_position_distance_m: float = 0.5
    visited_direction_angle_rad: float = 0.4
    # The released OpenFrontier configuration subtracts 2000 raw FrontierNet
    # gain units. Extraction stores gain after its 0.01 volume scale, so 20.0
    # is the exactly equivalent penalty in this graph's internal units.
    visited_gain_reduction_m3: float = 20.0

    def __post_init__(self) -> None:
        values = (
            self.merge_position_distance_m,
            self.merge_direction_weight,
            self.maximum_merge_cost,
            self.visited_position_distance_m,
            self.visited_direction_angle_rad,
            self.visited_gain_reduction_m3,
        )
        if not all(
            math.isfinite(float(value)) and float(value) > 0 for value in values
        ):
            raise ValueError("Frontier graph thresholds must be finite and positive.")


@dataclass(frozen=True, slots=True)
class PoseNode:
    pose_id: str
    observation_step: int
    world_T_camera: tuple[tuple[float, float, float, float], ...]

    @property
    def world_xyz(self) -> tuple[float, float, float]:
        return tuple(float(self.world_T_camera[index][3]) for index in range(3))

    def as_dict(self) -> dict[str, Any]:
        return {
            "pose_id": self.pose_id,
            "observation_step": self.observation_step,
            "world_T_camera": [list(row) for row in self.world_T_camera],
        }


def _normalized(vector: Any, *, field_name: str) -> np.ndarray:
    values = np.asarray(vector, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError(f"{field_name} must contain three finite values.")
    norm = float(np.linalg.norm(values))
    if norm <= 1e-8:
        raise ValueError(f"{field_name} must be non-zero.")
    return values / norm


def _frontier_camera_rotation(view_direction_world: Any) -> np.ndarray:
    """Build the OpenFrontier camera frame whose +Z looks at the frontier."""

    forward = _normalized(
        view_direction_world,
        field_name="frontier view direction",
    )
    camera_down = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
    camera_down -= float(np.dot(camera_down, forward)) * forward
    if float(np.linalg.norm(camera_down)) <= 1e-8:
        camera_down = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
        camera_down -= float(np.dot(camera_down, forward)) * forward
    camera_down = _normalized(camera_down, field_name="frontier camera down")
    camera_right = _normalized(
        np.cross(camera_down, forward),
        field_name="frontier camera right",
    )
    camera_down = _normalized(
        np.cross(forward, camera_right),
        field_name="frontier camera down",
    )
    return np.column_stack((camera_right, camera_down, forward))


def _rotation_difference_rad(first: np.ndarray, second: np.ndarray) -> float:
    relative = np.asarray(second) @ np.asarray(first).T
    cosine = float(np.clip((np.trace(relative) - 1.0) * 0.5, -1.0, 1.0))
    return math.acos(cosine)


class FrontierGraph:
    """Maintain stable frontier identities without owning map or policy state."""

    def __init__(self, config: FrontierGraphConfig | None = None) -> None:
        self.config = config or FrontierGraphConfig()
        self._frontiers: dict[str, TrackedFrontier] = {}
        self._poses: dict[str, PoseNode] = {}
        self._edges: dict[tuple[str, str], float] = {}
        self._next_frontier_index = 1
        self._next_pose_index = 1

    def reset(self) -> None:
        self._frontiers.clear()
        self._poses.clear()
        self._edges.clear()
        self._next_frontier_index = 1
        self._next_pose_index = 1

    @property
    def frontiers(self) -> Mapping[str, TrackedFrontier]:
        return dict(self._frontiers)

    def get(self, frontier_id: str) -> TrackedFrontier:
        try:
            return self._frontiers[str(frontier_id)]
        except KeyError as exc:
            raise KeyError(f"Unknown frontier id {frontier_id!r}.") from exc

    def add_observation(
        self,
        *,
        observation_step: int,
        world_T_camera: Any,
        frontiers: tuple[ObservedFrontier, ...],
    ) -> dict[int, str]:
        """Add one camera pose and associate its candidates with stable IDs."""

        transform = np.asarray(world_T_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("Frontier graph camera pose must be finite 4x4.")
        pose_id = f"P{self._next_pose_index:06d}"
        self._next_pose_index += 1
        pose = PoseNode(
            pose_id=pose_id,
            observation_step=int(observation_step),
            world_T_camera=tuple(
                tuple(float(value) for value in row) for row in transform
            ),
        )
        previous_pose_id = next(reversed(self._poses), None)
        self._poses[pose_id] = pose
        if previous_pose_id is not None:
            previous = self._poses[previous_pose_id]
            self._set_edge(
                previous_pose_id,
                pose_id,
                math.dist(previous.world_xyz, pose.world_xyz),
            )

        assignments: dict[int, str] = {}
        # Visited frontiers remain association candidates. Otherwise the same
        # explored opening is assigned a fresh active ID on every revisit and
        # no history-based information-gain reduction can persist.
        available_tracks = set(self._frontiers)
        pairs: list[tuple[float, int, str]] = []
        for observed in frontiers:
            for frontier_id in available_tracks:
                track = self._frontiers[frontier_id]
                cost = self._association_cost(observed, track)
                if cost is not None:
                    pairs.append((cost, observed.observation_index, frontier_id))
        used_observations: set[int] = set()
        used_tracks: set[str] = set()
        for _, observation_index, frontier_id in sorted(pairs):
            if observation_index in used_observations or frontier_id in used_tracks:
                continue
            observed = next(
                item
                for item in frontiers
                if item.observation_index == observation_index
            )
            self._merge_observation(
                self._frontiers[frontier_id],
                observed,
                pose_id=pose_id,
                observation_step=int(observation_step),
            )
            assignments[observation_index] = frontier_id
            used_observations.add(observation_index)
            used_tracks.add(frontier_id)

        for observed in frontiers:
            if observed.observation_index in used_observations:
                continue
            frontier_id = f"F{self._next_frontier_index:06d}"
            self._next_frontier_index += 1
            self._frontiers[frontier_id] = TrackedFrontier(
                frontier_id=frontier_id,
                world_xyz=observed.world_xyz,
                view_direction_world=observed.view_direction_world,
                base_information_gain_m3=float(observed.information_gain_m3),
                updated_information_gain_m3=float(observed.information_gain_m3),
                first_observation_step=int(observation_step),
                last_observation_step=int(observation_step),
                observation_count=1,
                parent_pose_ids=[pose_id],
            )
            self._set_edge(
                pose_id,
                frontier_id,
                math.dist(pose.world_xyz, observed.world_xyz),
            )
            assignments[observed.observation_index] = frontier_id
        self.adjust_gains_for_history()
        return assignments

    def _association_cost(
        self,
        observed: ObservedFrontier,
        tracked: TrackedFrontier,
    ) -> float | None:
        position_distance = float(
            np.sum(
                np.abs(np.asarray(observed.world_xyz) - np.asarray(tracked.world_xyz))
            )
        )
        if position_distance > self.config.merge_position_distance_m:
            return None
        first = _normalized(
            observed.view_direction_world,
            field_name="observed frontier direction",
        )
        second = _normalized(
            tracked.view_direction_world,
            field_name="tracked frontier direction",
        )
        direction_distance = 1.0 - float(np.clip(np.dot(first, second), -1.0, 1.0))
        cost = (
            position_distance + self.config.merge_direction_weight * direction_distance
        )
        return cost if cost <= self.config.maximum_merge_cost else None

    def _merge_observation(
        self,
        tracked: TrackedFrontier,
        observed: ObservedFrontier,
        *,
        pose_id: str,
        observation_step: int,
    ) -> None:
        # Geometry fusion must not depend on exploration utility. A visited
        # track may have near-zero adjusted gain but still owns valid geometry.
        old_weight = max(tracked.base_information_gain_m3, 1e-4)
        new_weight = max(observed.information_gain_m3, 1e-4)
        total_weight = old_weight + new_weight
        position = (
            old_weight * np.asarray(tracked.world_xyz)
            + new_weight * np.asarray(observed.world_xyz)
        ) / total_weight
        direction = _normalized(
            tracked.view_direction_world, field_name="tracked direction"
        ) + _normalized(observed.view_direction_world, field_name="observed direction")
        if np.linalg.norm(direction) <= 1e-8:
            direction = np.asarray(observed.view_direction_world, dtype=np.float64)
        direction /= np.linalg.norm(direction)
        tracked.world_xyz = tuple(float(value) for value in position)
        tracked.view_direction_world = tuple(float(value) for value in direction)
        tracked.base_information_gain_m3 = float(
            0.5 * (tracked.base_information_gain_m3 + observed.information_gain_m3)
        )
        tracked.updated_information_gain_m3 = tracked.base_information_gain_m3
        tracked.last_observation_step = int(observation_step)
        tracked.observation_count += 1
        if pose_id not in tracked.parent_pose_ids:
            tracked.parent_pose_ids.append(pose_id)
        self._set_edge(
            pose_id,
            tracked.frontier_id,
            math.dist(self._poses[pose_id].world_xyz, tracked.world_xyz),
        )

    def adjust_gains_for_history(self) -> tuple[str, ...]:
        """Apply OpenFrontier's pose-history visitation penalty to all tracks."""

        newly_visited: list[str] = []
        for frontier_id, frontier in self._frontiers.items():
            frontier_rotation = _frontier_camera_rotation(
                frontier.view_direction_world,
            )
            close_pose_count = 0
            for pose in self._poses.values():
                pose_transform = np.asarray(
                    pose.world_T_camera,
                    dtype=np.float64,
                )
                distance = float(
                    np.linalg.norm(
                        np.asarray(frontier.world_xyz) - pose_transform[:3, 3]
                    )
                )
                angle = _rotation_difference_rad(
                    frontier_rotation,
                    pose_transform[:3, :3],
                )
                if (
                    distance < self.config.visited_position_distance_m
                    and angle < self.config.visited_direction_angle_rad
                ):
                    close_pose_count += 1

            was_visited = frontier.state is FrontierState.VISITED
            frontier.updated_information_gain_m3 = max(
                frontier.base_information_gain_m3
                - self.config.visited_gain_reduction_m3 * close_pose_count,
                1e-4,
            )
            frontier.state = (
                FrontierState.VISITED if close_pose_count else FrontierState.ACTIVE
            )
            if close_pose_count and not was_visited:
                newly_visited.append(frontier_id)
        return tuple(newly_visited)

    def snapshot(self) -> dict[str, Any]:
        return {
            "poses": [pose.as_dict() for pose in self._poses.values()],
            "frontiers": [frontier.as_dict() for frontier in self._frontiers.values()],
            "edges": [
                {"source": source, "target": target, "distance_m": distance}
                for (source, target), distance in sorted(self._edges.items())
            ],
        }

    def _set_edge(self, first: str, second: str, distance_m: float) -> None:
        if not math.isfinite(float(distance_m)) or distance_m < 0.0:
            raise ValueError("Frontier graph edge distance must be non-negative.")
        edge = tuple(sorted((str(first), str(second))))
        self._edges[edge] = float(distance_m)


__all__ = ["FrontierGraph", "FrontierGraphConfig", "PoseNode"]
