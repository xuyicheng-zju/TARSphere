"""Robot-aware A* planning for a Human clearing a task passage."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.contracts.human_interaction import HUMAN_COMMUNICATION_RADIUS_M
from tarsphere.geometry import HUMAN_BODY_RADIUS_M
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    ROBOT_HUMAN_CENTER_CLEARANCE_M,
    ROBOT_HUMAN_MOTION_CLEARANCE_M,
)


_MIN_CLEAR_DISTANCE_M = ROBOT_HUMAN_MOTION_CLEARANCE_M + 0.10
_MAX_CLEAR_DISTANCE_M = 2.0
_CLEAR_DISTANCE_STEP_M = 0.25
_CANDIDATE_ANGLES = 32
_DIRECTION_EPSILON_M = 0.05
# A request stance is where dialogue happens, not already a successful yield.
# Moving one G1 footprint farther along the approach side creates observable
# retreat without prescribing an exact coordinate or a fixed action sequence.
_MIN_RETREAT_BEYOND_CONTACT_M = G1_NAVIGATION_FOOTPRINT_RADIUS_M


@dataclass(frozen=True, slots=True)
class SpatialClearancePlan:
    """One live decision about whether and where the blocker can move."""

    status: str
    reason: str
    robot_position: tuple[float, float]
    blocker_position: tuple[float, float]
    target: tuple[float, float, float] | None = None
    waypoints: tuple[tuple[float, float, float], ...] = ()

    @property
    def ready(self) -> bool:
        return self.status == "ready"

    def as_dict(self) -> dict:
        return {
            "status": self.status,
            "reason": self.reason,
            "robot_position": list(self.robot_position),
            "blocker_position": list(self.blocker_position),
            "target": list(self.target) if self.target is not None else None,
            "waypoints": [list(point) for point in self.waypoints],
        }


def plan_spatial_blocker_clearance(
    occupancy,
    *,
    blocker_position: Sequence[float],
    robot_position: Sequence[float],
    passage_contact_xy: Sequence[float],
    task_goal_xy: Sequence[float],
    other_human_positions: Sequence[Sequence[float]] = (),
) -> SpatialClearancePlan:
    """Choose a nearby clear pose only after Robot-aware A* becomes feasible.

    The passage contact is the authored approach-side request stance.  Its
    vector toward the blocker defines the local task direction, so moving to
    the far side of the blocker can never count as yielding.  The blocker
    destination itself is selected live from nearby free cells; no authored
    clear coordinate is used for execution. Other live Humans retain their
    shared body proxies in both the Human exit and the restored Robot route;
    the source occupancy map is never changed.
    """

    blocker = _finite_xy(blocker_position, "blocker_position")
    robot = _finite_xy(robot_position, "robot_position")
    contact = _finite_xy(passage_contact_xy, "passage_contact_xy")
    goal = _finite_xy(task_goal_xy, "task_goal_xy")
    z = _finite_z(blocker_position)
    if not isinstance(other_human_positions, Sequence):
        raise ValueError("other_human_positions must contain world positions")
    others = []
    for index, position in enumerate(other_human_positions):
        name = f"other_human_positions[{index}]"
        if not isinstance(position, Sequence) or len(position) not in (2, 3):
            raise ValueError(f"{name} must contain world XY or XYZ coordinates")
        try:
            point = _finite_xy(position, name)
            if len(position) == 3 and not math.isfinite(float(position[2])):
                raise ValueError(f"{name} must contain finite world coordinates")
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name} must contain finite world coordinates") from exc
        others.append(point)

    task_direction = np.asarray(blocker) - np.asarray(contact)
    norm = float(np.linalg.norm(task_direction))
    if norm <= 1e-6:
        raise ValueError("passage contact must differ from the blocker position")
    task_direction /= norm
    robot_from_blocker = np.asarray(robot) - np.asarray(blocker)
    forward_projection = float(np.dot(robot_from_blocker, task_direction))
    if forward_projection >= -_DIRECTION_EPSILON_M:
        return _waiting(
            "robot_not_on_approach_side",
            robot=robot,
            blocker=blocker,
        )
    contact_projection = float(
        np.dot(
            np.asarray(contact) - np.asarray(blocker),
            task_direction,
        )
    )
    if forward_projection > (contact_projection - _MIN_RETREAT_BEYOND_CONTACT_M):
        return _waiting(
            "robot_has_not_retreated_beyond_request_stance",
            robot=robot,
            blocker=blocker,
        )
    if math.dist(robot, blocker) <= ROBOT_HUMAN_MOTION_CLEARANCE_M:
        return _waiting(
            "robot_still_blocks_human_start",
            robot=robot,
            blocker=blocker,
        )

    robot_route_planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        world_exclusion_zones=[
            (x, y, ROBOT_HUMAN_CENTER_CLEARANCE_M) for x, y in others
        ],
        cache_static_nav=True,
    )
    if others and (
        not robot_route_planner.is_line_collision_free(robot, robot)
        or not robot_route_planner.is_line_collision_free(goal, goal)
    ):
        return _waiting(
            "task_route_unavailable_from_robot_pose",
            robot=robot,
            blocker=blocker,
        )
    robot_route = robot_route_planner.plan(
        PlanInput(
            start_xy=robot,
            start_yaw_rad=0.0,
            goal_xy=goal,
        )
    )
    if robot_route.status not in {"ok", "already_there"}:
        return _waiting(
            "task_route_unavailable_from_robot_pose",
            robot=robot,
            blocker=blocker,
        )
    task_polyline = [
        robot,
        *((float(point[0]), float(point[1])) for point in robot_route.waypoints),
    ]
    if len(task_polyline) == 1:
        task_polyline.append(goal)

    robot_nav = robot_route_planner.active_navigable_mask()
    human_nav = Planner(
        occupancy,
        backend="numpy",
        simplify=False,
        clearance_radius_m=HUMAN_BODY_RADIUS_M,
        world_exclusion_zones=[(x, y, 2.0 * HUMAN_BODY_RADIUS_M) for x, y in others],
        cache_static_nav=True,
    ).active_navigable_mask()

    candidates = _clear_pose_candidates(
        occupancy,
        blocker=blocker,
        robot=robot,
        task_polyline=task_polyline,
        prepared_human_nav=human_nav,
        other_human_positions=others,
    )
    if not candidates:
        return _waiting(
            "no_nearby_clear_pose",
            robot=robot,
            blocker=blocker,
        )

    human_planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        clearance_radius_m=HUMAN_BODY_RADIUS_M,
        prepared_navigable_mask=human_nav,
        world_exclusion_zones=[
            (
                robot[0],
                robot[1],
                ROBOT_HUMAN_MOTION_CLEARANCE_M,
            )
        ],
        cache_static_nav=True,
    )
    if others and not human_planner.is_line_collision_free(blocker, blocker):
        return _waiting(
            "other_human_blocks_human_start",
            robot=robot,
            blocker=blocker,
        )
    remaining = list(candidates)
    while remaining:
        human_route = human_planner.plan(
            PlanInput(
                start_xy=blocker,
                start_yaw_rad=0.0,
                goal_candidates_xy=tuple(remaining),
            )
        )
        if human_route.status not in {"ok", "already_there"}:
            return _waiting(
                "robot_aware_human_route_unavailable",
                robot=robot,
                blocker=blocker,
            )
        if human_route.goal_xy is None:
            raise RuntimeError("Human clearance A* returned no selected goal")
        selected = (
            float(human_route.goal_xy[0]),
            float(human_route.goal_xy[1]),
        )
        if _candidate_restores_task_route(
            occupancy,
            robot=robot,
            goal=goal,
            blocker_target=selected,
            prepared_robot_nav=robot_nav,
        ):
            waypoints_xy = [
                (float(point[0]), float(point[1])) for point in human_route.waypoints
            ]
            if not waypoints_xy or math.dist(waypoints_xy[-1], selected) > 1e-6:
                waypoints_xy.append(selected)
            waypoints = tuple((x, y, z) for x, y in waypoints_xy)
            target = (selected[0], selected[1], z)
            return SpatialClearancePlan(
                status="ready",
                reason="robot_aware_astar_ready",
                robot_position=robot,
                blocker_position=blocker,
                target=target,
                waypoints=waypoints,
            )
        remaining = [point for point in remaining if math.dist(point, selected) > 1e-6]

    return _waiting(
        "no_clear_pose_restores_task_route",
        robot=robot,
        blocker=blocker,
    )


def _clear_pose_candidates(
    occupancy,
    *,
    blocker: tuple[float, float],
    robot: tuple[float, float],
    task_polyline: Sequence[tuple[float, float]],
    prepared_human_nav: np.ndarray,
    other_human_positions: Sequence[tuple[float, float]] = (),
) -> tuple[tuple[float, float], ...]:
    nav = np.asarray(prepared_human_nav, dtype=bool)
    candidates: list[tuple[float, float]] = []
    seen_cells: set[tuple[int, int]] = set()
    radius = _MIN_CLEAR_DISTANCE_M
    while radius <= _MAX_CLEAR_DISTANCE_M + 1e-9:
        for index in range(_CANDIDATE_ANGLES):
            angle = 2.0 * math.pi * index / _CANDIDATE_ANGLES
            point = (
                blocker[0] + radius * math.cos(angle),
                blocker[1] + radius * math.sin(angle),
            )
            if any(
                math.dist(point, other) <= 2.0 * HUMAN_BODY_RADIUS_M
                for other in other_human_positions
            ):
                continue
            if not (
                ROBOT_HUMAN_MOTION_CLEARANCE_M
                < math.dist(point, robot)
                <= HUMAN_COMMUNICATION_RADIUS_M
            ):
                continue
            if _distance_to_polyline(point, task_polyline) <= (
                ROBOT_HUMAN_CENTER_CLEARANCE_M + 0.05
            ):
                continue
            cell_array = occupancy.world_to_cell(np.asarray([point], dtype=float))[0]
            cell = (int(cell_array[0]), int(cell_array[1]))
            if cell in seen_cells:
                continue
            seen_cells.add(cell)
            if not (
                0 <= cell[0] < nav.shape[0]
                and 0 <= cell[1] < nav.shape[1]
                and bool(nav[cell])
            ):
                continue
            candidates.append(point)
        radius += _CLEAR_DISTANCE_STEP_M
    return tuple(candidates)


def _candidate_restores_task_route(
    occupancy,
    *,
    robot: tuple[float, float],
    goal: tuple[float, float],
    blocker_target: tuple[float, float],
    prepared_robot_nav: np.ndarray,
) -> bool:
    planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        prepared_navigable_mask=prepared_robot_nav,
        world_exclusion_zones=[
            (
                blocker_target[0],
                blocker_target[1],
                ROBOT_HUMAN_CENTER_CLEARANCE_M,
            )
        ],
        cache_static_nav=True,
    )
    result = planner.plan(
        PlanInput(
            start_xy=robot,
            start_yaw_rad=0.0,
            goal_xy=goal,
        )
    )
    return result.status in {"ok", "already_there"}


def _distance_to_polyline(
    point: tuple[float, float],
    polyline: Sequence[tuple[float, float]],
) -> float:
    if len(polyline) < 2:
        return math.dist(point, polyline[0])
    p = np.asarray(point, dtype=float)
    best = math.inf
    for first, second in zip(polyline, polyline[1:]):
        a = np.asarray(first, dtype=float)
        b = np.asarray(second, dtype=float)
        delta = b - a
        scale = float(np.dot(delta, delta))
        ratio = (
            0.0
            if scale <= 1e-12
            else float(np.clip(np.dot(p - a, delta) / scale, 0.0, 1.0))
        )
        best = min(best, float(np.linalg.norm(p - (a + ratio * delta))))
    return best


def _finite_xy(value: Sequence[float], name: str) -> tuple[float, float]:
    if not isinstance(value, Sequence) or len(value) < 2:
        raise ValueError(f"{name} must contain world XY coordinates")
    point = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{name} must contain finite world XY coordinates")
    return point


def _finite_z(value: Sequence[float]) -> float:
    z = float(value[2]) if len(value) >= 3 else 0.0
    if not math.isfinite(z):
        raise ValueError("blocker_position Z must be finite")
    return z


def _waiting(
    reason: str,
    *,
    robot: tuple[float, float],
    blocker: tuple[float, float],
) -> SpatialClearancePlan:
    return SpatialClearancePlan(
        status="waiting_for_robot",
        reason=reason,
        robot_position=robot,
        blocker_position=blocker,
    )


__all__ = ["SpatialClearancePlan", "plan_spatial_blocker_clearance"]
