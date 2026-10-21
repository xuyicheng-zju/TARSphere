"""Shared exact-goal evidence for physical-task occupancy planning."""

from __future__ import annotations

from collections import deque
import math
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np

from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.harness.planner.contracts import PlanInput, PlanOutput
from tarsphere.harness.planner.planner import (
    Planner,
    erode_navigable_mask,
)
from tarsphere.assets import tarsphere_assets_root
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


Point2 = Tuple[float, float]
POINT_MATCH_TOLERANCE_M = 1e-6


def require_exact_occupancy_point(
    planner: Planner,
    position_xy: Point2,
    *,
    label: str,
) -> None:
    """Reject a position that the planner must move or snap."""
    output = planner.plan(
        PlanInput(
            start_xy=position_xy,
            start_yaw_rad=0.0,
            goal_xy=position_xy,
        )
    )
    if (
        output.status != "already_there"
        or output.goal_xy is None
        or math.dist(output.goal_xy, position_xy) > POINT_MATCH_TOLERANCE_M
    ):
        raise ValueError(f"{label} is not an exact G1-clear occupancy point")


def route_reaches_exact_goal(
    output: PlanOutput,
    *,
    start_xy: Point2,
    goal_xy: Point2,
) -> bool:
    """Return whether planning reached both endpoints without snapping."""
    if math.dist(start_xy, goal_xy) <= POINT_MATCH_TOLERANCE_M:
        return (
            output.status == "already_there"
            and output.goal_xy is not None
            and math.dist(output.goal_xy, goal_xy) <= POINT_MATCH_TOLERANCE_M
        )
    return (
        output.status == "ok"
        and float(output.metadata.get("start_snapped_m", math.inf))
        <= POINT_MATCH_TOLERANCE_M
        and output.goal_xy is not None
        and math.dist(output.goal_xy, goal_xy) <= POINT_MATCH_TOLERANCE_M
    )


def complete_path(
    output: PlanOutput,
    *,
    start_xy: Point2,
    goal_xy: Point2,
) -> tuple[Point2, ...]:
    """Return the planner polyline with exact start and goal included."""
    points = [start_xy]
    for waypoint in output.waypoints:
        point = (float(waypoint[0]), float(waypoint[1]))
        if math.dist(points[-1], point) > POINT_MATCH_TOLERANCE_M:
            points.append(point)
    if math.dist(points[-1], goal_xy) > POINT_MATCH_TOLERANCE_M:
        points.append(goal_xy)
    else:
        points[-1] = goal_xy
    return tuple(points)


def path_length(path_xy: tuple[Point2, ...]) -> float:
    """Return the Euclidean length of one ordered world-XY polyline."""
    return sum(math.dist(first, second) for first, second in zip(path_xy, path_xy[1:]))


def official_occupancy(
    scene: GroundedScene,
    *,
    assets_root: Optional[Path | str] = None,
) -> OfficialOccupancyBevMap:
    """Load the official occupancy map that owns one grounded scene."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root()
    )
    path = (root / scene.map.occupancy).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"official scene occupancy is missing: {path}")
    return OfficialOccupancyBevMap(occupancy_path=path)


def g1_navigation_planner(occupancy: OfficialOccupancyBevMap) -> Planner:
    """Build the frozen G1 static planner used by every physical-task gate."""
    return Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        cache_static_nav=True,
    )


def occupancy_cell_center(
    occupancy: OfficialOccupancyBevMap,
    position_xy: Sequence[float],
) -> Point2:
    """Return the exact centre of the occupancy cell holding one position."""
    row, column = occupancy.world_to_cell(
        np.asarray([tuple(float(value) for value in position_xy)], dtype=float)
    )[0]
    center = occupancy.cell_to_world(int(row), int(column))
    return (float(center[0]), float(center[1]))


def sample_radial_occupancy_points(
    occupancy: OfficialOccupancyBevMap,
    *,
    center_xy: Point2,
    reference_xy: Point2,
    minimum_distance_m: float,
    maximum_distance_m: float,
    radial_spacing_m: float,
    angular_samples: int,
) -> tuple[Point2, ...]:
    """Return exact occupancy-cell centres sampled around one task target."""
    if (
        minimum_distance_m < 0.0
        or maximum_distance_m <= minimum_distance_m
        or radial_spacing_m <= 0.0
        or angular_samples < 8
    ):
        raise ValueError("radial occupancy sampling contract is invalid")
    radial_intervals = max(
        1,
        math.ceil((maximum_distance_m - minimum_distance_m) / radial_spacing_m),
    )
    radii = tuple(
        minimum_distance_m
        + (maximum_distance_m - minimum_distance_m) * index / radial_intervals
        for index in range(radial_intervals + 1)
    )
    raw_positions = [reference_xy]
    raw_positions.extend(
        (
            center_xy[0] + radius * math.cos(angle),
            center_xy[1] + radius * math.sin(angle),
        )
        for radius in radii
        for angle in (
            2.0 * math.pi * index / angular_samples for index in range(angular_samples)
        )
    )
    cells = occupancy.world_to_cell(np.asarray(raw_positions, dtype=float))
    positions = {
        occupancy.cell_to_world(int(row), int(column)) for row, column in cells
    }
    return tuple(
        sorted(
            (
                position
                for position in positions
                if (
                    minimum_distance_m
                    <= math.dist(position, center_xy)
                    <= maximum_distance_m
                )
            ),
            key=lambda position: (
                math.dist(position, center_xy),
                position,
            ),
        )
    )


def reachable_radial_occupancy_points(
    occupancy: OfficialOccupancyBevMap,
    *,
    center_xy: Point2,
    reference_xy: Point2,
    minimum_distance_m: float,
    maximum_distance_m: float,
    radial_spacing_m: float,
    angular_samples: int,
    clearance_radius_m: float,
) -> tuple[Point2, ...]:
    """Filter radial samples to the reference point's clearance component."""
    pixels_per_meter = float(occupancy.pixels_per_meter)
    if (
        not math.isfinite(pixels_per_meter)
        or pixels_per_meter <= 0.0
        or not math.isfinite(clearance_radius_m)
        or clearance_radius_m < 0.0
    ):
        raise ValueError(
            "reachable occupancy sampling requires finite map scale and "
            "non-negative clearance"
        )
    navigation = occupancy.navigable_mask()
    clearance_cells = int(math.ceil(clearance_radius_m * pixels_per_meter))
    navigation = erode_navigable_mask(
        navigation,
        clearance_cells,
    )
    start_cell = occupancy.world_to_cell(np.asarray([reference_xy], dtype=float))[0]
    start = (int(start_cell[0]), int(start_cell[1]))
    rows, columns = navigation.shape
    if not (0 <= start[0] < rows and 0 <= start[1] < columns) or not bool(
        navigation[start]
    ):
        raise ValueError("reference point is not G1-clear in official occupancy")

    reachable = np.zeros_like(navigation, dtype=bool)
    reachable[start] = True
    pending = deque([start])
    movements = (
        (-1, -1),
        (-1, 0),
        (-1, 1),
        (0, -1),
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),
    )
    while pending:
        row, column = pending.popleft()
        for delta_row, delta_column in movements:
            next_row = row + delta_row
            next_column = column + delta_column
            if (
                not (0 <= next_row < rows and 0 <= next_column < columns)
                or reachable[next_row, next_column]
                or not navigation[next_row, next_column]
            ):
                continue
            if (
                delta_row != 0
                and delta_column != 0
                and (
                    not navigation[row + delta_row, column]
                    or not navigation[row, column + delta_column]
                )
            ):
                continue
            reachable[next_row, next_column] = True
            pending.append((next_row, next_column))

    candidates = sample_radial_occupancy_points(
        occupancy,
        center_xy=center_xy,
        reference_xy=reference_xy,
        minimum_distance_m=minimum_distance_m,
        maximum_distance_m=maximum_distance_m,
        radial_spacing_m=radial_spacing_m,
        angular_samples=angular_samples,
    )
    result = []
    for position in candidates:
        cell = occupancy.world_to_cell(np.asarray([position], dtype=float))[0]
        if reachable[int(cell[0]), int(cell[1])]:
            result.append(position)
    return tuple(result)


__all__ = [
    "POINT_MATCH_TOLERANCE_M",
    "complete_path",
    "g1_navigation_planner",
    "occupancy_cell_center",
    "official_occupancy",
    "path_length",
    "reachable_radial_occupancy_points",
    "require_exact_occupancy_point",
    "route_reaches_exact_goal",
    "sample_radial_occupancy_points",
]
