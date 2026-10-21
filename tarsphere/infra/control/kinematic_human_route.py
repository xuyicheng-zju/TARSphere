"""Resolve authored Human destinations with the shared collision-aware A*."""

from __future__ import annotations

import math
from collections.abc import Sequence

from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner


def resolve_human_route(
    planner: Planner,
    *,
    start: Sequence[float],
    target: Sequence[float],
    waypoints: Sequence[Sequence[float]] | None = None,
) -> list[list[float]]:
    """Plan through authored stops, preserving exact targets without snapping.

    The planner must be prepared with the Human collision radius on the
    private static scene occupancy, not the Robot's partial online map.
    Authored waypoints are required intermediate destinations, not an unchecked
    polyline. Every planned segment and the exact endpoint connection is checked.
    """
    stops = (
        [list(point) for point in waypoints]
        if waypoints is not None
        else [list(target)]
    )
    points = [list(start), list(target), *stops]
    if not stops or any(
        len(point) != 3 or not all(math.isfinite(float(value)) for value in point)
        for point in points
    ):
        raise ValueError("Human route requires finite world XYZ points")
    if math.dist(stops[-1], target) > 1e-6:
        raise ValueError("Human route must end at the requested destination")
    if any(abs(float(point[2]) - float(start[2])) > 1e-6 for point in points):
        raise ValueError("Kinematic Human routes currently require a level floor")
    previous = list(start)
    route: list[list[float]] = []
    for stop in stops:
        result = planner.plan(
            PlanInput(
                start_xy=(float(previous[0]), float(previous[1])),
                start_yaw_rad=0,
                goal_xy=(float(stop[0]), float(stop[1])),
            )
        )
        if result.status not in {"ok", "already_there"}:
            raise RuntimeError(f"Human A* route {previous} -> {stop}: {result.status}")
        leg = [
            [float(point[0]), float(point[1]), float(start[2])]
            for point in result.waypoints
        ]
        leg.append([float(value) for value in stop])
        for point in leg:
            if not planner.is_line_collision_free(previous[:2], point[:2]):
                raise RuntimeError(
                    f"Human A* exact route segment is blocked: {previous} -> {point}"
                )
            route.append(point)
            previous = point
    return route
