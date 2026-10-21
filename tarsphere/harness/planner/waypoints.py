"""Turn a planned A* path into the waypoint list a path controller accepts.

The planner emits one vertex per navigable BEV cell, which is far denser than
TARSphere's ``move_along_path`` controller wants: it steers toward the next
waypoint, so centimetre-scale vertices make it oscillate instead of advancing.
Resampling to metre-scale targets is pure geometry over an already
collision-checked path — it decides nothing about where to go — so it lives
beside the planner rather than inside any skill.

The only consumers are the authoring and certification layers (oracle route
scripts and the ``move_along_path`` smoke check). The evaluated agent has no
path controller: it owns every leg of its own trajectory through locomotion
primitives.
"""

from __future__ import annotations

import math


# Two segments meeting at a sharper angle than this are treated as a corner
# whose vertex must be kept, even when uniform spacing would skip it. Cutting a
# corner is what drives the robot into the wall the planner routed it around.
_CORNER_ANGLE_DEG = 5.0


def resample_path_for_controller(
    path: list[list[float]],
    *,
    start_xy: tuple[float, float] | list[float] | None = None,
    spacing_m: float,
) -> list[list[float]]:
    """Resample a collision-checked path into controller-scale targets.

    Args:
        path: Planned ``[x, y]`` vertices in metres, world frame, goal last.
        start_xy: Current robot ``[x, y]``, prepended so the first spacing
            interval is measured from where the robot actually stands rather
            than from the first planned vertex. Omit when the path already
            starts at the robot.
        spacing_m: Target distance in metres between consecutive emitted
            waypoints.

    Returns:
        Waypoints in metres, always ending at the final planned vertex.
        Intermediate points are spaced by roughly ``spacing_m`` along the path,
        plus the vertex of every corner sharper than 5 degrees.
    """
    if not path:
        return []

    spacing = max(float(spacing_m), 1e-6)
    if start_xy is None:
        vertices = [[float(p[0]), float(p[1])] for p in path]
    else:
        vertices = [
            [float(start_xy[0]), float(start_xy[1])],
            *[[float(p[0]), float(p[1])] for p in path],
        ]

    if len(vertices) == 1:
        return [list(vertices[0])]

    sampled: list[list[float]] = []
    travelled = 0.0
    next_emit = spacing
    for i in range(1, len(vertices)):
        start = vertices[i - 1]
        end = vertices[i]
        dx = end[0] - start[0]
        dy = end[1] - start[1]
        seg_len = math.hypot(dx, dy)
        if seg_len <= 1e-9:
            continue

        while travelled + seg_len >= next_emit:
            t = (next_emit - travelled) / seg_len
            sampled.append([start[0] + dx * t, start[1] + dy * t])
            next_emit += spacing
        travelled += seg_len
        if i < len(vertices) - 1 and _is_corner(
            vertices[i - 1], vertices[i], vertices[i + 1]
        ):
            _append_if_new(sampled, [float(end[0]), float(end[1])])

    _append_if_new(sampled, [float(path[-1][0]), float(path[-1][1])])

    return sampled


def _append_if_new(
    points: list[list[float]], point: list[float], *, eps: float = 1e-6
) -> None:
    """Append ``point`` unless it duplicates the last one already emitted.

    Uniform spacing and corner preservation can land on the same coordinate; a
    repeated waypoint would make the controller report arrival twice.
    """
    if points and math.hypot(points[-1][0] - point[0], points[-1][1] - point[1]) <= eps:
        return
    points.append(point)


def _is_corner(prev: list[float], cur: list[float], nxt: list[float]) -> bool:
    """Whether the path turns at ``cur`` by more than ``_CORNER_ANGLE_DEG``."""
    ax = cur[0] - prev[0]
    ay = cur[1] - prev[1]
    bx = nxt[0] - cur[0]
    by = nxt[1] - cur[1]
    la = math.hypot(ax, ay)
    lb = math.hypot(bx, by)
    if la <= 1e-9 or lb <= 1e-9:
        return False
    cross = abs(ax * by - ay * bx) / (la * lb)
    return cross > math.sin(math.radians(_CORNER_ANGLE_DEG))


__all__ = ["resample_path_for_controller"]
