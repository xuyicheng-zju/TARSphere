"""Semantic alignment and static reachability for physical-task routes."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Iterable, Optional, Protocol

from tarsphere.bench.grounding.schema import GroundedObject, GroundedScene
from tarsphere.bench.spatial.physical_task.schema import (
    NavigationPoint,
    NavigationRoute,
    PhysicalTaskRoute,
    StaticNavigationEvidence,
    StaticNavigationSegment,
)
from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.assets import tarsphere_assets_root
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


_POINT_MATCH_TOLERANCE_M = 1e-6
_SURFACE_HEIGHT_TOLERANCE_M = 1e-6


class PhysicalTaskBinding(Protocol):
    """Task fields needed to verify a resolved physical route."""

    scene_id: str
    target_object_id: str
    final_destination_object_id: str


def validate_route_against_task(
    route: PhysicalTaskRoute,
    task: PhysicalTaskBinding,
    scene: GroundedScene,
) -> None:
    """Reject a route that diverges from its fixed task or grounded scene."""
    if route.scene_id != task.scene_id or scene.scene_id != task.scene_id:
        raise ValueError(
            "physical-task route, fixed task, and grounded scene must "
            "identify the same scene"
        )
    if route.grasp.target_object_id != task.target_object_id:
        raise ValueError("route grasp target differs from the fixed task")
    if route.place.destination_object_id != task.final_destination_object_id:
        raise ValueError("route placement destination differs from the fixed task")

    objects = {item.object_id: item for item in scene.objects}
    target = _require_object(objects, route.grasp.target_object_id)
    if target.interaction != "pickable":
        raise ValueError("physical-task grasp target must be pickable")
    destination = _require_object(objects, route.place.destination_object_id)

    surface_x, surface_y, surface_z = route.place.support_surface_xyz
    lower = destination.bounds_min_xyz
    upper = destination.bounds_max_xyz
    if not (lower[0] <= surface_x <= upper[0] and lower[1] <= surface_y <= upper[1]):
        raise ValueError(
            "placement surface target lies outside the destination footprint"
        )
    if abs(surface_z - upper[2]) > _SURFACE_HEIGHT_TOLERANCE_M:
        raise ValueError("placement surface height differs from the destination top")

    _require_facing(
        route.grasp.robot_stance_xy,
        target.position_xyz[:2],
        route.grasp.robot_yaw_rad,
        label="grasp",
    )
    _require_facing(
        route.place.robot_stance_xy,
        route.place.support_surface_xyz[:2],
        route.place.robot_yaw_rad,
        label="placement",
    )


class PhysicalNavigationRoute(Protocol):
    """Spatial fields required to resolve static G1 navigation evidence."""

    scene_id: str
    navigate_to_target: NavigationRoute
    transport_to_destination: NavigationRoute


def resolve_static_navigation(
    route: PhysicalNavigationRoute,
    scene: GroundedScene,
    *,
    assets_root: Optional[Path | str] = None,
) -> StaticNavigationEvidence:
    """Resolve every authored navigation segment over official occupancy.

    This proves static G1 reachability only. It does not claim camera
    visibility, Human NavMesh feasibility, manipulation success, or runtime
    milestone timing.
    """
    if route.scene_id != scene.scene_id:
        raise ValueError(
            "physical-task route and grounded scene must identify the same scene"
        )
    planner = _navigation_planner(
        scene,
        assets_root=assets_root,
    )

    segments: list[StaticNavigationSegment] = []
    for route_name, navigation_route in (
        ("navigate_to_target", route.navigate_to_target),
        ("transport_to_destination", route.transport_to_destination),
    ):
        segments.extend(
            _compile_route(
                route_name=route_name,
                route=navigation_route,
                planner=planner,
            )
        )
    total_length = sum(segment.path_length_m for segment in segments)
    return StaticNavigationEvidence(
        scene_id=scene.scene_id,
        occupancy=scene.map.occupancy,
        robot_clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        segments=tuple(segments),
        total_path_length_m=total_length,
    )


def resolve_target_navigation(
    scene: GroundedScene,
    route: NavigationRoute,
    *,
    assets_root: Optional[Path | str] = None,
) -> StaticNavigationEvidence:
    """Resolve one unladen route used to certify a pickable object."""
    planner = _navigation_planner(
        scene,
        assets_root=assets_root,
    )
    segments = tuple(
        _compile_route(
            route_name="navigate_to_target",
            route=route,
            planner=planner,
        )
    )
    return StaticNavigationEvidence(
        scene_id=scene.scene_id,
        occupancy=scene.map.occupancy,
        robot_clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        segments=segments,
        total_path_length_m=sum(segment.path_length_m for segment in segments),
    )


def _navigation_planner(
    scene: GroundedScene,
    *,
    assets_root: Optional[Path | str],
) -> Planner:
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root
        else tarsphere_assets_root()
    )
    occupancy_path = (root / scene.map.occupancy).resolve()
    if not occupancy_path.is_file():
        raise FileNotFoundError(
            f"official scene occupancy is missing: {occupancy_path}"
        )
    return Planner(
        OfficialOccupancyBevMap(occupancy_path=occupancy_path),
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        cache_static_nav=True,
    )


def _compile_route(
    *,
    route_name: str,
    route: NavigationRoute,
    planner: Planner,
) -> list[StaticNavigationSegment]:
    segments: list[StaticNavigationSegment] = []
    for start, end in zip(route.points, route.points[1:]):
        output = planner.plan(
            PlanInput(
                start_xy=start.position_xy,
                start_yaw_rad=0.0,
                goal_xy=end.position_xy,
            )
        )
        if output.status != "ok":
            raise ValueError(
                f"static navigation failed for {route_name}: "
                f"{start.point_id} -> {end.point_id}; status={output.status}"
            )
        start_snap = float(output.metadata.get("start_snapped_m", math.inf))
        if start_snap > _POINT_MATCH_TOLERANCE_M:
            raise ValueError(
                f"navigation start is not G1-clear: {start.point_id}; "
                f"snapped by {start_snap:.6f} m"
            )
        if (
            output.goal_xy is None
            or math.dist(output.goal_xy, end.position_xy) > _POINT_MATCH_TOLERANCE_M
        ):
            raise ValueError(f"navigation goal is not G1-clear: {end.point_id}")
        path = _complete_path(start, end, output.waypoints)
        segments.append(
            StaticNavigationSegment(
                route_name=route_name,
                start_point_id=start.point_id,
                end_point_id=end.point_id,
                path_xy=path,
                path_length_m=_path_length(path),
            )
        )
    return segments


def _complete_path(
    start: NavigationPoint,
    end: NavigationPoint,
    waypoints: Iterable[Iterable[float]],
) -> tuple[tuple[float, float], ...]:
    path: list[tuple[float, float]] = [start.position_xy]
    for waypoint in waypoints:
        point = tuple(float(value) for value in tuple(waypoint)[:2])
        if math.dist(path[-1], point) > _POINT_MATCH_TOLERANCE_M:
            path.append(point)
    if math.dist(path[-1], end.position_xy) > _POINT_MATCH_TOLERANCE_M:
        path.append(end.position_xy)
    else:
        path[-1] = end.position_xy
    return tuple(path)


def _path_length(path: tuple[tuple[float, float], ...]) -> float:
    return sum(math.dist(first, second) for first, second in zip(path, path[1:]))


def _require_object(
    objects: dict[str, GroundedObject],
    object_id: str,
) -> GroundedObject:
    try:
        return objects[object_id]
    except KeyError as exc:
        raise ValueError(
            f"physical-task route references an unknown object: {object_id}"
        ) from exc


def _require_facing(
    stance_xy: tuple[float, float],
    target_xy: tuple[float, float],
    yaw_rad: float,
    *,
    label: str,
) -> None:
    expected = math.atan2(
        target_xy[1] - stance_xy[1],
        target_xy[0] - stance_xy[0],
    )
    error = abs((yaw_rad - expected + math.pi) % (2.0 * math.pi) - math.pi)
    if error > 1e-3:
        raise ValueError(f"{label} Robot yaw does not face its physical target")


__all__ = [
    "PhysicalNavigationRoute",
    "resolve_static_navigation",
    "resolve_target_navigation",
    "validate_route_against_task",
]
