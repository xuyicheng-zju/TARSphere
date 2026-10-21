"""Resolve one official object path to the nearest G1-clear approach."""

from __future__ import annotations

from collections import Counter
import math
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.grounding.schema import (
    GroundedScene,
    ObjectApproachPath,
)
from tarsphere.bench.inputs.scenes import GRSceneMetadata, Route
from tarsphere.bench.spatial.physical_task.objects import (
    ManipulationTargetCandidate,
    discover_manipulation_targets,
)
from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.bench.spatial.physical_task.navigation_evidence import (
    POINT_MATCH_TOLERANCE_M as _POINT_MATCH_TOLERANCE_M,
    complete_path as _complete_path,
    path_length as _path_length,
    require_exact_occupancy_point,
    route_reaches_exact_goal as _route_reaches_exact_goal,
)
from tarsphere.assets import tarsphere_assets_root
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


Point2 = Tuple[float, float]


class StrictModel(BaseModel):
    """Immutable current-release target-approach schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TargetApproachRoute(StrictModel):
    """Collision-free route to the closest exact G1-clear object approach."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene containing the selected manipulation target.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded pickable object approached by this route.",
    )
    source_path_index: int = Field(
        ge=0,
        description="Selected object-conditioned route index in paths.json.",
    )
    source_waypoint_index: int = Field(
        ge=0,
        description=(
            "Waypoint on the selected source route that is nearest the target "
            "while remaining exactly G1-clear and reachable."
        ),
    )
    robot_start_xy: Point2 = Field(
        description="Official source-route start in scene world metres.",
    )
    navigation_approach_xy: Point2 = Field(
        description=(
            "Closest source-route waypoint that the official G1 footprint can "
            "occupy without planner snapping."
        ),
    )
    path_xy: Tuple[Point2, ...] = Field(
        min_length=2,
        description=(
            "Collision-free G1 occupancy polyline including the official "
            "start and resolved navigation approach."
        ),
    )
    path_length_m: float = Field(
        gt=0.0,
        description="Length of the collision-free approach polyline in metres.",
    )

    @model_validator(mode="after")
    def validate_route(self) -> "TargetApproachRoute":
        values = (
            *self.robot_start_xy,
            *self.navigation_approach_xy,
            *(value for point in self.path_xy for value in point),
            self.path_length_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("target approach route geometry must be finite")
        if math.dist(self.path_xy[0], self.robot_start_xy) > _POINT_MATCH_TOLERANCE_M:
            raise ValueError("target approach path must begin at robot_start_xy")
        if (
            math.dist(self.path_xy[-1], self.navigation_approach_xy)
            > _POINT_MATCH_TOLERANCE_M
        ):
            raise ValueError("target approach path must end at navigation_approach_xy")
        duplicates = sorted(
            point for point, count in Counter(self.path_xy).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"target approach path repeats points: {duplicates}")
        measured_length = sum(
            math.dist(first, second)
            for first, second in zip(self.path_xy, self.path_xy[1:])
        )
        if not math.isclose(
            measured_length,
            self.path_length_m,
            abs_tol=1e-6,
        ):
            raise ValueError("target approach path_length_m differs from path geometry")
        return self


class TargetApproachResolutionError(ValueError):
    """Raised when an official object path has no usable G1 approach."""


def resolve_target_approach_route(
    scene: GroundedScene,
    target: ManipulationTargetCandidate,
    *,
    approach_path_index: int,
    robot_start_xy: Optional[Point2] = None,
    assets_root: Optional[Path | str] = None,
) -> TargetApproachRoute:
    """Resolve the closest G1-clear point on one official target path.

    GRScenes object paths terminate near the object reference position. Their
    endpoints are not embodiment stances and may lie inside G1 footprint
    clearance. This resolver walks the selected dense source path backward
    from the object, finds the nearest exact G1-clear waypoint reachable from
    the official path start, and replans a collision-free G1 route to it.

    The result is navigation evidence only. Precise grasp stance, certified
    right-arm IK, attachment, and retreat remain later gates. Human and task events
    are authored only after this physical route exists.
    """
    current_targets = {
        candidate.target_object_id: candidate
        for candidate in discover_manipulation_targets(scene).targets
    }
    if current_targets.get(target.target_object_id) != target:
        raise ValueError(
            "target approach resolution requires an unmodified manipulation "
            "candidate from the same grounded scene"
        )
    return resolve_object_approach_route(
        scene,
        object_id=target.target_object_id,
        approach_path_index=approach_path_index,
        robot_start_xy=robot_start_xy,
        assets_root=assets_root,
    )


def resolve_object_approach_route(
    scene: GroundedScene,
    *,
    object_id: str,
    approach_path_index: int,
    robot_start_xy: Optional[Point2] = None,
    assets_root: Optional[Path | str] = None,
) -> TargetApproachRoute:
    """Resolve any grounded object's dense official path for the G1 base.

    Object-path endpoints are semantic hints rather than embodiment poses.
    The current resolver therefore walks the tracked dense route backward and
    accepts the closest waypoint that is both exactly G1-clear and connected
    to the requested Robot start.  Manipulation-specific admission remains in
    ``resolve_target_approach_route``; abandoned narrative destinations use
    this general navigation contract directly.
    """
    summary = _approach_summary(
        scene,
        target_object_id=object_id,
        path_index=approach_path_index,
    )
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root()
    )
    metadata = GRSceneMetadata.load(scene.scene_id, assets_root=root)
    try:
        source_route = metadata.paths[object_id][approach_path_index]
    except (KeyError, IndexError) as exc:
        raise ValueError(
            "selected target approach path is absent from paths.json: "
            f"{object_id}[{approach_path_index}]"
        ) from exc
    _validate_source_route(source_route, summary)

    occupancy_path = (root / scene.map.occupancy).resolve()
    if not occupancy_path.is_file():
        raise FileNotFoundError(
            f"official scene occupancy is missing: {occupancy_path}"
        )
    occupancy = OfficialOccupancyBevMap(
        occupancy_path=occupancy_path,
    )
    planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        cache_static_nav=True,
    )
    start_xy = (
        tuple(source_route.start_point[:2])
        if robot_start_xy is None
        else tuple(float(value) for value in robot_start_xy)
    )
    if not _point_is_exactly_navigable(planner, start_xy):
        raise TargetApproachResolutionError(
            f"selected Robot start is not an exact G1-clear point: {start_xy!r}"
        )

    for waypoint_index in range(len(source_route.path) - 1, -1, -1):
        candidate_xy = tuple(source_route.path[waypoint_index][:2])
        if not _point_is_exactly_navigable(planner, candidate_xy):
            continue
        output = planner.plan(
            PlanInput(
                start_xy=start_xy,
                start_yaw_rad=0.0,
                goal_xy=candidate_xy,
            )
        )
        if not _route_reaches_exact_goal(
            output,
            start_xy=start_xy,
            goal_xy=candidate_xy,
        ):
            continue
        path_xy = _complete_path(
            output,
            start_xy=start_xy,
            goal_xy=candidate_xy,
        )
        if len(path_xy) < 3:
            continue
        return TargetApproachRoute(
            scene_id=scene.scene_id,
            target_object_id=object_id,
            source_path_index=approach_path_index,
            source_waypoint_index=waypoint_index,
            robot_start_xy=start_xy,
            navigation_approach_xy=candidate_xy,
            path_xy=path_xy,
            path_length_m=_path_length(path_xy),
        )
    raise TargetApproachResolutionError(
        "official object path contains no reachable exact G1 approach: "
        f"{object_id}[{approach_path_index}]"
    )


def _approach_summary(
    scene: GroundedScene,
    *,
    target_object_id: str,
    path_index: int,
) -> ObjectApproachPath:
    for path in scene.object_approach_paths:
        if path.object_id == target_object_id and path.path_index == path_index:
            return path
    raise ValueError(
        "target approach resolution requires a grounded path summary for "
        f"{target_object_id}[{path_index}]"
    )


def _validate_source_route(
    route: Route,
    summary: ObjectApproachPath,
) -> None:
    if (
        route.object_id != summary.object_id
        or math.dist(route.start_point[:2], summary.start_xy) > _POINT_MATCH_TOLERANCE_M
        or math.dist(route.path[-1], summary.approach_xy) > _POINT_MATCH_TOLERANCE_M
        or len(route.path) != summary.waypoint_count
        or not math.isclose(
            route.distance,
            summary.distance_m,
            abs_tol=_POINT_MATCH_TOLERANCE_M,
        )
    ):
        raise ValueError(
            "dense paths.json route differs from its grounded summary: "
            f"{summary.object_id}[{summary.path_index}]"
        )


def _point_is_exactly_navigable(
    planner: Planner,
    position_xy: Point2,
) -> bool:
    try:
        require_exact_occupancy_point(
            planner,
            position_xy,
            label="target approach candidate",
        )
    except ValueError:
        return False
    return True


__all__ = [
    "TargetApproachResolutionError",
    "TargetApproachRoute",
    "resolve_object_approach_route",
    "resolve_target_approach_route",
]
