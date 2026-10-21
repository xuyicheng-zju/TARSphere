"""Resolve the fixed G1 navigation–grasp–return calibration."""

from __future__ import annotations

import math
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.spatial.physical_task.navigation import (
    resolve_target_navigation,
)
from tarsphere.bench.spatial.physical_task.schema import (
    GraspAction,
    NavigationPoint,
    NavigationRoute,
    StaticNavigationEvidence,
)


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]

GRASP_MOTION_SCENE_ID = "g1_grasp_table_scene"
GRASP_MOTION_SCENE_ASSET = "scenes/g1_grasp_table.usda"
GRASP_MOTION_NAVIGATION_SCENE_ID = "MVUCSQAKTKJ5EAABAAAAABY8_usd"
GRASP_MOTION_TARGET_OBJECT_ID = "bottle/model_fdda7f8e37cde39156b218f568160b87_0"
GRASP_MOTION_TARGET_PRIM = "/World/env_0/scene/Bottle"
GRASP_MOTION_TABLE_OBJECT_ID = "simple_grasp_table"
GRASP_MOTION_TABLE_PRIM = "/World/env_0/scene/Table"
LOW_GRASP_MOTION_SCENE_ID = "g1_low_grasp_table_scene"
LOW_GRASP_MOTION_SCENE_ASSET = "scenes/g1_low_grasp_table.usda"

GRASP_MOTION_MINIMUM_TARGET_SUPPORT_INSET_M = 0.10

_ROBOT_START_XY: Point2 = (-0.7362313866615295, 0.1476239114999771)
_NAVIGATION_APPROACH_XY: Point2 = (
    0.6793661117553711,
    0.1476239114999771,
)
_RETURN_STOP_XY: Point2 = (-0.08394628018140793, 0.1476239114999771)
GRASP_MOTION_TARGET_CENTER_XYZ = (
    1.6360537198185922,
    0.1476239114999771,
    0.70,
)
# A current-view grasp stance must lie in the overlap between two physical
# workspaces: the whole target has to fit inside the ego camera frustum, and a
# partial-body plan still has to reach it without translating the feet.  The
# calibrated point is deliberately inside that overlap, rather than at either
# boundary: closer stances lose the tabletop below the camera while farther
# stances leave the fixed-foot manipulation workspace.
_GRASP_VIEW_REACH_STANCE_DISTANCE_M = 0.55
_ROBOT_GRASP_STANCE_XY: Point2 = (
    GRASP_MOTION_TARGET_CENTER_XYZ[0] - _GRASP_VIEW_REACH_STANCE_DISTANCE_M,
    GRASP_MOTION_TARGET_CENTER_XYZ[1],
)
GRASP_MOTION_TARGET_BOUNDS_MIN_XYZ = (
    1.6006642198185922,
    0.1122789114999771,
    0.6480085,
)
GRASP_MOTION_TARGET_BOUNDS_MAX_XYZ = (
    1.6714432198185922,
    0.1829689114999771,
    0.7519915,
)
GRASP_MOTION_TABLE_CENTER_XYZ = (
    1.7760537198185922,
    0.1476239114999771,
    0.324,
)
GRASP_MOTION_TABLE_BOUNDS_MIN_XYZ = (
    1.4760537198185921,
    -0.3523760885000229,
    0.0,
)
GRASP_MOTION_TABLE_BOUNDS_MAX_XYZ = (
    2.076053719818592,
    0.6476239114999771,
    0.648,
)
LOW_GRASP_MOTION_TARGET_CENTER_XYZ = (
    1.6360537198185922,
    0.1476239114999771,
    0.402,
)
LOW_GRASP_MOTION_TARGET_BOUNDS_MIN_XYZ = (
    1.6006642198185922,
    0.1122789114999771,
    0.3500085,
)
LOW_GRASP_MOTION_TARGET_BOUNDS_MAX_XYZ = (
    1.6714432198185922,
    0.1829689114999771,
    0.4539915,
)
LOW_GRASP_MOTION_TABLE_CENTER_XYZ = (
    1.7760537198185922,
    0.1476239114999771,
    0.175,
)
LOW_GRASP_MOTION_TABLE_BOUNDS_MIN_XYZ = (
    1.4760537198185921,
    -0.3523760885000229,
    0.0,
)
LOW_GRASP_MOTION_TABLE_BOUNDS_MAX_XYZ = (
    2.076053719818592,
    0.6476239114999771,
    0.35,
)


class StrictModel(BaseModel):
    """Immutable current-release grasp-motion schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspMotionTrial(StrictModel):
    """One real-asset G1 navigation, grasp, retraction, and return trial."""

    scene_id: str = Field(
        min_length=1,
        description="Simple table scene in which the calibration executes.",
    )
    scene_asset_path: str = Field(
        min_length=1,
        description="Repository asset path for the exact calibration scene.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Real GRScene pickable that G1 must contact and carry.",
    )
    source_support_object_id: str = Field(
        min_length=1,
        description="Simple table supporting the target before contact.",
    )
    target_position_xyz: Point3 = Field(
        description="Authored world-space centre of the real bottle asset.",
    )
    target_bounds_min_xyz: Point3 = Field(
        description="Minimum world-space corner of the bottle bounds.",
    )
    target_bounds_max_xyz: Point3 = Field(
        description="Maximum world-space corner of the bottle bounds.",
    )
    source_support_position_xyz: Point3 = Field(
        description="Authored world-space centre of the supporting table.",
    )
    source_support_bounds_min_xyz: Point3 = Field(
        description="Minimum world-space corner of the table bounds.",
    )
    source_support_bounds_max_xyz: Point3 = Field(
        description="Maximum world-space corner of the table bounds.",
    )
    navigate_to_target: NavigationRoute = Field(
        description="Short unladen route from the Robot start to the table.",
    )
    grasp: GraspAction = Field(
        description="Target-facing stance used by the shared grasp executor.",
    )
    return_from_target: NavigationRoute = Field(
        description="Loaded route that visibly carries the target away.",
    )
    approach_evidence: StaticNavigationEvidence = Field(
        description="Occupancy proof for the unladen table approach.",
    )
    return_evidence: StaticNavigationEvidence = Field(
        description="Occupancy proof for the loaded return route.",
    )

    @model_validator(mode="after")
    def validate_trial(self) -> "GraspMotionTrial":
        if (
            self.scene_id != self.approach_evidence.scene_id
            or self.scene_id != self.return_evidence.scene_id
        ):
            raise ValueError(
                "grasp-motion routes and navigation evidence identify different scenes"
            )
        if self.grasp.target_object_id != self.target_object_id:
            raise ValueError("grasp-motion target bindings are inconsistent")
        bound_pairs = (
            (self.target_bounds_min_xyz, self.target_bounds_max_xyz),
            (
                self.source_support_bounds_min_xyz,
                self.source_support_bounds_max_xyz,
            ),
        )
        if any(
            minimum[axis] > maximum[axis]
            for minimum, maximum in bound_pairs
            for axis in range(3)
        ):
            raise ValueError("grasp-motion object bounds must be ordered")
        if not math.isclose(
            self.target_bounds_min_xyz[2],
            self.source_support_bounds_max_xyz[2],
            abs_tol=0.002,
        ):
            raise ValueError("grasp-motion target must rest on the source support")
        target_insets = (
            self.target_bounds_min_xyz[0] - self.source_support_bounds_min_xyz[0],
            self.source_support_bounds_max_xyz[0] - self.target_bounds_max_xyz[0],
            self.target_bounds_min_xyz[1] - self.source_support_bounds_min_xyz[1],
            self.source_support_bounds_max_xyz[1] - self.target_bounds_max_xyz[1],
        )
        if min(target_insets) < GRASP_MOTION_MINIMUM_TARGET_SUPPORT_INSET_M:
            raise ValueError(
                "grasp-motion target must sit fully on the support with at "
                "least "
                f"{GRASP_MOTION_MINIMUM_TARGET_SUPPORT_INSET_M:.2f}m "
                "planar inset"
            )
        if not _same_xy(
            self.navigate_to_target.points[-1].position_xy,
            self.grasp.navigation_approach_xy,
        ):
            raise ValueError("navigate_to_target must end at the grasp approach")
        if not _same_xy(
            self.return_from_target.points[0].position_xy,
            self.grasp.retreat_xy,
        ):
            raise ValueError("return_from_target must start at the grasp retreat")
        _validate_evidence(
            self.navigate_to_target,
            self.approach_evidence,
        )
        _validate_evidence(
            self.return_from_target,
            self.return_evidence,
        )
        return self

    @property
    def robot_start_xy(self) -> Point2:
        """Return the first point of the certified unladen route."""
        return self.navigate_to_target.points[0].position_xy

    @property
    def robot_start_yaw_rad(self) -> float:
        """Return the first non-zero heading of the certified approach."""
        first_path = self.approach_evidence.segments[0].path_xy
        start = first_path[0]
        for point in first_path[1:]:
            if not _same_xy(start, point):
                return math.atan2(
                    point[1] - start[1],
                    point[0] - start[0],
                )
        raise ValueError("grasp-motion approach has no forward segment")


def resolve_grasp_motion_trial(
    navigation_scene: GroundedScene,
) -> GraspMotionTrial:
    """Resolve the standing-height table trial over certified occupancy."""
    return _resolve_grasp_motion_trial(
        navigation_scene,
        scene_id=GRASP_MOTION_SCENE_ID,
        scene_asset_path=GRASP_MOTION_SCENE_ASSET,
        target_position_xyz=GRASP_MOTION_TARGET_CENTER_XYZ,
        target_bounds_min_xyz=GRASP_MOTION_TARGET_BOUNDS_MIN_XYZ,
        target_bounds_max_xyz=GRASP_MOTION_TARGET_BOUNDS_MAX_XYZ,
        support_position_xyz=GRASP_MOTION_TABLE_CENTER_XYZ,
        support_bounds_min_xyz=GRASP_MOTION_TABLE_BOUNDS_MIN_XYZ,
        support_bounds_max_xyz=GRASP_MOTION_TABLE_BOUNDS_MAX_XYZ,
    )


def resolve_low_grasp_motion_trial(
    navigation_scene: GroundedScene,
) -> GraspMotionTrial:
    """Resolve the low-table trial that requires crouching before reach."""
    return _resolve_grasp_motion_trial(
        navigation_scene,
        scene_id=LOW_GRASP_MOTION_SCENE_ID,
        scene_asset_path=LOW_GRASP_MOTION_SCENE_ASSET,
        target_position_xyz=LOW_GRASP_MOTION_TARGET_CENTER_XYZ,
        target_bounds_min_xyz=LOW_GRASP_MOTION_TARGET_BOUNDS_MIN_XYZ,
        target_bounds_max_xyz=LOW_GRASP_MOTION_TARGET_BOUNDS_MAX_XYZ,
        support_position_xyz=LOW_GRASP_MOTION_TABLE_CENTER_XYZ,
        support_bounds_min_xyz=LOW_GRASP_MOTION_TABLE_BOUNDS_MIN_XYZ,
        support_bounds_max_xyz=LOW_GRASP_MOTION_TABLE_BOUNDS_MAX_XYZ,
    )


def _resolve_grasp_motion_trial(
    navigation_scene: GroundedScene,
    *,
    scene_id: str,
    scene_asset_path: str,
    target_position_xyz: Point3,
    target_bounds_min_xyz: Point3,
    target_bounds_max_xyz: Point3,
    support_position_xyz: Point3,
    support_bounds_min_xyz: Point3,
    support_bounds_max_xyz: Point3,
) -> GraspMotionTrial:
    """Resolve one reviewed table height over the shared open XY geometry."""
    if navigation_scene.scene_id != GRASP_MOTION_NAVIGATION_SCENE_ID:
        raise ValueError(
            "grasp-motion calibration requires its reviewed navigation map: "
            f"{GRASP_MOTION_NAVIGATION_SCENE_ID!r}"
        )
    route_scene = navigation_scene.model_copy(
        update={"scene_id": scene_id},
    )
    navigate_to_target = NavigationRoute(
        points=(
            NavigationPoint(
                point_id="robot_start",
                position_xy=_ROBOT_START_XY,
            ),
            NavigationPoint(
                point_id="table_approach",
                position_xy=_NAVIGATION_APPROACH_XY,
            ),
        )
    )
    return_from_target = NavigationRoute(
        points=(
            NavigationPoint(
                point_id="grasp_exit",
                position_xy=_NAVIGATION_APPROACH_XY,
            ),
            NavigationPoint(
                point_id="return_stop",
                position_xy=_RETURN_STOP_XY,
            ),
        )
    )
    return GraspMotionTrial(
        scene_id=scene_id,
        scene_asset_path=scene_asset_path,
        target_object_id=GRASP_MOTION_TARGET_OBJECT_ID,
        source_support_object_id=GRASP_MOTION_TABLE_OBJECT_ID,
        target_position_xyz=target_position_xyz,
        target_bounds_min_xyz=target_bounds_min_xyz,
        target_bounds_max_xyz=target_bounds_max_xyz,
        source_support_position_xyz=support_position_xyz,
        source_support_bounds_min_xyz=support_bounds_min_xyz,
        source_support_bounds_max_xyz=support_bounds_max_xyz,
        navigate_to_target=navigate_to_target,
        grasp=GraspAction(
            target_object_id=GRASP_MOTION_TARGET_OBJECT_ID,
            navigation_approach_xy=_NAVIGATION_APPROACH_XY,
            robot_stance_xy=_ROBOT_GRASP_STANCE_XY,
            robot_yaw_rad=0.0,
            retreat_xy=_NAVIGATION_APPROACH_XY,
        ),
        return_from_target=return_from_target,
        approach_evidence=resolve_target_navigation(
            route_scene,
            navigate_to_target,
        ),
        return_evidence=resolve_target_navigation(
            route_scene,
            return_from_target,
        ),
    )


def _validate_evidence(
    route: NavigationRoute,
    evidence: StaticNavigationEvidence,
) -> None:
    expected = tuple(
        (start.point_id, end.point_id)
        for start, end in zip(route.points, route.points[1:])
    )
    actual = tuple(
        (segment.start_point_id, segment.end_point_id) for segment in evidence.segments
    )
    if actual != expected:
        raise ValueError("grasp-motion navigation evidence does not cover its route")


def _same_xy(first: Point2, second: Point2) -> bool:
    return math.dist(first, second) <= 1e-9


__all__ = [
    "GRASP_MOTION_NAVIGATION_SCENE_ID",
    "GRASP_MOTION_MINIMUM_TARGET_SUPPORT_INSET_M",
    "GRASP_MOTION_SCENE_ASSET",
    "GRASP_MOTION_SCENE_ID",
    "GRASP_MOTION_TABLE_BOUNDS_MAX_XYZ",
    "GRASP_MOTION_TABLE_BOUNDS_MIN_XYZ",
    "GRASP_MOTION_TABLE_CENTER_XYZ",
    "GRASP_MOTION_TABLE_OBJECT_ID",
    "GRASP_MOTION_TABLE_PRIM",
    "GRASP_MOTION_TARGET_BOUNDS_MAX_XYZ",
    "GRASP_MOTION_TARGET_BOUNDS_MIN_XYZ",
    "GRASP_MOTION_TARGET_CENTER_XYZ",
    "GRASP_MOTION_TARGET_OBJECT_ID",
    "GRASP_MOTION_TARGET_PRIM",
    "LOW_GRASP_MOTION_SCENE_ASSET",
    "LOW_GRASP_MOTION_SCENE_ID",
    "LOW_GRASP_MOTION_TARGET_BOUNDS_MAX_XYZ",
    "LOW_GRASP_MOTION_TARGET_BOUNDS_MIN_XYZ",
    "LOW_GRASP_MOTION_TARGET_CENTER_XYZ",
    "GraspMotionTrial",
    "resolve_low_grasp_motion_trial",
    "resolve_grasp_motion_trial",
]
