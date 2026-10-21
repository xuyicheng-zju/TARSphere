"""Resolve G1-clear entry poses for live grasp certification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.infra.simulation.scene_query import (
    belongs_to_prim_tree,
)
from tarsphere.bench.certification.source_support import (
    SourceSupportEvidence,
)
from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.spatial.physical_task.navigation_evidence import (
    reachable_radial_occupancy_points,
)
from tarsphere.bench.spatial.physical_task.objects import (
    ManipulationTargetCandidate,
    discover_manipulation_targets,
)
from tarsphere.bench.spatial.physical_task.target_approach import (
    TargetApproachRoute,
)
from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.assets import tarsphere_assets_root
from tarsphere.infra.control.physical_contracts import (
    G1_GRASP_ENTRY_MAX_HORIZONTAL_DISTANCE_M,
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


Point2 = Tuple[float, float]


class StrictModel(BaseModel):
    """Immutable current-release grasp-entry schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspEntrySearch(StrictModel):
    """Bounded occupancy search before live grasp IK and attachment."""

    minimum_target_distance_m: float = Field(
        ge=0.0,
        description=(
            "Smallest horizontal distance sampled between the G1 base "
            "centre and live target centre, in metres."
        ),
    )
    maximum_target_distance_m: float = Field(
        gt=0.0,
        description=(
            "Largest horizontal distance sampled between the G1 base centre "
            "and live target centre, in metres."
        ),
    )
    radial_spacing_m: float = Field(
        gt=0.0,
        description="Maximum spacing between sampled entry rings, in metres.",
    )
    angular_samples: int = Field(
        ge=8,
        description="Number of evenly spaced base positions per stance ring.",
    )

    @model_validator(mode="after")
    def validate_range(self) -> "GraspEntrySearch":
        if self.minimum_target_distance_m >= self.maximum_target_distance_m:
            raise ValueError("grasp entry distance range must be strictly increasing")
        return self


GRASP_ENTRY_SEARCH = GraspEntrySearch(
    minimum_target_distance_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    maximum_target_distance_m=G1_GRASP_ENTRY_MAX_HORIZONTAL_DISTANCE_M,
    radial_spacing_m=0.05,
    angular_samples=64,
)


class GraspEntryPose(StrictModel):
    """One G1-clear starting pose for live near-field grasp closure."""

    robot_entry_xy: Point2 = Field(
        description=(
            "Exact official-occupancy cell centre where G1 may begin live "
            "near-field grasp closure, in world metres."
        ),
    )
    robot_yaw_rad: float = Field(
        ge=-math.pi,
        le=math.pi,
        description="World yaw pointing the G1 torso at the live target centre.",
    )
    target_distance_xy_m: float = Field(
        ge=0.0,
        description=(
            "Horizontal distance from the base centre to the live target "
            "centre, in metres."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "GraspEntryPose":
        values = (
            *self.robot_entry_xy,
            self.robot_yaw_rad,
            self.target_distance_xy_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("grasp entry geometry must be finite")
        return self


class GraspEntryPoseSet(StrictModel):
    """Occupancy-reachable entry poses for one live-supported target."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene containing the target and occupancy map.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded manipulation target faced by every entry pose.",
    )
    source_support_object_id: str = Field(
        min_length=1,
        description="Live-certified object initially supporting the target.",
    )
    target_approach_path_index: int = Field(
        ge=0,
        description=(
            "Official target-conditioned path whose G1-clear endpoint seeded "
            "these near-field entry poses."
        ),
    )
    target_center_xy: Point2 = Field(
        description=(
            "Horizontal centre of the live target USD bounds in world metres."
        ),
    )
    navigation_approach_xy: Point2 = Field(
        description=("Exact G1-clear endpoint of the selected target approach route."),
    )
    robot_clearance_radius_m: float = Field(
        gt=0.0,
        description=("G1 footprint radius applied to official occupancy planning."),
    )
    poses: Tuple[GraspEntryPose, ...] = Field(
        description=(
            "Deterministically ordered occupancy-reachable entry poses; an "
            "empty tuple means no static grasp entry was found."
        ),
    )

    @model_validator(mode="after")
    def validate_poses(self) -> "GraspEntryPoseSet":
        values = (*self.target_center_xy, *self.navigation_approach_xy)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("grasp entry set geometry must be finite")
        positions = tuple(pose.robot_entry_xy for pose in self.poses)
        if len(set(positions)) != len(positions):
            raise ValueError("grasp entry poses must be unique")
        return self


def find_grasp_entry_poses(
    scene: GroundedScene,
    target: ManipulationTargetCandidate,
    target_approach: TargetApproachRoute,
    source_support: SourceSupportEvidence,
    *,
    search: GraspEntrySearch = GRASP_ENTRY_SEARCH,
    assets_root: Optional[Path | str] = None,
) -> GraspEntryPoseSet:
    """Return exact occupancy poses; do not infer visibility, IK, or grasp."""
    current_targets = {
        candidate.target_object_id: candidate
        for candidate in discover_manipulation_targets(scene).targets
    }
    if current_targets.get(target.target_object_id) != target:
        raise ValueError(
            "grasp entry search requires an unmodified manipulation "
            "candidate from the same grounded scene"
        )
    if (
        target_approach.scene_id != scene.scene_id
        or target_approach.target_object_id != target.target_object_id
        or target_approach.source_path_index not in target.approach_path_indexes
    ):
        raise ValueError(
            "grasp entry search requires a matching certified target approach"
        )
    if source_support.target_object_id != target.target_object_id:
        raise ValueError(
            "grasp entry search requires source evidence for the same target"
        )
    if source_support.support_object_id not in target.source_support_object_ids:
        raise ValueError("live source support is absent from the target candidate")
    objects = {obj.object_id: obj for obj in scene.objects}
    support = objects[source_support.support_object_id]
    if not support.runtime_prim or any(
        not belongs_to_prim_tree(path, support.runtime_prim)
        for path in source_support.sampled_prim_paths
    ):
        raise ValueError(
            "source-support prim evidence does not match the grounded support"
        )

    target_center_xy = (
        (
            source_support.target_bounds_min_xyz[0]
            + source_support.target_bounds_max_xyz[0]
        )
        / 2.0,
        (
            source_support.target_bounds_min_xyz[1]
            + source_support.target_bounds_max_xyz[1]
        )
        / 2.0,
    )
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root()
    )
    occupancy_path = (root / scene.map.occupancy).resolve()
    if not occupancy_path.is_file():
        raise FileNotFoundError(
            f"official scene occupancy is missing: {occupancy_path}"
        )
    occupancy = OfficialOccupancyBevMap(
        occupancy_path=occupancy_path,
    )
    poses = []
    for position_xy in reachable_radial_occupancy_points(
        occupancy,
        center_xy=target_center_xy,
        reference_xy=target_approach.navigation_approach_xy,
        minimum_distance_m=search.minimum_target_distance_m,
        maximum_distance_m=search.maximum_target_distance_m,
        radial_spacing_m=search.radial_spacing_m,
        angular_samples=search.angular_samples,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    ):
        poses.append(
            GraspEntryPose(
                robot_entry_xy=position_xy,
                robot_yaw_rad=math.atan2(
                    target_center_xy[1] - position_xy[1],
                    target_center_xy[0] - position_xy[0],
                ),
                target_distance_xy_m=math.dist(
                    position_xy,
                    target_center_xy,
                ),
            )
        )
    return GraspEntryPoseSet(
        scene_id=scene.scene_id,
        target_object_id=target.target_object_id,
        source_support_object_id=source_support.support_object_id,
        target_approach_path_index=target_approach.source_path_index,
        target_center_xy=target_center_xy,
        navigation_approach_xy=target_approach.navigation_approach_xy,
        robot_clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        poses=tuple(
            sorted(
                poses,
                key=lambda pose: (
                    pose.target_distance_xy_m,
                    pose.robot_entry_xy,
                ),
            )
        ),
    )


__all__ = [
    "GRASP_ENTRY_SEARCH",
    "GraspEntryPose",
    "GraspEntryPoseSet",
    "GraspEntrySearch",
    "find_grasp_entry_poses",
]
