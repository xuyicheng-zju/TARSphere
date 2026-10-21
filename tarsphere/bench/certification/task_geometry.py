"""Resolve one semantic task proposal into physically grounded candidates."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.task_candidates import (
    PhysicalTaskShortlist,
)
from tarsphere.bench.certification.grasp_entry import (
    GraspEntryPoseSet,
    find_grasp_entry_poses,
)
from tarsphere.bench.certification.placement import (
    PlacementStanceResolutionError,
    PlacementStanceSet,
    PlacementSurfacePatch,
    find_placement_stance_candidates,
)
from tarsphere.bench.certification.scene_client import (
    SceneServerClient,
    SourceSupportProbeRejected,
)
from tarsphere.bench.certification.source_support import (
    SourceSupportEvidence,
)
from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.spatial.physical_task.objects import (
    discover_placement_destinations,
)
from tarsphere.bench.spatial.physical_task.target_approach import (
    TargetApproachResolutionError,
    TargetApproachRoute,
    resolve_target_approach_route,
)
from tarsphere.bench.spatial.physical_task.target_screening import (
    ManipulationTargetScreening,
)


class StrictModel(BaseModel):
    """Immutable current-release task-geometry candidate base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class TaskGeometryCandidates(StrictModel):
    """Live and occupancy evidence preceding grasp/place rollout trials."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene shared by every candidate and live probe.",
    )
    proposal_index: int = Field(
        ge=0,
        description="Ranked shortlist proposal resolved by this result.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded pickable selected by the semantic proposal.",
    )
    destination_object_id: str = Field(
        min_length=1,
        description="Grounded final destination selected by the proposal.",
    )
    goal_relation: Literal["on"] = Field(
        description=(
            "Final target–destination relation whose live placement geometry "
            "these candidates are intended to realize."
        ),
    )
    scene_server_id: str = Field(
        min_length=1,
        description=(
            "Unique baked Scene Server process shared by both live geometry probes."
        ),
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description=(
            "Repository revision loaded by the Scene Server that produced "
            "both live geometry probes."
        ),
    )
    source_support: SourceSupportEvidence = Field(
        description="Live five-ray evidence for the target initial support.",
    )
    examined_target_path_indexes: Tuple[int, ...] = Field(
        min_length=1,
        description=(
            "All official target path indexes examined for exact G1 approach "
            "and grasp-entry candidates."
        ),
    )
    target_approaches: Tuple[TargetApproachRoute, ...] = Field(
        min_length=1,
        description=(
            "Exact G1-clear target routes that also retain at least one "
            "near-field grasp-entry pose."
        ),
    )
    grasp_entry_sets: Tuple[GraspEntryPoseSet, ...] = Field(
        min_length=1,
        description=(
            "Target approaches retaining at least one occupancy-reachable "
            "entry pose; none yet proves arm reach or attachment."
        ),
    )
    placement_patches: Tuple[PlacementSurfacePatch, ...] = Field(
        min_length=1,
        description=(
            "Live collision patches large enough for the current target footprint."
        ),
    )
    examined_destination_path_indexes: Tuple[int, ...] = Field(
        min_length=1,
        description=(
            "All official destination path indexes examined against every "
            "live placement patch."
        ),
    )
    placement_stance_sets: Tuple[PlacementStanceSet, ...] = Field(
        min_length=1,
        description=(
            "Patch/path pairs retaining at least one occupancy-reachable G1 "
            "stance; none yet proves carried-object IK, release, or stability."
        ),
    )

    @model_validator(mode="after")
    def validate_bindings(self) -> "TaskGeometryCandidates":
        if self.target_object_id == self.destination_object_id:
            raise ValueError("task geometry target and destination must differ")
        if self.source_support.target_object_id != self.target_object_id:
            raise ValueError("source-support evidence belongs to a different target")
        if len(set(self.examined_target_path_indexes)) != len(
            self.examined_target_path_indexes
        ):
            raise ValueError("examined target path indexes must be unique")
        if any(index < 0 for index in self.examined_target_path_indexes):
            raise ValueError("examined target path indexes must be non-negative")
        if len(set(self.examined_destination_path_indexes)) != len(
            self.examined_destination_path_indexes
        ):
            raise ValueError("examined destination path indexes must be unique")
        if any(index < 0 for index in self.examined_destination_path_indexes):
            raise ValueError("examined destination path indexes must be non-negative")
        target_indexes = set(self.examined_target_path_indexes)
        destination_indexes = set(self.examined_destination_path_indexes)
        approach_indexes = tuple(
            approach.source_path_index for approach in self.target_approaches
        )
        if len(set(approach_indexes)) != len(approach_indexes) or any(
            approach.scene_id != self.scene_id
            or approach.target_object_id != self.target_object_id
            or approach.source_path_index not in target_indexes
            for approach in self.target_approaches
        ):
            raise ValueError("target approaches are not unique task bindings")
        grasp_indexes = tuple(
            entry.target_approach_path_index for entry in self.grasp_entry_sets
        )
        if len(set(grasp_indexes)) != len(grasp_indexes):
            raise ValueError("grasp-entry sets must identify unique target paths")
        if set(grasp_indexes) != set(approach_indexes):
            raise ValueError(
                "target approaches and grasp-entry sets must cover the same "
                "official paths"
            )
        if any(
            entry.scene_id != self.scene_id
            or entry.target_object_id != self.target_object_id
            or entry.source_support_object_id != self.source_support.support_object_id
            or entry.target_approach_path_index not in target_indexes
            or not entry.poses
            for entry in self.grasp_entry_sets
        ):
            raise ValueError("grasp-entry sets are not bound to this task")
        patch_centers = {
            patch.center_xyz
            for patch in self.placement_patches
            if (
                patch.target_object_id == self.target_object_id
                and patch.destination_object_id == self.destination_object_id
            )
        }
        if len(patch_centers) != len(self.placement_patches):
            raise ValueError("placement patches are not unique task bindings")
        stance_bindings = tuple(
            (
                stance.surface_center_xyz,
                stance.destination_approach_path_index,
            )
            for stance in self.placement_stance_sets
        )
        if len(set(stance_bindings)) != len(stance_bindings):
            raise ValueError(
                "placement stance sets must identify unique patch/path pairs"
            )
        if any(
            stance.scene_id != self.scene_id
            or stance.target_object_id != self.target_object_id
            or stance.destination_object_id != self.destination_object_id
            or stance.surface_center_xyz not in patch_centers
            or stance.destination_approach_path_index not in destination_indexes
            or not stance.candidates
            for stance in self.placement_stance_sets
        ):
            raise ValueError("placement stance sets are not bound to this task")
        return self


class TaskGeometryResolutionError(ValueError):
    """Raised when a proposal has no physically grounded candidate geometry."""


def _validate_shortlist_against_screening(
    scene: GroundedScene,
    shortlist: PhysicalTaskShortlist,
    target_screening: ManipulationTargetScreening,
) -> None:
    """Prove every semantic proposal belongs to the offline authority.

    Physical-task selection once routed this check through an LLM-facing query
    helper. The live geometry resolver is now the sole consumer, so it validates
    the same target and destination facts directly instead of retaining an
    authoring tool with no Agent caller.
    """
    if target_screening.scene_id != scene.scene_id:
        raise ValueError("target screening and grounded scene must match")
    for proposal in shortlist.proposals:
        target = target_screening.require(proposal.target_object_id).target
        destination_ids = {
            candidate.destination_object_id
            for candidate in discover_placement_destinations(
                scene,
                target,
            ).destinations
        }
        if proposal.destination_object_id not in destination_ids:
            raise ValueError(
                f"Unknown destination {proposal.destination_object_id!r} "
                f"for target {proposal.target_object_id!r}"
            )


def resolve_task_geometry_candidates(
    scene: GroundedScene,
    shortlist: PhysicalTaskShortlist,
    target_screening: ManipulationTargetScreening,
    *,
    proposal_index: int,
    scene_server: SceneServerClient,
    robot_start_xy: Optional[Tuple[float, float]] = None,
    assets_root: Optional[Path | str] = None,
) -> TaskGeometryCandidates:
    """Run live surface gates and enumerate exact G1 geometry candidates.

    This stage intentionally stops before Robot motion. It proves the target's
    current support, target and destination navigation approach candidates,
    live destination patches, and occupancy-reachable near-field base poses.
    Arm motion, FixedJoint attachment, loaded transport, release, and settled
    placement remain later rollout gates. Human and task events are authored
    only after the physical task is certified.
    """
    if shortlist.scene_id != scene.scene_id:
        raise ValueError("physical-task shortlist and grounded scene must match")
    _validate_shortlist_against_screening(
        scene,
        shortlist,
        target_screening,
    )
    if proposal_index < 0:
        raise ValueError(f"proposal_index is outside the shortlist: {proposal_index}")
    try:
        proposal = shortlist.proposals[proposal_index]
    except IndexError as exc:
        raise ValueError(
            f"proposal_index is outside the shortlist: {proposal_index}"
        ) from exc
    target = target_screening.require(proposal.target_object_id).target
    objects = {obj.object_id: obj for obj in scene.objects}
    destination = objects[proposal.destination_object_id]

    try:
        source_result = scene_server.probe_source_support(
            scene_id=scene.scene_id,
            target_object_id=proposal.target_object_id,
        )
    except SourceSupportProbeRejected as exc:
        raise TaskGeometryResolutionError(
            f"Live source-support contract rejected the target: {exc}"
        ) from exc
    placement_result = scene_server.probe_placement_surfaces(
        scene_id=scene.scene_id,
        target_object_id=proposal.target_object_id,
        destination_object_id=proposal.destination_object_id,
    )
    if source_result.scene_server_id != placement_result.scene_server_id:
        raise TaskGeometryResolutionError(
            "Live geometry probes came from different Scene Server instances"
        )
    if source_result.scene_server_git_sha != placement_result.scene_server_git_sha:
        raise TaskGeometryResolutionError(
            "Live geometry probes came from different Scene Server revisions"
        )
    if not placement_result.patches:
        raise TaskGeometryResolutionError(
            "Live destination probe found no supported placement patch"
        )
    _require_live_patch_footprint(
        source_result.evidence,
        placement_result.patches,
        edge_margin_m=placement_result.search.edge_margin_m,
    )

    target_path_indexes = tuple(sorted(target.approach_path_indexes))
    target_approaches = []
    grasp_entries = []
    for path_index in target_path_indexes:
        try:
            target_approach = resolve_target_approach_route(
                scene,
                target,
                approach_path_index=path_index,
                robot_start_xy=robot_start_xy,
                assets_root=assets_root,
            )
        except TargetApproachResolutionError:
            continue
        entries = find_grasp_entry_poses(
            scene,
            target,
            target_approach,
            source_result.evidence,
            assets_root=assets_root,
        )
        if entries.poses:
            target_approaches.append(target_approach)
            grasp_entries.append(entries)
    if not grasp_entries:
        raise TaskGeometryResolutionError(
            "No official target path yields a G1-clear grasp entry"
        )

    destination_path_indexes = tuple(
        sorted(
            path.path_index
            for path in scene.object_approach_paths
            if path.object_id == proposal.destination_object_id
        )
    )
    if not destination_path_indexes:
        raise TaskGeometryResolutionError(
            "Selected destination has no official approach path"
        )
    placement_stances = []
    for patch in placement_result.patches:
        for path_index in destination_path_indexes:
            try:
                stance_set = find_placement_stance_candidates(
                    scene,
                    destination,
                    patch,
                    approach_path_index=path_index,
                    assets_root=assets_root,
                )
            except PlacementStanceResolutionError:
                continue
            if stance_set.candidates:
                placement_stances.append(stance_set)
    if not placement_stances:
        raise TaskGeometryResolutionError(
            "No live placement patch has a G1-clear destination stance"
        )

    return TaskGeometryCandidates(
        scene_id=scene.scene_id,
        proposal_index=proposal_index,
        target_object_id=proposal.target_object_id,
        destination_object_id=proposal.destination_object_id,
        goal_relation=proposal.goal_relation,
        scene_server_id=source_result.scene_server_id,
        scene_server_git_sha=source_result.scene_server_git_sha,
        source_support=source_result.evidence,
        examined_target_path_indexes=target_path_indexes,
        target_approaches=tuple(target_approaches),
        grasp_entry_sets=tuple(grasp_entries),
        placement_patches=placement_result.patches,
        examined_destination_path_indexes=destination_path_indexes,
        placement_stance_sets=tuple(placement_stances),
    )


def _require_live_patch_footprint(
    source_support: SourceSupportEvidence,
    patches: Tuple[PlacementSurfacePatch, ...],
    *,
    edge_margin_m: float,
) -> None:
    lower = source_support.target_bounds_min_xyz
    upper = source_support.target_bounds_max_xyz
    expected = (
        upper[0] - lower[0] + 2.0 * edge_margin_m,
        upper[1] - lower[1] + 2.0 * edge_margin_m,
    )
    if not all(
        math.isclose(
            patch.supported_footprint_xy_m[axis],
            expected[axis],
            abs_tol=1e-6,
        )
        for patch in patches
        for axis in range(2)
    ):
        raise TaskGeometryResolutionError(
            "Placement patches were not conditioned on the source probe live "
            "target footprint"
        )


__all__ = [
    "TaskGeometryCandidates",
    "TaskGeometryResolutionError",
    "resolve_task_geometry_candidates",
]
