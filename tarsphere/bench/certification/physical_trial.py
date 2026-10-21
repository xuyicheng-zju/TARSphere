"""Resolve one explicit geometry choice into a standalone physical trial."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Iterable, Literal, Optional, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.certification.task_geometry import (
    TaskGeometryCandidates,
)
from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.bench.spatial.physical_task.navigation import (
    resolve_static_navigation,
)
from tarsphere.bench.spatial.physical_task.schema import (
    GraspAction,
    NavigationPoint,
    NavigationRoute,
    PlaceAction,
    StaticNavigationEvidence,
)


Point2 = tuple[float, float]
_POINT_MATCH_TOLERANCE_M = 1e-6
_T = TypeVar("_T")


class StrictModel(BaseModel):
    """Immutable current-release physical-trial schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PhysicalTrialSelection(StrictModel):
    """Identify one bounded trial within a task-geometry candidate set."""

    target_approach_path_index: int = Field(
        ge=0,
        description=(
            "Official target approach-path index whose navigation endpoint "
            "and grasp-entry set are used by this trial."
        ),
    )
    grasp_entry_index: int = Field(
        ge=0,
        description=(
            "Zero-based pose index within the selected target path’s "
            "deterministically ordered grasp-entry set."
        ),
    )
    placement_patch_index: int = Field(
        ge=0,
        description=(
            "Zero-based index of the live-probed destination surface patch "
            "used by this trial."
        ),
    )
    destination_approach_path_index: int = Field(
        ge=0,
        description=(
            "Official destination approach-path index used to reach the "
            "selected placement patch."
        ),
    )
    placement_stance_index: int = Field(
        ge=0,
        description=(
            "Zero-based pose index within the selected patch/path stance set."
        ),
    )


class PhysicalTrial(StrictModel):
    """One Human-free nav–grasp–transport–place rollout candidate."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene in which the complete trial must execute.",
    )
    scene_server_id: str = Field(
        min_length=1,
        description=(
            "Unique baked Scene Server process that produced the live "
            "geometry used by this trial and must also execute its rollout."
        ),
    )
    scene_server_git_sha: str = Field(
        min_length=1,
        description=(
            "Scene Server revision that produced the live source and "
            "destination surface evidence used by this trial."
        ),
    )
    selection: PhysicalTrialSelection = Field(
        description=(
            "Exact bounded choice within the preceding task-geometry candidate set."
        ),
    )
    goal_relation: Literal["on"] = Field(
        description="Final physical relation the release must establish.",
    )
    source_support_object_id: str = Field(
        min_length=1,
        description=("Live-certified object supporting the target before grasp."),
    )
    navigate_to_target: NavigationRoute = Field(
        description=(
            "Unladen route from the official Robot start to the selected "
            "target navigation approach."
        ),
    )
    grasp: GraspAction = Field(
        description=(
            "Selected occupancy entry and retreat geometry for live grasp "
            "reach and attachment testing."
        ),
    )
    transport_to_destination: NavigationRoute = Field(
        description=(
            "Loaded route from the grasp retreat to the selected destination "
            "navigation approach."
        ),
    )
    place: PlaceAction = Field(
        description=(
            "Selected placement stance and live-probed support point for "
            "release testing."
        ),
    )
    navigation_evidence: StaticNavigationEvidence = Field(
        description=(
            "Exact G1-footprint occupancy evidence for both navigation legs; "
            "its total length is the trial’s navigation-efficiency reference."
        ),
    )

    @model_validator(mode="after")
    def validate_trial(self) -> "PhysicalTrial":
        if self.scene_id != self.navigation_evidence.scene_id:
            raise ValueError(
                "physical trial and navigation evidence identify different scenes"
            )
        if self.grasp.target_object_id == self.place.destination_object_id:
            raise ValueError("physical trial target and destination must be distinct")
        if not _same_xy(
            self.navigate_to_target.points[-1].position_xy,
            self.grasp.navigation_approach_xy,
        ):
            raise ValueError("navigate_to_target must end at the grasp approach")
        if not _same_xy(
            self.transport_to_destination.points[0].position_xy,
            self.grasp.retreat_xy,
        ):
            raise ValueError("transport_to_destination must begin at the grasp retreat")
        if not _same_xy(
            self.transport_to_destination.points[-1].position_xy,
            self.place.navigation_approach_xy,
        ):
            raise ValueError("transport_to_destination must end at the place approach")
        expected_segments = tuple(
            (
                route_name,
                start.point_id,
                end.point_id,
            )
            for route_name, route in (
                ("navigate_to_target", self.navigate_to_target),
                (
                    "transport_to_destination",
                    self.transport_to_destination,
                ),
            )
            for start, end in zip(route.points, route.points[1:])
        )
        actual_segments = tuple(
            (
                segment.route_name,
                segment.start_point_id,
                segment.end_point_id,
            )
            for segment in self.navigation_evidence.segments
        )
        if actual_segments != expected_segments:
            raise ValueError(
                "navigation evidence does not cover the physical trial routes"
            )
        return self

    @property
    def robot_start_yaw_rad(self) -> float:
        """Return the first collision-free path heading for Robot spawn."""
        first_segment = self.navigation_evidence.segments[0]
        start = first_segment.path_xy[0]
        for point in first_segment.path_xy[1:]:
            if not _same_xy(start, point):
                return math.atan2(
                    point[1] - start[1],
                    point[0] - start[0],
                )
        raise ValueError(
            "physical trial navigation evidence has no forward start segment"
        )


class PhysicalTrialResolutionError(ValueError):
    """Raised when an explicit candidate choice cannot form a physical trial."""


@dataclass(frozen=True)
class _TrialNavigation:
    scene_id: str
    navigate_to_target: NavigationRoute
    transport_to_destination: NavigationRoute


def resolve_physical_trial(
    scene: GroundedScene,
    candidates: TaskGeometryCandidates,
    selection: PhysicalTrialSelection,
    *,
    assets_root: Optional[Path | str] = None,
) -> PhysicalTrial:
    """Resolve one explicit candidate tuple without claiming rollout success."""
    if scene.scene_id != candidates.scene_id:
        raise ValueError("physical-trial candidates and grounded scene must match")

    target_approach = _single(
        (
            approach
            for approach in candidates.target_approaches
            if approach.source_path_index == selection.target_approach_path_index
        ),
        label=(f"target approach path {selection.target_approach_path_index}"),
    )
    grasp_entries = _single(
        (
            entry_set
            for entry_set in candidates.grasp_entry_sets
            if entry_set.target_approach_path_index
            == selection.target_approach_path_index
        ),
        label=(
            f"grasp-entry set for target path {selection.target_approach_path_index}"
        ),
    )
    grasp_entry = _indexed(
        grasp_entries.poses,
        selection.grasp_entry_index,
        label="grasp entry",
    )
    placement_patch = _indexed(
        candidates.placement_patches,
        selection.placement_patch_index,
        label="placement patch",
    )
    placement_stances = _single(
        (
            stance_set
            for stance_set in candidates.placement_stance_sets
            if (
                stance_set.surface_center_xyz == placement_patch.center_xyz
                and stance_set.destination_approach_path_index
                == selection.destination_approach_path_index
            )
        ),
        label=(
            "placement stance set for patch "
            f"{selection.placement_patch_index} and destination path "
            f"{selection.destination_approach_path_index}"
        ),
    )
    placement_stance = _indexed(
        placement_stances.candidates,
        selection.placement_stance_index,
        label="placement stance",
    )

    navigate_to_target = NavigationRoute(
        points=(
            NavigationPoint(
                point_id="robot_start",
                position_xy=target_approach.robot_start_xy,
            ),
            NavigationPoint(
                point_id="target_approach",
                position_xy=target_approach.navigation_approach_xy,
            ),
        )
    )
    grasp = GraspAction(
        target_object_id=candidates.target_object_id,
        navigation_approach_xy=grasp_entries.navigation_approach_xy,
        robot_stance_xy=grasp_entry.robot_entry_xy,
        robot_yaw_rad=grasp_entry.robot_yaw_rad,
        retreat_xy=grasp_entries.navigation_approach_xy,
    )
    transport_to_destination = NavigationRoute(
        points=(
            NavigationPoint(
                point_id="grasp_retreat",
                position_xy=grasp.retreat_xy,
            ),
            NavigationPoint(
                point_id="destination_approach",
                position_xy=placement_stances.destination_approach_xy,
            ),
        )
    )
    place = PlaceAction(
        destination_object_id=candidates.destination_object_id,
        navigation_approach_xy=(placement_stances.destination_approach_xy),
        robot_stance_xy=placement_stance.robot_stance_xy,
        robot_yaw_rad=placement_stance.robot_yaw_rad,
        support_surface_xyz=placement_patch.center_xyz,
    )
    navigation_evidence = resolve_static_navigation(
        _TrialNavigation(
            scene_id=scene.scene_id,
            navigate_to_target=navigate_to_target,
            transport_to_destination=transport_to_destination,
        ),
        scene,
        assets_root=assets_root,
    )
    return PhysicalTrial(
        scene_id=scene.scene_id,
        scene_server_id=candidates.scene_server_id,
        scene_server_git_sha=candidates.scene_server_git_sha,
        selection=selection,
        goal_relation=candidates.goal_relation,
        source_support_object_id=(candidates.source_support.support_object_id),
        navigate_to_target=navigate_to_target,
        grasp=grasp,
        transport_to_destination=transport_to_destination,
        place=place,
        navigation_evidence=navigation_evidence,
    )


def _same_xy(first: Point2, second: Point2) -> bool:
    return math.dist(first, second) <= _POINT_MATCH_TOLERANCE_M


def _single(values: Iterable[_T], *, label: str) -> _T:
    matches = tuple(values)
    if len(matches) != 1:
        raise PhysicalTrialResolutionError(
            f"Expected exactly one {label}; found {len(matches)}"
        )
    return matches[0]


def _indexed(values: Iterable[_T], index: int, *, label: str) -> _T:
    items = tuple(values)
    try:
        return items[index]
    except IndexError as exc:
        raise PhysicalTrialResolutionError(
            f"{label} index {index} is outside {len(items)} candidates"
        ) from exc


__all__ = [
    "PhysicalTrial",
    "PhysicalTrialResolutionError",
    "PhysicalTrialSelection",
    "resolve_physical_trial",
]
