"""Typed nav–grasp–transport–place route contracts."""

from __future__ import annotations

from collections import Counter
import math
from typing import Annotated, Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


PHYSICAL_TASK_ROUTE_SCHEMA_VERSION = "tarsphere.physical_task_route.v1"

Identifier = Annotated[
    str,
    Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase name for one route point.",
    ),
]
Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


def _same_xy(first: Point2, second: Point2) -> bool:
    return math.dist(first, second) <= 1e-9


class StrictModel(BaseModel):
    """Immutable current-release spatial schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class NavigationPoint(StrictModel):
    """One meaningful control point on a Robot navigation route."""

    point_id: Identifier = Field(
        description=(
            "Task-readable name such as robot_start, object_observation, or "
            "initial_destination_commitment."
        ),
    )
    position_xy: Point2 = Field(
        description="Robot base position in scene world metres.",
    )

    @model_validator(mode="after")
    def validate_position(self) -> "NavigationPoint":
        if not all(math.isfinite(value) for value in self.position_xy):
            raise ValueError("navigation point position must be finite")
        return self


class NavigationRoute(StrictModel):
    """Ordered control points for one collision-checked navigation leg."""

    points: Tuple[NavigationPoint, ...] = Field(
        min_length=2,
        description=(
            "Ordered sparse control points; static occupancy planning fills "
            "the collision-free path between each adjacent pair."
        ),
    )

    @model_validator(mode="after")
    def validate_points(self) -> "NavigationRoute":
        duplicates = sorted(
            point_id
            for point_id, count in Counter(
                point.point_id for point in self.points
            ).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate navigation point identifiers: {duplicates}")
        for first, second in zip(self.points, self.points[1:]):
            if _same_xy(first.position_xy, second.position_xy):
                raise ValueError(
                    "adjacent navigation points must have distinct positions"
                )
        return self


class GraspAction(StrictModel):
    """Near-field Robot geometry for one physical object grasp."""

    target_object_id: str = Field(
        min_length=1,
        description="Grounded pickable object that must become attached.",
    )
    navigation_approach_xy: Point2 = Field(
        description=("Last occupancy-safe base position before precise manipulation."),
    )
    robot_stance_xy: Point2 = Field(
        description=(
            "Near-field base position used for grasp IK; this is verified in "
            "Isaac rather than treated as a normal navigation waypoint."
        ),
    )
    robot_yaw_rad: float = Field(
        ge=-math.pi,
        le=math.pi,
        description="Robot yaw that faces the grasp target.",
    )
    retreat_xy: Point2 = Field(
        description=(
            "Occupancy-safe base position reached after attachment and arm "
            "retraction, before loaded navigation begins."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "GraspAction":
        values = (
            *self.navigation_approach_xy,
            *self.robot_stance_xy,
            self.robot_yaw_rad,
            *self.retreat_xy,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("grasp geometry must be finite")
        return self


class PlaceAction(StrictModel):
    """Near-field Robot geometry for stable release onto a destination."""

    destination_object_id: str = Field(
        min_length=1,
        description="Grounded surface that must support the released object.",
    )
    navigation_approach_xy: Point2 = Field(
        description=("Last occupancy-safe base position before precise placement."),
    )
    robot_stance_xy: Point2 = Field(
        description=("Near-field base position used for placement IK and release."),
    )
    robot_yaw_rad: float = Field(
        ge=-math.pi,
        le=math.pi,
        description="Robot yaw that faces the placement surface target.",
    )
    support_surface_xyz: Point3 = Field(
        description=(
            "World-space point on the clear destination surface beneath the "
            "released object."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "PlaceAction":
        values = (
            *self.navigation_approach_xy,
            *self.robot_stance_xy,
            self.robot_yaw_rad,
            *self.support_surface_xyz,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("placement geometry must be finite")
        return self


class PhysicalTaskRoute(StrictModel):
    """One shared nav–grasp–transport–place route for an episode task line."""

    schema_version: Literal["tarsphere.physical_task_route.v1"] = Field(
        default=PHYSICAL_TASK_ROUTE_SCHEMA_VERSION,
        description="Version of the physical-task route contract.",
    )
    scene_id: str = Field(
        min_length=1,
        description="Grounded scene containing every route point and object.",
    )
    robot_start_yaw_rad: float = Field(
        ge=-math.pi,
        le=math.pi,
        description=(
            "Shared Robot yaw at the first navigation point before any "
            "episode-specific Human behavior begins."
        ),
    )
    target_observation_point_id: Identifier = Field(
        description=(
            "Point on navigate_to_target where the task object first becomes "
            "naturally observable during normal approach."
        ),
    )
    destination_commitment_point_id: Identifier = Field(
        description=(
            "Point on transport_to_destination reached only after the Robot "
            "has behaviorally committed to the initially public destination."
        ),
    )
    navigate_to_target: NavigationRoute = Field(
        description="Unladen Robot route from the shared start to the object.",
    )
    grasp: GraspAction = Field(
        description="Precise approach, attachment target, and safe retreat.",
    )
    transport_to_destination: NavigationRoute = Field(
        description=(
            "Loaded Robot route from the grasp retreat through the committed "
            "task trajectory to the final destination."
        ),
    )
    place: PlaceAction = Field(
        description="Precise destination approach and stable release target.",
    )

    @model_validator(mode="after")
    def validate_continuity(self) -> "PhysicalTaskRoute":
        target_end = self.navigate_to_target.points[-1].position_xy
        if not _same_xy(target_end, self.grasp.navigation_approach_xy):
            raise ValueError(
                "navigate_to_target must end at grasp.navigation_approach_xy"
            )
        transport_start = self.transport_to_destination.points[0].position_xy
        if not _same_xy(transport_start, self.grasp.retreat_xy):
            raise ValueError("transport_to_destination must start at grasp.retreat_xy")
        transport_end = self.transport_to_destination.points[-1].position_xy
        if not _same_xy(transport_end, self.place.navigation_approach_xy):
            raise ValueError(
                "transport_to_destination must end at place.navigation_approach_xy"
            )
        point_ids = tuple(
            point.point_id
            for route in (
                self.navigate_to_target,
                self.transport_to_destination,
            )
            for point in route.points
        )
        duplicates = sorted(
            point_id for point_id, count in Counter(point_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(
                f"navigation point identifiers must be route-unique: {duplicates}"
            )
        navigate_ids = tuple(point.point_id for point in self.navigate_to_target.points)
        transport_ids = tuple(
            point.point_id for point in self.transport_to_destination.points
        )
        if self.target_observation_point_id not in navigate_ids:
            raise ValueError(
                "target_observation_point_id must reference navigate_to_target"
            )
        if self.destination_commitment_point_id not in transport_ids:
            raise ValueError(
                "destination_commitment_point_id must reference "
                "transport_to_destination"
            )
        if self.target_observation_point_id == navigate_ids[0]:
            raise ValueError(
                "target observation must occur after the shared Robot start"
            )
        if self.destination_commitment_point_id in {
            transport_ids[0],
            transport_ids[-1],
        }:
            raise ValueError(
                "destination commitment must occur after grasp retreat and "
                "before final destination approach"
            )
        return self

    def task_stage_point_id(self, task_stage: str) -> str:
        """Return the physical milestone that grounds one task stage."""
        try:
            return {
                "task_start": self.navigate_to_target.points[0].point_id,
                "information_gathering": (self.navigate_to_target.points[0].point_id),
                "target_approach": self.target_observation_point_id,
                "before_grasp": self.navigate_to_target.points[-1].point_id,
                "transport": self.destination_commitment_point_id,
                "destination_approach": (
                    self.transport_to_destination.points[-1].point_id
                ),
                "task_complete": (self.transport_to_destination.points[-1].point_id),
            }[task_stage]
        except KeyError as exc:
            raise ValueError(f"unknown physical task stage: {task_stage!r}") from exc


class StaticNavigationSegment(StrictModel):
    """One A*-resolved segment between adjacent authored control points."""

    route_name: Literal[
        "navigate_to_target",
        "transport_to_destination",
    ] = Field(
        description="Physical-task navigation leg containing this segment.",
    )
    start_point_id: Identifier = Field(
        description="Authored control point at the segment start.",
    )
    end_point_id: Identifier = Field(
        description="Authored control point at the segment end.",
    )
    path_xy: Tuple[Point2, ...] = Field(
        min_length=2,
        description=(
            "Collision-checked world-space polyline including both authored "
            "segment endpoints."
        ),
    )
    path_length_m: float = Field(
        gt=0.0,
        description="Length of the resolved segment polyline in metres.",
    )


class StaticNavigationEvidence(StrictModel):
    """Offline occupancy evidence for all navigation legs in one route."""

    scene_id: str = Field(
        min_length=1,
        description="Scene whose official occupancy layer was used.",
    )
    occupancy: str = Field(
        min_length=1,
        description="Asset-relative official occupancy file.",
    )
    robot_clearance_radius_m: float = Field(
        gt=0.0,
        description="G1 base-clearance radius applied during path search.",
    )
    segments: Tuple[StaticNavigationSegment, ...] = Field(
        min_length=1,
        description="Resolved evidence for every adjacent control-point pair.",
    )
    total_path_length_m: float = Field(
        gt=0.0,
        description="Sum of all unladen and loaded navigation segment lengths.",
    )


__all__ = [
    "PHYSICAL_TASK_ROUTE_SCHEMA_VERSION",
    "GraspAction",
    "NavigationPoint",
    "NavigationRoute",
    "PhysicalTaskRoute",
    "PlaceAction",
    "StaticNavigationEvidence",
    "StaticNavigationSegment",
]
