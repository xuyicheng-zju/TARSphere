"""Strict output schema for deterministic scene grounding."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field


GROUNDING_SCHEMA_VERSION = "tarsphere.grounded_scene.v1"

Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


class SceneSources(BaseModel):
    """Canonical files from which object and navigation facts were read."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_usd: str = Field(
        description="Asset-relative runtime USD for the grounded scene.",
    )
    object_metadata: str = Field(
        description="Asset-relative captioned object metadata source.",
    )
    object_paths: str = Field(
        description="Asset-relative official object-approach path source.",
    )
    interactive_objects: str = Field(
        description="Asset-relative physical interaction declaration.",
    )


class SceneMap(BaseModel):
    """BEV and occupancy references sharing one world projection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rgb_bev: str = Field(
        description="Asset-relative unannotated RGB bird-eye-view image.",
    )
    room_bev: str = Field(
        description="Asset-relative room-labelled RGB bird-eye-view image.",
    )
    projection: str = Field(
        description="Asset-relative world-to-BEV projection metadata.",
    )
    room_annotations: str = Field(
        description="Asset-relative semantic room polygons and labels.",
    )
    occupancy: str = Field(
        description="Asset-relative official navigation occupancy array.",
    )
    resolution: int = Field(
        gt=0,
        description="Width and height of the square BEV images in pixels.",
    )
    bounds_min_xy: Point2 = Field(
        description="Minimum world XY represented by the BEV projection.",
    )
    bounds_max_xy: Point2 = Field(
        description="Maximum world XY represented by the BEV projection.",
    )


class SceneRoom(BaseModel):
    """One semantically named room region."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    room_id: str
    label: str
    room_type: str
    source_room_id: Optional[str]
    polygon_xy: Tuple[Point2, ...] = Field(min_length=3)
    label_xy: Point2
    area_m2: float = Field(gt=0.0)


class GroundedObject(BaseModel):
    """One semantic scene object available to later authoring stages."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: str
    category: str
    room_id: str
    position_xyz: Point3
    bounds_min_xyz: Point3
    bounds_max_xyz: Point3
    caption: Optional[str]
    source_prim: Optional[str]
    runtime_prim: Optional[str]
    interaction: Optional[Literal["articulated", "pickable"]]


class ObjectApproachPath(BaseModel):
    """Compact reference to one official object-conditioned path."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    object_id: str
    path_index: int = Field(ge=0)
    start_xy: Point2
    initial_heading_rad: float = Field(
        description=("World-frame yaw from the first official waypoint to the second."),
    )
    approach_xy: Point2
    distance_m: float = Field(ge=0.0)
    waypoint_count: int = Field(gt=0)


class GroundedScene(BaseModel):
    """Minimal deterministic scene facts consumed by benchmark tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tarsphere.grounded_scene.v1"] = GROUNDING_SCHEMA_VERSION
    source_id: str
    scene_id: str
    scene_scale: Point3 = Field(
        description=(
            "Isotropic scale applied when the authored scene is referenced "
            "into the benchmark metre-based world."
        ),
    )
    sources: SceneSources
    map: SceneMap
    rooms: Tuple[SceneRoom, ...]
    objects: Tuple[GroundedObject, ...]
    object_approach_paths: Tuple[ObjectApproachPath, ...]


__all__ = [
    "GROUNDING_SCHEMA_VERSION",
    "GroundedObject",
    "GroundedScene",
    "ObjectApproachPath",
    "SceneMap",
    "SceneRoom",
    "SceneSources",
]
