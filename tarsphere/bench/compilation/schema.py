"""Private grounded records used by deterministic episode builders."""

from __future__ import annotations

from typing import Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field

from tarsphere.bench.spatial.physical_task.schema import (
    GraspAction,
    PlaceAction,
)


Point3 = Tuple[float, float, float]


class StrictModel(BaseModel):
    """Immutable current-release compilation schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class EpisodeObject(StrictModel):
    """One task-relevant scene object retained for runtime and certification."""

    object_id: str = Field(
        min_length=1,
        description="Stable identifier from the grounded GRScene metadata.",
    )
    name: Optional[str] = Field(
        default=None,
        description=(
            "Reviewed natural name when the object participates in language "
            "or perception evaluation; physical-only obstacles have no name."
        ),
    )
    category: str = Field(
        min_length=1,
        description="Grounded object category declared by the scene asset.",
    )
    room_id: str = Field(
        min_length=1,
        description="Grounded semantic room containing the object.",
    )
    position_xyz: Point3 = Field(
        description="Object reference position in scene world metres.",
    )
    bounds_min_xyz: Point3 = Field(
        description="Minimum corner of the object world-space bounding box.",
    )
    bounds_max_xyz: Point3 = Field(
        description="Maximum corner of the object world-space bounding box.",
    )
    runtime_prim: str = Field(
        min_length=1,
        description="Isaac runtime prim used for physical evidence and actions.",
    )
    interaction: Optional[str] = Field(
        default=None,
        description="Grounded physical interaction kind when one is declared.",
    )


class CompiledPhysicalTask(StrictModel):
    """Shared nav–grasp–transport task executed by every H level."""

    robot_start_yaw_rad: float = Field(
        description="Shared G1 root yaw at the first navigation point.",
    )
    target_object_id: str = Field(
        min_length=1,
        description="Object that must be grasped, retained, and transported.",
    )
    target_support_object_id: Optional[str] = Field(
        default=None,
        description="Object supporting the target at the observation event.",
    )
    distractor_object_ids: Tuple[str, ...] = Field(
        default=(),
        description="Grounded alternatives that make clarification necessary.",
    )
    initial_destination_object_id: str = Field(
        min_length=1,
        description="Destination active before the adaptation milestone.",
    )
    final_destination_object_id: str = Field(
        min_length=1,
        description="Destination visited after adaptation and before handover.",
    )
    goal_relation: Literal["on", "in"] = Field(
        description="Certified set-down relation used only by diagnostic tasks.",
    )
    grasp: GraspAction = Field(
        description="Near-field stance, hand target, and retreat for grasp.",
    )
    place: PlaceAction = Field(
        description="Optional diagnostic set-down geometry, not a Golden goal.",
    )


__all__ = [
    "CompiledPhysicalTask",
    "EpisodeObject",
    "StrictModel",
]
