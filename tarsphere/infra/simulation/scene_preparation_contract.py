"""Reversible scene edits that make one physical grasp target live-ready."""

from __future__ import annotations

from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    """Immutable current-release scene-preparation schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class RelocatedPickable(StrictModel):
    """Seat one pickable on a flat support, optionally moving it in XY."""

    object_id: str = Field(
        min_length=1,
        description="Grounded pickable whose PhysX body is vertically aligned.",
    )
    runtime_prim: str = Field(
        min_length=1,
        description="Exact USD prim whose world bounds are measured and shifted.",
    )
    support_surface_z_m: float = Field(
        description=(
            "Requested world-space support height in metres. The runtime "
            "aligns the pickable bottom to this Z before attachment start()."
        ),
    )
    support_xy: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Requested world-space XY for the pickable AABB centre. When "
            "omitted, only vertical alignment is applied."
        ),
    )


class PhysicalScenePreparation(StrictModel):
    """Hide unstable intermediates and seat the target on its flat support."""

    hidden_runtime_prims: Tuple[str, ...] = Field(
        description=(
            "Native scene prims deactivated for this episode so the target "
            "rests on the declared flat support instead of an uneven mesh."
        ),
    )
    relocated_pickables: Tuple[RelocatedPickable, ...] = Field(
        description=(
            "Pickables whose authored USD pose is shifted before G1 "
            "attachment records the episode initial transform."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_ids(self) -> "PhysicalScenePreparation":
        object_ids = tuple(item.object_id for item in self.relocated_pickables)
        prim_paths = tuple(item.runtime_prim for item in self.relocated_pickables)
        if len(object_ids) != len(set(object_ids)):
            raise ValueError("relocated pickables must use unique object ids")
        if len(prim_paths) != len(set(prim_paths)):
            raise ValueError("relocated pickables must use unique runtime prims")
        return self


__all__ = ["PhysicalScenePreparation", "RelocatedPickable"]
