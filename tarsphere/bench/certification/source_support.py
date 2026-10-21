"""Certify the live furniture surface supporting a manipulation target.

Target selection and batch episode generation must place each manipulation
target in a clean, grasp-ready environment: one flat direct support surface,
no unstable intermediate mesh (for example woven tray or metal dish on table),
and PhysX contact matching ``target_support_object_id``. Offline AABB overlap
and five-ray certification are necessary but not sufficient when the live
mesh is uneven; use ``physical_scene_preparation`` or asset adaptation when
the native scene layout does not already satisfy this contract.
"""

from __future__ import annotations

import math
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.infra.simulation.scene_query import (
    SurfaceRayHit,
    SurfaceRaycastAll,
    belongs_to_prim_tree,
)
from tarsphere.bench.grounding.schema import (
    GroundedObject,
    GroundedScene,
)
from tarsphere.bench.spatial.physical_task.objects import (
    ManipulationTargetCandidate,
    discover_manipulation_targets,
)


Point3 = Tuple[float, float, float]


class StrictModel(BaseModel):
    """Immutable current-release source-support schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceSupportSearch(StrictModel):
    """Bounded live query used to identify one target's initial support."""

    footprint_sample_fraction: float = Field(
        gt=0.0,
        lt=0.5,
        description=(
            "Fraction of target bbox width and depth separating each corner "
            "sample from the target centre."
        ),
    )
    maximum_surface_misalignment_m: float = Field(
        ge=0.0,
        description=(
            "Largest absolute vertical difference allowed between the "
            "authored target bottom and each live support hit, in metres."
        ),
    )
    minimum_normal_z: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "Minimum upward world-Z normal component required from every support hit."
        ),
    )
    ray_origin_offset_m: float = Field(
        gt=0.0,
        description=(
            "Vertical distance above the authored target bounds where each "
            "downward ray begins, in metres."
        ),
    )
    ray_depth_below_target_m: float = Field(
        gt=0.0,
        description=(
            "Distance below the authored target bottom covered by each live "
            "downward ray, in metres."
        ),
    )


SOURCE_SUPPORT_SEARCH = SourceSupportSearch(
    footprint_sample_fraction=0.25,
    maximum_surface_misalignment_m=0.03,
    minimum_normal_z=math.cos(math.radians(15.0)),
    ray_origin_offset_m=0.02,
    ray_depth_below_target_m=0.08,
)


class SourceSupportEvidence(StrictModel):
    """Live evidence that one grounded object supports the target."""

    target_object_id: str = Field(
        min_length=1,
        description="Grounded manipulation target tested by the live query.",
    )
    support_object_id: str = Field(
        min_length=1,
        description=(
            "Grounded candidate whose prim hierarchy received every first "
            "non-target ray hit."
        ),
    )
    target_bounds_min_xyz: Point3 = Field(
        description=(
            "Minimum corner of the target live USD world bounds in metres at "
            "certification time."
        ),
    )
    target_bounds_max_xyz: Point3 = Field(
        description=(
            "Maximum corner of the target live USD world bounds in metres at "
            "certification time."
        ),
    )
    sampled_positions_xyz: Tuple[Point3, ...] = Field(
        min_length=5,
        max_length=5,
        description=(
            "Five live support intersections below the target centre and "
            "inset footprint corners, in world metres."
        ),
    )
    sampled_prim_paths: Tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
        description="USD support prim hit by each of the five live rays.",
    )
    maximum_surface_misalignment_m: float = Field(
        ge=0.0,
        description=(
            "Largest measured absolute vertical difference between target "
            "bottom and the five support hits, in metres."
        ),
    )
    minimum_normal_z: float = Field(
        gt=0.0,
        le=1.0,
        description="Smallest upward normal component among all support hits.",
    )

    @model_validator(mode="after")
    def validate_evidence(self) -> "SourceSupportEvidence":
        values = (
            *self.target_bounds_min_xyz,
            *self.target_bounds_max_xyz,
            *(value for point in self.sampled_positions_xyz for value in point),
            self.maximum_surface_misalignment_m,
            self.minimum_normal_z,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("source-support evidence must be finite")
        if any(
            self.target_bounds_min_xyz[axis] > self.target_bounds_max_xyz[axis]
            for axis in range(3)
        ):
            raise ValueError("source-support target bounds must be ordered")
        return self


class SourceSupportCertificationError(ValueError):
    """Raised when live geometry does not prove one unique source support."""


def certify_source_support(
    scene: GroundedScene,
    target: ManipulationTargetCandidate,
    *,
    target_bounds_xyz: tuple[Point3, Point3],
    raycast_all: SurfaceRaycastAll,
    search: SourceSupportSearch = SOURCE_SUPPORT_SEARCH,
) -> SourceSupportEvidence:
    """Return live five-ray evidence for the target's unique source support."""
    current_targets = {
        candidate.target_object_id: candidate
        for candidate in discover_manipulation_targets(scene).targets
    }
    if current_targets.get(target.target_object_id) != target:
        raise ValueError(
            "source-support certification requires an unmodified "
            "manipulation candidate from the same grounded scene"
        )
    objects = {obj.object_id: obj for obj in scene.objects}
    target_object = objects[target.target_object_id]
    if not target_object.runtime_prim:
        raise ValueError("source-support certification requires a target runtime prim")
    target_bounds_min_xyz, target_bounds_max_xyz = _validated_live_bounds(
        target_bounds_xyz
    )
    supports = tuple(
        objects[object_id] for object_id in target.source_support_object_ids
    )
    if any(not support.runtime_prim for support in supports):
        missing = sorted(
            support.object_id for support in supports if not support.runtime_prim
        )
        raise ValueError(f"source-support candidates lack runtime prims: {missing}")

    hits = tuple(
        _first_non_target_hit(
            origin_xyz=origin_xyz,
            target_prim_path=target_object.runtime_prim,
            raycast_all=raycast_all,
            ray_distance_m=(
                target_bounds_max_xyz[2]
                - target_bounds_min_xyz[2]
                + search.ray_origin_offset_m
                + search.ray_depth_below_target_m
            ),
        )
        for origin_xyz in _sample_origins(
            target_bounds_min_xyz,
            target_bounds_max_xyz,
            search,
        )
    )
    if any(hit is None for hit in hits):
        raise SourceSupportCertificationError(
            f"Live support rays missed below target {target.target_object_id!r}"
        )
    complete_hits = tuple(hit for hit in hits if hit is not None)
    support_ids = tuple(_support_for_hit(hit, supports) for hit in complete_hits)
    if any(support_id is None for support_id in support_ids):
        paths = sorted(
            {
                hit.prim_path
                for hit, support_id in zip(complete_hits, support_ids)
                if support_id is None
            }
        )
        raise SourceSupportCertificationError(
            "First non-target live hits do not belong to any grounded "
            f"source-support candidate: {paths}"
        )
    unique_support_ids = set(support_ids)
    if len(unique_support_ids) != 1:
        raise SourceSupportCertificationError(
            "Target footprint spans multiple source-support candidates: "
            f"{sorted(str(value) for value in unique_support_ids)}"
        )
    misalignments = tuple(
        abs(target_bounds_min_xyz[2] - hit.position_xyz[2]) for hit in complete_hits
    )
    maximum_misalignment = max(misalignments)
    if maximum_misalignment > search.maximum_surface_misalignment_m:
        raise SourceSupportCertificationError(
            "Live source surface is vertically inconsistent with the target "
            f"bottom: measured={maximum_misalignment:.4f} m, "
            f"allowed={search.maximum_surface_misalignment_m:.4f} m"
        )
    minimum_normal_z = min(hit.normal_xyz[2] for hit in complete_hits)
    if minimum_normal_z < search.minimum_normal_z:
        raise SourceSupportCertificationError(
            "Live source surface is too steep: "
            f"measured_normal_z={minimum_normal_z:.4f}, "
            f"required={search.minimum_normal_z:.4f}"
        )
    return SourceSupportEvidence(
        target_object_id=target.target_object_id,
        support_object_id=next(iter(unique_support_ids)),
        target_bounds_min_xyz=target_bounds_min_xyz,
        target_bounds_max_xyz=target_bounds_max_xyz,
        sampled_positions_xyz=tuple(hit.position_xyz for hit in complete_hits),
        sampled_prim_paths=tuple(hit.prim_path for hit in complete_hits),
        maximum_surface_misalignment_m=maximum_misalignment,
        minimum_normal_z=minimum_normal_z,
    )


def _sample_origins(
    bounds_min_xyz: Point3,
    bounds_max_xyz: Point3,
    search: SourceSupportSearch,
) -> tuple[Point3, ...]:
    center_x = (bounds_min_xyz[0] + bounds_max_xyz[0]) / 2.0
    center_y = (bounds_min_xyz[1] + bounds_max_xyz[1]) / 2.0
    offset_x = (
        bounds_max_xyz[0] - bounds_min_xyz[0]
    ) * search.footprint_sample_fraction
    offset_y = (
        bounds_max_xyz[1] - bounds_min_xyz[1]
    ) * search.footprint_sample_fraction
    origin_z = bounds_max_xyz[2] + search.ray_origin_offset_m
    return (
        (center_x, center_y, origin_z),
        (center_x - offset_x, center_y - offset_y, origin_z),
        (center_x - offset_x, center_y + offset_y, origin_z),
        (center_x + offset_x, center_y - offset_y, origin_z),
        (center_x + offset_x, center_y + offset_y, origin_z),
    )


def _validated_live_bounds(
    bounds_xyz: tuple[Point3, Point3],
) -> tuple[Point3, Point3]:
    try:
        lower, upper = bounds_xyz
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_bounds_xyz must contain minimum and maximum 3-D points"
        ) from exc
    values = (*lower, *upper)
    if (
        len(lower) != 3
        or len(upper) != 3
        or not all(math.isfinite(float(value)) for value in values)
        or any(float(lower[axis]) > float(upper[axis]) for axis in range(3))
    ):
        raise ValueError("target_bounds_xyz must contain finite ordered 3-D points")
    return (
        tuple(float(value) for value in lower),
        tuple(float(value) for value in upper),
    )


def _first_non_target_hit(
    *,
    origin_xyz: Point3,
    target_prim_path: str,
    raycast_all: SurfaceRaycastAll,
    ray_distance_m: float,
) -> SurfaceRayHit | None:
    hits = raycast_all(
        origin_xyz,
        (0.0, 0.0, -1.0),
        ray_distance_m,
    )
    for hit in hits:
        travelled = origin_xyz[2] - hit.position_xyz[2]
        if travelled < -1e-6 or travelled > ray_distance_m + 1e-6:
            raise RuntimeError(
                "PhysX ray hit lies outside the requested downward segment"
            )
        if not belongs_to_prim_tree(
            hit.prim_path,
            target_prim_path,
        ):
            return hit
    return None


def _support_for_hit(
    hit: SurfaceRayHit,
    supports: tuple[GroundedObject, ...],
) -> str | None:
    matches = tuple(
        support.object_id
        for support in supports
        if belongs_to_prim_tree(
            hit.prim_path,
            support.runtime_prim or "",
        )
    )
    if len(matches) > 1:
        raise SourceSupportCertificationError(
            "One live hit belongs to multiple grounded support hierarchies: "
            f"{sorted(matches)}"
        )
    return matches[0] if matches else None


__all__ = [
    "SOURCE_SUPPORT_SEARCH",
    "SourceSupportCertificationError",
    "SourceSupportEvidence",
    "SourceSupportSearch",
    "certify_source_support",
]
