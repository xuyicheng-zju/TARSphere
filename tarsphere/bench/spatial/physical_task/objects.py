"""Discover grounded object combinations suitable for task construction."""

from __future__ import annotations

from collections import Counter
import math
from typing import Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.grounding.schema import (
    GroundedObject,
    GroundedScene,
)


# GRScenes bounds occasionally overlap a supporting surface by a few
# centimetres. This tolerance admits that annotation error while remaining a
# candidate-generation rule, not a grasp or contact certificate.
_SUPPORT_BOUNDS_TOLERANCE_M = 0.03

# Offline geometry may only nominate object classes whose primary asset
# affordance includes an upper support surface. A large AABB plus an approach
# path is not enough: curtains, shoes, mirrors, and similar objects can pass
# those two tests while remaining semantically invalid place destinations.
# Live five-ray probing still owns actual surface flatness and support.
# Batch generation must additionally require a clean grasp environment: the
# target's direct PhysX support is a flat hard surface (table/desk/counter),
# not an uneven tray, dish stack, or other intermediate mesh. See
# docs/EPISODE_DIVERSITY.md.
STATIC_PLACEMENT_SUPPORT_CATEGORIES = frozenset(
    {
        "bed",
        "cabinet",
        "chestofdrawers",
        "counter",
        "desk",
        "dishwasher",
        "nightstand",
        "oven",
        "refrigerator",
        "shelf",
        "sideboardcabinet",
        "stool",
        "table",
        "teatable",
        "tray",
        "tvstand",
        "washingmachine",
    }
)


class StrictModel(BaseModel):
    """Immutable current-release physical-task object schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ManipulationTargetCandidate(StrictModel):
    """One grounded pickable object with evidence needed for task design."""

    target_object_id: str = Field(
        min_length=1,
        description=(
            "Captioned pickable scene object with a verified runtime prim, "
            "an official approach path, and at least one possible support."
        ),
    )
    source_support_object_ids: Tuple[str, ...] = Field(
        min_length=1,
        description=(
            "Grounded objects whose bounds are geometrically consistent with "
            "containing a support surface beneath the target at episode "
            "start, ordered by the highest external upper surface and then "
            "compact enclosing hierarchy; physical contact remains "
            "uncertified."
        ),
    )
    distractor_object_ids: Tuple[str, ...] = Field(
        description=(
            "Other complete manipulation-target candidates in the scene with "
            "the same grounded category, available for a genuine target "
            "ambiguity; semantic similarity still requires authoring review."
        ),
    )
    approach_path_indexes: Tuple[int, ...] = Field(
        min_length=1,
        description=(
            "Indexes of official object-conditioned approach paths for the "
            "target; dense waypoints remain owned by the scene assets."
        ),
    )

    @model_validator(mode="after")
    def validate_references(self) -> "ManipulationTargetCandidate":
        if self.target_object_id in self.source_support_object_ids:
            raise ValueError("target object cannot support itself")
        if self.target_object_id in self.distractor_object_ids:
            raise ValueError("target object cannot be its own distractor")
        for label, values in (
            ("source support object", self.source_support_object_ids),
            ("distractor object", self.distractor_object_ids),
            ("approach path index", self.approach_path_indexes),
        ):
            duplicates = sorted(
                str(value) for value, count in Counter(values).items() if count > 1
            )
            if duplicates:
                raise ValueError(f"duplicate {label}: {duplicates}")
        return self


class ManipulationTargetSet(StrictModel):
    """All statically eligible manipulation targets in one grounded scene."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene from which every candidate was derived.",
    )
    targets: Tuple[ManipulationTargetCandidate, ...] = Field(
        description=(
            "Deterministically ordered targets; an empty tuple means this "
            "scene offers no target satisfying the current static contract."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_targets(self) -> "ManipulationTargetSet":
        duplicates = sorted(
            target_id
            for target_id, count in Counter(
                item.target_object_id for item in self.targets
            ).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate manipulation target candidates: {duplicates}")
        return self


class PlacementDestinationCandidate(StrictModel):
    """One grounded object whose bounds could contain the selected target."""

    destination_object_id: str = Field(
        min_length=1,
        description=(
            "Grounded object distinct from all target-side roles and large "
            "enough to contain the target footprint in its world-aligned "
            "bounds."
        ),
    )
    room_id: str = Field(
        min_length=1,
        description="Semantic room containing the destination object.",
    )
    approach_path_indexes: Tuple[int, ...] = Field(
        min_length=1,
        description=(
            "Indexes of official object-conditioned approach paths for this "
            "destination; dense waypoints remain owned by scene assets."
        ),
    )
    same_room_as_target: bool = Field(
        description=(
            "Whether target and destination belong to the same semantic room."
        ),
    )
    center_distance_xy_m: float = Field(
        ge=0.0,
        description=(
            "Straight-line world-XY distance between target and destination "
            "centres in metres; this is not navigation-path length."
        ),
    )
    bbox_top_height_m: float = Field(
        description=(
            "Maximum world Z of the destination bounds in metres; it is only "
            "a surface-search hint, not evidence of a real horizontal patch."
        ),
    )
    bbox_placement_margin_xy_m: Tuple[float, float] = Field(
        description=(
            "Non-negative half-width and half-depth remaining when the target "
            "footprint is centred in the better of its current or 90-degree "
            "orientation inside destination bounds."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "PlacementDestinationCandidate":
        values = (
            self.center_distance_xy_m,
            self.bbox_top_height_m,
            *self.bbox_placement_margin_xy_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("placement destination geometry must be finite")
        if any(value < 0.0 for value in self.bbox_placement_margin_xy_m):
            raise ValueError("bbox placement margins must be non-negative")
        duplicates = sorted(
            str(value)
            for value, count in Counter(self.approach_path_indexes).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate destination approach path index: {duplicates}")
        return self


class PlacementDestinationSet(StrictModel):
    """Static destination options conditioned on one manipulation target."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene from which every destination was derived.",
    )
    target_object_id: str = Field(
        min_length=1,
        description=(
            "Manipulation target whose footprint and semantic role exclusions "
            "condition every destination candidate."
        ),
    )
    destinations: Tuple[PlacementDestinationCandidate, ...] = Field(
        description=(
            "Deterministically ordered destination candidates; each still "
            "requires surface, navigation, IK, and release certification."
        ),
    )

    @model_validator(mode="after")
    def validate_unique_destinations(self) -> "PlacementDestinationSet":
        duplicates = sorted(
            object_id
            for object_id, count in Counter(
                item.destination_object_id for item in self.destinations
            ).items()
            if count > 1
        )
        if duplicates:
            raise ValueError(
                f"duplicate placement destination candidates: {duplicates}"
            )
        if any(
            item.destination_object_id == self.target_object_id
            for item in self.destinations
        ):
            raise ValueError("target object cannot be a placement destination")
        return self


def discover_manipulation_targets(
    scene: GroundedScene,
) -> ManipulationTargetSet:
    """Return objects eligible for later nav–grasp task resolution.

    Eligibility is intentionally static. A returned target still requires
    Isaac certification for visibility, stance reachability, IK, attachment,
    transport stability, and placement.
    """
    approach_indexes: dict[str, list[int]] = {}
    for path in scene.object_approach_paths:
        approach_indexes.setdefault(path.object_id, []).append(path.path_index)

    eligible_objects = {
        obj.object_id: obj
        for obj in scene.objects
        if _has_task_object_evidence(obj, approach_indexes)
    }
    supported_objects: dict[
        str,
        tuple[GroundedObject, tuple[GroundedObject, ...]],
    ] = {}
    for target_object_id in sorted(eligible_objects):
        target = eligible_objects[target_object_id]
        supports = _possible_supports(
            target,
            scene.objects,
        )
        if supports:
            supported_objects[target_object_id] = (target, supports)

    candidates = []
    for target_object_id in sorted(supported_objects):
        target, supports = supported_objects[target_object_id]
        distractors = tuple(
            object_id
            for object_id, (obj, _) in supported_objects.items()
            if (object_id != target_object_id and obj.category == target.category)
        )
        candidates.append(
            ManipulationTargetCandidate(
                target_object_id=target_object_id,
                source_support_object_ids=tuple(item.object_id for item in supports),
                distractor_object_ids=distractors,
                approach_path_indexes=tuple(sorted(approach_indexes[target_object_id])),
            )
        )
    return ManipulationTargetSet(
        scene_id=scene.scene_id,
        targets=tuple(candidates),
    )


def discover_source_supports(
    scene: GroundedScene,
    target_object_id: str,
) -> tuple[GroundedObject, ...]:
    """Return grounded supports beneath an object used only as route geometry.

    An inference-time replacement does not inherit the routed object's task
    identity or interaction annotation. It needs only the real anchor pose and
    a support that contains that pose and exposes a runtime prim. The grasp
    approach belongs to the routed target, not to the furniture beneath it.
    This query deliberately omits the target-side
    ``pickable`` requirement enforced by ``discover_manipulation_targets``.
    """
    objects = {obj.object_id: obj for obj in scene.objects}
    try:
        target = objects[target_object_id]
    except KeyError as error:
        raise ValueError(
            f"route anchor is absent from grounded scene: {target_object_id}"
        ) from error
    return _possible_supports(
        target,
        scene.objects,
    )


def discover_placement_destinations(
    scene: GroundedScene,
    target: ManipulationTargetCandidate,
) -> PlacementDestinationSet:
    """Return bbox-eligible destinations for one discovered target.

    The function intentionally does not choose a destination or invent a
    placement point. Irregular geometry, empty regions, reachable surface
    patches, Robot stance, IK, and stable release belong to live physical
    resolution.
    """
    discovered_targets = {
        item.target_object_id: item
        for item in discover_manipulation_targets(scene).targets
    }
    if discovered_targets.get(target.target_object_id) != target:
        raise ValueError(
            "placement destination discovery requires an unmodified target "
            "candidate from the same grounded scene"
        )
    return discover_object_placement_destinations(
        scene,
        target_object_id=target.target_object_id,
        excluded_object_ids=(
            *target.source_support_object_ids,
            *target.distractor_object_ids,
        ),
    )


def discover_object_placement_destinations(
    scene: GroundedScene,
    *,
    target_object_id: str,
    excluded_object_ids: Tuple[str, ...] = (),
) -> PlacementDestinationSet:
    """Return destination candidates for an explicit grounded target.

    This is the placement-side authority for inference-time assets. Their
    grasp route is owned by ``GroundedTaskProgram`` and need not be duplicated
    into the scene's native object-path index merely so target footprint
    geometry can condition destination discovery.
    """
    objects = {obj.object_id: obj for obj in scene.objects}
    try:
        target_object = objects[target_object_id]
    except KeyError as error:
        raise ValueError(
            f"placement target is absent from grounded scene: {target_object_id}"
        ) from error
    if (
        target_object.interaction != "pickable"
        or not target_object.caption
        or not target_object.runtime_prim
    ):
        raise ValueError("placement target must be a captioned pickable runtime object")
    approach_indexes: dict[str, list[int]] = {}
    for path in scene.object_approach_paths:
        approach_indexes.setdefault(path.object_id, []).append(path.path_index)

    excluded_object_ids = {
        target_object_id,
        *excluded_object_ids,
    }
    destinations = []
    for object_id in sorted(objects):
        obj = objects[object_id]
        if (
            object_id in excluded_object_ids
            or obj.category not in STATIC_PLACEMENT_SUPPORT_CATEGORIES
            or not obj.caption
            or not obj.runtime_prim
            or object_id not in approach_indexes
        ):
            continue
        margins = _placement_margins(
            target_object,
            destination=obj,
        )
        if margins is None:
            continue
        destinations.append(
            PlacementDestinationCandidate(
                destination_object_id=object_id,
                room_id=obj.room_id,
                approach_path_indexes=tuple(sorted(approach_indexes[object_id])),
                same_room_as_target=(obj.room_id == target_object.room_id),
                center_distance_xy_m=math.dist(
                    target_object.position_xyz[:2],
                    obj.position_xyz[:2],
                ),
                bbox_top_height_m=obj.bounds_max_xyz[2],
                bbox_placement_margin_xy_m=margins,
            )
        )
    return PlacementDestinationSet(
        scene_id=scene.scene_id,
        target_object_id=target_object_id,
        destinations=tuple(destinations),
    )


def _has_task_object_evidence(
    obj: GroundedObject,
    approach_indexes: dict[str, list[int]],
) -> bool:
    return (
        obj.interaction == "pickable"
        and bool(obj.caption)
        and bool(obj.runtime_prim)
        and obj.object_id in approach_indexes
    )


def _possible_supports(
    target: GroundedObject,
    objects: Tuple[GroundedObject, ...],
) -> tuple[GroundedObject, ...]:
    target_x, target_y, _ = target.position_xyz
    target_area = _footprint_area(target)
    supports = []
    for obj in objects:
        if (
            obj.object_id == target.object_id
            or obj.room_id != target.room_id
            or not obj.runtime_prim
        ):
            continue
        lower = obj.bounds_min_xyz
        upper = obj.bounds_max_xyz
        if not (lower[0] <= target_x <= upper[0] and lower[1] <= target_y <= upper[1]):
            continue
        target_bottom_z = target.bounds_min_xyz[2]
        if not (
            lower[2] - _SUPPORT_BOUNDS_TOLERANCE_M
            <= target_bottom_z
            <= upper[2] + _SUPPORT_BOUNDS_TOLERANCE_M
        ):
            continue
        if _footprint_area(obj) < target_area:
            continue
        supports.append(obj)
    return tuple(
        sorted(
            supports,
            key=lambda obj: (
                (
                    0
                    if obj.bounds_max_xyz[2]
                    <= target.position_xyz[2] + _SUPPORT_BOUNDS_TOLERANCE_M
                    else 1
                ),
                -obj.bounds_max_xyz[2],
                _footprint_area(obj),
                obj.bounds_max_xyz[2] - obj.bounds_min_xyz[2],
                obj.object_id,
            ),
        )
    )


def _footprint_area(obj: GroundedObject) -> float:
    return (obj.bounds_max_xyz[0] - obj.bounds_min_xyz[0]) * (
        obj.bounds_max_xyz[1] - obj.bounds_min_xyz[1]
    )


def _placement_margins(
    target: GroundedObject,
    *,
    destination: GroundedObject,
) -> tuple[float, float] | None:
    target_width, target_depth = _footprint_size(target)
    destination_width, destination_depth = _footprint_size(destination)
    orientations = (
        (target_width, target_depth),
        (target_depth, target_width),
    )
    valid = [
        (
            (destination_width - width) / 2.0,
            (destination_depth - depth) / 2.0,
        )
        for width, depth in orientations
        if (width <= destination_width and depth <= destination_depth)
    ]
    if not valid:
        return None
    return max(
        valid,
        key=lambda margins: (
            min(margins),
            sum(margins),
            margins,
        ),
    )


def _footprint_size(obj: GroundedObject) -> tuple[float, float]:
    return (
        obj.bounds_max_xyz[0] - obj.bounds_min_xyz[0],
        obj.bounds_max_xyz[1] - obj.bounds_min_xyz[1],
    )


__all__ = [
    "ManipulationTargetCandidate",
    "ManipulationTargetSet",
    "PlacementDestinationCandidate",
    "PlacementDestinationSet",
    "STATIC_PLACEMENT_SUPPORT_CATEGORIES",
    "discover_manipulation_targets",
    "discover_object_placement_destinations",
    "discover_placement_destinations",
    "discover_source_supports",
]
