"""Resolve and measure certified G1 Dex3 reaches around a target.

The target-facing geometry itself lives in :mod:`tarsphere.grasp_poses` and is
used only by deterministic dataset/Golden certification. Evaluated Robot agents
instead receive fresh SAM3/depth-conditioned GraspGenX poses. What stays
here is the certification side's own concern: the bounded shell of palm
candidates a promoted entry pose may try, and measurement of the arm's achieved
pose.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.grasp_poses import body_center_position, facing_surface
from tarsphere.infra.manipulation.g1_attachment import bbox_surface_gap
from tarsphere.infra.manipulation.g1_grasp_contact import G1GraspContact


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]
Hand = Literal["left", "right"]


class StrictModel(BaseModel):
    """Immutable current-release grasp-reach schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class GraspReachSearch(StrictModel):
    """Bounded hand-target search performed from one certified entry pose."""

    contact_depths_m: Tuple[float, ...] = Field(
        min_length=1,
        description=(
            "Ordered desired hand displacements from the Robot-facing target "
            "surface toward the target interior, in metres. The collider "
            "must stop the physical hand at the surface; depth never proves "
            "contact by itself."
        ),
    )
    vertical_offsets_from_top_m: Tuple[float, ...] = Field(
        min_length=1,
        description=(
            "Ordered vertical offsets from the live target AABB top, in "
            "metres. Candidates below the target bottom are excluded."
        ),
    )
    lateral_offsets_m: Tuple[float, ...] = Field(
        min_length=1,
        description=(
            "Ordered offsets tangent to the Robot-facing target surface, in "
            "metres. Zero tests the surface centreline."
        ),
    )
    maximum_arm_trials: int = Field(
        ge=1,
        description=(
            "Largest number of physically executed arm targets for one entry "
            "pose before that pose is rejected."
        ),
    )

    @model_validator(mode="after")
    def validate_search(self) -> "GraspReachSearch":
        groups = (
            self.contact_depths_m,
            self.vertical_offsets_from_top_m,
            self.lateral_offsets_m,
        )
        if any(not all(math.isfinite(value) for value in group) for group in groups):
            raise ValueError("grasp reach search values must be finite")
        if any(value < 0.0 for value in self.contact_depths_m):
            raise ValueError("contact depths cannot be negative")
        if any(len(set(group)) != len(group) for group in groups):
            raise ValueError("grasp reach search values must be unique")
        available = math.prod(len(group) for group in groups)
        if self.maximum_arm_trials > available:
            raise ValueError(
                "maximum_arm_trials cannot exceed the candidate search size"
            )
        return self


GRASP_REACH_SEARCH = GraspReachSearch(
    contact_depths_m=(0.0, 0.02, 0.04),
    vertical_offsets_from_top_m=(0.05, 0.10, 0.0, -0.05, -0.10),
    lateral_offsets_m=(0.0, -0.04, 0.04),
    maximum_arm_trials=12,
)


class HandTargetCandidate(StrictModel):
    """One world-space Dex3 palm target derived from live geometry."""

    target_position_world: Point3 = Field(
        description=("Desired Dex3 palm position in scene world metres."),
    )
    contact_depth_m: float = Field(
        ge=0.0,
        description=(
            "Desired inward displacement from the Robot-facing target "
            "surface; native collision, not this target, establishes contact."
        ),
    )
    vertical_offset_from_top_m: float = Field(
        description="Vertical displacement from the live target AABB top.",
    )
    lateral_offset_m: float = Field(
        description=(
            "Signed displacement along the target surface tangent, in metres."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "HandTargetCandidate":
        values = (
            *self.target_position_world,
            self.contact_depth_m,
            self.vertical_offset_from_top_m,
            self.lateral_offset_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("hand target geometry must be finite")
        return self


class GraspCenterTarget(StrictModel):
    """Dex3 closing-centre target derived from one live object AABB."""

    target_position_world: Point3 = Field(
        description=(
            "World-space point that must lie between the closing Dex3 "
            "fingers, in metres."
        ),
    )
    height_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the live object height measured from its lower AABB "
            "face. The canonical 0.60 value encloses the body below its top "
            "without driving the fingertips into the support surface."
        ),
    )


class PhysXGraspContactEvidence(StrictModel):
    """Native collision evidence for one exact G1 hand–object pair."""

    hand: Hand = Field(
        description="G1 Dex3 hand named by the PhysX contact pair.",
    )
    hand_link: str = Field(
        min_length=1,
        description=("Exact Dex3 palm or finger link named by the PhysX contact pair."),
    )
    target_object_id: str = Field(
        min_length=1,
        description="Grounded object named by the PhysX contact pair.",
    )
    actor_paths: Tuple[str, str] = Field(
        description="Rigid-body USD paths reported by PhysX.",
    )
    collider_paths: Tuple[str, str] = Field(
        description="Exact collision-shape USD paths reported by PhysX.",
    )
    position_xyz: Point3 = Field(
        description="World-space PhysX contact point in metres.",
    )
    separation_m: float = Field(
        description="Signed collider separation reported by PhysX in metres.",
    )

    @classmethod
    def from_recorded_contact(
        cls,
        payload: Mapping[str, Any],
    ) -> "PhysXGraspContactEvidence":
        """Project a live PhysX contact record onto this certificate.

        Runtime ``G1GraspContact`` traces also carry force-closure fields
        owned by the runtime record. Those names are dropped here. Any
        other extra key still fails, so a new certified field cannot pass
        through an unreviewed name.
        """
        if not isinstance(payload, Mapping):
            raise ValueError("PhysX grasp contact evidence must be a mapping")
        certified_fields = frozenset(cls.model_fields)
        runtime_only_fields = (
            frozenset(G1GraspContact.__dataclass_fields__) - certified_fields
        )
        unknown = frozenset(payload) - certified_fields - runtime_only_fields
        if unknown:
            raise ValueError(
                "PhysX grasp contact evidence has unknown fields: "
                + ", ".join(sorted(unknown))
            )
        return cls.model_validate(
            {field: payload[field] for field in cls.model_fields if field in payload}
        )

    @model_validator(mode="after")
    def validate_contact(self) -> "PhysXGraspContactEvidence":
        if (
            any(not path.startswith("/") for path in self.actor_paths)
            or any(not path.startswith("/") for path in self.collider_paths)
            or not all(math.isfinite(value) for value in self.position_xyz)
            or not math.isfinite(self.separation_m)
        ):
            raise ValueError("PhysX grasp contact evidence is malformed")
        return self


class GraspReachEvidence(StrictModel):
    """Live pre-attachment geometry after one arm target was executed."""

    target_object_id: str = Field(
        min_length=1,
        description="Grounded pickable object approached by the Dex3 hand.",
    )
    hand: Literal["right"] = Field(
        description="Certified right Dex3 hand that executed the reach.",
    )
    requested_target_position_world: Point3 = Field(
        description="World-space hand target supplied to the executed IK reach.",
    )
    predicted_ik_error_m: float = Field(
        ge=0.0,
        description=(
            "Position error predicted by the official G1 IK solution, in "
            "metres, before articulated execution."
        ),
    )
    hand_position_world: Point3 = Field(
        description="Live Dex3 palm origin after arm execution.",
    )
    grasp_center_world: Point3 = Field(
        description=(
            "Live centre of the closing Dex3 finger envelope after execution."
        ),
    )
    hand_bounds_min_xyz: Point3 = Field(
        description="Minimum corner of the live hand world AABB, in metres.",
    )
    hand_bounds_max_xyz: Point3 = Field(
        description="Maximum corner of the live hand world AABB, in metres.",
    )
    target_bounds_min_xyz: Point3 = Field(
        description="Minimum corner of the live target world AABB, in metres.",
    )
    target_bounds_max_xyz: Point3 = Field(
        description="Maximum corner of the live target world AABB, in metres.",
    )
    aabb_surface_gap_m: float = Field(
        ge=0.0,
        description=(
            "Diagnostic Euclidean gap between live hand and target AABBs; "
            "this value never establishes contact."
        ),
    )
    grasp_center_to_target_m: float = Field(
        ge=0.0,
        description=(
            "Distance from the live Dex3 closing centre to the exact target "
            "AABB; zero means the target lies inside the closing envelope."
        ),
    )
    physx_contact: PhysXGraspContactEvidence | None = Field(
        default=None,
        description=(
            "Native PhysX contact point for the exact hand–target pair, or "
            "null when the executed reach did not make contact."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "GraspReachEvidence":
        values = (
            *self.requested_target_position_world,
            self.predicted_ik_error_m,
            *self.hand_position_world,
            *self.grasp_center_world,
            *self.hand_bounds_min_xyz,
            *self.hand_bounds_max_xyz,
            *self.target_bounds_min_xyz,
            *self.target_bounds_max_xyz,
            self.aabb_surface_gap_m,
            self.grasp_center_to_target_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("grasp reach evidence must be finite")
        if any(
            self.hand_bounds_min_xyz[axis] > self.hand_bounds_max_xyz[axis]
            for axis in range(3)
        ):
            raise ValueError("hand bounds must be ordered")
        if any(
            self.target_bounds_min_xyz[axis] > self.target_bounds_max_xyz[axis]
            for axis in range(3)
        ):
            raise ValueError("target bounds must be ordered")
        if self.physx_contact is not None and (
            self.physx_contact.hand != self.hand
            or self.physx_contact.target_object_id != self.target_object_id
        ):
            raise ValueError("PhysX contact evidence belongs to another hand or target")
        return self

    @property
    def contact_is_verified(self) -> bool:
        """Whether native PhysX evidence records target contact."""
        return self.physx_contact is not None


def generate_hand_targets(
    *,
    target_bounds_min_xyz: Point3,
    target_bounds_max_xyz: Point3,
    robot_entry_xy: Point2,
    search: GraspReachSearch = GRASP_REACH_SEARCH,
) -> Tuple[HandTargetCandidate, ...]:
    """Derive a bounded target-facing hand shell from live target bounds."""
    lower = tuple(float(value) for value in target_bounds_min_xyz)
    upper = tuple(float(value) for value in target_bounds_max_xyz)
    surface = facing_surface(
        target_bounds_min_xyz=target_bounds_min_xyz,
        target_bounds_max_xyz=target_bounds_max_xyz,
        robot_position_xy=robot_entry_xy,
    )
    surface_x, surface_y = surface.surface_xy
    outward_x, outward_y = surface.outward_xy
    tangent_x, tangent_y = surface.tangent_xy

    candidates = []
    for vertical_offset in search.vertical_offsets_from_top_m:
        for lateral_offset in search.lateral_offsets_m:
            for contact_depth in search.contact_depths_m:
                target_z = upper[2] + vertical_offset
                if target_z < lower[2]:
                    continue
                candidates.append(
                    HandTargetCandidate(
                        target_position_world=(
                            surface_x
                            - outward_x * contact_depth
                            + tangent_x * lateral_offset,
                            surface_y
                            - outward_y * contact_depth
                            + tangent_y * lateral_offset,
                            target_z,
                        ),
                        contact_depth_m=contact_depth,
                        vertical_offset_from_top_m=vertical_offset,
                        lateral_offset_m=lateral_offset,
                    )
                )
    if not candidates:
        raise ValueError(
            "grasp reach search produced no target above the object bottom"
        )
    return tuple(candidates)


def grasp_center_target(
    *,
    target_bounds_min_xyz: Point3,
    target_bounds_max_xyz: Point3,
    height_fraction: float = 0.60,
) -> GraspCenterTarget:
    """Place the Dex3 closing centre inside one bounded object body."""
    return GraspCenterTarget(
        target_position_world=body_center_position(
            target_bounds_min_xyz=target_bounds_min_xyz,
            target_bounds_max_xyz=target_bounds_max_xyz,
            height_fraction=height_fraction,
        ),
        height_fraction=float(height_fraction),
    )


def measure_grasp_reach(
    manipulation_snapshot: dict[str, Any],
    *,
    target_object_id: str,
    hand: Hand,
    requested_target_position_world: Point3,
    predicted_ik_error_m: float,
) -> GraspReachEvidence:
    """Measure one executed arm reach from the private live scene snapshot."""
    if hand != "right":
        raise ValueError("Dataset grasp certification uses only the right Dex3 hand.")
    hands = manipulation_snapshot.get("hands")
    objects = manipulation_snapshot.get("objects")
    if not isinstance(hands, dict) or not isinstance(objects, dict):
        raise ValueError("grasp reach measurement requires live hands and objects")
    hand_state = hands.get(hand)
    target_state = objects.get(target_object_id)
    if not isinstance(hand_state, dict):
        raise ValueError(f"live hand geometry is missing for {hand!r}")
    if not isinstance(target_state, dict):
        raise ValueError(f"live target geometry is missing for {target_object_id!r}")
    if bool(target_state.get("attached", False)):
        raise ValueError("grasp reach must be measured before target attachment")

    hand_position = _point3(
        hand_state.get("position_world"),
        label="hand position",
    )
    grasp_center = _point3(
        hand_state.get("grasp_center_world"),
        label="grasp center",
    )
    hand_min, hand_max = _bounds(
        hand_state,
        label="hand",
    )
    target_min, target_max = _bounds(
        target_state,
        label="target",
    )
    requested_target = _point3(
        requested_target_position_world,
        label="requested hand target",
    )
    predicted_error = float(predicted_ik_error_m)
    if not math.isfinite(predicted_error) or predicted_error < 0.0:
        raise ValueError("predicted IK error must be finite and non-negative")
    gap = bbox_surface_gap(
        (list(hand_min), list(hand_max)),
        (list(target_min), list(target_max)),
    )
    grasp_center_gap = _point_to_bounds_distance(
        grasp_center,
        target_min,
        target_max,
    )
    contacts = manipulation_snapshot.get("grasp_contacts")
    if not isinstance(contacts, list):
        raise ValueError("grasp reach measurement requires live PhysX contact evidence")
    matching_contacts = [
        PhysXGraspContactEvidence.from_recorded_contact(item)
        for item in contacts
        if (
            isinstance(item, dict)
            and item.get("hand") == hand
            and item.get("target_object_id") == target_object_id
        )
    ]
    contact = min(
        matching_contacts,
        key=lambda item: item.separation_m,
        default=None,
    )
    return GraspReachEvidence(
        target_object_id=str(target_object_id),
        hand=hand,
        requested_target_position_world=requested_target,
        predicted_ik_error_m=predicted_error,
        hand_position_world=hand_position,
        grasp_center_world=grasp_center,
        hand_bounds_min_xyz=hand_min,
        hand_bounds_max_xyz=hand_max,
        target_bounds_min_xyz=target_min,
        target_bounds_max_xyz=target_max,
        aabb_surface_gap_m=gap,
        grasp_center_to_target_m=grasp_center_gap,
        physx_contact=contact,
    )


def _point_to_bounds_distance(
    point: Point3,
    lower: Point3,
    upper: Point3,
) -> float:
    return math.sqrt(
        sum(
            max(
                float(lower[axis]) - float(point[axis]),
                0.0,
                float(point[axis]) - float(upper[axis]),
            )
            ** 2
            for axis in range(3)
        )
    )


def _point3(value: Any, *, label: str) -> Point3:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(f"{label} must contain three coordinates")
    point = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{label} must be finite")
    return point  # type: ignore[return-value]


def _bounds(
    state: dict[str, Any],
    *,
    label: str,
) -> tuple[Point3, Point3]:
    lower = _point3(state.get("bbox_min"), label=f"{label} bounds minimum")
    upper = _point3(state.get("bbox_max"), label=f"{label} bounds maximum")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        raise ValueError(f"{label} bounds must be ordered")
    return lower, upper


__all__ = [
    "GRASP_REACH_SEARCH",
    "GraspReachEvidence",
    "GraspReachSearch",
    "GraspCenterTarget",
    "HandTargetCandidate",
    "PhysXGraspContactEvidence",
    "generate_hand_targets",
    "grasp_center_target",
    "measure_grasp_reach",
]
