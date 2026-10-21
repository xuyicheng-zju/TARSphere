"""Record exact PhysX contacts between Dex3 links and task objects."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping, Optional, Sequence

from tarsphere.embodiment.dex3 import (
    DEX3_MAXIMUM_OPPOSING_NORMAL_DOT,
    DEX3_MINIMUM_CLOSING_AXIS_NORMAL_PROJECTION,
    DEX3_MINIMUM_NORMAL_IMPULSE_NS,
    DEX3_MINIMUM_OPPOSING_CONTACT_SPAN_M,
)
from tarsphere.embodiment.gripper_profiles import GripperProfile
from tarsphere.infra.control.physical_contracts import G1_GRASP_CONTACT_GAP_M

FORCE_CLOSURE_GATE_OPPOSITE_SIDES = "opposite_sides"
FORCE_CLOSURE_GATE_CONTACT_SPAN = "contact_span"
FORCE_CLOSURE_GATE_OPPOSITE_NORMAL_SIGNS = "opposite_normal_signs"
FORCE_CLOSURE_GATE_CLOSING_AXIS_NORMAL_PROJECTION = "closing_axis_normal_projection"
FORCE_CLOSURE_GATE_OPPOSING_NORMAL_DOT = "opposing_normal_dot"
FORCE_CLOSURE_GATE_NORMAL_IMPULSE = "normal_impulse"


@dataclass(frozen=True)
class G1GraspContact:
    """One live PhysX contact point eligible for a grasp attachment.

    Attributes:
        hand: Dex3 hand participating in the contact.
        hand_link: Exact palm or finger link participating in the contact.
        target_object_id: Episode object touched by that hand.
        actor_paths: PhysX rigid-body paths for the reported pair.
        collider_paths: Exact PhysX collider paths for the reported pair.
        target_collider_index: Index of the target in ``collider_paths``.
            This is resolved from the episode's USD binding when the contact
            is recorded; object ids are semantic labels and need not appear
            in USD paths.
        position_xyz: World-space contact point in metres.
        normal_xyz: World-space contact normal reported by PhysX.
        normal_impulse_xyz: World-space normal impulse applied at this contact
            during the latest physics step, in newton-seconds.
        friction_impulse_xyz: Sum of the world-space friction-anchor impulses
            for the same collider pair and physics step, in newton-seconds.
        separation_m: Signed collider separation reported by PhysX in metres.
    """

    hand: str
    hand_link: str
    target_object_id: str
    actor_paths: tuple[str, str]
    collider_paths: tuple[str, str]
    target_collider_index: int
    position_xyz: tuple[float, float, float]
    normal_xyz: tuple[float, float, float]
    normal_impulse_xyz: tuple[float, float, float]
    friction_impulse_xyz: tuple[float, float, float]
    separation_m: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible evidence for the private runtime trace."""
        return asdict(self)


@dataclass(frozen=True)
class G1SupportContact:
    """One live PhysX contact between a carried object and its support."""

    # Carried task object sampled immediately before hand opening.
    object_id: str
    # Scene object intended to support the carried object after release.
    destination_object_id: str
    # Exact collider pair reported by PhysX for the contact.
    collider_paths: tuple[str, str]
    # Signed separation between colliders in metres.
    separation_m: float

    def as_dict(self) -> dict[str, Any]:
        """Return JSON-compatible private evaluator evidence."""

        return asdict(self)


@dataclass(frozen=True)
class G1OpposingContactPair:
    """Measured force-closure evidence across one generated closing axis.

    ``first`` and ``second`` follow the two ordered contact-role groups in the
    selected gripper profile.  Point and normal projections use the same
    signed world-space closing axis, so a trace can distinguish contacts on
    opposite sides from several links pressing the same object surface.
    """

    first: G1GraspContact
    second: G1GraspContact
    first_point_projection_m: float
    second_point_projection_m: float
    first_normal_projection: float
    second_normal_projection: float
    normal_dot: float
    contact_span_m: float

    def as_dict(self) -> dict[str, Any]:
        """Return evaluator-private evidence without recomputing geometry."""

        return {
            "first": self.first.as_dict(),
            "second": self.second.as_dict(),
            "first_point_projection_m": self.first_point_projection_m,
            "second_point_projection_m": self.second_point_projection_m,
            "first_normal_projection": self.first_normal_projection,
            "second_normal_projection": self.second_normal_projection,
            "normal_dot": self.normal_dot,
            "contact_span_m": self.contact_span_m,
        }


def grasp_contact_role_pair(
    contacts: Sequence[G1GraspContact],
    *,
    gripper: GripperProfile,
) -> Optional[tuple[G1GraspContact, G1GraspContact]]:
    """Return real target contacts from both gripper roles, or ``None``.

    This is deliberately weaker than force closure. It is only suitable for
    pose-preserving diagnostic retention after the generated closing volume
    has independently enclosed the target. Formal physical grasp uses
    :func:`force_closure_contact_pair` instead.
    """

    role_groups = tuple(gripper.opposing_contact_roles)
    if len(role_groups) != 2:
        raise ValueError(
            f"{gripper.name} must define exactly two opposing contact roles."
        )
    grouped = tuple(
        tuple(
            contact
            for contact in contacts
            if any(marker in contact.hand_link for marker in markers)
        )
        for markers in role_groups
    )
    if not grouped[0] or not grouped[1]:
        return None
    return min(
        ((first, second) for first in grouped[0] for second in grouped[1]),
        key=lambda pair: (
            pair[0].separation_m + pair[1].separation_m,
            -_magnitude(pair[0].normal_impulse_xyz)
            - _magnitude(pair[1].normal_impulse_xyz),
        ),
    )


def opposing_grasp_contact_pair(
    contacts: Sequence[G1GraspContact],
    *,
    object_center_world: Sequence[float],
    closing_axis_world: Sequence[float],
    gripper: GripperProfile,
) -> Optional[G1OpposingContactPair]:
    """Return contacts that straddle the object centre, or ``None``.

    This is the diagnostic-retention gate. It proves that a FixedJoint
    preserves a physically realized closed-hand pose, but it deliberately does
    not claim that friction alone can retain the object.
    """

    return _opposing_contact_pair(
        contacts,
        object_center_world=object_center_world,
        closing_axis_world=closing_axis_world,
        gripper=gripper,
        require_force_closure=False,
        require_current_impulse=False,
    )


def force_closure_contact_pair(
    contacts: Sequence[G1GraspContact],
    *,
    object_center_world: Sequence[float],
    closing_axis_world: Sequence[float],
    gripper: GripperProfile,
    require_current_impulse: bool,
) -> Optional[G1OpposingContactPair]:
    """Return the strongest measured friction-retaining pair, or ``None``."""

    return _opposing_contact_pair(
        contacts,
        object_center_world=object_center_world,
        closing_axis_world=closing_axis_world,
        gripper=gripper,
        require_force_closure=True,
        require_current_impulse=require_current_impulse,
    )


@dataclass(frozen=True)
class ForceClosureContactDiagnostics:
    """Best cross-role measurements without force-closure accept thresholds."""

    contact_links: tuple[str, ...]
    first_role_contact: bool
    second_role_contact: bool
    best_contact_span_m: float | None
    best_normal_dot: float | None
    best_minimum_closing_axis_normal_projection: float | None
    best_first_normal_projection: float | None
    best_second_normal_projection: float | None
    best_first_point_projection_m: float | None
    best_second_point_projection_m: float | None
    best_failed_gate_names: tuple[str, ...]
    best_span_deficit_m: float | None
    best_projection_deficit: float | None
    best_normal_dot_deficit: float | None
    best_first_normal_impulse_ns: float | None
    best_second_normal_impulse_ns: float | None

    def as_report_fields(self) -> dict[str, Any]:
        """Return the wrap-report / failure-message diagnostic fields."""

        return {
            "contact_links": list(self.contact_links),
            "first_role_contact": self.first_role_contact,
            "second_role_contact": self.second_role_contact,
            "best_contact_span_m": self.best_contact_span_m,
            "best_normal_dot": self.best_normal_dot,
            "best_minimum_closing_axis_normal_projection": (
                self.best_minimum_closing_axis_normal_projection
            ),
            "best_first_normal_projection": self.best_first_normal_projection,
            "best_second_normal_projection": self.best_second_normal_projection,
            "best_first_point_projection_m": self.best_first_point_projection_m,
            "best_second_point_projection_m": (self.best_second_point_projection_m),
            "best_failed_gate_names": list(self.best_failed_gate_names),
            "best_span_deficit_m": self.best_span_deficit_m,
            "best_projection_deficit": self.best_projection_deficit,
            "best_normal_dot_deficit": self.best_normal_dot_deficit,
            "best_first_normal_impulse_ns": self.best_first_normal_impulse_ns,
            "best_second_normal_impulse_ns": self.best_second_normal_impulse_ns,
        }


@dataclass(frozen=True)
class ForceClosureGateEvaluation:
    """One pair measured against every Dex3 force-closure accept gate."""

    opposite_sides: bool
    opposed_normals: bool
    span_deficit: float
    projection_deficit: float
    normal_dot_deficit: float
    first_point_projection_m: float
    second_point_projection_m: float
    first_normal_projection: float
    second_normal_projection: float
    min_abs_normal_projection: float
    normal_dot: float
    contact_span_m: float
    first_normal_impulse_ns: float
    second_normal_impulse_ns: float
    impulse_passed: bool

    def failed_gate_names(
        self,
        *,
        require_current_impulse: bool,
    ) -> tuple[str, ...]:
        failed: list[str] = []
        if not self.opposite_sides:
            failed.append(FORCE_CLOSURE_GATE_OPPOSITE_SIDES)
        if self.span_deficit != 0.0:
            failed.append(FORCE_CLOSURE_GATE_CONTACT_SPAN)
        if not self.opposed_normals:
            failed.append(FORCE_CLOSURE_GATE_OPPOSITE_NORMAL_SIGNS)
        if self.projection_deficit != 0.0:
            failed.append(FORCE_CLOSURE_GATE_CLOSING_AXIS_NORMAL_PROJECTION)
        if self.normal_dot_deficit != 0.0:
            failed.append(FORCE_CLOSURE_GATE_OPPOSING_NORMAL_DOT)
        if require_current_impulse and not self.impulse_passed:
            failed.append(FORCE_CLOSURE_GATE_NORMAL_IMPULSE)
        return tuple(failed)

    def ranking_key(
        self,
        *,
        require_current_impulse: bool = False,
    ) -> tuple[int, float]:
        return (
            len(
                self.failed_gate_names(
                    require_current_impulse=require_current_impulse,
                )
            ),
            self.span_deficit + self.projection_deficit + self.normal_dot_deficit,
        )


def _evaluate_force_closure_gates(
    pair: G1OpposingContactPair,
) -> ForceClosureGateEvaluation:
    """Score one pair against the Dex3 force-closure accept thresholds."""

    opposite_sides = (
        pair.first_point_projection_m * pair.second_point_projection_m < 0.0
    )
    opposed_normals = pair.first_normal_projection * pair.second_normal_projection < 0.0
    min_projection = min(
        abs(pair.first_normal_projection),
        abs(pair.second_normal_projection),
    )
    first_impulse = _magnitude(pair.first.normal_impulse_xyz)
    second_impulse = _magnitude(pair.second.normal_impulse_xyz)
    return ForceClosureGateEvaluation(
        opposite_sides=opposite_sides,
        opposed_normals=opposed_normals,
        span_deficit=max(
            0.0,
            DEX3_MINIMUM_OPPOSING_CONTACT_SPAN_M - pair.contact_span_m,
        ),
        projection_deficit=max(
            0.0,
            DEX3_MINIMUM_CLOSING_AXIS_NORMAL_PROJECTION - min_projection,
        ),
        normal_dot_deficit=max(
            0.0,
            pair.normal_dot - DEX3_MAXIMUM_OPPOSING_NORMAL_DOT,
        ),
        first_point_projection_m=pair.first_point_projection_m,
        second_point_projection_m=pair.second_point_projection_m,
        first_normal_projection=pair.first_normal_projection,
        second_normal_projection=pair.second_normal_projection,
        min_abs_normal_projection=min_projection,
        normal_dot=pair.normal_dot,
        contact_span_m=pair.contact_span_m,
        first_normal_impulse_ns=first_impulse,
        second_normal_impulse_ns=second_impulse,
        impulse_passed=(
            first_impulse >= DEX3_MINIMUM_NORMAL_IMPULSE_NS
            and second_impulse >= DEX3_MINIMUM_NORMAL_IMPULSE_NS
        ),
    )


def _force_closure_gate_deficit(
    pair: G1OpposingContactPair,
    *,
    require_current_impulse: bool = False,
) -> tuple[int, float]:
    """Rank one pair by how far it is from the force-closure accept gates.

    Smaller is closer to passing. The numeric gates are one-sided, so extra
    span or extra opposition past the threshold does not count as distance.
    """

    return _evaluate_force_closure_gates(pair).ranking_key(
        require_current_impulse=require_current_impulse,
    )


def force_closure_contact_diagnostics(
    contacts: Sequence[G1GraspContact],
    *,
    object_center_world: Sequence[float],
    closing_axis_world: Sequence[float],
    gripper: GripperProfile,
    require_current_impulse: bool = False,
) -> ForceClosureContactDiagnostics:
    """Summarize the cross-role pair closest to the force-closure gate."""

    centre = _unit_geometry_vector(
        object_center_world,
        name="object_center_world",
        unit_length=False,
    )
    axis = _unit_geometry_vector(
        closing_axis_world,
        name="closing_axis_world",
        unit_length=True,
    )
    role_groups = tuple(gripper.opposing_contact_roles)
    if len(role_groups) != 2:
        raise ValueError(
            f"{gripper.name} must define exactly two opposing contact roles."
        )
    grouped = tuple(
        tuple(
            contact
            for contact in contacts
            if any(marker in contact.hand_link for marker in markers)
        )
        for markers in role_groups
    )
    contact_links = tuple(sorted({contact.hand_link for contact in contacts}))
    measured: list[G1OpposingContactPair] = []
    for first in grouped[0]:
        for second in grouped[1]:
            first_point = _dot(_subtract(first.position_xyz, centre), axis)
            second_point = _dot(_subtract(second.position_xyz, centre), axis)
            first_normal = _target_to_hand_normal(first)
            second_normal = _target_to_hand_normal(second)
            first_axis = _dot(first_normal, axis)
            second_axis = _dot(second_normal, axis)
            measured.append(
                G1OpposingContactPair(
                    first=first,
                    second=second,
                    first_point_projection_m=first_point,
                    second_point_projection_m=second_point,
                    first_normal_projection=first_axis,
                    second_normal_projection=second_axis,
                    normal_dot=_dot(first_normal, second_normal),
                    contact_span_m=abs(first_point - second_point),
                )
            )
    best = min(
        measured,
        key=lambda pair: _force_closure_gate_deficit(
            pair,
            require_current_impulse=require_current_impulse,
        ),
        default=None,
    )
    evaluation = None if best is None else _evaluate_force_closure_gates(best)
    return ForceClosureContactDiagnostics(
        contact_links=contact_links,
        first_role_contact=bool(grouped[0]),
        second_role_contact=bool(grouped[1]),
        best_contact_span_m=(
            None if evaluation is None else float(evaluation.contact_span_m)
        ),
        best_normal_dot=(None if evaluation is None else float(evaluation.normal_dot)),
        best_minimum_closing_axis_normal_projection=(
            None if evaluation is None else float(evaluation.min_abs_normal_projection)
        ),
        best_first_normal_projection=(
            None if evaluation is None else float(evaluation.first_normal_projection)
        ),
        best_second_normal_projection=(
            None if evaluation is None else float(evaluation.second_normal_projection)
        ),
        best_first_point_projection_m=(
            None if evaluation is None else float(evaluation.first_point_projection_m)
        ),
        best_second_point_projection_m=(
            None if evaluation is None else float(evaluation.second_point_projection_m)
        ),
        best_failed_gate_names=(
            ()
            if evaluation is None
            else evaluation.failed_gate_names(
                require_current_impulse=require_current_impulse,
            )
        ),
        best_span_deficit_m=(
            None if evaluation is None else float(evaluation.span_deficit)
        ),
        best_projection_deficit=(
            None if evaluation is None else float(evaluation.projection_deficit)
        ),
        best_normal_dot_deficit=(
            None if evaluation is None else float(evaluation.normal_dot_deficit)
        ),
        best_first_normal_impulse_ns=(
            None if evaluation is None else float(evaluation.first_normal_impulse_ns)
        ),
        best_second_normal_impulse_ns=(
            None if evaluation is None else float(evaluation.second_normal_impulse_ns)
        ),
    )


def _opposing_contact_pair(
    contacts: Sequence[G1GraspContact],
    *,
    object_center_world: Sequence[float],
    closing_axis_world: Sequence[float],
    gripper: GripperProfile,
    require_force_closure: bool,
    require_current_impulse: bool,
) -> Optional[G1OpposingContactPair]:
    """Reduce live contacts under one explicit geometry/force contract.

    PhysX reports a contact normal from the second collider toward the first.
    This reducer flips normals when the target occupies the first collider, so
    every retained normal points from the target surface toward the contacting
    hand link.  It then requires two gripper roles, contact points on opposite
    sides of the live object centre, opposed normals, and normals aligned with
    the same closing axis.  Link names alone never establish a grasp.

    ``require_current_impulse`` is true when a closure is first accepted.  A
    static retained grasp may legitimately have a zero impulse on a later
    solver tick, so hold-state revalidation uses the same geometry with this
    flag false.
    """

    centre = _unit_geometry_vector(
        object_center_world,
        name="object_center_world",
        unit_length=False,
    )
    axis = _unit_geometry_vector(
        closing_axis_world,
        name="closing_axis_world",
        unit_length=True,
    )
    role_groups = tuple(gripper.opposing_contact_roles)
    if len(role_groups) != 2:
        raise ValueError(
            f"{gripper.name} must define exactly two opposing contact roles."
        )
    grouped = tuple(
        tuple(
            contact
            for contact in contacts
            if any(marker in contact.hand_link for marker in markers)
        )
        for markers in role_groups
    )
    candidates: list[G1OpposingContactPair] = []
    for first in grouped[0]:
        for second in grouped[1]:
            first_point = _dot(_subtract(first.position_xyz, centre), axis)
            second_point = _dot(_subtract(second.position_xyz, centre), axis)
            first_normal = _target_to_hand_normal(first)
            second_normal = _target_to_hand_normal(second)
            first_axis = _dot(first_normal, axis)
            second_axis = _dot(second_normal, axis)
            pair = G1OpposingContactPair(
                first=first,
                second=second,
                first_point_projection_m=first_point,
                second_point_projection_m=second_point,
                first_normal_projection=first_axis,
                second_normal_projection=second_axis,
                normal_dot=_dot(first_normal, second_normal),
                contact_span_m=abs(first_point - second_point),
            )
            evaluation = _evaluate_force_closure_gates(pair)
            if not evaluation.opposite_sides:
                continue
            if evaluation.span_deficit != 0.0:
                continue
            if require_force_closure:
                if not evaluation.opposed_normals:
                    continue
                if evaluation.projection_deficit != 0.0:
                    continue
                if evaluation.normal_dot_deficit != 0.0:
                    continue
                if require_current_impulse and not evaluation.impulse_passed:
                    continue
            candidates.append(pair)
    return max(
        candidates,
        key=lambda pair: (
            pair.contact_span_m,
            -pair.normal_dot,
            abs(pair.first_normal_projection) + abs(pair.second_normal_projection),
        ),
        default=None,
    )


class G1GraspContactMonitor:
    """Maintain current hand–object contacts from native PhysX reports."""

    def __init__(
        self,
        *,
        hand_link_paths: Mapping[str, Mapping[str, str]],
        target_paths: Mapping[str, str],
        contact_report_prim_path: str,
        scene_object_paths: Mapping[str, str] | None = None,
    ) -> None:
        if set(hand_link_paths) != {"left", "right"}:
            raise ValueError(
                "G1 grasp contact monitor requires left and right Dex3 links"
            )
        if any(not paths for paths in hand_link_paths.values()):
            raise ValueError("Each Dex3 hand requires at least one physical link path")
        if not target_paths:
            raise ValueError(
                "G1 grasp contact monitor requires at least one target path"
            )
        self.hand_link_paths = {
            str(hand): {
                str(link_name): _normalized_path(path)
                for link_name, path in paths.items()
            }
            for hand, paths in hand_link_paths.items()
        }
        self.target_paths = {
            str(object_id): _normalized_path(path)
            for object_id, path in target_paths.items()
        }
        self.contact_report_prim_path = _normalized_path(contact_report_prim_path)
        self.scene_object_paths = {
            str(object_id): _normalized_path(path)
            for object_id, path in (scene_object_paths or target_paths).items()
        }
        self._subscription: Any = None
        self._contacts: dict[
            tuple[str, str],
            tuple[
                tuple[str, str],
                tuple[str, str],
                tuple[
                    tuple[float, float, float],
                    tuple[float, float, float],
                    tuple[float, float, float],
                    tuple[float, float, float],
                    float,
                ],
            ],
        ] = {}
        self._callback_error: Optional[Exception] = None

    def start(self, stage: Any) -> None:
        """Subscribe through the contact reporter authored before simulation.

        PhysX contact reporting is part of the robot asset contract.  Applying
        ``PhysxContactReportAPI`` to task objects after the stage is live asks
        PhysX to rebuild report state while simulation is stepping and can
        crash inside the native physics-sensor plugin.  One zero-threshold
        reporter on the articulation is sufficient: the callback receives its
        hand contacts and this monitor filters them to the declared targets.
        """
        if self._subscription is not None:
            raise RuntimeError("G1 grasp contact monitor is already running")
        from omni.physx import get_physx_simulation_interface
        from pxr import PhysxSchema

        report_prim = stage.GetPrimAtPath(self.contact_report_prim_path)
        if not report_prim or not report_prim.IsValid():
            raise RuntimeError(
                "Grasp contact reporting requires its authored robot prim: "
                f"{self.contact_report_prim_path}."
            )
        if not report_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            raise RuntimeError(
                "Grasp contact reporting was not authored before the robot "
                f"became live: {self.contact_report_prim_path}."
            )
        threshold = report_prim.GetAttribute("physxContactReport:threshold").Get()
        if threshold is None or float(threshold) != 0.0:
            raise RuntimeError(
                "Grasp contact reporting requires a zero-threshold authored "
                f"reporter: {self.contact_report_prim_path}."
            )
        for object_id, target_path in self.target_paths.items():
            target_prim = stage.GetPrimAtPath(target_path)
            if not target_prim or not target_prim.IsValid():
                raise RuntimeError(
                    f"Cannot monitor missing grasp target {object_id!r}: {target_path}"
                )

        interface = get_physx_simulation_interface()
        if interface is None:
            raise RuntimeError(
                "PhysX simulation interface is unavailable for grasp contact"
            )
        self._subscription = interface.subscribe_full_contact_report_events(
            self._on_contact_report
        )
        if self._subscription is None:
            raise RuntimeError("PhysX grasp contact subscription could not be created")

    def close(self) -> None:
        """Release the PhysX subscription and all transient contact state."""
        self._subscription = None
        self._contacts.clear()
        self._callback_error = None

    def contact(
        self,
        *,
        hand: str,
        target_object_id: str,
    ) -> Optional[G1GraspContact]:
        """Return the closest current contact for one exact hand–target pair."""
        return min(
            self.contacts(
                hand=hand,
                target_object_id=target_object_id,
            ),
            key=lambda item: item.separation_m,
            default=None,
        )

    def contacts(
        self,
        *,
        hand: str,
        target_object_id: str,
    ) -> tuple[G1GraspContact, ...]:
        """Return every eligible gripper-link contact for one hand and target."""
        self._raise_callback_error()
        hand_links = self.hand_link_paths.get(str(hand))
        target_path = self.target_paths.get(str(target_object_id))
        if hand_links is None:
            raise ValueError(f"Unknown G1 hand for contact query: {hand!r}")
        if target_path is None:
            raise ValueError(
                f"Unknown grasp target for contact query: {target_object_id!r}"
            )

        matches: list[G1GraspContact] = []
        for actor_paths, collider_paths, point in self._contacts.values():
            first_paths = (actor_paths[0], collider_paths[0])
            second_paths = (actor_paths[1], collider_paths[1])
            for link_name, link_path in hand_links.items():
                hand_first = _contains_path(first_paths, link_path) and _contains_path(
                    second_paths, target_path
                )
                hand_second = _contains_path(
                    second_paths, link_path
                ) and _contains_path(first_paths, target_path)
                if hand_first or hand_second:
                    matches.append(
                        G1GraspContact(
                            hand=str(hand),
                            hand_link=link_name,
                            target_object_id=str(target_object_id),
                            actor_paths=actor_paths,
                            collider_paths=collider_paths,
                            target_collider_index=1 if hand_first else 0,
                            position_xyz=point[0],
                            normal_xyz=point[1],
                            normal_impulse_xyz=point[2],
                            friction_impulse_xyz=point[3],
                            separation_m=point[4],
                        )
                    )
                    break
        return tuple(
            item for item in matches if item.separation_m <= G1_GRASP_CONTACT_GAP_M
        )

    def current_contacts(self) -> tuple[G1GraspContact, ...]:
        """Return every currently eligible task-object contact."""
        contacts = []
        for hand in self.hand_link_paths:
            for object_id in self.target_paths:
                contacts.extend(
                    self.contacts(
                        hand=hand,
                        target_object_id=object_id,
                    )
                )
        return tuple(contacts)

    def support_contact(
        self,
        *,
        object_id: str,
        destination_object_id: str,
    ) -> Optional[G1SupportContact]:
        """Return the closest live object--destination PhysX contact."""

        self._raise_callback_error()
        object_path = self.scene_object_paths.get(str(object_id))
        destination_path = self.scene_object_paths.get(str(destination_object_id))
        if object_path is None or destination_path is None:
            raise ValueError(
                "Unknown manipulation object pair for support contact: "
                f"{object_id!r}, {destination_object_id!r}."
            )
        matches = []
        for _, collider_paths, point in self._contacts.values():
            if not (
                _contains_path((collider_paths[0],), object_path)
                and _contains_path((collider_paths[1],), destination_path)
                or _contains_path((collider_paths[1],), object_path)
                and _contains_path((collider_paths[0],), destination_path)
            ):
                continue
            matches.append(
                G1SupportContact(
                    object_id=str(object_id),
                    destination_object_id=str(destination_object_id),
                    collider_paths=collider_paths,
                    separation_m=float(point[4]),
                )
            )
        return min(matches, key=lambda item: item.separation_m, default=None)

    def _on_contact_report(
        self,
        headers: Any,
        contact_data: Any,
        friction_anchors: Any,
    ) -> None:
        """Copy one-step PhysX buffers before their callback lifetime ends."""
        try:
            self._consume_contact_report(
                headers,
                contact_data,
                friction_anchors,
            )
        except Exception as exc:
            self._callback_error = exc

    def _consume_contact_report(
        self,
        headers: Any,
        contact_data: Any,
        friction_anchors: Any,
    ) -> None:
        from pxr import PhysicsSchemaTools

        for header in headers:
            actors = (
                str(PhysicsSchemaTools.intToSdfPath(header.actor0)),
                str(PhysicsSchemaTools.intToSdfPath(header.actor1)),
            )
            colliders = (
                str(PhysicsSchemaTools.intToSdfPath(header.collider0)),
                str(PhysicsSchemaTools.intToSdfPath(header.collider1)),
            )
            key = tuple(sorted(colliders))
            event_name = str(getattr(header.type, "name", header.type))
            event_name = event_name.rsplit(".", 1)[-1]
            if event_name == "CONTACT_LOST":
                self._contacts.pop(key, None)
                continue
            if event_name not in {"CONTACT_FOUND", "CONTACT_PERSIST"}:
                raise RuntimeError(f"Unknown PhysX contact event type: {event_name!r}")
            start = int(header.contact_data_offset)
            end = start + int(header.num_contact_data)
            friction_start = int(header.friction_anchors_offset)
            friction_end = friction_start + int(header.num_friction_anchors_data)
            friction_impulse = tuple(
                sum(
                    _point3(friction_anchors[index].impulse)[axis]
                    for index in range(friction_start, friction_end)
                )
                for axis in range(3)
            )
            points = [
                (
                    _point3(contact_data[index].position),
                    _point3(contact_data[index].normal),
                    _point3(contact_data[index].impulse),
                    friction_impulse,
                    float(contact_data[index].separation),
                )
                for index in range(start, end)
            ]
            if not points:
                raise RuntimeError("PhysX contact event omitted its contact points")
            if any(not math.isfinite(separation) for _, _, _, _, separation in points):
                raise RuntimeError(
                    "PhysX contact event contained a non-finite separation"
                )
            self._contacts[key] = (
                actors,
                colliders,
                min(points, key=lambda value: value[4]),
            )

    def _raise_callback_error(self) -> None:
        if self._callback_error is not None:
            raise RuntimeError(
                "PhysX grasp contact reporting failed"
            ) from self._callback_error


def _normalized_path(value: object) -> str:
    path = str(value or "").strip().rstrip("/")
    if not path.startswith("/"):
        raise ValueError(f"USD prim path must be absolute: {value!r}")
    return path


def _contains_path(paths: tuple[str, ...], root: str) -> bool:
    return any(path == root or path.startswith(root + "/") for path in paths)


def _unit_geometry_vector(
    values: Sequence[float],
    *,
    name: str,
    unit_length: bool,
) -> tuple[float, float, float]:
    try:
        vector = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite three-vector.") from exc
    if len(vector) != 3 or not all(math.isfinite(value) for value in vector):
        raise ValueError(f"{name} must be a finite three-vector.")
    if not unit_length:
        return vector  # type: ignore[return-value]
    length = _magnitude(vector)
    if not math.isclose(length, 1.0, rel_tol=0.0, abs_tol=2e-3):
        raise ValueError(f"{name} must be a unit vector; norm={length:.6f}.")
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def _target_to_hand_normal(
    contact: G1GraspContact,
) -> tuple[float, float, float]:
    """Normalize PhysX's shape-1→shape-0 normal to target→hand."""

    target_index = int(contact.target_collider_index)
    if target_index not in (0, 1):
        raise ValueError(
            "A grasp contact target_collider_index must be 0 or 1; got "
            f"{target_index!r}."
        )
    normal = _unit_geometry_vector(
        contact.normal_xyz,
        name="contact.normal_xyz",
        unit_length=True,
    )
    # PxContactPairPoint.normal points from shape 1 to shape 0. It already
    # points target→hand when the target is shape 1; otherwise flip it.
    sign = 1.0 if target_index == 1 else -1.0
    return tuple(sign * value for value in normal)  # type: ignore[return-value]


def _subtract(
    first: Sequence[float],
    second: Sequence[float],
) -> tuple[float, float, float]:
    return tuple(
        float(left) - float(right) for left, right in zip(first, second, strict=True)
    )  # type: ignore[return-value]


def _dot(first: Sequence[float], second: Sequence[float]) -> float:
    return float(
        sum(
            float(left) * float(right)
            for left, right in zip(first, second, strict=True)
        )
    )


def _magnitude(vector: Sequence[float]) -> float:
    return math.sqrt(_dot(vector, vector))


def _point3(value: Any) -> tuple[float, float, float]:
    try:
        point = tuple(float(value[index]) for index in range(3))
    except (IndexError, KeyError, TypeError, ValueError) as exc:
        raise RuntimeError(
            f"PhysX contact position is not a finite 3-D point: {value!r}"
        ) from exc
    if not all(math.isfinite(item) for item in point):
        raise RuntimeError(f"PhysX contact position is not finite: {point!r}")
    return point  # type: ignore[return-value]


__all__ = [
    "FORCE_CLOSURE_GATE_CLOSING_AXIS_NORMAL_PROJECTION",
    "FORCE_CLOSURE_GATE_CONTACT_SPAN",
    "FORCE_CLOSURE_GATE_NORMAL_IMPULSE",
    "FORCE_CLOSURE_GATE_OPPOSING_NORMAL_DOT",
    "FORCE_CLOSURE_GATE_OPPOSITE_NORMAL_SIGNS",
    "FORCE_CLOSURE_GATE_OPPOSITE_SIDES",
    "force_closure_contact_diagnostics",
    "force_closure_contact_pair",
    "ForceClosureContactDiagnostics",
    "grasp_contact_role_pair",
    "G1GraspContact",
    "G1GraspContactMonitor",
    "G1OpposingContactPair",
    "G1SupportContact",
    "opposing_grasp_contact_pair",
]
