"""Reduce one measured ``place_object`` transition to a shared verdict.

Runtime supplies measured values. Evaluation and certification use the same
verdict function; runtime success flags do not establish placement success.

``pre_release`` owns the instant immediately before the hand starts opening::

    {
        "decision_step": int,
        "physics_step": int,
        "object_id": str,
        "destination_object_id": str,
        "hand": "left" | "right",
        "retention": { ... raw physical state ... },
        "target_bbox_min": [x, y, z],
        "target_bbox_max": [x, y, z],
        "destination_bbox_min": [x, y, z],
        "destination_bbox_max": [x, y, z],
        "support_contact": {
            "object_id": str,
            "destination_object_id": str,
            "separation_m": float
        }
    }

For ``physical`` retention, the raw state contains ``attached``, both
attachment-joint flags, ``physically_held``, ``held_by_hand``, and
``live_hand_contact_count``.

``hand_opening`` owns the measured joint state after the same macro-action's
opening trajectory::

    {
        "decision_step": int,
        "physics_step": int,
        "hand": "left" | "right",
        "measured_joint_positions_rad": [float, ...],
        "open_joint_positions_rad": [float, ...],
        "joint_tolerance_rad": float
    }

``post_snapshot`` and every settle snapshot use the collector's ordinary
``{step, decision, manipulation.objects}`` shape. The post snapshot is the
state after ``place_object`` finishes. Settle snapshots must begin at the next
decision step and remain consecutive; missing frames are never skipped.
"""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Optional, Sequence, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.infra.control.physical_contracts import (
    G1_PLACEMENT_SETTLE_MOTION_M,
    G1_PLACEMENT_RELATION_TOLERANCE_M,
    G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M,
)


Point3 = Tuple[float, float, float]
_EPS = 1e-9


class PlacementTransitionUnavailableError(RuntimeError):
    """Raised when a trace contains no structurally measurable place action."""


class PlacementTransitionVerdict(BaseModel):
    """Immutable placement result shared by scoring and certification."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    target_object_id: str = Field(
        min_length=1,
        description="Grounded object whose held-to-supported transition was measured.",
    )
    destination_object_id: str = Field(
        min_length=1,
        description="Grounded object intended to support the released target.",
    )
    retention_mode: Literal["physical"] = Field(
        default="physical",
        description="Finite-friction physical retention contract.",
    )
    place_step: int = Field(
        ge=0,
        description="Decision step containing the explicit place_object action.",
    )
    required_settle_samples: int = Field(
        ge=2,
        description="Number of consecutive post-release snapshots required to pass.",
    )
    observed_settle_samples: int = Field(
        ge=0,
        description="Number of present snapshots inside the required settle prefix.",
    )
    retained_before_release: bool = Field(
        description=(
            "Whether the target satisfied the configured retention mode before opening."
        ),
    )
    support_contact_valid: bool = Field(
        description=(
            "Whether target and destination support met within the fixed "
            "5 mm contact band."
        ),
    )
    opening_follows_support_contact: bool = Field(
        description=(
            "Whether measured hand opening started after the pre-release "
            "support sample."
        ),
    )
    hand_opened: bool = Field(
        description=(
            "Whether every measured hand joint reached its authored open target."
        ),
    )
    released_state_valid: bool = Field(
        description=(
            "Whether the same-step post state is dynamic, unretained, and on "
            "destination."
        ),
    )
    complete_settle_window: bool = Field(
        description="Whether every snapshot in the fixed settle prefix is present.",
    )
    consecutive_settle_window: bool = Field(
        description=(
            "Whether settle steps are exactly place_step+1 through place_step+N."
        ),
    )
    settled_relation_valid: bool = Field(
        description=(
            "Whether every settle snapshot remains dynamic, released, and on "
            "destination."
        ),
    )
    settle_motion_valid: bool = Field(
        description=(
            "Whether all consecutive target-centre motions remain within tolerance."
        ),
    )
    support_contact_separation_m: float = Field(
        description=(
            "Signed target-destination contact separation sampled before opening."
        ),
    )
    maximum_opening_error_rad: float = Field(
        ge=0.0,
        description="Largest absolute measured-to-open hand-joint error in radians.",
    )
    maximum_settle_motion_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Largest target-centre motion from release through the settle window."
        ),
    )
    passed: bool = Field(
        description=(
            "Whether every causal and physical requirement of placement passed."
        ),
    )
    failure_reason: Optional[str] = Field(
        default=None,
        description=(
            "First failed placement requirement in causal order; omitted on pass."
        ),
    )

    @model_validator(mode="after")
    def validate_outcome(self) -> "PlacementTransitionVerdict":
        conditions = (
            self.retained_before_release,
            self.support_contact_valid,
            self.opening_follows_support_contact,
            self.hand_opened,
            self.released_state_valid,
            self.complete_settle_window,
            self.consecutive_settle_window,
            self.settled_relation_valid,
            self.settle_motion_valid,
        )
        if self.passed != all(conditions):
            raise ValueError("placement verdict disagrees with its measured conditions")
        if self.passed != (self.failure_reason is None):
            raise ValueError("placement failure_reason presence disagrees with passed")
        return self


def evaluate_placement_transition(
    *,
    pre_release: Mapping[str, Any],
    hand_opening: Mapping[str, Any],
    post_snapshot: Mapping[str, Any],
    settle_snapshots: Sequence[Mapping[str, Any] | None],
    target_object_id: str,
    destination_object_id: str,
    required_settle_samples: int,
    relation_tolerance_m: float = G1_PLACEMENT_RELATION_TOLERANCE_M,
    motion_tolerance_m: float = G1_PLACEMENT_SETTLE_MOTION_M,
) -> PlacementTransitionVerdict:
    """Return the formal placement verdict from raw transition evidence.

    ``relation_tolerance_m`` belongs only to released task geometry.
    Pre-release support contact always uses the frozen 5 mm physical contract.
    Structurally malformed evidence raises; well-formed failed evidence returns
    a negative verdict with one causal failure reason.
    """

    target_id = _nonempty_text(target_object_id, "target_object_id")
    destination_id = _nonempty_text(destination_object_id, "destination_object_id")
    if target_id == destination_id:
        raise ValueError("placement target and destination must be distinct")
    if (
        isinstance(required_settle_samples, bool)
        or not isinstance(required_settle_samples, int)
        or required_settle_samples < 2
    ):
        raise ValueError("required_settle_samples must be an integer of at least two")
    if not math.isfinite(relation_tolerance_m) or relation_tolerance_m <= 0.0:
        raise ValueError("relation_tolerance_m must be positive and finite")
    if not math.isfinite(motion_tolerance_m) or motion_tolerance_m < 0.0:
        raise ValueError("motion_tolerance_m must be non-negative and finite")

    place_step, post_target, post_destination = _snapshot(
        post_snapshot,
        target_object_id=target_id,
        destination_object_id=destination_id,
        require_place_action=True,
    )
    pre_step = _required_int(pre_release, "decision_step", "pre_release")
    pre_physics_step = _required_int(pre_release, "physics_step", "pre_release")
    opening_step = _required_int(hand_opening, "decision_step", "hand_opening")
    opening_physics_step = _required_int(hand_opening, "physics_step", "hand_opening")
    pre_hand = _hand(_required(pre_release, "hand", "pre_release"))
    opening_hand = _hand(_required(hand_opening, "hand", "hand_opening"))
    if pre_step != place_step or opening_step != place_step:
        raise ValueError(
            "pre-release, hand-opening, and post snapshots must share one decision step"
        )
    if (
        _nonempty_text(
            _required(pre_release, "object_id", "pre_release"),
            "pre_release.object_id",
        )
        != target_id
    ):
        raise ValueError("pre_release object_id does not match target_object_id")
    if (
        _nonempty_text(
            _required(pre_release, "destination_object_id", "pre_release"),
            "pre_release.destination_object_id",
        )
        != destination_id
    ):
        raise ValueError(
            "pre_release destination_object_id does not match destination_object_id"
        )

    retention = _required_mapping(pre_release, "retention", "pre_release")
    retained = _retention_holds(
        retention,
        target_object_id=target_id,
        hand=pre_hand,
    )
    support_separation = _support_contact_separation(
        pre_release,
        target_object_id=target_id,
        destination_object_id=destination_id,
    )
    support_valid = _support_contact_holds(
        pre_release,
        separation_m=support_separation,
    )
    opening_after_contact = bool(
        opening_hand == pre_hand and opening_physics_step > pre_physics_step
    )
    opening_error, opened = _hand_opening_holds(hand_opening)

    released_state_valid = bool(
        _released(post_target)
        and post_target.get("kinematic_enabled") is False
        and _support_relation_holds(
            post_target,
            post_destination,
            tolerance_m=relation_tolerance_m,
        )
    )

    required_prefix = tuple(settle_snapshots[:required_settle_samples])
    observed = sum(isinstance(item, Mapping) for item in required_prefix)
    complete = bool(
        len(required_prefix) == required_settle_samples
        and all(isinstance(item, Mapping) for item in required_prefix)
    )
    settle_records: tuple[tuple[int, Mapping[str, Any], Mapping[str, Any]], ...] = ()
    if complete:
        settle_records = tuple(
            _snapshot(
                item,
                target_object_id=target_id,
                destination_object_id=destination_id,
                require_place_action=False,
            )
            for item in required_prefix
            if isinstance(item, Mapping)
        )
    consecutive = bool(
        complete
        and tuple(record[0] for record in settle_records)
        == tuple(range(place_step + 1, place_step + 1 + required_settle_samples))
    )
    settled_relation = bool(
        complete
        and all(
            _released(target)
            and target.get("kinematic_enabled") is False
            and _support_relation_holds(
                target,
                destination,
                tolerance_m=relation_tolerance_m,
            )
            for _, target, destination in settle_records
        )
    )
    centers = (
        (_bbox_center(post_target),)
        + tuple(_bbox_center(target) for _, target, _ in settle_records)
        if complete
        else ()
    )
    motions = tuple(
        math.dist(first, second) for first, second in zip(centers, centers[1:])
    )
    maximum_motion = max(motions) if motions else None
    motion_valid = bool(
        complete
        and maximum_motion is not None
        and maximum_motion <= motion_tolerance_m + _EPS
    )

    conditions = (
        ("target_not_retained_before_release", retained),
        ("support_contact_not_established", support_valid),
        ("hand_opening_did_not_follow_support_contact", opening_after_contact),
        ("hand_did_not_reach_open_target", opened),
        ("released_state_invalid", released_state_valid),
        ("settle_window_incomplete", complete),
        ("settle_window_not_consecutive", consecutive),
        ("support_relation_failed_during_settle", settled_relation),
        ("target_motion_exceeded_settle_tolerance", motion_valid),
    )
    failure_reason = next(
        (reason for reason, condition in conditions if not condition),
        None,
    )
    return PlacementTransitionVerdict(
        target_object_id=target_id,
        destination_object_id=destination_id,
        place_step=place_step,
        required_settle_samples=required_settle_samples,
        observed_settle_samples=observed,
        retained_before_release=retained,
        support_contact_valid=support_valid,
        opening_follows_support_contact=opening_after_contact,
        hand_opened=opened,
        released_state_valid=released_state_valid,
        complete_settle_window=complete,
        consecutive_settle_window=consecutive,
        settled_relation_valid=settled_relation,
        settle_motion_valid=motion_valid,
        support_contact_separation_m=support_separation,
        maximum_opening_error_rad=opening_error,
        maximum_settle_motion_m=maximum_motion,
        passed=failure_reason is None,
        failure_reason=failure_reason,
    )


def evaluate_placement_trace(
    steps: Sequence[Any],
    *,
    target_object_id: str,
    destination_object_id: str,
    required_settle_samples: int,
    relation_tolerance_m: float = G1_PLACEMENT_RELATION_TOLERANCE_M,
    motion_tolerance_m: float = G1_PLACEMENT_SETTLE_MOTION_M,
) -> PlacementTransitionVerdict:
    """Measure the first passing place attempt, or the last valid attempt.

    An Agent may recover from an unsuccessful placement and issue another
    ``place_object`` action. Every attempt must still carry the complete raw
    evidence contract. Malformed attempts are recorded as diagnostics but
    never interpreted as a release; if none is measurable the trace fails
    loudly instead of falling back to a relation-only snapshot.
    """

    normalized = tuple(_trace_step(item) for item in steps)
    verdicts: list[PlacementTransitionVerdict] = []
    errors: list[str] = []
    for index, step in enumerate(normalized):
        decision = step.get("decision")
        if (
            not isinstance(decision, Mapping)
            or decision.get("action") != "place_object"
        ):
            continue
        report = step.get("placement_release")
        if not isinstance(report, Mapping):
            errors.append(f"step {step.get('step')}: placement_release missing")
            continue
        pre_release = report.get("pre_release")
        hand_opening = report.get("hand_opening")
        if not isinstance(pre_release, Mapping) or not isinstance(
            hand_opening, Mapping
        ):
            errors.append(f"step {step.get('step')}: raw release evidence incomplete")
            continue
        try:
            verdict = evaluate_placement_transition(
                pre_release=pre_release,
                hand_opening=hand_opening,
                post_snapshot=step,
                settle_snapshots=normalized[
                    index + 1 : index + 1 + required_settle_samples
                ],
                target_object_id=target_object_id,
                destination_object_id=destination_object_id,
                required_settle_samples=required_settle_samples,
                relation_tolerance_m=relation_tolerance_m,
                motion_tolerance_m=motion_tolerance_m,
            )
        except (TypeError, ValueError) as exc:
            errors.append(f"step {step.get('step')}: {exc}")
            continue
        verdicts.append(verdict)
        if verdict.passed:
            return verdict
    if verdicts:
        return verdicts[-1]
    detail = "; ".join(errors) if errors else "no place_object action"
    raise PlacementTransitionUnavailableError(
        f"Placement transition is not measurable: {detail}."
    )


def _trace_step(value: Any) -> Mapping[str, Any]:
    if isinstance(value, Mapping):
        return value
    serializer = getattr(value, "as_dict", None)
    if callable(serializer):
        serialized = serializer()
        if isinstance(serialized, Mapping):
            return serialized
    raise TypeError("Placement trace steps must be mappings or expose as_dict().")


def _retention_holds(
    evidence: Mapping[str, Any],
    *,
    target_object_id: str,
    hand: str,
) -> bool:
    if (
        _nonempty_text(
            _required(evidence, "object_id", "pre_release.retention"),
            "pre_release.retention.object_id",
        )
        != target_object_id
    ):
        return False
    return bool(
        not _required_bool(evidence, "attached", "pre_release.retention")
        and not _required_bool(
            evidence, "attachment_joint_present", "pre_release.retention"
        )
        and not _required_bool(
            evidence, "attachment_joint_valid", "pre_release.retention"
        )
        and _required_bool(evidence, "physically_held", "pre_release.retention")
        and _required(evidence, "held_by_hand", "pre_release.retention") == hand
        and _required_int(evidence, "live_hand_contact_count", "pre_release.retention")
        > 0
    )


def _support_contact_separation(
    pre_release: Mapping[str, Any],
    *,
    target_object_id: str,
    destination_object_id: str,
) -> float:
    contact = _required_mapping(pre_release, "support_contact", "pre_release")
    if (
        _nonempty_text(
            _required(contact, "object_id", "pre_release.support_contact"),
            "pre_release.support_contact.object_id",
        )
        != target_object_id
    ):
        raise ValueError("support contact object_id does not match target")
    if (
        _nonempty_text(
            _required(contact, "destination_object_id", "pre_release.support_contact"),
            "pre_release.support_contact.destination_object_id",
        )
        != destination_object_id
    ):
        raise ValueError(
            "support contact destination_object_id does not match destination"
        )
    return _required_float(contact, "separation_m", "pre_release.support_contact")


def _support_contact_holds(
    pre_release: Mapping[str, Any],
    *,
    separation_m: float,
) -> bool:
    target_min, target_max = _bounds_from_fields(
        pre_release,
        minimum_field="target_bbox_min",
        maximum_field="target_bbox_max",
        label="pre_release target",
    )
    destination_min, destination_max = _bounds_from_fields(
        pre_release,
        minimum_field="destination_bbox_min",
        maximum_field="destination_bbox_max",
        label="pre_release destination",
    )
    overlap_x = min(target_max[0], destination_max[0]) - max(
        target_min[0], destination_min[0]
    )
    overlap_y = min(target_max[1], destination_max[1]) - max(
        target_min[1], destination_min[1]
    )
    signed_vertical_gap = target_min[2] - destination_max[2]
    tolerance = G1_PRE_RELEASE_SUPPORT_CONTACT_TOLERANCE_M + _EPS
    return bool(
        overlap_x > 0.0
        and overlap_y > 0.0
        and abs(signed_vertical_gap) <= tolerance
        and abs(separation_m) <= tolerance
    )


def _hand_opening_holds(evidence: Mapping[str, Any]) -> tuple[float, bool]:
    measured = _float_sequence(
        _required(evidence, "measured_joint_positions_rad", "hand_opening"),
        "hand_opening.measured_joint_positions_rad",
    )
    target = _float_sequence(
        _required(evidence, "open_joint_positions_rad", "hand_opening"),
        "hand_opening.open_joint_positions_rad",
    )
    if len(measured) != len(target):
        raise ValueError("measured and open hand-joint vectors must have equal length")
    tolerance = _required_float(evidence, "joint_tolerance_rad", "hand_opening")
    if tolerance <= 0.0:
        raise ValueError("hand_opening.joint_tolerance_rad must be positive")
    maximum_error = max(
        abs(actual - expected)
        for actual, expected in zip(measured, target, strict=True)
    )
    return maximum_error, maximum_error <= tolerance + _EPS


def _released(target: Mapping[str, Any]) -> bool:
    unconstrained = bool(
        not _required_bool(target, "attached", "target snapshot")
        and not _required_bool(target, "attachment_joint_present", "target snapshot")
        and not _required_bool(target, "attachment_joint_valid", "target snapshot")
    )
    return bool(
        unconstrained
        and not _required_bool(target, "physically_held", "target snapshot")
        and target.get("held_by_hand") not in {"left", "right"}
    )


def _support_relation_holds(
    target: Mapping[str, Any],
    destination: Mapping[str, Any],
    *,
    tolerance_m: float,
) -> bool:
    target_min, target_max = _bounds(target, label="target snapshot")
    destination_min, destination_max = _bounds(
        destination, label="destination snapshot"
    )
    overlap_x = min(target_max[0], destination_max[0]) - max(
        target_min[0], destination_min[0]
    )
    overlap_y = min(target_max[1], destination_max[1]) - max(
        target_min[1], destination_min[1]
    )
    return bool(
        overlap_x > 0.0
        and overlap_y > 0.0
        and abs(target_min[2] - destination_max[2]) <= tolerance_m + _EPS
    )


def _snapshot(
    value: Mapping[str, Any],
    *,
    target_object_id: str,
    destination_object_id: str,
    require_place_action: bool,
) -> tuple[int, Mapping[str, Any], Mapping[str, Any]]:
    label = "post_snapshot" if require_place_action else "settle snapshot"
    step = _required_int(value, "step", label)
    if require_place_action:
        decision = _required_mapping(value, "decision", label)
        action = _nonempty_text(
            _required(decision, "action", f"{label}.decision"),
            f"{label}.decision.action",
        )
        if action != "place_object":
            raise ValueError(
                f"post_snapshot requires explicit place_object, got {action!r}"
            )
    manipulation = _required_mapping(value, "manipulation", label)
    objects = _required_mapping(manipulation, "objects", f"{label}.manipulation")
    target = _required_mapping(objects, target_object_id, f"{label}.objects")
    destination = _required_mapping(objects, destination_object_id, f"{label}.objects")
    return step, target, destination


def _bbox_center(state: Mapping[str, Any]) -> Point3:
    lower, upper = _bounds(state, label="target snapshot")
    return tuple((lower[axis] + upper[axis]) / 2.0 for axis in range(3))


def _bounds(state: Mapping[str, Any], *, label: str) -> tuple[Point3, Point3]:
    return _bounds_from_fields(
        state,
        minimum_field="bbox_min",
        maximum_field="bbox_max",
        label=label,
    )


def _bounds_from_fields(
    state: Mapping[str, Any],
    *,
    minimum_field: str,
    maximum_field: str,
    label: str,
) -> tuple[Point3, Point3]:
    lower = _point3(_required(state, minimum_field, label), f"{label}.{minimum_field}")
    upper = _point3(_required(state, maximum_field, label), f"{label}.{maximum_field}")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        raise ValueError(f"{label} requires an ordered world AABB")
    return lower, upper


def _point3(value: Any, label: str) -> Point3:
    values = _float_sequence(value, label)
    if len(values) != 3:
        raise ValueError(f"{label} must contain exactly three finite values")
    return values[0], values[1], values[2]


def _float_sequence(value: Any, label: str) -> tuple[float, ...]:
    if not isinstance(value, (list, tuple)) or not value:
        raise ValueError(f"{label} must be a non-empty list of finite numbers")
    result = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in result):
        raise ValueError(f"{label} must contain only finite numbers")
    return result


def _required(mapping: Mapping[str, Any], field: str, label: str) -> Any:
    if field not in mapping:
        raise ValueError(f"{label} is missing required field {field!r}")
    return mapping[field]


def _required_mapping(
    mapping: Mapping[str, Any], field: str, label: str
) -> Mapping[str, Any]:
    value = _required(mapping, field, label)
    if not isinstance(value, Mapping):
        raise ValueError(f"{label}.{field} must be a mapping")
    return value


def _required_bool(mapping: Mapping[str, Any], field: str, label: str) -> bool:
    value = _required(mapping, field, label)
    if not isinstance(value, bool):
        raise ValueError(f"{label}.{field} must be a Boolean")
    return value


def _required_int(mapping: Mapping[str, Any], field: str, label: str) -> int:
    value = _required(mapping, field, label)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{label}.{field} must be a non-negative integer")
    return value


def _required_float(mapping: Mapping[str, Any], field: str, label: str) -> float:
    value = float(_required(mapping, field, label))
    if not math.isfinite(value):
        raise ValueError(f"{label}.{field} must be finite")
    return value


def _nonempty_text(value: Any, label: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be text")
    result = value.strip()
    if not result:
        raise ValueError(f"{label} must be non-empty")
    return result


def _hand(value: Any) -> str:
    hand = str(value).strip().lower()
    if hand not in {"left", "right"}:
        raise ValueError(f"hand must be left or right, got {value!r}")
    return hand


__all__ = [
    "PlacementTransitionUnavailableError",
    "PlacementTransitionVerdict",
    "evaluate_placement_trace",
    "evaluate_placement_transition",
]
