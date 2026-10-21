"""Grasp failures and authorized-target geometry shared with certification.

SAM3, synchronized depth and GraspGenX supply evaluated grasp proposals.
This module matches observed target centres to authorized pickables and defines
stance/reach failures and canonical certification geometry.

``GraspNotPossibleHere`` reports a recoverable stance or reach failure to the
agent and raises during certification. Invalid authored geometry, missing
object-class grasps and incompatible gripper openings remain authoring errors."""

from __future__ import annotations

import math
from typing import Any, Dict, Mapping, Sequence, Tuple

from tarsphere.infra.action_errors import ActionRefusedError


class GraspFailureCode:
    """Stable reasons an attempted grasp can fail without ending an episode."""

    TARGET_NOT_RESOLVED = "target_not_resolved"
    TARGET_OUT_OF_VIEW = "target_out_of_view"
    GROUNDING_GEOMETRY_STALE = "grounding_geometry_stale"
    STANCE_FACING = "stance_facing"
    REACH_ENVELOPE = "reach_envelope"
    # A planned reach executed, but the measured palm did not reach its goal.
    REACH_ALIGNMENT_FAILED = "reach_alignment_failed"
    CONTACT_MISS = "contact_miss"
    RETENTION_LOST = "retention_lost"
    SUPPORT_UNSTABLE = "support_unstable"


GRASP_FAILURE_CODES = (
    GraspFailureCode.TARGET_NOT_RESOLVED,
    GraspFailureCode.TARGET_OUT_OF_VIEW,
    GraspFailureCode.GROUNDING_GEOMETRY_STALE,
    GraspFailureCode.STANCE_FACING,
    GraspFailureCode.REACH_ENVELOPE,
    GraspFailureCode.REACH_ALIGNMENT_FAILED,
    GraspFailureCode.CONTACT_MISS,
    GraspFailureCode.RETENTION_LOST,
    GraspFailureCode.SUPPORT_UNSTABLE,
)


class GraspNotPossibleHere(ActionRefusedError):
    """An attempted grasp failed for a condition the Agent can change."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        evidence: Mapping[str, Any] | None = None,
        reachability: Mapping[str, Any] | None = None,
    ) -> None:
        if code not in GRASP_FAILURE_CODES:
            raise ValueError(
                f"{code!r} is not a grasp failure code; expected one of "
                f"{list(GRASP_FAILURE_CODES)}."
            )
        super().__init__(message)
        self.code = code
        self.evidence = dict(evidence or {})
        self.reachability = dict(reachability or {})

    def as_feedback(self) -> Dict[str, Any]:
        """Return the public, serializable failure reported to the Agent."""

        return {
            "code": self.code,
            "message": str(self),
            "evidence": dict(self.evidence),
        }


def closing_palm_residual_m(planned_world, measured_world) -> float:
    """Measure finite reach error; only real contact and retention prove grasp.

    The 5 mm feedback-correction stopping target is not a physical grasp
    acceptance threshold. The subsequent joint-tracking, Dex3 force-closure,
    unconstrained retention and support-clearance checks remain mandatory.
    """
    if (
        planned_world is None
        or measured_world is None
        or len(planned_world) != 3
        or len(measured_world) != 3
        or not all(math.isfinite(float(v)) for v in (*planned_world, *measured_world))
    ):
        raise GraspNotPossibleHere(
            GraspFailureCode.REACH_ALIGNMENT_FAILED,
            "Closing requires finite planned and measured world palm positions; fingers remain open.",
        )
    return math.dist(planned_world, measured_world)


# Largest horizontal registration gap between the current SAM3-mask depth
# centre and the authorized pickable it names. The bound absorbs sensor noise;
# it does not generate grasp geometry, which always comes from GraspGenX.
GRASP_GROUNDING_MATCH_RADIUS_M = 0.35

# Runtime-owned labels for which mask produced the executed grasp.
GRASP_GROUNDING_SAM3 = "sam3"
GRASP_GROUNDING_MEMORY = "memory"

# Minimum disk radius when re-projecting a remembered bbox into the current
# view. GraspGenX needs roughly thirty valid depth pixels inside the mask;
# a smaller disk would hand it mostly background at long range.
GRASP_REMEMBERED_GROUNDING_MIN_DISK_RADIUS_PX = 4

# Agent-facing copy for runtime-owned grounding fallbacks. The evaluated
# agent may only hear that the target is not usable from here; it must not
# learn that a ground-truth mask exists or where the true object projects.
GRASP_FALLBACK_TARGET_NOT_IN_VIEW = (
    "The grounder did not lock onto the target you named from the current "
    "view; it may have missed it (small, backlit, or unusual phrasing), which "
    "is not proof of absence. Try a plainer or more specific phrase, or "
    "reposition for a closer view, then ask again."
)
GRASP_FALLBACK_VIEW_UNUSABLE = (
    "The current view has no usable depth for grasping. Reposition or look "
    "around, then observe the target again."
)
GRASP_FALLBACK_VIEW_TOO_SPARSE = (
    "Nothing you can pick up is seen clearly enough from here: the current "
    "view is too sparse for a grasp. Observe it again from closer up."
)
GRASP_FALLBACK_TARGET_UNAVAILABLE = (
    "Nothing you can pick up matches the current target: no graspable object "
    "is available from here. Observe it again from closer up."
)
GRASP_FALLBACK_NO_GRASP_POSE = (
    "Nothing you can pick up has a usable grasp from the current view. "
    "Observe it again from closer up or a different angle."
)
GRASP_FALLBACK_NEVER_GROUNDED = (
    "Nothing you can pick up matches the target you named: you have not yet "
    "grounded it in this episode. Use locate_object or grasp_object while the "
    "target is visible in the current ego view."
)
GRASP_FALLBACK_GROUNDING_GEOMETRY_STALE = (
    "You still know which object you were grasping, but its remembered position "
    "is no longer trustworthy after something moved it. Observe that target "
    "again in the current ego view before grasping."
)
GRASP_REMEMBERED_GROUNDING_NOTICE = (
    "The target you named is not visible in the current view. Grasping uses "
    "grounding from your earlier observation at action step {action_step}."
)


def agent_safe_grasp_refusal(
    code: str,
    message: str,
    *,
    evaluator_detail: str,
    evidence: Mapping[str, Any] | None = None,
) -> GraspNotPossibleHere:
    """Build a grasp refusal whose public text stays agent-legible.

    ``message`` is what the evaluated agent may read. ``evaluator_detail``
    stays on the private evidence record with any coordinates or
    ground-truth diagnostics.
    """

    payload = dict(evidence or {})
    payload["evaluator_detail"] = str(evaluator_detail)
    return GraspNotPossibleHere(code, message, evidence=payload)


def horizontal_gap_to_bounds(
    point_xy: Sequence[float],
    *,
    bounds_min_xyz: Sequence[float],
    bounds_max_xyz: Sequence[float],
) -> float:
    """Horizontal distance from a point to an AABB footprint, zero inside it.

    Horizontal because this function resolves object identity on the floor
    plane; GraspGenX independently owns the full 3-D hand pose.
    """
    point = tuple(float(value) for value in point_xy)
    lower = tuple(float(value) for value in bounds_min_xyz)
    upper = tuple(float(value) for value in bounds_max_xyz)
    if len(point) != 2:
        raise ValueError("a grounded target coordinate is an [x, y] pair")
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("object bounds need three coordinates")
    if not all(math.isfinite(value) for value in (*point, *lower, *upper)):
        raise ValueError("grasp target matching needs finite geometry")
    if any(lower[axis] > upper[axis] for axis in range(2)):
        raise ValueError("object bounds must be ordered")
    return math.hypot(
        max(lower[0] - point[0], 0.0, point[0] - upper[0]),
        max(lower[1] - point[1], 0.0, point[1] - upper[1]),
    )


def nearest_grounded_pickable(
    *,
    grounded_world_xy: Sequence[float],
    candidate_bounds: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
) -> Tuple[str, float]:
    """Return the nearest authorized pickable without applying the match radius.

    Used to measure SAM3-vs-GT discrepancy even when the observed centre is
    too far to authorize a grasp. Identity matching for execution still goes
    through :func:`match_grounded_pickable`.
    """
    point = tuple(float(value) for value in grounded_world_xy)
    if len(point) != 2 or not all(math.isfinite(value) for value in point):
        raise ValueError("a grounded target coordinate is a finite [x, y] pair")
    gaps = {
        str(object_id): horizontal_gap_to_bounds(
            point,
            bounds_min_xyz=lower,
            bounds_max_xyz=upper,
        )
        for object_id, (lower, upper) in candidate_bounds.items()
    }
    if not gaps:
        raise GraspNotPossibleHere(
            GraspFailureCode.TARGET_NOT_RESOLVED,
            "Nothing you can pick up is available: the object you observed is "
            "not one of the graspable objects in this scene.",
        )
    object_id = min(gaps, key=lambda key: (gaps[key], key))
    return object_id, gaps[object_id]


def match_grounded_pickable(
    *,
    grounded_world_xy: Sequence[float],
    candidate_bounds: Mapping[str, Tuple[Sequence[float], Sequence[float]]],
    maximum_match_gap_m: float = GRASP_GROUNDING_MATCH_RADIUS_M,
) -> Tuple[str, float]:
    """Match a current observed target centre to one authorized pickable.

    ``candidate_bounds`` maps object id to its live ``(bounds_min, bounds_max)``
    and is expected to hold only the objects this grasp is allowed to take —
    the episode's authorized pickables, minus anything already in a hand. The
    caller owns that filtering, so this function never has to decide what
    "allowed" means.

    Returns the matched object id and the measured horizontal gap.

    Raises:
        GraspNotPossibleHere: when nothing in ``candidate_bounds`` lies within
            ``maximum_match_gap_m`` of the grounded coordinate. The message
            carries the nearest measured gap, which is what tells the agent
            whether it is looking at the wrong thing or standing too far from
            the right one.
    """
    limit = float(maximum_match_gap_m)
    if not math.isfinite(limit) or limit <= 0.0:
        raise ValueError("the grasp target match radius must be positive")
    object_id, gap = nearest_grounded_pickable(
        grounded_world_xy=grounded_world_xy,
        candidate_bounds=candidate_bounds,
    )
    if gap > limit:
        raise GraspNotPossibleHere(
            GraspFailureCode.TARGET_NOT_RESOLVED,
            "Nothing you can pick up is at the current grounded coordinate: the "
            f"nearest graspable object is {gap:.2f}m away from it and the "
            f"grasp aims within {limit:.2f}m. Either the object you located is "
            "not graspable, or observe it again from closer up.",
            evidence={
                "nearest_object_gap_m": float(gap),
                "maximum_match_gap_m": float(limit),
            },
        )
    return object_id, gap


def aabb_center_world_m(
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
) -> Tuple[float, float, float]:
    """Return the geometric centre of one finite ordered AABB."""

    lower = tuple(float(value) for value in bbox_min)
    upper = tuple(float(value) for value in bbox_max)
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("object bounds need three coordinates")
    if not all(math.isfinite(value) for value in (*lower, *upper)):
        raise ValueError("object bounds must be finite")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        raise ValueError("object bounds must be ordered")
    return (
        0.5 * (lower[0] + upper[0]),
        0.5 * (lower[1] + upper[1]),
        0.5 * (lower[2] + upper[2]),
    )


def point_inside_aabb(
    point_world_m: Sequence[float],
    *,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
) -> bool:
    """Return whether a world point lies inside or on a finite ordered AABB."""

    point = tuple(float(value) for value in point_world_m)
    lower = tuple(float(value) for value in bbox_min)
    upper = tuple(float(value) for value in bbox_max)
    if len(point) != 3:
        raise ValueError("a world point needs three coordinates")
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("object bounds need three coordinates")
    if not all(math.isfinite(value) for value in (*point, *lower, *upper)):
        raise ValueError("AABB membership needs finite geometry")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        raise ValueError("object bounds must be ordered")
    return all(lower[axis] <= point[axis] <= upper[axis] for axis in range(3))


def signed_aabb_surface_gap_m(
    point_world_m: Sequence[float],
    *,
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
) -> float:
    """Signed distance from a point to an AABB surface.

    Negative inside (distance to the nearest face), zero on the surface, and
    positive outside (Euclidean gap to the box). The unsigned outside case
    matches the closing-centre gate already used by the live target AABB.
    """
    point = tuple(float(value) for value in point_world_m)
    lower = tuple(float(value) for value in bbox_min)
    upper = tuple(float(value) for value in bbox_max)
    if len(point) != 3:
        raise ValueError("a world point needs three coordinates")
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("object bounds need three coordinates")
    if not all(math.isfinite(value) for value in (*point, *lower, *upper)):
        raise ValueError("AABB surface gap needs finite geometry")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        raise ValueError("object bounds must be ordered")
    if all(lower[axis] <= point[axis] <= upper[axis] for axis in range(3)):
        return -min(
            min(point[axis] - lower[axis], upper[axis] - point[axis])
            for axis in range(3)
        )
    squared = 0.0
    for axis in range(3):
        delta = max(
            lower[axis] - point[axis],
            0.0,
            point[axis] - upper[axis],
        )
        squared += delta * delta
    return math.sqrt(squared)


def sam3_gt_aabb_discrepancy(
    *,
    sam3_center_world_m: Sequence[float],
    bbox_min: Sequence[float],
    bbox_max: Sequence[float],
) -> Dict[str, Any]:
    """Measure how far one SAM3 world centre sits from the live GT AABB.

    These numbers are a record of the model's grounding. They must not decide
    whether the first grasp attempt may run.
    """
    sam3 = tuple(float(value) for value in sam3_center_world_m)
    if len(sam3) != 3 or not all(math.isfinite(value) for value in sam3):
        raise ValueError("sam3_center_world_m must contain three finite coordinates")
    gt_center = aabb_center_world_m(bbox_min, bbox_max)
    lower = tuple(float(value) for value in bbox_min)
    upper = tuple(float(value) for value in bbox_max)
    distance = math.sqrt(sum((sam3[axis] - gt_center[axis]) ** 2 for axis in range(3)))
    return {
        "sam3_target_center_world_m": [sam3[0], sam3[1], sam3[2]],
        "gt_aabb_center_world_m": [gt_center[0], gt_center[1], gt_center[2]],
        "gt_aabb_min_world_m": [lower[0], lower[1], lower[2]],
        "gt_aabb_max_world_m": [upper[0], upper[1], upper[2]],
        "center_distance_m": float(distance),
        "sam3_center_inside_gt_aabb": point_inside_aabb(
            sam3,
            bbox_min=lower,
            bbox_max=upper,
        ),
        "signed_surface_gap_m": signed_aabb_surface_gap_m(
            sam3,
            bbox_min=lower,
            bbox_max=upper,
        ),
    }


def grasp_target_observation(
    *,
    target_center_camera_m: Sequence[float],
    world_T_camera: Sequence[Sequence[float]],
) -> Dict[str, float]:
    """Summarize a depth-grounded target relative to the ego camera.

    The target point comes only from the current RGB mask and synchronized
    depth. Camera calibration rotates that point into the gravity-aligned
    world basis long enough to measure horizontal range and height from the
    camera; no world coordinate is returned. Bearing stays relative to the
    ego camera heading, positive to the left. These are observations, not an
    IK distance, end-effector residual, or prescribed recovery action.
    """

    values = tuple(float(value) for value in target_center_camera_m)
    if len(values) != 3 or not all(math.isfinite(value) for value in values):
        raise ValueError("target_center_camera_m must contain three finite values")
    right_m, down_m, forward_depth_m = values
    if forward_depth_m <= 0.0:
        raise ValueError("target forward depth must be positive")

    transform = tuple(tuple(float(value) for value in row) for row in world_T_camera)
    if len(transform) != 4 or any(len(row) != 4 for row in transform):
        raise ValueError("world_T_camera must be a 4x4 transform")
    if not all(math.isfinite(value) for row in transform for value in row):
        raise ValueError("world_T_camera must contain finite values")

    # Translation cancels because both endpoints are the target and camera
    # origin. Only the calibrated camera rotation is required.
    target_delta_world = tuple(
        sum(transform[row][column] * values[column] for column in range(3))
        for row in range(3)
    )
    horizontal_distance_m = math.hypot(
        target_delta_world[0],
        target_delta_world[1],
    )
    camera_forward_xy = (transform[0][2], transform[1][2])
    forward_norm = math.hypot(*camera_forward_xy)
    if forward_norm <= 1e-9:
        raise ValueError("ego camera forward axis has no horizontal projection")
    forward_xy = tuple(value / forward_norm for value in camera_forward_xy)
    left_xy = (-forward_xy[1], forward_xy[0])
    forward_component = sum(
        target_delta_world[index] * forward_xy[index] for index in range(2)
    )
    left_component = sum(
        target_delta_world[index] * left_xy[index] for index in range(2)
    )
    return {
        "horizontal_distance_m": horizontal_distance_m,
        "height_from_camera_m": target_delta_world[2],
        "bearing_deg": math.degrees(math.atan2(left_component, forward_component)),
    }


def frame_from_world_pose(
    *,
    frame_position_world_m: Sequence[float],
    frame_orientation_world_wxyz: Sequence[float],
) -> Tuple[Tuple[float, float, float, float], ...]:
    """Build a world-to-frame transform from one live rigid-body pose.

    Isaac rigid bodies report a world position and a ``wxyz`` orientation.
    Grasp planning needs the inverse transform. Keeping the conversion here
    makes the live-pose authority testable without importing Isaac or USD.
    """

    position = tuple(float(value) for value in frame_position_world_m)
    quaternion = tuple(float(value) for value in frame_orientation_world_wxyz)
    if len(position) != 3 or not all(math.isfinite(value) for value in position):
        raise ValueError("the live frame position must be a finite xyz point")
    if len(quaternion) != 4 or not all(math.isfinite(value) for value in quaternion):
        raise ValueError("the live frame orientation must be a finite wxyz quaternion")
    norm = math.sqrt(sum(value * value for value in quaternion))
    if norm <= 1e-8:
        raise ValueError("the live frame orientation quaternion has zero norm")
    w, x, y, z = (value / norm for value in quaternion)
    world_from_frame = (
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
        ),
        (
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
        ),
        (
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
    )
    frame_from_world = tuple(zip(*world_from_frame))
    offset = tuple(
        -sum(frame_from_world[row][axis] * position[axis] for axis in range(3))
        for row in range(3)
    )
    return tuple(tuple(frame_from_world[row]) + (offset[row],) for row in range(3)) + (
        (0.0, 0.0, 0.0, 1.0),
    )


__all__ = [
    "GRASP_GROUNDING_MATCH_RADIUS_M",
    "GRASP_GROUNDING_SAM3",
    "GRASP_GROUNDING_MEMORY",
    "GRASP_REMEMBERED_GROUNDING_MIN_DISK_RADIUS_PX",
    "GRASP_FALLBACK_TARGET_NOT_IN_VIEW",
    "GRASP_FALLBACK_VIEW_UNUSABLE",
    "GRASP_FALLBACK_VIEW_TOO_SPARSE",
    "GRASP_FALLBACK_TARGET_UNAVAILABLE",
    "GRASP_FALLBACK_NO_GRASP_POSE",
    "GRASP_FALLBACK_NEVER_GROUNDED",
    "GRASP_FALLBACK_GROUNDING_GEOMETRY_STALE",
    "GRASP_REMEMBERED_GROUNDING_NOTICE",
    "GRASP_FAILURE_CODES",
    "GraspFailureCode",
    "GraspNotPossibleHere",
    "agent_safe_grasp_refusal",
    "aabb_center_world_m",
    "grasp_target_observation",
    "match_grounded_pickable",
    "nearest_grounded_pickable",
    "horizontal_gap_to_bounds",
    "point_inside_aabb",
    "sam3_gt_aabb_discrepancy",
    "signed_aabb_surface_gap_m",
]
