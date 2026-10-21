"""Authored grasp geometry for asset and reach certification.

``CanonicalGrasp`` describes an object's approach direction, closing axis and
height in a gravity-aligned frame. ``resolve_grasp_pose`` combines it with live
bounds and robot bearing to obtain pregrasp and closing poses. Yaw-dependent
grasps require an explicit object yaw.

Palm +X is the approach axis and palm Y is the closing axis. The arm solver
constructs the palm rotation; these helpers return axes and positions.
``facing_surface`` also defines the bounded target shell used by grasp-reach
certification. Evaluated grasp proposals come from GraspGenX."""

from __future__ import annotations

import math
from typing import Literal, Tuple

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]

GraspApproach = Literal["frontal_horizontal", "top_down"]
ClosingCenter = Literal["body_center", "facing_rim"]


class StrictModel(BaseModel):
    """Immutable current-release grasp-pose schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class FacingSurface(StrictModel):
    """The robot-facing frame of one live target AABB.

    Everything a grasp needs about where the robot stands relative to the
    object reduces to this frame: the point where the horizontal bearing
    leaves the AABB, the outward direction from object to robot, and the
    tangent the jaws close across. Deriving it from the live bearing rather
    than from a world axis is what lets one authored grasp serve an object
    approached from any side.
    """

    surface_xy: Point2 = Field(
        description=(
            "Point where the object-to-robot bearing crosses the horizontal "
            "AABB boundary, in world metres."
        ),
    )
    center_xy: Point2 = Field(
        description="Horizontal centre of the live target AABB, in world metres.",
    )
    outward_xy: Point2 = Field(
        description=(
            "Unit horizontal direction from the object centre toward the "
            "robot. The approach direction is its negation."
        ),
    )
    tangent_xy: Point2 = Field(
        description=(
            "Unit horizontal direction tangent to the faced surface, right "
            "handed with world +Z. The jaws close across this direction at "
            "zero wrist roll."
        ),
    )
    surface_distance_m: float = Field(
        gt=0.0,
        description=(
            "Distance from the horizontal AABB centre to the faced surface "
            "along the bearing, in metres."
        ),
    )


class CanonicalGrasp(StrictModel):
    """How one object class is taken, in its own gravity-aligned frame.

    Authored once per class and reviewed, the way the Dex3 finger presets are.
    Nothing here is per-episode: the numbers describe a shape, and
    :func:`resolve_grasp_pose` supplies the live pose and size.
    """

    approach: GraspApproach = Field(
        description=(
            "Direction the hand travels to reach the closing centre. "
            "`frontal_horizontal` approaches along the horizontal bearing "
            "from the robot; `top_down` descends along world -Z."
        ),
    )
    closing_center: ClosingCenter = Field(
        description=(
            "Where the jaws close. `body_center` puts the closing centre on "
            "the vertical axis of the live AABB, which is scale invariant and "
            "right for a body the jaws can span. `facing_rim` puts it "
            "`rim_inset_m` inward from the robot-facing surface, which is how "
            "a flat or hollow object is taken by its edge."
        ),
    )
    rim_inset_m: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Distance inward from the robot-facing surface to the closing "
            "centre, for `facing_rim` only. Must be zero for `body_center`."
        ),
    )
    height_fraction: float = Field(
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of the live object height, measured from its lower AABB "
            "face, at which the jaws close. The 0.60 used for upright bodies "
            "encloses them below the top without driving the fingertips into "
            "the support surface."
        ),
    )
    standoff_m: float = Field(
        gt=0.0,
        description=(
            "Pre-grasp retreat from the closing centre along the reverse of "
            "the approach direction, in metres. The arm reaches this pose "
            "first so the final motion into the object is a straight line."
        ),
    )
    jaw_span_m: float = Field(
        gt=0.0,
        description=(
            "Opening the jaws must reach to take this class, in metres. "
            "Checked against the selected gripper profile before a reach is "
            "commanded, and against the live AABB when the closing centre is "
            "the body centre."
        ),
    )
    wrist_roll_rad: float = Field(
        default=0.0,
        description=(
            "Roll of the closing axis about the approach axis. Zero closes "
            "across the horizontal tangent of the faced surface, which is the "
            "upright palm the arm solver produces today."
        ),
    )
    requires_object_yaw: bool = Field(
        default=False,
        description=(
            "True when the grasp is only correct at a particular object yaw "
            "(a handle, a tool grip). There is no yaw estimator behind this "
            "module, so resolving such a class raises instead of guessing."
        ),
    )

    @model_validator(mode="after")
    def validate_grasp(self) -> "CanonicalGrasp":
        values = (
            self.rim_inset_m,
            self.height_fraction,
            self.standoff_m,
            self.jaw_span_m,
            self.wrist_roll_rad,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("canonical grasp values must be finite")
        if self.closing_center == "body_center" and self.rim_inset_m != 0.0:
            raise ValueError(
                "a body_center grasp closes on the AABB vertical axis, so "
                "rim_inset_m must be zero"
            )
        if self.closing_center == "facing_rim" and self.rim_inset_m <= 0.0:
            raise ValueError(
                "a facing_rim grasp needs a positive rim_inset_m to place the "
                "closing centre inside the edge"
            )
        return self


class ResolvedGrasp(StrictModel):
    """One canonical grasp composed with one live target AABB and bearing."""

    approach: GraspApproach = Field(
        description="Approach the resolved poses were built for.",
    )
    grasp_position_world: Point3 = Field(
        description=(
            "World point that must end up between the closing jaws, in metres."
        ),
    )
    standoff_position_world: Point3 = Field(
        description=(
            "World pre-grasp position, one `standoff_m` back along the "
            "approach axis, in metres."
        ),
    )
    approach_axis_world: Point3 = Field(
        description=(
            "Unit world direction the hand travels from standoff to closing "
            "centre. The palm +X aligns with it."
        ),
    )
    closing_axis_world: Point3 = Field(
        description=("Unit world direction the jaws close across, after wrist roll."),
    )
    wrist_roll_rad: float = Field(
        description="Roll about the approach axis carried from the canonical grasp.",
    )
    jaw_span_m: float = Field(
        gt=0.0,
        description="Opening the jaws must reach, carried from the canonical grasp.",
    )
    object_span_across_jaws_m: float = Field(
        ge=0.0,
        description=(
            "Live AABB extent along the closing axis, in metres. Diagnostic "
            "for a rim grasp, where the jaws close on an edge rather than on "
            "the whole extent; a hard bound for a body-centre grasp."
        ),
    )

    @model_validator(mode="after")
    def validate_resolved(self) -> "ResolvedGrasp":
        values = (
            *self.grasp_position_world,
            *self.standoff_position_world,
            *self.approach_axis_world,
            *self.closing_axis_world,
            self.wrist_roll_rad,
            self.jaw_span_m,
            self.object_span_across_jaws_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("resolved grasp geometry must be finite")
        for label, axis in (
            ("approach", self.approach_axis_world),
            ("closing", self.closing_axis_world),
        ):
            if abs(float(np.linalg.norm(axis)) - 1.0) > 1e-9:
                raise ValueError(f"resolved {label} axis must be a unit vector")
        return self


def facing_surface(
    *,
    target_bounds_min_xyz: Point3,
    target_bounds_max_xyz: Point3,
    robot_position_xy: Point2,
) -> FacingSurface:
    """Derive the robot-facing frame of one live target AABB.

    The bearing is horizontal on purpose: a grasp is planned around which side
    of the object the robot stands on, and the vertical placement is a separate
    authored fraction. A robot standing on the object's own vertical axis has
    no bearing at all and is rejected rather than resolved to an arbitrary side.
    """
    lower = tuple(float(value) for value in target_bounds_min_xyz)
    upper = tuple(float(value) for value in target_bounds_max_xyz)
    robot = tuple(float(value) for value in robot_position_xy)
    if len(lower) != 3 or len(upper) != 3 or len(robot) != 2:
        raise ValueError(
            "grasp target bounds need three coordinates and the Robot position "
            "needs two"
        )
    if not all(math.isfinite(value) for value in (*lower, *upper, *robot)):
        raise ValueError("grasp target bounds and Robot entry must be finite")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        raise ValueError("grasp target bounds must be ordered")

    center_x = 0.5 * (lower[0] + upper[0])
    center_y = 0.5 * (lower[1] + upper[1])
    toward_robot_x = robot[0] - center_x
    toward_robot_y = robot[1] - center_y
    horizontal_distance = math.hypot(toward_robot_x, toward_robot_y)
    if horizontal_distance <= 1e-9:
        raise ValueError(
            "Robot entry cannot coincide with the target horizontal centre"
        )
    outward_x = toward_robot_x / horizontal_distance
    outward_y = toward_robot_y / horizontal_distance
    half_x = 0.5 * (upper[0] - lower[0])
    half_y = 0.5 * (upper[1] - lower[1])
    boundary_distances = [
        half_extent / abs(direction)
        for half_extent, direction in (
            (half_x, outward_x),
            (half_y, outward_y),
        )
        if abs(direction) > 1e-12
    ]
    if not boundary_distances:
        raise ValueError("target-facing AABB surface cannot be resolved")
    surface_distance = min(boundary_distances)
    if surface_distance <= 0.0:
        raise ValueError("target-facing AABB surface has no horizontal extent to grasp")
    return FacingSurface(
        surface_xy=(
            center_x + outward_x * surface_distance,
            center_y + outward_y * surface_distance,
        ),
        center_xy=(center_x, center_y),
        outward_xy=(outward_x, outward_y),
        tangent_xy=(-outward_y, outward_x),
        surface_distance_m=surface_distance,
    )


def body_center_position(
    *,
    target_bounds_min_xyz: Point3,
    target_bounds_max_xyz: Point3,
    height_fraction: float,
) -> Point3:
    """Place a closing centre inside one bounded object body.

    Horizontally the AABB centre, vertically a fraction of the object's own
    height. Being scale free is the point: the same fraction survives episode
    scaling and a different model of the same class.
    """
    lower = tuple(float(value) for value in target_bounds_min_xyz)
    upper = tuple(float(value) for value in target_bounds_max_xyz)
    fraction = float(height_fraction)
    if len(lower) != 3 or len(upper) != 3:
        raise ValueError("grasp-centre bounds need three coordinates")
    if not all(math.isfinite(value) for value in (*lower, *upper, fraction)):
        raise ValueError("grasp-centre geometry must be finite")
    if any(lower[axis] > upper[axis] for axis in range(3)):
        raise ValueError("grasp-centre bounds must be ordered")
    if not 0.0 <= fraction <= 1.0:
        raise ValueError("grasp-centre height fraction must lie in [0, 1]")
    return (
        0.5 * (lower[0] + upper[0]),
        0.5 * (lower[1] + upper[1]),
        lower[2] + fraction * (upper[2] - lower[2]),
    )


def resolve_grasp_pose(
    *,
    canonical: CanonicalGrasp,
    target_bounds_min_xyz: Point3,
    target_bounds_max_xyz: Point3,
    robot_position_xy: Point2,
) -> ResolvedGrasp:
    """Compose one canonical grasp with one live target AABB and bearing.

    Raises:
        ValueError: when the class needs an object yaw nobody can supply, when
            the bearing is degenerate, or when a body-centre grasp is asked of
            an object wider across the jaws than the authored opening. Each of
            these is a contract the caller must surface as a failure; none of
            them has a defensible substitute pose.
    """
    surface = facing_surface(
        target_bounds_min_xyz=target_bounds_min_xyz,
        target_bounds_max_xyz=target_bounds_max_xyz,
        robot_position_xy=robot_position_xy,
    )
    if canonical.requires_object_yaw:
        raise ValueError(
            "this object class is only grasped at a known object yaw, and no "
            "yaw source exists; authoring must supply a yaw-free grasp or the "
            "class stays ungraspable"
        )

    lower = tuple(float(value) for value in target_bounds_min_xyz)
    upper = tuple(float(value) for value in target_bounds_max_xyz)
    outward = np.asarray(
        [surface.outward_xy[0], surface.outward_xy[1], 0.0],
        dtype=np.float64,
    )
    tangent = np.asarray(
        [surface.tangent_xy[0], surface.tangent_xy[1], 0.0],
        dtype=np.float64,
    )
    if canonical.approach == "frontal_horizontal":
        approach_axis = -outward
    else:
        approach_axis = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)

    body_center = body_center_position(
        target_bounds_min_xyz=target_bounds_min_xyz,
        target_bounds_max_xyz=target_bounds_max_xyz,
        height_fraction=canonical.height_fraction,
    )
    if canonical.closing_center == "body_center":
        grasp_xy = np.asarray(body_center[:2], dtype=np.float64)
    else:
        grasp_xy = (
            np.asarray(surface.surface_xy, dtype=np.float64)
            - outward[:2] * canonical.rim_inset_m
        )
    grasp_position = np.asarray(
        [grasp_xy[0], grasp_xy[1], body_center[2]],
        dtype=np.float64,
    )
    standoff_position = grasp_position - approach_axis * canonical.standoff_m
    closing_axis = _rotated_about(
        tangent,
        axis=approach_axis,
        angle_rad=canonical.wrist_roll_rad,
    )
    span = sum(
        abs(float(closing_axis[axis])) * (upper[axis] - lower[axis])
        for axis in range(3)
    )
    if canonical.closing_center == "body_center" and span > canonical.jaw_span_m:
        raise ValueError(
            f"the live object spans {span:.3f} m across the jaws but this "
            f"class authorises an opening of {canonical.jaw_span_m:.3f} m; a "
            "body-centre grasp of it would not close"
        )
    return ResolvedGrasp(
        approach=canonical.approach,
        grasp_position_world=(
            float(grasp_position[0]),
            float(grasp_position[1]),
            float(grasp_position[2]),
        ),
        standoff_position_world=(
            float(standoff_position[0]),
            float(standoff_position[1]),
            float(standoff_position[2]),
        ),
        approach_axis_world=(
            float(approach_axis[0]),
            float(approach_axis[1]),
            float(approach_axis[2]),
        ),
        closing_axis_world=(
            float(closing_axis[0]),
            float(closing_axis[1]),
            float(closing_axis[2]),
        ),
        wrist_roll_rad=float(canonical.wrist_roll_rad),
        jaw_span_m=float(canonical.jaw_span_m),
        object_span_across_jaws_m=float(span),
    )


def _rotated_about(
    vector: np.ndarray,
    *,
    axis: np.ndarray,
    angle_rad: float,
) -> np.ndarray:
    """Rotate ``vector`` about a unit ``axis`` by Rodrigues' formula."""
    if angle_rad == 0.0:
        return np.asarray(vector, dtype=np.float64)
    cos = math.cos(angle_rad)
    sin = math.sin(angle_rad)
    return (
        vector * cos
        + np.cross(axis, vector) * sin
        + axis * float(np.dot(axis, vector)) * (1.0 - cos)
    )


__all__ = [
    "CanonicalGrasp",
    "ClosingCenter",
    "FacingSurface",
    "GraspApproach",
    "ResolvedGrasp",
    "body_center_position",
    "facing_surface",
    "resolve_grasp_pose",
]
