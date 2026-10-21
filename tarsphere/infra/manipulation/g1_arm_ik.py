"""Inverse kinematics for the G1 robot's two arms.

The ``G1MoveBySpeedController`` accepts a 15-D upper-body target in
the order ``waist + left arm (7) + right arm (7)``.  It does not provide an IK
layer.  This module supplies that missing geometric layer for TARSphere while keeping
the locomotion-controlled body unchanged. Arm IK ends at the Dex3 palm;
finger closure is controlled independently by the two seven-joint controllers.

The chain constants and limits below are the official Unitree G1 bilateral arm
kinematics used by TARSphere's ``g1_29dof_color.usd``. The final fixed
offset is the Dex3 palm mount shared by the donor asset. Placement controls the
palm origin. Grasp controls the physical closing centre forward of that origin,
so the target lies between the fingers rather than visibly inside the palm.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable, Literal, Sequence

import numpy as np


G1_UPPER_COMMAND_SCALE = 0.25
G1ArmSide = Literal["left", "right"]
G1_LEFT_ARM_JOINT_NAMES = (
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
)
G1_RIGHT_ARM_JOINT_NAMES = (
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)

# Official Unitree G1 joint limits, in the order above.
G1_RIGHT_ARM_LOWER = np.asarray(
    [-3.0892, -2.2515, -2.6180, -1.0472, -1.972222054, -1.614429558, -1.614429558],
    dtype=np.float64,
)
G1_RIGHT_ARM_UPPER = np.asarray(
    [2.6704, 1.5882, 2.6180, 2.0944, 1.972222054, 1.614429558, 1.614429558],
    dtype=np.float64,
)
G1_LEFT_ARM_LOWER = np.asarray(
    [-3.0892, -1.5882, -2.6180, -1.0472, -1.972222054, -1.614429558, -1.614429558],
    dtype=np.float64,
)
G1_LEFT_ARM_UPPER = np.asarray(
    [2.6704, 2.2515, 2.6180, 2.0944, 1.972222054, 1.614429558, 1.614429558],
    dtype=np.float64,
)


@dataclass(frozen=True)
class G1ArmIKResult:
    """One bounded bilateral-arm solve.

    ``hand_position_torso`` is always the Dex3 palm origin.
    ``reference_position_torso`` is the palm-frame point actually aligned to
    ``target_position_torso``; it equals the palm for ordinary reaches and the
    Dex3 closing centre for contact grasps.
    """

    hand: G1ArmSide
    joint_positions: tuple[float, ...]
    hand_position_torso: tuple[float, float, float]
    target_position_torso: tuple[float, float, float]
    position_error_m: float
    converged: bool
    iterations: int
    reference_position_torso: tuple[float, float, float] | None = None
    reference_from_palm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    hand_rotation_torso: tuple[tuple[float, float, float], ...] | None = None
    target_rotation_torso: tuple[tuple[float, float, float], ...] | None = None
    orientation_error_rad: float = 0.0

    def upper_body_command(self, *, balance_contralateral: bool = False) -> list[float]:
        """Return TARSphere's 15-D policy-space upper-body command.

        A carried object breaks the nominal bilateral upper-body symmetry seen
        by the locomotion policy. ``balance_contralateral`` commands the free
        arm to the exact kinematic mirror of the solved arm while leaving the
        attached hand unchanged.
        """
        command = [0.0] * 15
        start = 1 if self.hand == "left" else 8
        command[start : start + 7] = [
            float(value) / G1_UPPER_COMMAND_SCALE for value in self.joint_positions
        ]
        if balance_contralateral:
            mirrored = _mirror_right_joint_vector(self.joint_positions)
            other_start = 8 if self.hand == "left" else 1
            command[other_start : other_start + 7] = [
                float(value) / G1_UPPER_COMMAND_SCALE for value in mirrored
            ]
        return command


@dataclass(frozen=True)
class _Joint:
    xyz: tuple[float, float, float]
    rpy: tuple[float, float, float]
    axis: tuple[float, float, float] | None


_RIGHT_ARM_CHAIN = (
    _Joint(
        (0.0039563, -0.10021, 0.24778),
        (-0.27931, 5.4949e-05, 0.00019159),
        (0.0, 1.0, 0.0),
    ),
    _Joint((0.0, -0.038, -0.013831), (0.27925, 0.0, 0.0), (1.0, 0.0, 0.0)),
    _Joint((0.0, -0.00624, -0.1032), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _Joint((0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _Joint((0.100, -0.00188791, -0.010), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    _Joint((0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _Joint((0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    # Fixed wrist-to-Dex3-palm mount.
    _Joint((0.0415, -0.003, 0.0), (0.0, 0.0, 0.0), None),
)

_LEFT_ARM_CHAIN = (
    _Joint(
        (0.0039563, 0.10022, 0.24778),
        (0.27931, 5.4949e-05, -0.00019159),
        (0.0, 1.0, 0.0),
    ),
    _Joint((0.0, 0.038, -0.013831), (-0.27925, 0.0, 0.0), (1.0, 0.0, 0.0)),
    _Joint((0.0, 0.00624, -0.1032), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _Joint((0.015783, 0.0, -0.080518), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _Joint((0.100, 0.00188791, -0.010), (0.0, 0.0, 0.0), (1.0, 0.0, 0.0)),
    _Joint((0.038, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
    _Joint((0.046, 0.0, 0.0), (0.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    _Joint((0.0415, 0.003, 0.0), (0.0, 0.0, 0.0), None),
)

_ARM_CHAINS = {"left": _LEFT_ARM_CHAIN, "right": _RIGHT_ARM_CHAIN}
_ARM_LOWER = {"left": G1_LEFT_ARM_LOWER, "right": G1_RIGHT_ARM_LOWER}
_ARM_UPPER = {"left": G1_LEFT_ARM_UPPER, "right": G1_RIGHT_ARM_UPPER}


def right_hand_position_in_torso(
    joint_positions: Sequence[float],
) -> np.ndarray:
    """Compute the G1 right Dex3 palm position in ``torso_link`` frame."""
    return hand_position_in_torso(joint_positions, hand="right")


def left_hand_position_in_torso(
    joint_positions: Sequence[float],
) -> np.ndarray:
    """Compute the G1 left Dex3 palm position in ``torso_link`` frame."""
    return hand_position_in_torso(joint_positions, hand="left")


def hand_position_in_torso(
    joint_positions: Sequence[float], *, hand: G1ArmSide
) -> np.ndarray:
    """Compute either G1 Dex3 palm position in ``torso_link`` frame."""
    return hand_transform_in_torso(joint_positions, hand=hand)[:3, 3].copy()


def shifted_hand_pose_ik_target_torso(
    *,
    measured_palm_torso: Sequence[float],
    model_palm_torso: Sequence[float],
    planned_palm_torso: Sequence[float],
) -> np.ndarray:
    """Shift an IK palm target by the live model-versus-measurement offset.

    The analytic FK shares the bias that put the real palm beside the planned
    goal.  Re-solving for the planned goal would land in the same wrong place.
    Subtracting the offset measured at the current joints makes the biased model
    predict the planned goal at the corrected command.  Any constant tool-point
    convention difference between FK and measurement is absorbed into the offset
    because both sides of the subtraction use the same live measurement frame.
    """
    measured = np.asarray(measured_palm_torso, dtype=np.float64).reshape(-1)
    model = np.asarray(model_palm_torso, dtype=np.float64).reshape(-1)
    planned = np.asarray(planned_palm_torso, dtype=np.float64).reshape(-1)
    for label, vector in (
        ("measured palm", measured),
        ("model palm", model),
        ("planned palm", planned),
    ):
        if vector.size != 3 or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"G1 reach correction {label} must contain three finite values."
            )
    model_offset_torso = measured - model
    return planned - model_offset_torso


def clamp_palm_correction_step_torso(
    *,
    model_palm_torso: Sequence[float],
    shifted_target_torso: Sequence[float],
    max_step_m: float,
) -> np.ndarray:
    """Bound how far the IK solve asks the model palm to travel in one step."""
    model = np.asarray(model_palm_torso, dtype=np.float64).reshape(-1)
    shifted = np.asarray(shifted_target_torso, dtype=np.float64).reshape(-1)
    if (
        model.size != 3
        or shifted.size != 3
        or not np.all(np.isfinite(model))
        or not np.all(np.isfinite(shifted))
    ):
        raise ValueError(
            "G1 reach correction palm vectors must contain three finite values."
        )
    if not math.isfinite(max_step_m) or max_step_m <= 0.0:
        raise ValueError(
            "G1 reach correction max step must be a positive finite length."
        )
    step = shifted - model
    step_length = float(np.linalg.norm(step))
    if step_length <= max_step_m or step_length == 0.0:
        return shifted.copy()
    return model + step * (max_step_m / step_length)


def actuator_command_from_measured_ik_delta(
    *,
    previous_command: Sequence[float],
    measured_joints: Sequence[float],
    desired_measured_joints: Sequence[float],
    hand: G1ArmSide,
) -> tuple[float, ...]:
    """Apply a live-state IK correction as a delta to the actuator command.

    The upper-body controller has finite steady-state tracking error.  An IK
    solve warm-started from the measured joints therefore describes the next
    *measured* joint state, not an absolute command for that controller.  Keep
    the controller's existing bias compensation by adding the requested live
    joint delta to its previous command.
    """
    side = _arm_side(hand)
    command = np.asarray(previous_command, dtype=np.float64).reshape(-1)
    measured = np.asarray(measured_joints, dtype=np.float64).reshape(-1)
    desired = np.asarray(
        desired_measured_joints,
        dtype=np.float64,
    ).reshape(-1)
    for label, vector in (
        ("previous command", command),
        ("measured joints", measured),
        ("desired measured joints", desired),
    ):
        if vector.size != 7 or not np.all(np.isfinite(vector)):
            raise ValueError(
                f"G1 reach correction {label} must contain seven finite joint values."
            )
    corrected = command + desired - measured
    lower = _ARM_LOWER[side]
    upper = _ARM_UPPER[side]
    # The desired measured state already came from bounded IK.  Controller
    # bias compensation can nevertheless put the *actuator command* a few
    # milliradians beyond an official limit.  Project that command back into
    # the same embodiment limits instead of turning a feasible measured goal
    # into a terminal exception.  Subsequent live tracking, Cartesian, contact,
    # and retention gates still decide whether the saturated correction worked.
    bounded = np.clip(corrected, lower, upper)
    return tuple(float(value) for value in bounded)


def hand_reference_position_in_torso(
    joint_positions: Sequence[float],
    *,
    hand: G1ArmSide,
    reference_from_palm: Sequence[float],
) -> np.ndarray:
    """Transform one palm-frame manipulation reference into the torso frame."""
    offset = np.asarray(reference_from_palm, dtype=np.float64).reshape(-1)
    if offset.size != 3 or not np.all(np.isfinite(offset)):
        raise ValueError("G1 hand reference offset must contain three finite values.")
    transform = hand_transform_in_torso(joint_positions, hand=hand)
    return transform[:3, 3] + transform[:3, :3] @ offset


def hand_transform_in_torso(
    joint_positions: Sequence[float], *, hand: G1ArmSide
) -> np.ndarray:
    """Compute the full Dex3 palm transform in ``torso_link`` frame."""
    side = _arm_side(hand)
    q = _joint_vector(joint_positions, hand=side)
    transform = np.eye(4, dtype=np.float64)
    for index, joint in enumerate(_ARM_CHAINS[side]):
        transform = transform @ _origin_transform(joint.xyz, joint.rpy)
        if joint.axis is not None:
            transform = transform @ _axis_transform(joint.axis, float(q[index]))
    return transform


def wrist_yaw_transform_in_torso(
    joint_positions: Sequence[float], *, hand: G1ArmSide
) -> np.ndarray:
    """Compute ``{left,right}_wrist_yaw_link`` in ``torso_link``.

    Unlike :func:`hand_transform_in_torso`, this stops before the final fixed
    41.5 mm Dex3 palm mount. Policies that name a wrist EEF must not silently
    receive the palm/tool origin: that offset rotates with the wrist and makes
    redundant wrist-pose and arm-joint actions physically inconsistent.
    """

    side = _arm_side(hand)
    q = _joint_vector(joint_positions, hand=side)
    transform = np.eye(4, dtype=np.float64)
    for index, joint in enumerate(_ARM_CHAINS[side][:-1]):
        transform = transform @ _origin_transform(joint.xyz, joint.rpy)
        if joint.axis is not None:
            transform = transform @ _axis_transform(joint.axis, float(q[index]))
    return transform


def frontal_grasp_palm_rotation_in_torso(
    target_position_torso: Sequence[float],
) -> np.ndarray:
    """Return the upright, target-facing Dex3 palm rotation for a grasp.

    Dex3 fingers extend along palm +X and close across palm Y. Aligning +X
    with the horizontal torso-to-target bearing keeps the fingers parallel to
    the support surface; aligning palm +Z with torso +Z prevents the point-only
    IK solution from sweeping the fingertips downward through that surface.

    This is the frame every promoted grasp trajectory was certified with, and
    the arithmetic below is deliberately left in place rather than expressed
    through :func:`grasp_palm_rotation_in_torso`: the general builder
    orthonormalises, which would perturb this frame in the last bits and move
    an already-certified iterative solve for no gain. Use it when all that is
    known is where the hand should end up — a caller that knows how the object
    is approached should say so and use the general builder.
    """
    target = np.asarray(target_position_torso, dtype=np.float64).reshape(-1)
    if target.size != 3 or not np.all(np.isfinite(target)):
        raise ValueError(
            "G1 grasp target must contain three finite torso-frame values."
        )
    forward = np.asarray([target[0], target[1], 0.0], dtype=np.float64)
    forward_norm = float(np.linalg.norm(forward))
    if forward_norm < 1e-6:
        raise ValueError("G1 grasp target has no horizontal torso-frame bearing.")
    palm_x = forward / forward_norm
    palm_z = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    palm_y = np.cross(palm_z, palm_x)
    return np.column_stack((palm_x, palm_y, palm_z))


def grasp_palm_rotation_in_torso(
    *,
    approach_axis_torso: Sequence[float],
    closing_axis_torso: Sequence[float],
) -> np.ndarray:
    """Build the Dex3 palm rotation from how an object is approached.

    Dex3 fingers extend along palm +X and close across palm Y, so the two axes
    a grasp is planned around are exactly the two the palm frame needs: the
    direction the hand travels into the object, and the direction the jaws close
    across. Taking both from the caller is what lets one builder serve a side
    approach and a top-down one — a top-down grasp has no horizontal bearing to
    derive a frame from, and its roll is a property of the object's shape rather
    than of where the robot stands.

    The closing axis is orthonormalised against the approach axis rather than
    trusted, because both arrive through a world-to-torso transform that leaves
    them very slightly non-perpendicular; a rotation matrix that is not
    orthonormal makes the IK residual meaningless. Axes that disagree by more
    than a degree are a caller defect and raise.
    """
    palm_x = _unit_axis(approach_axis_torso, name="approach")
    closing = _unit_axis(closing_axis_torso, name="closing")
    alignment = float(np.dot(palm_x, closing))
    if abs(alignment) > 0.0175:
        raise ValueError(
            "A Dex3 grasp closes across its approach, but the requested "
            f"closing axis is {math.degrees(math.asin(min(1.0, abs(alignment)))):.1f}"
            " deg out of perpendicular."
        )
    palm_y = closing - palm_x * alignment
    palm_y_norm = float(np.linalg.norm(palm_y))
    if palm_y_norm < 1e-6:
        raise ValueError(
            "A Dex3 grasp closing axis cannot be parallel to its approach."
        )
    palm_y = palm_y / palm_y_norm
    palm_z = np.cross(palm_x, palm_y)
    return np.column_stack((palm_x, palm_y, palm_z))


def _unit_axis(axis: Sequence[float], *, name: str) -> np.ndarray:
    """Return ``axis`` as a unit vector, or raise if it cannot be one."""
    values = np.asarray(axis, dtype=np.float64).reshape(-1)
    if values.size != 3 or not np.all(np.isfinite(values)):
        raise ValueError(
            f"A G1 grasp {name} axis must contain three finite torso-frame values."
        )
    norm = float(np.linalg.norm(values))
    if norm < 1e-6:
        raise ValueError(f"A G1 grasp {name} axis must have a direction.")
    return values / norm


def solve_right_hand_position(
    target_position_torso: Sequence[float],
    *,
    warm_start: Sequence[float] | None = None,
    position_tolerance_m: float = 0.015,
    solver_tolerance_m: float | None = None,
    max_iterations: int = 180,
) -> G1ArmIKResult:
    """Solve position-only right-arm IK with bounded damped least squares.

    Multiple deterministic seeds avoid treating one open-loop pose as the
    kinematic truth.  The returned result always contains the best candidate;
    callers must require ``converged`` before executing a certification grasp.
    """
    return solve_hand_pose(
        target_position_torso,
        hand="right",
        warm_start=warm_start,
        position_tolerance_m=position_tolerance_m,
        solver_tolerance_m=solver_tolerance_m,
        max_iterations=max_iterations,
    )


def solve_left_hand_position(
    target_position_torso: Sequence[float],
    *,
    warm_start: Sequence[float] | None = None,
    position_tolerance_m: float = 0.015,
    solver_tolerance_m: float | None = None,
    max_iterations: int = 180,
) -> G1ArmIKResult:
    return solve_hand_pose(
        target_position_torso,
        hand="left",
        warm_start=warm_start,
        position_tolerance_m=position_tolerance_m,
        solver_tolerance_m=solver_tolerance_m,
        max_iterations=max_iterations,
    )


def solve_hand_pose(
    target_position_torso: Sequence[float],
    *,
    hand: G1ArmSide,
    warm_start: Sequence[float] | None = None,
    position_tolerance_m: float = 0.015,
    solver_tolerance_m: float | None = None,
    reference_from_palm: Sequence[float] = (0.0, 0.0, 0.0),
    target_rotation_torso: Sequence[Sequence[float]] | None = None,
    orientation_tolerance_rad: float = math.radians(12.0),
    max_iterations: int = 180,
) -> G1ArmIKResult:
    """Solve bounded position or full-pose IK for one official G1 arm.

    ``position_tolerance_m`` is the task-level reachability envelope.
    ``solver_tolerance_m`` optionally requests a tighter numerical solution.
    Keeping the two thresholds separate prevents a permissive manipulation
    envelope from prematurely terminating IK at a visibly inaccurate pose.
    ``reference_from_palm`` selects the physical point aligned to the target.
    ``target_rotation_torso`` constrains the Dex3 palm when grasp geometry
    requires it; ordinary reaches remain position-only by passing ``None``.
    ``warm_start`` is a deterministic seed, not a second pose objective.
    """
    side = _arm_side(hand)
    lower = _ARM_LOWER[side]
    upper = _ARM_UPPER[side]
    target = np.asarray(target_position_torso, dtype=np.float64).reshape(-1)
    if target.size != 3 or not np.all(np.isfinite(target)):
        raise ValueError(
            "G1 arm IK target must be three finite torso-frame coordinates."
        )
    tolerance = float(position_tolerance_m)
    if tolerance <= 0:
        raise ValueError("G1 arm IK position tolerance must be positive.")
    solver_tolerance = (
        tolerance if solver_tolerance_m is None else float(solver_tolerance_m)
    )
    if solver_tolerance <= 0 or solver_tolerance > tolerance:
        raise ValueError(
            "G1 arm IK solver tolerance must be positive and no greater "
            "than the task position tolerance."
        )
    reference = np.asarray(reference_from_palm, dtype=np.float64).reshape(-1)
    if reference.size != 3 or not np.all(np.isfinite(reference)):
        raise ValueError("G1 hand reference offset must contain three finite values.")
    target_rotation = _validated_rotation(target_rotation_torso)
    orientation_tolerance = float(orientation_tolerance_rad)
    if orientation_tolerance <= 0.0 or orientation_tolerance > math.pi:
        raise ValueError("G1 arm IK orientation tolerance must lie in (0, pi].")

    reference_q = None if warm_start is None else _joint_vector(warm_start, hand=side)
    seeds: list[np.ndarray] = []
    if reference_q is not None:
        seeds.append(reference_q.copy())
    right_seeds = (
        (0.0, -0.55, 0.0, 1.35, 0.0, 0.0, 0.0),
        (0.3, -0.70, -0.4, 1.20, 0.0, 0.0, 0.0),
        (-0.3, -0.45, 0.4, 1.55, 0.0, 0.0, 0.0),
        (0.0, -1.00, 0.0, 0.50, 0.0, 0.0, 0.0),
        (0.0, -0.25, 0.0, 1.70, 0.0, 0.0, 0.0),
    )
    seeds.extend(
        np.asarray(
            seed if side == "right" else _mirror_right_joint_vector(seed),
            dtype=np.float64,
        )
        for seed in right_seeds
    )

    best_q = seeds[0].copy()
    best_reference = hand_reference_position_in_torso(
        best_q,
        hand=side,
        reference_from_palm=reference,
    )
    best_error = float(np.linalg.norm(target - best_reference))
    best_rotation = hand_transform_in_torso(
        best_q,
        hand=side,
    )[:3, :3]
    best_orientation_error = _rotation_distance(
        target_rotation,
        best_rotation,
    )
    best_score = (
        best_error + _G1_HAND_ORIENTATION_LENGTH_SCALE_M * best_orientation_error
    )
    best_iterations = 0

    for seed in seeds:
        q = np.clip(seed, lower, upper)
        for iteration in range(1, int(max_iterations) + 1):
            reached_reference = hand_reference_position_in_torso(
                q,
                hand=side,
                reference_from_palm=reference,
            )
            error = target - reached_reference
            error_norm = float(np.linalg.norm(error))
            reached_rotation = hand_transform_in_torso(
                q,
                hand=side,
            )[:3, :3]
            orientation_error_vector = _rotation_error_vector(
                target_rotation,
                reached_rotation,
            )
            orientation_error = float(np.linalg.norm(orientation_error_vector))
            score = error_norm + _G1_HAND_ORIENTATION_LENGTH_SCALE_M * orientation_error
            if score < best_score:
                best_q = q.copy()
                best_reference = reached_reference.copy()
                best_error = error_norm
                best_rotation = reached_rotation.copy()
                best_orientation_error = orientation_error
                best_score = score
                best_iterations = iteration
            if (
                error_norm <= solver_tolerance
                and orientation_error <= orientation_tolerance
            ):
                return _result(
                    side,
                    q,
                    reached_reference,
                    target,
                    reference,
                    error_norm,
                    target_rotation,
                    orientation_error,
                    True,
                    iteration,
                )

            task_jacobian = _pose_jacobian(
                q,
                hand=side,
                reference_from_palm=reference,
                include_orientation=target_rotation is not None,
            )
            task_error = error
            if target_rotation is not None:
                task_error = np.concatenate(
                    (
                        error,
                        _G1_HAND_ORIENTATION_LENGTH_SCALE_M * orientation_error_vector,
                    )
                )
            # Damped least-squares pseudoinverse. Ordinary reaches pull toward
            # a neutral arm in the task nullspace. The live warm start remains
            # the first seed, preserving continuity without sacrificing reach.
            damping = 2.5e-3 + 1.5e-2 * min(1.0, error_norm)
            system = task_jacobian @ task_jacobian.T + (damping**2) * np.eye(
                task_jacobian.shape[0]
            )
            delta = task_jacobian.T @ np.linalg.solve(
                system,
                task_error,
            )
            right_neutral = np.asarray(
                [0.0, -0.55, 0.0, 1.20, 0.0, 0.0, 0.0],
                dtype=np.float64,
            )
            neutral = (
                right_neutral
                if side == "right"
                else np.asarray(
                    _mirror_right_joint_vector(right_neutral),
                    dtype=np.float64,
                )
            )
            nullspace = np.eye(7) - task_jacobian.T @ np.linalg.solve(
                system, task_jacobian
            )
            delta += 0.015 * (nullspace @ (neutral - q))
            delta_norm = float(np.linalg.norm(delta))
            if delta_norm > 0.16:
                delta *= 0.16 / delta_norm
            q = np.clip(q + delta, lower, upper)

    return _result(
        side,
        best_q,
        best_reference,
        target,
        reference,
        best_error,
        target_rotation,
        best_orientation_error,
        (best_error <= tolerance and best_orientation_error <= orientation_tolerance),
        best_iterations,
    )


_G1_HAND_ORIENTATION_LENGTH_SCALE_M = 0.10


def _pose_jacobian(
    q: np.ndarray,
    *,
    hand: G1ArmSide,
    reference_from_palm: np.ndarray,
    include_orientation: bool,
    epsilon: float = 1e-4,
) -> np.ndarray:
    lower_limits = _ARM_LOWER[hand]
    upper_limits = _ARM_UPPER[hand]
    rows = 6 if include_orientation else 3
    jacobian = np.empty((rows, 7), dtype=np.float64)
    for index in range(7):
        lower = max(lower_limits[index], q[index] - epsilon)
        upper = min(upper_limits[index], q[index] + epsilon)
        if upper <= lower:
            jacobian[:, index] = 0.0
            continue
        q_lower = q.copy()
        q_upper = q.copy()
        q_lower[index] = lower
        q_upper[index] = upper
        jacobian[:3, index] = (
            hand_reference_position_in_torso(
                q_upper,
                hand=hand,
                reference_from_palm=reference_from_palm,
            )
            - hand_reference_position_in_torso(
                q_lower,
                hand=hand,
                reference_from_palm=reference_from_palm,
            )
        ) / (upper - lower)
        if include_orientation:
            lower_rotation = hand_transform_in_torso(
                q_lower,
                hand=hand,
            )[:3, :3]
            upper_rotation = hand_transform_in_torso(
                q_upper,
                hand=hand,
            )[:3, :3]
            jacobian[3:, index] = (
                _G1_HAND_ORIENTATION_LENGTH_SCALE_M
                * _rotation_vector(upper_rotation @ lower_rotation.T)
                / (upper - lower)
            )
    return jacobian


def _validated_rotation(
    values: Sequence[Sequence[float]] | None,
) -> np.ndarray | None:
    if values is None:
        return None
    rotation = np.asarray(values, dtype=np.float64)
    if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
        raise ValueError("G1 arm IK target rotation must be a finite 3x3 matrix.")
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-6) or not math.isclose(
        float(np.linalg.det(rotation)),
        1.0,
        abs_tol=1e-6,
    ):
        raise ValueError("G1 arm IK target rotation must belong to SO(3).")
    return rotation


def _rotation_error_vector(
    target_rotation: np.ndarray | None,
    reached_rotation: np.ndarray,
) -> np.ndarray:
    if target_rotation is None:
        return np.zeros(3, dtype=np.float64)
    return _rotation_vector(target_rotation @ reached_rotation.T)


def _rotation_distance(
    target_rotation: np.ndarray | None,
    reached_rotation: np.ndarray,
) -> float:
    return float(
        np.linalg.norm(_rotation_error_vector(target_rotation, reached_rotation))
    )


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    cosine = float(
        np.clip(
            (np.trace(rotation) - 1.0) * 0.5,
            -1.0,
            1.0,
        )
    )
    angle = math.acos(cosine)
    skew = np.asarray(
        [
            rotation[2, 1] - rotation[1, 2],
            rotation[0, 2] - rotation[2, 0],
            rotation[1, 0] - rotation[0, 1],
        ],
        dtype=np.float64,
    )
    if angle < 1e-8:
        return 0.5 * skew
    if math.pi - angle < 1e-5:
        eigenvalues, eigenvectors = np.linalg.eig(rotation)
        axis = np.real(eigenvectors[:, int(np.argmin(np.abs(eigenvalues - 1.0)))])
        axis_norm = float(np.linalg.norm(axis))
        if axis_norm < 1e-8:
            raise RuntimeError("G1 arm IK could not resolve a pi rotation axis.")
        axis /= axis_norm
        if float(np.dot(axis, skew)) < 0.0:
            axis = -axis
        return angle * axis
    return angle * skew / (2.0 * math.sin(angle))


def rotation_matrix_from_vector(values: Sequence[float]) -> np.ndarray:
    """Return the SO(3) exponential of one finite axis-angle vector."""
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != 3 or not np.all(np.isfinite(vector)):
        raise ValueError("Rotation vector must contain three finite values.")
    angle = float(np.linalg.norm(vector))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / angle
    x, y, z = (float(value) for value in axis)
    skew = np.asarray(
        [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def rotation_vector_from_matrix(rotation: Sequence[Sequence[float]]) -> np.ndarray:
    """Return the principal axis-angle vector of one valid rotation matrix."""
    matrix = _validated_rotation(rotation)
    if matrix is None:  # pragma: no cover - a concrete matrix was provided
        raise ValueError("Rotation matrix is required.")
    return _rotation_vector(matrix)


def _joint_vector(values: Sequence[float], *, hand: G1ArmSide) -> np.ndarray:
    vector = np.asarray(values, dtype=np.float64).reshape(-1)
    if vector.size != 7 or not np.all(np.isfinite(vector)):
        raise ValueError("G1 arm joint vector must contain seven finite values.")
    return np.clip(vector, _ARM_LOWER[hand], _ARM_UPPER[hand])


def _result(
    hand: G1ArmSide,
    q: np.ndarray,
    reached_reference: np.ndarray,
    target: np.ndarray,
    reference_from_palm: np.ndarray,
    error: float,
    target_rotation: np.ndarray | None,
    orientation_error: float,
    converged: bool,
    iterations: int,
) -> G1ArmIKResult:
    palm_transform = hand_transform_in_torso(q, hand=hand)
    palm_position = palm_transform[:3, 3]
    palm_rotation = palm_transform[:3, :3]
    return G1ArmIKResult(
        hand=hand,
        joint_positions=tuple(float(value) for value in q),
        hand_position_torso=tuple(float(value) for value in palm_position),
        target_position_torso=tuple(float(value) for value in target),
        position_error_m=float(error),
        converged=bool(converged),
        iterations=int(iterations),
        reference_position_torso=tuple(float(value) for value in reached_reference),
        reference_from_palm=tuple(float(value) for value in reference_from_palm),
        hand_rotation_torso=tuple(
            tuple(float(value) for value in row) for row in palm_rotation
        ),
        target_rotation_torso=(
            None
            if target_rotation is None
            else tuple(tuple(float(value) for value in row) for row in target_rotation)
        ),
        orientation_error_rad=float(orientation_error),
    )


def _arm_side(value: str) -> G1ArmSide:
    side = str(value).strip().lower()
    if side not in {"left", "right"}:
        raise ValueError(f"G1 arm side must be left or right, got {value!r}.")
    return side  # type: ignore[return-value]


def _mirror_right_joint_vector(values: Sequence[float]) -> tuple[float, ...]:
    q = np.asarray(values, dtype=np.float64).reshape(7)
    signs = np.asarray([1.0, -1.0, -1.0, 1.0, -1.0, 1.0, -1.0])
    return tuple(float(value) for value in q * signs)


def _origin_transform(xyz: Iterable[float], rpy: Iterable[float]) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = _rpy_rotation(rpy)
    transform[:3, 3] = np.asarray(tuple(xyz), dtype=np.float64)
    return transform


def _axis_transform(axis: Iterable[float], angle: float) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    unit = np.asarray(tuple(axis), dtype=np.float64)
    unit /= np.linalg.norm(unit)
    x, y, z = unit
    cosine = math.cos(angle)
    sine = math.sin(angle)
    one_minus = 1.0 - cosine
    transform[:3, :3] = np.asarray(
        [
            [
                cosine + x * x * one_minus,
                x * y * one_minus - z * sine,
                x * z * one_minus + y * sine,
            ],
            [
                y * x * one_minus + z * sine,
                cosine + y * y * one_minus,
                y * z * one_minus - x * sine,
            ],
            [
                z * x * one_minus - y * sine,
                z * y * one_minus + x * sine,
                cosine + z * z * one_minus,
            ],
        ],
        dtype=np.float64,
    )
    return transform


def _rpy_rotation(rpy: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(value) for value in rpy)
    rx = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(roll), -math.sin(roll)],
            [0.0, math.sin(roll), math.cos(roll)],
        ],
        dtype=np.float64,
    )
    ry = np.asarray(
        [
            [math.cos(pitch), 0.0, math.sin(pitch)],
            [0.0, 1.0, 0.0],
            [-math.sin(pitch), 0.0, math.cos(pitch)],
        ],
        dtype=np.float64,
    )
    rz = np.asarray(
        [
            [math.cos(yaw), -math.sin(yaw), 0.0],
            [math.sin(yaw), math.cos(yaw), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return rz @ ry @ rx


__all__ = [
    "G1ArmIKResult",
    "G1ArmSide",
    "G1_LEFT_ARM_JOINT_NAMES",
    "G1_RIGHT_ARM_JOINT_NAMES",
    "G1_LEFT_ARM_LOWER",
    "G1_LEFT_ARM_UPPER",
    "G1_UPPER_COMMAND_SCALE",
    "frontal_grasp_palm_rotation_in_torso",
    "grasp_palm_rotation_in_torso",
    "hand_transform_in_torso",
    "hand_position_in_torso",
    "hand_reference_position_in_torso",
    "left_hand_position_in_torso",
    "right_hand_position_in_torso",
    "rotation_matrix_from_vector",
    "rotation_vector_from_matrix",
    "actuator_command_from_measured_ik_delta",
    "clamp_palm_correction_step_torso",
    "shifted_hand_pose_ik_target_torso",
    "solve_hand_pose",
    "solve_left_hand_position",
    "solve_right_hand_position",
]
