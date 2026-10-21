"""Bounded proprioceptive tracking for one Agent-requested body pose.

The controller never reads occupancy, depth, contacts, or task geometry.  It
only adjusts the velocity command sent to the released G1 locomotion policy
from the live root pose.  Consequently PhysX remains free to block, deflect,
or topple the robot; a target that cannot be reached simply exhausts its fixed
action window and reports the measured result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from tarsphere.harness.actions.router import LocomotionControlProfile
from tarsphere.infra.control.physical_contracts import (
    G1_RELATIVE_POSE_MIN_TRANSLATION_COMMAND_MPS,
    G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M,
    G1_RELATIVE_POSE_LATERAL_YAW_TOLERANCE_RAD,
    G1_RELATIVE_POSE_ROTATION_TRANSLATION_TOLERANCE_M,
    G1_RELATIVE_POSE_TRANSLATION_GAIN_S_INV,
    G1_RELATIVE_POSE_TRANSLATION_TOLERANCE_M,
    G1_RELATIVE_POSE_YAW_GAIN_S_INV,
    G1_RELATIVE_POSE_YAW_TOLERANCE_RAD,
    G1_ROTATION_POLICY_COMMAND,
    SIMULATION_PHYSICS_HZ,
)


def _wrap_angle(value: float) -> float:
    return math.atan2(math.sin(value), math.cos(value))


@dataclass(frozen=True)
class BodyPose2D:
    x_m: float
    y_m: float
    yaw_rad: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (
                self.x_m,
                self.y_m,
                self.yaw_rad,
            )
        ):
            raise ValueError(f"Body pose must be finite: {self!r}.")


BodyPoseSampler = Callable[[], BodyPose2D]


def world_xy_from_body_displacement(
    *,
    displacement_forward_left_m: Any,
    body_world_xy: Any,
    body_yaw_rad: float,
) -> tuple[float, float]:
    """Resolve a planar body-frame displacement in the world XY frame."""

    displacement = tuple(float(value) for value in displacement_forward_left_m)
    origin = tuple(float(value) for value in body_world_xy)
    yaw = float(body_yaw_rad)
    if len(displacement) != 2 or len(origin) != 2:
        raise ValueError("body displacement and world origin must each be planar")
    if not all(math.isfinite(value) for value in (*displacement, *origin, yaw)):
        raise ValueError("body-to-world PointGoal inputs must be finite")
    forward, left = displacement
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return (
        origin[0] + cosine * forward - sine * left,
        origin[1] + sine * forward + cosine * left,
    )


def _scaled_twist_pose(
    twist: tuple[float, float, float],
    fraction: float,
) -> tuple[float, float, float]:
    """Apply ``Exp(fraction * twist)`` in the starting body frame."""

    forward_twist, left_twist, yaw = twist
    fraction = min(1.0, max(0.0, float(fraction)))
    scaled_yaw = fraction * yaw
    scaled_forward = fraction * forward_twist
    scaled_left = fraction * left_twist
    if abs(scaled_yaw) <= 1e-9:
        return scaled_forward, scaled_left, scaled_yaw
    sine = math.sin(scaled_yaw) / scaled_yaw
    cosine = (1.0 - math.cos(scaled_yaw)) / scaled_yaw
    return (
        sine * scaled_forward - cosine * scaled_left,
        cosine * scaled_forward + sine * scaled_left,
        scaled_yaw,
    )


def _world_from_start(
    start: BodyPose2D,
    relative: tuple[float, float, float],
) -> BodyPose2D:
    forward, left, yaw = relative
    cosine = math.cos(start.yaw_rad)
    sine = math.sin(start.yaw_rad)
    return BodyPose2D(
        x_m=start.x_m + cosine * forward - sine * left,
        y_m=start.y_m + sine * forward + cosine * left,
        yaw_rad=_wrap_angle(start.yaw_rad + yaw),
    )


def _body_error(current: BodyPose2D, target: BodyPose2D) -> tuple[float, float, float]:
    dx = target.x_m - current.x_m
    dy = target.y_m - current.y_m
    cosine = math.cos(current.yaw_rad)
    sine = math.sin(current.yaw_rad)
    return (
        cosine * dx + sine * dy,
        -sine * dx + cosine * dy,
        _wrap_angle(target.yaw_rad - current.yaw_rad),
    )


def _replace_base_velocity(
    template: Mapping[str, Any],
    velocity: tuple[float, float, float],
) -> dict[str, Any]:
    action = dict(template)
    move = action.get("move_by_speed")
    if not isinstance(move, (list, tuple)) or len(move) < 3:
        raise ValueError(
            "Relative-pose tracking requires a move_by_speed command template."
        )
    updated = list(move)
    updated[:3] = [float(value) for value in velocity]
    action["move_by_speed"] = updated
    return action


class RelativePoseController:
    """Track one fixed relative pose through bounded G1 velocity commands."""

    def __init__(
        self,
        *,
        target_delta: tuple[float, float, float],
        profile: LocomotionControlProfile,
        command_template: Mapping[str, Any],
        sample_pose: BodyPoseSampler,
        stable_zero_ticks: int,
        physics_hz: float = SIMULATION_PHYSICS_HZ,
        maximum_linear_speed_mps: float = 0.5,
    ) -> None:
        if stable_zero_ticks < 1:
            raise ValueError("Relative-pose tracking requires a zero-speed hold.")
        self.target_delta = tuple(float(value) for value in target_delta)
        self.profile = profile
        self.command_template = dict(command_template)
        self.sample_pose = sample_pose
        self.stable_zero_ticks_required = int(stable_zero_ticks)
        self.physics_hz = float(physics_hz)
        self.maximum_linear_speed_mps = float(maximum_linear_speed_mps)
        if self.physics_hz <= 0.0 or self.maximum_linear_speed_mps <= 0.0:
            raise ValueError("Relative-pose control rates and limits must be positive.")

        self.start = self.sample_pose()
        self.goal = _world_from_start(self.start, self.target_delta)
        forward_target, left_target, yaw_target = self.target_delta
        self.pure_rotation = bool(
            math.hypot(forward_target, left_target) <= 1e-9 and abs(yaw_target) > 1e-9
        )
        pure_lateral = bool(
            abs(forward_target) <= 1e-9
            and abs(left_target) > 1e-9
            and abs(yaw_target) <= 1e-9
        )
        self.translation_tolerance_m = (
            G1_RELATIVE_POSE_ROTATION_TRANSLATION_TOLERANCE_M
            if self.pure_rotation
            else (
                G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M
                if pure_lateral
                else G1_RELATIVE_POSE_TRANSLATION_TOLERANCE_M
            )
        )
        self.yaw_tolerance_rad = (
            G1_RELATIVE_POSE_LATERAL_YAW_TOLERANCE_RAD
            if pure_lateral
            else G1_RELATIVE_POSE_YAW_TOLERANCE_RAD
        )
        self.nominal_ticks = max(
            1,
            int(round(self.profile.duration_s * self.physics_hz)),
        )
        self._settling = False
        self._stable_zero_ticks = 0
        # A pure turn leaves the released gait in its turning limit cycle.
        # Give zero velocity one complete settle interval before accepting it.
        # Appending a translation leg is not an in-place turn and empirically
        # makes the settled pose worse, so pure turns remain yaw-only.
        self._rotation_transition_complete = not self.pure_rotation
        self._rotation_transition_zero_ticks = 0
        self._rotation_transition_pose: BodyPose2D | None = None
        self._rotation_terminal_tick_phase = "inactive"
        self._settling_entry_count = 0
        self._maximum_stable_zero_ticks = 0
        self._last_command_zero = False
        self._last_tick = -1
        self._completed = False
        self._maximum_reference_error_m = 0.0
        self._final_translation_error_m = math.inf
        self._final_yaw_error_rad = math.inf

    @property
    def completed(self) -> bool:
        """Whether the target survived the required consecutive zero ticks."""

        return self._completed

    @staticmethod
    def _bounded_linear(
        forward: float,
        left: float,
        maximum: float,
    ) -> tuple[float, float]:
        magnitude = math.hypot(forward, left)
        if magnitude <= maximum:
            return forward, left
        scale = maximum / magnitude
        return forward * scale, left * scale

    def _minimum_linear_correction(
        self,
        forward: float,
        left: float,
        error_m: float,
    ) -> tuple[float, float]:
        magnitude = math.hypot(forward, left)
        minimum = min(
            G1_RELATIVE_POSE_MIN_TRANSLATION_COMMAND_MPS,
            self.maximum_linear_speed_mps,
        )
        if (
            error_m <= self.translation_tolerance_m
            or magnitude <= 1e-12
            or magnitude >= minimum
        ):
            return forward, left
        scale = minimum / magnitude
        return forward * scale, left * scale

    def _reference(self, tick: int) -> BodyPose2D:
        fraction = min(1.0, float(tick + 1) / float(self.nominal_ticks))
        return _world_from_start(
            self.start,
            _scaled_twist_pose(self.profile.body_twist, fraction),
        )

    def action_for_tick(self, tick: int) -> dict[str, Any]:
        """Return the next policy velocity; never modify physical pose state."""

        self._last_tick = int(tick)
        if self._settling:
            self._last_command_zero = True
            return _replace_base_velocity(
                self.command_template,
                (0.0, 0.0, 0.0),
            )

        if (
            self.pure_rotation
            and tick >= self.nominal_ticks
            and not self._rotation_transition_complete
        ):
            current = self.sample_pose()
            _, _, error_yaw = _body_error(current, self.goal)
            if abs(error_yaw) <= self.yaw_tolerance_rad:
                velocity = (0.0, 0.0, 0.0)
            else:
                velocity = (
                    0.0,
                    0.0,
                    math.copysign(G1_ROTATION_POLICY_COMMAND, error_yaw),
                )
            self._last_command_zero = not any(abs(value) > 1e-12 for value in velocity)
            return _replace_base_velocity(self.command_template, velocity)

        if (
            self.pure_rotation
            and tick >= self.nominal_ticks
            and self._rotation_transition_complete
        ):
            self._rotation_terminal_tick_phase = "turn_zero_hold"
            self._last_command_zero = True
            return _replace_base_velocity(
                self.command_template,
                (0.0, 0.0, 0.0),
            )

        current = self.sample_pose()
        reference = self._reference(tick)
        error_forward, error_left, error_yaw = _body_error(current, reference)
        reference_error = math.hypot(error_forward, error_left)
        self._maximum_reference_error_m = max(
            self._maximum_reference_error_m,
            reference_error,
        )
        nominal_active = tick < self.nominal_ticks
        feed_forward, feed_left, feed_yaw = (
            self.profile.move_by_speed if nominal_active else (0.0, 0.0, 0.0)
        )
        forward = feed_forward + G1_RELATIVE_POSE_TRANSLATION_GAIN_S_INV * error_forward
        left = feed_left + G1_RELATIVE_POSE_TRANSLATION_GAIN_S_INV * error_left
        forward, left = self._bounded_linear(
            forward,
            left,
            self.maximum_linear_speed_mps,
        )
        if not nominal_active:
            forward, left = self._minimum_linear_correction(
                forward,
                left,
                math.hypot(error_forward, error_left),
            )

        yaw = feed_yaw + G1_RELATIVE_POSE_YAW_GAIN_S_INV * error_yaw
        yaw = max(
            -G1_ROTATION_POLICY_COMMAND,
            min(G1_ROTATION_POLICY_COMMAND, yaw),
        )
        if (
            not nominal_active
            and abs(error_yaw) > self.yaw_tolerance_rad
            and 1e-12 < abs(yaw) < G1_ROTATION_POLICY_COMMAND
        ):
            yaw = math.copysign(G1_ROTATION_POLICY_COMMAND, yaw)

        self._last_command_zero = bool(
            abs(forward) <= 1e-12 and abs(left) <= 1e-12 and abs(yaw) <= 1e-12
        )
        return _replace_base_velocity(
            self.command_template,
            (forward, left, yaw),
        )

    def observe_after_tick(self) -> bool:
        """Update goal/zero-speed stability and report physical completion."""

        current = self.sample_pose()
        error_forward, error_left, error_yaw = _body_error(current, self.goal)
        self._final_translation_error_m = math.hypot(error_forward, error_left)
        self._final_yaw_error_rad = abs(error_yaw)
        reference_complete = self._last_tick + 1 >= self.nominal_ticks
        if (
            self.pure_rotation
            and reference_complete
            and not self._rotation_transition_complete
        ):
            if self._final_yaw_error_rad > self.yaw_tolerance_rad:
                self._rotation_transition_zero_ticks = 0
                return False
            if self._last_command_zero:
                self._rotation_transition_zero_ticks += 1
            else:
                self._rotation_transition_zero_ticks = 0
            self._rotation_transition_complete = bool(
                self._rotation_transition_zero_ticks >= self.stable_zero_ticks_required
            )
            if self._rotation_transition_complete:
                self._rotation_transition_pose = current
                self._rotation_terminal_tick_phase = "turn_zero_hold"
                if self._final_translation_error_m <= self.translation_tolerance_m:
                    self._settling = True
                    self._stable_zero_ticks = self._rotation_transition_zero_ticks
                    self._maximum_stable_zero_ticks = max(
                        self._maximum_stable_zero_ticks,
                        self._stable_zero_ticks,
                    )
                    self._completed = True
                    return True
            return False
        within = bool(
            reference_complete
            and self._final_translation_error_m <= self.translation_tolerance_m
            and self._final_yaw_error_rad <= self.yaw_tolerance_rad
        )
        if not within:
            self._settling = False
            self._stable_zero_ticks = 0
            return False
        if self._last_command_zero:
            self._stable_zero_ticks += 1
        else:
            self._stable_zero_ticks = 0
        if not self._settling:
            self._settling_entry_count += 1
        self._settling = True
        self._maximum_stable_zero_ticks = max(
            self._maximum_stable_zero_ticks,
            self._stable_zero_ticks,
        )
        self._completed = self._stable_zero_ticks >= self.stable_zero_ticks_required
        return self._completed

    def report(self, *, timed_out: bool) -> dict[str, Any]:
        """Return observable controller status without claiming blocked success."""

        current = self.sample_pose()
        error_forward, error_left, error_yaw = _body_error(current, self.goal)
        final_translation = math.hypot(error_forward, error_left)
        final_yaw = abs(error_yaw)
        return {
            "target_reached": bool(
                self._completed
                and final_translation <= self.translation_tolerance_m
                and final_yaw <= self.yaw_tolerance_rad
            ),
            "timed_out": bool(timed_out),
            "translation_tolerance_m": (self.translation_tolerance_m),
            "yaw_tolerance_deg": math.degrees(self.yaw_tolerance_rad),
            "final_translation_error_m": float(final_translation),
            "final_yaw_error_deg": math.degrees(final_yaw),
            "stable_zero_ticks": int(self._stable_zero_ticks),
            "required_stable_zero_ticks": int(self.stable_zero_ticks_required),
            "rotation_transition_zero_ticks": int(self._rotation_transition_zero_ticks),
            "rotation_transition_complete": bool(self._rotation_transition_complete),
            "rotation_transition_pose": (
                {
                    "x_m": float(self._rotation_transition_pose.x_m),
                    "y_m": float(self._rotation_transition_pose.y_m),
                    "yaw_rad": float(self._rotation_transition_pose.yaw_rad),
                }
                if self._rotation_transition_pose is not None
                else None
            ),
            "rotation_terminal_tick_phase": str(self._rotation_terminal_tick_phase),
            "settling_entry_count": int(self._settling_entry_count),
            "maximum_stable_zero_ticks": int(self._maximum_stable_zero_ticks),
            "control_mode": (
                "continuous_live_pose_feedback_then_gait_phase_handoff_then_zero_settle"
            ),
            "maximum_reference_position_error_m": float(
                self._maximum_reference_error_m
            ),
            "controller": "bounded_proprioceptive_relative_pose",
            "uses_scene_geometry": False,
            "overrides_physx_pose": False,
        }


__all__ = [
    "BodyPose2D",
    "RelativePoseController",
    "world_xy_from_body_displacement",
]
