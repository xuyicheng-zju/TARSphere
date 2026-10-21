"""Robot action adapter.

The robot agent decides symbolic actions such as ``move_forward`` or
``speak``.  This adapter maps those symbols into controller-level command
payloads.  It is intentionally robot-owned; the simulator runtime only
executes the payload it receives.

This module is the single authority for the feed-forward velocity sent by each
locomotion primitive.  The simulator-adjacent executor closes the requested
relative pose using live proprioception within a fixed, geometry-blind action
window; the learned gait and native PhysX response remain authoritative.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Tuple

from tarsphere.embodiment.dex3 import (
    G1_MANIPULATION_HAND,
    dex3_continuous_joint_positions,
    dex3_controller_name,
    dex3_joint_positions,
)
from tarsphere.infra.control.physical_contracts import (
    G1_LOCOMOTION_BACKWARD_RESPONSE,
    G1_LOCOMOTION_FORWARD_RESPONSE,
    G1_LOCOMOTION_LEFT_RESPONSE,
    G1_LOCOMOTION_RIGHT_RESPONSE,
    G1_LOCOMOTION_TURN_LEFT_RESPONSE,
    G1_LOCOMOTION_TURN_RIGHT_RESPONSE,
    G1_POST_MOTION_SETTLE_S,
    G1_RELATIVE_POSE_CORRECTION_ALLOWANCE_S,
    G1LocomotionAxisResponse,
    G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M,
    G1_ROTATION_POLICY_COMMAND,
    G1_STANDING_HEIGHT_COMMAND_M,
    SIMULATION_PHYSICS_HZ,
)

_LOCOMOTION_AXIS_EPSILON = 1e-6


def _float_env(name: str, default: float) -> float:
    value = os.environ.get(name, "").strip()
    if not value:
        return float(default)
    try:
        return float(value)
    except ValueError:
        return float(default)


def commanded_linear_speed_mps() -> float:
    """Speed at which translation primitives are commanded."""
    return _float_env("TARSPHERE_MOVE_BY_SPEED_LINEAR", 0.5)


def g1_rotation_policy_command() -> float:
    """Certified normalized yaw input for the released G1 locomotion policy."""

    return float(G1_ROTATION_POLICY_COMMAND)


def _body_delta(params: Dict[str, Any]) -> tuple[float, float, float]:
    """Read the validated `[x, y, theta]` body delta without aliases."""

    raw = params.get("delta")
    if not isinstance(raw, (list, tuple)) or len(raw) != 3:
        raise ValueError("move_body requires delta=[x, y, theta]")
    x_m, y_m, theta_deg = (float(value) for value in raw)
    if not all(math.isfinite(value) for value in (x_m, y_m, theta_deg)):
        raise ValueError("move_body delta components must be finite")
    return x_m, y_m, theta_deg


def duration_physics_steps(seconds: float) -> int:
    """Physics ticks a velocity command must be held to last ``seconds``."""
    return max(1, int(round(float(seconds) * SIMULATION_PHYSICS_HZ)))


@dataclass(frozen=True)
class LocomotionControlProfile:
    """One calibrated feed-forward twist and its nominal traversal duration."""

    move_by_speed: tuple[float, float, float]
    duration_s: float
    body_twist: tuple[float, float, float]


def _se2_log_translation(
    forward_m: float,
    left_m: float,
    turn_rad: float,
) -> tuple[float, float]:
    """Return body-twist translation whose SE(2) exponential reaches a pose."""

    if abs(turn_rad) <= _LOCOMOTION_AXIS_EPSILON:
        return float(forward_m), float(left_m)
    sine = math.sin(turn_rad) / turn_rad
    cosine = (1.0 - math.cos(turn_rad)) / turn_rad
    determinant = sine * sine + cosine * cosine
    if determinant <= 1e-12:
        raise ValueError(
            "move_body cannot resolve a finite body twist for this yaw delta"
        )
    return (
        (sine * forward_m + cosine * left_m) / determinant,
        (-cosine * forward_m + sine * left_m) / determinant,
    )


def _translation_response(value: float) -> G1LocomotionAxisResponse:
    return (
        G1_LOCOMOTION_FORWARD_RESPONSE
        if value >= 0.0
        else G1_LOCOMOTION_BACKWARD_RESPONSE
    )


def _lateral_response(value: float) -> G1LocomotionAxisResponse:
    return G1_LOCOMOTION_LEFT_RESPONSE if value >= 0.0 else G1_LOCOMOTION_RIGHT_RESPONSE


def _rotation_response(value: float) -> G1LocomotionAxisResponse:
    return (
        G1_LOCOMOTION_TURN_LEFT_RESPONSE
        if value >= 0.0
        else G1_LOCOMOTION_TURN_RIGHT_RESPONSE
    )


def _profile_for_body_pose(
    forward_m: float,
    left_m: float,
    turn_rad: float,
) -> LocomotionControlProfile:
    """Convert one desired relative pose into one calibrated open-loop twist.

    Translation and rotation remain simultaneous. The SE(2) logarithm fixes
    the old component-wise integration error without selecting a path or
    reading scene geometry. Direction-specific response fits then choose a
    shared duration and bounded controller velocities.
    """

    twist_forward, twist_left = _se2_log_translation(
        forward_m,
        left_m,
        turn_rad,
    )
    active_translation: list[tuple[float, G1LocomotionAxisResponse]] = []
    if abs(twist_forward) > _LOCOMOTION_AXIS_EPSILON:
        active_translation.append((twist_forward, _translation_response(twist_forward)))
    if abs(twist_left) > _LOCOMOTION_AXIS_EPSILON:
        active_translation.append((twist_left, _lateral_response(twist_left)))

    translation_duration = 0.0
    if active_translation:
        shared_startup_loss = max(
            response.startup_loss_s for _, response in active_translation
        )
        calibrated_components = [
            value / response.gain for value, response in active_translation
        ]
        calibrated_twist_norm = (
            math.hypot(*calibrated_components)
            if len(calibrated_components) == 2
            else abs(calibrated_components[0])
        )
        maximum_linear_speed = commanded_linear_speed_mps()
        if not math.isfinite(maximum_linear_speed) or maximum_linear_speed <= 0:
            raise ValueError(
                "TARSPHERE_MOVE_BY_SPEED_LINEAR must be finite and positive"
            )
        translation_duration = (
            shared_startup_loss + calibrated_twist_norm / maximum_linear_speed
        )

    rotation_response = _rotation_response(turn_rad)
    rotation_duration = (
        rotation_response.startup_loss_s
        + abs(turn_rad) / (rotation_response.gain * g1_rotation_policy_command())
        if abs(turn_rad) > _LOCOMOTION_AXIS_EPSILON
        else 0.0
    )
    duration_s = max(translation_duration, rotation_duration)
    if duration_s <= 0.0:
        raise ValueError("move_body requires a non-zero body delta")

    def command_for_axis(
        value: float,
        response: G1LocomotionAxisResponse,
    ) -> float:
        if abs(value) <= _LOCOMOTION_AXIS_EPSILON:
            return 0.0
        effective_seconds = duration_s - response.startup_loss_s
        if effective_seconds <= 0.0:
            raise RuntimeError("calibrated locomotion window is non-positive")
        return value / (response.gain * effective_seconds)

    forward_speed = command_for_axis(
        twist_forward,
        _translation_response(twist_forward),
    )
    left_speed = command_for_axis(
        twist_left,
        _lateral_response(twist_left),
    )
    rotation_speed = command_for_axis(turn_rad, rotation_response)
    linear_speed = math.hypot(forward_speed, left_speed)
    maximum_linear_speed = commanded_linear_speed_mps()
    if linear_speed > maximum_linear_speed + 1e-9:
        raise RuntimeError(
            "calibrated locomotion profile exceeded the linear command limit: "
            f"{linear_speed} > {maximum_linear_speed}"
        )
    if abs(rotation_speed) > g1_rotation_policy_command() + 1e-9:
        raise RuntimeError(
            "calibrated locomotion profile exceeded the yaw command limit: "
            f"{rotation_speed}"
        )
    return LocomotionControlProfile(
        move_by_speed=(forward_speed, left_speed, rotation_speed),
        duration_s=duration_s,
        body_twist=(twist_forward, twist_left, turn_rad),
    )


def locomotion_control_profile(
    action: str,
    *,
    params: Dict[str, Any] | None = None,
) -> LocomotionControlProfile | None:
    """Return the single authoritative metric-motion controller profile."""

    target = metric_body_delta(action, params=params)
    return _profile_for_body_pose(*target) if target is not None else None


def metric_body_delta(
    action: str,
    *,
    params: Dict[str, Any] | None = None,
) -> tuple[float, float, float] | None:
    """Return one public metric command as ``(forward, left, yaw_rad)``.

    This is the relative-pose contract shared by the feed-forward router and
    the runtime feedback controller.  Actions without an explicit metric
    magnitude do not acquire a hidden target.
    """

    values = params or {}
    if action in {
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
    }:
        distance_m = values.get("distance_m")
        if distance_m is None:
            return None
        distance = abs(float(distance_m))
        return {
            "move_forward": (distance, 0.0, 0.0),
            "move_backward": (-distance, 0.0, 0.0),
            "move_left": (0.0, distance, 0.0),
            "move_right": (0.0, -distance, 0.0),
        }[action]
    if action in {"turn_left", "turn_right"}:
        degrees = values.get("degrees")
        if degrees is None:
            return None
        yaw = abs(math.radians(float(degrees)))
        return (0.0, 0.0, yaw if action == "turn_left" else -yaw)
    if action == "move_body":
        forward_m, left_m, turn_deg = _body_delta(values)
        return forward_m, left_m, math.radians(turn_deg)
    return None


def locomotion_physics_steps(
    action: str,
    *,
    params: Dict[str, Any] | None = None,
) -> int | None:
    """Convert one public motion magnitude into its nominal traversal window.

    Metric motions use the same calibrated profile that supplies their
    feed-forward velocity. The runtime adds one fixed correction/settling
    allowance around this nominal window; it never grows the allowance in
    response to an obstacle or residual error.

    Actions without a public magnitude return ``None`` so their caller-owned
    physical horizon remains authoritative.
    """

    values = params or {}
    profile = locomotion_control_profile(action, params=values)
    if profile is not None:
        return duration_physics_steps(profile.duration_s)
    if action == "wait":
        seconds = values.get("seconds")
        if seconds is None:
            return None
        return duration_physics_steps(float(seconds))
    return None


def relative_pose_control_physics_steps(
    action: str,
    *,
    params: Dict[str, Any] | None = None,
    post_motion_settle_seconds: float = G1_POST_MOTION_SETTLE_S,
) -> int | None:
    """Return the bounded feedback window for a metric body-pose command.

    The window contains the calibrated nominal traversal, a fixed correction
    allowance, and the zero-speed interval whose final pose must remain within
    tolerance.  It never grows in response to an obstacle or residual error.
    """

    profile = locomotion_control_profile(action, params=params)
    target = metric_body_delta(action, params=params)
    if profile is None or target is None:
        return None
    settle_seconds = float(post_motion_settle_seconds)
    if not math.isfinite(settle_seconds) or settle_seconds < 0.0:
        raise ValueError("post_motion_settle_seconds must be finite and non-negative")
    return (
        duration_physics_steps(profile.duration_s)
        + duration_physics_steps(G1_RELATIVE_POSE_CORRECTION_ALLOWANCE_S)
        + max(1, int(round(settle_seconds * SIMULATION_PHYSICS_HZ)))
    )


_GRASP_CONTROLLER_KEYS = frozenset({"hand", "preset", "reach"})


def decision_has_compiled_controller(decision: Mapping[str, Any]) -> bool:
    """Whether a private policy supplied a controller for the runtime to execute.

    Certification policies may emit primitives outside the public action
    vocabulary, with a ready controller payload or placement IK metadata.
    Both action-space admission and physical dispatch use this same contract.
    """

    if "controller_action" in decision:
        return True
    metadata = decision.get("controller_metadata")
    return isinstance(metadata, dict) and isinstance(metadata.get("g1_place_ik"), dict)


def requires_environment_execution(decision: Mapping[str, Any]) -> bool:
    """True when the environment controller must run this payload.

    A completed in-handler tool records the tool identity as ``action``.
    That identity is not a controller command. Failed calls keep the same
    identity and never move the body. Closed-loop NavDP is physical even
    when the public action name is the navigation tool rather than
    ``move_body``.
    """

    if decision.get("is_error"):
        return False
    results = decision.get("tool_results")
    if isinstance(results, list) and any(
        isinstance(row, Mapping) and row.get("is_error") for row in results
    ):
        return False
    name = str(decision.get("action") or "")
    if not name:
        raise ValueError(
            "decision has no action; every completed tool owns a runtime payload"
        )
    if decision_has_compiled_controller(decision):
        return True
    navigation = decision.get("navigation_command")
    if isinstance(navigation, Mapping) and navigation.get("backend") in {
        "x_navdp",
        "x_navdp_fallback",
    }:
        return True
    try:
        ActionRouter().route(name, params=dict(decision))
    except (TypeError, ValueError):
        return False
    if name in {"grasp_object", "place_object"}:
        return any(key in decision for key in _GRASP_CONTROLLER_KEYS)
    return True


class ActionRouter:
    """Map robot action names to controller command dicts.

    Each route() call returns ``(env_action, public_feedback)``.  The current
    TARSphere controller uses payloads such as
    ``{'move_by_speed': [vx, vy, vtheta]}``.
    """

    def route(
        self,
        action: str,
        *,
        params: Dict[str, Any] | None = None,
    ) -> Tuple[Dict[str, Any], str]:
        params = params or {}
        profile = locomotion_control_profile(action, params=params)

        if action == "move_forward":
            if profile is not None:
                return {"move_by_speed": list(profile.move_by_speed)}, ""
            return {
                "move_by_speed": [
                    commanded_linear_speed_mps(),
                    0.0,
                    0.0,
                ],
            }, ""
        if action == "move_backward":
            if profile is not None:
                return {"move_by_speed": list(profile.move_by_speed)}, ""
            return {
                "move_by_speed": [
                    -commanded_linear_speed_mps(),
                    0.0,
                    0.0,
                ],
            }, ""
        if action == "move_left":
            if profile is not None:
                return {"move_by_speed": list(profile.move_by_speed)}, ""
            return {
                "move_by_speed": [
                    0.0,
                    commanded_linear_speed_mps(),
                    0.0,
                ],
            }, ""
        if action == "move_right":
            if profile is not None:
                return {"move_by_speed": list(profile.move_by_speed)}, ""
            return {
                "move_by_speed": [
                    0.0,
                    -commanded_linear_speed_mps(),
                    0.0,
                ],
            }, ""
        if action == "turn_left":
            if profile is not None:
                return {"move_by_speed": list(profile.move_by_speed)}, ""
            return {
                "move_by_speed": [
                    0.0,
                    0.0,
                    g1_rotation_policy_command(),
                ],
            }, ""
        if action == "turn_right":
            if profile is not None:
                return {"move_by_speed": list(profile.move_by_speed)}, ""
            return {
                "move_by_speed": [
                    0.0,
                    0.0,
                    -g1_rotation_policy_command(),
                ],
            }, ""
        if action == "move_body":
            if profile is None:
                raise ValueError("move_body requires a calibrated profile")
            return {"move_by_speed": list(profile.move_by_speed)}, ""
        if action == "rotate_to":
            return {
                "rotate": [self._quat_required(params.get("target_orientation"))],
            }, ""
        if action == "crouch":
            pelvis_height_m = float(params.get("pelvis_height_m"))
            if not (
                G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M
                <= pelvis_height_m
                <= G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M
            ):
                raise ValueError(
                    "crouch pelvis_height_m must be between "
                    f"{G1_MINIMUM_CROUCH_HEIGHT_COMMAND_M} and "
                    f"{G1_MAXIMUM_CROUCH_HEIGHT_COMMAND_M} metres"
                )
            return {
                "move_by_speed": [
                    0.0,
                    0.0,
                    0.0,
                    pelvis_height_m,
                ],
            }, ""
        if action == "stand_up":
            return {
                "move_by_speed": [
                    0.0,
                    0.0,
                    0.0,
                    G1_STANDING_HEIGHT_COMMAND_M,
                ],
            }, ""

        if action in {"speak", "ask", "report_done", "handover_object"}:
            # Language is dispatched by AutonomousEpisodeRunner, not by an
            # TARSphere robot controller.  The learned G1 locomotion policy
            # must still run at zero velocity throughout the decision window
            # or the unsupported action/message keys leave the humanoid
            # uncontrolled long enough to fall.
            return {"move_by_speed": [0.0, 0.0, 0.0]}, ""

        if action in {"grasp_object", "place_object"}:
            hand = self._g1_hand(
                params.get("hand", G1_MANIPULATION_HAND),
                working_hand_only=True,
            )
            preset = self._dex3_preset(action, params)
            return {
                "move_by_speed": self._reach_or_stand(params),
                dex3_controller_name(hand): [list(dex3_joint_positions(hand, preset))],
            }, ""

        if action == "move_end_effector":
            delta = params.get("delta")
            if not isinstance(delta, (list, tuple)) or len(delta) != 7:
                raise ValueError("move_end_effector requires a seven-number delta.")
            closure = float(delta[6])
            if not math.isfinite(closure):
                raise ValueError("move_end_effector gripper closure must be finite.")
            hand = G1_MANIPULATION_HAND
            return {
                dex3_controller_name(hand): [
                    [*dex3_continuous_joint_positions(hand, closure)]
                ],
            }, ""

        if action == "set_g1_gripper_preset":
            hand = self._g1_hand(params.get("hand", G1_MANIPULATION_HAND))
            preset = self._dex3_preset(action, params)
            return {
                "move_by_speed": self._reach_or_stand(params),
                dex3_controller_name(hand): [list(dex3_joint_positions(hand, preset))],
            }, ""

        if action in {"wait", "tilt_camera"}:
            # Same as speech: G1's learned loco policy must keep receiving
            # zero-velocity commands or an empty env action leaves the body
            # uncontrolled and it falls within a few physics steps.
            # ``tilt_camera`` belongs here because it aims the camera prim rather
            # than the body: the runtime applies the angle, and the body's only
            # job for the step is to stand still.
            return {"move_by_speed": [0.0, 0.0, 0.0]}, ""

        raise ValueError(f"ActionRouter: unknown action {action!r}")

    @staticmethod
    def _reach_or_stand(params: Dict[str, Any]) -> list[float]:
        reach = params.get("reach")
        if reach is None:
            return [0.0, 0.0, 0.0]
        items = [float(v) for v in reach]
        if len(items) not in {3, 18, 19}:
            raise ValueError(
                f"reach move_by_speed must be length 3, 18, or 19; got {len(items)}."
            )
        return items

    @staticmethod
    def _quat_required(value: Any) -> list[float]:
        try:
            items = [float(v) for v in value]
        except TypeError as exc:
            raise ValueError(
                "target_orientation must be a quaternion [w, x, y, z]"
            ) from exc
        if len(items) != 4:
            raise ValueError(
                f"target_orientation must contain exactly 4 numbers, got {items!r}"
            )
        return items

    @staticmethod
    def _g1_hand(value: Any, *, working_hand_only: bool = False) -> str:
        hand = str(value or "").strip().lower()
        if hand not in {"left", "right"}:
            raise ValueError(f"G1 gripper hand must be left or right, got {value!r}")
        if working_hand_only and hand != G1_MANIPULATION_HAND:
            raise ValueError("G1 manipulation requires the certified right work hand.")
        return hand

    @staticmethod
    def _dex3_preset(
        action: str,
        params: Dict[str, Any],
    ) -> str:
        if action == "grasp_object":
            return "grasp"
        if action == "place_object":
            return "open"
        preset = str(params.get("preset") or "").strip().lower()
        if preset not in {"open", "grasp"}:
            raise ValueError(f"Dex3 preset must be open or grasp, got {preset!r}.")
        return preset


__all__ = [
    "ActionRouter",
    "commanded_linear_speed_mps",
    "duration_physics_steps",
    "g1_rotation_policy_command",
    "locomotion_control_profile",
    "locomotion_physics_steps",
    "metric_body_delta",
    "relative_pose_control_physics_steps",
    "requires_environment_execution",
]
