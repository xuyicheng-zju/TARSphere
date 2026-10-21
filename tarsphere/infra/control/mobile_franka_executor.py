"""Mobile Franka executor for the benchmark Physical Task contract.

The oracle sequence is intentionally the same one used by the G1 executor:
the robot navigates to a source stance, reaches a natural grasp pose, closes
its gripper, creates a FixedJoint only after the pose gate passes, carries the
object through ordinary base control, and releases it at the destination.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from tarsphere.infra.manipulation.g1_attachment import (
    G1ManipulationObject,
    _activate_pickable_collider,
    _create_attachment_joint,
    _enable_pickable_physics,
    _read_object_local_transform,
    _restore_pickable_local_transform,
    _set_pickable_kinematic,
    _set_pickable_mass,
    _set_pickable_metric_world_pose,
    _world_pose,
    bbox_surface_gap,
)
from tarsphere.infra.simulation.usd_geometry import world_bounds


MOBILE_FRANKA_PICKABLE_MASS_KG = 1e-6
MOBILE_FRANKA_NAVIGATION_TOLERANCE_M = 0.10
MOBILE_FRANKA_MANIPULATION_HEADING_TOLERANCE_RAD = 0.15
MOBILE_FRANKA_PLACEMENT_TOLERANCE_M = 0.05
MOBILE_FRANKA_CARRY_FORWARD_OFFSET_M = 0.30
MOBILE_FRANKA_CARRY_HEIGHT_M = 1.05
MOBILE_FRANKA_VERTICAL_APPROACH_M = 0.20
# Radius of the complete folded robot collision envelope in the authored USD.
# It is deliberately larger than the 0.21 m chassis half-length.
MOBILE_FRANKA_FOLDED_COLLISION_RADIUS_M = 0.28
# The certified Mobile Franka dynamics and arm target slew use 240 Hz physics.
MOBILE_FRANKA_DEFAULT_PHYSICS_HZ = 240
MOBILE_FRANKA_NAVIGATION_CONTROL_HZ = 20
MOBILE_FRANKA_NAVIGATION_TIMEOUT_S = 60.0
MOBILE_FRANKA_NAVIGATION_PROGRESS_WINDOW_S = 2.0
MOBILE_FRANKA_NAVIGATION_MIN_TRANSLATION_M = 0.03
MOBILE_FRANKA_NAVIGATION_MIN_ROTATION_RAD = 0.05
MOBILE_FRANKA_ROTATION_PROGRESS_WINDOW_S = 2.0
MOBILE_FRANKA_ROTATION_MIN_PROGRESS_RAD = 0.05
MOBILE_FRANKA_ROTATION_TIMEOUT_S = 12.5
MOBILE_FRANKA_ARM_TIMEOUT_S = 12.0
MOBILE_FRANKA_GRIPPER_COMMAND_S = 0.375
MOBILE_FRANKA_PLACEMENT_SETTLE_S = 1.0
MOBILE_FRANKA_ROTATE_SETTLE_S = 0.5
MOBILE_FRANKA_ARM_SETTLE_S = 0.125
MOBILE_FRANKA_STATIONARY_CONFIRMATION_S = 0.25
MOBILE_FRANKA_STATIONARY_TIMEOUT_S = 3.0
MOBILE_FRANKA_NAVIGATION_WHEEL_MAX_FORCE_NM = 10.0
MOBILE_FRANKA_TURN_AND_HOLD_WHEEL_MAX_FORCE_NM = 50.0
MOBILE_FRANKA_STATIONARY_LINEAR_SPEED_M_S = 0.015
MOBILE_FRANKA_STATIONARY_ANGULAR_SPEED_RAD_S = 0.03


def _positive_hz(value: float) -> int:
    hz = int(round(float(value)))
    if hz < 1:
        raise ValueError("physics_hz must be a positive integer.")
    return hz


def _ticks_for_seconds(seconds: float, *, physics_hz: int) -> int:
    return max(1, int(round(float(seconds) * float(physics_hz))))


def _control_interval_ticks(
    *,
    physics_hz: int,
    control_hz: float,
) -> int:
    return max(1, int(round(float(physics_hz) / float(control_hz))))


@dataclass(frozen=True)
class MobileFrankaTask:
    """One metric navigation–grasp–carry–place task."""

    # Exact semantic object id written into certificate evidence.
    object_id: str
    # Live USD prim for the movable task object.
    object_prim_path: str
    # Metres represented by one authored unit inside the object's USD asset.
    object_local_unit_in_meters: float
    # Collision-free route from spawn to the source manipulation stance.
    source_route_xyz: tuple[tuple[float, float, float], ...]
    # Near-field source stance entered only after facing the object.
    source_stance_xyz: tuple[float, float, float]
    # Robot yaw at the source manipulation stance.
    source_yaw_rad: float
    # World-space gripper target that encloses the object between the fingers.
    grasp_position_xyz: tuple[float, float, float]
    # Collision-free object-carrying route to the destination stance.
    carry_route_xyz: tuple[tuple[float, float, float], ...]
    # Near-field destination stance entered only after facing the surface.
    destination_stance_xyz: tuple[float, float, float]
    # Robot yaw at the destination manipulation stance.
    destination_yaw_rad: float
    # World-space gripper target used immediately before release.
    place_position_xyz: tuple[float, float, float]
    # Expected placement surface centre in world coordinates.
    destination_surface_xyz: tuple[float, float, float]
    # Half-size of the valid destination surface in world XY.
    destination_half_extent_xy: tuple[float, float]


class MobileFrankaPhysicalTaskExecutor:
    """Execute and certify one Mobile Franka Physical Task in a live Env."""

    def __init__(
        self,
        *,
        env: Any,
        task: MobileFrankaTask,
        robot_root_path: str,
        step_callback: Callable[[str, int], None] | None = None,
        physics_hz: int = MOBILE_FRANKA_DEFAULT_PHYSICS_HZ,
    ) -> None:
        # Live TARSphere environment advanced by every task phase.
        self.env = env
        # Frozen metric task geometry.
        self.task = task
        # Live robot root containing the mobile base and Panda articulation.
        self.robot_root_path = str(robot_root_path).rstrip("/")
        # Optional evidence callback called after each control update.
        self.step_callback = step_callback
        # Physics rate of the live Env; navigation budgets are derived from it.
        self.physics_hz = _positive_hz(physics_hz)
        self.navigation_control_interval_ticks = _control_interval_ticks(
            physics_hz=self.physics_hz,
            control_hz=MOBILE_FRANKA_NAVIGATION_CONTROL_HZ,
        )
        # Arm IK targets stay per-tick so joint-rate limiting remains smooth.
        self.arm_control_interval_ticks = 1
        self.stationary_control_interval_ticks = self.navigation_control_interval_ticks
        self.placement_settle_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_PLACEMENT_SETTLE_S,
            physics_hz=self.physics_hz,
        )
        self.rotate_settle_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_ROTATE_SETTLE_S,
            physics_hz=self.physics_hz,
        )
        self.arm_settle_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_ARM_SETTLE_S,
            physics_hz=self.physics_hz,
        )
        self.stationary_confirmation_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_STATIONARY_CONFIRMATION_S,
            physics_hz=self.physics_hz,
        )
        self.stationary_timeout_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_STATIONARY_TIMEOUT_S,
            physics_hz=self.physics_hz,
        )
        self.navigation_timeout_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_NAVIGATION_TIMEOUT_S,
            physics_hz=self.physics_hz,
        )
        self.navigation_progress_window_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_NAVIGATION_PROGRESS_WINDOW_S,
            physics_hz=self.physics_hz,
        )
        self.rotation_timeout_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_ROTATION_TIMEOUT_S,
            physics_hz=self.physics_hz,
        )
        self.rotation_progress_window_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_ROTATION_PROGRESS_WINDOW_S,
            physics_hz=self.physics_hz,
        )
        self.arm_timeout_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_ARM_TIMEOUT_S,
            physics_hz=self.physics_hz,
        )
        self.gripper_command_ticks = _ticks_for_seconds(
            MOBILE_FRANKA_GRIPPER_COMMAND_S,
            physics_hz=self.physics_hz,
        )
        # Number of physics ticks consumed by this executor.
        self.physics_steps = 0
        # Live USD stage, resolved once target physics is prepared.
        self.stage: Any = None
        # Initial target centre used to prove non-trivial transport.
        self.initial_object_position: list[float] | None = None
        # Initial target geometry used to reject any rigid-body unit drift.
        self.initial_object_bounds: tuple[list[float], list[float]] | None = None
        # Evidence captured at the FixedJoint transition.
        self.grasp_evidence: dict[str, Any] | None = None
        # Open-finger enclosure captured before physical finger closure.
        self.grasp_reach_evidence: dict[str, Any] | None = None

    @property
    def hand_prim_path(self) -> str:
        """Central Franka hand body used for FixedJoint retention."""
        return f"{self.robot_root_path}/arm_mount/panda/panda_hand"

    @property
    def finger_prim_paths(self) -> tuple[str, str]:
        """The two parallel fingers used by the natural-pose geometry gate."""
        prefix = f"{self.robot_root_path}/arm_mount/panda"
        return (
            f"{prefix}/panda_leftfinger",
            f"{prefix}/panda_rightfinger",
        )

    @property
    def joint_prim_path(self) -> str:
        """Object-local FixedJoint created only during the carry interval."""
        return f"{self.task.object_prim_path}/pick_joint"

    def execute(self) -> dict[str, Any]:
        """Run the complete task and return machine-readable evidence."""
        self._grasp_object_at_source()
        self._navigate(
            self.task.carry_route_xyz,
            phase="carry_to_destination",
        )
        self._navigate(
            (self.task.destination_stance_xyz,),
            phase="approach_destination_stance",
        )
        self._rotate(
            self.task.destination_yaw_rad,
            phase="face_destination",
        )
        self._set_wheel_max_force(MOBILE_FRANKA_TURN_AND_HOLD_WHEEL_MAX_FORCE_NM)
        try:
            self._settle_base(phase="settle_before_place")
            place_orientation = self._gripper_orientation(self.task.destination_yaw_rad)
            self._move_arm(
                self._vertical_approach(self.task.place_position_xyz),
                place_orientation,
                phase="reach_preplace_pose",
            )
            self._move_arm(
                self.task.place_position_xyz,
                None,
                phase="reach_place_pose",
            )
            release_evidence = self._release()
            self._set_gripper("open", phase="open_after_release")
        finally:
            self._set_wheel_max_force(MOBILE_FRANKA_NAVIGATION_WHEEL_MAX_FORCE_NM)
        self._hold(self.placement_settle_ticks, phase="placement_settle")
        placement_evidence = self._placement_evidence()
        if not placement_evidence["passed"]:
            raise RuntimeError(
                "Mobile Franka placement did not settle inside the authored "
                f"destination: {placement_evidence!r}"
            )
        return {
            "passed": True,
            "robot_type": "MobileFrankaRobot",
            "physics_hz": self.physics_hz,
            "task_sequence": [
                "navigate",
                "grasp",
                "carry",
                "place",
            ],
            "physics_steps": self.physics_steps,
            "sim_time_s": round(
                float(self.physics_steps) / float(self.physics_hz),
                3,
            ),
            "control_intervals_ticks": {
                "navigation": self.navigation_control_interval_ticks,
                "arm": self.arm_control_interval_ticks,
                "stationary": self.stationary_control_interval_ticks,
            },
            "turn_and_hold_wheel_max_force_nm": (
                MOBILE_FRANKA_TURN_AND_HOLD_WHEEL_MAX_FORCE_NM
            ),
            "grasp": dict(self.grasp_evidence),
            "release": release_evidence,
            "placement": placement_evidence,
        }

    def certify_source_grasp(self) -> dict[str, Any]:
        """Certify source approach, grasp attachment, and carry preparation."""
        self._grasp_object_at_source()
        return {
            "passed": True,
            "robot_type": "MobileFrankaRobot",
            "physics_hz": self.physics_hz,
            "task_sequence": ["navigate", "grasp", "prepare_carry"],
            "physics_steps": self.physics_steps,
            "sim_time_s": round(
                float(self.physics_steps) / float(self.physics_hz),
                3,
            ),
            "grasp": dict(self.grasp_evidence),
        }

    def _grasp_object_at_source(self) -> None:
        """Reach the source, attach the enclosed object, and retract."""
        self._resolve_target()
        self._set_gripper("open", phase="open_before_grasp")
        self._require_target_geometry_unchanged(
            phase="target_physics_initialization",
        )
        self._prepare_navigation_pose()
        self._navigate(
            self.task.source_route_xyz,
            phase="navigate_to_source",
        )
        self._navigate(
            (self.task.source_stance_xyz,),
            phase="approach_source_stance",
        )
        self._rotate(
            self.task.source_yaw_rad,
            phase="face_source",
        )
        self._set_wheel_max_force(MOBILE_FRANKA_TURN_AND_HOLD_WHEEL_MAX_FORCE_NM)
        try:
            self._settle_base(phase="settle_before_grasp")
            self._move_arm(
                self._vertical_approach(self.task.grasp_position_xyz),
                None,
                phase="reach_pregrasp_pose",
            )
            self._reach_grasp_pose()
            self._set_gripper("close", phase="close_gripper")
            self._require_target_geometry_unchanged(
                phase="grasp_attachment",
            )
            self.grasp_evidence = self._attach_after_pose_gate()
            self._move_arm(
                self._carry_position(self.task.source_yaw_rad),
                self._gripper_orientation(self.task.source_yaw_rad),
                phase="prepare_carry",
            )
        finally:
            self._set_wheel_max_force(MOBILE_FRANKA_NAVIGATION_WHEEL_MAX_FORCE_NM)

    def _prepare_navigation_pose(self) -> None:
        """Lower the arm into the compact pose before wheeled navigation."""
        self._set_wheel_max_force(MOBILE_FRANKA_TURN_AND_HOLD_WHEEL_MAX_FORCE_NM)
        try:
            self._settle_base(phase="settle_before_navigation_pose")
            self._move_arm(
                self._carry_position(self._robot_yaw()),
                self._gripper_orientation(self._robot_yaw()),
                phase="prepare_navigation_pose",
            )
        finally:
            self._set_wheel_max_force(MOBILE_FRANKA_NAVIGATION_WHEEL_MAX_FORCE_NM)

    def _resolve_target(self) -> None:
        """Resolve task prims and initialize one stationary movable target."""
        import omni.usd

        stage = omni.usd.get_context().get_stage()
        if stage is None:
            raise RuntimeError("Mobile Franka executor requires an attached USD stage.")
        for prim_path in (
            self.robot_root_path,
            self.hand_prim_path,
            *self.finger_prim_paths,
            self.task.object_prim_path,
        ):
            prim = stage.GetPrimAtPath(prim_path)
            if not prim or not prim.IsValid():
                raise RuntimeError(f"Mobile Franka task prim is missing: {prim_path}")
        self.stage = stage
        self._set_wheel_max_force(MOBILE_FRANKA_NAVIGATION_WHEEL_MAX_FORCE_NM)
        self._initialize_target_physics(stage)

    def _initialize_target_physics(self, stage: Any) -> None:
        """Create one kinematic rigid target before task motion begins."""
        target = self._target()
        bounds_before = world_bounds(stage, target.prim_path)
        local_transform = _read_object_local_transform(
            stage,
            target.prim_path,
        )
        _enable_pickable_physics(stage, target)
        _restore_pickable_local_transform(
            stage,
            target,
            local_transform,
        )
        _set_pickable_mass(
            stage,
            target,
            MOBILE_FRANKA_PICKABLE_MASS_KG,
        )
        _set_pickable_kinematic(stage, target, enabled=True)
        self.initial_object_bounds = (
            list(bounds_before[0]),
            list(bounds_before[1]),
        )
        self.initial_object_position = self._bounds_center(bounds_before)

    def _require_target_geometry_unchanged(self, *, phase: str) -> None:
        """Reject target drift after PhysX has consumed the authored body."""
        bounds_before = self.initial_object_bounds
        if bounds_before is None:
            raise RuntimeError(
                "Mobile Franka initial target geometry was not recorded."
            )
        bounds_after = world_bounds(
            self._require_stage(),
            self.task.object_prim_path,
        )
        maximum_drift = max(
            abs(float(after) - float(before))
            for before_bound, after_bound in zip(
                bounds_before,
                bounds_after,
                strict=True,
            )
            for before, after in zip(
                before_bound,
                after_bound,
                strict=True,
            )
        )
        if maximum_drift > 1e-3:
            raise RuntimeError(
                "Mobile Franka target geometry changed after physics "
                f"initialization during {phase}: "
                f"maximum_bbox_drift_m={maximum_drift:.6f}, "
                "bounds_before="
                f"{[list(bound) for bound in bounds_before]!r}, "
                f"bounds_after="
                f"{[list(bound) for bound in bounds_after]!r}, "
                "object_local_unit_in_meters="
                f"{self.task.object_local_unit_in_meters:g}."
            )

    def _navigate(
        self,
        route_xyz: Sequence[Sequence[float]],
        *,
        phase: str,
        maximum_steps: int | None = None,
    ) -> None:
        path = [[float(value) for value in point] for point in route_xyz]
        if not path or any(len(point) != 3 for point in path):
            raise ValueError(f"{phase} requires at least one 3-D route point.")
        terminal = np.asarray(path[-1][:2], dtype=np.float64)
        remaining_steps = (
            self.navigation_timeout_ticks
            if maximum_steps is None
            else int(maximum_steps)
        )
        progress_position = self._robot_position()
        progress_yaw = self._robot_yaw()
        progress_ticks = 0
        latest_observation: dict[str, Any] = {}
        while remaining_steps > 0:
            physics_ticks = min(
                self.navigation_control_interval_ticks,
                remaining_steps,
            )
            latest_observation = self._step(
                {"move_along_path": [path]},
                phase=phase,
                physics_ticks=physics_ticks,
            )
            remaining_steps -= physics_ticks
            progress_ticks += physics_ticks
            position = self._robot_position()
            distance = float(np.linalg.norm(position[:2] - terminal))
            finished = bool(
                (latest_observation.get("controllers") or {})
                .get("move_along_path", {})
                .get("finished", False)
            )
            if finished:
                if distance > MOBILE_FRANKA_NAVIGATION_TOLERANCE_M:
                    raise RuntimeError(
                        "Mobile Franka path controller reported completion "
                        f"outside the navigation contract in {phase}: "
                        f"distance_m={distance:.4f}, "
                        "maximum_distance_m="
                        f"{MOBILE_FRANKA_NAVIGATION_TOLERANCE_M:.4f}."
                    )
                self._hold(self.rotate_settle_ticks, phase=f"{phase}_settle")
                return
            if progress_ticks >= self.navigation_progress_window_ticks:
                yaw = self._robot_yaw()
                translation = float(
                    np.linalg.norm(position[:2] - progress_position[:2])
                )
                rotation = abs(
                    (yaw - progress_yaw + math.pi) % (2.0 * math.pi) - math.pi
                )
                if (
                    translation < MOBILE_FRANKA_NAVIGATION_MIN_TRANSLATION_M
                    and rotation < MOBILE_FRANKA_NAVIGATION_MIN_ROTATION_RAD
                ):
                    evidence = self._navigation_motion_evidence(
                        terminal=terminal,
                        observation=latest_observation,
                    )
                    raise RuntimeError(
                        "Mobile Franka navigation made no physical progress "
                        f"in {phase}: translation_m={translation:.4f}, "
                        f"rotation_rad={rotation:.4f}, "
                        "minimum_translation_m="
                        f"{MOBILE_FRANKA_NAVIGATION_MIN_TRANSLATION_M:.4f}, "
                        "minimum_rotation_rad="
                        f"{MOBILE_FRANKA_NAVIGATION_MIN_ROTATION_RAD:.4f}, "
                        f"evidence={evidence!r}."
                    )
                progress_position = position
                progress_yaw = yaw
                progress_ticks = 0
        evidence = self._navigation_motion_evidence(
            terminal=terminal,
            observation=latest_observation,
        )
        raise RuntimeError(
            f"Mobile Franka navigation timed out in {phase}: evidence={evidence!r}."
        )

    def _navigation_motion_evidence(
        self,
        *,
        terminal: np.ndarray,
        observation: dict[str, Any],
    ) -> dict[str, Any]:
        """Expose the commanded and measured state behind a navigation stall."""
        evidence = self._wheel_motion_evidence(
            observation=observation,
            controller_name="move_along_path",
        )
        evidence["goal_xy"] = terminal.tolist()
        return evidence

    def _wheel_motion_evidence(
        self,
        *,
        observation: dict[str, Any],
        controller_name: str,
    ) -> dict[str, Any]:
        """Expose wheel commands and measured chassis motion on failure."""
        robot = self._runtime_robot()
        wheel_indices = [
            robot.articulation.get_dof_index(name)
            for name in ("left_wheel_joint", "right_wheel_joint")
        ]
        wheel_velocities = np.asarray(
            robot.articulation.get_joint_velocities(wheel_indices),
            dtype=np.float64,
        ).reshape(-1)
        linear_speed, angular_speed = self._base_speed()
        return {
            "position_xyz": self._robot_position().tolist(),
            "yaw_rad": self._robot_yaw(),
            "base_linear_speed_m_s": linear_speed,
            "base_angular_speed_rad_s": angular_speed,
            "wheel_velocities_rad_s": wheel_velocities.tolist(),
            "last_articulation_action": robot.get_last_action(),
            "controller": dict(
                (observation.get("controllers") or {}).get(controller_name, {})
            ),
        }

    def _rotate(
        self,
        yaw_rad: float,
        *,
        phase: str,
        maximum_steps: int | None = None,
    ) -> None:
        target_yaw = float(yaw_rad)
        orientation = self._yaw_quaternion(target_yaw)
        self._set_wheel_max_force(MOBILE_FRANKA_TURN_AND_HOLD_WHEEL_MAX_FORCE_NM)
        try:
            self._settle_base(phase=f"{phase}_entry_settle")
            remaining_steps = (
                self.rotation_timeout_ticks
                if maximum_steps is None
                else int(maximum_steps)
            )
            latest_yaw = self._robot_yaw()
            progress_yaw = latest_yaw
            progress_ticks = 0
            latest_observation: dict[str, Any] = {}
            while remaining_steps > 0:
                physics_ticks = min(
                    self.navigation_control_interval_ticks,
                    remaining_steps,
                )
                latest_observation = self._step(
                    {"rotate": [orientation]},
                    phase=phase,
                    physics_ticks=physics_ticks,
                )
                remaining_steps -= physics_ticks
                progress_ticks += physics_ticks
                latest_yaw = self._robot_yaw()
                heading_error = (target_yaw - latest_yaw + math.pi) % (
                    2.0 * math.pi
                ) - math.pi
                controller_finished = bool(
                    (latest_observation.get("controllers") or {})
                    .get("rotate", {})
                    .get("finished", False)
                )
                if (
                    controller_finished
                    or abs(heading_error)
                    <= MOBILE_FRANKA_MANIPULATION_HEADING_TOLERANCE_RAD
                ):
                    self._hold(
                        self.rotate_settle_ticks,
                        phase=f"{phase}_settle",
                    )
                    return
                if progress_ticks >= self.rotation_progress_window_ticks:
                    rotation = abs(
                        (latest_yaw - progress_yaw + math.pi) % (2.0 * math.pi)
                        - math.pi
                    )
                    if rotation < MOBILE_FRANKA_ROTATION_MIN_PROGRESS_RAD:
                        evidence = self._wheel_motion_evidence(
                            observation=latest_observation,
                            controller_name="rotate",
                        )
                        raise RuntimeError(
                            "Mobile Franka rotation made no physical progress "
                            f"in {phase}: rotation_rad={rotation:.4f}, "
                            "minimum_rotation_rad="
                            f"{MOBILE_FRANKA_ROTATION_MIN_PROGRESS_RAD:.4f}, "
                            f"evidence={evidence!r}."
                        )
                    progress_yaw = latest_yaw
                    progress_ticks = 0
            heading_error = (target_yaw - latest_yaw + math.pi) % (
                2.0 * math.pi
            ) - math.pi
            evidence = self._wheel_motion_evidence(
                observation=latest_observation,
                controller_name="rotate",
            )
            raise RuntimeError(
                f"Mobile Franka rotation timed out in {phase}: "
                f"current_yaw_rad={latest_yaw:.4f}, "
                f"target_yaw_rad={target_yaw:.4f}, "
                f"heading_error_rad={heading_error:.4f}, "
                f"evidence={evidence!r}."
            )
        finally:
            self._set_wheel_max_force(MOBILE_FRANKA_NAVIGATION_WHEEL_MAX_FORCE_NM)

    def _move_arm(
        self,
        target_position_xyz: Sequence[float],
        target_orientation_wxyz: Sequence[float] | None,
        *,
        phase: str,
        maximum_steps: int | None = None,
    ) -> None:
        position = [float(value) for value in target_position_xyz]
        orientation = (
            None
            if target_orientation_wxyz is None
            else [float(value) for value in target_orientation_wxyz]
        )
        if len(position) != 3 or (orientation is not None and len(orientation) != 4):
            raise ValueError(
                f"{phase} requires one 3-D position and, when provided, one quaternion."
            )
        latest: dict[str, Any] = {}
        remaining_steps = (
            self.arm_timeout_ticks if maximum_steps is None else int(maximum_steps)
        )
        while remaining_steps > 0:
            physics_ticks = min(
                self.arm_control_interval_ticks,
                remaining_steps,
            )
            observation = self._step(
                {
                    "move_by_speed": [0.0, 0.0],
                    "arm_ik_controller": [position, orientation],
                },
                phase=phase,
                physics_ticks=physics_ticks,
            )
            remaining_steps -= physics_ticks
            latest = (observation.get("controllers") or {}).get("arm_ik_controller", {})
            if bool(latest.get("finished", False)):
                self._hold(self.arm_settle_ticks, phase=f"{phase}_settle")
                return
        raise RuntimeError(
            f"Mobile Franka IK timed out in {phase}: "
            f"target={position!r}, "
            f"robot_position={self._robot_position().tolist()!r}, "
            f"controller={latest!r}."
        )

    def _reach_grasp_pose(
        self,
        *,
        maximum_steps: int | None = None,
    ) -> None:
        """Use position IK until the controller reports the grasp pose.

        Finger/object AABB gaps are diagnostic snapshots, not the completion
        gate. Orientation stays unconstrained so the arm can reach across
        source surfaces; tabletop placement uses its own downward orientation.
        """
        position = [float(value) for value in self.task.grasp_position_xyz]
        latest_controller: dict[str, Any] = {}
        remaining_steps = (
            self.arm_timeout_ticks if maximum_steps is None else int(maximum_steps)
        )
        while remaining_steps > 0:
            physics_ticks = min(
                self.arm_control_interval_ticks,
                remaining_steps,
            )
            observation = self._step(
                {
                    "move_by_speed": [0.0, 0.0],
                    "arm_ik_controller": [position, None],
                },
                phase="reach_grasp_pose",
                physics_ticks=physics_ticks,
            )
            remaining_steps -= physics_ticks
            latest_controller = (observation.get("controllers") or {}).get(
                "arm_ik_controller", {}
            )
            if bool(latest_controller.get("finished", False)):
                self.grasp_reach_evidence = dict(self._grasp_geometry())
                self._hold(self.arm_settle_ticks, phase="reach_grasp_pose_settle")
                return
        raise RuntimeError(
            "Mobile Franka grasp reach timed out before IK finished: "
            f"target={position!r}, "
            f"controller={latest_controller!r}, "
            f"geometry={self._grasp_geometry()!r}."
        )

    def _set_gripper(
        self,
        command: str,
        *,
        phase: str,
        steps: int | None = None,
    ) -> None:
        if command not in {"open", "close"}:
            raise ValueError(
                f"Mobile Franka gripper command must be open or close: {command!r}"
            )
        self._repeat_action(
            {
                "move_by_speed": [0.0, 0.0],
                "gripper_controller": [command],
            },
            phase=phase,
            physics_ticks=(self.gripper_command_ticks if steps is None else int(steps)),
            control_interval_ticks=(self.stationary_control_interval_ticks),
        )

    def _attach_after_pose_gate(self) -> dict[str, Any]:
        stage = self._require_stage()
        reach_geometry, closed_geometry = self._require_grasp_geometry_for_attachment()

        from tarsphere.simulator.core.util.physics import deactivate_collider
        from omni.isaac.core.utils.prims import get_prim_at_path

        target = self._target()
        deactivate_collider(get_prim_at_path(target.prim_path))
        _set_pickable_kinematic(stage, target, enabled=False)
        _create_attachment_joint(
            stage,
            prim_path=self.joint_prim_path,
            body0=target.prim_path,
            body1=self.hand_prim_path,
        )
        joint = stage.GetPrimAtPath(self.joint_prim_path)
        if not joint or not joint.IsValid():
            raise RuntimeError("Mobile Franka FixedJoint was not created.")
        return {
            "object_id": self.task.object_id,
            "joint_path": self.joint_prim_path,
            "hand_prim_path": self.hand_prim_path,
            "open_grasp_center_to_object_m": reach_geometry["grasp_center_to_object_m"],
            "open_object_bbox_center_offset_m": reach_geometry[
                "object_bbox_center_offset_m"
            ],
            "closed_grasp_center_to_object_m": closed_geometry[
                "grasp_center_to_object_m"
            ],
            "closed_object_bbox_center_offset_m": closed_geometry[
                "object_bbox_center_offset_m"
            ],
            "closed_finger_surface_gaps_m": closed_geometry["finger_surface_gaps_m"],
            "object_position_xyz": closed_geometry["object_center_xyz"],
        }

    def _require_grasp_geometry_for_attachment(
        self,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Require a completed grasp reach, then snapshot closed-finger geometry."""
        reach_geometry = self.grasp_reach_evidence
        if reach_geometry is None:
            raise RuntimeError(
                "Mobile Franka cannot attach without a completed grasp reach."
            )
        return dict(reach_geometry), self._grasp_geometry()

    def _grasp_geometry(self) -> dict[str, Any]:
        """Measure target enclosure by the two live Franka fingers."""
        stage = self._require_stage()
        target_bounds = world_bounds(stage, self.task.object_prim_path)
        finger_bounds = [world_bounds(stage, path) for path in self.finger_prim_paths]
        object_center = self._bounds_center(target_bounds)
        finger_centres = [self._bounds_center(bounds) for bounds in finger_bounds]
        grasp_center = [
            (finger_centres[0][axis] + finger_centres[1][axis]) / 2.0
            for axis in range(3)
        ]
        grasp_center_to_object = self._point_to_bounds_distance(
            grasp_center,
            target_bounds,
        )
        surface_gaps = [
            bbox_surface_gap(bounds, target_bounds) for bounds in finger_bounds
        ]
        return {
            "object_center_xyz": object_center,
            "finger_center_xyz": grasp_center,
            "grasp_center_to_object_m": grasp_center_to_object,
            "object_bbox_center_offset_m": math.dist(
                object_center,
                grasp_center,
            ),
            "finger_surface_gaps_m": surface_gaps,
        }

    def _release(self) -> dict[str, Any]:
        stage = self._require_stage()
        joint = stage.GetPrimAtPath(self.joint_prim_path)
        if not joint or not joint.IsValid():
            raise RuntimeError(
                "Mobile Franka cannot release without a live FixedJoint."
            )
        target = self._target()
        release_position, release_orientation = _world_pose(
            stage,
            target.prim_path,
        )
        _set_pickable_kinematic(stage, target, enabled=True)
        stage.RemovePrim(self.joint_prim_path)
        _set_pickable_metric_world_pose(
            stage,
            target,
            position_world=release_position,
            orientation_wxyz=release_orientation,
            object_local_unit_in_meters=(self.task.object_local_unit_in_meters),
        )
        _activate_pickable_collider(target)
        _set_pickable_kinematic(stage, target, enabled=False)
        return {
            "object_id": self.task.object_id,
            "joint_removed": True,
            "released_position_xyz": release_position,
        }

    def _placement_evidence(self) -> dict[str, Any]:
        bounds = world_bounds(
            self._require_stage(),
            self.task.object_prim_path,
        )
        center = self._bounds_center(bounds)
        surface = self.task.destination_surface_xyz
        half_x, half_y = self.task.destination_half_extent_xy
        xy_inside = bool(
            abs(center[0] - surface[0]) <= half_x + MOBILE_FRANKA_PLACEMENT_TOLERANCE_M
            and abs(center[1] - surface[1])
            <= half_y + MOBILE_FRANKA_PLACEMENT_TOLERANCE_M
        )
        height_error = abs(float(bounds[0][2]) - float(surface[2]))
        initial = self.initial_object_position
        if initial is None:
            raise RuntimeError("Initial object position was not recorded.")
        transport_distance = math.dist(initial, center)
        return {
            "object_center_xyz": center,
            "object_bounds_xyz": [list(bounds[0]), list(bounds[1])],
            "destination_surface_xyz": list(surface),
            "xy_inside_destination": xy_inside,
            "height_error_m": height_error,
            "placement_tolerance_m": MOBILE_FRANKA_PLACEMENT_TOLERANCE_M,
            "transport_distance_m": transport_distance,
            "joint_present_after_release": bool(
                self._require_stage().GetPrimAtPath(self.joint_prim_path).IsValid()
            ),
            "passed": bool(
                xy_inside
                and height_error <= MOBILE_FRANKA_PLACEMENT_TOLERANCE_M
                and transport_distance >= 0.30
            ),
        }

    def _hold(self, steps: int, *, phase: str) -> None:
        self._repeat_action(
            {"move_by_speed": [0.0, 0.0]},
            phase=phase,
            physics_ticks=int(steps),
            control_interval_ticks=(self.stationary_control_interval_ticks),
        )

    def _set_wheel_max_force(self, maximum_force_nm: float) -> None:
        set_mobile_franka_wheel_max_force(
            self._require_stage(),
            robot_root_path=self.robot_root_path,
            maximum_force_nm=maximum_force_nm,
        )

    def _settle_base(self, *, phase: str) -> None:
        """Require measured chassis rest before rotation or manipulation."""
        stable_ticks = 0
        remaining_ticks = self.stationary_timeout_ticks
        latest_motion: tuple[float, float] | None = None
        while remaining_ticks > 0:
            physics_ticks = min(
                self.stationary_control_interval_ticks,
                remaining_ticks,
            )
            self._step(
                {"move_by_speed": [0.0, 0.0]},
                phase=phase,
                physics_ticks=physics_ticks,
            )
            remaining_ticks -= physics_ticks
            linear_speed, angular_speed = self._base_speed()
            latest_motion = (linear_speed, angular_speed)
            if (
                linear_speed <= MOBILE_FRANKA_STATIONARY_LINEAR_SPEED_M_S
                and angular_speed <= MOBILE_FRANKA_STATIONARY_ANGULAR_SPEED_RAD_S
            ):
                stable_ticks += physics_ticks
                if stable_ticks >= self.stationary_confirmation_ticks:
                    return
            else:
                stable_ticks = 0
        if latest_motion is None:
            raise RuntimeError("Mobile Franka stationary gate did not run.")
        raise RuntimeError(
            f"Mobile Franka did not become stationary in {phase}: "
            f"latest_linear_speed_m_s={latest_motion[0]:.4f}, "
            f"latest_angular_speed_rad_s={latest_motion[1]:.4f}."
        )

    def _base_speed(self) -> tuple[float, float]:
        base = self._runtime_robot().get_robot_base()
        linear_velocity = np.asarray(
            base.get_linear_velocity(),
            dtype=np.float64,
        ).reshape(-1)
        angular_velocity = np.asarray(
            base.get_angular_velocity(),
            dtype=np.float64,
        ).reshape(-1)
        if (
            linear_velocity.size != 3
            or angular_velocity.size != 3
            or not np.all(np.isfinite(linear_velocity))
            or not np.all(np.isfinite(angular_velocity))
        ):
            raise RuntimeError("Mobile Franka base returned invalid physical velocity.")
        return (
            float(np.linalg.norm(linear_velocity)),
            float(np.linalg.norm(angular_velocity)),
        )

    def _repeat_action(
        self,
        action: dict[str, Any],
        *,
        phase: str,
        physics_ticks: int,
        control_interval_ticks: int,
    ) -> None:
        """Hold one controller command for an exact physics-tick budget."""
        remaining = int(physics_ticks)
        if remaining < 0:
            raise ValueError("physics_ticks must be non-negative.")
        interval = int(control_interval_ticks)
        if interval < 1:
            raise ValueError("control_interval_ticks must be a positive integer.")
        while remaining > 0:
            current_window = min(
                interval,
                remaining,
            )
            self._step(
                action,
                phase=phase,
                physics_ticks=current_window,
            )
            remaining -= current_window

    def _step(
        self,
        action: dict[str, Any],
        *,
        phase: str,
        physics_ticks: int | None = None,
    ) -> dict[str, Any]:
        window = 1 if physics_ticks is None else int(physics_ticks)
        if window < 1:
            raise ValueError("physics_ticks must be a positive integer.")
        observation, _, terminated, truncated, _ = self.env.step(action=action)
        self.physics_steps += 1
        if self.step_callback is not None:
            self.step_callback(phase, self.physics_steps)
        if terminated or truncated:
            raise RuntimeError(f"Mobile Franka environment terminated during {phase}.")
        if window > 1:
            self.env.warm_up(
                steps=window - 1,
                render=False,
                physics=True,
            )
            self.physics_steps += window - 1
        return dict(observation or {})

    def _runtime_robot(self) -> Any:
        tasks = tuple(self.env.runner.current_tasks.values())
        if len(tasks) != 1:
            raise RuntimeError(f"Expected one Mobile Franka task, found {len(tasks)}.")
        robots = tuple(tasks[0].robots.values())
        if len(robots) != 1:
            raise RuntimeError(
                f"Expected one Mobile Franka robot, found {len(robots)}."
            )
        return robots[0]

    def _live_base_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Return one finite, normalized pose from the runtime rigid body."""
        position, orientation = self._runtime_robot().get_pose()
        position = np.asarray(position, dtype=np.float64).reshape(-1)
        orientation = np.asarray(
            orientation,
            dtype=np.float64,
        ).reshape(-1)
        if (
            position.size != 3
            or orientation.size != 4
            or not np.all(np.isfinite(position))
            or not np.all(np.isfinite(orientation))
        ):
            raise RuntimeError(
                "Mobile Franka base returned an invalid live pose: "
                f"position={position.tolist()!r}, "
                f"orientation={orientation.tolist()!r}."
            )
        orientation_norm = float(np.linalg.norm(orientation))
        if orientation_norm <= 1e-9:
            raise RuntimeError("Mobile Franka base returned a zero-norm orientation.")
        return position, orientation / orientation_norm

    def _robot_position(self) -> np.ndarray:
        return self._live_base_pose()[0]

    def _robot_yaw(self) -> float:
        orientation = self._live_base_pose()[1]
        w, x, y, z = orientation
        return float(
            math.atan2(
                2.0 * (w * z + x * y),
                1.0 - 2.0 * (y * y + z * z),
            )
        )

    def _object_center(self) -> list[float]:
        return self._bounds_center(
            world_bounds(
                self._require_stage(),
                self.task.object_prim_path,
            )
        )

    def _carry_position(self, yaw_rad: float) -> tuple[float, float, float]:
        """Return the embodiment-level compact pose used during navigation."""
        base = self._robot_position()
        return (
            float(
                base[0]
                + MOBILE_FRANKA_CARRY_FORWARD_OFFSET_M * math.cos(float(yaw_rad))
            ),
            float(
                base[1]
                + MOBILE_FRANKA_CARRY_FORWARD_OFFSET_M * math.sin(float(yaw_rad))
            ),
            MOBILE_FRANKA_CARRY_HEIGHT_M,
        )

    @staticmethod
    def _vertical_approach(
        target_position_xyz: Sequence[float],
    ) -> tuple[float, float, float]:
        """Return a collision-clear pose directly above a tabletop target."""
        if len(target_position_xyz) != 3:
            raise ValueError("Mobile Franka approach target must contain three values.")
        return (
            float(target_position_xyz[0]),
            float(target_position_xyz[1]),
            float(target_position_xyz[2]) + MOBILE_FRANKA_VERTICAL_APPROACH_M,
        )

    def _require_stage(self) -> Any:
        if self.stage is None:
            raise RuntimeError("Mobile Franka target physics has not been prepared.")
        return self.stage

    def _target(self) -> G1ManipulationObject:
        return G1ManipulationObject(
            object_id=self.task.object_id,
            prim_path=self.task.object_prim_path,
            pickable=True,
        )

    @staticmethod
    def _bounds_center(
        bounds: tuple[list[float], list[float]],
    ) -> list[float]:
        return [
            (float(bounds[0][axis]) + float(bounds[1][axis])) / 2.0 for axis in range(3)
        ]

    @staticmethod
    def _point_to_bounds_distance(
        point: Sequence[float],
        bounds: tuple[list[float], list[float]],
    ) -> float:
        """Return zero exactly when a point lies inside one world AABB."""
        if len(point) != 3:
            raise ValueError("Bounds distance requires one 3-D point.")
        lower, upper = bounds
        squared_distance = 0.0
        for axis in range(3):
            coordinate = float(point[axis])
            offset = max(
                float(lower[axis]) - coordinate,
                0.0,
                coordinate - float(upper[axis]),
            )
            squared_distance += offset * offset
        return math.sqrt(squared_distance)

    @staticmethod
    def _yaw_quaternion(yaw_rad: float) -> list[float]:
        return [
            math.cos(float(yaw_rad) / 2.0),
            0.0,
            0.0,
            math.sin(float(yaw_rad) / 2.0),
        ]

    @classmethod
    def _gripper_orientation(cls, yaw_rad: float) -> list[float]:
        # Isaac Sim's Franka pick-place controller defines the standard
        # downward tabletop pose as Euler [0, pi, 0]. Rotate this robot-local
        # pose by the mobile base heading at each manipulation stance.
        yaw = cls._yaw_quaternion(yaw_rad)
        downward = [0.0, 0.0, 1.0, 0.0]
        return cls._quaternion_product(yaw, downward)

    @staticmethod
    def _quaternion_product(
        left: Sequence[float],
        right: Sequence[float],
    ) -> list[float]:
        lw, lx, ly, lz = (float(value) for value in left)
        rw, rx, ry, rz = (float(value) for value in right)
        return [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ]


def set_mobile_franka_wheel_max_force(
    stage: Any,
    *,
    robot_root_path: str,
    maximum_force_nm: float,
) -> None:
    """Set the explicit effort limit shared by both wheel drives."""
    from pxr import UsdPhysics

    maximum_force = float(maximum_force_nm)
    if not math.isfinite(maximum_force) or maximum_force <= 0.0:
        raise ValueError(
            "Mobile Franka wheel maximum force must be finite and positive."
        )
    root = str(robot_root_path).rstrip("/")
    for joint_name in ("left_wheel_joint", "right_wheel_joint"):
        joint_path = f"{root}/{joint_name}"
        prim = stage.GetPrimAtPath(joint_path)
        if not prim or not prim.IsValid():
            raise RuntimeError(f"Mobile Franka wheel joint is missing: {joint_path}")
        drive = UsdPhysics.DriveAPI.Get(prim, "angular")
        if not drive:
            raise RuntimeError(f"Mobile Franka wheel drive is missing: {joint_path}")
        drive.GetMaxForceAttr().Set(float(maximum_force))


__all__ = [
    "MobileFrankaPhysicalTaskExecutor",
    "MobileFrankaTask",
    "set_mobile_franka_wheel_max_force",
]
