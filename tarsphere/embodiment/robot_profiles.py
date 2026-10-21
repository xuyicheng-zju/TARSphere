"""TARSphere robot profile registry.

Each profile binds a robot model name (as written in episode YAML's
``robot.model``) to its TARSphere config module, standing pelvis
height, USD asset, and camera wiring.  ``IsaacAutonomousEnv`` looks the
profile up by name and uses it to build the TARSphere ``RobotCfg`` — no
hardcoded ``H1RobotCfg`` import.

Adding a new robot:
    1. Ensure the TARSphere config module exists (e.g.
       ``tarsphere.simulator.extensions.configs.robots.foo`` exposing ``FooRobotCfg``
       + ``foo_camera_cfg`` + ``move_by_speed_cfg`` + ``rotate_cfg``).
    2. Add a :class:`RobotProfile` entry to :data:`ROBOT_PROFILES` below.

Standing height note: Isaac's robot root link is the pelvis, not the foot.
``standing_z`` must match the pelvis height of the robot in its default
standing pose, otherwise the robot spawns inside the floor and jams.
Typical values: G1 ~0.8 m, H1 ~1.05 m.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import numpy as np


@dataclass(frozen=True)
class RobotProfile:
    """One robot's binding to TARSphere config + geometry."""

    # Canonical model name authored in episode YAML ``robot.model``.
    name: str
    # Dotted import path of the TARSphere config module that exposes
    # ``{RobotCls}``, ``{prefix}_camera_cfg``, ``move_by_speed_cfg``,
    # ``rotate_cfg``.  Imported lazily by isaac_env (which only runs under
    # the Isaac conda env), so a missing module does not break unit tests.
    config_module: str
    # RobotCfg class name inside config_module.
    robot_cfg_class: str
    # Camera cfg attribute name inside config_module.
    camera_cfg_attr: str
    # Standing pelvis height (metres). start_position.z should match this.
    standing_z: float
    # USD asset path relative to the flat TARSphere assets root.
    usd_rel_path: str
    # Optional camera prim path override relative to the robot prim.  This keeps
    # TARSPHERE-owned robot wiring independent from the base simulator defaults.
    camera_prim_path: str = ""
    # Fixed camera mount pitch applied once to the camera prim after the robot
    # is loaded. Negative values tilt a USD camera downward around its local
    # right axis. This is asset calibration, not an agent action.
    camera_mount_pitch_deg: float = 0.0
    # Optional env var for local calibration experiments on one robot profile.
    camera_mount_pitch_env: str = ""
    # Human-readable description for docs / errors.
    description: str = ""
    # Controller cfg attributes to load from config_module, in env creation order.
    controller_cfg_attrs: Tuple[str, ...] = ("move_by_speed_cfg", "rotate_cfg")


ROBOT_PROFILES: Dict[str, RobotProfile] = {
    "G1Robot": RobotProfile(
        name="G1Robot",
        # The stable 29-DOF G1 remains the locomotion source model. Runtime
        # composition mounts articulated Dex3 hands without replacing it.
        config_module="tarsphere.embodiment.robots.g1",
        robot_cfg_class="G1RobotCfg",
        camera_cfg_attr="g1_camera_cfg",
        camera_prim_path="d435_link/Camera",
        camera_mount_pitch_deg=-15.0,
        camera_mount_pitch_env="TARSPHERE_G1_CAMERA_MOUNT_PITCH_DEG",
        standing_z=0.8,
        usd_rel_path="robots/g1/g1_29dof_color.usd",
        description=(
            "Unitree G1 29-DOF with two 7-DOF Dex3-1 hands. Mounted-hand "
            "gravity is neutralized for locomotion; benchmark grasp retention "
            "requires a converged full-pose reach, opposing finger contact, "
            "and measured closure before FixedJoint transport."
        ),
        # MoveAlongPathPointsController is defined in TARSphere's G1 config
        # (move_along_path_cfg, wrapping move_to_point -> move_by_speed). List
        # it here so the oracle route scripts' follow_path action resolves to a
        # registered controller instead of being dropped with
        # "unknown controller move_along_path". The evaluated agent never emits
        # follow_path — it walks with locomotion primitives — but certification
        # needs the path controller to prove a route is physically traversable.
        controller_cfg_attrs=(
            "move_by_speed_cfg",
            "move_to_point_cfg",
            "move_along_path_cfg",
            "rotate_cfg",
            "left_hand_joint_controller_cfg",
            "right_hand_joint_controller_cfg",
            "right_arm_joint_controller_cfg",
            "waist_pitch_joint_controller_cfg",
        ),
    ),
    "H1Robot": RobotProfile(
        name="H1Robot",
        config_module="tarsphere.simulator.extensions.configs.robots.h1",
        robot_cfg_class="H1RobotCfg",
        camera_cfg_attr="h1_camera_cfg",
        camera_prim_path="logo_link/Camera",
        standing_z=1.05,
        usd_rel_path="robots/h1/h1.usd",
        description="Unitree H1 humanoid.",
    ),
    "H1WithHandRobot": RobotProfile(
        name="H1WithHandRobot",
        config_module="tarsphere.embodiment.robots.h1_with_hand",
        robot_cfg_class="H1WithHandRobotCfg",
        camera_cfg_attr="h1_with_hand_camera_cfg",
        camera_prim_path="logo_link/Camera",
        camera_mount_pitch_deg=-18.0,
        camera_mount_pitch_env="TARSPHERE_H1_WITH_HAND_CAMERA_MOUNT_PITCH_DEG",
        standing_z=1.05,
        usd_rel_path="robots/h1_with_hand/h1_with_hand_rt_v2.usd",
        description="Unitree H1 humanoid with right-arm IK and hand joints.",
        controller_cfg_attrs=(
            "move_by_speed_cfg",
            "rotate_cfg",
            "right_arm_ik_controller_cfg",
            "right_arm_joint_controller_cfg",
            "right_hand_joint_controller_cfg",
        ),
    ),
    "GR1Robot": RobotProfile(
        name="GR1Robot",
        config_module="tarsphere.simulator.extensions.configs.robots.gr1",
        robot_cfg_class="GR1RobotCfg",
        camera_cfg_attr="camera_left_cfg",
        standing_z=0.75,
        usd_rel_path="robots/gr1/gr1.usd",
        description="Fourier GR1 humanoid.",
    ),
    "MobileFrankaRobot": RobotProfile(
        name="MobileFrankaRobot",
        config_module="tarsphere.simulator.extensions.configs.robots.mobile_franka",
        robot_cfg_class="MobileFrankaRobotCfg",
        camera_cfg_attr="mobile_franka_camera_cfg",
        camera_prim_path="base_link/Camera",
        # The USD already authors base_link 0.4195 m above its root. Episode
        # positions locate that root on the floor and must not add the offset
        # again.
        standing_z=0.0,
        usd_rel_path="robots/mobile_franka/mobile_franka.usda",
        description=(
            "Differential-drive pedestal + Franka Panda (in-repo composition). "
            "Optional cross-embodiment executor with metric arm IK and physical "
            "gripper control."
        ),
        controller_cfg_attrs=(
            "move_by_speed_cfg",
            "move_to_point_cfg",
            "move_along_path_cfg",
            "rotate_cfg",
            "arm_ik_cfg",
            "gripper_cfg",
        ),
    ),
}

# Default robot for new episodes / when episode YAML omits robot_type.
DEFAULT_ROBOT_TYPE = "G1Robot"


def get_robot_profile(robot_type: str) -> RobotProfile:
    """Look up a robot profile by its canonical name.  Fail loud if unknown."""
    profile = ROBOT_PROFILES.get(robot_type)
    if profile is None:
        raise KeyError(
            f"Unknown robot_type {robot_type!r}. Known robots: "
            f"{sorted(ROBOT_PROFILES)}. Add a RobotProfile entry in "
            f"tarsphere.embodiment.robot_profiles to support a new robot."
        )
    return profile


def load_simulator_robot_cfg(robot_type: str) -> Tuple[Any, Any, Any, Any]:
    """Import the TARSphere config module for ``robot_type`` and return
    ``(RobotCfgClass, camera_cfg, move_by_speed_cfg, rotate_cfg)``.

    Importing TARSphere requires the Isaac conda env (``pxr`` etc.), so
    this is intentionally a lazy call site used only by ``isaac_env``.
    """
    profile = get_robot_profile(robot_type)
    import importlib

    mod = importlib.import_module(profile.config_module)
    robot_cfg_cls = getattr(mod, profile.robot_cfg_class)
    camera_cfg = getattr(mod, profile.camera_cfg_attr)
    if profile.camera_prim_path:
        camera_cfg = camera_cfg.update(prim_path=profile.camera_prim_path)
    move_by_speed_cfg = getattr(mod, "move_by_speed_cfg")
    rotate_cfg = getattr(mod, "rotate_cfg")
    return robot_cfg_cls, camera_cfg, move_by_speed_cfg, rotate_cfg


def load_simulator_robot_components(robot_type: str) -> Tuple[Any, Any, list[Any]]:
    """Load ``(RobotCfgClass, camera_cfg, controller_cfgs)`` for a robot type.

    This is the preferred runtime helper.  Profiles can expose additional
    controllers, e.g. H1WithHand's right-arm IK and right-hand joint controller,
    without adding robot-specific branches in IsaacAutonomousEnv.
    """
    profile = get_robot_profile(robot_type)
    import importlib

    mod = importlib.import_module(profile.config_module)
    robot_cfg_cls = getattr(mod, profile.robot_cfg_class)
    camera_cfg = getattr(mod, profile.camera_cfg_attr)
    if profile.camera_prim_path:
        camera_cfg = camera_cfg.update(prim_path=profile.camera_prim_path)
    controller_cfgs: list[Any] = []
    missing: list[str] = []
    for attr in profile.controller_cfg_attrs:
        if not hasattr(mod, attr):
            missing.append(attr)
            continue
        controller_cfgs.append(getattr(mod, attr))
    if missing:
        raise AttributeError(
            f"Robot profile {robot_type!r} references missing controller cfg "
            f"attribute(s) {missing} in module {profile.config_module!r}."
        )
    return robot_cfg_cls, camera_cfg, controller_cfgs


def patch_simulator_robot_observation_layout(robot_type: str) -> None:
    """Normalize robot observation layout for TARSPHERE.

    The G1 extension creates sensors correctly, but its
    ``get_obs()`` writes sensor data at top-level ``obs['camera']`` while H1/GR1
    write the common ``obs['sensors']['camera']`` shape.  TARSphere expects one robot
    observation contract, so patch G1 at import time instead of making every
    camera consumer support two incompatible layouts.
    """
    if robot_type != "G1Robot":
        return

    try:
        from tarsphere.simulator.extensions.robots.g1 import G1Robot
        from tarsphere.embodiment.robots.g1 import set_g1_dex3_gains
    except Exception as exc:
        raise RuntimeError(f"Cannot patch G1 observation layout: {exc}") from exc

    if getattr(G1Robot, "_tarsphere_runtime_contract_patch", False):
        return

    def iter_controllers(self):  # type: ignore[no-untyped-def]
        seen: set[int] = set()

        def visit(controller):
            if id(controller) in seen:
                return
            seen.add(id(controller))
            yield controller
            for child in getattr(controller, "sub_controllers", None) or []:
                yield from visit(child)

        for controller in self.controllers.values():
            yield from visit(controller)

    def clone_state_value(self, value):  # type: ignore[no-untyped-def]
        if isinstance(value, dict):
            return {key: clone_state_value(self, item) for key, item in value.items()}
        if isinstance(value, list):
            return [clone_state_value(self, item) for item in value]
        if isinstance(value, tuple):
            return tuple(clone_state_value(self, item) for item in value)
        clone = getattr(value, "clone", None)
        if callable(clone):
            return clone()
        if isinstance(value, np.ndarray):
            return np.array(value, copy=True)
        return copy.deepcopy(value)

    def share_locomotion_policy_controller(self):  # type: ignore[no-untyped-def]
        policy = self.controllers.get("move_by_speed")
        move_to_point = self.controllers.get("move_to_point")
        path = self.controllers.get("move_along_path")
        rotate = self.controllers.get("rotate")
        if policy is None or move_to_point is None or path is None or rotate is None:
            raise RuntimeError(
                "G1 locomotion requires move_by_speed, move_to_point, "
                "move_along_path, and rotate controllers."
            )
        move_to_point.sub_controllers = [policy]
        path.sub_controllers = [move_to_point]
        rotate.sub_controllers = [policy]

    def save_episode_reset_state(self):  # type: ignore[no-untyped-def]
        self.articulation.save_status()
        self._episode_articulation_reset_state = clone_state_value(
            self, self.articulation.status
        )
        snapshots = {}
        for controller in iter_controllers(self):
            state = {}
            old_policy_obs = getattr(controller, "_old_policy_obs", None)
            if old_policy_obs is not None:
                state["old_policy_obs"] = old_policy_obs.clone()
            last_targets = getattr(controller, "_last_target_joint_positions", None)
            if last_targets is not None:
                state["last_target_joint_positions"] = np.array(last_targets, copy=True)
            applied = getattr(controller, "applied_joint_positions", None)
            if applied is not None:
                state["applied_joint_positions"] = np.array(applied, copy=True)
            if hasattr(controller, "_apply_times_left"):
                state["apply_times_left"] = int(controller._apply_times_left)
            snapshots[id(controller)] = state
        self._episode_controller_reset_state = snapshots
        self._episode_reset_state_saved = True

    def restore_episode_controller_state(self):  # type: ignore[no-untyped-def]
        for controller in iter_controllers(self):
            state = self._episode_controller_reset_state.get(id(controller))
            if state is None:
                raise RuntimeError(
                    f"G1 controller {type(controller).__name__} has no saved reset state."
                )
            old_policy_obs = getattr(controller, "_old_policy_obs", None)
            if old_policy_obs is not None:
                saved = state.get("old_policy_obs")
                if saved is None:
                    raise RuntimeError("Saved G1 policy history is missing.")
                controller._old_policy_obs = saved.clone()
            last_targets = getattr(controller, "_last_target_joint_positions", None)
            if last_targets is not None:
                saved = state.get("last_target_joint_positions")
                if saved is None:
                    raise RuntimeError("Saved G1 target joint state is missing.")
                controller._last_target_joint_positions = np.array(saved, copy=True)
            if hasattr(controller, "_apply_times_left"):
                controller._apply_times_left = int(state["apply_times_left"])
            if "applied_joint_positions" in state:
                controller.applied_joint_positions = np.array(
                    state["applied_joint_positions"], copy=True
                )
            if hasattr(controller, "path_points"):
                controller.path_points = []
                controller.path_point_idx = 0
                controller.current_path_point = None
            if hasattr(controller, "goal_position"):
                controller.goal_position = None
            if hasattr(controller, "last_threshold"):
                controller.last_threshold = None
            if hasattr(controller, "goal_orientation"):
                controller.goal_orientation = None
            controller._obs = {}

    def restore_episode_reset_state(self):  # type: ignore[no-untyped-def]
        if not getattr(self, "_episode_reset_state_saved", False):
            raise RuntimeError("G1 episode reset state has not been saved.")
        if not self._episode_articulation_reset_state:
            raise RuntimeError("G1 articulation reset snapshot is empty.")
        self.articulation.status = clone_state_value(
            self, self._episode_articulation_reset_state
        )
        self.articulation.restore_status()
        restore_episode_controller_state(self)
        self.set_gains()

    def get_obs(self):  # type: ignore[no-untyped-def]
        position, orientation = self._robot_base.get_pose()
        obs = {
            "position": position,
            "orientation": orientation,
            "joint_positions": self.articulation.get_joint_positions(),
            "joint_velocities": self.articulation.get_joint_velocities(),
            "controllers": {},
            "sensors": {},
        }
        for controller_name, controller in self.controllers.items():
            obs["controllers"][controller_name] = controller.get_obs()
        for sensor_name, sensor in self.sensors.items():
            obs["sensors"][sensor_name] = sensor.get_data()
        return self._make_ordered(obs)

    def post_reset(self):  # type: ignore[no-untyped-def]
        """Bind the stable G1 links after TARSphere reset."""
        super(G1Robot, self).post_reset()
        root = self.config.prim_path
        self._robot_base = self._rigid_body_map[f"{root}/pelvis"]
        self._imu_in_torso = self._rigid_body_map[f"{root}/imu_link"]
        self.set_gains()
        share_locomotion_policy_controller(self)

    G1Robot.post_reset = post_reset
    G1Robot.get_obs = get_obs
    G1Robot.set_gains = set_g1_dex3_gains
    G1Robot._iter_controllers = iter_controllers
    G1Robot._clone_state_value = clone_state_value
    G1Robot._share_locomotion_policy_controller = share_locomotion_policy_controller
    G1Robot.save_episode_reset_state = save_episode_reset_state
    G1Robot._restore_episode_controller_state = restore_episode_controller_state
    G1Robot.restore_episode_reset_state = restore_episode_reset_state
    G1Robot._tarsphere_nested_sensor_obs_patch = True
    G1Robot._tarsphere_runtime_contract_patch = True


__all__ = [
    "RobotProfile",
    "ROBOT_PROFILES",
    "DEFAULT_ROBOT_TYPE",
    "get_robot_profile",
    "load_simulator_robot_cfg",
    "load_simulator_robot_components",
    "patch_simulator_robot_observation_layout",
]
