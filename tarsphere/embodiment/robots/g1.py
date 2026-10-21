"""TARSphere controller binding for Unitree G1 with two Dex3-1 hands.

The locomotion policy remains the bundled 27-joint G1 policy.
Dex3 joints use two independent seven-joint controllers and therefore never
change the policy observation or action layout.
"""

import numpy as np

from tarsphere.simulator.macros import gm
from tarsphere.simulator.extensions.configs.controllers import (
    G1MoveBySpeedControllerCfg,
    JointControllerCfg,
    MoveAlongPathPointsControllerCfg,
    MoveToPointBySpeedControllerCfg,
    RotateControllerCfg,
)
from tarsphere.simulator.extensions.configs.robots.g1 import G1RobotCfg
from tarsphere.simulator.extensions.configs.sensors import RepCameraCfg

# Import registers the TARSphere adapters before TARSphere builds controller
# trees. The module side effect is the contract; no symbol is consumed here.
import tarsphere.embodiment.controllers  # noqa: F401
from tarsphere.embodiment.dex3 import (
    DEX3_JOINT_NAMES,
    arm_controller_name,
    dex3_controller_name,
    waist_pitch_controller_name,
    waist_roll_controller_name,
    waist_yaw_controller_name,
)
from tarsphere.infra.manipulation.g1_arm_ik import (
    G1_LEFT_ARM_JOINT_NAMES,
    G1_RIGHT_ARM_JOINT_NAMES,
)
from tarsphere.infra.control.physical_contracts import (
    NAVIGATION_CONTROLLER_REACH_TOLERANCE_M,
    G1_LOWER_POLICY_JOINT_NAMES,
)


G1_LOCOMOTION_JOINT_NAMES = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
]

move_by_speed_cfg = G1MoveBySpeedControllerCfg(
    name="move_by_speed",
    type="TARSphereG1MoveBySpeedController",
    policy_weights_path=(
        gm.ASSET_PATH + "/robots/g1/policy/move_by_speed/g1_15000.onnx"
    ),
    joint_names=G1_LOCOMOTION_JOINT_NAMES,
)

move_to_point_cfg = MoveToPointBySpeedControllerCfg(
    name="move_to_point",
    # This controller is also registered at top level for the final short,
    # collision-sensitive manipulation approach. The path wrapper supplies
    # its own 0.5/1.0 commands, so these lower values affect only direct
    # move_to_point calls.
    forward_speed=0.20,
    rotation_speed=0.50,
    threshold=0.05,
    sub_controllers=[move_by_speed_cfg],
)

move_along_path_cfg = MoveAlongPathPointsControllerCfg(
    name="move_along_path",
    type="HcaMoveAlongPathPointsController",
    forward_speed=0.5,
    rotation_speed=1.0,
    # The official point controller uses this value both to advance internal
    # waypoints and to stop at the terminal point. It therefore shares the
    # exact runtime reach contract used by the executor and Golden oracle.
    threshold=NAVIGATION_CONTROLLER_REACH_TOLERANCE_M,
    sub_controllers=[move_to_point_cfg],
)

rotate_cfg = RotateControllerCfg(
    name="rotate",
    rotation_speed=1.0,
    threshold=0.1,
    sub_controllers=[move_by_speed_cfg],
)

left_hand_joint_controller_cfg = JointControllerCfg(
    name=dex3_controller_name("left"),
    joint_names=list(DEX3_JOINT_NAMES["left"]),
)

right_hand_joint_controller_cfg = JointControllerCfg(
    name=dex3_controller_name("right"),
    joint_names=list(DEX3_JOINT_NAMES["right"]),
)


waist_pitch_joint_controller_cfg = JointControllerCfg(
    name=waist_pitch_controller_name(),
    joint_names=["waist_pitch_joint"],
)

# Psi0 needs all five direct upper-body controllers. The ordinary G1 profile
# additionally registers only the right-arm controller for the public local
# end-effector action; grasp/carry macros keep their reviewed 15-D locomotion
# upper-body path.
waist_yaw_joint_controller_cfg = JointControllerCfg(
    name=waist_yaw_controller_name(),
    joint_names=["waist_yaw_joint"],
)

waist_roll_joint_controller_cfg = JointControllerCfg(
    name=waist_roll_controller_name(),
    joint_names=["waist_roll_joint"],
)

left_arm_joint_controller_cfg = JointControllerCfg(
    name=arm_controller_name("left"),
    joint_names=list(G1_LEFT_ARM_JOINT_NAMES),
)

right_arm_joint_controller_cfg = JointControllerCfg(
    name=arm_controller_name("right"),
    joint_names=list(G1_RIGHT_ARM_JOINT_NAMES),
)

g1_camera_cfg = RepCameraCfg(
    name="camera",
    prim_path="d435_link/Camera",
    resolution=(640, 480),
    depth=True,
)


# Leg joints keep the locomotion-policy gains. Waist yaw and both arms use the
# Dex3 donor manipulation stiffness (625) because the runtime commands them
# directly; damping scales by sqrt(kp_ratio) to preserve zeta, not the donor's 0.
_G1_MANIPULATION_STIFFNESS = 625.0
# The fingers share the arm's 625 stiffness, so they take the wrist damping the
# same sqrt(kp) scaling produced, and the wrist force ceiling they press against.
_G1_FINGER_DAMPING = 2.80
_G1_FINGER_MAX_EFFORT_NM = 5.0
# USD authors wrist pitch/yaw at 5 N·m while wrist roll and the rest of the
# arm sit at 25 N·m. A live stall commanded 0.237 rad, measured 0.730 rad, and
# pinned effort at -4.9997 N·m — the drive was following the command and still
# could not move the joint. Match the arm ceiling so those two distal axes are
# not the only saturated servos.
_G1_WRIST_MAX_EFFORT_NM = 25.0
_G1_WRIST_FORCE_JOINT_NAMES = (
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_G1_POLICY_GAIN_JOINT_NAMES = (
    *G1_LOWER_POLICY_JOINT_NAMES,
    "waist_yaw_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
_G1_POLICY_STIFFNESS = (
    150.0,
    150.0,
    150.0,
    300.0,
    50.0,
    50.0,
    150.0,
    150.0,
    150.0,
    300.0,
    50.0,
    50.0,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
    _G1_MANIPULATION_STIFFNESS,
)
_G1_POLICY_DAMPING = (
    2.0,
    2.0,
    2.0,
    4.0,
    5.0,
    5.0,
    2.0,
    2.0,
    2.0,
    4.0,
    5.0,
    5.0,
    7.22,
    7.07,
    7.07,
    7.07,
    2.50,
    2.80,
    2.80,
    2.80,
    7.07,
    7.07,
    7.07,
    2.50,
    2.80,
    2.80,
    2.80,
)


def set_g1_dex3_gains(robot: object) -> None:
    """Apply locomotion-policy and Dex3 gains to their exact joint subsets."""
    articulation = robot.articulation
    stiffness_by_name = dict(zip(_G1_POLICY_GAIN_JOINT_NAMES, _G1_POLICY_STIFFNESS))
    damping_by_name = dict(zip(_G1_POLICY_GAIN_JOINT_NAMES, _G1_POLICY_DAMPING))
    policy_indices = np.asarray(
        [articulation.get_dof_index(name) for name in G1_LOCOMOTION_JOINT_NAMES],
        dtype=np.int64,
    )
    articulation.set_gains(
        kps=np.asarray(
            [stiffness_by_name[name] for name in G1_LOCOMOTION_JOINT_NAMES],
            dtype=np.float64,
        ),
        kds=np.asarray(
            [damping_by_name[name] for name in G1_LOCOMOTION_JOINT_NAMES],
            dtype=np.float64,
        ),
        joint_indices=policy_indices,
    )

    # The waist-pitch joint retains its USD-authored drive; leg joints keep the
    # locomotion-policy values.
    finger_names = tuple(
        name for hand in ("left", "right") for name in DEX3_JOINT_NAMES[hand]
    )
    finger_indices = np.asarray(
        [articulation.get_dof_index(name) for name in finger_names],
        dtype=np.int64,
    )
    # The USD authors the fingers at stiffness 625 with zero damping, so they
    # ring against a contact instead of loading it, and caps them at 1.40 N·m
    # while the arm they hang off runs at 25 N·m. Closure then registers a
    # contact point that carries no measured normal impulse.
    articulation.set_gains(
        kds=np.full(
            len(finger_names),
            _G1_FINGER_DAMPING,
            dtype=np.float64,
        ),
        joint_indices=finger_indices,
    )
    articulation._articulation._articulation_view.set_max_efforts(
        np.full((1, len(finger_names)), _G1_FINGER_MAX_EFFORT_NM, np.float32),
        joint_indices=finger_indices,
    )
    wrist_indices = np.asarray(
        [articulation.get_dof_index(name) for name in _G1_WRIST_FORCE_JOINT_NAMES],
        dtype=np.int64,
    )
    articulation._articulation._articulation_view.set_max_efforts(
        np.full(
            (1, len(_G1_WRIST_FORCE_JOINT_NAMES)),
            _G1_WRIST_MAX_EFFORT_NM,
            np.float32,
        ),
        joint_indices=wrist_indices,
    )


__all__ = [
    "G1RobotCfg",
    "g1_camera_cfg",
    "left_arm_joint_controller_cfg",
    "left_hand_joint_controller_cfg",
    "move_along_path_cfg",
    "move_by_speed_cfg",
    "move_to_point_cfg",
    "right_arm_joint_controller_cfg",
    "right_hand_joint_controller_cfg",
    "rotate_cfg",
    "set_g1_dex3_gains",
    "waist_pitch_controller_name",
    "waist_pitch_joint_controller_cfg",
    "waist_roll_controller_name",
    "waist_roll_joint_controller_cfg",
    "waist_yaw_controller_name",
    "waist_yaw_joint_controller_cfg",
]
