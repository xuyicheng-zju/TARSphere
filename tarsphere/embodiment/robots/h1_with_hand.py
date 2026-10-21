"""TARSphere H1-with-hand configuration.

The base H1WithHand binding already provides locomotion and
right-arm IK.  TARSphere adds a thin right-hand joint controller so manipulation
skills can command hand presets through the same controller interface.
"""

from __future__ import annotations

from tarsphere.simulator.extensions.configs.controllers import JointControllerCfg
from tarsphere.simulator.extensions.configs.robots.h1_with_hand import (
    H1WithHandRobotCfg,
    h1_with_hand_camera_cfg,
    move_by_speed_cfg,
    right_arm_ik_controller_cfg,
    right_arm_joint_controller_cfg,
    rotate_cfg,
)


RIGHT_HAND_JOINT_NAMES = [
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
]


right_hand_joint_controller_cfg = JointControllerCfg(
    name="right_hand_joint_controller",
    joint_names=RIGHT_HAND_JOINT_NAMES,
)


__all__ = [
    "H1WithHandRobotCfg",
    "h1_with_hand_camera_cfg",
    "move_by_speed_cfg",
    "rotate_cfg",
    "right_arm_ik_controller_cfg",
    "right_arm_joint_controller_cfg",
    "right_hand_joint_controller_cfg",
    "RIGHT_HAND_JOINT_NAMES",
]
