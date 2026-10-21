from typing import Optional

from tarsphere.simulator.core.config import RobotCfg
from tarsphere.simulator.macros import gm
from tarsphere.simulator.extensions.configs.controllers import (
    DifferentialDriveControllerCfg,
    GripperControllerCfg,
    InverseKinematicsControllerCfg,
    MoveAlongPathPointsControllerCfg,
    MoveToPointBySpeedControllerCfg,
    RotateControllerCfg,
)
from tarsphere.simulator.extensions.configs.sensors import RepCameraCfg

MOBILE_FRANKA_WHEEL_RADIUS_M = 0.055
MOBILE_FRANKA_WHEEL_BASE_M = 0.28

move_by_speed_cfg = DifferentialDriveControllerCfg(
    name="move_by_speed",
    wheel_base=MOBILE_FRANKA_WHEEL_BASE_M,
    wheel_radius=MOBILE_FRANKA_WHEEL_RADIUS_M,
)

move_to_point_cfg = MoveToPointBySpeedControllerCfg(
    name="move_to_point",
    forward_speed=0.35,
    rotation_speed=0.80,
    threshold=0.05,
    sub_controllers=[move_by_speed_cfg],
)

move_along_path_cfg = MoveAlongPathPointsControllerCfg(
    name="move_along_path",
    forward_speed=0.35,
    rotation_speed=0.80,
    threshold=0.10,
    sub_controllers=[move_to_point_cfg],
)

rotate_cfg = RotateControllerCfg(
    name="rotate",
    rotation_speed=0.80,
    threshold=0.05,
    sub_controllers=[move_by_speed_cfg],
)

arm_ik_cfg = InverseKinematicsControllerCfg(
    name="arm_ik_controller",
    robot_description_path=gm.ASSET_PATH
    + "/robots/franka/rmpflow/robot_descriptor.yaml",
    robot_urdf_path=gm.ASSET_PATH + "/robots/franka/lula_franka_gen.urdf",
    end_effector_frame_name="right_gripper",
    threshold=0.01,
)

gripper_cfg = GripperControllerCfg(
    name="gripper_controller",
)

mobile_franka_camera_cfg = RepCameraCfg(
    name="camera",
    prim_path="base_link/Camera",
    resolution=(640, 480),
    depth=True,
)


class MobileFrankaRobotCfg(RobotCfg):
    name: Optional[str] = "mobile_franka"
    type: Optional[str] = "MobileFrankaRobot"
    prim_path: Optional[str] = "/mobile_franka"
    usd_path: Optional[str] = gm.ASSET_PATH + "/robots/mobile_franka/mobile_franka.usda"
