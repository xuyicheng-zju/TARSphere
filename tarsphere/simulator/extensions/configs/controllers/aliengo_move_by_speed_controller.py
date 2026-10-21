from typing import List, Optional

from tarsphere.simulator.core.config.robot import ControllerCfg


class AliengoMoveBySpeedControllerCfg(ControllerCfg):
    type: Optional[str] = "AliengoMoveBySpeedController"
    joint_names: List[str]
    policy_weights_path: str
