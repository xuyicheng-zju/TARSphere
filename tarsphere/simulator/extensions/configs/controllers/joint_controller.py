from typing import List, Optional

from tarsphere.simulator.core.config.robot import ControllerCfg


class JointControllerCfg(ControllerCfg):
    type: Optional[str] = "JointController"
    joint_names: List[str]
