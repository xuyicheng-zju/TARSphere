from typing import Optional

from tarsphere.simulator.core.config.robot import ControllerCfg


class GripperControllerCfg(ControllerCfg):
    type: Optional[str] = "GripperController"
