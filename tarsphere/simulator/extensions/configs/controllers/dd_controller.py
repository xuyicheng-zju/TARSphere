from typing import Optional

from tarsphere.simulator.core.config.robot import ControllerCfg


class DifferentialDriveControllerCfg(ControllerCfg):
    type: Optional[str] = "DifferentialDriveController"
    wheel_radius: float
    wheel_base: float
