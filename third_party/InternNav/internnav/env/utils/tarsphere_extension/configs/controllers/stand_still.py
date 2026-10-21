from typing import Optional

from tarsphere.simulator.core.config.robot import ControllerCfg


class StandStillControllerCfg(ControllerCfg):
    type: Optional[str] = 'StandStillController'
