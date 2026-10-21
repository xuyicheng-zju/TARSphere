from typing import Optional, Tuple

from tarsphere.simulator.core.config.robot import SensorCfg


class VLNCameraCfg(SensorCfg):
    type: Optional[str] = 'VLNCamera'
    enable: Optional[bool] = True
    resolution: Optional[Tuple[int, int]]
