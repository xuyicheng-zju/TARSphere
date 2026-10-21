from typing import Optional, Tuple

from tarsphere.simulator.core.config.robot import ControllerCfg


class LayoutEditMocapControllerCfg(ControllerCfg):
    type: Optional[str] = "LayoutEditMocapController"
    scale: Tuple[float, float, float]
    target_position: Tuple[float, float, float]
    rh_origin_xyz: Optional[Tuple[float, float, float]] = None
    lh_origin_xyz: Optional[Tuple[float, float, float]] = None
    origin_xyz_angle: Optional[Tuple[float, float, float]] = None
