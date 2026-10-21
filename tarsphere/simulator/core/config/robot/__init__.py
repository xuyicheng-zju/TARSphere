from typing import List, Optional, Tuple

from pydantic import BaseModel

from tarsphere.simulator.core.config.sensor import SensorCfg


class BaseCfg(BaseModel):
    def update(self, **kwargs):
        return self.model_copy(update=kwargs, deep=True)


class ControllerCfg(BaseCfg, extra="allow"):
    """Controller type, instance name, and optional nested controllers."""

    name: str
    type: str
    sub_controllers: Optional[List["ControllerCfg"]] = None


class RobotCfg(BaseCfg):
    """Robot asset, initial transform, controllers, and sensors."""

    # meta info
    name: str
    type: str
    prim_path: str
    usd_path: Optional[str] = None  # If Optional, use default usd_path

    # common config
    position: Optional[Tuple[float, float, float]] = (0.0, 0.0, 0.0)
    orientation: Optional[Tuple[float, float, float, float]] = None
    scale: Optional[Tuple[float, float, float]] = None

    controllers: Optional[List[ControllerCfg]] = None
    sensors: Optional[List[SensorCfg]] = None
