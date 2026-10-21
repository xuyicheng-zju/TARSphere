from typing import Optional, Tuple

from tarsphere.simulator.core.config.robot import SensorCfg


class RepCameraCfg(SensorCfg):
    # Fields from params.
    type: Optional[str] = "RepCamera"
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    rgba: Optional[bool] = True
    landmarks: Optional[bool] = False
    depth: Optional[bool] = False
    pointcloud: Optional[bool] = False
    camera_params: Optional[bool] = False


class MocapControlledCameraCfg(SensorCfg):
    # Fields from params.
    type: Optional[str] = "MocapControlledCamera"
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    translation: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = (
        None  # Quaternion in local frame
    )


class LayoutEditMocapControlledCameraCfg(SensorCfg):
    type: Optional[str] = "LayoutEditMocapControlledCamera"
    resolution: Optional[Tuple[int, int]] = None  # Camera only
    translation: Optional[Tuple[float, float, float]] = None
    orientation: Optional[Tuple[float, float, float, float]] = (
        None  # Quaternion in local frame
    )
