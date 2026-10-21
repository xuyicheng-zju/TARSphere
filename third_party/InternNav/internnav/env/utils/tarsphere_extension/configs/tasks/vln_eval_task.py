from typing import Dict, List, Optional

from tarsphere.simulator.core.config.metric import MetricCfg
from tarsphere.simulator.core.config.task import TaskCfg


class VLNEvalTaskCfg(TaskCfg):
    type: Optional[str] = 'VLNEvalTask'
    max_step: int
    check_fall_and_stuck: bool
    robot_ankle_height: float
    fall_height_threshold: float
    metrics: List[MetricCfg]
    warm_up_step: int
    data: Dict
    robot_flash: Optional[bool] = False
    one_step_stand_still: Optional[bool] = False
