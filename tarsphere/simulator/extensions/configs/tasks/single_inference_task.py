from typing import Optional

from tarsphere.simulator.core.config.task import TaskCfg


class SingleInferenceTaskCfg(TaskCfg):
    type: Optional[str] = "SingleInferenceTask"
