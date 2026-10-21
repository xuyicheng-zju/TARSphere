import numpy as np

from tarsphere.simulator.core.config import TaskCfg
from tarsphere.simulator.core.task.metric import BaseMetric
from tarsphere.simulator.core.util import log
from tarsphere.simulator.extensions.configs.metrics.traveled_distance_metric import (
    TraveledDistanceMetricCfg,
)


@BaseMetric.register("TraveledDistanceMetric")
class TraveledDistanceMetric(BaseMetric):
    """
    Calculate the total distance a robot moves
    """

    def __init__(self, config: TraveledDistanceMetricCfg, task_config: TaskCfg):
        super().__init__(config, task_config)
        self.distance: float = 0.0
        self.position = None
        self.robot_name = self.config.robot_name

    def update(self, obs: dict):
        """
        This function is called at each world step.
        """
        if self.position is None:
            self.position = obs[self.robot_name]["position"]
            return
        self.distance += np.linalg.norm(
            self.position - obs[self.robot_name]["position"]
        )
        self.position = obs[self.robot_name]["position"]
        return

    def calc(self):
        """
        This function is called to calculate the metrics when the episode is terminated.
        """
        log.info("TraveledDistanceMetric calc() called.")
        return self.distance
