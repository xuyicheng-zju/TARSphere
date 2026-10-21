"""Decouple path waypoint completion from MoveToPoint speed tapering.

The ``MoveAlongPathPointsController`` wrapper forwards one ``threshold`` to two
different decisions: advancing a waypoint and cubically reducing locomotion
speed. A loaded G1 can therefore enter the policy dead zone just outside the
same radius required to advance. This adapter preserves the controller tree and action API while assigning each decision its own physical
contract.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np

from tarsphere.simulator.core.robot.controller import BaseController
from tarsphere.simulator.core.util import log
from tarsphere.simulator.extensions.controllers.move_along_path_points_controller import (
    MoveAlongPathPointsController,
)

from tarsphere.infra.control.physical_contracts import (
    NAVIGATION_CONTROLLER_SPEED_TAPER_THRESHOLD_M,
)


@BaseController.register("HcaMoveAlongPathPointsController")
class HcaMoveAlongPathPointsController(MoveAlongPathPointsController):
    """Path controller with independent reach and taper thresholds."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.speed_taper_threshold = float(
            NAVIGATION_CONTROLLER_SPEED_TAPER_THRESHOLD_M
        )
        if self.speed_taper_threshold <= 0.0:
            raise ValueError("Path speed-taper threshold must be positive.")
        if self.speed_taper_threshold >= float(self.threshold):
            raise ValueError(
                "Path speed-taper threshold must be smaller than the "
                "waypoint reach threshold."
            )

    def forward(
        self,
        start_position: np.ndarray,
        start_orientation: np.ndarray,
        path_points: np.ndarray,
        forward_speed: float = 1,
        rotation_speed: float = 8,
        threshold: float = 0.02,
    ):
        reach_threshold = float(threshold)
        if reach_threshold <= 0.0:
            raise ValueError("Path waypoint reach threshold must be positive.")

        if self.path_points != path_points:
            self.path_points = path_points
            self.path_point_idx = 0
            log.info("reset path points")
            self.current_path_point = np.array(
                deepcopy(self.path_points[self.path_point_idx])
            )
            self.current_path_point[-1] = 0

        start_position[-1] = 0
        distance = float(np.linalg.norm(start_position - self.current_path_point))
        while (
            distance < reach_threshold
            and self.path_point_idx < len(self.path_points) - 1
        ):
            self.path_point_idx += 1
            self.current_path_point = np.array(
                deepcopy(self.path_points[self.path_point_idx])
            )
            self.current_path_point[-1] = 0
            log.info(f"switch to next path point: {self.current_path_point}")
            distance = float(np.linalg.norm(start_position - self.current_path_point))

        terminal_reached = (
            self.path_point_idx == len(self.path_points) - 1
            and distance < reach_threshold
        )
        return self.sub_controllers[0].forward(
            start_position=start_position,
            start_orientation=start_orientation,
            goal_position=self.current_path_point,
            forward_speed=0.0 if terminal_reached else forward_speed,
            rotation_speed=0.0 if terminal_reached else rotation_speed,
            threshold=(
                reach_threshold if terminal_reached else self.speed_taper_threshold
            ),
        )


__all__ = ["HcaMoveAlongPathPointsController"]
