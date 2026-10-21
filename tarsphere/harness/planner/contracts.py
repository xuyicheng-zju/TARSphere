"""Typed IO contracts for the robot navigation planner.

The planner turns a goal into a collision-checked list of world-space waypoints
over the live :class:`BevMap`.  It is a leaf module: it reads the map and the
robot pose, and writes only a path.  It never touches the simulator, the VLM,
or any hidden/evaluator state.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class PlanInput:
    """A single planning request.

    Plans to ``goal_xy`` or any member of ``goal_candidates_xy`` in world metres.
    ``start_xy``/``start_yaw_rad`` are the robot's current world pose.

    Candidate sets are caller-defined geometric domains, never enlarged or
    snapped by the planner. Search determines the reachable destination.
    """

    start_xy: tuple[float, float]
    start_yaw_rad: float
    goal_xy: Optional[tuple[float, float]] = None
    # Exact allowed world-space endpoints. None selects single-goal mode;
    # an empty tuple has no destination. Search may not snap these points.
    goal_candidates_xy: Optional[tuple[tuple[float, float], ...]] = None


@dataclass
class PlanOutput:
    """A collision-checked waypoint path in world metres.

    ``waypoints`` excludes the start cell; the first entry is the immediate
    sub-goal the controller should head toward.  ``status`` is one of:
    ``ok`` | ``already_there`` | ``unreachable`` | ``no_map``.
    """

    waypoints: list[list[float]] = field(default_factory=list)
    goal_xy: Optional[list[float]] = None
    status: str = "ok"
    metadata: dict[str, Any] = field(default_factory=dict)
