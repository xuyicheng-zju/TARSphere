"""Small deterministic helpers shared by benchmark episode builders."""

from __future__ import annotations

import math

from tarsphere.bench.schema import ActionSpec


# Navigation tools exposed by the default evaluated harness (OPEN_FRONTIER).
DEFAULT_NAVIGATION_ACTIONS = (
    "navigate_to_pointgoal",
    "navigate_to_frontier",
    "navigate_to_entity",
)


def agent_action_space() -> ActionSpec:
    """Return the one task-agnostic action surface for evaluated Agents."""

    return ActionSpec(
        cognitive=["wait", "tilt_camera"],
        perceptual=[
            "locate_object",
            "remember_entity",
            "inspect_entity",
        ],
        communicative=["ask", "speak", "handover_object"],
        physical=[
            "move_body",
            "move_end_effector",
            "crouch",
            "stand_up",
            "grasp_object",
        ],
    )


def yaw_quaternion(
    yaw_rad: float,
) -> tuple[float, float, float, float]:
    """Convert a planar yaw angle to the runtime WXYZ quaternion."""

    return (
        math.cos(float(yaw_rad) / 2.0),
        0.0,
        0.0,
        math.sin(float(yaw_rad) / 2.0),
    )


__all__ = ["agent_action_space", "yaw_quaternion"]
