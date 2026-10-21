"""Shared state objects for kinematic and animated Human bodies.

Human policy decides what a human wants to do.  The Isaac backend executes the
visible body action through the selected motion backend; this module defines the
structured state/result objects shared between the environment and executor.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional


PoseSyncCallback = Callable[[str, "HumanBodyState"], None]


@dataclass
class HumanBodyState:
    """Public body state for one Human actor."""

    human_id: str
    position: list[float]
    yaw_rad: float = 0.0
    body_status: str = "anim_people_pending"
    animation_state: str = "idle"
    action_status: str = "idle"
    physical_status: str = "navmesh_pending"
    active_action: str = ""
    target_position: Optional[list[float]] = None
    pose_stale: bool = False
    last_error: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_observation(self) -> Dict[str, Any]:
        """Return the body fields safe for human-policy ego observations."""
        observation: Dict[str, Any] = {
            "position": list(self.position),
            "yaw_rad": float(self.yaw_rad),
            "body_status": self.body_status,
            "animation_state": self.animation_state,
            "action_status": self.action_status,
            "physical_status": self.physical_status,
            "pose_stale": bool(self.pose_stale),
        }
        if self.active_action:
            observation["active_action"] = self.active_action
        if self.target_position is not None:
            observation["target_position"] = list(self.target_position)
        if self.last_error:
            observation["body_error"] = self.last_error
        return observation


@dataclass
class HumanBodyResult:
    """Structured result from applying a high-level human command."""

    human_id: str
    command: str
    executed: bool
    feedback: str = ""
    state: Optional[HumanBodyState] = None
