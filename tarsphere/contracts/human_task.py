"""Authored symbolic Human work, grounded in a real execution stance."""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


# Position tolerance for an authored operation stance, not speech range or
# navigation clearance. A goto must also have actually finished.
HUMAN_TASK_STANCE_TOLERANCE_M = 0.25


class HumanTaskAction(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    human: str = Field(
        min_length=1, description="Episode Human assigned this auxiliary work."
    )
    command: Literal["perform_task"] = "perform_task"
    target_anchor: str = Field(
        min_length=1,
        description="Real operation stance in world_knowledge.spatial_anchors.",
    )
    task_description: str = Field(
        min_length=1,
        description="Authored symbolic auxiliary work; does not mutate geometry or complete Robot outcomes.",
    )
    partner_human_id: str | None = Field(
        default=None,
        min_length=1,
        description="Other Human whose current pose must be within the shared auditory interaction radius for the auxiliary exchange.",
    )


def human_task_stance_error(position, target, *, moving: bool, stale: bool) -> str:
    """Return a factual rejection reason; absent or nonfinite poses never pass."""
    if moving:
        return "Human motion is still active"
    if stale:
        return "Human pose is stale"
    if any(
        not isinstance(p, (list, tuple))
        or len(p) != 3
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in p
        )
        for p in (position, target)
    ):
        return "Human task requires finite world XYZ poses"
    distance = math.dist(position[:3], target[:3])
    if distance > HUMAN_TASK_STANCE_TOLERANCE_M:
        return f"Human is {distance:.3f} m from the operation stance"
    return ""
