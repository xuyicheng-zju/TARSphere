"""The single public structure for an Agent task plan."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class StructuredPlan(BaseModel):
    """A submitted task goal and its non-empty ordered execution steps."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    goal: str = Field(
        min_length=1,
        description="Non-empty task outcome currently pursued by the Agent.",
    )
    steps: tuple[str, ...] = Field(
        min_length=1,
        description="Non-empty ordered physical or interaction subgoals.",
    )

    @field_validator("goal")
    @classmethod
    def normalize_goal(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("plan.goal must be a non-empty string")
        return normalized

    @field_validator("steps")
    @classmethod
    def normalize_steps(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(step.strip() for step in value)
        if any(not step for step in normalized):
            raise ValueError("plan.steps entries must be non-empty strings")
        return normalized

    def as_public_dict(self) -> dict[str, Any]:
        """Return the exact JSON shape accepted from and shown to the Agent."""

        return self.model_dump(mode="json")


__all__ = ["StructuredPlan"]
