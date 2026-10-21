"""Distinguish invalid initialization from task-time physical failures."""

from __future__ import annotations

from typing import Any


class EpisodeInitializationError(RuntimeError):
    """No valid initial observation exists; this run cannot be scored."""


class RobotFallError(RuntimeError):
    """Report a measured robot fall with the physical evidence that caused it."""

    def __init__(
        self,
        *,
        action_name: str,
        env_action: dict[str, Any],
        physics_steps: int,
        stability: dict[str, float],
        observation: dict[str, Any],
        physical_contacts: list[dict[str, Any]],
        physical_contact_details: list[dict[str, Any]],
        human_safety_violations: list[dict[str, Any]],
        contact_monitor_snapshot: dict[str, Any] | None,
    ) -> None:
        # Preserve the actual terminal camera/pose sample separately from
        # JSON-compatible diagnostic details; never substitute a pre-action pose.
        self.observation = observation
        self.details = {
            "action_name": action_name,
            "env_action": dict(env_action),
            "physics_steps": int(physics_steps),
            "stability": dict(stability),
            "physical_contacts": physical_contacts,
            "physical_contact_details": physical_contact_details,
            "human_safety_violations": human_safety_violations,
            "contact_monitor_snapshot": contact_monitor_snapshot,
        }
        super().__init__(
            "Robot fall detected after controller execution: "
            f"action={action_name!r}, env_action={env_action!r}, "
            f"physics_steps={physics_steps}, stability={stability!r}."
        )


__all__ = ["EpisodeInitializationError", "RobotFallError"]
