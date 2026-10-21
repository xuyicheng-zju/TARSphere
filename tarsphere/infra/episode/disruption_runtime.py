"""Physics-clock execution contingencies for Disruption Adaptation episodes.

The clock is read from the simulator, never from model calls or decision steps.
No state here changes a task binding or awards a progress point.
"""

from __future__ import annotations

from typing import Any

from tarsphere.contracts.task_profile import (
    COMPONENT_OUTAGE_DURATION_PHYSICS_STEPS,
    TEMPORARY_UNAVAILABILITY_DURATION_PHYSICS_STEPS,
    CapabilitySlice,
    ComponentOutageProfile,
    TemporaryUnavailabilityProfile,
)


class DisruptionRuntime:
    def __init__(self, episode: Any) -> None:
        evaluation = getattr(episode, "evaluation", None)
        profile = getattr(evaluation, "task_profile", None)
        self.profile = (
            profile
            if getattr(profile, "capability_slice", None)
            is CapabilitySlice.DISRUPTION_ADAPTATION
            else None
        )
        self._human_unavailable_since: int | None = None
        self._component_unavailable_since: int | None = None

    def reset(self) -> None:
        self._human_unavailable_since = None
        self._component_unavailable_since = None

    def human_unavailable(
        self,
        human_id: str,
        physics_step: int,
        *,
        trigger: bool = True,
    ) -> bool:
        """Start one refusal window on the first valid target-Human contact."""
        profile = self.profile
        if (
            not isinstance(profile, TemporaryUnavailabilityProfile)
            or profile.human_id != human_id
        ):
            return False
        if self._human_unavailable_since is None:
            if not trigger:
                return False
            self._human_unavailable_since = physics_step
        return (
            self._human_unavailable_since
            <= physics_step
            < self._human_unavailable_since
            + TEMPORARY_UNAVAILABILITY_DURATION_PHYSICS_STEPS
        )

    def component_unavailable(
        self,
        tool_name: str,
        physics_step: int,
        *,
        trigger: bool = True,
    ) -> bool:
        """Start one outage on the first target-tool call and report its state.

        Evaluator observation passes ``trigger=False`` so reading private state
        cannot itself create the disruption.
        """
        profile = self.profile
        if (
            not isinstance(profile, ComponentOutageProfile)
            or profile.component_tool_name != tool_name
        ):
            return False
        if self._component_unavailable_since is None:
            if not trigger:
                return False
            self._component_unavailable_since = physics_step
        return (
            self._component_unavailable_since
            <= physics_step
            < self._component_unavailable_since
            + COMPONENT_OUTAGE_DURATION_PHYSICS_STEPS
        )

    def unavailable_response(
        self,
        human_id: str,
        physics_step: int,
        *,
        trigger: bool = True,
    ) -> str | None:
        if self.human_unavailable(human_id, physics_step, trigger=trigger):
            return self.profile.unavailable_response
        return None

    def trace_state(self, physics_step: int) -> dict[str, Any]:
        """Evaluator-private state for checking whether the disturbance occurred."""
        profile = self.profile
        if isinstance(profile, TemporaryUnavailabilityProfile):
            return {
                "subtype": profile.subtype,
                "active": self.human_unavailable(
                    profile.human_id,
                    physics_step,
                    trigger=False,
                ),
                "triggered": self._human_unavailable_since is not None,
                "human_id": profile.human_id,
            }
        if isinstance(profile, ComponentOutageProfile):
            return {
                "subtype": profile.subtype,
                "active": self.component_unavailable(
                    profile.component_tool_name,
                    physics_step,
                    trigger=False,
                ),
                "triggered": self._component_unavailable_since is not None,
                "component_tool_name": profile.component_tool_name,
            }
        if profile is not None:
            return {"subtype": profile.subtype}
        return {}
