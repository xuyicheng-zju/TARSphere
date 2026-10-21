"""Robot perception module implementation."""

from __future__ import annotations

from typing import Any

from tarsphere.harness.perception.contracts import (
    PerceptionInput,
    PerceptionOutput,
)


class RobotPerceptionModule:
    """Deterministic current-frame facts, computed every step.

    This is the fixed textual half of the agent's perceptual input: whatever
    public text the environment surfaced this step. RGB remains the policy's
    visual input; synchronized depth is available privately inside
    explicitly called grounding/manipulation tools. Depth never supplies hidden target identity,
    destination, or route state.

    Semantic recognition is deliberately not here. Running an open-vocabulary
    grounder on every frame both costs a forward pass the agent did not ask
    for and forces a guess at what to look for; the agent instead calls the
    ``locate_object`` tool with its own phrase for an object it has already
    identified in the current frame. See ``tools/handlers.LocateObjectHandler``.
    """

    name = "robot_perception"

    def reset(self, context: Any = None) -> None:
        del context

    def run(self, data: PerceptionInput, context: Any = None) -> PerceptionOutput:
        del context
        return self._deterministic_output(data)

    def process(
        self,
        observation,
        *,
        metadata: dict[str, Any] | None = None,
    ) -> PerceptionOutput:
        """Process one RobotAgent observation."""
        return self.run(
            PerceptionInput(
                observation=observation,
                metadata=dict(metadata or {}),
            )
        )

    @staticmethod
    def _deterministic_output(data: PerceptionInput) -> PerceptionOutput:
        observation = data.observation
        facts: list[str] = []

        ego = observation.views.get("ego")
        if ego is None:
            raise ValueError("robot perception requires the normalized 'ego' view")

        if observation.text:
            facts.append(str(observation.text))

        return PerceptionOutput(
            actor_id=observation.actor_id,
            semantic_facts=facts,
            metadata={"module": RobotPerceptionModule.name},
        )
