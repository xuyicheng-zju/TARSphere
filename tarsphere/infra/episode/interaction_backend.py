"""Deterministic routing for optional, episode-authored Human answers."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Mapping

from tarsphere.bench.schema import (
    AuthoredInteractionAnswer,
    InteractionIntentSpec,
    TARSphereEpisode,
    RelativeDirectionInteractionAnswer,
)


@dataclass(frozen=True, slots=True)
class RoutedInteraction:
    """One matched intent and the public Human answer it produces."""

    intent_id: str
    message: str


class DeterministicInteractionBackend:
    """Route an ``ask`` through only the intents enabled by one episode."""

    def __init__(self, episode: TARSphereEpisode) -> None:
        self._episode = episode
        self._intents = tuple(episode.interaction.intents)
        self._anchors = dict((episode.world_knowledge.get("spatial_anchors") or {}))
        profile = getattr(getattr(episode, "evaluation", None), "task_profile", None)
        capability = getattr(profile, "capability_slice", "")
        self._broad_contextual_routing = (
            str(getattr(capability, "value", capability)) == "intent_understanding"
        )

    def route(
        self,
        *,
        recipient_id: str,
        intent_id: str,
        world_snapshot: Mapping[str, Any],
        completed_milestones: frozenset[str] = frozenset(),
        allow_private: bool = False,
    ) -> RoutedInteraction | None:
        """Return the selected public answer, or ``None`` when unsupported."""

        intent = next(
            (item for item in self._intents if item.intent_id == str(intent_id)),
            None,
        )
        if intent is None:
            return None
        # Public IDs are the Agent's deterministic interface. Hidden topics
        # may be selected only by the authorized NPC language router, never
        # by guessing their stable identifiers in an Agent tool argument.
        if not intent.public and not allow_private:
            return None
        if not {item.milestone_ref for item in intent.preconditions}.issubset(
            completed_milestones
        ):
            return None
        if (
            intent.eligible_human_ids
            and str(recipient_id) not in intent.eligible_human_ids
        ):
            return None
        return RoutedInteraction(
            intent_id=intent.intent_id,
            message=self._render_answer(intent, world_snapshot),
        )

    def available_topics(
        self, *, recipient_id: str, completed_milestones: frozenset[str]
    ) -> list[dict[str, str]]:
        """Return authorized semantic scopes, never canonical question strings."""
        return [
            {
                "intent_id": intent.intent_id,
                "description": intent.description,
                "match_policy": (
                    "broad_contextual_clarification"
                    if self._broad_contextual_routing and not intent.public
                    else "semantic_topic"
                ),
            }
            for intent in self._intents
            if (
                not intent.eligible_human_ids
                or recipient_id in intent.eligible_human_ids
            )
            and {ref.milestone_ref for ref in intent.preconditions}.issubset(
                completed_milestones
            )
        ]

    def _render_answer(
        self,
        intent: InteractionIntentSpec,
        world_snapshot: Mapping[str, Any],
    ) -> str:
        answer = intent.answer
        if isinstance(answer, AuthoredInteractionAnswer):
            return answer.text
        if not isinstance(answer, RelativeDirectionInteractionAnswer):
            raise TypeError(
                f"Unsupported interaction answer type: {type(answer).__name__}"
            )
        anchor = self._anchors[answer.target_anchor_id]
        robot = world_snapshot.get("robot")
        if not isinstance(robot, Mapping):
            raise RuntimeError(
                "relative_direction interaction requires live Robot state"
            )
        position = robot.get("position")
        yaw = robot.get("yaw_rad")
        if not (
            isinstance(position, (list, tuple))
            and len(position) >= 2
            and all(math.isfinite(float(value)) for value in position[:2])
            and yaw is not None
            and math.isfinite(float(yaw))
        ):
            raise RuntimeError(
                "relative_direction interaction requires finite Robot XY and "
                "yaw_rad from the live simulator snapshot"
            )
        delta_x = float(anchor[0]) - float(position[0])
        delta_y = float(anchor[1]) - float(position[1])
        distance_m = math.hypot(delta_x, delta_y)
        yaw_rad = float(yaw)
        forward_m = math.cos(yaw_rad) * delta_x + math.sin(yaw_rad) * delta_y
        left_m = -math.sin(yaw_rad) * delta_x + math.cos(yaw_rad) * delta_y
        bearing_deg = math.degrees(math.atan2(left_m, forward_m))
        direction = self._relative_direction_phrase(bearing_deg)
        return (
            f"{answer.target_name.capitalize()} is about {distance_m:.1f} m "
            f"{direction} relative to your current body heading "
            f"(straight-line distance, not a walking route). "
            f"{answer.approach_description}"
        )

    @staticmethod
    def _relative_direction_phrase(bearing_deg: float) -> str:
        directions = (
            "ahead",
            "ahead-left",
            "to your left",
            "behind-left",
            "behind you",
            "behind-right",
            "to your right",
            "ahead-right",
        )
        sector = int(math.floor((float(bearing_deg) + 22.5) / 45.0)) % 8
        return directions[sector]


__all__ = [
    "DeterministicInteractionBackend",
    "RoutedInteraction",
]
