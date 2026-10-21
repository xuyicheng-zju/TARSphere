"""Episode-long Human identities, dialogue, and visual sightings.

This channel carries public appearance and episode role descriptions, public language
events, and positions admitted only by current named-Human RGB-D visibility.
It never carries invisible simulator state, evaluator goldens, or private
persona/goals.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from tarsphere.harness.memory.contracts import (
    HumanDialogueTurn,
    HumanRecord,
)


class HumanMemory:
    """Own public Human identity and dialogue for one episode."""

    def __init__(self, *, actor_id: str = "robot") -> None:
        self.actor_id = actor_id
        self._humans: dict[str, HumanRecord] = {}
        self._next_sequence: int = 0

    def configure_episode_profiles(
        self,
        profiles: list[dict[str, Any]],
    ) -> None:
        """Seed the episode cast from resolved public Human profiles.

        ``caption`` combines catalog appearance and authored public context;
        ``job`` is the episode occupation, or catalog occupation if unspecified.
        These public facts are loaded
        before the first observation and remain available for the full episode.
        """

        if self._humans:
            raise RuntimeError(
                "Human profiles must be configured before episode observations."
            )
        for index, profile in enumerate(profiles):
            if not isinstance(profile, dict):
                raise TypeError(
                    f"human profile {index} must be a dict, got "
                    f"{type(profile).__name__}"
                )
            name = str(profile.get("name") or "").strip()
            caption = str(profile.get("caption") or "").strip()
            job = str(profile.get("job") or "").strip()
            if not name or not caption or not job:
                raise ValueError(
                    f"human profile {index} requires non-empty name, caption, and job"
                )
            if name in self._humans:
                raise ValueError(f"duplicate Human profile name: {name!r}")
            self._humans[name] = HumanRecord(
                speaker=name,
                caption=caption,
                job=job,
                public_profile_facts=deepcopy(profile.get("public_profile_facts", [])),
            )

    def begin_action_step(
        self,
        *,
        action_step: int,
        observation: Any,
    ) -> None:
        """Ingest public language without inventing a sound-source position."""

        for event in getattr(observation, "public_events", None) or []:
            utterance = self._as_utterance(event, action_step=int(action_step))
            speaker = utterance["speaker"]
            recipient = ""
            delivered = None
            if speaker == self.actor_id:
                recipient = str(event.get("recipient") or "").strip()
                delivered = event.get("delivered")
                if not recipient or not isinstance(delivered, bool):
                    raise ValueError(
                        "Robot language requires recipient and actual delivery status"
                    )
            record = self._humans.setdefault(
                speaker,
                HumanRecord(speaker=speaker),
            )
            record.dialogue.append(
                HumanDialogueTurn(
                    speaker=speaker,
                    text=utterance["text"],
                    action_step=int(action_step),
                    sequence=self._next_sequence,
                    recipient=recipient,
                    delivered=delivered,
                )
            )
            if speaker != self.actor_id:
                record.last_utterance_action_step = int(action_step)
            self._next_sequence += 1

    def retrieved_humans(self) -> list[HumanRecord]:
        """Return all human records with full dialogue retention."""

        return deepcopy(list(self._humans.values()))

    def record_visible_sighting(
        self,
        human_id: str,
        position_xy: tuple[float, float],
        *,
        action_step: int,
        place: str,
    ) -> None:
        """Retain one current RGB-D-grounded named-Human position."""

        name = str(human_id).strip()
        if not name:
            raise ValueError("Visible Human sighting requires a non-empty id.")
        xy = (float(position_xy[0]), float(position_xy[1]))
        if not all(math.isfinite(value) for value in xy):
            raise ValueError("Visible Human sighting requires finite world XY.")
        record = self._humans.setdefault(name, HumanRecord(speaker=name))
        record.last_seen_action_step = int(action_step)
        record.last_seen_position_xy = xy
        record.last_seen_place = str(place).strip()

    @staticmethod
    def _as_utterance(event: Any, *, action_step: int) -> dict[str, str | int]:
        if not isinstance(event, dict):
            raise TypeError(f"public event must be a dict, got {type(event).__name__}")
        if event.get("event_type") != "language":
            raise ValueError(
                f"public event type {event.get('event_type')!r} is not "
                "deliverable; only 'language' is public"
            )
        speaker = str(event.get("speaker") or "").strip()
        text = str(event.get("text") or "").strip()
        if not speaker or not text:
            raise ValueError("language event requires non-empty speaker and text")
        return {
            "speaker": speaker,
            "text": text,
            "action_step": int(action_step),
        }

    def reset(self) -> None:
        self._humans.clear()
        self._next_sequence = 0


__all__ = ["HumanMemory"]
