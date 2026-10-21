"""Structured outputs required at the current public task event."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DecisionRequirements:
    """Structured fields the Agent must emit for one public task event.

    The Agent harness derives these requirements from the
    episode's current public event. Event-local scorers consume the resulting
    outputs after the rollout. Ground-truth answers are never included here.

    ``event_milestone_ids`` identifies which events opened the requirement. It
    is harness plumbing, never model-visible content. Requirements that arrive
    during a serial action block are accumulated by the Agent and presented at
    the next policy inference without interrupting the authored suffix.
    """

    planning: dict[str, Any] | None = None
    event_milestone_ids: tuple[str, ...] = ()

    def any_required(self) -> bool:
        return any(v is not None for v in (self.planning,))


__all__ = [
    "DecisionRequirements",
]
