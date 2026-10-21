"""Diagnostic event log for public actions and utterances in an episode.

The authoritative evaluator reads typed trajectory state, milestone conditions,
and measured physical outcomes.  This append-only bus is a replay/debugging
view of selected public events; it never grants progress by itself.

- Language events (robot speaks, human responds, permission requested)
- Action events (robot approaches, robot picks up object)
- Physical events (grasp succeeds, navigation reaches target, collision)
- Social events (robot enters personal space, human accepts/rejects help)
- State events (human changes phase, milestone triggered)
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional


class EventType(str, Enum):
    """Categories of events in a TARSphere episode."""

    LANGUAGE = "language"
    ACTION = "action"
    PHYSICAL = "physical"
    SOCIAL = "social"
    STATE = "state"
    MILESTONE = "milestone"
    ERROR = "error"


@dataclass
class Event:
    """A single event during an episode."""

    step: int
    event_type: EventType
    name: str
    source: str  # 'robot', 'human:<id>', 'environment'
    data: Dict[str, Any] = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "step": self.step,
            "event_type": self.event_type.value,
            "name": self.name,
            "source": self.source,
            "data": self.data,
            "timestamp": self.timestamp,
        }


class EventBus:
    """Append-only diagnostic recording and retrieval system.

    Usage::

        bus = EventBus()
        bus.emit(Event(step=10, event_type=EventType.ACTION, name='approach',
                       source='robot', data={'target': 'Nora'}))
        recent = bus.get_events(since_step=5)
    """

    def __init__(self):
        self._events: List[Event] = []
        self._step = 0

    @property
    def step(self) -> int:
        return self._step

    @step.setter
    def step(self, value: int):
        self._step = value

    def emit(self, event: Event):
        """Record an event."""
        self._events.append(event)

    def emit_action(self, name: str, source: str, **data):
        """Shorthand for emitting an action event."""
        self.emit(
            Event(
                step=self._step,
                event_type=EventType.ACTION,
                name=name,
                source=source,
                data=data,
            )
        )

    def emit_language(self, name: str, source: str, message: str, **data):
        """Shorthand for emitting a language event."""
        d = {"message": message}
        d.update(data)
        self.emit(
            Event(
                step=self._step,
                event_type=EventType.LANGUAGE,
                name=name,
                source=source,
                data=d,
            )
        )

    def emit_social(self, name: str, source: str, **data):
        """Shorthand for emitting a social event."""
        self.emit(
            Event(
                step=self._step,
                event_type=EventType.SOCIAL,
                name=name,
                source=source,
                data=data,
            )
        )

    def get_events(
        self,
        since_step: int = 0,
        event_type: Optional[EventType] = None,
        source: Optional[str] = None,
    ) -> List[Event]:
        """Retrieve events with optional filtering."""
        result = []
        for e in self._events:
            if e.step < since_step:
                continue
            if event_type and e.event_type != event_type:
                continue
            if source and e.source != source:
                continue
            result.append(e)
        return result

    def get_latest(self, event_type: Optional[EventType] = None) -> Optional[Event]:
        """Get the most recent event, optionally filtered by type."""
        for e in reversed(self._events):
            if event_type is None or e.event_type == event_type:
                return e
        return None

    def get_all(self) -> List[Event]:
        """Return all recorded events."""
        return list(self._events)

    def clear(self):
        """Clear all events (e.g., on episode reset)."""
        self._events.clear()
        self._step = 0

    def to_dicts(self) -> List[Dict[str, Any]]:
        """Serialize all events to dicts."""
        return [e.to_dict() for e in self._events]
