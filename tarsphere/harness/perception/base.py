"""Base protocol for robot perception backends."""

from __future__ import annotations

from typing import Protocol

from tarsphere.harness.perception.contracts import (
    PerceptionInput,
    PerceptionOutput,
)


class PerceptionBackend(Protocol):
    """Semantic/current-frame perception backend."""

    name: str

    def reset(self) -> None:
        """Clear backend state at episode reset."""

    def process(self, data: PerceptionInput) -> PerceptionOutput:
        """Return current-frame perception output."""
