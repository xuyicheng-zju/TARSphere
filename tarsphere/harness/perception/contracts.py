"""Typed IO contracts for robot perception modules."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np


@dataclass
class PerceptionInput:
    """Current public robot observation passed to perception tools."""

    observation: Any
    text_prompts: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class Detection2D:
    """A semantic 2D detection from a perception backend."""

    label: str
    score: float
    box_xyxy: tuple[float, float, float, float]
    mask: Optional[np.ndarray] = None
    source: str = ""


@dataclass
class PerceptionOutput:
    """Current-frame perception result.

    This is a parallel input to decision, not a required downstream input to
    memory.  Memory may selectively absorb parts of it.
    """

    actor_id: str
    detections: list[Detection2D] = field(default_factory=list)
    semantic_facts: list[str] = field(default_factory=list)
    tool_trace: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)
