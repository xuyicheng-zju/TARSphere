"""Agent-curated semantic landmarks with observation keyframes.

The map is sparse by design. Nothing is inserted merely because a detector or
the simulator knows it exists: the evaluated Agent must explicitly choose a
currently observed Human, object, or place worth retaining. Metric location and
the supporting frame remain runtime-owned evidence behind that choice.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Literal

import numpy as np


SemanticEntityType = Literal["human", "object", "place"]


@dataclass(frozen=True)
class SemanticEntitySummary:
    """Model-facing directory entry; metric coordinates stay tool-private."""

    entity_id: str
    entity_type: SemanticEntityType
    label: str
    description: str
    observed_action_step: int

    def describe(self) -> str:
        detail = f"; {self.description}" if self.description else ""
        return (
            f'{self.entity_id}: {self.entity_type} "{self.label}"{detail}; '
            f"keyframe saved at action step {self.observed_action_step}"
        )


@dataclass(frozen=True)
class SemanticEntityRecord:
    """One measured landmark plus its supporting current-view keyframe."""

    entity_id: str
    entity_type: SemanticEntityType
    label: str
    description: str
    world_xyz: tuple[float, float, float]
    observed_action_step: int
    capture_id: int
    keyframe_rgb: np.ndarray
    bbox_2d: tuple[float, float, float, float] | None = None

    def __post_init__(self) -> None:
        if not self.entity_id or not self.label:
            raise ValueError("Semantic entity id and label must be non-empty.")
        if self.entity_type not in {"human", "object", "place"}:
            raise ValueError(f"Unsupported semantic entity type: {self.entity_type!r}")
        if len(self.world_xyz) != 3 or not all(
            math.isfinite(float(value)) for value in self.world_xyz
        ):
            raise ValueError("Semantic entity world_xyz must be finite XYZ.")
        frame = np.asarray(self.keyframe_rgb)
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise ValueError("Semantic keyframe must be an HxWx3 RGB array.")

    @property
    def summary(self) -> SemanticEntitySummary:
        return SemanticEntitySummary(
            entity_id=self.entity_id,
            entity_type=self.entity_type,
            label=self.label,
            description=self.description,
            observed_action_step=self.observed_action_step,
        )


class SemanticMap:
    """Episode-scoped, explicitly curated semantic entity store."""

    def __init__(self) -> None:
        self._records: dict[str, SemanticEntityRecord] = {}

    @property
    def landmarks(self) -> tuple[SemanticEntityRecord, ...]:
        """Saved entity anchors for BEV; the same records own navigation goals.

        No automatic live-perception update is applied to these historical
        anchors. Rendering must not mutate their keyframes or map state.
        """
        return tuple(record for _, record in sorted(self._records.items()))

    @property
    def summaries(self) -> list[SemanticEntitySummary]:
        return [record.summary for _, record in sorted(self._records.items())]

    def get(self, entity_id: str) -> SemanticEntityRecord | None:
        record = self._records.get(str(entity_id).strip())
        return deepcopy(record) if record is not None else None

    def put(self, record: SemanticEntityRecord) -> None:
        self._records[record.entity_id] = deepcopy(record)

    def remove(self, entity_id: str) -> None:
        self._records.pop(str(entity_id).strip(), None)

    def reset(self) -> None:
        self._records.clear()


class ReplaceSemanticEntityEffect:
    """Reversible map replacement committed at the action-block boundary."""

    def __init__(self, semantic_map: SemanticMap, record: SemanticEntityRecord):
        self._map = semantic_map
        self._record = deepcopy(record)
        self._previous = semantic_map.get(record.entity_id)
        self._committed = False

    def commit(self) -> None:
        if self._committed:
            raise RuntimeError("Semantic entity effect was committed twice.")
        self._map.put(self._record)
        self._committed = True

    def rollback(self) -> None:
        if not self._committed:
            return
        if self._previous is None:
            self._map.remove(self._record.entity_id)
        else:
            self._map.put(self._previous)
        self._committed = False


__all__ = [
    "ReplaceSemanticEntityEffect",
    "SemanticEntityRecord",
    "SemanticEntitySummary",
    "SemanticEntityType",
    "SemanticMap",
]
