"""Deterministic scene grounding for benchmark authoring tools."""

from tarsphere.bench.grounding.resolve import resolve_scene
from tarsphere.bench.grounding.room_references import (
    require_map_specific_room_text,
    room_roster,
)
from tarsphere.bench.grounding.schema import (
    GroundedObject,
    GroundedScene,
    ObjectApproachPath,
    SceneMap,
    SceneRoom,
    SceneSources,
)

__all__ = [
    "GroundedObject",
    "GroundedScene",
    "ObjectApproachPath",
    "SceneMap",
    "SceneRoom",
    "SceneSources",
    "resolve_scene",
    "require_map_specific_room_text",
    "room_roster",
]
