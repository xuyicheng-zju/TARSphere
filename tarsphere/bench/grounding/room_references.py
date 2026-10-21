"""Enforce one scene-specific room vocabulary across authoring layers."""

from __future__ import annotations

import re
from typing import Iterable

from tarsphere.bench.grounding.schema import GroundedScene
from tarsphere.scene_graph.room_references import (
    SceneRoomReference,
    SceneRoomRoster,
)


def room_roster(scene: GroundedScene) -> SceneRoomRoster:
    """Derive the reference vocabulary already owned by a grounded scene."""
    return SceneRoomRoster(
        scene_id=scene.scene_id,
        rooms=tuple(
            SceneRoomReference(
                display_id=room.room_id,
                display_label=room.label,
                room_type=room.room_type,
                metadata_room_id=room.source_room_id,
            )
            for room in scene.rooms
        ),
    )


def require_map_specific_room_text(
    texts: Iterable[tuple[str, str]],
    scene: GroundedScene,
) -> None:
    """Keep private room keys and numbered room labels out of public language."""
    roster = room_roster(scene)
    aliases = tuple(
        room.metadata_room_id
        for room in roster.rooms
        if room.metadata_room_id is not None
    )
    for field, text in texts:
        for alias in aliases:
            if re.search(
                r"(?<!\\w)" + re.escape(alias) + r"(?!\\w)", text, re.IGNORECASE
            ):
                raise ValueError(f"{field} exposes private room key {alias!r}")
        numbered = re.search(
            r"\\b[RS]\\d{2}\\b|\\b[a-z]+(?:_[a-z]+)*_\\d{2}\\b"
            r"|\\b(?:room|bedroom|kitchen|corridor|bathroom|balcony)\\s+\\d{2}\\b",
            text,
            re.IGNORECASE,
        )
        if numbered:
            raise ValueError(
                f"{field} uses numbered scene/room label {numbered.group(0)!r}; "
                "use an unnumbered room name and a physical landmark"
            )


__all__ = [
    "require_map_specific_room_text",
    "room_roster",
]
