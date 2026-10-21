"""Canonical room identities shared by maps, authoring, and episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


ROOM_ANNOTATION_SCHEMA = "tarsphere.scene_room_annotations.v1"


class SceneRoomReference(BaseModel):
    """One room name copied verbatim from a scene annotation."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    display_id: str = Field(
        pattern=r"^R[0-9]{2}$",
        description="Private map-specific binding key; never drawn or included in task language.",
    )
    display_label: str = Field(
        min_length=1,
        description="Exact human-readable room label drawn on the semantic map.",
    )
    room_type: str = Field(
        min_length=1,
        description="Semantic room category; never an episode identifier.",
    )
    metadata_room_id: Optional[str] = Field(
        default=None,
        min_length=1,
        description=(
            "GRScenes source alias accepted only while importing source data."
        ),
    )

    @model_validator(mode="after")
    def validate_label(self) -> "SceneRoomReference":
        if self.display_label != self.room_type.replace("_", " ").title():
            raise ValueError(
                "room display_label must be the unnumbered, title-cased room type"
            )
        return self


class SceneRoomRoster(BaseModel):
    """Complete map-specific room vocabulary for one scene."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_id: str = Field(
        min_length=1,
        description="Pinned scene whose semantic map owns these room names.",
    )
    rooms: Tuple[SceneRoomReference, ...] = Field(
        min_length=1,
        description="Rooms in the display order declared by the annotation.",
    )

    @model_validator(mode="after")
    def validate_unique_names(self) -> "SceneRoomRoster":
        for name, values in (
            ("display_id", tuple(room.display_id for room in self.rooms)),
        ):
            if len(set(values)) != len(values):
                raise ValueError(f"room {name} values must be unique")
        aliases = tuple(
            room.metadata_room_id
            for room in self.rooms
            if room.metadata_room_id is not None
        )
        if len(set(aliases)) != len(aliases):
            raise ValueError("room metadata_room_id values must be unique")
        return self

    def require_display_id(self, room_id: str, *, field: str) -> str:
        """Require an exact room id from this scene without normalization."""
        if room_id not in {room.display_id for room in self.rooms}:
            raise ValueError(
                f"{field} must use a display_id from scene {self.scene_id!r}; "
                f"got {room_id!r}"
            )
        return room_id

    def import_metadata_room_id(self, source_room_id: str, *, field: str) -> str:
        """Resolve one GRScenes alias at the source-import boundary."""
        matches = tuple(
            room.display_id
            for room in self.rooms
            if room.metadata_room_id == source_room_id
        )
        if len(matches) != 1:
            raise ValueError(
                f"{field} references unknown metadata_room_id "
                f"{source_room_id!r} in scene {self.scene_id!r}"
            )
        return matches[0]


def load_scene_room_roster(
    path: Path | str,
    *,
    expected_scene_id: str,
) -> SceneRoomRoster:
    """Load the sole room vocabulary from one room-annotation artifact."""
    annotation_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(annotation_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"cannot load room annotations: {annotation_path}") from exc
    if not isinstance(payload, Mapping):
        raise TypeError(f"room annotations must be a mapping: {annotation_path}")
    return parse_scene_room_roster(
        payload,
        expected_scene_id=expected_scene_id,
    )


def parse_scene_room_roster(
    payload: Mapping[str, object],
    *,
    expected_scene_id: str,
) -> SceneRoomRoster:
    """Validate room identities from an already-loaded annotation mapping."""
    if payload.get("schema_version") != ROOM_ANNOTATION_SCHEMA:
        raise ValueError(
            f"unsupported room annotation schema for {expected_scene_id!r}: "
            f"{payload.get('schema_version')!r}"
        )
    if payload.get("scene_id") != expected_scene_id:
        raise ValueError(
            f"room annotations identify {payload.get('scene_id')!r}, "
            f"expected {expected_scene_id!r}"
        )
    raw_rooms = payload.get("rooms")
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise ValueError(f"room annotations contain no rooms for {expected_scene_id!r}")
    rooms = []
    for index, raw in enumerate(raw_rooms):
        if not isinstance(raw, Mapping):
            raise TypeError(f"room annotation #{index} must be a mapping")
        rooms.append(
            SceneRoomReference(
                display_id=str(raw.get("display_id") or "").strip(),
                display_label=str(raw.get("display_label") or "").strip(),
                room_type=str(raw.get("room_type") or "").strip(),
                metadata_room_id=(
                    str(raw["metadata_room_id"]).strip()
                    if raw.get("metadata_room_id") is not None
                    else None
                ),
            )
        )
    return SceneRoomRoster(
        scene_id=expected_scene_id,
        rooms=tuple(rooms),
    )


__all__ = [
    "ROOM_ANNOTATION_SCHEMA",
    "SceneRoomReference",
    "SceneRoomRoster",
    "load_scene_room_roster",
    "parse_scene_room_roster",
]
