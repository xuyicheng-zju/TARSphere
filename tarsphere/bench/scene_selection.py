"""Typed authority for the frozen production scene selection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator

from tarsphere.bench.inputs.sources import (
    AssetSourceRegistry,
    SceneSourceSpec,
    load_asset_source_registry,
)


SCENE_SELECTION_SCHEMA_VERSION = "tarsphere.scene_selection.v1"


class SceneSelectionManifest(BaseModel):
    """Exact source-scene set approved for production authoring."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["tarsphere.scene_selection.v1"] = Field(
        description="Current production scene-selection contract version.",
    )
    source_id: Literal["grscenes"] = Field(
        description="Active asset source whose registered scenes may be selected.",
    )
    scenes: Tuple[SceneSourceSpec, ...] = Field(
        min_length=1,
        description="Complete frozen scene set used to expand production episodes.",
    )

    @field_validator("scenes")
    @classmethod
    def validate_unique_scenes(
        cls,
        scenes: Tuple[SceneSourceSpec, ...],
    ) -> Tuple[SceneSourceSpec, ...]:
        bindings = tuple((scene.collection, scene.scene_id) for scene in scenes)
        if len(bindings) != len(set(bindings)):
            raise ValueError("scene selection contains duplicate scene bindings")
        return scenes

    @property
    def scene_ids(self) -> Tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes)


def validate_scene_selection(
    manifest: SceneSelectionManifest,
    *,
    registry: AssetSourceRegistry,
) -> None:
    """Require every selected binding to exist in the active source registry."""

    source = registry.active_source
    if source.source_id != manifest.source_id:
        raise ValueError(
            f"scene selection uses source {manifest.source_id!r}, but the active "
            f"source is {source.source_id!r}"
        )
    registered = {(scene.collection, scene.scene_id) for scene in source.scenes}
    unknown = sorted(
        (scene.collection, scene.scene_id)
        for scene in manifest.scenes
        if (scene.collection, scene.scene_id) not in registered
    )
    if unknown:
        raise ValueError(
            f"scene selection contains unregistered source bindings: {unknown}"
        )


def load_scene_selection_manifest(path: Path) -> SceneSelectionManifest:
    """Load and bind the production selection to the current source registry."""

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise ValueError(f"scene selection manifest does not exist: {path}") from exc
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"cannot read scene selection manifest {path}: {exc}") from exc
    manifest = SceneSelectionManifest.model_validate(payload)
    validate_scene_selection(
        manifest,
        registry=load_asset_source_registry(),
    )
    return manifest


__all__ = [
    "SCENE_SELECTION_SCHEMA_VERSION",
    "SceneSelectionManifest",
    "load_scene_selection_manifest",
    "validate_scene_selection",
]
