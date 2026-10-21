"""Current-release asset sources and GRScenes eligibility policy."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from tarsphere.assets import tarsphere_assets_root


SOURCE_REGISTRY_PATH = Path(__file__).with_name("sources.json")
SOURCE_REGISTRY_SCHEMA_VERSION = "2.0"
GRSCENES_SOURCE_ID = "grscenes"
CURRENT_GRSCENES_SCENE_COUNT = 40
GRSCENES_ROOT = Path("scenes") / "GRScenes-100"

_SCENE_SOURCE_KEYS = {
    "start_result_navigation.usd": "scene_usd",
    "interactive_obj_list.json": "interactive_objects",
}
_METADATA_SOURCE_KEYS = {
    "object_dict.json": "object_dict",
    "object_dict_with_caption.json": "object_dict_with_caption",
    "model_mapping.json": "model_mapping",
    "paths.json": "paths",
    "annotation.json": "annotation",
    "occupancy.npy": "occupancy_npy",
    "rgb_init.png": "rgb_init",
}


class AssetSourceRegistryError(RuntimeError):
    """Raised when the current asset-source policy is malformed."""


class SceneCatalogError(RuntimeError):
    """Raised when a pinned scene cannot satisfy the active source contract."""


class SceneSourceSpec(BaseModel):
    """One scene binding in the current 40-scene main set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    scene_id: str
    collection: Literal["home_scenes", "commercial_scenes"]

    @field_validator("scene_id")
    @classmethod
    def _validate_scene_id(cls, value: str) -> str:
        normalized = str(value).strip()
        if not normalized:
            raise ValueError("scene_id cannot be empty")
        return normalized


class AssetSourceSpec(BaseModel):
    """One physical asset source used by deterministic benchmark tools."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    role: Literal["primary", "secondary"]
    status: Literal["active", "reserved"]
    scenes: Tuple[SceneSourceSpec, ...] = ()
    required_scene_files: Tuple[str, ...] = ()
    required_metadata_files: Tuple[str, ...] = ()

    @field_validator(
        "source_id",
        mode="before",
    )
    @classmethod
    def _strip_optional_text(cls, value: object) -> object:
        if value is None:
            return None
        return str(value).strip()

    @field_validator("scenes")
    @classmethod
    def _validate_scenes(
        cls,
        values: Tuple[SceneSourceSpec, ...],
    ) -> Tuple[SceneSourceSpec, ...]:
        keys = tuple((value.collection, value.scene_id) for value in values)
        if len(set(keys)) != len(keys):
            raise ValueError("source scenes must contain unique bindings")
        scene_ids = tuple(value.scene_id for value in values)
        if len(set(scene_ids)) != len(scene_ids):
            raise ValueError(
                "source scene_id values must be globally unique because "
                "public resolvers are keyed by scene_id"
            )
        if keys != tuple(sorted(keys)):
            raise ValueError("source scenes must be sorted by collection and scene_id")
        return values

    @field_validator("required_scene_files", "required_metadata_files")
    @classmethod
    def _validate_required_files(
        cls,
        values: Tuple[str, ...],
    ) -> Tuple[str, ...]:
        normalized = tuple(str(value).strip() for value in values)
        if any(not value for value in normalized):
            raise ValueError("required file lists cannot contain empty values")
        if len(set(normalized)) != len(normalized):
            raise ValueError("required file lists must contain unique values")
        return normalized

    @model_validator(mode="after")
    def _validate_status_contract(self) -> "AssetSourceSpec":
        if not self.source_id:
            raise ValueError("source_id cannot be empty")
        if self.status == "active":
            if not self.scenes:
                raise ValueError("active asset sources require pinned scenes")
            if not self.required_scene_files or not self.required_metadata_files:
                raise ValueError("active asset sources require complete file contracts")
            if self.source_id == GRSCENES_SOURCE_ID:
                if set(self.required_scene_files) != set(_SCENE_SOURCE_KEYS):
                    raise ValueError(
                        "GRScenes required_scene_files must match the implemented "
                        "source map exactly"
                    )
                if set(self.required_metadata_files) != set(_METADATA_SOURCE_KEYS):
                    raise ValueError(
                        "GRScenes required_metadata_files must match the implemented "
                        "source map exactly"
                    )
        else:
            if self.scenes or self.required_scene_files or self.required_metadata_files:
                raise ValueError("reserved asset sources cannot expose loadable assets")
        return self

    @property
    def scene_ids(self) -> Tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.scenes)

    def require_scene(self, scene_id: str) -> SceneSourceSpec:
        matches = tuple(scene for scene in self.scenes if scene.scene_id == scene_id)
        if len(matches) != 1:
            raise KeyError(f"scene binding is not unique in source: {scene_id!r}")
        return matches[0]


class AssetSourceRegistry(BaseModel):
    """Validated physical source policy for the current main set."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str
    current_source_id: str
    sources: Tuple[AssetSourceSpec, ...]

    @model_validator(mode="after")
    def _validate_release_contract(self) -> "AssetSourceRegistry":
        if self.schema_version != SOURCE_REGISTRY_SCHEMA_VERSION:
            raise ValueError(
                f"unsupported source registry schema_version {self.schema_version!r}; "
                f"expected {SOURCE_REGISTRY_SCHEMA_VERSION!r}"
            )
        source_ids = tuple(source.source_id for source in self.sources)
        if len(set(source_ids)) != len(source_ids):
            raise ValueError("asset source ids must be unique")
        active = tuple(source for source in self.sources if source.status == "active")
        primary = tuple(source for source in self.sources if source.role == "primary")
        if len(active) != 1 or len(primary) != 1 or active[0] is not primary[0]:
            raise ValueError(
                "current release requires exactly one active primary source"
            )
        if active[0].source_id != self.current_source_id:
            raise ValueError(
                "current_source_id must identify the active primary source"
            )
        if (
            active[0].source_id == GRSCENES_SOURCE_ID
            and len(active[0].scene_ids) != CURRENT_GRSCENES_SCENE_COUNT
        ):
            raise ValueError(
                "current GRScenes release must pin exactly "
                f"{CURRENT_GRSCENES_SCENE_COUNT} scenes"
            )
        return self

    @property
    def active_source(self) -> AssetSourceSpec:
        return next(source for source in self.sources if source.status == "active")

    def require_source(self, source_id: str) -> AssetSourceSpec:
        for source in self.sources:
            if source.source_id == source_id:
                return source
        raise KeyError(f"asset source is not registered: {source_id!r}")


class SceneCatalogEntry(BaseModel):
    """Eligibility and caption coverage for one discovered GRScenes scene."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    collection: str
    scene_id: str
    authorable: bool
    exclusion_reason: str = ""
    scene_asset_path: str = ""
    metadata_sources: Dict[str, str] = Field(default_factory=dict)
    object_count: int = 0
    captioned_object_count: int = 0
    uncaptioned_object_count: int = 0

    @model_validator(mode="after")
    def _validate_entry(self) -> "SceneCatalogEntry":
        if self.authorable:
            if self.exclusion_reason:
                raise ValueError("authorable scenes cannot have an exclusion reason")
            if not self.scene_asset_path or not self.metadata_sources:
                raise ValueError("authorable scenes require source paths")
            if self.object_count <= 0 or self.captioned_object_count <= 0:
                raise ValueError("authorable scenes require captioned semantic objects")
            if (
                self.captioned_object_count + self.uncaptioned_object_count
                != self.object_count
            ):
                raise ValueError("scene caption counts do not sum to object_count")
        elif not self.exclusion_reason:
            raise ValueError("excluded scenes require an exclusion reason")
        return self


class GRScenesCatalog(BaseModel):
    """Live inventory under the pinned current-release GRScenes policy."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str
    authorable_scenes: Tuple[SceneCatalogEntry, ...]
    excluded_scenes: Tuple[SceneCatalogEntry, ...]

    @property
    def authorable_scene_ids(self) -> Tuple[str, ...]:
        return tuple(scene.scene_id for scene in self.authorable_scenes)

    def require_scene(self, scene_id: str) -> SceneCatalogEntry:
        for scene in self.authorable_scenes:
            if scene.scene_id == scene_id:
                return scene
        raise KeyError(f"scene is not authorable in the current release: {scene_id!r}")


@dataclass(frozen=True)
class GRSceneSourcePaths:
    """Resolved on-disk sources for one pinned GRScenes scene."""

    assets_root: Path
    source: AssetSourceSpec
    scene_id: str
    collection: str
    scene_dir: Path
    meta_dir: Path

    @property
    def scene_usd(self) -> Path:
        return self.scene_dir / "start_result_navigation.usd"

    def source_map(self) -> Dict[str, str]:
        sources: Dict[str, str] = {}
        for filename in self.source.required_scene_files:
            key = _SCENE_SOURCE_KEYS.get(filename)
            if key:
                sources[key] = _relative(self.scene_dir / filename, self.assets_root)
        for filename in self.source.required_metadata_files:
            key = _METADATA_SOURCE_KEYS.get(filename)
            if key:
                sources[key] = _relative(self.meta_dir / filename, self.assets_root)
        return sources


def load_asset_source_registry(
    path: Optional[Path | str] = None,
) -> AssetSourceRegistry:
    """Load the single machine-readable source policy for this release."""
    registry_path = Path(path).expanduser().resolve() if path else SOURCE_REGISTRY_PATH
    try:
        raw = json.loads(registry_path.read_text(encoding="utf-8"))
        return AssetSourceRegistry.model_validate(raw)
    except Exception as exc:
        raise AssetSourceRegistryError(
            f"cannot load asset source registry {registry_path}: {exc}"
        ) from exc


def resolve_active_grscene(
    scene_id: str,
    *,
    assets_root: Optional[Path | str] = None,
) -> GRSceneSourcePaths:
    """Resolve one main-set GRScene and require its full physical source pack."""
    registry = load_asset_source_registry()
    source = registry.active_source
    if source.source_id != GRSCENES_SOURCE_ID:
        raise AssetSourceRegistryError(
            f"active source {source.source_id!r} has no implemented adapter"
        )
    if scene_id not in source.scene_ids:
        raise SceneCatalogError(
            f"GRScenes scene {scene_id!r} is not authorable in the current "
            f"{len(source.scene_ids)}-scene release"
        )

    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root
        else tarsphere_assets_root()
    )
    return _resolve_grscene(
        scene=source.require_scene(scene_id),
        root=root,
        source=source,
    )


def _resolve_grscene(
    *,
    scene: SceneSourceSpec,
    root: Path,
    source: AssetSourceSpec,
) -> GRSceneSourcePaths:
    scene_id = scene.scene_id
    scene_dir = root / GRSCENES_ROOT / scene.collection / "scenes" / scene_id
    meta_dir = root / "benchmark" / "meta" / scene_id
    for filename in source.required_scene_files:
        _require_file(scene_dir / filename, scene_id=scene_id)
    for filename in source.required_metadata_files:
        _require_file(meta_dir / filename, scene_id=scene_id)
    return GRSceneSourcePaths(
        assets_root=root,
        source=source,
        scene_id=scene_id,
        collection=scene.collection,
        scene_dir=scene_dir,
        meta_dir=meta_dir,
    )


def build_grscenes_catalog(
    *,
    assets_root: Optional[Path | str] = None,
) -> GRScenesCatalog:
    """Inventory the scenes pinned by the active source registry."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root
        else tarsphere_assets_root()
    )
    registry = load_asset_source_registry()
    source = registry.active_source
    if source.source_id != GRSCENES_SOURCE_ID:
        raise AssetSourceRegistryError(
            f"active source {source.source_id!r} has no implemented adapter"
        )

    authorable: List[SceneCatalogEntry] = []
    for scene in source.scenes:
        scene_id = scene.scene_id
        paths = _resolve_grscene(scene=scene, root=root, source=source)
        object_path = paths.meta_dir / "object_dict_with_caption.json"
        try:
            object_raw = json.loads(object_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise SceneCatalogError(
                f"cannot parse caption metadata for {scene_id}: {object_path}"
            ) from exc
        if not isinstance(object_raw, dict) or not object_raw:
            raise SceneCatalogError(
                f"caption metadata for {scene_id} must be a non-empty object mapping"
            )
        invalid = [
            object_id
            for object_id, raw in object_raw.items()
            if not isinstance(object_id, str) or not isinstance(raw, dict)
        ]
        if invalid:
            raise SceneCatalogError(
                f"caption metadata for {scene_id} has invalid object entries: "
                f"{invalid[:3]}"
            )
        captioned_count = sum(
            bool(str(raw.get("caption") or "").strip()) for raw in object_raw.values()
        )
        object_count = len(object_raw)
        source_map = paths.source_map()
        authorable.append(
            SceneCatalogEntry(
                source_id=source.source_id,
                collection=scene.collection,
                scene_id=scene_id,
                authorable=True,
                scene_asset_path=source_map["scene_usd"],
                metadata_sources=source_map,
                object_count=object_count,
                captioned_object_count=captioned_count,
                uncaptioned_object_count=object_count - captioned_count,
            )
        )

    active_keys = {(scene.collection, scene.scene_id) for scene in source.scenes}
    excluded: List[SceneCatalogEntry] = []
    for collection in ("commercial_scenes", "home_scenes"):
        scenes_dir = root / GRSCENES_ROOT / collection / "scenes"
        if not scenes_dir.is_dir():
            continue
        for scene_dir in sorted(path for path in scenes_dir.iterdir() if path.is_dir()):
            key = (collection, scene_dir.name)
            if key in active_keys:
                continue
            excluded.append(
                SceneCatalogEntry(
                    source_id=source.source_id,
                    collection=collection,
                    scene_id=scene_dir.name,
                    authorable=False,
                    exclusion_reason="not_in_active_source",
                )
            )

    return GRScenesCatalog(
        source_id=source.source_id,
        authorable_scenes=tuple(authorable),
        excluded_scenes=tuple(excluded),
    )


def _require_file(path: Path, *, scene_id: str) -> None:
    if not path.is_file():
        raise SceneCatalogError(
            f"pinned GRScenes scene {scene_id!r} is missing required file: {path}"
        )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise SceneCatalogError(
            f"asset source escaped configured root {root}: {path}"
        ) from exc


__all__ = [
    "AssetSourceRegistry",
    "AssetSourceRegistryError",
    "AssetSourceSpec",
    "CURRENT_GRSCENES_SCENE_COUNT",
    "GRSCENES_SOURCE_ID",
    "GRSceneSourcePaths",
    "GRScenesCatalog",
    "SceneCatalogEntry",
    "SceneCatalogError",
    "SceneSourceSpec",
    "SOURCE_REGISTRY_PATH",
    "build_grscenes_catalog",
    "load_asset_source_registry",
    "resolve_active_grscene",
]
