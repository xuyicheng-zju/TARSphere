"""GRScene metadata loader for scene-grounded episode authoring."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from tarsphere.bench.inputs.sources import resolve_active_grscene
from tarsphere.assets import tarsphere_assets_root


DEFAULT_SCENE_ID = "MV7J6NIKTKJZ2AABAAAAADY8_usd"
RUNTIME_SCENE_PRIM = "/World/env_0/scene"


def _float_list(value: Iterable[Any], *, length: int, field: str) -> List[float]:
    values = list(value)
    if len(values) < length:
        raise ValueError(f"{field} needs at least {length} values: {value!r}")
    try:
        return [float(item) for item in values[:length]]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} must be numeric: {value!r}") from exc


def point3(xy_or_xyz: Iterable[Any], *, z: float = 0.0) -> List[float]:
    values = list(xy_or_xyz)
    if len(values) >= 3:
        return _round_point(_float_list(values, length=3, field="point"))
    point = _float_list(values, length=2, field="point")
    point.append(float(z))
    return _round_point(point)


def _round_point(point: Iterable[float], digits: int = 6) -> List[float]:
    return [round(float(value), digits) for value in point]


@dataclass(frozen=True)
class SceneObject:
    """One object entry from GRScenes object metadata."""

    instance_id: str
    category: str
    room: str
    position: List[float]
    min_points: List[float]
    max_points: List[float]
    caption: str = ""
    source_prim_path: str = ""
    runtime_prim_path: str = ""
    interactive_kind: str = ""

    @classmethod
    def from_raw(
        cls,
        instance_id: str,
        raw: Dict[str, Any],
        *,
        source_prim_path: str = "",
        interactive_kind: str = "",
    ) -> "SceneObject":
        runtime_path = ""
        if source_prim_path:
            runtime_path = source_prim_path.replace("/Root", RUNTIME_SCENE_PRIM, 1)
        return cls(
            instance_id=instance_id,
            category=str(raw.get("category") or ""),
            room=str(raw.get("room") or ""),
            position=point3(raw.get("position") or []),
            min_points=point3(raw.get("min_points") or []),
            max_points=point3(raw.get("max_points") or []),
            caption=str(raw.get("caption") or ""),
            source_prim_path=source_prim_path,
            runtime_prim_path=runtime_path,
            interactive_kind=interactive_kind,
        )


@dataclass(frozen=True)
class Route:
    """One object path entry from GRScenes paths.json."""

    object_id: str
    start_point: List[float]
    target_point: List[float]
    distance: float
    path: List[List[float]]

    @classmethod
    def from_raw(cls, object_id: str, raw: Dict[str, Any]) -> "Route":
        path = [point3(item)[:2] for item in raw.get("path") or []]
        if not path:
            raise ValueError(f"paths.json route for {object_id!r} has an empty path.")
        return cls(
            object_id=object_id,
            start_point=point3(raw.get("start_point") or []),
            target_point=point3(raw.get("target_point") or []),
            distance=float(raw.get("distance") or 0.0),
            path=path,
        )


class GRSceneMetadata:
    """Load and query the GRScene files needed for episode authoring."""

    def __init__(
        self,
        *,
        scene_id: str,
        assets_root: Path,
        meta_dir: Path,
        scene_usd: Path,
        objects: Dict[str, SceneObject],
        paths: Dict[str, List[Route]],
    ):
        self.scene_id = scene_id
        self.assets_root = assets_root
        self.meta_dir = meta_dir
        self.scene_usd = scene_usd
        self.objects = objects
        self.paths = paths

    @classmethod
    def load(
        cls,
        scene_id: str = DEFAULT_SCENE_ID,
        *,
        assets_root: Optional[Path | str] = None,
    ) -> "GRSceneMetadata":
        """Load scene metadata.

        The authoring layer reads the full GRScenes asset tree directly.  This
        keeps provenance paths aligned with the benchmark assets and avoids a
        second scene-cache directory drifting away from the canonical data.
        ``benchmark/meta`` is always required.
        """
        root = (
            Path(assets_root).expanduser().resolve()
            if assets_root
            else tarsphere_assets_root()
        )
        source_paths = resolve_active_grscene(scene_id, assets_root=root)
        meta_dir = source_paths.meta_dir
        scene_usd = source_paths.scene_usd
        scene_asset_dir = source_paths.scene_dir

        object_raw = cls._read_json(meta_dir / "object_dict_with_caption.json")
        path_raw = cls._read_json(meta_dir / "paths.json")
        interactive_path = scene_asset_dir / "interactive_obj_list.json"
        if not interactive_path.is_file():
            raise FileNotFoundError(
                f"Interactive object list not found: {interactive_path}"
            )
        interactive_objects = cls._read_json(interactive_path)

        source_prim_paths = cls._load_source_prim_paths(scene_usd, object_raw.keys())
        interactive_kinds = cls._load_interactive_kinds(interactive_objects)

        objects = {
            object_id: SceneObject.from_raw(
                object_id,
                raw,
                source_prim_path=source_prim_paths.get(object_id, ""),
                interactive_kind=interactive_kinds.get(object_id, ""),
            )
            for object_id, raw in object_raw.items()
        }
        paths = {
            object_id: [Route.from_raw(object_id, item) for item in entries]
            for object_id, entries in path_raw.items()
        }
        return cls(
            scene_id=scene_id,
            assets_root=root,
            meta_dir=meta_dir,
            scene_usd=scene_usd,
            objects=objects,
            paths=paths,
        )

    @staticmethod
    def _read_json(path: Path) -> Any:
        if not path.is_file():
            raise FileNotFoundError(f"Required scene metadata file not found: {path}")
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _load_source_prim_paths(
        scene_usd: Path, object_ids: Iterable[str]
    ) -> Dict[str, str]:
        try:
            from pxr import Usd
        except Exception as exc:
            raise RuntimeError(
                "pxr.Usd is required for scene-grounded authoring."
            ) from exc

        stage = Usd.Stage.Open(str(scene_usd))
        if stage is None:
            raise RuntimeError(
                f"Could not open GRScene USD for prim verification: {scene_usd}"
            )
        wanted = set(object_ids)
        mapping: Dict[str, str] = {}
        for prim in stage.Traverse():
            path = str(prim.GetPath())
            if not path.startswith("/Root/Meshes/"):
                continue
            parts = path.split("/")
            if len(parts) < 2:
                continue
            object_id = f"{parts[-2]}/{parts[-1]}"
            if object_id in wanted and object_id not in mapping:
                mapping[object_id] = path
        return mapping

    @staticmethod
    def _load_interactive_kinds(raw: Dict[str, Any]) -> Dict[str, str]:
        kinds: Dict[str, str] = {}
        for kind in ("articulated", "pickable"):
            for prim_path in raw.get(kind) or []:
                parts = str(prim_path).split("/")
                if len(parts) < 2:
                    continue
                object_id = f"{parts[-2]}/{parts[-1]}"
                kinds[object_id] = kind
        return kinds

    def require_object(self, object_id: str) -> SceneObject:
        obj = self.objects.get(object_id)
        if obj is None:
            raise KeyError(f"Scene object not found in metadata: {object_id}")
        return obj
