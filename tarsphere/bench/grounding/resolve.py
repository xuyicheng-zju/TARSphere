"""Ground current-release scene assets into minimal deterministic facts."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from tarsphere.bench.grounding.schema import (
    GroundedObject,
    GroundedScene,
    ObjectApproachPath,
    SceneMap,
    SceneRoom,
    SceneSources,
)
from tarsphere.bench.inputs.scenes import GRSceneMetadata
from tarsphere.bench.inputs.sources import (
    GRSceneSourcePaths,
    resolve_active_grscene,
)
from tarsphere.assets import tarsphere_assets_root
from tarsphere.scene_graph.room_references import parse_scene_room_roster


def resolve_scene(
    scene_id: str,
    *,
    assets_root: Optional[Path | str] = None,
) -> GroundedScene:
    """Ground one scene without introducing authoring task semantics."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root
        else tarsphere_assets_root()
    )
    paths = resolve_active_grscene(scene_id, assets_root=root)
    metadata = GRSceneMetadata.load(scene_id, assets_root=root)

    annotation_path = paths.scene_dir / "room_annotations.json"
    projection_path = paths.scene_dir / "bev_meta.json"
    rgb_bev_path = paths.scene_dir / "bev_rgb.png"
    room_bev_path = paths.scene_dir / "bev_rooms.png"
    for required in (
        annotation_path,
        projection_path,
        rgb_bev_path,
        room_bev_path,
    ):
        if not required.is_file():
            raise FileNotFoundError(
                f"Grounded scene input is missing for {scene_id!r}: {required}"
            )

    annotations = _read_mapping(annotation_path)
    projection = _read_mapping(projection_path)
    rooms, room_ids = _build_rooms(scene_id, annotations)
    scene_map = _build_map(
        scene_id=scene_id,
        root=root,
        paths=paths,
        annotations=annotations,
        projection=projection,
    )
    objects = _build_objects(metadata, room_ids)
    approach_paths = _build_approach_paths(metadata)
    source_map = paths.source_map()

    return GroundedScene(
        source_id=paths.source.source_id,
        scene_id=scene_id,
        scene_scale=_scene_scale(paths.scene_usd),
        sources=SceneSources(
            scene_usd=source_map["scene_usd"],
            object_metadata=source_map["object_dict_with_caption"],
            object_paths=source_map["paths"],
            interactive_objects=source_map["interactive_objects"],
        ),
        map=scene_map,
        rooms=rooms,
        objects=objects,
        object_approach_paths=approach_paths,
    )


def _scene_scale(scene_usd: Path) -> tuple[float, float, float]:
    """Read the scale that maps authored scene coordinates into metres."""
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError(
            "pxr.Usd is required to resolve the scene unit contract"
        ) from exc
    stage = Usd.Stage.Open(str(scene_usd), load=Usd.Stage.LoadNone)
    if stage is None:
        raise RuntimeError(f"Could not open scene USD for units: {scene_usd}")
    unit_in_meters = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not math.isfinite(unit_in_meters) or unit_in_meters <= 0.0:
        raise ValueError(
            f"Scene USD declares invalid metres per unit: "
            f"{scene_usd} -> {unit_in_meters!r}"
        )
    return (unit_in_meters, unit_in_meters, unit_in_meters)


def _build_rooms(
    scene_id: str,
    annotations: Mapping[str, Any],
) -> tuple[tuple[SceneRoom, ...], dict[str, str]]:
    roster = parse_scene_room_roster(
        annotations,
        expected_scene_id=scene_id,
    )
    raw_rooms = annotations.get("rooms")
    if not isinstance(raw_rooms, list) or not raw_rooms:
        raise ValueError(f"Room annotations contain no rooms for {scene_id!r}")

    rooms: list[SceneRoom] = []
    source_to_room: dict[str, str] = {}
    room_ids: set[str] = set()
    for index, raw in enumerate(raw_rooms):
        if not isinstance(raw, Mapping):
            raise TypeError(f"Room annotation #{index} must be a mapping")
        reference = roster.rooms[index]
        room_id = reference.display_id
        if room_id in room_ids:
            raise ValueError(f"Duplicate room id in {scene_id!r}: {room_id!r}")
        room_ids.add(room_id)

        source_room_id = reference.metadata_room_id
        if source_room_id:
            if source_room_id in source_to_room:
                raise ValueError(
                    f"Duplicate source room id in {scene_id!r}: {source_room_id!r}"
                )
            source_to_room[source_room_id] = room_id

        rooms.append(
            SceneRoom(
                room_id=room_id,
                label=reference.display_label,
                room_type=reference.room_type,
                source_room_id=source_room_id,
                polygon_xy=_point_sequence(
                    raw.get("polygon_world_xy"),
                    length=2,
                    field=f"rooms[{index}].polygon_world_xy",
                    minimum=3,
                ),
                label_xy=_point(
                    raw.get("label_world_xy"),
                    length=2,
                    field=f"rooms[{index}].label_world_xy",
                ),
                area_m2=float(raw.get("polygon_area_m2")),
            )
        )
    return tuple(rooms), source_to_room


def _build_map(
    *,
    scene_id: str,
    root: Path,
    paths: GRSceneSourcePaths,
    annotations: Mapping[str, Any],
    projection: Mapping[str, Any],
) -> SceneMap:
    if projection.get("scene_id") != scene_id:
        raise ValueError(
            f"BEV projection identifies {projection.get('scene_id')!r}, "
            f"expected {scene_id!r}"
        )
    resolution = int(projection.get("resolution") or 0)
    bounds_min = _point(
        projection.get("bev_bounds_min_xy"),
        length=2,
        field="bev_meta.bev_bounds_min_xy",
    )
    bounds_max = _point(
        projection.get("bev_bounds_max_xy"),
        length=2,
        field="bev_meta.bev_bounds_max_xy",
    )

    coordinate_contract = annotations.get("coordinate_contract")
    if not isinstance(coordinate_contract, Mapping):
        raise TypeError("room_annotations.coordinate_contract must be a mapping")
    annotation_resolution = int(coordinate_contract.get("bev_resolution") or 0)
    annotation_min = _point(
        coordinate_contract.get("bev_bounds_min_xy"),
        length=2,
        field="room_annotations.coordinate_contract.bev_bounds_min_xy",
    )
    annotation_max = _point(
        coordinate_contract.get("bev_bounds_max_xy"),
        length=2,
        field="room_annotations.coordinate_contract.bev_bounds_max_xy",
    )
    if (
        resolution != annotation_resolution
        or bounds_min != annotation_min
        or bounds_max != annotation_max
    ):
        raise ValueError(f"RGB and room BEV projections disagree for {scene_id!r}")

    source_map = paths.source_map()
    return SceneMap(
        rgb_bev=_relative(paths.scene_dir / "bev_rgb.png", root),
        room_bev=_relative(paths.scene_dir / "bev_rooms.png", root),
        projection=_relative(paths.scene_dir / "bev_meta.json", root),
        room_annotations=_relative(
            paths.scene_dir / "room_annotations.json",
            root,
        ),
        occupancy=source_map["occupancy_npy"],
        resolution=resolution,
        bounds_min_xy=bounds_min,
        bounds_max_xy=bounds_max,
    )


def _build_objects(
    metadata: GRSceneMetadata,
    room_ids: Mapping[str, str],
) -> tuple[GroundedObject, ...]:
    grounded: list[GroundedObject] = []
    for object_id in sorted(metadata.objects):
        obj = metadata.objects[object_id]
        room_id = room_ids.get(obj.room)
        if not room_id:
            raise ValueError(
                f"Object {object_id!r} references unknown room {obj.room!r}"
            )
        grounded.append(
            GroundedObject(
                object_id=object_id,
                category=obj.category,
                room_id=room_id,
                position_xyz=tuple(obj.position),
                bounds_min_xyz=tuple(obj.min_points),
                bounds_max_xyz=tuple(obj.max_points),
                caption=_optional_text(obj.caption),
                source_prim=_optional_text(obj.source_prim_path),
                runtime_prim=_optional_text(obj.runtime_prim_path),
                interaction=_optional_text(obj.interactive_kind),
            )
        )
    return tuple(grounded)


def _build_approach_paths(
    metadata: GRSceneMetadata,
) -> tuple[ObjectApproachPath, ...]:
    grounded: list[ObjectApproachPath] = []
    for object_id in sorted(metadata.paths):
        obj = metadata.require_object(object_id)
        for path_index, route in enumerate(metadata.paths[object_id]):
            if len(route.path) < 2:
                raise ValueError(
                    "Object approach path needs at least two waypoints to "
                    f"define an initial heading: {object_id!r}[{path_index}]"
                )
            target_xy = tuple(route.target_point[:2])
            if math.dist(target_xy, tuple(obj.position[:2])) > 1e-4:
                raise ValueError(
                    f"Object approach path target disagrees with object position: "
                    f"{object_id!r}[{path_index}]"
                )
            grounded.append(
                ObjectApproachPath(
                    object_id=object_id,
                    path_index=path_index,
                    start_xy=tuple(route.start_point[:2]),
                    initial_heading_rad=math.atan2(
                        float(route.path[1][1]) - float(route.path[0][1]),
                        float(route.path[1][0]) - float(route.path[0][0]),
                    ),
                    approach_xy=tuple(route.path[-1][:2]),
                    distance_m=route.distance,
                    waypoint_count=len(route.path),
                )
            )
    return tuple(grounded)


def _read_mapping(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Cannot read grounded scene input: {path}") from exc
    if not isinstance(value, Mapping):
        raise TypeError(f"Grounded scene input must be a mapping: {path}")
    return value


def _required_text(raw: Mapping[str, Any], field: str) -> str:
    value = _optional_text(raw.get(field))
    if not value:
        raise ValueError(f"{field} must be non-empty")
    return value


def _optional_text(value: Any) -> Optional[str]:
    text = str(value or "").strip()
    return text or None


def _point(
    value: Any,
    *,
    length: int,
    field: str,
) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a numeric sequence")
    if len(value) != length:
        raise ValueError(f"{field} must contain exactly {length} values")
    point = tuple(float(item) for item in value)
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{field} must contain finite values")
    return point


def _point_sequence(
    value: Any,
    *,
    length: int,
    field: str,
    minimum: int,
) -> tuple[tuple[float, ...], ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise TypeError(f"{field} must be a sequence")
    if len(value) < minimum:
        raise ValueError(f"{field} must contain at least {minimum} points")
    return tuple(
        _point(point, length=length, field=f"{field}[{index}]")
        for index, point in enumerate(value)
    )


def _relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Grounded scene source escaped configured root {root}: {path}"
        ) from exc


__all__ = ["resolve_scene"]
