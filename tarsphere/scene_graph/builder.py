"""Build a full scene-level graph from GRScene metadata and USD.

This graph is an oracle scene asset for dataset authoring and debugging.  It
is not an evaluated robot observation path.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from tarsphere.bench.inputs.scenes import DEFAULT_SCENE_ID
from tarsphere.assets import tarsphere_assets_root


SCHEMA = "tarsphere.scenegraph.full.v1"
RUNTIME_SCENE_PRIM = "/World/env_0/scene"
STRUCTURE_CATEGORIES = {"wall", "ground", "ceiling", "column", "threshold"}
SEMANTIC_USD_BBOX_WARN_TOLERANCE_M = 0.08


class SceneGraphBuildError(RuntimeError):
    """Raised when required scene graph coverage is incomplete."""


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required scene graph source file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _rel(path: Path, root: Path) -> str:
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError:
        return str(path.resolve())


def _point3(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        raise ValueError(f"{field} must be a 3D point, got {value!r}")
    point = [float(value[0]), float(value[1]), float(value[2])]
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{field} contains non-finite values: {value!r}")
    return point


def _point2(value: Any, *, field: str) -> list[float]:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        raise ValueError(f"{field} must be a 2D point, got {value!r}")
    point = [float(value[0]), float(value[1])]
    if not all(math.isfinite(item) for item in point):
        raise ValueError(f"{field} contains non-finite values: {value!r}")
    return point


def _bbox_payload(lo: Any, hi: Any, *, field: str) -> dict[str, Any]:
    bmin = np.asarray(_point3(lo, field=f"{field}.min"), dtype=float)
    bmax = np.asarray(_point3(hi, field=f"{field}.max"), dtype=float)
    lo_arr = np.minimum(bmin, bmax)
    hi_arr = np.maximum(bmin, bmax)
    size = hi_arr - lo_arr
    if not np.all(np.isfinite(size)):
        raise ValueError(f"{field} has non-finite bbox size.")
    return {
        "min": [round(float(v), 6) for v in lo_arr],
        "max": [round(float(v), 6) for v in hi_arr],
        "center": [round(float(v), 6) for v in ((lo_arr + hi_arr) * 0.5)],
        "size": [round(float(v), 6) for v in size],
        "area_xy_m2": round(float(max(size[0], 0.0) * max(size[1], 0.0)), 6),
    }


def _bbox_from_object(raw: dict[str, Any], *, object_id: str) -> dict[str, Any]:
    return _bbox_payload(
        raw.get("min_points"),
        raw.get("max_points"),
        field=f"object_dict_with_caption.{object_id}",
    )


def _bbox_from_arrays(lo: np.ndarray, hi: np.ndarray, *, field: str) -> dict[str, Any]:
    return _bbox_payload(lo.tolist(), hi.tolist(), field=field)


def _bbox_diff_m(lhs: dict[str, Any], rhs: dict[str, Any]) -> float:
    lhs_min = np.asarray(lhs["min"], dtype=float)
    lhs_max = np.asarray(lhs["max"], dtype=float)
    rhs_min = np.asarray(rhs["min"], dtype=float)
    rhs_max = np.asarray(rhs["max"], dtype=float)
    return float(max(np.abs(lhs_min - rhs_min).max(), np.abs(lhs_max - rhs_max).max()))


def _locate_scene_assets(root: Path, scene_id: str) -> tuple[Path, Path]:
    grscenes_root = root / "scenes" / "GRScenes-100"
    candidates = [
        item
        for item in sorted(grscenes_root.glob(f"*/scenes/{scene_id}"))
        if (item / "start_result_navigation.usd").is_file()
    ]
    if not candidates:
        raise FileNotFoundError(
            f"GRScene USD not found for scene {scene_id!r}; looked under {grscenes_root}"
        )
    if len(candidates) > 1:
        raise SceneGraphBuildError(
            f"Ambiguous GRScene asset directories for scene {scene_id!r}: "
            + ", ".join(str(item) for item in candidates)
        )
    asset_dir = candidates[0]
    return asset_dir / "start_result_navigation.usd", asset_dir


def _runtime_prim_path(source_prim_path: str) -> str:
    if source_prim_path.startswith("/Root"):
        return source_prim_path.replace("/Root", RUNTIME_SCENE_PRIM, 1)
    return ""


def _model_id_from_prim_path(path: str) -> str | None:
    parts = path.split("/")
    if len(parts) < 6 or not path.startswith("/Root/Meshes/"):
        return None
    return f"{parts[-2]}/{parts[-1]}"


def _scope_from_prim_path(path: str) -> str:
    parts = path.split("/")
    if len(parts) >= 6 and parts[1:3] == ["Root", "Meshes"]:
        return parts[3]
    return ""


def _load_model_prim_paths(stage: Any, model_ids: set[str]) -> dict[str, str]:
    found: dict[str, str] = {}
    for prim in stage.TraverseAll():
        path = str(prim.GetPath())
        model_id = _model_id_from_prim_path(path)
        if model_id in model_ids and model_id not in found:
            found[model_id] = path
    return found


def _range_bounds(
    cache: Any, prim: Any, meters_per_unit: float
) -> tuple[np.ndarray, np.ndarray] | None:
    try:
        aligned = cache.ComputeWorldBound(prim).ComputeAlignedBox()
        mesh_range = aligned.GetRange() if hasattr(aligned, "GetRange") else aligned
        if hasattr(mesh_range, "IsEmpty") and mesh_range.IsEmpty():
            return None
        lo = np.asarray(mesh_range.GetMin(), dtype=float) * meters_per_unit
        hi = np.asarray(mesh_range.GetMax(), dtype=float) * meters_per_unit
    except Exception:
        return None
    if not np.all(np.isfinite(lo)) or not np.all(np.isfinite(hi)):
        return None
    if np.any((np.maximum(lo, hi) - np.minimum(lo, hi)) > 0):
        return np.minimum(lo, hi), np.maximum(lo, hi)
    return None


def _compute_usd_bounds(
    *,
    stage: Any,
    prim_path: str,
    bbox_cache: Any,
    meters_per_unit: float,
) -> dict[str, Any]:
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim:
        raise SceneGraphBuildError(
            f"USD prim disappeared while building scene graph: {prim_path}"
        )

    mins: list[np.ndarray] = []
    maxs: list[np.ndarray] = []
    mesh_count = 0
    bounded_mesh_count = 0
    for child in Usd.PrimRange(prim):
        if not child.IsA(UsdGeom.Mesh):
            continue
        mesh_count += 1
        bounds = _range_bounds(bbox_cache, child, meters_per_unit)
        if bounds is None:
            continue
        lo, hi = bounds
        mins.append(lo)
        maxs.append(hi)
        bounded_mesh_count += 1

    if mins:
        lo_arr = np.min(np.stack(mins, axis=0), axis=0)
        hi_arr = np.max(np.stack(maxs, axis=0), axis=0)
        return {
            "bbox": _bbox_from_arrays(lo_arr, hi_arr, field=f"usd.{prim_path}"),
            "bbox_source": "usd_descendant_mesh_bounds",
            "mesh_count": mesh_count,
            "bounded_mesh_count": bounded_mesh_count,
        }

    root_bounds = _range_bounds(bbox_cache, prim, meters_per_unit)
    if root_bounds is not None:
        lo_arr, hi_arr = root_bounds
        return {
            "bbox": _bbox_from_arrays(lo_arr, hi_arr, field=f"usd.{prim_path}"),
            "bbox_source": "usd_root_bounds",
            "mesh_count": mesh_count,
            "bounded_mesh_count": bounded_mesh_count,
        }

    return {
        "bbox": None,
        "bbox_source": "",
        "mesh_count": mesh_count,
        "bounded_mesh_count": bounded_mesh_count,
    }


def _load_interactive_kinds(
    raw: dict[str, Any], model_ids: set[str]
) -> dict[str, list[str]]:
    kinds: dict[str, list[str]] = {}
    fatal: list[str] = []
    for kind in ("articulated", "pickable"):
        entries = raw.get(kind) or []
        if not isinstance(entries, list):
            fatal.append(f"interactive_obj_list.{kind} must be a list.")
            continue
        for prim_path in entries:
            model_id = _model_id_from_prim_path(str(prim_path))
            if model_id not in model_ids:
                fatal.append(
                    f"interactive {kind} path does not map to model_mapping: {prim_path}"
                )
                continue
            kinds.setdefault(model_id, [])
            if kind not in kinds[model_id]:
                kinds[model_id].append(kind)
    if fatal:
        raise SceneGraphBuildError("\n".join(fatal))
    return kinds


def _route_payloads(
    raw: dict[str, Any], *, include_waypoints: bool
) -> dict[str, list[dict[str, Any]]]:
    routes: dict[str, list[dict[str, Any]]] = {}
    for object_id, entries in raw.items():
        if not isinstance(entries, list) or not entries:
            raise ValueError(
                f"paths.json route list for {object_id!r} must be non-empty."
            )
        route_list = []
        for index, entry in enumerate(entries):
            if not isinstance(entry, dict):
                raise ValueError(
                    f"paths.json route {object_id}[{index}] must be a mapping."
                )
            path = entry.get("path") or []
            if not isinstance(path, list) or not path:
                raise ValueError(
                    f"paths.json route {object_id}[{index}].path must be non-empty."
                )
            payload = {
                "route_index": index,
                "start_point": _point2(
                    entry.get("start_point"),
                    field=f"paths.{object_id}[{index}].start_point",
                ),
                "target_point": _point3(
                    entry.get("target_point"),
                    field=f"paths.{object_id}[{index}].target_point",
                ),
                "distance_m": round(float(entry.get("distance") or 0.0), 6),
                "waypoint_count": len(path),
            }
            if include_waypoints:
                payload["path"] = [
                    _point2(point, field=f"paths.{object_id}[{index}].path")
                    for point in path
                ]
            route_list.append(payload)
        routes[object_id] = route_list
    return routes


def _room_nodes(object_raw: dict[str, Any]) -> list[dict[str, Any]]:
    rooms: dict[str, list[tuple[str, dict[str, Any]]]] = {}
    for object_id, raw in object_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"object_dict_with_caption.{object_id} must be a mapping.")
        room = str(raw.get("room") or "").strip()
        if not room:
            raise ValueError(f"object_dict_with_caption.{object_id}.room is empty.")
        rooms.setdefault(room, []).append((object_id, raw))

    def sort_key(room_id: str) -> tuple[int, str]:
        prefix = room_id.split("/", 1)[0]
        return (int(prefix) if prefix.isdigit() else 10_000, room_id)

    nodes: list[dict[str, Any]] = []
    for room_id in sorted(rooms, key=sort_key):
        entries = rooms[room_id]
        bboxes = [
            _bbox_from_object(raw, object_id=object_id) for object_id, raw in entries
        ]
        lo = np.min(np.asarray([bbox["min"] for bbox in bboxes], dtype=float), axis=0)
        hi = np.max(np.asarray([bbox["max"] for bbox in bboxes], dtype=float), axis=0)
        categories = sorted(
            {
                str(raw.get("category") or "")
                for _, raw in entries
                if raw.get("category")
            }
        )
        room_type = room_id.split("/", 1)[-1]
        nodes.append(
            {
                "id": f"room:{room_id}",
                "room_id": room_id,
                "room_type": room_type,
                "bbox": _bbox_from_arrays(
                    lo, hi, field=f"room.{room_id}.object_bbox_union"
                ),
                "geometry_source": "object_dict_with_caption.room + semantic object bbox union",
                "semantic_object_count": len(entries),
                "semantic_object_ids": [
                    f"model:{object_id}" for object_id, _raw in entries
                ],
                "categories": categories,
            }
        )
    return nodes


def default_scenegraph_output_path(scenegraph: dict[str, Any]) -> Path:
    return (
        Path(scenegraph["source_files"]["scene_usd_abs"]).parent
        / "scenegraph_full.json"
    )


def build_scenegraph(
    *,
    scene_id: str = DEFAULT_SCENE_ID,
    assets_root: str | Path | None = None,
    include_route_waypoints: bool = True,
    semantic_bbox_tolerance_m: float = SEMANTIC_USD_BBOX_WARN_TOLERANCE_M,
) -> dict[str, Any]:
    """Build the full GRScene graph for one scene id."""
    try:
        from pxr import Usd, UsdGeom
    except Exception as exc:
        raise RuntimeError(
            "pxr.Usd/UsdGeom are required to build the full scene graph."
        ) from exc

    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root
        else tarsphere_assets_root()
    )
    scene_usd, scene_asset_dir = _locate_scene_assets(root, scene_id)
    meta_dir = root / "benchmark" / "meta" / scene_id
    if not meta_dir.is_dir():
        raise FileNotFoundError(f"GRScene metadata directory not found: {meta_dir}")

    object_raw = _read_json(meta_dir / "object_dict_with_caption.json")
    object_plain = _read_json(meta_dir / "object_dict.json")
    model_mapping = _read_json(meta_dir / "model_mapping.json")
    path_raw = _read_json(meta_dir / "paths.json")
    annotations = _read_json(meta_dir / "annotation.json")
    interactive_raw = _read_json(scene_asset_dir / "interactive_obj_list.json")

    if not all(
        isinstance(item, dict)
        for item in (object_raw, object_plain, model_mapping, path_raw, interactive_raw)
    ):
        raise ValueError(
            "Scene graph JSON sources must be mappings except annotation.json."
        )
    if not isinstance(annotations, list):
        raise ValueError("annotation.json must be a list of room annotations.")

    object_ids = set(object_raw)
    model_ids = set(model_mapping)
    fatal: list[str] = []
    warnings: list[str] = []

    if set(object_plain) != object_ids:
        fatal.append(
            "object_dict.json and object_dict_with_caption.json have different object ids."
        )
    object_without_model = sorted(object_ids - model_ids)
    if object_without_model:
        fatal.append(
            f"{len(object_without_model)} semantic objects are missing from model_mapping.json."
        )

    stage = Usd.Stage.Open(str(scene_usd))
    if stage is None:
        raise RuntimeError(f"Could not open GRScene USD: {scene_usd}")
    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage) or 1.0)
    purposes = [
        token
        for token in (
            getattr(UsdGeom.Tokens, "default_", None),
            getattr(UsdGeom.Tokens, "render", None),
            getattr(UsdGeom.Tokens, "proxy", None),
        )
        if token is not None
    ]
    bbox_cache = UsdGeom.BBoxCache(0.0, purposes or ["default"], True)
    prim_paths = _load_model_prim_paths(stage, model_ids)
    missing_prims = sorted(model_ids - set(prim_paths))
    if missing_prims:
        fatal.append(
            f"{len(missing_prims)} model_mapping entries have no USD root prim."
        )

    route_object_ids = set(path_raw)
    routes_without_object = sorted(route_object_ids - object_ids)
    if routes_without_object:
        fatal.append(
            f"{len(routes_without_object)} paths.json entries are not semantic objects."
        )

    if fatal:
        raise SceneGraphBuildError("\n".join(fatal))

    interactive_kinds = _load_interactive_kinds(interactive_raw, model_ids)
    routes = _route_payloads(path_raw, include_waypoints=include_route_waypoints)
    source_files = {
        "scene_usd": _rel(scene_usd, root),
        "scene_usd_abs": str(scene_usd.resolve()),
        "object_dict_with_caption": _rel(
            meta_dir / "object_dict_with_caption.json", root
        ),
        "object_dict": _rel(meta_dir / "object_dict.json", root),
        "model_mapping": _rel(meta_dir / "model_mapping.json", root),
        "paths": _rel(meta_dir / "paths.json", root),
        "annotation": _rel(meta_dir / "annotation.json", root),
        "occupancy": _rel(meta_dir / "occupancy.npy", root),
        "rgb_init": _rel(meta_dir / "rgb_init.png", root),
        "interactive_objects": _rel(
            scene_asset_dir / "interactive_obj_list.json", root
        ),
    }

    semantic_usd_bounds_missing: list[str] = []
    semantic_bbox_mismatches: list[dict[str, Any]] = []
    nonsemantic_bounds_missing: list[str] = []
    nodes: list[dict[str, Any]] = []

    for model_id in sorted(model_mapping):
        category = model_id.split("/", 1)[0]
        prim_path = prim_paths[model_id]
        usd_bounds = _compute_usd_bounds(
            stage=stage,
            prim_path=prim_path,
            bbox_cache=bbox_cache,
            meters_per_unit=meters_per_unit,
        )
        raw = object_raw.get(model_id)
        semantic = raw is not None
        if semantic:
            semantic_bbox = _bbox_from_object(raw, object_id=model_id)
            bbox = semantic_bbox
            bbox_source = "object_dict_with_caption.json"
            position = _point3(
                raw.get("position"),
                field=f"object_dict_with_caption.{model_id}.position",
            )
            if usd_bounds["bbox"] is None:
                semantic_usd_bounds_missing.append(model_id)
            else:
                diff = _bbox_diff_m(semantic_bbox, usd_bounds["bbox"])
                if diff > semantic_bbox_tolerance_m:
                    semantic_bbox_mismatches.append(
                        {
                            "object_id": model_id,
                            "category": category,
                            "max_abs_diff_m": round(diff, 6),
                        }
                    )
        else:
            if usd_bounds["bbox"] is None:
                nonsemantic_bounds_missing.append(model_id)
                bbox = None
                position = [0.0, 0.0, 0.0]
            else:
                bbox = usd_bounds["bbox"]
                position = bbox["center"]
            bbox_source = usd_bounds["bbox_source"]

        kind = (
            "semantic_object"
            if semantic
            else (
                "structure_model"
                if category in STRUCTURE_CATEGORIES
                else "unlabeled_model"
            )
        )
        node_interactive_kinds = interactive_kinds.get(model_id, [])
        node: dict[str, Any] = {
            "id": f"model:{model_id}",
            "model_id": model_id,
            "kind": kind,
            "category": category,
            "scope": _scope_from_prim_path(prim_path),
            "model_asset": model_mapping[model_id],
            "position": [round(float(v), 6) for v in position],
            "bbox": bbox,
            "bbox_source": bbox_source,
            "source_prim_path": prim_path,
            "runtime_prim_path": _runtime_prim_path(prim_path),
            "usd_bounds": usd_bounds,
            "semantic": semantic,
            "interactive_kinds": node_interactive_kinds,
            "routable": model_id in routes,
            "routes": routes.get(model_id, []),
        }
        if semantic:
            node.update(
                {
                    "room": str(raw.get("room") or ""),
                    "caption": str(raw.get("caption") or ""),
                    "nearby_objects": raw.get("nearby_objects") or {},
                    "semantic_scope": str(raw.get("scope") or ""),
                    "sources": {
                        "object_metadata": source_files["object_dict_with_caption"],
                        "model_mapping": source_files["model_mapping"],
                        "usd_prim": source_files["scene_usd"],
                    },
                }
            )
        else:
            node.update(
                {
                    "room": "",
                    "caption": "",
                    "nearby_objects": {},
                    "sources": {
                        "model_mapping": source_files["model_mapping"],
                        "usd_prim": source_files["scene_usd"],
                    },
                }
            )
        nodes.append(node)

    if nonsemantic_bounds_missing:
        raise SceneGraphBuildError(
            f"{len(nonsemantic_bounds_missing)} non-semantic model nodes have no USD bounds: "
            + ", ".join(nonsemantic_bounds_missing[:20])
        )

    rooms = _room_nodes(object_raw)
    room_ids = {room["id"] for room in rooms}
    node_ids = {node["id"] for node in nodes}
    relations: list[dict[str, Any]] = []
    missing_nearby_targets: list[str] = []

    for object_id, raw in sorted(object_raw.items()):
        subject = f"model:{object_id}"
        room_id = f"room:{raw.get('room')}"
        if room_id not in room_ids:
            raise SceneGraphBuildError(
                f"{object_id} references missing room node {room_id}."
            )
        relations.append(
            {
                "type": "in_room",
                "subject": subject,
                "object": room_id,
                "source": "object_dict_with_caption.room",
            }
        )
        nearby = raw.get("nearby_objects") or {}
        if not isinstance(nearby, dict):
            raise ValueError(
                f"object_dict_with_caption.{object_id}.nearby_objects must be a mapping."
            )
        for target_id, relation_raw in sorted(nearby.items()):
            target_node = f"model:{target_id}"
            if target_node not in node_ids:
                missing_nearby_targets.append(f"{object_id} -> {target_id}")
                continue
            if (
                not isinstance(relation_raw, (list, tuple))
                or len(relation_raw) < 1
                or not str(relation_raw[0]).strip()
            ):
                raise ValueError(
                    f"Invalid nearby relation for object_dict_with_caption.{object_id}: {target_id}={relation_raw!r}"
                )
            relation = {
                "type": str(relation_raw[0]),
                "subject": subject,
                "object": target_node,
                "source": "object_dict_with_caption.nearby_objects",
            }
            if len(relation_raw) >= 2:
                relation["distance_m"] = round(float(relation_raw[1]), 6)
            relations.append(relation)

    if missing_nearby_targets:
        raise SceneGraphBuildError(
            f"{len(missing_nearby_targets)} nearby_objects references are missing graph nodes: "
            + ", ".join(missing_nearby_targets[:20])
        )

    if semantic_usd_bounds_missing:
        warnings.append(
            f"{len(semantic_usd_bounds_missing)} semantic objects have object metadata bbox but no USD mesh bounds."
        )
    if semantic_bbox_mismatches:
        warnings.append(
            f"{len(semantic_bbox_mismatches)} semantic object bboxes differ from USD mesh bounds by "
            f"more than {semantic_bbox_tolerance_m:.2f} m."
        )

    semantic_count = len(object_ids)
    structure_count = sum(1 for node in nodes if node["kind"] == "structure_model")
    unlabeled_count = sum(1 for node in nodes if node["kind"] == "unlabeled_model")
    multi_interactive = {
        model_id: kinds
        for model_id, kinds in sorted(interactive_kinds.items())
        if len(kinds) > 1
    }
    graph = {
        "schema": SCHEMA,
        "scene_id": scene_id,
        "oracle_asset": True,
        "asset_scope": "scene_level_static_asset",
        "excluded_entities": ["robot", "human_actor_pose"],
        "source_files": source_files,
        "provenance": {
            "semantic_objects": "object_dict_with_caption.json",
            "all_model_roots": "model_mapping.json + verified USD root prim paths",
            "structure_models": "model_mapping entries not present in object_dict_with_caption.json",
            "room_nodes": "object_dict_with_caption.room grouped by semantic object bbox union",
            "spatial_relations": "object_dict_with_caption.nearby_objects and room membership",
            "navigation_routes": "paths.json",
            "interactive_kind": "interactive_obj_list.json",
            "raw_room_annotations": (
                "annotation.json is recorded as a source but not used as primary room geometry "
                "because current GRScene room polygons need separate validation against object metadata."
            ),
        },
        "summary": {
            "node_count": len(nodes),
            "model_mapping_count": len(model_mapping),
            "semantic_object_count": semantic_count,
            "structure_model_count": structure_count,
            "unlabeled_model_count": unlabeled_count,
            "room_count": len(rooms),
            "route_target_count": len(routes),
            "interactive_object_count": len(interactive_kinds),
            "multi_interactive_object_count": len(multi_interactive),
            "relation_count": len(relations),
            "annotation_room_count": len(annotations),
        },
        "quality_report": {
            "fatal_errors": [],
            "warnings": warnings,
            "coverage": {
                "object_dict_keys_match_caption_version": set(object_plain)
                == object_ids,
                "semantic_objects_missing_model_mapping": object_without_model,
                "model_mapping_missing_usd_prim": missing_prims,
                "paths_missing_semantic_object": routes_without_object,
                "all_model_mapping_entries_have_usd_prim": not missing_prims,
                "all_semantic_objects_have_model_mapping": not object_without_model,
                "all_paths_reference_semantic_objects": not routes_without_object,
            },
            "usd_bounds": {
                "stage_meters_per_unit": meters_per_unit,
                "semantic_usd_bounds_missing_count": len(semantic_usd_bounds_missing),
                "semantic_usd_bounds_missing": semantic_usd_bounds_missing,
                "semantic_bbox_mismatch_tolerance_m": semantic_bbox_tolerance_m,
                "semantic_bbox_mismatch_count": len(semantic_bbox_mismatches),
                "semantic_bbox_mismatches": sorted(
                    semantic_bbox_mismatches,
                    key=lambda item: item["max_abs_diff_m"],
                    reverse=True,
                )[:50],
            },
            "interactive_objects": {
                "multi_kind_objects": multi_interactive,
            },
        },
        "rooms": rooms,
        "nodes": nodes,
        "relations": relations,
        "raw_room_annotations": {
            "source": source_files["annotation"],
            "used_for_primary_room_geometry": False,
            "count": len(annotations),
            "room_types": [
                str(item.get("room_type") or "")
                for item in annotations
                if isinstance(item, dict)
            ],
        },
    }
    if graph["summary"]["node_count"] != graph["summary"]["model_mapping_count"]:
        raise SceneGraphBuildError(
            "Internal graph count mismatch: node_count must equal model_mapping_count."
        )
    return graph


def write_scenegraph(
    scenegraph: dict[str, Any], output_path: str | Path | None = None
) -> Path:
    path = (
        Path(output_path).expanduser().resolve()
        if output_path
        else default_scenegraph_output_path(scenegraph)
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(scenegraph, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if not path.is_file():
        raise RuntimeError(f"Scene graph writer did not create expected file: {path}")
    return path
