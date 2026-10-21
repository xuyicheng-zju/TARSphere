"""Compile offline room-annotation artifacts for authoring and review.

Room identities come from released annotations. Traversable room entrances are
derived offline from those polygons and official navigation occupancy. These
artifacts support benchmark authoring and visual review; they are not exposed
to the evaluated Agent and are not the runtime online semantic map.
"""

from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import distance_transform_edt

from tarsphere.assets import tarsphere_assets_root
from tarsphere.bench.inputs.sources import (
    load_asset_source_registry,
    resolve_active_grscene,
)
from tarsphere.scene_graph.room_references import ROOM_ANNOTATION_SCHEMA
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


ROOM_ANNOTATION_MANIFEST_SCHEMA = "tarsphere.scene_room_annotation_batch.v1"
ROOM_MAP_FILENAME = "bev_rooms.png"
ROOM_ANNOTATION_FILENAME = "room_annotations.json"
ROOM_ANNOTATION_MANIFEST_FILENAME = "room_annotation_manifest.json"

# Released room polygons occasionally stop just short of a doorway's free
# cells. Bound the bridge to one G1 planning-footprint radius, then separately
# require full footprint clearance at the selected portal centre. This absorbs
# polygon quantization without turning arbitrary nearby rooms into a connection.
ROOM_ENTRANCE_MAX_ANNOTATION_GAP_M = 0.25

_ROOM_BOUNDARY_COLOR = (148, 163, 184)


@dataclass(frozen=True)
class TopDownBevProjection:
    """World-XY to canonical top-down BEV pixel projection."""

    bounds_min_xy: tuple[float, float]
    bounds_max_xy: tuple[float, float]
    resolution: int

    def __post_init__(self) -> None:
        if self.resolution <= 1:
            raise ValueError(f"BEV resolution must exceed one pixel: {self.resolution}")
        lo = np.asarray(self.bounds_min_xy, dtype=float)
        hi = np.asarray(self.bounds_max_xy, dtype=float)
        span = hi - lo
        if (
            lo.shape != (2,)
            or hi.shape != (2,)
            or not np.all(np.isfinite(lo))
            or not np.all(np.isfinite(hi))
            or np.any(span <= 0)
        ):
            raise ValueError(
                f"Invalid BEV world bounds: {self.bounds_min_xy}..{self.bounds_max_xy}"
            )
        if not math.isclose(float(span[0]), float(span[1]), rel_tol=0.0, abs_tol=1e-5):
            raise ValueError(
                f"Top-down BEV must use square world bounds, got span {span.tolist()}"
            )

    @property
    def span_m(self) -> float:
        return float(self.bounds_max_xy[0] - self.bounds_min_xy[0])

    def world_to_pixel_float(self, point_xy: Iterable[float]) -> tuple[float, float]:
        point = np.asarray(list(point_xy), dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"World point must be finite XY: {point_xy!r}")
        u = (
            (float(point[0]) - self.bounds_min_xy[0])
            / self.span_m
            * (self.resolution - 1)
        )
        v = (
            (self.bounds_max_xy[1] - float(point[1]))
            / self.span_m
            * (self.resolution - 1)
        )
        return u, v

    def pixel_to_world(self, point_uv: Iterable[float]) -> tuple[float, float]:
        point = np.asarray(list(point_uv), dtype=float)
        if point.shape != (2,) or not np.all(np.isfinite(point)):
            raise ValueError(f"Pixel point must be finite UV: {point_uv!r}")
        x = (
            self.bounds_min_xy[0]
            + float(point[0]) / (self.resolution - 1) * self.span_m
        )
        y = (
            self.bounds_max_xy[1]
            - float(point[1]) / (self.resolution - 1) * self.span_m
        )
        return x, y


def _read_json(path: Path) -> Any:
    if not path.is_file():
        raise FileNotFoundError(f"Required room-annotation source is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _asset_relative(path: Path, root: Path) -> str:
    try:
        return path.resolve().relative_to(root.resolve()).as_posix()
    except ValueError as exc:
        raise ValueError(
            f"Room-annotation artifact is outside the asset root: {path}"
        ) from exc


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    temporary.replace(path)


def _save_image_atomic(image: Image.Image, path: Path) -> None:
    temporary = path.with_name(f".{path.stem}.tmp{path.suffix}")
    image.save(temporary)
    temporary.replace(path)


def annotation_polygon_row_col_to_world(
    polygon_row_col: Iterable[Iterable[float]],
    *,
    x_axis_m: np.ndarray,
    y_axis_m: np.ndarray,
) -> list[list[float]]:
    """Convert GRScenes ``[row, column]`` room vertices to world XY."""
    x_axis = np.asarray(x_axis_m, dtype=float)
    y_axis = np.asarray(y_axis_m, dtype=float)
    if (
        x_axis.ndim != 1
        or y_axis.ndim != 1
        or x_axis.size <= 1
        or y_axis.size <= 1
        or not np.all(np.isfinite(x_axis))
        or not np.all(np.isfinite(y_axis))
    ):
        raise ValueError("Official occupancy coordinate axes must be finite 1D arrays.")

    points = np.asarray([list(point) for point in polygon_row_col], dtype=float)
    if (
        points.ndim != 2
        or points.shape[0] < 3
        or points.shape[1] != 2
        or not np.all(np.isfinite(points))
    ):
        raise ValueError(
            f"Room annotation needs at least three finite [row, column] points: {points}"
        )
    rows = points[:, 0]
    columns = points[:, 1]
    if (
        np.any(rows < 0)
        or np.any(rows > y_axis.size - 1)
        or np.any(columns < 0)
        or np.any(columns > x_axis.size - 1)
    ):
        raise ValueError(
            "Room annotation lies outside the official occupancy coordinate axes: "
            f"row={rows.min()}..{rows.max()}, column={columns.min()}..{columns.max()}"
        )

    x_world = np.interp(columns, np.arange(x_axis.size, dtype=float), x_axis)
    y_world = np.interp(rows, np.arange(y_axis.size, dtype=float), y_axis)
    return [
        [round(float(x), 6), round(float(y), 6)]
        for x, y in zip(x_world, y_world, strict=True)
    ]


def point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    """Even-odd containment test for a world-XY point against a room polygon.

    Shared authority for room localization: the annotation builder uses it to
    place labels, and the agent's semantic map uses it to answer "which room
    am I standing in".
    """
    x, y = float(point[0]), float(point[1])
    inside = False
    previous = len(polygon) - 1
    for current, (x_current, y_current) in enumerate(polygon):
        x_previous, y_previous = polygon[previous]
        intersects = ((y_current > y) != (y_previous > y)) and (
            x
            < (x_previous - x_current)
            * (y - y_current)
            / ((y_previous - y_current) or 1e-12)
            + x_current
        )
        if intersects:
            inside = not inside
        previous = current
    return inside


@dataclass(frozen=True)
class RoomOwnershipRegion:
    """Geometry that gives one annotated room ownership of a body centre.

    ``room_id`` is the display id printed on the Room BEV. ``label_xy`` is the
    world-metre anchor at which that display label is drawn. ``polygon_xy`` is
    the room boundary in the same world frame. The label anchor is the frozen
    tie-break authority when authored room polygons overlap.
    """

    room_id: str
    label_xy: tuple[float, float]
    polygon_xy: tuple[tuple[float, float], ...]

    def __post_init__(self) -> None:
        room_id = str(self.room_id).strip()
        label = tuple(float(value) for value in self.label_xy)
        polygon = tuple(
            tuple(float(value) for value in point) for point in self.polygon_xy
        )
        polygon_array = np.asarray(polygon, dtype=float)
        if not room_id:
            raise ValueError("Room ownership region requires a room id.")
        if len(label) != 2 or not all(math.isfinite(value) for value in label):
            raise ValueError(f"{room_id} room label anchor must be finite world XY.")
        if (
            polygon_array.ndim != 2
            or polygon_array.shape[0] < 3
            or polygon_array.shape[1] != 2
            or not np.all(np.isfinite(polygon_array))
        ):
            raise ValueError(
                f"{room_id} room polygon must contain at least three finite "
                "world-XY points."
            )
        object.__setattr__(self, "room_id", room_id)
        object.__setattr__(self, "label_xy", label)
        object.__setattr__(self, "polygon_xy", polygon)


def resolve_room_owner_id(
    position_xy: Iterable[float],
    regions: Iterable[RoomOwnershipRegion],
) -> str | None:
    """Return the unique annotated owner of one measured body centre.

    A position outside every polygon has no room. If polygons overlap, the
    containing room whose displayed label anchor is closest owns the point;
    the stable room id breaks an exact-distance tie. This is the sole room
    ownership rule shared by agent localization and hidden evaluation.
    """

    position = tuple(float(value) for value in position_xy)
    if len(position) != 2 or not all(math.isfinite(value) for value in position):
        raise ValueError("Room ownership position must be finite world XY.")
    declared = tuple(regions)
    if not declared:
        raise ValueError("Room ownership requires at least one region.")
    room_ids = tuple(region.room_id for region in declared)
    if len(set(room_ids)) != len(room_ids):
        raise ValueError("Room ownership region ids must be unique.")

    point = np.asarray(position, dtype=float)
    containing = tuple(
        region
        for region in declared
        if point_in_polygon(
            point,
            np.asarray(region.polygon_xy, dtype=float),
        )
    )
    if not containing:
        return None
    return min(
        containing,
        key=lambda region: (
            (position[0] - region.label_xy[0]) ** 2
            + (position[1] - region.label_xy[1]) ** 2,
            region.room_id,
        ),
    ).room_id


def load_room_ownership_regions(
    annotation_path: Path | str,
) -> tuple[RoomOwnershipRegion, ...]:
    """Load evaluator-grade room ownership geometry from one annotation file."""

    path = Path(annotation_path)
    payload = _read_json(path)
    if not isinstance(payload, dict):
        raise ValueError(f"Room annotation must be a JSON object: {path}")
    if payload.get("schema_version") != ROOM_ANNOTATION_SCHEMA:
        raise ValueError(
            f"{path} declares schema {payload.get('schema_version')!r}, "
            f"expected {ROOM_ANNOTATION_SCHEMA!r}"
        )
    rooms = payload.get("rooms")
    if not isinstance(rooms, list) or not rooms:
        raise ValueError(f"{path} declares no rooms")
    regions = tuple(
        RoomOwnershipRegion(
            room_id=str(room["display_id"]),
            label_xy=tuple(room["label_world_xy"]),
            polygon_xy=tuple(tuple(point) for point in room["polygon_world_xy"]),
        )
        for room in rooms
    )
    # Validate uniqueness even before the first queried pose.
    if len({region.room_id for region in regions}) != len(regions):
        raise ValueError(f"{path} declares duplicate room display ids")
    return regions


def _distance_to_polygon_edges(point: np.ndarray, polygon: np.ndarray) -> float:
    starts = polygon
    ends = np.roll(polygon, -1, axis=0)
    segments = ends - starts
    lengths_squared = np.sum(segments * segments, axis=1)
    offsets = point - starts
    ratios = np.divide(
        np.sum(offsets * segments, axis=1),
        lengths_squared,
        out=np.zeros_like(lengths_squared),
        where=lengths_squared > 1e-12,
    )
    ratios = np.clip(ratios, 0.0, 1.0)
    projections = starts + ratios[:, None] * segments
    return float(np.min(np.linalg.norm(point - projections, axis=1)))


def polygon_interior_label_point(
    polygon_xy: Iterable[Iterable[float]],
    *,
    grid_size: int = 31,
) -> tuple[float, float]:
    """Find a deterministic high-clearance label anchor inside a polygon."""
    polygon = np.asarray([list(point) for point in polygon_xy], dtype=float)
    if (
        polygon.ndim != 2
        or polygon.shape[0] < 3
        or polygon.shape[1] != 2
        or not np.all(np.isfinite(polygon))
    ):
        raise ValueError(f"Invalid polygon for label placement: {polygon_xy!r}")
    if grid_size < 3:
        raise ValueError(f"Label-point grid_size must be at least three: {grid_size}")

    lo = np.min(polygon, axis=0)
    hi = np.max(polygon, axis=0)
    if np.any(hi <= lo):
        raise ValueError(f"Degenerate polygon bounds for label placement: {lo}..{hi}")

    candidates: list[np.ndarray] = []
    for y in np.linspace(float(lo[1]), float(hi[1]), grid_size):
        for x in np.linspace(float(lo[0]), float(hi[0]), grid_size):
            point = np.asarray([x, y], dtype=float)
            if point_in_polygon(point, polygon):
                candidates.append(point)
    if not candidates:
        raise ValueError("Room polygon has no interior label candidate.")

    center = np.mean(polygon, axis=0)
    best = max(
        candidates,
        key=lambda point: (
            _distance_to_polygon_edges(point, polygon),
            -float(np.linalg.norm(point - center)),
        ),
    )
    return float(best[0]), float(best[1])


def _polygon_area(points_xy: Iterable[Iterable[float]]) -> float:
    points = np.asarray([list(point) for point in points_xy], dtype=float)
    x = points[:, 0]
    y = points[:, 1]
    return float(abs(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)) * 0.5)


def _room_metadata(
    object_raw: dict[str, Any],
) -> tuple[dict[int, str], dict[str, int]]:
    indexed_rooms: dict[int, str] = {}
    object_counts: dict[str, int] = {}
    for object_id, raw in object_raw.items():
        if not isinstance(raw, dict):
            raise ValueError(f"Object metadata for {object_id!r} must be a mapping.")
        room_id = str(raw.get("room") or "").strip()
        if not room_id or "/" not in room_id:
            raise ValueError(f"Object {object_id!r} has invalid room id: {room_id!r}")
        prefix = room_id.split("/", 1)[0]
        if not prefix.isdigit():
            raise ValueError(
                f"Object {object_id!r} has non-numeric room prefix: {room_id!r}"
            )
        index = int(prefix)
        existing = indexed_rooms.get(index)
        if existing is not None and existing != room_id:
            raise ValueError(
                f"Room index {index} maps to both {existing!r} and {room_id!r}"
            )
        indexed_rooms[index] = room_id
        object_counts[room_id] = object_counts.get(room_id, 0) + 1
    return indexed_rooms, object_counts


def _room_source_mask(
    polygon_row_col: Iterable[Iterable[float]],
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Rasterize one released room polygon in official occupancy pixels."""

    image = Image.new("1", (width, height), 0)
    points = [(float(point[1]), float(point[0])) for point in polygon_row_col]
    if len(points) < 3:
        raise ValueError(
            "Room entrance derivation needs at least three polygon points."
        )
    ImageDraw.Draw(image).polygon(points, fill=1)
    return np.asarray(image, dtype=bool)


def _connected_pixel_components(
    points: set[tuple[int, int]],
) -> tuple[tuple[tuple[int, int], ...], ...]:
    """Return deterministic 8-connected components from sparse pixels."""

    remaining = set(points)
    components: list[tuple[tuple[int, int], ...]] = []
    while remaining:
        start = min(remaining)
        remaining.remove(start)
        queue: deque[tuple[int, int]] = deque([start])
        component: list[tuple[int, int]] = []
        while queue:
            row, column = queue.popleft()
            component.append((row, column))
            for row_offset in (-1, 0, 1):
                for column_offset in (-1, 0, 1):
                    if row_offset == 0 and column_offset == 0:
                        continue
                    neighbour = (row + row_offset, column + column_offset)
                    if neighbour in remaining:
                        remaining.remove(neighbour)
                        queue.append(neighbour)
        components.append(tuple(sorted(component)))
    return tuple(components)


def derive_traversable_room_entrances(
    rooms: Iterable[dict[str, Any]],
    *,
    official_occupancy: np.ndarray,
    projection: TopDownBevProjection,
    maximum_annotation_gap_m: float = ROOM_ENTRANCE_MAX_ANNOTATION_GAP_M,
    minimum_clearance_m: float = G1_NAVIGATION_FOOTPRINT_RADIUS_M,
) -> list[dict[str, Any]]:
    """Derive G1-traversable portals between annotated room regions.

    Room polygons are seeds on the official free-space layer. A bounded
    four-connected wavefront bridges only small unannotated seams. Wherever
    wavefronts from two rooms meet, the connected seam is a portal candidate.
    A candidate is retained only if its best free-space point admits the G1
    circular planning footprint. This rejects apparent adjacency through a
    wall and one-pixel annotation noise without exposing an occupancy map.
    """

    declared_rooms = tuple(rooms)
    if len(declared_rooms) < 2:
        return []
    occupancy = np.asarray(official_occupancy)
    if occupancy.ndim != 2 or occupancy.shape[0] <= 2 or occupancy.shape[1] <= 2:
        raise ValueError(
            "Room entrance derivation requires official occupancy with axes."
        )
    if not math.isfinite(maximum_annotation_gap_m) or maximum_annotation_gap_m <= 0.0:
        raise ValueError("maximum_annotation_gap_m must be finite and positive.")
    if not math.isfinite(minimum_clearance_m) or minimum_clearance_m < 0.0:
        raise ValueError("minimum_clearance_m must be finite and non-negative.")

    x_axis = np.asarray(occupancy[0, 1:], dtype=float)
    y_axis = np.asarray(occupancy[1:, 0], dtype=float)
    labels = np.asarray(occupancy[1:, 1:], dtype=float)
    if (
        x_axis.size != labels.shape[1]
        or y_axis.size != labels.shape[0]
        or not np.all(np.isfinite(x_axis))
        or not np.all(np.isfinite(y_axis))
    ):
        raise ValueError("Official occupancy axes do not match its label grid.")
    x_spacing = float(np.median(np.abs(np.diff(x_axis))))
    y_spacing = float(np.median(np.abs(np.diff(y_axis))))
    if x_spacing <= 0.0 or y_spacing <= 0.0:
        raise ValueError("Official occupancy axes must have positive spacing.")
    pixel_step_m = max(x_spacing, y_spacing)
    maximum_gap_steps = int(math.ceil(maximum_annotation_gap_m / pixel_step_m))
    free = labels == 1.0
    height, width = free.shape

    # A pixel inside overlapping annotations is deliberately left unowned and
    # resolved by the same bounded wavefront as a tiny annotation seam. This
    # avoids choosing a portal merely from room declaration order.
    owner = np.full((height, width), -1, dtype=np.int16)
    overlap = np.zeros((height, width), dtype=bool)
    for room_index, room in enumerate(declared_rooms):
        mask = (
            _room_source_mask(
                room["source_polygon_row_col"],
                width=width,
                height=height,
            )
            & free
        )
        overlap |= mask & (owner >= 0)
        owner[mask & (owner < 0)] = room_index
    owner[overlap] = -1

    distance_steps = np.full((height, width), -1, dtype=np.int16)
    seed_rows, seed_columns = np.nonzero(owner >= 0)
    if seed_rows.size == 0:
        raise ValueError("No annotated room overlaps official traversable space.")
    distance_steps[seed_rows, seed_columns] = 0
    frontier: deque[tuple[int, int]] = deque(
        zip(seed_rows.tolist(), seed_columns.tolist(), strict=True)
    )
    meeting_pixels: dict[tuple[int, int], dict[tuple[int, int], int]] = defaultdict(
        dict
    )

    while frontier:
        row, column = frontier.popleft()
        next_distance = int(distance_steps[row, column]) + 1
        if next_distance > maximum_gap_steps:
            continue
        room_index = int(owner[row, column])
        for neighbour_row, neighbour_column in (
            (row - 1, column),
            (row + 1, column),
            (row, column - 1),
            (row, column + 1),
        ):
            if (
                neighbour_row < 0
                or neighbour_row >= height
                or neighbour_column < 0
                or neighbour_column >= width
                or not free[neighbour_row, neighbour_column]
            ):
                continue
            neighbour_owner = int(owner[neighbour_row, neighbour_column])
            if neighbour_owner < 0:
                owner[neighbour_row, neighbour_column] = room_index
                distance_steps[neighbour_row, neighbour_column] = next_distance
                frontier.append((neighbour_row, neighbour_column))
                continue
            if neighbour_owner == room_index:
                continue
            gap_steps = (
                int(distance_steps[row, column])
                + int(distance_steps[neighbour_row, neighbour_column])
                + 1
            )
            if gap_steps > maximum_gap_steps:
                continue
            pair = tuple(sorted((room_index, neighbour_owner)))
            for pixel in (
                (row, column),
                (neighbour_row, neighbour_column),
            ):
                previous = meeting_pixels[pair].get(pixel)
                if previous is None or gap_steps < previous:
                    meeting_pixels[pair][pixel] = gap_steps

    clearance_m = distance_transform_edt(
        free,
        sampling=(y_spacing, x_spacing),
    )
    entrances: list[dict[str, Any]] = []
    clearance_tolerance_m = pixel_step_m
    for room_pair in sorted(meeting_pixels):
        gaps = meeting_pixels[room_pair]
        for component in _connected_pixel_components(set(gaps)):
            coordinates = np.asarray(component, dtype=int)
            component_clearance = clearance_m[coordinates[:, 0], coordinates[:, 1]]
            best_clearance = float(np.max(component_clearance))
            if best_clearance + clearance_tolerance_m < minimum_clearance_m:
                continue
            centroid = np.mean(coordinates, axis=0)
            best_candidates = np.flatnonzero(
                np.isclose(component_clearance, best_clearance, atol=1e-9)
            )
            best_index = min(
                best_candidates.tolist(),
                key=lambda index: (
                    float(np.sum((coordinates[index] - centroid) ** 2)),
                    int(coordinates[index, 0]),
                    int(coordinates[index, 1]),
                ),
            )
            row = int(coordinates[best_index, 0])
            column = int(coordinates[best_index, 1])
            world_xy = (float(x_axis[column]), float(y_axis[row]))
            bev_pixel = projection.world_to_pixel_float(world_xy)
            room_ids = [str(declared_rooms[index]["display_id"]) for index in room_pair]
            entrances.append(
                {
                    "entrance_id": "",
                    "room_ids": room_ids,
                    "source_row_col": [row, column],
                    "world_xy": [round(value, 6) for value in world_xy],
                    "bev_pixel_xy": [int(round(value)) for value in bev_pixel],
                    "annotation_gap_m": round(
                        min(gaps[pixel] for pixel in component) * pixel_step_m,
                        4,
                    ),
                    "free_space_clearance_m": round(best_clearance, 4),
                }
            )

    entrances.sort(
        key=lambda item: (
            tuple(item["room_ids"]),
            int(item["source_row_col"][0]),
            int(item["source_row_col"][1]),
        )
    )
    for index, entrance in enumerate(entrances, start=1):
        entrance["entrance_id"] = f"E{index:02d}"
    return entrances


def build_room_annotations(
    scene_id: str,
    *,
    assets_root: Path | str | None = None,
) -> dict[str, Any]:
    """Build region semantics for one pinned GRScenes scene."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root(require=True)
    )
    paths = resolve_active_grscene(scene_id, assets_root=root)
    annotation_path = paths.meta_dir / "annotation.json"
    occupancy_path = paths.meta_dir / "occupancy.npy"
    rgb_init_path = paths.meta_dir / "rgb_init.png"
    object_path = paths.meta_dir / "object_dict_with_caption.json"
    bev_meta_path = paths.scene_dir / "bev_meta.json"
    bev_rgb_path = paths.scene_dir / "bev_rgb.png"

    annotations = _read_json(annotation_path)
    object_raw = _read_json(object_path)
    bev_meta = _read_json(bev_meta_path)
    if not isinstance(annotations, list) or not annotations:
        raise ValueError(
            f"annotation.json must contain a non-empty room list: {annotation_path}"
        )
    if not isinstance(object_raw, dict) or not object_raw:
        raise ValueError(f"Object metadata must be a non-empty mapping: {object_path}")
    if not isinstance(bev_meta, dict):
        raise ValueError(f"BEV metadata must be a mapping: {bev_meta_path}")
    if not occupancy_path.is_file():
        raise FileNotFoundError(f"Official occupancy is missing: {occupancy_path}")
    if not rgb_init_path.is_file():
        raise FileNotFoundError(f"Official RGB overview is missing: {rgb_init_path}")
    if not bev_rgb_path.is_file():
        raise FileNotFoundError(f"Canonical RGB BEV is missing: {bev_rgb_path}")

    occupancy = np.load(occupancy_path)
    if occupancy.ndim != 2 or occupancy.shape[0] <= 2 or occupancy.shape[1] <= 2:
        raise ValueError(f"Official occupancy has invalid shape: {occupancy.shape}")
    x_axis = np.asarray(occupancy[0, 1:], dtype=float)
    y_axis = np.asarray(occupancy[1:, 0], dtype=float)
    with Image.open(rgb_init_path) as official_rgb:
        official_size = official_rgb.size
    if official_size != (x_axis.size, y_axis.size):
        raise ValueError(
            "Official RGB and occupancy coordinate axes disagree: "
            f"rgb={official_size}, axes={(x_axis.size, y_axis.size)}"
        )

    resolution = int(bev_meta.get("resolution") or 0)
    with Image.open(bev_rgb_path) as bev_rgb:
        bev_size = bev_rgb.size
    if bev_size != (resolution, resolution):
        raise ValueError(
            f"Canonical RGB BEV size {bev_size} disagrees with metadata resolution {resolution}."
        )
    projection = TopDownBevProjection(
        bounds_min_xy=tuple(float(v) for v in bev_meta["bev_bounds_min_xy"]),
        bounds_max_xy=tuple(float(v) for v in bev_meta["bev_bounds_max_xy"]),
        resolution=resolution,
    )
    indexed_rooms, object_counts = _room_metadata(object_raw)
    type_ordinals: dict[str, int] = {}
    room_payloads: list[dict[str, Any]] = []
    annotation_only_count = 0

    for index, raw in enumerate(annotations):
        if not isinstance(raw, dict):
            raise ValueError(f"Room annotation {index} must be a mapping.")
        room_type = str(raw.get("room_type") or "").strip()
        polygon_row_col = raw.get("polygon")
        if not room_type:
            raise ValueError(f"Room annotation {index} has an empty room_type.")
        if not isinstance(polygon_row_col, list):
            raise ValueError(f"Room annotation {index} polygon must be a list.")

        metadata_room_id = indexed_rooms.get(index)
        expected_room_id = f"{index}/{room_type}"
        if metadata_room_id is not None and metadata_room_id != expected_room_id:
            raise ValueError(
                f"Room annotation {index} type {room_type!r} disagrees with "
                f"object metadata room {metadata_room_id!r}."
            )
        if metadata_room_id is None:
            annotation_only_count += 1

        world_polygon = annotation_polygon_row_col_to_world(
            polygon_row_col,
            x_axis_m=x_axis,
            y_axis_m=y_axis,
        )
        pixel_polygon_float = [
            projection.world_to_pixel_float(point) for point in world_polygon
        ]
        pixel_polygon = [[int(round(u)), int(round(v))] for u, v in pixel_polygon_float]
        coordinates = np.asarray(pixel_polygon, dtype=int)
        if (
            np.any(coordinates[:, 0] < 0)
            or np.any(coordinates[:, 0] >= resolution)
            or np.any(coordinates[:, 1] < 0)
            or np.any(coordinates[:, 1] >= resolution)
        ):
            raise ValueError(
                f"Projected room {index} lies outside the canonical RGB BEV: "
                f"{coordinates.min(axis=0).tolist()}..{coordinates.max(axis=0).tolist()}"
            )

        label_pixel_float = polygon_interior_label_point(pixel_polygon_float)
        label_pixel = [int(round(v)) for v in label_pixel_float]
        label_world = projection.pixel_to_world(label_pixel_float)
        type_ordinals[room_type] = type_ordinals.get(room_type, 0) + 1
        room_ordinal = type_ordinals[room_type]
        display_id = f"R{index + 1:02d}"
        display_type = room_type.replace("_", " ").title()
        display_label = display_type
        room_payloads.append(
            {
                "annotation_id": f"annotation:{index:02d}",
                "display_id": display_id,
                "display_label": display_label,
                "annotation_index": index,
                "room_type": room_type,
                "room_type_ordinal": room_ordinal,
                "metadata_room_id": metadata_room_id,
                "annotation_only": metadata_room_id is None,
                "semantic_object_count": (
                    object_counts.get(metadata_room_id, 0)
                    if metadata_room_id is not None
                    else 0
                ),
                "source_polygon_row_col": polygon_row_col,
                "polygon_world_xy": world_polygon,
                "polygon_bev_pixel_xy": pixel_polygon,
                "label_world_xy": [round(float(v), 6) for v in label_world],
                "label_bev_pixel_xy": label_pixel,
                "polygon_area_m2": round(_polygon_area(world_polygon), 4),
            }
        )

    missing_annotation_indices = sorted(
        set(indexed_rooms) - set(range(len(annotations)))
    )
    if missing_annotation_indices:
        raise ValueError(
            "Object metadata references room indices missing from annotation.json: "
            f"{missing_annotation_indices}"
        )

    entrances = derive_traversable_room_entrances(
        room_payloads,
        official_occupancy=occupancy,
        projection=projection,
    )

    return {
        "schema_version": ROOM_ANNOTATION_SCHEMA,
        "scene_id": scene_id,
        "oracle_asset": True,
        "annotation_scope": "static_scene_regions_and_traversable_entrances",
        "excluded_entities": [
            "object_marker",
            "robot",
            "human_actor_pose",
            "trajectory",
        ],
        "coordinate_contract": {
            "annotation_polygon_order": "row_column",
            "world_mapping": "x=x_axis[column], y=y_axis[row]",
            "official_image_size": [int(official_size[0]), int(official_size[1])],
            "bev_resolution": resolution,
            "bev_bounds_min_xy": [float(v) for v in projection.bounds_min_xy],
            "bev_bounds_max_xy": [float(v) for v in projection.bounds_max_xy],
            "bev_world_to_pixel": (
                "u=(x-x_min)/span*(resolution-1), v=(y_max-y)/span*(resolution-1)"
            ),
            "entrance_derivation": {
                "free_label": 1,
                "connectivity": 4,
                "maximum_annotation_gap_m": ROOM_ENTRANCE_MAX_ANNOTATION_GAP_M,
                "minimum_clearance_m": G1_NAVIGATION_FOOTPRINT_RADIUS_M,
                "meaning": (
                    "a G1-traversable portal between two annotated rooms; "
                    "not a route or task target"
                ),
            },
        },
        "source_files": {
            "annotation": _asset_relative(annotation_path, root),
            "occupancy": _asset_relative(occupancy_path, root),
            "rgb_init": _asset_relative(rgb_init_path, root),
            "object_metadata": _asset_relative(object_path, root),
            "bev_meta": _asset_relative(bev_meta_path, root),
            "bev_rgb": _asset_relative(bev_rgb_path, root),
        },
        "quality_report": {
            "annotation_room_count": len(room_payloads),
            "object_metadata_room_count": len(indexed_rooms),
            "annotation_only_room_count": annotation_only_count,
            "metadata_room_type_mismatch_count": 0,
            "projected_out_of_bounds_count": 0,
            "traversable_entrance_count": len(entrances),
        },
        "rooms": room_payloads,
        "entrances": entrances,
    }


def _rectangles_overlap(
    lhs: tuple[int, int, int, int], rhs: tuple[int, int, int, int]
) -> bool:
    return not (
        lhs[2] < rhs[0] or rhs[2] < lhs[0] or lhs[3] < rhs[1] or rhs[3] < lhs[1]
    )


def _label_rectangle(
    *,
    anchor: tuple[int, int],
    text_size: tuple[int, int],
    image_size: tuple[int, int],
    occupied: list[tuple[int, int, int, int]],
    padding: tuple[int, int],
) -> tuple[int, int, int, int]:
    width = text_size[0] + padding[0] * 2
    height = text_size[1] + padding[1] * 2
    offsets = (
        (0, 0),
        (0, -height - 10),
        (0, height + 10),
        (-width - 12, 0),
        (width + 12, 0),
        (-width - 12, -height - 10),
        (width + 12, -height - 10),
        (-width - 12, height + 10),
        (width + 12, height + 10),
    )
    best: tuple[int, int, int, int] | None = None
    for offset_x, offset_y in offsets:
        center_x = anchor[0] + offset_x
        center_y = anchor[1] + offset_y
        left = int(np.clip(center_x - width // 2, 2, max(2, image_size[0] - width - 2)))
        top = int(
            np.clip(center_y - height // 2, 2, max(2, image_size[1] - height - 2))
        )
        candidate = (left, top, left + width, top + height)
        if best is None:
            best = candidate
        if not any(_rectangles_overlap(candidate, previous) for previous in occupied):
            return candidate
    if best is None:
        raise RuntimeError("No room-label rectangle candidate was generated.")
    return best


def render_room_annotation_image(
    payload: dict[str, Any],
    *,
    base_image_path: Path | str,
    size_px: int | None = None,
) -> Image.Image:
    """Render room semantics directly at the requested output resolution."""

    scene_id = str(payload.get("scene_id") or "")
    if payload.get("schema_version") != ROOM_ANNOTATION_SCHEMA or not scene_id:
        raise ValueError(
            "Room annotation payload has an unsupported schema or empty scene id."
        )
    base_path = Path(base_image_path).expanduser().resolve()
    if not base_path.is_file():
        raise FileNotFoundError(f"Canonical RGB BEV is missing: {base_path}")

    base = Image.open(base_path).convert("RGBA")
    if size_px is not None:
        if int(size_px) <= 1:
            raise ValueError("Room annotation output size must exceed one pixel.")
        target_size = (int(size_px), int(size_px))
        if base.size != target_size:
            base = base.resize(target_size, Image.Resampling.LANCZOS)
    coordinate_contract = payload.get("coordinate_contract") or {}
    source_resolution = int(coordinate_contract.get("bev_resolution") or 0)
    if source_resolution <= 1:
        raise ValueError("Room annotations require a valid source BEV resolution.")
    scale_x = float(base.width - 1) / float(source_resolution - 1)
    scale_y = float(base.height - 1) / float(source_resolution - 1)

    def scaled_point(point: Iterable[float]) -> tuple[int, int]:
        values = tuple(float(value) for value in point)
        if len(values) != 2 or not all(math.isfinite(value) for value in values):
            raise ValueError(f"Room annotation point must be finite XY: {point!r}")
        return int(round(values[0] * scale_x)), int(round(values[1] * scale_y))

    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")
    line_width = max(2, int(round(base.width / 480)))
    font_size = max(16, int(round(base.width / 72)))
    font = ImageFont.load_default(size=font_size)

    rooms = list(payload.get("rooms") or [])
    if not rooms:
        raise ValueError("Room annotation payload has no rooms to render.")
    for room in rooms:
        polygon = [scaled_point(point) for point in room["polygon_bev_pixel_xy"]]
        draw.line(
            polygon + [polygon[0]],
            fill=_ROOM_BOUNDARY_COLOR + (185,),
            width=line_width,
        )

    occupied: list[tuple[int, int, int, int]] = []
    entrance_color = (6, 182, 212)
    for entrance in payload.get("entrances") or []:
        anchor = scaled_point(entrance["bev_pixel_xy"])
        marker_radius = max(7, line_width + 4)
        occupied.append(
            (
                anchor[0] - marker_radius - 5,
                anchor[1] - marker_radius - 5,
                anchor[0] + marker_radius + 5,
                anchor[1] + marker_radius + 5,
            )
        )
        glow_radius = marker_radius + 4
        draw.ellipse(
            [
                anchor[0] - glow_radius,
                anchor[1] - glow_radius,
                anchor[0] + glow_radius,
                anchor[1] + glow_radius,
            ],
            fill=entrance_color + (55,),
        )
        draw.ellipse(
            [
                anchor[0] - marker_radius,
                anchor[1] - marker_radius,
                anchor[0] + marker_radius,
                anchor[1] + marker_radius,
            ],
            fill=entrance_color + (105,),
            outline=entrance_color + (255,),
            width=max(2, line_width),
        )
        inner_radius = max(2, marker_radius // 4)
        draw.ellipse(
            [
                anchor[0] - inner_radius,
                anchor[1] - inner_radius,
                anchor[0] + inner_radius,
                anchor[1] + inner_radius,
            ],
            fill=entrance_color + (255,),
        )

    label_rooms = sorted(
        enumerate(rooms),
        key=lambda item: (float(item[1]["polygon_area_m2"]), int(item[0])),
    )
    for _, room in label_rooms:
        anchor = scaled_point(room["label_bev_pixel_xy"])
        label = str(room["room_type"]).replace("_", " ").title()
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_size = (text_bbox[2] - text_bbox[0], text_bbox[3] - text_bbox[1])
        padding = (8, 5)
        rectangle = _label_rectangle(
            anchor=anchor,
            text_size=text_size,
            image_size=base.size,
            occupied=occupied,
            padding=padding,
        )
        occupied.append(rectangle)
        rectangle_center = (
            (rectangle[0] + rectangle[2]) // 2,
            (rectangle[1] + rectangle[3]) // 2,
        )
        if rectangle_center != anchor:
            draw.line(
                [anchor, rectangle_center],
                fill=_ROOM_BOUNDARY_COLOR + (200,),
                width=line_width,
            )
        marker_radius = max(3, line_width + 1)
        draw.ellipse(
            [
                anchor[0] - marker_radius,
                anchor[1] - marker_radius,
                anchor[0] + marker_radius,
                anchor[1] + marker_radius,
            ],
            fill=_ROOM_BOUNDARY_COLOR + (255,),
            outline=(255, 255, 255, 240),
            width=1,
        )
        draw.rounded_rectangle(
            rectangle,
            radius=5,
            fill=(15, 23, 42, 218),
            outline=_ROOM_BOUNDARY_COLOR + (255,),
            width=max(1, line_width - 1),
        )
        draw.text(
            (rectangle[0] + padding[0], rectangle[1] + padding[1] - 1),
            label,
            fill=(248, 250, 252, 255),
            font=font,
        )

    return Image.alpha_composite(base, overlay).convert("RGB")


def render_room_annotation_map(
    payload: dict[str, Any],
    *,
    assets_root: Path | str | None = None,
    output_path: Path | str | None = None,
) -> Path:
    """Render one native-resolution room-map asset from a compiled payload."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root(require=True)
    )
    scene_id = str(payload.get("scene_id") or "")
    if payload.get("schema_version") != ROOM_ANNOTATION_SCHEMA or not scene_id:
        raise ValueError(
            "Room annotation payload has an unsupported schema or empty scene id."
        )
    source = load_asset_source_registry().active_source
    scene = source.require_scene(scene_id)
    scene_dir = (
        root / "scenes" / "GRScenes-100" / scene.collection / "scenes" / scene_id
    )
    base_path = root / str(payload["source_files"]["bev_rgb"])
    destination = (
        Path(output_path).expanduser().resolve()
        if output_path is not None
        else scene_dir / ROOM_MAP_FILENAME
    )
    image = render_room_annotation_image(
        payload,
        base_image_path=base_path,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    _save_image_atomic(image, destination)
    return destination


def generate_room_annotation_asset(
    scene_id: str,
    *,
    assets_root: Path | str | None = None,
) -> dict[str, Any]:
    """Compile JSON and render the region map for one pinned scene."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root(require=True)
    )
    paths = resolve_active_grscene(scene_id, assets_root=root)
    started = time.monotonic()
    payload = build_room_annotations(scene_id, assets_root=root)
    image_path = render_room_annotation_map(payload, assets_root=root)
    annotation_path = paths.scene_dir / ROOM_ANNOTATION_FILENAME
    payload["output_files"] = {
        "room_map": _asset_relative(image_path, root),
        "room_annotations": _asset_relative(annotation_path, root),
    }
    payload["elapsed_sec"] = round(time.monotonic() - started, 3)
    _write_json_atomic(annotation_path, payload)
    return {
        "scene_id": scene_id,
        "status": "generated",
        "room_count": len(payload["rooms"]),
        "entrance_count": len(payload["entrances"]),
        "annotation_only_room_count": payload["quality_report"][
            "annotation_only_room_count"
        ],
        "room_map": payload["output_files"]["room_map"],
        "room_annotations": payload["output_files"]["room_annotations"],
        "elapsed_sec": payload["elapsed_sec"],
    }


def generate_pinned_room_annotation_assets(
    *,
    scene_ids: Iterable[str] | None = None,
    assets_root: Path | str | None = None,
) -> dict[str, Any]:
    """Generate region semantics for pinned scenes in deterministic serial order."""
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root(require=True)
    )
    source = load_asset_source_registry().active_source
    requested = tuple(scene_ids) if scene_ids is not None else source.scene_ids
    if not requested:
        raise ValueError("Room annotation generation requires at least one scene.")
    if len(set(requested)) != len(requested):
        raise ValueError("Room annotation scene ids must be unique.")
    unknown = sorted(set(requested) - set(source.scene_ids))
    if unknown:
        raise ValueError(
            f"Room annotation generation requested unpinned scenes: {unknown}"
        )

    started = time.monotonic()
    results: list[dict[str, Any]] = []
    for index, scene_id in enumerate(requested, start=1):
        print(f"[ROOMS] [{index}/{len(requested)}] generating {scene_id}", flush=True)
        results.append(generate_room_annotation_asset(scene_id, assets_root=root))

    manifest = {
        "schema_version": ROOM_ANNOTATION_MANIFEST_SCHEMA,
        "source_id": source.source_id,
        "execution_mode": "deterministic_serial",
        "requested_scene_count": len(requested),
        "generated_scene_count": len(results),
        "failed_scene_count": 0,
        "room_count": sum(int(result["room_count"]) for result in results),
        "entrance_count": sum(int(result["entrance_count"]) for result in results),
        "annotation_only_room_count": sum(
            int(result["annotation_only_room_count"]) for result in results
        ),
        "elapsed_sec": round(time.monotonic() - started, 3),
        "results": results,
        "failures": [],
    }
    manifest_path = (
        root
        / "scenes"
        / "GRScenes-100"
        / "home_scenes"
        / ROOM_ANNOTATION_MANIFEST_FILENAME
    )
    _write_json_atomic(manifest_path, manifest)
    print(
        f"[ROOMS] completed {len(results)}/{len(requested)}; manifest={manifest_path}",
        flush=True,
    )
    return manifest
