"""Render the selected occupancy grid as the Agent's exploration BEV.

The default image uses sensor-derived online occupancy and measured mapping
poses. The ground-truth navigation diagnostic instead supplies the complete
certified occupancy to this same renderer. Candidate selection still uses the
current RGB-D observation; it does not consult the occupancy grid.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tarsphere.harness.memory.semantic import SemanticEntityRecord
from tarsphere.infra.navigation.open_frontier.contracts import DisplayFrontier

EXPLORATION_MEMORY_SIZE_PX = 320
EXPLORATION_MEMORY_MINIMUM_SPAN_M = 8.0
EXPLORATION_MEMORY_MARGIN_M = 1.0

_UNKNOWN_RGB = (205, 211, 221)
_FREE_RGB = (231, 246, 235)
_OCCUPIED_RGB = (42, 51, 68)
_FRONTIER_RGB = (34, 197, 94)
_CANDIDATE_RGB = (14, 165, 233)
_TRAIL_RGB = (37, 99, 235)
_ROBOT_RGB = (225, 29, 72)
_VIEW_CONE_RGB = (245, 158, 11)
_VIEW_CONE_LENGTH_M = 1.75
_SAVED_ENTITY_RGB = (124, 58, 237)


def _font(size: int, *, bold: bool = False) -> ImageFont.ImageFont:
    candidates = (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        (
            "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
            if bold
            else "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
        ),
    )
    for path in candidates:
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _finite_xy(value: Any, *, field_name: str) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64).reshape(-1)
    if point.size != 2 or not np.all(np.isfinite(point)):
        raise ValueError(f"{field_name} must contain two finite values.")
    return point


def _square_crop(
    mask: np.ndarray,
    *,
    required_cells: tuple[tuple[int, int], ...],
    minimum_span_cells: int,
    margin_cells: int,
) -> tuple[int, int, int, int]:
    points: list[tuple[int, int]] = [
        (int(row), int(col)) for row, col in required_cells
    ]
    if np.any(mask):
        rows, cols = np.nonzero(mask)
        points.extend(
            ((int(rows.min()), int(cols.min())), (int(rows.max()), int(cols.max())))
        )
    if not points:
        centre = (mask.shape[0] // 2, mask.shape[1] // 2)
        points.append(centre)

    row_values = [item[0] for item in points]
    col_values = [item[1] for item in points]
    row_centre = 0.5 * (min(row_values) + max(row_values))
    col_centre = 0.5 * (min(col_values) + max(col_values))
    side = max(
        int(minimum_span_cells),
        max(row_values) - min(row_values) + 1 + 2 * int(margin_cells),
        max(col_values) - min(col_values) + 1 + 2 * int(margin_cells),
    )
    side = min(side, mask.shape[0], mask.shape[1])

    def bounded_start(centre: float, extent: int, limit: int) -> int:
        return max(0, min(int(round(centre - 0.5 * extent)), limit - extent))

    row_start = bounded_start(row_centre, side, mask.shape[0])
    col_start = bounded_start(col_centre, side, mask.shape[1])
    return row_start, row_start + side, col_start, col_start + side


def _draw_dashed_line(
    draw: ImageDraw.ImageDraw,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    fill: tuple[int, int, int, int],
    width: int,
    dash_px: float = 10.0,
    gap_px: float = 7.0,
) -> None:
    delta = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
    length = float(np.linalg.norm(delta))
    if length <= 1e-6:
        return
    direction = delta / length
    cursor = 0.0
    while cursor < length:
        finish = min(length, cursor + dash_px)
        first = np.asarray(start, dtype=np.float64) + cursor * direction
        second = np.asarray(start, dtype=np.float64) + finish * direction
        draw.line([tuple(first), tuple(second)], fill=fill, width=width)
        cursor += dash_px + gap_px


def _online_frontier_mask(
    *,
    observed: np.ndarray,
    free: np.ndarray,
) -> np.ndarray:
    """Return explored-free cells touching unknown space in 8-connectivity."""

    unknown = ~observed
    adjacent_unknown = np.zeros_like(unknown, dtype=bool)
    padded = np.pad(unknown, 1, mode="constant", constant_values=False)
    height, width = unknown.shape
    for row_offset in range(3):
        for col_offset in range(3):
            if row_offset == 1 and col_offset == 1:
                continue
            adjacent_unknown |= padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
    return free & adjacent_unknown


def render_exploration_memory(
    *,
    occupancy_grid: Any,
    robot_world_xy: Any,
    robot_yaw_rad: float,
    camera_horizontal_fov_deg: float,
    displayed_frontiers: tuple[DisplayFrontier, ...],
    saved_entities: tuple[SemanticEntityRecord, ...] = (),
    output_size_px: int = EXPLORATION_MEMORY_SIZE_PX,
    suppress_frontier_glyphs: bool = False,
    font_factory: Any = None,
) -> np.ndarray:
    """Render occupancy and Agent-saved historical entity anchors.

    World ``+X`` is up and world ``+Y`` is left. This keeps the map stable
    across turns while matching the harness's right-handed motion signs when
    the Robot yaw is zero. The live heading arrow disambiguates other yaws.
    """

    if str(getattr(occupancy_grid, "map_source", "")) not in {
        "online",
        "ground_truth",
    }:
        raise ValueError("Exploration BEV requires online or ground-truth occupancy.")
    if not math.isfinite(float(robot_yaw_rad)):
        raise ValueError("robot_yaw_rad must be finite.")
    horizontal_fov_deg = float(camera_horizontal_fov_deg)
    if not math.isfinite(horizontal_fov_deg) or not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("camera_horizontal_fov_deg must be finite and in (0, 180).")
    size = int(output_size_px)
    if size < 128:
        raise ValueError("Exploration BEV output must be at least 128 pixels.")
    resolution = float(getattr(occupancy_grid, "resolution"))
    if not math.isfinite(resolution) or resolution <= 0.0:
        raise ValueError("Occupancy-map resolution must be finite and positive.")

    layers = occupancy_grid.exploration_layers()
    observed = np.asarray(layers["observed"], dtype=bool)
    free = np.asarray(layers["free"], dtype=bool)
    occupied = np.asarray(layers["occupied"], dtype=bool)
    if observed.ndim != 2 or any(
        layer.shape != observed.shape for layer in (free, occupied)
    ):
        raise ValueError("Exploration-BEV layers must be aligned 2-D arrays.")
    frontier = _online_frontier_mask(observed=observed, free=free)

    robot_xy = _finite_xy(robot_world_xy, field_name="robot_world_xy")
    pose_points = tuple(
        _finite_xy(point, field_name="exploration pose")
        for point in occupancy_grid.exploration_pose_history()
    )
    candidate_points = tuple(
        _finite_xy(
            item.world_xyz[:2],
            field_name=f"frontier {item.label} world anchor",
        )
        for item in displayed_frontiers
    )
    half_fov_rad = math.radians(horizontal_fov_deg) * 0.5
    cone_boundary_points = tuple(
        robot_xy
        + _VIEW_CONE_LENGTH_M
        * np.asarray(
            [
                math.cos(float(robot_yaw_rad) + offset),
                math.sin(float(robot_yaw_rad) + offset),
            ]
        )
        for offset in (-half_fov_rad, half_fov_rad)
    )
    required_world = (
        robot_xy,
        *pose_points,
        *candidate_points,
        *(entity.world_xyz[:2] for entity in saved_entities),
        *cone_boundary_points,
    )
    required_cells = tuple(
        tuple(int(value) for value in occupancy_grid.world_to_grid(point))
        for point in required_world
    )
    row0, row1, col0, col1 = _square_crop(
        observed,
        required_cells=required_cells,
        minimum_span_cells=int(
            math.ceil(EXPLORATION_MEMORY_MINIMUM_SPAN_M / resolution)
        ),
        margin_cells=int(math.ceil(EXPLORATION_MEMORY_MARGIN_M / resolution)),
    )

    rgb = np.empty((*observed.shape, 3), dtype=np.uint8)
    rgb[:, :] = _UNKNOWN_RGB
    rgb[free] = _FREE_RGB
    rgb[occupied] = _OCCUPIED_RGB
    rgb[frontier] = _FRONTIER_RGB
    # Flip both grid axes: image up is world +X and image left is world +Y.
    crop = np.ascontiguousarray(rgb[row0:row1, col0:col1][::-1, ::-1])
    image = (
        Image.fromarray(crop, mode="RGB")
        .resize(
            (size, size),
            resample=Image.Resampling.NEAREST,
        )
        .convert("RGBA")
    )
    scale_x = size / float(col1 - col0)
    scale_y = size / float(row1 - row0)

    def image_point(world_xy: Any) -> tuple[float, float]:
        row, col = (
            int(value)
            for value in occupancy_grid.world_to_grid(
                _finite_xy(world_xy, field_name="map annotation")
            )
        )
        x = (col1 - 1 - col + 0.5) * scale_x
        y = (row1 - 1 - row + 0.5) * scale_y
        return (
            float(np.clip(x, 0.0, size - 1.0)),
            float(np.clip(y, 0.0, size - 1.0)),
        )

    # This compact wedge communicates only the current horizontal camera
    # bearing. Its fixed drawing length is deliberately not a visibility,
    # sensor-range, explored-space, or reachability claim.
    centre = np.asarray(image_point(robot_xy), dtype=np.float64)
    cone_angles = np.linspace(
        float(robot_yaw_rad) - half_fov_rad,
        float(robot_yaw_rad) + half_fov_rad,
        25,
    )
    cone_arc = [
        image_point(
            robot_xy
            + _VIEW_CONE_LENGTH_M
            * np.asarray(
                [
                    math.cos(float(angle)),
                    math.sin(float(angle)),
                ]
            )
        )
        for angle in cone_angles
    ]
    # Alpha-composite the fill onto the map. Drawing a translucent RGBA value
    # directly into the base image leaves solid orange RGB behind when alpha
    # is discarded for the model-visible image.
    cone_layer = Image.new("RGBA", image.size, (0, 0, 0, 0))
    cone_draw = ImageDraw.Draw(cone_layer, mode="RGBA")
    cone_draw.polygon(
        [tuple(centre), *cone_arc],
        fill=(*_VIEW_CONE_RGB, 38),
    )
    image = Image.alpha_composite(image, cone_layer)
    draw = ImageDraw.Draw(image, mode="RGBA")
    cone_width = max(2, size // 180)
    draw.line(
        [tuple(centre), cone_arc[0]],
        fill=(*_VIEW_CONE_RGB, 255),
        width=cone_width,
    )
    draw.line(
        [tuple(centre), cone_arc[-1]],
        fill=(*_VIEW_CONE_RGB, 255),
        width=cone_width,
    )
    draw.line(
        cone_arc,
        fill=(*_VIEW_CONE_RGB, 205),
        width=cone_width,
    )

    if len(pose_points) >= 2:
        pixels = [image_point(point) for point in pose_points]
        for first, second in zip(pixels, pixels[1:]):
            _draw_dashed_line(
                draw,
                first,
                second,
                fill=(*_TRAIL_RGB, 235),
                width=max(3, size // 128),
            )

    make_font = _font if font_factory is None else font_factory
    label_font = make_font(max(16, size // 25), bold=True)
    marker_radius = max(12, size // 26)
    marker_groups: list[dict[str, Any]] = []
    for candidate in () if suppress_frontier_glyphs else displayed_frontiers:
        point = np.asarray(
            image_point(candidate.world_xyz[:2]),
            dtype=np.float64,
        )
        group = next(
            (
                item
                for item in marker_groups
                if float(np.linalg.norm(item["point"] - point)) <= marker_radius
            ),
            None,
        )
        if group is None:
            marker_groups.append(
                {
                    "point": point,
                    "labels": [candidate.label],
                    "count": 1,
                }
            )
        else:
            count = int(group["count"])
            group["point"] = (group["point"] * count + point) / (count + 1)
            group["labels"].append(candidate.label)
            group["count"] = count + 1
    for group in marker_groups:
        x, y = group["point"]
        label = "/".join(group["labels"])
        box = draw.textbbox((0, 0), label, font=label_font)
        text_width = box[2] - box[0]
        half_width = max(marker_radius, text_width / 2 + 7)
        draw.rounded_rectangle(
            (
                x - half_width,
                y - marker_radius,
                x + half_width,
                y + marker_radius,
            ),
            radius=marker_radius,
            fill=(15, 23, 42, 238),
            outline=(*_CANDIDATE_RGB, 255),
            width=max(2, size // 160),
        )
        draw.text(
            (
                x - text_width / 2 - box[0],
                y - (box[3] - box[1]) / 2 - box[1],
            ),
            label,
            font=label_font,
            fill=(255, 255, 255, 255),
        )

    # Overlay only: these bookmarks neither reveal nor clear map cells.
    entity_font = make_font(max(13, size // 25), bold=True)
    for entity in saved_entities:
        x, y = image_point(entity.world_xyz[:2])
        ink = (*_SAVED_ENTITY_RGB, 255)
        if entity.entity_type == "human":
            # Feet touch the saved coordinate; no implied heading.
            draw.ellipse(
                (x - 3, y - 17, x + 3, y - 11),
                fill=ink,
                outline=(255, 255, 255, 255),
                width=1,
            )
            draw.line([(x, y - 10), (x, y - 4)], fill=ink, width=3)
            draw.line([(x - 5, y - 7), (x, y - 10), (x + 5, y - 7)], fill=ink, width=2)
            draw.line([(x - 4, y), (x, y - 4), (x + 4, y)], fill=ink, width=2)
        elif entity.entity_type == "object":
            draw.rounded_rectangle(
                (x - 5, y - 5, x + 5, y + 5),
                radius=2,
                fill=ink,
                outline=(255, 255, 255, 255),
                width=1,
            )
        else:
            draw.polygon(
                [(x, y - 6), (x + 6, y), (x, y + 6), (x - 6, y)],
                fill=ink,
                outline=(255, 255, 255, 255),
                width=1,
            )
        box = draw.textbbox((0, 0), entity.label, font=entity_font)
        width, height = box[2] - box[0], box[3] - box[1]
        label_x = max(2, min(x + 9, size - width - 8))
        label_y = max(2, min(y - 17, size - height - 8))
        draw.rounded_rectangle(
            (label_x - 3, label_y - 3, label_x + width + 3, label_y + height + 3),
            radius=4,
            fill=(255, 255, 255, 240),
            outline=ink,
            width=1,
        )
        draw.text(
            (label_x - box[0], label_y - box[1]),
            entity.label,
            font=entity_font,
            fill=ink,
        )

    forward_world = robot_xy + 0.75 * np.asarray(
        [
            math.cos(float(robot_yaw_rad)),
            math.sin(float(robot_yaw_rad)),
        ]
    )
    tip = np.asarray(image_point(forward_world), dtype=np.float64)
    direction = tip - centre
    norm = float(np.linalg.norm(direction))
    if norm <= 1e-6:
        direction = np.asarray([0.0, -1.0])
    else:
        direction /= norm
    left = np.asarray([-direction[1], direction[0]])
    length = max(18, size // 17)
    half_width = max(10, size // 30)
    rear = centre - direction * length * 0.55
    notch = centre - direction * length * 0.02
    arrow = [
        tuple(centre + direction * length),
        tuple(rear + left * half_width),
        tuple(notch),
        tuple(rear - left * half_width),
    ]
    draw.polygon(
        arrow,
        fill=(*_ROBOT_RGB, 245),
        outline=(255, 255, 255, 255),
        width=max(2, size // 170),
    )

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


__all__ = [
    "EXPLORATION_MEMORY_SIZE_PX",
    "render_exploration_memory",
]
