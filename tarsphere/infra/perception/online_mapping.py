"""Private occupancy mapping from synchronized ego RGB-D, LiDAR and robot pose.

Robot pose comes from the simulator; SLAM is outside the benchmark. Depth rays
clear traversed 3-D voxels and mark occupied endpoints. Vertical columns project
to a raw 2-D layer, and the planner applies the G1 footprint once per route.

The policy can receive a visualization of observed free and occupied space.
It contains sensor-derived geometry, not official scene occupancy or task
bindings. Planner masks and metric map state remain private."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np

from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.infra.control.physical_contracts import (
    G1_BODY_HULL_XY,
    G1_FOOTPRINT_HALF_WIDTH_M,
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    G1_RELATIVE_POSE_ROTATION_TRANSLATION_TOLERANCE_M,
    G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M,
    LOCOMOTION_MAX_BACKWARD_DISTANCE_M,
    LOCOMOTION_MAX_LATERAL_DISTANCE_M,
    LOCOMOTION_MIN_STEP_DISTANCE_M,
    G1_RGBD_SELF_FILTER_PADDING_M,
    G1_STANDING_BODY_TOP_M,
    LIDAR_COARSE_AZIMUTH_STEP_DEG,
    g1_body_surface_offset_m,
)
from tarsphere.infra.perception.lidar import LidarScan


# Depth rays used for free-space clearing and far endpoints skip this many
# pixels. Near-field endpoints stay full-resolution. Stride 2 is the planning
# default: stride 4 left jagged walls and holes; stride 1 is a full-image cost.
ONLINE_DEPTH_SUBSAMPLE = 2
_PLANNING_BUFFER_RGB = (200, 170, 170)
_UNKNOWN_RGB = (200, 200, 200)
_FREE_RGB = (230, 245, 230)
_OCCUPIED_RGB = (55, 55, 75)


_UINT8_BIT_COUNTS = np.asarray(
    [int(value).bit_count() for value in range(256)],
    dtype=np.uint8,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bresenham(r0: int, c0: int, r1: int, c1: int):
    """Yield (r, c) integer grid cells along the line from (r0,c0) to (r1,c1)."""
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r0 < r1 else -1
    sc = 1 if c0 < c1 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        yield r, c
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc


def _dilate_4(mask: np.ndarray) -> np.ndarray:
    """One-cell 4-neighbour dilation, including the original support."""

    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    rows, cols = source.shape
    return (
        padded[1 : 1 + rows, 1 : 1 + cols]
        | padded[0:rows, 1 : 1 + cols]
        | padded[2 : 2 + rows, 1 : 1 + cols]
        | padded[1 : 1 + rows, 0:cols]
        | padded[1 : 1 + rows, 2 : 2 + cols]
    )


def _erode_4(mask: np.ndarray) -> np.ndarray:
    """One-cell 4-neighbour erosion."""

    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    rows, cols = source.shape
    return (
        padded[1 : 1 + rows, 1 : 1 + cols]
        & padded[0:rows, 1 : 1 + cols]
        & padded[2 : 2 + rows, 1 : 1 + cols]
        & padded[1 : 1 + rows, 0:cols]
        & padded[1 : 1 + rows, 2 : 2 + cols]
    )


def _morphological_close_4(mask: np.ndarray) -> np.ndarray:
    """Fill one-cell 4-neighbour holes without growing the outer boundary."""

    return _erode_4(_dilate_4(mask))


def _four_neighbor_support(mask: np.ndarray) -> np.ndarray:
    """True where a cell has at least one 4-connected True neighbour."""

    source = np.asarray(mask, dtype=bool)
    padded = np.pad(source, 1, mode="constant", constant_values=False)
    rows, cols = source.shape
    return (
        padded[0:rows, 1 : 1 + cols]
        | padded[2 : 2 + rows, 1 : 1 + cols]
        | padded[1 : 1 + rows, 0:cols]
        | padded[1 : 1 + rows, 2 : 2 + cols]
    )


def _close_ground_raster_gaps(mask: np.ndarray) -> np.ndarray:
    """Close one-cell 4-neighbour gaps between sampled ground endpoints.

    Projected neighbouring pixels describe a continuous image surface, while
    Bresenham produces discrete one-cell lines. A one-cell cross restores that
    raster support without filling diagonal corners or adding obstacle margin.
    Current surface endpoints are removed from this layer afterwards.
    """

    return _dilate_4(mask)


def _wrap_pi(angle: np.ndarray | float) -> np.ndarray | float:
    """Wrap radians into ``(-pi, pi]``."""

    return (
        np.mod(np.asarray(angle, dtype=np.float64) + math.pi, 2.0 * math.pi) - math.pi
    )


def _sector_range_limited_by_body_width_m(step_deg: float) -> float:
    """Farthest range at which a sector still cannot hide a G1-wide object.

    Two bearings ``step_deg`` apart are ``2 r sin(step/2)`` apart at range
    ``r``. Beyond the G1 body width that gap is large enough for an unobserved
    obstacle, so the inverse sensor must stop claiming free space.
    """

    half = math.radians(abs(float(step_deg))) * 0.5
    sine = math.sin(half)
    if sine <= 1e-12:
        return 0.0
    return (2.0 * G1_FOOTPRINT_HALF_WIDTH_M) / (2.0 * sine)


def _points_in_padded_polygon(
    points_xy: np.ndarray,
    polygon_xy: np.ndarray,
    *,
    padding_m: float,
) -> np.ndarray:
    """Classify points inside a polygon or its metric padding.

    This is the depth-camera equivalent of ``robot_self_filter``: the released
    G1 silhouette is the source geometry and the already-certified tracking
    margin covers small pose changes during gait. It is deliberately applied
    only to RGB-D endpoints. PhysX LiDAR still starts at the exact body surface
    and can therefore retain a real obstacle immediately outside the robot.
    """

    points = np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
    polygon = np.asarray(polygon_xy, dtype=np.float64).reshape(-1, 2)
    if points.shape[0] == 0:
        return np.zeros(0, dtype=bool)
    if polygon.shape[0] < 3:
        raise ValueError("self-filter polygon must contain at least 3 points")

    # Even-odd containment. Work only on the small near-body candidate set;
    # this avoids an image-pixels x polygon-edges allocation.
    px = points[:, 0]
    py = points[:, 1]
    inside = np.zeros(points.shape[0], dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        x0, y0 = float(previous[0]), float(previous[1])
        x1, y1 = float(current[0]), float(current[1])
        crosses = (y0 > py) != (y1 > py)
        denominator = y1 - y0
        if abs(denominator) > 1e-12:
            edge_x = x0 + (py - y0) * (x1 - x0) / denominator
            inside ^= crosses & (px < edge_x)
        previous = current

    padding = float(padding_m)
    if padding <= 0.0:
        return inside

    padding2 = padding * padding
    within_padding = np.zeros(points.shape[0], dtype=bool)
    previous = polygon[-1]
    for current in polygon:
        edge = current - previous
        length2 = float(np.dot(edge, edge))
        if length2 > 1e-12:
            offset = points - previous[None, :]
            t = np.clip((offset @ edge) / length2, 0.0, 1.0)
            closest = previous[None, :] + t[:, None] * edge[None, :]
            within_padding |= np.sum((points - closest) ** 2, axis=1) <= padding2
        previous = current
    return inside | within_padding


def _convex_hull_polygon(points_xy: np.ndarray) -> np.ndarray:
    """Return a deterministic counter-clockwise outer hull."""

    points = sorted(
        {
            (float(point[0]), float(point[1]))
            for point in np.asarray(points_xy, dtype=np.float64).reshape(-1, 2)
        }
    )
    if len(points) < 3:
        raise ValueError("convex hull requires at least 3 unique points")

    def cross(
        origin: tuple[float, float],
        first: tuple[float, float],
        second: tuple[float, float],
    ) -> float:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[float, float]] = []
    for point in points:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 0.0:
            lower.pop()
        lower.append(point)
    upper: list[tuple[float, float]] = []
    for point in reversed(points):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 0.0:
            upper.pop()
        upper.append(point)
    return np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)


_G1_SELF_FILTER_HULL_XY = _convex_hull_polygon(
    np.asarray(G1_BODY_HULL_XY, dtype=np.float64),
)


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------


class OnlineOccupancyGrid:
    """Incremental height-resolved occupancy projected to a 2-D costmap.

    Cell encoding (int8):
      UNKNOWN  =  0  — no stable classification (unseen or one pending hit)
      FREE     = -1  — observed and clear
      OCCUPIED =  1  — observed obstacle

    The raw grid contains sensor evidence only. G1 clearance is applied once,
    by the shared :class:`Planner`, when a route is requested.

    Coordinates:
      World XY in metres (Isaac world frame, Z-up).
      Grid row increases with world X; grid col increases with world Y.
    """

    UNKNOWN: int = 0
    FREE: int = -1
    OCCUPIED: int = 1

    # Retained as a serialized/debug contract. Depth and PhysX LiDAR are exact
    # simulator measurements, so one endpoint is a real occupied observation;
    # arbitrary temporal confirmation would only delay collision geometry.
    _OCCUPIED_EVIDENCE_THRESHOLD: int = 1
    _EVIDENCE_LIMIT: int = 1

    def __init__(
        self,
        *,
        origin_xy: tuple[float, float] = (0.0, 0.0),
        size_m: float = 24.0,
        resolution_m: float = 0.05,
        robot_radius_m: float = G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        obstacle_z_min_m: float = 0.04,
        obstacle_z_max_m: float = G1_STANDING_BODY_TOP_M,
        voxel_z_resolution_m: float = 0.10,
        map_source: str = "online",
    ) -> None:
        self.resolution = float(resolution_m)
        self.pixels_per_meter = 1.0 / self.resolution
        self.robot_radius_m = float(robot_radius_m)
        self.obstacle_z_min_m = float(obstacle_z_min_m)
        self.obstacle_z_max_m = float(obstacle_z_max_m)
        self.voxel_z_resolution_m = float(voxel_z_resolution_m)
        if self.voxel_z_resolution_m <= 0.0:
            raise ValueError("voxel_z_resolution_m must be positive")
        if self.obstacle_z_max_m <= self.obstacle_z_min_m:
            raise ValueError("obstacle_z_max_m must exceed obstacle_z_min_m")
        self._voxel_z_count = int(
            math.ceil(
                (self.obstacle_z_max_m - self.obstacle_z_min_m)
                / self.voxel_z_resolution_m
            )
        )
        if self._voxel_z_count > 63:
            raise ValueError(
                "height-resolved navigation supports at most 63 z voxels; "
                f"got {self._voxel_z_count}"
            )
        if map_source not in {"online", "ground_truth"}:
            raise ValueError(f"unknown navigation map source: {map_source!r}")
        self.map_source = str(map_source)

        n = int(math.ceil(size_m / resolution_m))
        self._n = n
        self._origin = np.asarray(origin_xy, dtype=np.float64)
        self._grid = np.zeros((n, n), dtype=np.int8)
        self._evidence = np.zeros((n, n), dtype=np.int8)
        self._observed = np.zeros((n, n), dtype=bool)
        self._ground_free = np.zeros((n, n), dtype=bool)
        # Packed vertical columns keep the map height-aware without allocating
        # a dense 960x960x13 boolean volume. Bit k is z voxel k.
        self._voxel_observed = np.zeros((n, n), dtype=np.uint64)
        self._voxel_occupied = np.zeros((n, n), dtype=np.uint64)
        # LiDAR sees current PhysX geometry, including moving people.
        # Free sectors accumulate across views. Occupied hits remain current
        # so moving people do not leave permanent ghosts. New obstacle
        # evidence overrides remembered free space in the projected map.
        self._lidar_free = np.zeros((n, n), dtype=bool)
        self._lidar_occupied = np.zeros((n, n), dtype=bool)
        # The body currently occupies this disk, so it is current free support
        # for the centre-point planner. It is replaced at every observation and
        # never erases persistent RGB-D/LiDAR obstacle evidence.
        self._transient_robot_free = np.zeros((n, n), dtype=bool)
        # Every accepted mapping capture contributes one measured pose cell.
        # This is the high-cadence route memory rendered by OpenFrontier; it is
        # distinct from a planned path and never interpolates through walls.
        self._pose_history_xy: list[tuple[float, float]] = []

        self._clearance_cells = int(math.ceil(self.robot_radius_m / self.resolution))
        self._planning_navigable: Optional[np.ndarray] = None

    @classmethod
    def from_official_occupancy(
        cls,
        occupancy_path: str | Path,
        *,
        robot_radius_m: float = G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        static_obstacles: tuple[tuple[float, float, float], ...] = (),
    ) -> "OnlineOccupancyGrid":
        """Adapt one official scene occupancy to the shared runtime planner.

        The returned grid is complete and immutable and belongs only to private
        dataset certification. Evaluated agents build occupancy from live sensors.
        G1 clearance is still applied by the same :class:`Planner` used in the
        benchmark and Oracle route gates. ``static_obstacles`` are authored
        (world X, world Y, body radius) disks absent from the baked scene;
        rasterize their bodies before the shared planner adds Robot clearance.
        """

        official = np.load(Path(occupancy_path))
        if official.ndim != 2 or official.shape[0] < 3:
            raise ValueError(f"invalid official occupancy shape: {official.shape!r}")
        x_axis = np.asarray(official[0, 1:], dtype=np.float64)
        y_axis = np.asarray(official[1:, 0], dtype=np.float64)
        labels = np.asarray(official[1:, 1:])
        if (
            labels.shape[0] != labels.shape[1]
            or x_axis.size != labels.shape[1]
            or y_axis.size != labels.shape[0]
            or not np.all(np.diff(x_axis) > 0.0)
            or not np.all(np.diff(y_axis) < 0.0)
        ):
            raise ValueError(f"invalid official occupancy axes: {occupancy_path}")
        resolution = float(np.median(np.diff(x_axis)))
        grid = cls(
            origin_xy=(float(x_axis[0]), float(y_axis[-1])),
            size_m=resolution * labels.shape[0],
            resolution_m=resolution,
            robot_radius_m=robot_radius_m,
            map_source="ground_truth",
        )
        grid._grid = np.where(
            labels.T[:, ::-1] == 1.0,
            grid.FREE,
            grid.OCCUPIED,
        ).astype(np.int8)
        rows, cols = np.indices(grid._grid.shape)
        world_x = grid._origin[0] + rows * grid.resolution
        world_y = grid._origin[1] + cols * grid.resolution
        for x, y, radius in static_obstacles:
            if not np.all(np.isfinite((x, y, radius))) or radius <= 0:
                raise ValueError(
                    "Static obstacles require finite XY and positive body radius."
                )
            # Mark every cell touched by a body, not only its cell centre.
            dx = np.maximum(np.abs(world_x - x) - 0.5 * grid.resolution, 0)
            dy = np.maximum(np.abs(world_y - y) - 0.5 * grid.resolution, 0)
            grid._grid[dx * dx + dy * dy <= radius * radius] = grid.OCCUPIED
        grid._observed[:, :] = True
        grid._ground_free[:, :] = grid._grid == grid.FREE
        grid._transient_robot_free[:, :] = False
        grid._evidence = np.where(
            grid._grid == grid.OCCUPIED,
            grid._OCCUPIED_EVIDENCE_THRESHOLD,
            0,
        ).astype(np.int8)
        grid._n = int(labels.shape[0])
        grid._planning_navigable = None
        return grid

    @classmethod
    def from_npz(cls, path: str | Path) -> "OnlineOccupancyGrid":
        """Restore an exact runtime map snapshot written by :meth:`dump_npz`.

        This is an offline replay path for diagnostics and navigation-module
        validation. It restores sensor layers, map geometry, and the robot
        footprint, then deliberately recomputes the derived planning mask via
        the canonical :class:`Planner` on first use.
        """

        snapshot_path = Path(path).expanduser().resolve()
        if not snapshot_path.is_file():
            raise FileNotFoundError(
                f"online occupancy snapshot not found: {snapshot_path}"
            )
        required = {
            "grid",
            "evidence",
            "observed",
            "ground_free",
            "voxel_observed",
            "voxel_occupied",
            "lidar_free",
            "lidar_occupied",
            "transient_robot_free",
            "pose_history_xy",
            "origin_xy",
            "resolution_m",
            "voxel_z_resolution_m",
            "obstacle_z_range_m",
            "robot_radius_m",
            "map_source",
        }
        with np.load(snapshot_path, allow_pickle=False) as snapshot:
            missing = sorted(required - set(snapshot.files))
            if missing:
                raise ValueError(
                    f"online occupancy snapshot is missing fields: {missing}"
                )
            raw_grid = np.asarray(snapshot["grid"])
            if raw_grid.ndim != 2 or raw_grid.shape[0] != raw_grid.shape[1]:
                raise ValueError(
                    "online occupancy snapshot grid must be square and 2-D; "
                    f"got {raw_grid.shape!r}"
                )
            origin = np.asarray(snapshot["origin_xy"], dtype=np.float64).reshape(-1)
            obstacle_range = np.asarray(
                snapshot["obstacle_z_range_m"],
                dtype=np.float64,
            ).reshape(-1)
            if origin.size != 2 or not np.all(np.isfinite(origin)):
                raise ValueError(
                    "online occupancy origin_xy must contain two finite values"
                )
            if obstacle_range.size != 2 or not np.all(np.isfinite(obstacle_range)):
                raise ValueError(
                    "online occupancy obstacle_z_range_m must contain two finite values"
                )
            resolution = float(np.asarray(snapshot["resolution_m"]).item())
            voxel_resolution = float(
                np.asarray(snapshot["voxel_z_resolution_m"]).item()
            )
            robot_radius = float(np.asarray(snapshot["robot_radius_m"]).item())
            map_source = str(np.asarray(snapshot["map_source"]).item())
            grid = cls(
                origin_xy=(float(origin[0]), float(origin[1])),
                size_m=float(raw_grid.shape[0]) * resolution,
                resolution_m=resolution,
                robot_radius_m=robot_radius,
                obstacle_z_min_m=float(obstacle_range[0]),
                obstacle_z_max_m=float(obstacle_range[1]),
                voxel_z_resolution_m=voxel_resolution,
                map_source=map_source,
            )
            layer_specs = {
                "_grid": ("grid", np.int8),
                "_evidence": ("evidence", np.int8),
                "_observed": ("observed", bool),
                "_ground_free": ("ground_free", bool),
                "_voxel_observed": ("voxel_observed", np.uint64),
                "_voxel_occupied": ("voxel_occupied", np.uint64),
                "_lidar_free": ("lidar_free", bool),
                "_lidar_occupied": ("lidar_occupied", bool),
                "_transient_robot_free": ("transient_robot_free", bool),
            }
            for attribute, (field, dtype) in layer_specs.items():
                layer = np.asarray(snapshot[field], dtype=dtype)
                if layer.shape != raw_grid.shape:
                    raise ValueError(
                        f"online occupancy field {field!r} has shape "
                        f"{layer.shape!r}, expected {raw_grid.shape!r}"
                    )
                setattr(grid, attribute, layer.copy())
            pose_history = np.asarray(
                snapshot["pose_history_xy"],
                dtype=np.float64,
            )
            if (
                pose_history.ndim != 2
                or pose_history.shape[1:] != (2,)
                or not np.all(np.isfinite(pose_history))
            ):
                raise ValueError("online occupancy pose_history_xy must be finite Nx2")
            grid._pose_history_xy = [
                (float(point[0]), float(point[1])) for point in pose_history
            ]
        valid_values = {grid.UNKNOWN, grid.FREE, grid.OCCUPIED}
        observed_values = set(int(value) for value in np.unique(grid._grid))
        if not observed_values <= valid_values:
            raise ValueError(
                "online occupancy grid contains invalid cell values: "
                f"{sorted(observed_values - valid_values)}"
            )
        grid._n = int(raw_grid.shape[0])
        grid._planning_navigable = None
        return grid

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def reset(
        self,
        *,
        origin_xy: Optional[tuple[float, float]] = None,
        size_m: Optional[float] = None,
    ) -> None:
        """Wipe the grid; optionally re-centre on a new origin / extent."""
        if origin_xy is not None:
            self._origin = np.asarray(origin_xy, dtype=np.float64)
        if size_m is not None:
            self._n = int(math.ceil(size_m / self.resolution))
        self._grid = np.zeros((self._n, self._n), dtype=np.int8)
        self._evidence = np.zeros((self._n, self._n), dtype=np.int8)
        self._observed = np.zeros((self._n, self._n), dtype=bool)
        self._ground_free = np.zeros((self._n, self._n), dtype=bool)
        self._voxel_observed = np.zeros((self._n, self._n), dtype=np.uint64)
        self._voxel_occupied = np.zeros((self._n, self._n), dtype=np.uint64)
        self._lidar_free = np.zeros((self._n, self._n), dtype=bool)
        self._lidar_occupied = np.zeros((self._n, self._n), dtype=bool)
        self._transient_robot_free = np.zeros((self._n, self._n), dtype=bool)
        self._pose_history_xy = []
        self._planning_navigable = None

    # ------------------------------------------------------------------
    # Coordinate transforms
    # ------------------------------------------------------------------

    def world_to_grid(self, xy: np.ndarray) -> np.ndarray:
        """World XY (metres) → grid (row, col); may be out of bounds."""
        rc = (np.asarray(xy, dtype=np.float64) - self._origin) / self.resolution
        return np.round(rc).astype(np.int64)

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """Planner-compatible world XY batch to ``(row, col)`` cells."""

        points = np.asarray(xy, dtype=np.float64)
        if points.size == 0:
            return np.zeros((0, 2), dtype=np.int64)
        return self.world_to_grid(points.reshape(-1, 2))

    def grid_to_world(self, rc: np.ndarray) -> np.ndarray:
        """Grid (row, col) → world XY centre of that cell (metres)."""
        return np.asarray(rc, dtype=np.float64) * self.resolution + self._origin

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        """Planner-compatible cell centre to world XY."""

        point = self.grid_to_world(np.asarray([row, col], dtype=np.float64))
        return float(point[0]), float(point[1])

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self._n and 0 <= c < self._n

    # ------------------------------------------------------------------
    # Grid update
    # ------------------------------------------------------------------

    def update_from_depth_with_K(
        self,
        *,
        depth: np.ndarray,
        K: np.ndarray,
        world_T_cam: np.ndarray,
        robot_xy: np.ndarray,
        lidar: Mapping[str, object] | None = None,
        robot_yaw_rad: float | None = None,
        depth_min_m: float = 0.10,
        depth_max_m: float = 10.0,
        subsample: int = ONLINE_DEPTH_SUBSAMPLE,
        near_field_max_m: float = 3.0,
    ) -> None:
        """Integrate one synchronized ego RGBD/LiDAR observation.

        Stride-sampled depth rays clear only the *3-D voxels* they traverse.
        Near-field endpoints are retained at full image resolution so narrow
        obstacle surfaces are not skipped. The default stride is
        ``ONLINE_DEPTH_SUBSAMPLE`` (2): it is dense enough for planning-quality
        walls without a full-image cost. Endpoint marking happens after ray
        clearing and wins within a frame. A high ray can therefore clear a
        high voxel without erasing a low obstacle in the same XY column.

        Direct ground endpoints establish persistent walkable support. Sparse
        PhysX range scans contribute bounded free-sector estimates between
        adjacent bearings. These estimates persist; occupied scan hits are
        current-frame evidence. This is not dense RTX LiDAR coverage. The
        current G1 footprint is planner start support, not persistent sensor
        evidence.
        """
        if self.map_source == "ground_truth":
            return
        self._planning_navigable = None
        self._transient_robot_free.fill(False)

        if int(subsample) <= 0:
            raise ValueError("subsample must be positive")
        if float(near_field_max_m) <= float(depth_min_m):
            raise ValueError("near_field_max_m must exceed depth_min_m")

        depth = np.asarray(depth, dtype=np.float32)
        K = np.asarray(K, dtype=np.float64)
        world_T_cam = np.asarray(world_T_cam, dtype=np.float64)
        robot_xy = np.asarray(robot_xy, dtype=np.float64).reshape(-1)
        if depth.ndim != 2 or not all(size > 0 for size in depth.shape):
            raise ValueError("depth must be a non-empty HxW array")
        if K.shape != (3, 3) or world_T_cam.shape != (4, 4):
            raise ValueError("K and world_T_cam must be 3x3 and 4x4")
        if robot_xy.size < 2 or not np.all(np.isfinite(robot_xy[:2])):
            raise ValueError("robot_xy must contain finite world x and y")
        if not np.all(np.isfinite(K)) or not np.all(np.isfinite(world_T_cam)):
            raise ValueError("camera calibration must be finite")
        fx, fy = float(K[0, 0]), float(K[1, 1])
        cx, cy = float(K[0, 2]), float(K[1, 2])
        if fx <= 0.0 or fy <= 0.0:
            raise ValueError("camera focal lengths must be positive")
        if not self._pose_history_xy or not np.array_equal(
            self.world_to_grid(robot_xy[:2]),
            self.world_to_grid(self._pose_history_xy[-1]),
        ):
            self._pose_history_xy.append((float(robot_xy[0]), float(robot_xy[1])))
        H, W = depth.shape

        valid_depth = (
            np.isfinite(depth) & (depth >= depth_min_m) & (depth <= depth_max_m)
        )

        def project_pixels(rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
            d = depth[rows, cols].astype(np.float64)
            xc = d * (cols.astype(np.float64) - cx) / fx
            yc = d * (rows.astype(np.float64) - cy) / fy
            pts_cam = np.stack([xc, yc, d, np.ones_like(d)], axis=1)
            return (world_T_cam @ pts_cam.T).T[:, :3]

        # Retain every near-field endpoint for thin obstacle surfaces. Far
        # endpoints and all inverse-sensor rays use the bounded stride grid.
        near_depth = valid_depth & (depth <= float(near_field_max_m))
        near_endpoint_r, near_endpoint_c = np.nonzero(near_depth)
        base_r_grid, base_c_grid = np.mgrid[
            0 : H : int(subsample), 0 : W : int(subsample)
        ]
        base_r = base_r_grid.ravel()
        base_c = base_c_grid.ravel()
        base_valid = valid_depth[base_r, base_c]
        base_r = base_r[base_valid]
        base_c = base_c[base_valid]

        endpoint_flat = np.unique(
            np.concatenate(
                [
                    near_endpoint_r.astype(np.int64) * W
                    + near_endpoint_c.astype(np.int64),
                    base_r.astype(np.int64) * W + base_c.astype(np.int64),
                ]
            )
        )
        endpoint_r = endpoint_flat // W
        endpoint_c = endpoint_flat % W
        endpoint_world = project_pixels(endpoint_r, endpoint_c)
        endpoint_wx = endpoint_world[:, 0]
        endpoint_wy = endpoint_world[:, 1]
        endpoint_wz = endpoint_world[:, 2]
        endpoint_obstacle = (endpoint_wz >= self.obstacle_z_min_m) & (
            endpoint_wz < self.obstacle_z_max_m
        )
        endpoint_ground = (endpoint_wz >= self.obstacle_z_min_m - 0.15) & (
            endpoint_wz < self.obstacle_z_min_m
        )
        # Reject rendered self-returns. This does not clear persistent external
        # geometry; it only prevents the current body becoming a new endpoint.
        endpoint_distance2 = (endpoint_wx - robot_xy[0]) ** 2 + (
            endpoint_wy - robot_xy[1]
        ) ** 2
        if self.robot_radius_m > 0.0 and endpoint_obstacle.any():
            if robot_yaw_rad is None or not math.isfinite(robot_yaw_rad):
                raise ValueError(
                    "robot_yaw_rad is required for RGB-D robot self-filtering"
                )
            hull = _G1_SELF_FILTER_HULL_XY
            self_filter_extent = (
                float(np.max(np.linalg.norm(hull, axis=1)))
                + G1_RGBD_SELF_FILTER_PADDING_M
            )
            near_self = endpoint_distance2 <= self_filter_extent**2
            self_return = np.zeros(endpoint_obstacle.shape, dtype=bool)
            if near_self.any():
                relative_world = np.stack(
                    [
                        endpoint_wx[near_self] - robot_xy[0],
                        endpoint_wy[near_self] - robot_xy[1],
                    ],
                    axis=1,
                )
                cosine = math.cos(float(robot_yaw_rad))
                sine = math.sin(float(robot_yaw_rad))
                # world -> body, where +X is forward and +Y is left.
                relative_body = np.stack(
                    [
                        cosine * relative_world[:, 0] + sine * relative_world[:, 1],
                        -sine * relative_world[:, 0] + cosine * relative_world[:, 1],
                    ],
                    axis=1,
                )
                self_return[near_self] = _points_in_padded_polygon(
                    relative_body,
                    hull,
                    padding_m=G1_RGBD_SELF_FILTER_PADDING_M,
                )
            endpoint_obstacle &= ~self_return

        if base_r.size:
            self._integrate_depth_free_rays(
                camera_world=world_T_cam[:3, 3],
                endpoint_world=project_pixels(base_r, base_c),
            )

        frame_ground = np.zeros_like(self._observed)
        if endpoint_ground.any():
            free_r = np.round(
                (endpoint_wx[endpoint_ground] - self._origin[0]) / self.resolution
            ).astype(np.int64)
            free_c = np.round(
                (endpoint_wy[endpoint_ground] - self._origin[1]) / self.resolution
            ).astype(np.int64)
            in_bounds = (
                (free_r >= 0) & (free_r < self._n) & (free_c >= 0) & (free_c < self._n)
            )
            frame_ground[free_r[in_bounds], free_c[in_bounds]] = True
        frame_ground = _close_ground_raster_gaps(frame_ground)
        self._ground_free |= frame_ground
        # Ground support and an obstacle may legitimately project into the
        # same XY cell (for example a chair leg beside a visible floor patch).
        # Do not clear the complete height column here. The inverse sensor
        # model above already clears only the exact voxels traversed by a
        # current depth ray; endpoint evidence is then applied below and wins
        # within the frame. Clearing a whole column from a ground endpoint
        # destroys valid vertical geometry and is not voxel-map semantics.

        # Mark exact endpoint height voxels after ray clearing.
        if endpoint_obstacle.any():
            obs_r = np.round(
                (endpoint_wx[endpoint_obstacle] - self._origin[0]) / self.resolution
            ).astype(np.int64)
            obs_c = np.round(
                (endpoint_wy[endpoint_obstacle] - self._origin[1]) / self.resolution
            ).astype(np.int64)
            obs_z = np.floor(
                (endpoint_wz[endpoint_obstacle] - self.obstacle_z_min_m)
                / self.voxel_z_resolution_m
            ).astype(np.int64)
            in_bounds = (
                (obs_r >= 0)
                & (obs_r < self._n)
                & (obs_c >= 0)
                & (obs_c < self._n)
                & (obs_z >= 0)
                & (obs_z < self._voxel_z_count)
            )
            if in_bounds.any():
                bits = np.left_shift(np.uint64(1), obs_z[in_bounds].astype(np.uint64))
                np.bitwise_or.at(
                    self._voxel_observed,
                    (obs_r[in_bounds], obs_c[in_bounds]),
                    bits,
                )
                np.bitwise_or.at(
                    self._voxel_occupied,
                    (obs_r[in_bounds], obs_c[in_bounds]),
                    bits,
                )

        self._lidar_occupied.fill(False)
        if lidar is not None:
            if robot_yaw_rad is None or not math.isfinite(robot_yaw_rad):
                raise ValueError("robot_yaw_rad is required when LiDAR is integrated")
            frame_free, self._lidar_occupied = self._lidar_frame_inverse_sensor(
                scan=LidarScan.from_mapping(lidar),
                robot_xy=robot_xy,
                robot_yaw_rad=float(robot_yaw_rad),
            )
            self._lidar_free |= frame_free

        self._rebuild_projected_grid()

        robot_rc = self.world_to_grid(robot_xy)
        r0, c0 = int(robot_rc[0]), int(robot_rc[1])
        if self._in_bounds(r0, c0):
            radius = self._clearance_cells
            row_min = max(0, r0 - radius)
            row_max = min(self._n, r0 + radius + 1)
            col_min = max(0, c0 - radius)
            col_max = min(self._n, c0 + radius + 1)
            rows, cols = np.ogrid[row_min:row_max, col_min:col_max]
            inside = (rows - r0) ** 2 + (cols - c0) ** 2 <= radius**2
            self._transient_robot_free[
                row_min:row_max,
                col_min:col_max,
            ][inside] = True

    def _integrate_depth_free_rays(
        self,
        *,
        camera_world: np.ndarray,
        endpoint_world: np.ndarray,
        chunk_size: int = 1024,
    ) -> None:
        """Clear and observe exact 3-D voxels traversed by depth rays."""

        camera = np.asarray(camera_world, dtype=np.float64).reshape(3)
        endpoints = np.asarray(endpoint_world, dtype=np.float64).reshape(-1, 3)
        if not endpoints.size:
            return
        scales = np.asarray(
            [
                self.resolution,
                self.resolution,
                self.voxel_z_resolution_m,
            ],
            dtype=np.float64,
        )
        for offset in range(0, endpoints.shape[0], int(chunk_size)):
            target = endpoints[offset : offset + int(chunk_size)]
            delta = target - camera[None, :]
            steps = np.maximum(
                1,
                np.ceil(np.max(np.abs(delta) / scales[None, :], axis=1)).astype(
                    np.int64
                ),
            )
            sample_index = np.arange(int(steps.max()), dtype=np.int64)[None, :]
            valid_sample = sample_index < steps[:, None]
            t = sample_index / steps[:, None]
            points = camera[None, None, :] + delta[:, None, :] * t[..., None]
            points = points[valid_sample]
            rows = np.round((points[:, 0] - self._origin[0]) / self.resolution).astype(
                np.int64
            )
            cols = np.round((points[:, 1] - self._origin[1]) / self.resolution).astype(
                np.int64
            )
            z_indices = np.floor(
                (points[:, 2] - self.obstacle_z_min_m) / self.voxel_z_resolution_m
            ).astype(np.int64)
            valid = (
                (rows >= 0)
                & (rows < self._n)
                & (cols >= 0)
                & (cols < self._n)
                & (z_indices >= 0)
                & (z_indices < self._voxel_z_count)
            )
            if not valid.any():
                continue
            rows = rows[valid]
            cols = cols[valid]
            z_indices = z_indices[valid]
            keys = np.unique((rows * self._n + cols) * self._voxel_z_count + z_indices)
            z_indices = keys % self._voxel_z_count
            cells = keys // self._voxel_z_count
            rows = cells // self._n
            cols = cells % self._n
            bits = np.left_shift(np.uint64(1), z_indices.astype(np.uint64))
            np.bitwise_or.at(self._voxel_observed, (rows, cols), bits)
            np.bitwise_and.at(
                self._voxel_occupied,
                (rows, cols),
                np.bitwise_not(bits),
            )

    def _lidar_frame_inverse_sensor(
        self,
        *,
        scan: LidarScan,
        robot_xy: np.ndarray,
        robot_yaw_rad: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return current free sectors and occupied PhysX endpoints.

        G1 LiDAR is sparse (3.75 deg forward / 22.5 deg elsewhere). Tracing
        one Bresenham line per azimuth wrote 1-cell max-range tunnels and
        left the space between beams unknown. Those tunnels entered the
        planning grid.

        This inverse sensor fills the angular wedge between each pair of
        adjacent bearings. The first/last wrap-around pair is used only
        when that gap is at most the coarse azimuth step, so a partial
        forward scan does not invent a rear wedge:

        * both miss: free through the wedge, but only to the range where
          the inter-beam gap still fits inside the G1 body width. A
          max-range miss never writes a 1-cell-wide corridor to 5 m.
        * both hit: free to the nearer return; both endpoints occupied;
          the chord between similar-range hits is occupied so a wall is
          a continuous block rather than two dots.
        * mixed: free only to the measured hit. The miss does not claim
          space past that surface.
        * a single bearing with no neighbour: mark a hit endpoint only.
          No adjacent beam means no sector, so no free needle.

        This method returns one scan only. The caller accumulates its free
        sectors and replaces occupied hits; persistent RGB-D obstacle voxels
        and current LiDAR hits override remembered free space.
        """

        frame_free = np.zeros_like(self._observed)
        frame_occupied = np.zeros_like(self._observed)
        bearings = scan.bearings_deg
        ranges = scan.ranges_m
        if not bearings:
            return frame_free, frame_occupied

        def beam_endpoint(
            bearing_deg: float,
            measured_m: float | None,
        ) -> tuple[np.ndarray, float, float]:
            world_angle = robot_yaw_rad + math.radians(float(bearing_deg))
            direction = np.array(
                [
                    math.cos(world_angle),
                    math.sin(world_angle),
                ],
                dtype=np.float64,
            )
            offset_m = g1_body_surface_offset_m(float(bearing_deg))
            reach_m = (
                float(scan.max_range_m) if measured_m is None else float(measured_m)
            )
            return robot_xy + (offset_m + reach_m) * direction, offset_m, world_angle

        for bearing_deg, measured_m in zip(bearings, ranges):
            if measured_m is None:
                continue
            endpoint_xy, _, _ = beam_endpoint(bearing_deg, measured_m)
            obstacle_rc = self.world_to_grid(endpoint_xy)
            obstacle_r, obstacle_c = int(obstacle_rc[0]), int(obstacle_rc[1])
            if self._in_bounds(obstacle_r, obstacle_c):
                frame_occupied[obstacle_r, obstacle_c] = True

        if len(bearings) < 2:
            frame_free[frame_occupied] = False
            return frame_free, frame_occupied

        max_body_offset_m = max(
            g1_body_surface_offset_m(float(bearing)) for bearing in bearings
        )
        max_reach_m = float(scan.max_range_m) + max_body_offset_m
        radius_cells = int(math.ceil(max_reach_m / self.resolution)) + 1
        robot_rc = self.world_to_grid(robot_xy)
        center_r, center_c = int(robot_rc[0]), int(robot_rc[1])
        row_min = max(0, center_r - radius_cells)
        row_max = min(self._n, center_r + radius_cells + 1)
        col_min = max(0, center_c - radius_cells)
        col_max = min(self._n, center_c + radius_cells + 1)
        if row_min >= row_max or col_min >= col_max:
            frame_free[frame_occupied] = False
            return frame_free, frame_occupied

        rows, cols = np.ogrid[row_min:row_max, col_min:col_max]
        world_x = self._origin[0] + rows.astype(np.float64) * self.resolution
        world_y = self._origin[1] + cols.astype(np.float64) * self.resolution
        delta_x = world_x - float(robot_xy[0])
        delta_y = world_y - float(robot_xy[1])
        distance_m = np.hypot(delta_x, delta_y)
        bearing_rad = np.arctan2(delta_y, delta_x)
        half_cell_m = 0.5 * self.resolution
        window_free = np.zeros(
            (row_max - row_min, col_max - col_min),
            dtype=bool,
        )

        pairs = [(index, index + 1) for index in range(len(bearings) - 1)]
        wrap_step_deg = abs(
            (float(bearings[-1]) - float(bearings[0]) + 180.0) % 360.0 - 180.0
        )
        if wrap_step_deg <= float(LIDAR_COARSE_AZIMUTH_STEP_DEG) + 1e-6:
            pairs.append((len(bearings) - 1, 0))
        for index, other in pairs:
            bearing_a = float(bearings[index])
            bearing_b = float(bearings[other])
            range_a = ranges[index]
            range_b = ranges[other]
            step_deg = abs((bearing_a - bearing_b + 180.0) % 360.0 - 180.0)
            if step_deg < 1e-6:
                continue
            if range_a is None and range_b is None:
                evidence_m = float(scan.max_range_m)
            elif range_a is None:
                evidence_m = float(range_b)
            elif range_b is None:
                evidence_m = float(range_a)
            else:
                evidence_m = min(float(range_a), float(range_b))
            free_m = min(
                evidence_m,
                _sector_range_limited_by_body_width_m(step_deg),
            )
            offset_m = min(
                g1_body_surface_offset_m(bearing_a),
                g1_body_surface_offset_m(bearing_b),
            )
            outer_m = offset_m + free_m + half_cell_m
            world_a = robot_yaw_rad + math.radians(bearing_a)
            world_b = robot_yaw_rad + math.radians(bearing_b)
            sector_delta = _wrap_pi(world_b - world_a)
            point_delta = _wrap_pi(bearing_rad - world_a)
            in_sector = (point_delta * sector_delta >= -1e-12) & (
                np.abs(point_delta) <= np.abs(sector_delta) + 1e-12
            )
            window_free |= (
                in_sector & (distance_m >= half_cell_m) & (distance_m <= outer_m)
            )
            if range_a is None or range_b is None:
                continue
            span_m = abs(float(range_a) - float(range_b))
            mean_m = 0.5 * (float(range_a) + float(range_b))
            gap_m = 2.0 * mean_m * math.sin(math.radians(step_deg) * 0.5)
            if span_m > max(0.30, gap_m):
                continue
            endpoint_a, _, _ = beam_endpoint(bearing_a, range_a)
            endpoint_b, _, _ = beam_endpoint(bearing_b, range_b)
            cell_a = self.world_to_grid(endpoint_a)
            cell_b = self.world_to_grid(endpoint_b)
            for row, col in _bresenham(
                int(cell_a[0]),
                int(cell_a[1]),
                int(cell_b[0]),
                int(cell_b[1]),
            ):
                if self._in_bounds(row, col):
                    frame_occupied[row, col] = True

        frame_free[row_min:row_max, col_min:col_max] = window_free
        frame_free[frame_occupied] = False
        return frame_free, frame_occupied

    def _rebuild_projected_grid(self) -> None:
        """Project height-resolved sensor state to UNKNOWN/FREE/OCCUPIED.

        Occupied wins when a column has more than one occupied height voxel,
        when a single occupied voxel has no visible ground (a wall or
        obstacle seen from the side), when that voxel is 4-connected to
        another occupied column (a surface, not a speckle), or when LiDAR
        reports a hit. An isolated ground-plus-one-hit cell keeps its 3-D
        occupied voxel but is not painted as a fat 2-D block: the visible
        floor remains free unless LiDAR contradicts it. A one-cell
        morphological close then seals stride holes without eating a
        doorway.
        """

        occupied_bits = self._voxel_occupied
        occupied_any = occupied_bits != np.uint64(0)
        multi_height = (occupied_bits & (occupied_bits - np.uint64(1))) != np.uint64(0)
        depth_occupied = (
            multi_height
            | (occupied_any & ~self._ground_free)
            | (occupied_any & _four_neighbor_support(occupied_any))
        )
        occupied = depth_occupied | self._lidar_occupied
        depth_observed = self._voxel_observed != np.uint64(0)
        observed = (
            depth_observed | self._ground_free | self._lidar_free | self._lidar_occupied
        )
        occupied = _morphological_close_4(occupied)
        free = _morphological_close_4(observed & ~occupied) & ~occupied
        self._grid.fill(self.UNKNOWN)
        self._grid[free] = self.FREE
        self._grid[occupied] = self.OCCUPIED
        self._observed[:, :] = observed | free | occupied
        self._evidence[:, :] = np.where(
            occupied,
            self._OCCUPIED_EVIDENCE_THRESHOLD,
            0,
        ).astype(np.int8)

    def coverage_stats(self) -> dict[str, object]:
        """Return unambiguous online-map extent diagnostics.

        A fraction over the fixed 48 m runtime canvas made a well-observed
        scene look almost empty. Metric area and the observed bounding box do
        not pretend that the canvas is the scene boundary.
        """

        free = int(np.count_nonzero(self._grid == self.FREE))
        occupied = int(np.count_nonzero(self._grid == self.OCCUPIED))
        observed = int(np.count_nonzero(self._observed))
        pending = int(np.count_nonzero(self._observed & (self._grid == self.UNKNOWN)))
        total = int(self._grid.size)
        observed_mask = self._observed
        result: dict[str, object] = {
            "free_cells": free,
            "occupied_cells": occupied,
            "height_voxel_count": self._voxel_z_count,
            "observed_height_voxels": int(
                _UINT8_BIT_COUNTS[self._voxel_observed.view(np.uint8)].sum(
                    dtype=np.uint64
                )
            ),
            "occupied_height_voxels": int(
                _UINT8_BIT_COUNTS[self._voxel_occupied.view(np.uint8)].sum(
                    dtype=np.uint64
                )
            ),
            "transient_robot_free_cells": int(
                np.count_nonzero(self._transient_robot_free)
            ),
            "pending_surface_cells": pending,
            "unobserved_cells": total - observed,
            "observed_cells": observed,
            "observed_area_m2": observed * self.resolution**2,
            "map_canvas_area_m2": total * self.resolution**2,
        }
        if observed:
            rows, cols = np.nonzero(observed_mask)
            row_min, row_max = int(rows.min()), int(rows.max())
            col_min, col_max = int(cols.min()), int(cols.max())
            world_min = self.grid_to_world(np.array([row_min, col_min]))
            world_max = self.grid_to_world(np.array([row_max, col_max]))
            bbox_cells = (row_max - row_min + 1) * (col_max - col_min + 1)
            result.update(
                {
                    "observed_bounds_world_xy": {
                        "min": world_min.tolist(),
                        "max": world_max.tolist(),
                    },
                    "observed_bbox_area_m2": bbox_cells * self.resolution**2,
                    "observed_bbox_fill_fraction": observed / bbox_cells,
                }
            )
        else:
            result.update(
                {
                    "observed_bounds_world_xy": None,
                    "observed_bbox_area_m2": 0.0,
                    "observed_bbox_fill_fraction": 0.0,
                }
            )
        return result

    def exploration_layers(self) -> dict[str, np.ndarray]:
        """Return immutable-by-copy occupancy layers for Agent-visible rendering.

        Online grids expose fused sensor evidence; ground-truth diagnostic
        grids expose complete official occupancy. Both exclude the inflated
        planning mask and task annotations.
        """

        occupied = self._grid == self.OCCUPIED
        free = self._grid == self.FREE
        pending = self._observed & (self._grid == self.UNKNOWN)
        return {
            "observed": self._observed.copy(),
            "free": free.copy(),
            "occupied": occupied.copy(),
            "pending": pending.copy(),
        }

    def exploration_pose_history(self) -> tuple[tuple[float, float], ...]:
        """Return measured mapping poses after consecutive-cell deduplication."""

        return tuple(self._pose_history_xy)

    # ------------------------------------------------------------------
    # Raw occupancy and G1 planning
    # ------------------------------------------------------------------

    def navigable_mask(self) -> np.ndarray:
        """Return sensor-derived free space before embodiment clearance.

        UNKNOWN and OCCUPIED cells remain blocked except under the currently
        measured body footprint: the robot physically occupies that region,
        so it is valid start support for a centre-point planner. Raw collision
        evidence remains in ``_grid`` for diagnostics and failure attribution.
        The shared Planner applies G1 clearance exactly once outside this
        method.
        """

        return (self._grid == self.FREE) | self._transient_robot_free

    def planning_navigable_mask(self) -> np.ndarray:
        """Return the cached G1-clear mask produced by the shared Planner."""

        if self._planning_navigable is None:
            planner = Planner(
                self,
                backend="numpy",
                simplify=True,
                time_limit_s=5.0,
                clearance_radius_m=self.robot_radius_m,
                cache_static_nav=False,
            )
            self._planning_navigable = np.asarray(
                planner.embodiment_navigable_mask(),
                dtype=bool,
            ).copy()
        return self._planning_navigable

    def _planner(
        self,
        *,
        world_exclusion_zones: Sequence[tuple[float, float, float]] = (),
    ) -> Planner:
        """Build the shared planner over canonical clearance plus transient disks.

        ``world_exclusion_zones`` never mutate the raw occupancy or the cached
        G1-clear mask.  They are applied by :class:`Planner` to a per-request
        copy, which lets a closed-loop caller avoid live actors without
        turning their last measured positions into persistent map geometry.
        """

        return Planner(
            self,
            backend="numpy",
            simplify=True,
            time_limit_s=5.0,
            clearance_radius_m=self.robot_radius_m,
            prepared_navigable_mask=self.planning_navigable_mask(),
            world_exclusion_zones=list(world_exclusion_zones),
            cache_static_nav=False,
        )

    def is_navigable(self, world_xy: np.ndarray) -> bool:
        rc = self.world_to_grid(np.asarray(world_xy))
        r, c = int(rc[0]), int(rc[1])
        if not self._in_bounds(r, c):
            return False
        return bool(self.planning_navigable_mask()[r, c])

    def is_raw_free(self, world_xy: np.ndarray) -> bool:
        """Whether a measured base centre is outside raw scene geometry.

        This deliberately does not apply the planning footprint.  It is used
        only to distinguish a stable pose just inside the conservative A*
        buffer from a pose whose centre actually entered geometry; it never
        authorizes route planning through the buffer.
        """

        rc = self.world_to_grid(np.asarray(world_xy))
        r, c = int(rc[0]), int(rc[1])
        return bool(self._in_bounds(r, c) and self._grid[r, c] == self.FREE)

    def can_turn_in_place(self, world_xy: np.ndarray) -> bool:
        """Prefer turning where the full body sweep and gait drift fit.

        This does not inflate the route footprint or veto a public body action.
        A* execution can translate along its existing path to a wider location.
        Check a small local disk against the already inflated mask, including
        partial cell intersections and the blocked canvas exterior.
        """
        xy = np.asarray(world_xy, dtype=float)
        if xy.shape != (2,) or not np.isfinite(xy).all():
            raise ValueError("Turn-clearance query requires finite world XY.")
        if not self.is_navigable(xy):
            return False
        extra_m = max(
            0.0,
            max(math.hypot(x, y) for x, y in G1_BODY_HULL_XY)
            + G1_RELATIVE_POSE_ROTATION_TRANSLATION_TOLERANCE_M
            - self.robot_radius_m,
        )
        return self._planning_disk_clear(xy, extra_m)

    def _planning_disk_clear(self, xy: np.ndarray, extra_m: float) -> bool:
        """Check additional local clearance without changing the planning mask."""
        center = self.world_to_grid(xy)
        extent = int(math.ceil(extra_m / self.resolution)) + 1
        rows, cols = np.meshgrid(
            np.arange(int(center[0]) - extent, int(center[0]) + extent + 1),
            np.arange(int(center[1]) - extent, int(center[1]) + extent + 1),
            indexing="ij",
        )
        centers = self._origin + np.stack((rows, cols), axis=-1) * self.resolution
        nearest = np.maximum(np.abs(centers - xy) - self.resolution / 2, 0.0)
        touched = np.linalg.norm(nearest, axis=-1) <= extra_m
        rows, cols = rows[touched], cols[touched]
        nav = self.planning_navigable_mask()
        if (
            np.any(rows < 0)
            or np.any(cols < 0)
            or np.any(rows >= nav.shape[0])
            or np.any(cols >= nav.shape[1])
        ):
            return False
        return bool(np.all(nav[rows, cols]))

    def translation_sweep_clear(self, start_xy, end_xy, yaw_rad: float) -> bool:
        """Check a fixed-heading body sweep against raw cell squares.

        This is a PointGoal motion preference, not another hard route footprint
        or a veto of public move_body. A convex hull of both translated body
        silhouettes contains the complete continuous translation. Separating
        axes include square edges, so checking only cell centres is avoided.
        """
        start, end = np.asarray(start_xy, dtype=float), np.asarray(end_xy, dtype=float)
        if (
            start.shape != (2,)
            or end.shape != (2,)
            or not np.isfinite([*start, *end, yaw_rad]).all()
        ):
            raise ValueError("Body sweep requires finite XY endpoints and yaw.")
        c, s = math.cos(yaw_rad), math.sin(yaw_rad)
        body = _G1_SELF_FILTER_HULL_XY @ np.array([[c, s], [-s, c]])
        polygon = _convex_hull_polygon(np.vstack((body + start, body + end)))
        lower = np.floor(
            (polygon.min(axis=0) - self._origin) / self.resolution - 0.5
        ).astype(int)
        upper = np.ceil(
            (polygon.max(axis=0) - self._origin) / self.resolution + 0.5
        ).astype(int)
        rows, cols = np.meshgrid(
            np.arange(lower[0], upper[0] + 1),
            np.arange(lower[1], upper[1] + 1),
            indexing="ij",
        )
        valid = (
            (rows >= 0)
            & (cols >= 0)
            & (rows < self._grid.shape[0])
            & (cols < self._grid.shape[1])
        )
        blocked = ~valid
        blocked[valid] = self._grid[rows[valid], cols[valid]] != self.FREE
        centers = self._origin + np.c_[rows[blocked], cols[blocked]] * self.resolution
        if not len(centers):
            return True
        edges = np.roll(polygon, -1, axis=0) - polygon
        axes = np.vstack((np.eye(2), np.c_[-edges[:, 1], edges[:, 0]]))
        axes /= np.linalg.norm(axes, axis=1)[:, None]
        projection = polygon @ axes.T
        cell_projection = centers @ axes.T
        half = self.resolution / 2 * np.abs(axes).sum(axis=1)
        overlap = (cell_projection + half >= projection.min(axis=0) - 1e-12) & (
            cell_projection - half <= projection.max(axis=0) + 1e-12
        )
        return not bool(np.any(np.all(overlap, axis=1)))

    def translation_turn_checkpoint(
        self, start_xy, yaw_rad: float, goal_xy
    ) -> Optional[np.ndarray]:
        """Prefer a short, physically reachable place to change heading.

        No new route is invented through obstacles: both the canonical centre
        segment and full directional body sweep must be clear. If none exists,
        the caller retains the existing footprint action and discloses that no
        extra-clearance alternative was found; ordinary contact is not a veto.
        """
        start, goal = (
            np.asarray(start_xy, dtype=float),
            np.asarray(goal_xy, dtype=float),
        )
        if goal.shape != (2,) or not np.isfinite(goal).all():
            raise ValueError("Turn checkpoint requires a finite goal XY.")
        if not self.translation_sweep_clear(
            start, start, yaw_rad
        ) or not self.is_navigable(start):
            return None
        limit = min(
            LOCOMOTION_MAX_BACKWARD_DISTANCE_M, LOCOMOTION_MAX_LATERAL_DISTANCE_M
        )
        directions = np.array(
            [
                (math.cos(a), math.sin(a))
                for a in np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
            ]
        )
        step = G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M
        # A one-tolerance checkpoint (currently 5 cm) is smaller than the
        # released gait's executable translation and can alternate around the
        # same narrow turn site forever.  Search from the public minimum body
        # step while retaining the same hard-clearance and sweep predicates.
        first_executable_distance = max(step, LOCOMOTION_MIN_STEP_DISTANCE_M)
        for distance in np.arange(first_executable_distance, limit + 1e-9, step):
            candidates = start + distance * directions
            for index in np.argsort(
                np.linalg.norm(candidates - goal, axis=1), kind="stable"
            ):
                end = candidates[index]
                if (
                    self.can_turn_in_place(end)
                    and self._planner().is_line_collision_free(start, end)
                    and self.translation_sweep_clear(start, end, yaw_rad)
                ):
                    return end
        return None

    def clearance_buffer_exit(
        self,
        world_xy: np.ndarray,
        *,
        preferred_goal_xy: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """Find a physical straight exit from accidental footprint-buffer entry.

        The measured centre must still be in a raw FREE cell. Throughout the
        segment, prefer a footprint that touches no new blocked cell and whose
        centre moves monotonically away from every initially overlapping cell.
        If opposing buffer contacts make that impossible, ordinary contact is
        not promoted to a hard veto: fall back to the shortest raw-cell-free
        centre segment. The caller executes that proven segment incrementally
        and re-observes after every step. The endpoint restores the canonical
        planning mask plus the shared lateral execution tolerance. This is not
        cell snapping, obstacle removal or permission to enter raw geometry.
        """
        start = np.asarray(world_xy, dtype=float)
        if start.shape != (2,) or not np.isfinite(start).all():
            raise ValueError("Buffer exit requires finite world XY.")
        preferred_goal = None
        if preferred_goal_xy is not None:
            preferred_goal = np.asarray(preferred_goal_xy, dtype=float)
            if preferred_goal.shape != (2,) or not np.isfinite(preferred_goal).all():
                raise ValueError("Preferred buffer-exit goal requires finite world XY.")
        rc = self.world_to_grid(start).astype(int)
        if (
            not self._in_bounds(*rc)
            or self._grid[tuple(rc)] != self.FREE
            or self.is_navigable(start)
        ):
            return None
        radius = self.robot_radius_m
        limit = min(
            LOCOMOTION_MAX_BACKWARD_DISTANCE_M, LOCOMOTION_MAX_LATERAL_DISTANCE_M
        )
        margin = G1_RELATIVE_POSE_LATERAL_TRANSLATION_TOLERANCE_M
        half = self.resolution / 2
        extent = int(math.ceil((limit + radius + margin) / self.resolution)) + 1
        rows, cols = np.meshgrid(
            np.arange(rc[0] - extent, rc[0] + extent + 1),
            np.arange(rc[1] - extent, rc[1] + extent + 1),
            indexing="ij",
        )
        valid = (
            (rows >= 0)
            & (cols >= 0)
            & (rows < self._grid.shape[0])
            & (cols < self._grid.shape[1])
        )
        blocked = ~valid
        blocked[valid] = self._grid[rows[valid], cols[valid]] != self.FREE
        centers = self._origin + np.c_[rows[blocked], cols[blocked]] * self.resolution
        nearest = np.maximum(np.abs(start - centers) - half, 0.0)
        initial_overlap = np.sum(nearest * nearest, axis=1) <= radius**2
        old_centers, new_centers = centers[initial_overlap], centers[~initial_overlap]

        def continuous_sweep_clear(end):
            delta = end - start
            # This inequality holds for every point in each square cell, not
            # just its centre. Distance to every old obstacle point can only
            # increase along the whole straight segment.
            if np.any(
                (start - old_centers) @ delta < half * np.abs(delta).sum() - 1e-12
            ):
                return False
            if not len(new_centers):
                return True
            lo, hi = new_centers - half, new_centers + half
            # Exact segment/AABB intersection via slab intervals.
            entry, leave = np.zeros(len(lo)), np.ones(len(lo))
            for axis in range(2):
                if abs(delta[axis]) < 1e-12:
                    outside = (start[axis] < lo[:, axis]) | (start[axis] > hi[:, axis])
                    entry[outside] = np.inf
                else:
                    times = np.c_[
                        (lo[:, axis] - start[axis]) / delta[axis],
                        (hi[:, axis] - start[axis]) / delta[axis],
                    ]
                    entry = np.maximum(entry, times.min(axis=1))
                    leave = np.minimum(leave, times.max(axis=1))
            if np.any(entry <= leave):
                return False
            # Disjoint segment/square distance is attained at an endpoint of
            # either segment: our endpoints to the square, or its corners to
            # our segment. This catches contact between samples as well.
            start_gap = np.maximum(np.abs(start - new_centers) - half, 0.0)
            end_gap = np.maximum(np.abs(end - new_centers) - half, 0.0)
            distance_sq = np.minimum(
                np.sum(start_gap**2, axis=1), np.sum(end_gap**2, axis=1)
            )
            for offset in ((-half, -half), (-half, half), (half, -half), (half, half)):
                corner = new_centers + offset
                fraction = np.clip((corner - start) @ delta / (delta @ delta), 0.0, 1.0)
                gap = corner - start - fraction[:, None] * delta
                distance_sq = np.minimum(distance_sq, np.sum(gap**2, axis=1))
            return bool(np.all(distance_sq > radius**2 + 1e-12))

        def raw_center_sweep_clear(end):
            """Keep the measured base centre outside every raw cell square."""
            delta = end - start
            lo, hi = centers - half, centers + half
            entry, leave = np.zeros(len(lo)), np.ones(len(lo))
            for axis in range(2):
                if abs(delta[axis]) < 1e-12:
                    outside = (start[axis] < lo[:, axis]) | (start[axis] > hi[:, axis])
                    entry[outside] = np.inf
                else:
                    times = np.c_[
                        (lo[:, axis] - start[axis]) / delta[axis],
                        (hi[:, axis] - start[axis]) / delta[axis],
                    ]
                    entry = np.maximum(entry, times.min(axis=1))
                    leave = np.minimum(leave, times.max(axis=1))
            return not bool(np.any(entry <= leave))

        # A bounded deterministic local search, not an episode-specific pose.
        # Do not return a sub-resolution exit. The PointGoal handler executes
        # this endpoint with the same physical G1 controller as move_body;
        # requests shorter than its public minimum are dominated by gait
        # endpoint drift and can oscillate forever inside the same buffer.
        first_executable_distance = max(margin, LOCOMOTION_MIN_STEP_DISTANCE_M)
        directions = np.array(
            [
                (math.cos(angle), math.sin(angle))
                for angle in np.linspace(0.0, 2 * math.pi, 72, endpoint=False)
            ]
        )

        def ordered_candidates(distance):
            candidates = start + distance * directions
            if preferred_goal is None:
                return candidates
            order = np.argsort(
                np.linalg.norm(candidates - preferred_goal, axis=1),
                kind="stable",
            )
            return candidates[order]

        for distance in np.arange(first_executable_distance, limit + 1e-9, margin / 2):
            for end in ordered_candidates(distance):
                # Cell-centre erosion alone is not continuous circle/square
                # clearance, especially on coarse or diagonal boundaries.
                end_gap = np.maximum(np.abs(end - centers) - half, 0.0)
                raw_clear = np.all(np.sum(end_gap**2, axis=1) >= (radius + margin) ** 2)
                if (
                    raw_clear
                    and self._planning_disk_clear(end, margin)
                    and continuous_sweep_clear(end)
                ):
                    return end
        # A robot already inside overlapping buffers can have obstacles on
        # opposing sides, so no direction increases distance to every one.
        # The 0.25 m planner contract still governs the recovered endpoint;
        # only ordinary contact along the short, repeatedly observed exit is
        # tolerated. The base centre itself never crosses raw scene geometry.
        for distance in np.arange(first_executable_distance, limit + 1e-9, margin / 2):
            for end in ordered_candidates(distance):
                if self._planning_disk_clear(end, margin) and raw_center_sweep_clear(
                    end
                ):
                    return end

        # A concave or narrow buffer can require more than one straight
        # correction before the complete robot disk is planning-clear.  The
        # previous search rejected every such case because it required the
        # first endpoint to leave the inflated buffer completely.  Choose a
        # raw-cell-free intermediate endpoint that strictly increases measured
        # obstacle clearance instead.  The handler executes only one bounded
        # gait step, re-observes, and calls this routine again, so this cannot
        # silently jump or snap the robot through geometry.
        def raw_obstacle_clearance(end):
            gaps = np.maximum(np.abs(end - centers) - half, 0.0)
            return float(np.sqrt(np.min(np.sum(gaps * gaps, axis=1))))

        start_clearance = raw_obstacle_clearance(start)
        improving: list[tuple[float, float, np.ndarray]] = []
        for distance in np.arange(first_executable_distance, limit + 1e-9, margin / 2):
            for end in ordered_candidates(distance):
                if not self.is_raw_free(end) or not raw_center_sweep_clear(end):
                    continue
                clearance = raw_obstacle_clearance(end)
                if clearance <= start_clearance + margin / 4:
                    continue
                goal_distance = (
                    0.0
                    if preferred_goal is None
                    else float(np.linalg.norm(end - preferred_goal))
                )
                improving.append((clearance, -goal_distance, end))
        if improving:
            return max(improving, key=lambda item: (item[0], item[1]))[2]
        return None

    def goal_candidates(
        self,
        robot_xy: np.ndarray,
        target_xy: np.ndarray,
        *,
        lateral_tolerance_m: Optional[float] = 0.25,
        search_radius_m: float = 0.50,
        world_exclusion_zones: Sequence[tuple[float, float, float]] = (),
    ) -> tuple[tuple[float, float], ...]:
        """Enumerate body-clear destinations in the caller's allowed region.

        Candidate centres must project onto the robot→target segment, stay
        within ``lateral_tolerance_m`` metres of that segment's line, and
        within ``search_radius_m`` metres of the original target. These are
        endpoint constraints, not restrictions on the later A* route. No
        connectivity preference or additional footprint inflation is applied.
        ``lateral_tolerance_m=None`` selects the full target disk for saved
        entities. A* chooses among these cells; this method never picks a
        single nearest cell or prunes isolated cells heuristically.
        """
        start = np.asarray(robot_xy, dtype=np.float64).reshape(-1)
        target = np.asarray(target_xy, dtype=np.float64).reshape(-1)
        if start.size != 2 or target.size != 2:
            raise ValueError("robot_xy and target_xy must each contain two values")
        if not (np.all(np.isfinite(start)) and np.all(np.isfinite(target))):
            raise ValueError("robot_xy and target_xy must be finite")

        if lateral_tolerance_m is not None and (
            not math.isfinite(lateral_tolerance_m) or lateral_tolerance_m < 0
        ):
            raise ValueError("lateral_tolerance_m must be finite and nonnegative")
        if not math.isfinite(search_radius_m) or search_radius_m <= 0:
            raise ValueError("search_radius_m must be finite and positive")
        navigable = (
            self._planner(
                world_exclusion_zones=world_exclusion_zones,
            ).active_navigable_mask()
            if world_exclusion_zones
            else self.planning_navigable_mask()
        )
        r0, c0 = (int(v) for v in self.world_to_grid(target))
        candidates = []
        if self._in_bounds(r0, c0) and navigable[r0, c0]:
            candidates.append(tuple(float(v) for v in target))
        delta = target - start
        length = float(np.linalg.norm(delta))
        if length <= 1e-9 and lateral_tolerance_m is not None:
            return tuple(candidates)
        direction = delta / length if length > 1e-9 else np.zeros(2)
        radius_cells = int(math.ceil(search_radius_m / self.resolution)) + 1
        rows = range(
            max(0, r0 - radius_cells), min(navigable.shape[0], r0 + radius_cells + 1)
        )
        cols = range(
            max(0, c0 - radius_cells), min(navigable.shape[1], c0 + radius_cells + 1)
        )
        for r in rows:
            for c in cols:
                if not navigable[r, c] or (r, c) == (r0, c0):
                    continue
                point = self.grid_to_world(np.asarray([r, c], dtype=np.float64))
                offset = point - start
                along = float(np.dot(offset, direction))
                lateral = abs(
                    float(direction[0] * offset[1] - direction[1] * offset[0])
                )
                distance = float(np.linalg.norm(point - target))
                if distance > search_radius_m + 1e-9 or (
                    lateral_tolerance_m is not None
                    and (
                        along < -1e-9
                        or along > length + 1e-9
                        or lateral > lateral_tolerance_m + 1e-9
                    )
                ):
                    continue
                candidates.append(tuple(float(v) for v in point))
        return tuple(candidates)

    def snap_to_navigable(
        self,
        world_xy: np.ndarray,
        *,
        search_radius_m: float = 2.0,
    ) -> Optional[np.ndarray]:
        """Return the closest body-clear cell inside a metric target disk.

        Distance is measured from the requested world point, not its rounded
        grid cell. There is no bearing restriction or preference for the
        robot's connected component. Route planning remains a separate step.
        """
        target_values = np.asarray(world_xy, dtype=np.float64).reshape(-1)
        if target_values.size != 2 or not np.all(np.isfinite(target_values)):
            raise ValueError("world_xy must contain two finite values")
        if not math.isfinite(float(search_radius_m)) or search_radius_m <= 0.0:
            raise ValueError("search_radius_m must be finite and positive")
        target = target_values
        rc = self.world_to_grid(target)
        r0, c0 = int(rc[0]), int(rc[1])
        radius_cells = int(math.ceil(search_radius_m / self.resolution))
        navigable = self.planning_navigable_mask()
        best_key: Optional[tuple[float, int, int]] = None
        best_rc: Optional[tuple[int, int]] = None
        for dr in range(-radius_cells, radius_cells + 1):
            for dc in range(-radius_cells, radius_cells + 1):
                r, c = r0 + dr, c0 + dc
                if not self._in_bounds(r, c) or not navigable[r, c]:
                    continue
                candidate = self.grid_to_world(np.array([r, c], dtype=np.float64))
                anchor_offset = candidate - target
                anchor_distance = float(np.linalg.norm(anchor_offset))
                if anchor_distance > search_radius_m + 1e-9:
                    continue
                key = (anchor_distance, r, c)
                if best_key is None or key < best_key:
                    best_key = key
                    best_rc = (r, c)
        if best_rc is None:
            return None
        return self.grid_to_world(np.array(best_rc, dtype=np.float64))

    def ground_approach_for_elevated(
        self,
        world_xyz: np.ndarray,
        *,
        approach_radius_m: float = 0.80,
        robot_xy: Optional[np.ndarray] = None,
    ) -> Optional[np.ndarray]:
        """For an object above the floor, find the nearest navigable approach point.

        Samples points on a circle of approach_radius_m around the XY footprint
        and returns the closest navigable candidate to the robot.
        """
        xy = np.asarray(world_xyz, dtype=np.float64)[:2]
        n_samples = 16
        angles = np.linspace(0.0, 2 * math.pi, n_samples, endpoint=False)
        candidates = []
        for theta in angles:
            pt = xy + approach_radius_m * np.array([math.cos(theta), math.sin(theta)])
            if self.is_navigable(pt):
                candidates.append(pt)
        if not candidates:
            return self.snap_to_navigable(xy, search_radius_m=approach_radius_m * 2.0)
        if robot_xy is not None:
            rxy = np.asarray(robot_xy, dtype=np.float64)
            candidates.sort(key=lambda p: float(np.linalg.norm(p - rxy)))
        return candidates[0]

    # ------------------------------------------------------------------
    # A* path planning
    # ------------------------------------------------------------------

    def astar_to_goal_region(
        self,
        start_xy: np.ndarray,
        target_xy: np.ndarray,
        *,
        lateral_tolerance_m: Optional[float] = 0.25,
        search_radius_m: float = 0.50,
        world_exclusion_zones: Sequence[tuple[float, float, float]] = (),
    ) -> tuple[Optional[list[np.ndarray]], Optional[np.ndarray], str]:
        """Search once to an allowed destination on a request-local actor mask."""
        candidates = self.goal_candidates(
            start_xy,
            target_xy,
            lateral_tolerance_m=lateral_tolerance_m,
            search_radius_m=search_radius_m,
            world_exclusion_zones=world_exclusion_zones,
        )
        if not candidates:
            return None, None, "no_valid_goal"
        start = np.asarray(start_xy, dtype=np.float64)
        planner = self._planner(world_exclusion_zones=world_exclusion_zones)
        output = planner.plan(
            PlanInput(
                start_xy=tuple(float(v) for v in start),
                start_yaw_rad=0.0,
                goal_candidates_xy=candidates,
            )
        )
        if output.status not in {"ok", "already_there"} or output.goal_xy is None:
            return None, None, output.status
        goal = np.asarray(output.goal_xy, dtype=np.float64)
        path = self._measured_start_path(start, goal, output, planner)
        return path, goal, "ok"

    def astar(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        *,
        world_exclusion_zones: Sequence[tuple[float, float, float]] = (),
    ) -> Optional[list[np.ndarray]]:
        """Plan with the exact A* implementation used by the Golden Oracle.

        The returned polyline includes the measured physical start. The goal
        is already resolved locally by :meth:`snap_to_navigable`. If map
        evidence marks the measured start unsafe, the shared planner's nearest
        safe start is retained as geometric search evidence, not permission
        for physical recovery. The navigation handler separately requires a
        clear measured start and a collision-free departure segment.
        """
        waypoints, _reason = self.astar_with_reason(
            start_xy,
            goal_xy,
            world_exclusion_zones=world_exclusion_zones,
        )
        return waypoints

    def astar_with_reason(
        self,
        start_xy: np.ndarray,
        goal_xy: np.ndarray,
        *,
        world_exclusion_zones: Sequence[tuple[float, float, float]] = (),
    ) -> tuple[Optional[list[np.ndarray]], str]:
        """Plan as :meth:`astar` and name why a ``None`` result failed.

        One of: ``no_map`` (no navigable evidence yet), ``goal_not_navigable``
        (the resolved goal cell is not body-clear), ``unreachable`` (no
        connected route), or ``goal_substituted`` (the planner returned a
        different goal). Fallback labels use the specific cause instead of
        one overloaded ``astar_map_disconnected`` string.
        """
        start = np.asarray(start_xy, dtype=np.float64).reshape(-1)
        goal = np.asarray(goal_xy, dtype=np.float64).reshape(-1)
        if start.size < 2 or goal.size < 2:
            return None, "no_map"
        goal_rc = tuple(int(value) for value in self.world_to_grid(goal[:2]))
        planner = self._planner(world_exclusion_zones=world_exclusion_zones)
        navigable = planner.active_navigable_mask()
        if not self._in_bounds(*goal_rc) or not navigable[goal_rc]:
            return None, "goal_not_navigable"
        output = planner.plan(
            PlanInput(
                start_xy=(float(start[0]), float(start[1])),
                start_yaw_rad=0.0,
                goal_xy=(float(goal[0]), float(goal[1])),
            )
        )
        if output.status == "already_there":
            return self._measured_start_path(start, goal, output, planner), "ok"
        if output.status != "ok" or output.goal_xy is None:
            return None, "unreachable"
        if math.dist(output.goal_xy, goal[:2].tolist()) > 1e-6:
            return None, "goal_substituted"
        return self._measured_start_path(start, goal, output, planner), "ok"

    def _measured_start_path(self, start, goal, output, planner) -> list[np.ndarray]:
        if output.status == "already_there":
            if np.linalg.norm(goal[:2] - start[:2]) > 1e-6:
                return [start[:2].copy(), goal[:2].copy()]
            return [start[:2].copy()]
        start_rc = tuple(int(value) for value in self.world_to_grid(start[:2]))
        active_navigable = planner.active_navigable_mask()
        start_is_safe = self._in_bounds(*start_rc) and bool(active_navigable[start_rc])
        waypoints = [
            np.asarray(waypoint, dtype=np.float64) for waypoint in output.waypoints
        ]
        # The shared planner represents its resolved start as the centre of
        # the containing safe cell. A measured pose is usually a few
        # centimetres off that centre, so it can appear as waypoint zero even
        # though no meaningful translation is required. Never turn this grid
        # quantisation residual into a physical G1 command: drop only leading
        # waypoints that map back to the already validated start cell.
        while (
            start_is_safe
            and waypoints
            and tuple(int(value) for value in self.world_to_grid(waypoints[0]))
            == start_rc
        ):
            waypoints.pop(0)
        # Re-run the shared planner's conservative supercover validation from
        # the measured pose and keep the farthest directly visible path
        # vertex. This prevents a one-cell A* bend (a few centimetres) from
        # becoming a meaningless physical action while preserving the exact
        # collision semantics used to simplify the Oracle path.
        if start_is_safe and len(waypoints) >= 2:
            for candidate in range(len(waypoints) - 1, 0, -1):
                if planner.shortcut_preserves_route_cost(
                    [start[:2], *waypoints[: candidate + 1]]
                ):
                    waypoints = waypoints[candidate:]
                    break
        return [start[:2].copy(), *waypoints]

    # ------------------------------------------------------------------
    # Visualisation
    # ------------------------------------------------------------------

    def dump_png(
        self,
        path: str,
        *,
        robot_xy: Optional[np.ndarray] = None,
        path_waypoints: Optional[list[np.ndarray]] = None,
        selected_xy: Optional[np.ndarray] = None,
        goal_xy: Optional[np.ndarray] = None,
        include_planning_buffer: bool = False,
    ) -> None:
        """Write one raw-occupancy diagnostic image.

        The default PNG is UNKNOWN / FREE / OCCUPIED plus robot, path, and
        goal marks. It does not paint G1 inflation: that overlay made narrow
        corridors look like a broken frontier. Explicit navigation debugging
        may request a sibling ``*_planning.png`` with the planner buffer.

        ``selected_xy`` is the model's requested relative PointGoal transformed
        into world space. ``goal_xy`` is the collision-free point after
        occupancy snapping. Keeping both visible makes a poor visual choice
        distinguishable from a planner correction.
        """
        try:
            from PIL import Image, ImageDraw
        except ImportError:
            import warnings

            warnings.warn("Pillow not available; skipping online map PNG dump.")
            return

        occupancy_rgb = np.full(
            (self._n, self._n, 3),
            _UNKNOWN_RGB,
            dtype=np.uint8,
        )
        occupancy_rgb[self._grid == self.FREE] = _FREE_RGB
        occupancy_rgb[self._grid == self.OCCUPIED] = _OCCUPIED_RGB

        planning_rgb = occupancy_rgb.copy()
        safe = self.planning_navigable_mask()
        buffer_mask = self.navigable_mask() & ~safe
        planning_rgb[buffer_mask] = _PLANNING_BUFFER_RGB

        # Crop the oversized runtime canvas to actual evidence plus one metre.
        # This changes only diagnostics; world/grid coordinates stay intact.
        crop_mask = self._observed | self._transient_robot_free
        for point in (robot_xy, selected_xy, goal_xy):
            if point is None:
                continue
            rc = self.world_to_grid(np.asarray(point))
            r, c = int(rc[0]), int(rc[1])
            if self._in_bounds(r, c):
                crop_mask[r, c] = True
        if path_waypoints:
            for point in path_waypoints:
                rc = self.world_to_grid(np.asarray(point))
                r, c = int(rc[0]), int(rc[1])
                if self._in_bounds(r, c):
                    crop_mask[r, c] = True
        if crop_mask.any():
            rows, cols = np.nonzero(crop_mask)
            margin = int(math.ceil(1.0 / self.resolution))
            row_offset = max(0, int(rows.min()) - margin)
            row_end = min(self._n, int(rows.max()) + margin + 1)
            col_offset = max(0, int(cols.min()) - margin)
            col_end = min(self._n, int(cols.max()) + margin + 1)
        else:
            row_offset, row_end = 0, self._n
            col_offset, col_end = 0, self._n

        def compose_image(
            rgb: np.ndarray,
            *,
            legend: tuple[tuple[tuple[int, int, int], str], ...] = (),
        ):
            cropped = rgb[row_offset:row_end, col_offset:col_end]
            img = Image.fromarray(cropped)
            draw = ImageDraw.Draw(img)

            def image_point(world_xy: np.ndarray) -> tuple[int, int]:
                rc = self.world_to_grid(np.asarray(world_xy))
                return int(rc[1]) - col_offset, int(rc[0]) - row_offset

            if path_waypoints and len(path_waypoints) >= 2:
                pts = [image_point(np.asarray(wp)) for wp in path_waypoints]
                for i in range(len(pts) - 1):
                    draw.line([pts[i], pts[i + 1]], fill=(30, 110, 255), width=2)

            if selected_xy is not None:
                c, r = image_point(np.asarray(selected_xy))
                draw.line(
                    [(c - 5, r - 5), (c + 5, r + 5)],
                    fill=(245, 190, 30),
                    width=2,
                )
                draw.line(
                    [(c - 5, r + 5), (c + 5, r - 5)],
                    fill=(245, 190, 30),
                    width=2,
                )

            if goal_xy is not None:
                c, r = image_point(np.asarray(goal_xy))
                draw.ellipse(
                    [(c - 5, r - 5), (c + 5, r + 5)],
                    outline=(220, 50, 50),
                    width=2,
                )

            if robot_xy is not None:
                c, r = image_point(np.asarray(robot_xy))
                draw.ellipse(
                    [(c - 3, r - 3), (c + 3, r + 3)],
                    fill=(50, 200, 80),
                )

            # Preserve metric occupancy cells. NEAREST keeps UNKNOWN / FREE /
            # OCCUPIED sharp; denser mapping, not a blurrier upscale, is what
            # removes the 3 px jaggy look.
            longest_side = max(img.size)
            if longest_side and longest_side < 640:
                scale = int(math.ceil(640 / longest_side))
                img = img.resize(
                    (img.width * scale, img.height * scale),
                    resample=Image.Resampling.NEAREST,
                )
            if legend:
                swatch = 14
                pad = 6
                bar = Image.new("RGB", (img.width, swatch + 2 * pad), (236, 236, 236))
                bar_draw = ImageDraw.Draw(bar)
                cursor = pad
                for color, label in legend:
                    bar_draw.rectangle(
                        [cursor, pad, cursor + swatch, pad + swatch],
                        fill=color,
                        outline=(40, 40, 40),
                    )
                    bar_draw.text(
                        (cursor + swatch + 4, pad),
                        label,
                        fill=(30, 30, 30),
                    )
                    cursor += swatch + 8 + 7 * len(label)
                stacked = Image.new("RGB", (img.width, img.height + bar.height))
                stacked.paste(img, (0, 0))
                stacked.paste(bar, (0, img.height))
                img = stacked
            return img

        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        compose_image(occupancy_rgb).save(output)
        if not include_planning_buffer or output.stem.endswith("_planning"):
            return
        compose_image(
            planning_rgb,
            legend=(
                (_UNKNOWN_RGB, "unknown"),
                (_FREE_RGB, "free"),
                (_OCCUPIED_RGB, "occupied"),
                (_PLANNING_BUFFER_RGB, "G1 buffer"),
            ),
        ).save(output.with_name(f"{output.stem}_planning{output.suffix}"))

    def dump_npz(self, path: str) -> None:
        """Persist raw map state for offline comparison with official GT."""

        Path(path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            grid=self._grid,
            evidence=self._evidence,
            observed=self._observed,
            ground_free=self._ground_free,
            voxel_observed=self._voxel_observed,
            voxel_occupied=self._voxel_occupied,
            lidar_free=self._lidar_free,
            lidar_occupied=self._lidar_occupied,
            transient_robot_free=self._transient_robot_free,
            pose_history_xy=np.asarray(
                self._pose_history_xy,
                dtype=np.float64,
            ).reshape(-1, 2),
            origin_xy=self._origin,
            resolution_m=np.asarray(self.resolution, dtype=np.float64),
            voxel_z_resolution_m=np.asarray(
                self.voxel_z_resolution_m,
                dtype=np.float64,
            ),
            obstacle_z_range_m=np.asarray(
                [
                    self.obstacle_z_min_m,
                    self.obstacle_z_max_m,
                ],
                dtype=np.float64,
            ),
            robot_radius_m=np.asarray(self.robot_radius_m, dtype=np.float64),
            map_source=np.asarray(self.map_source),
        )


__all__ = ["ONLINE_DEPTH_SUBSAMPLE", "OnlineOccupancyGrid"]
