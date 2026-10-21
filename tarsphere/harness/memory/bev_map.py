"""Bird's-eye-view occupancy / navigable map (pure numpy).

The map is a top-down grid in world metres.  Each ``update()`` integrates one
RGBD frame: obstacle points (depth back-projected to the world frame and
height-filtered) mark occupied cells, and the cells the camera can see through
to are marked known-and-free.  The map answers one question — *where can the
robot walk* — and the planner reads exactly that from ``navigable_mask``.

It deliberately does **not** model exploration frontiers or semantics. Both
GT-backed subclasses mark every cell known at construction (the offline raster
already labels the whole floor free or occupied), so an unknown set never
exists and a frontier is by definition empty.

Free-space marking walks the grid cells along each segment from the camera
cell to a back-projected obstacle point (Bresenham in cell space), stopping
just before the obstacle. Cells behind the obstacle remain unknown.

The module is intentionally dependency-free at import time; scipy is used only
in :meth:`BevMap._dilate` when available, with a pure-numpy implementation
otherwise.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from tarsphere.harness.memory.geometry import (
    camera_to_world,
    depth_to_camera_points,
    resolve_intrinsics,
)
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


@dataclass
class BevMapState:
    """Public, serializable snapshot of the current BEV map for decision."""

    occupied_cells: int = 0
    known_cells: int = 0
    free_cells: int = 0
    total_cells: int = 0
    bounds_m: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    last_pose_xy: Optional[list[float]] = None
    # World [x, y] of the occupied cell nearest the last pose; lets a decision
    # module describe obstacle bearings without the raw mask.  None when there
    # is no pose or no occupied cell yet.
    nearest_obstacle_xy: Optional[list[float]] = None
    metadata: dict[str, Any] = field(default_factory=dict)


class BevMap:
    """Accumulating top-down occupancy map.

    Args:
        size_m: half-extent of the map in metres along each axis; the full map
            covers ``[-size_m, +size_m]`` in both X and Y, centred on the world
            origin (the robot starts near the centre).
        pixels_per_meter: grid resolution.  ``20`` gives 5 cm cells.
        min_height_m / max_height_m: world-Z band kept as obstacles; the floor
            (below ``min_height_m``) and the ceiling / overhead clutter (above
            ``max_height_m``) are ignored.
        max_depth_m: maximum reliable depth to integrate per frame.
        agent_radius_m: obstacle inflation radius so navigable cells clear the
            robot footprint (used by :meth:`navigable_mask`).
    """

    def __init__(
        self,
        *,
        size_m: float = 25.0,
        pixels_per_meter: int = 20,
        min_height_m: float = 0.10,
        max_height_m: float = 1.80,
        max_depth_m: float = 5.0,
        agent_radius_m: float = G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    ):
        if size_m <= 0 or pixels_per_meter <= 0:
            raise ValueError("size_m and pixels_per_meter must be positive.")
        self.size_m = float(size_m)
        self.pixels_per_meter = int(pixels_per_meter)
        self.min_height_m = float(min_height_m)
        self.max_height_m = float(max_height_m)
        self.max_depth_m = float(max_depth_m)
        self.agent_radius_m = float(agent_radius_m)

        self._cells_per_axis = int(2.0 * self.size_m * self.pixels_per_meter)
        self._center = self._cells_per_axis / 2.0
        # Occupancy / coverage grids (bool).  A cell is one of: unknown, free,
        # or occupied.  occupied=True overrides free.
        self._occupied = np.zeros(
            (self._cells_per_axis, self._cells_per_axis), dtype=bool
        )
        self._known = np.zeros_like(self._occupied)
        self._last_pose_xy: Optional[np.ndarray] = None

    def reset(self) -> None:
        self._occupied.fill(False)
        self._known.fill(False)
        self._last_pose_xy = None

    # ------------------------------------------------------------------
    # Coordinate conversion
    # ------------------------------------------------------------------

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        """World (x, y) -> integer (row, col) cells, clipped to the grid."""
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return np.zeros((0, 2), dtype=int)
        xy = xy.reshape(-1, 2)
        x = xy[..., 0]
        y = xy[..., 1]
        col = np.rint(x * self.pixels_per_meter + self._center).astype(int)
        # +Y world points up the image -> smaller row.
        row = np.rint(self._center - y * self.pixels_per_meter).astype(int)
        cells = np.stack([row, col], axis=-1)
        return self._clip_cells(cells)

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        x = (float(col) - self._center) / self.pixels_per_meter
        y = (self._center - float(row)) / self.pixels_per_meter
        return x, y

    def _clip_cells(self, cells: np.ndarray) -> np.ndarray:
        n = self._cells_per_axis
        cells[:, 0] = np.clip(cells[:, 0], 0, n - 1)
        cells[:, 1] = np.clip(cells[:, 1], 0, n - 1)
        return cells

    # ------------------------------------------------------------------
    # Integration
    # ------------------------------------------------------------------

    def update(
        self,
        points_world: np.ndarray,
        pose_xy: tuple[float, float],
        yaw_rad: float,
        *,
        horizontal_fov_deg: float = 90.0,
    ) -> None:
        """Integrate one frame's world-frame points into the map.

        ``pose_xy`` is the robot ``(x, y)`` and ``yaw_rad`` its heading.  Free
        space is marked along observed depth rays; ``horizontal_fov_deg``
        bounds the angular region considered.
        """
        pose_xy_arr = np.asarray(pose_xy, dtype=float).reshape(2)
        self._last_pose_xy = pose_xy_arr

        self._mark_obstacles(points_world)
        self._mark_free_raycast(points_world, pose_xy_arr, yaw_rad, horizontal_fov_deg)

    def record_pose(self, x: float, y: float) -> None:
        """Update the robot marker without integrating sensor data."""
        self._last_pose_xy = np.asarray([float(x), float(y)], dtype=float)

    def integrate_depth(
        self,
        depth: np.ndarray,
        pose: dict[str, Any],
        camera_params: Optional[dict[str, Any]] = None,
        *,
        horizontal_fov_deg: float = 90.0,
    ) -> None:
        """Convenience: back-project depth and integrate in one call.

        ``pose`` must carry ``position`` (``[x, y, z]``) and ``yaw_rad``; it may
        optionally carry ``camera_height_m`` and ``camera_pitch_deg``.
        """
        position = list(pose.get("position") or [0.0, 0.0, 0.0])
        if len(position) < 3:
            position = list(position) + [0.0] * (3 - len(position))
        yaw = float(pose.get("yaw_rad", pose.get("yaw", 0.0)) or 0.0)
        camera_height = float(pose.get("camera_height_m", 0.0) or 0.0)
        camera_pitch = float(pose.get("camera_pitch_deg", 0.0) or 0.0)

        fx, fy, cx, cy = resolve_intrinsics(
            camera_params, depth.shape, horizontal_fov_deg
        )
        points_cam = depth_to_camera_points(
            depth, fx, fy, cx, cy, max_depth_m=self.max_depth_m
        )
        points_world = camera_to_world(
            points_cam,
            position,
            yaw,
            camera_height_m=camera_height,
            camera_pitch_deg=camera_pitch,
        )
        self.update(
            points_world,
            (float(position[0]), float(position[1])),
            yaw,
            horizontal_fov_deg=horizontal_fov_deg,
        )

    def _mark_obstacles(self, points_world: np.ndarray) -> None:
        pts = np.asarray(points_world, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return
        in_band = (pts[:, 2] >= self.min_height_m) & (pts[:, 2] <= self.max_height_m)
        obstacles = pts[in_band][:, :2]
        if obstacles.shape[0] == 0:
            return
        cells = self.world_to_cell(obstacles)
        self._occupied[cells[:, 0], cells[:, 1]] = True

    def _mark_free_raycast(
        self,
        points_world: np.ndarray,
        pose_xy: np.ndarray,
        yaw_rad: float,
        horizontal_fov_deg: float,
    ) -> None:
        """Mark cells between the camera and each obstacle point as known-free.

        For every back-projected obstacle point that lies inside the height
        band and within the camera's FOV/range, walk the grid cells along the
        segment from the camera cell to the obstacle cell (Bresenham in cell
        space) marking each as known, stopping just before the obstacle cell
        — the obstacle itself is already marked occupied by
        :meth:`_mark_obstacles`. Cells behind an obstacle are
        never reached, so they stay unknown instead of being over-marked free.

        Points are subsampled deterministically (every Nth) for speed.
        """
        pts = np.asarray(points_world, dtype=float)
        if pts.ndim != 2 or pts.shape[0] == 0:
            return
        # Height-band filter identical to _mark_obstacles so the ray endpoints
        # match what actually got marked occupied.
        in_band = (pts[:, 2] >= self.min_height_m) & (pts[:, 2] <= self.max_height_m)
        band_pts = pts[in_band][:, :2]
        if band_pts.shape[0] == 0:
            return

        rel = band_pts - pose_xy.reshape(1, 2)
        dist = np.sqrt(rel[:, 0] ** 2 + rel[:, 1] ** 2)
        # Range filter (depth beyond max_depth_m is unreliable).
        in_range = dist <= self.max_depth_m
        # FOV filter: only treat points inside the forward cone as ray ends.
        half_fov = math.radians(float(horizontal_fov_deg)) / 2.0
        bearing = np.arctan2(rel[:, 1], rel[:, 0])
        in_fov = np.abs(_wrap_angle(bearing - yaw_rad)) <= half_fov
        valid = in_range & in_fov
        band_pts = band_pts[valid]
        if band_pts.shape[0] == 0:
            return

        # Deterministic subsampling bounds raycasting cost.
        n = band_pts.shape[0]
        stride = 4 if n > 64 else 1
        if stride > 1:
            band_pts = band_pts[::stride]

        cam_cell = self.world_to_cell(pose_xy.reshape(1, 2))[0]
        cam_r, cam_c = int(cam_cell[0]), int(cam_cell[1])
        end_cells = self.world_to_cell(band_pts)
        n_grid = self._cells_per_axis
        known = self._known

        for i in range(end_cells.shape[0]):
            er, ec = int(end_cells[i, 0]), int(end_cells[i, 1])
            if (er, ec) == (cam_r, cam_c):
                continue
            # Bresenham over the obstacle cell, marking all cells strictly
            # before it as known (the obstacle cell stays to _mark_obstacles).
            for r, c in _bresenham_cells(cam_r, cam_c, er, ec, n_grid):
                if (r, c) != (er, ec):
                    known[r, c] = True

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def free_mask(self) -> np.ndarray:
        return self._known & ~self._occupied

    def navigable_mask(self) -> np.ndarray:
        """Free cells inflated-clear of obstacles by the agent radius."""
        free = self.free_mask()
        if not free.any() or self.agent_radius_m <= 0:
            return free
        inflation = max(1, int(round(self.agent_radius_m * self.pixels_per_meter)))
        blocked = self._dilate(self._occupied, inflation)
        return free & ~blocked

    def is_line_collision_free(
        self,
        p0_xy,
        p1_xy,
        *,
        sample_step_m: float = 0.10,
    ) -> bool:
        """Return True iff the segment ``p0 -> p1`` is fully navigable.

        Samples world points along the segment at ``sample_step_m`` resolution
        and converts each to a grid cell.  The segment is collision-free only
        if *every* sampled cell is navigable (free, known, and clear of the
        inflated obstacle footprint).  Used by the planner for path
        simplification and by the controller for segment gating.
        """
        p0 = np.asarray(p0_xy, dtype=float).reshape(2)
        p1 = np.asarray(p1_xy, dtype=float).reshape(2)
        if sample_step_m <= 0:
            raise ValueError("sample_step_m must be positive.")
        seg = p1 - p0
        length = float(np.linalg.norm(seg))
        if length < 1e-9:
            # Degenerate segment: just check the single cell.
            cell = self.world_to_cell(p0.reshape(1, 2))[0]
            return bool(self.navigable_mask()[int(cell[0]), int(cell[1])])
        n_steps = max(1, int(math.ceil(length / sample_step_m)))
        ts = np.linspace(0.0, 1.0, n_steps + 1)
        samples = p0.reshape(1, 2) + ts.reshape(-1, 1) * seg.reshape(1, 2)
        cells = self.world_to_cell(samples)
        nav = self.navigable_mask()
        return bool(nav[cells[:, 0], cells[:, 1]].all())

    def nearest_free_cell(
        self,
        xy,
        *,
        max_search_m: float = 2.0,
    ) -> Optional[tuple[float, float]]:
        """Snap ``xy`` to the nearest navigable cell within ``max_search_m``.

        Searches outward from the cell containing ``xy`` over increasing
        Chebyshev (square-ring) radius, returning the world ``(x, y)`` of the
        first navigable cell found.  Returns ``None`` if no navigable cell
        exists within the radius.  Used to snap start/goal onto the mask.
        """
        if max_search_m <= 0:
            return None
        cell = self.world_to_cell(np.asarray(xy, dtype=float).reshape(1, 2))[0]
        cr, cc = int(cell[0]), int(cell[1])
        nav = self.navigable_mask()
        n_grid = self._cells_per_axis
        max_radius = max(1, int(math.ceil(max_search_m * self.pixels_per_meter)))

        # Radius 0 first (the query cell itself).
        if 0 <= cr < n_grid and 0 <= cc < n_grid and nav[cr, cc]:
            return self.cell_to_world(cr, cc)

        for radius in range(1, max_radius + 1):
            # Walk the clipped Chebyshev ring without revisiting its corners.
            top = max(cr - radius, 0)
            bot = min(cr + radius, n_grid - 1)
            left = max(cc - radius, 0)
            right = min(cc + radius, n_grid - 1)
            if top > bot or left > right:
                continue
            for c in range(left, right + 1):
                if nav[top, c]:
                    return self.cell_to_world(top, c)
                if top != bot and nav[bot, c]:
                    return self.cell_to_world(bot, c)
            for r in range(top + 1, bot):
                if nav[r, left]:
                    return self.cell_to_world(r, left)
                if left != right and nav[r, right]:
                    return self.cell_to_world(r, right)
        return None

    def cell_to_world_vec(self, rows: np.ndarray, cols: np.ndarray):
        x = (cols.astype(float) - self._center) / self.pixels_per_meter
        y = (self._center - rows.astype(float)) / self.pixels_per_meter
        return x, y

    def world_bounds_m(self) -> tuple[float, float, float, float]:
        """Return the map's covered world bounds as ``xmin, ymin, xmax, ymax``."""
        return (-self.size_m, -self.size_m, self.size_m, self.size_m)

    def snapshot(self) -> BevMapState:
        occupied = int(self._occupied.sum())
        known = int(self._known.sum())
        total = int(self._occupied.size)
        last_xy = (
            [float(self._last_pose_xy[0]), float(self._last_pose_xy[1])]
            if self._last_pose_xy is not None
            else None
        )
        return BevMapState(
            occupied_cells=occupied,
            known_cells=known,
            free_cells=max(0, known - occupied),
            total_cells=total,
            bounds_m=self.world_bounds_m(),
            last_pose_xy=last_xy,
            nearest_obstacle_xy=self._nearest_obstacle_xy(),
            metadata={
                "pixels_per_meter": self.pixels_per_meter,
                "size_m": self.size_m,
                "max_depth_m": self.max_depth_m,
            },
        )

    def _nearest_obstacle_xy(self) -> Optional[list[float]]:
        """World [x, y] of the occupied cell closest to the last pose, or None."""
        if self._last_pose_xy is None or not self._occupied.any():
            return None
        rows, cols = np.nonzero(self._occupied)
        xs, ys = self.cell_to_world_vec(rows, cols)
        dx = xs - self._last_pose_xy[0]
        dy = ys - self._last_pose_xy[1]
        idx = int(np.argmin(dx * dx + dy * dy))
        return [float(xs[idx]), float(ys[idx])]

    # ------------------------------------------------------------------
    # Small morphological helpers (numpy-only, no scipy/cv2)
    # ------------------------------------------------------------------

    @staticmethod
    def _shift(mask: np.ndarray, dr: int, dc: int) -> np.ndarray:
        shifted = np.zeros_like(mask)
        height, width = mask.shape
        src_r0 = max(0, -dr)
        src_r1 = min(height, height - dr)
        src_c0 = max(0, -dc)
        src_c1 = min(width, width - dc)
        dst_r0 = max(0, dr)
        dst_r1 = dst_r0 + (src_r1 - src_r0)
        dst_c0 = max(0, dc)
        dst_c1 = dst_c0 + (src_c1 - src_c0)
        if src_r1 > src_r0 and src_c1 > src_c0:
            shifted[dst_r0:dst_r1, dst_c0:dst_c1] = mask[src_r0:src_r1, src_c0:src_c1]
        return shifted

    def _dilate(self, mask: np.ndarray, radius: int) -> np.ndarray:
        """Dilate a boolean mask by ``radius`` (agent footprint).

        Uses :func:`scipy.ndimage.binary_dilation` with a disk structuring
        element when scipy is importable; otherwise uses iterative 4-neighbour
        OR shifts (a Manhattan-diamond structuring element). Both
        paths produce a superset of the input cells, so :meth:`navigable_mask`
        only ever *removes* free cells.
        """
        if radius <= 0:
            return mask.copy()
        try:
            from scipy.ndimage import binary_dilation  # lazy-optional
        except ImportError:
            binary_dilation = None  # type: ignore[assignment]

        if binary_dilation is not None:
            yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
            struct = (xx * xx + yy * yy) <= radius * radius  # disk
            return binary_dilation(mask, structure=struct, border_value=0)

        result = mask.copy()
        for _ in range(radius):
            result = (
                result
                | self._shift(result, 1, 0)
                | self._shift(result, -1, 0)
                | self._shift(result, 0, 1)
                | self._shift(result, 0, -1)
            )
        return result


def _wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Wrap radians to (-pi, pi]."""
    return (angle + math.pi) % (2.0 * math.pi) - math.pi


def _bresenham_cells(r0: int, c0: int, r1: int, c1: int, n: int):
    """Yield integer ``(row, col)`` grid cells along the segment ``(r0,c0)->(r1,c1)``.

    Standard integer Bresenham over cell space.  Cells outside the ``n x n``
    grid are skipped (the line is clipped implicitly).  The endpoint
    ``(r1, c1)`` *is* included so callers can decide whether to keep it.
    """
    dr = abs(r1 - r0)
    dc = abs(c1 - c0)
    sr = 1 if r1 >= r0 else -1
    sc = 1 if c1 >= c0 else -1
    err = dr - dc
    r, c = r0, c0
    while True:
        if 0 <= r < n and 0 <= c < n:
            yield (r, c)
        if r == r1 and c == c1:
            break
        e2 = 2 * err
        if e2 > -dc:
            err -= dc
            r += sr
        if e2 < dr:
            err += dr
            c += sc
