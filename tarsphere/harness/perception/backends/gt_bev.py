"""GT (oracle) BEV backend — static occupancy map from official scene metadata.

Replaces LingBot's online monocular reconstruction with the per-scene official
GRScene occupancy map shipped as ``benchmark/meta/<scene_id>/occupancy.npy``.
That file is the canonical navigation layer used by TARSphere metadata:
row 0 stores world-X coordinates, column 0 stores world-Y coordinates, and
interior value ``1`` marks navigable floor. ``bev_rgb.png`` is still used as
the visual top-down RGB for the model, but it is not the navigation truth.

The older TARSPHERE-local ``bev.png`` USD mesh-footprint raster is retained only as
a compatibility fallback for tests or scenes that do not ship official meta.
It is not preferred because it rasterizes mesh AABBs and can close real doors.

Layout of this module:

- :class:`OfficialOccupancyBevMap` — a :class:`BevMap` subclass backed by
  TARSphere's official ``occupancy.npy``. It uses the file's coordinate
  axes directly and does not apply another agent-radius dilation; the official
  map is already the evaluated navigation layer.
- :class:`GTBevMap` — a legacy :class:`BevMap` subclass whose coordinate methods
  honour the real scene bounds (``bev_bounds_min_xy`` / ``bev_span_m``)
  instead of BevMap's world-origin-centred ``[-size_m, +size_m]`` grid.
  The occupancy / known layers are baked once from the PNG and never
  re-integrated from ego depth (the map is static for the episode).
- :class:`GTBevBackend` — locates the per-scene ``bev.png`` /
  ``bev_meta.json`` for a given episode / scene asset, loads them, and
  hands back a ready :class:`GTBevMap`. Held by :class:`EpisodicMemory`
  in place of the old LingBot backend.

Both classes are importable without Isaac / torch (pure numpy + PIL + a
JSON read) so the headless mock path and unit tests can construct them.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Optional

import numpy as np

from tarsphere.assets import tarsphere_assets_root
from tarsphere.harness.memory.bev_map import BevMap
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


OFFICIAL_NAVIGABLE_VALUE = 1


class OfficialOccupancyBevMap(BevMap):
    """A :class:`BevMap` backed by TARSphere's official ``occupancy.npy``.

    The official array layout is:
        - ``occupancy[0, 1:]``: world-X coordinate for each data column.
        - ``occupancy[1:, 0]``: world-Y coordinate for each data row.
        - ``occupancy[1:, 1:]``: occupancy labels; value ``1`` is navigable.

    This backend returns the baked raw floor/obstacle layer without inflation.
    The shared runtime planner must apply the actual footprint once. Prepared
    runtime masks already include that footprint and must not be inflated again.
    """

    def __init__(
        self,
        *,
        occupancy_path: str | os.PathLike,
        navigable_value: int = OFFICIAL_NAVIGABLE_VALUE,
    ) -> None:
        arr = np.load(occupancy_path)
        if arr.ndim != 2 or arr.shape[0] < 3 or arr.shape[1] < 3:
            raise ValueError(f"Invalid official occupancy map shape: {arr.shape!r}")
        if arr.shape[0] != arr.shape[1]:
            raise ValueError(
                f"Official occupancy map must be square, got {arr.shape!r}"
            )

        self._occupancy_path = str(occupancy_path)
        self._official_occupancy = np.asarray(arr, dtype=np.float32)
        self._x_axis = np.asarray(arr[0, 1:], dtype=float)
        self._y_axis = np.asarray(arr[1:, 0], dtype=float)
        if not np.all(np.isfinite(self._x_axis)) or not np.all(
            np.isfinite(self._y_axis)
        ):
            raise ValueError(
                f"Official occupancy axes contain non-finite values: {occupancy_path}"
            )
        if not np.all(np.diff(self._x_axis) > 0):
            raise ValueError("Official occupancy X axis must be strictly increasing.")
        if not np.all(np.diff(self._y_axis) < 0):
            raise ValueError("Official occupancy Y axis must be strictly decreasing.")

        span_x = float(self._x_axis[-1] - self._x_axis[0])
        span_y = float(self._y_axis[0] - self._y_axis[-1])
        span = max(span_x, span_y, 1e-6)
        cell_size = float(np.median(np.diff(self._x_axis)))
        axis_epsilon = (
            np.finfo(arr.dtype).eps if np.issubdtype(arr.dtype, np.floating) else 0.0
        )
        axis_roundoff = max(
            1e-8,
            4
            * axis_epsilon
            * max(
                1.0,
                float(np.max(np.abs(self._x_axis))),
                float(np.max(np.abs(self._y_axis))),
            ),
        )
        if not np.allclose(
            np.diff(self._x_axis), cell_size, rtol=1e-5, atol=axis_roundoff
        ) or not np.allclose(
            -np.diff(self._y_axis), cell_size, rtol=1e-5, atol=axis_roundoff
        ):
            raise ValueError("Official occupancy requires uniform square metric cells.")
        ppm = 1.0 / cell_size
        super().__init__(
            size_m=span / 2.0 + 1e-6,
            pixels_per_meter=ppm,
            min_height_m=0.0,
            max_height_m=1e9,
            max_depth_m=5.0,
            agent_radius_m=0.0,
        )
        # BevMap accepts integer sensor-map resolution; official metric axes
        # must retain precision (100.168 px/m must not become 100 and lose an
        # entire clearance cell). Coordinate methods below use those axes.
        self.pixels_per_meter = ppm

        n = int(arr.shape[0])
        self._cells_per_axis = n
        self._center = n / 2.0
        free = np.zeros((n, n), dtype=bool)
        free[1:, 1:] = arr[1:, 1:] == float(navigable_value)
        self._official_free = free
        self._occupied = ~free
        # The official layer is complete; first row/col are coordinate axes
        # and intentionally non-navigable.
        self._known = np.ones((n, n), dtype=bool)
        self._last_pose_xy: Optional[np.ndarray] = None
        self._bounds_min = np.array([self._x_axis[0], self._y_axis[-1]], dtype=float)
        self._bounds_max = np.array([self._x_axis[-1], self._y_axis[0]], dtype=float)
        self._span = span

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return np.zeros((0, 2), dtype=int)
        xy = xy.reshape(-1, 2)
        cols = _nearest_axis_indices(self._x_axis, xy[:, 0]) + 1
        rows = (
            len(self._y_axis) - 1 - _nearest_axis_indices(self._y_axis[::-1], xy[:, 1])
        ) + 1
        cells = np.stack([rows, cols], axis=-1).astype(int)
        return self._clip_cells(cells)

    @property
    def occupancy_path(self) -> str:
        """Absolute source path for explicit diagnostic map substitution."""

        return self._occupancy_path

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        rr = int(np.clip(int(row), 1, self._cells_per_axis - 1))
        cc = int(np.clip(int(col), 1, self._cells_per_axis - 1))
        return float(self._x_axis[cc - 1]), float(self._y_axis[rr - 1])

    def cell_to_world_vec(self, rows: np.ndarray, cols: np.ndarray):
        rr = np.clip(rows.astype(int), 1, self._cells_per_axis - 1)
        cc = np.clip(cols.astype(int), 1, self._cells_per_axis - 1)
        return self._x_axis[cc - 1], self._y_axis[rr - 1]

    def world_bounds_m(self) -> tuple[float, float, float, float]:
        return (
            float(self._bounds_min[0]),
            float(self._bounds_min[1]),
            float(self._bounds_max[0]),
            float(self._bounds_max[1]),
        )

    def navigable_mask(self) -> np.ndarray:
        """Return the official navigable layer directly, with no extra dilation."""
        return self._official_free.copy()

    def reset(self) -> None:
        return

    def update(self, *args, **kwargs) -> None:  # noqa: D401
        """No-op: official occupancy is static; only track the latest pose."""
        pose = kwargs.get("pose_xy")
        if pose is None and args:
            pose = args[1] if len(args) > 1 else None
        if pose is not None:
            try:
                self._last_pose_xy = np.asarray(pose, dtype=float).reshape(2)
            except (ValueError, TypeError):
                pass
        return

    def integrate_depth(self, *args, **kwargs) -> None:  # noqa: D401
        """No-op: official occupancy is static; only track the latest pose."""
        pose = kwargs.get("pose")
        if pose is None and args:
            pose = args[1] if len(args) > 1 else None
        if isinstance(pose, dict):
            position = pose.get("position")
            if position and len(position) >= 2:
                self._last_pose_xy = np.asarray(
                    [float(position[0]), float(position[1])], dtype=float
                )
        return

    def snapshot(self):
        state = super().snapshot()
        state.metadata.update(
            {
                "source": "official_occupancy_npy",
                "occupancy_path": self._occupancy_path,
                "navigable_value": OFFICIAL_NAVIGABLE_VALUE,
            }
        )
        return state


def _nearest_axis_indices(axis: np.ndarray, values: np.ndarray) -> np.ndarray:
    idx = np.searchsorted(axis, values, side="left")
    idx = np.clip(idx, 0, len(axis) - 1)
    prev = np.clip(idx - 1, 0, len(axis) - 1)
    choose_prev = np.abs(values - axis[prev]) <= np.abs(values - axis[idx])
    return np.where(choose_prev, prev, idx).astype(int)


# ---------------------------------------------------------------------------
# Coordinate conventions for the offline GT BEV raster
# ---------------------------------------------------------------------------
# bev_meta.json (see assets/.../bev_meta.json) carries:
#   bev_bounds_min_xy : [xmin, ymin]   world metres, top-left of the raster
#   bev_span_m        : float          edge length of the (square) raster
#   resolution        : int            pixels per edge (e.g. 1440)
#   scene_scale       : [sx, sy, sz]   USD stage->world scale (usually 0.01)
# The PNG is laid out image-style: row 0 = +Y (top), row N = -Y (bottom);
# col 0 = +X-min (left), col N = +X-max (right). Pixel value 0 (black) is
# occupied/unknown, non-zero is free floor. We mirror BevMap's convention
# (world +X -> +col, world +Y -> -row) by deriving cells from world coords
# directly, so the PNG only seeds the occupancy/known boolean layers.


class GTBevMap(BevMap):
    """A :class:`BevMap` seeded from the offline GT BEV raster.

    Unlike the base class (world-origin-centred, ``[-size_m, +size_m]``),
    this map covers an arbitrary rectangular world region defined by the
    scene's ``bev_bounds_min_xy`` + ``bev_span_m``. Coordinate conversions
    are overridden to map world (x, y) into that region's pixel grid.

    The occupancy layer is baked once at construction from ``bev.png`` and
    is treated as fully known across the covered region (the GT raster
    already marks every cell as free or occupied). :meth:`reset` is a
    no-op so the baked layer survives :meth:`EpisodicMemory.reset`.
    """

    def __init__(
        self,
        *,
        bev_png_path: str | os.PathLike,
        bev_bounds_min_xy: tuple[float, float],
        bev_span_m: float,
        resolution: int,
        pixels_per_meter: Optional[int] = None,
        agent_radius_m: float = G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        occupied_threshold: int = 100,
    ) -> None:
        if bev_span_m <= 0 or resolution <= 1:
            raise ValueError("bev_span_m and resolution must be positive.")
        # Choose the grid resolution: honour an explicit ppm override,
        # otherwise keep the raster's native pixels-per-metre so we do not
        # silently resample the GT layer.
        if pixels_per_meter is None:
            pixels_per_meter = max(1, int(round(float(resolution) / float(bev_span_m))))
        # The base BevMap grid is square with cells_per_axis = 2*size_m*ppm.
        # We size it so it covers at least the GT span, then keep our own
        # bounds for the world<->cell math.
        size_m = float(bev_span_m) / 2.0 + 1e-6
        # Bump size so cells_per_axis lands on the raster resolution exactly
        # when ppm is native (avoids a 1-cell rounding gap).
        super().__init__(
            size_m=size_m,
            pixels_per_meter=int(pixels_per_meter),
            # Height band is irrelevant — the raster already encodes occupancy;
            # these only affect live depth integration, which we do not use.
            min_height_m=0.0,
            max_height_m=1e9,
            max_depth_m=5.0,
            agent_radius_m=agent_radius_m,
        )
        self._bounds_min = np.array(
            [float(bev_bounds_min_xy[0]), float(bev_bounds_min_xy[1])],
            dtype=float,
        )
        self._span = float(bev_span_m)
        self._resolution = int(resolution)
        self._ppm_native = float(resolution) / float(bev_span_m)
        self._bake_occupancy(bev_png_path, occupied_threshold)
        # Mark the whole covered region as known (the GT raster settles
        # every cell as free or occupied; there is no "unknown" inside it).
        self._known[:, :] = True

    # ------------------------------------------------------------------
    # Baking
    # ------------------------------------------------------------------

    def _bake_occupancy(
        self, png_path: str | os.PathLike, occupied_threshold: int
    ) -> None:
        """Seed ``_occupied`` from the GT PNG.

        ``occupied_threshold`` is the grayscale value below which a pixel is
        treated as occupied (the offline raster draws floor as bright — value
        ~224/246 — and obstacles as dark — value ~49-73; the default 100
        splits the two clusters cleanly). The GT raster covers the whole
        planner grid; world coordinates are mapped into this raster by
        ``bev_bounds_min_xy`` / ``bev_span_m``.
        """
        try:
            from PIL import Image
        except ImportError as exc:  # pragma: no cover - PIL is a hard dep
            raise RuntimeError("Pillow is required to load the GT BEV PNG") from exc

        img = Image.open(png_path)
        if img.mode not in ("L", "P", "RGB", "RGBA"):
            img = img.convert("L")
        elif img.mode != "L":
            img = img.convert("L")
        arr = np.asarray(img, dtype=np.uint8)
        if arr.ndim == 3:  # safety after convert('L')
            arr = arr.mean(axis=-1).astype(np.uint8)

        src_h, src_w = arr.shape[:2]
        n = self._cells_per_axis
        # The GT raster covers the full span; our grid is sized to the span,
        # so we resample the raster to the grid resolution. When ppm is
        # native these match (modulo a possible 1px rounding). Use nearest
        # neighbour to preserve crisp obstacle boundaries.
        sample_rows = np.linspace(0, src_h - 1, n).astype(int)
        sample_cols = np.linspace(0, src_w - 1, n).astype(int)
        grid = arr[np.ix_(sample_rows, sample_cols)]
        self._occupied[:, :] = grid < occupied_threshold

    def _gt_region_cells(self) -> tuple[int, int, int, int]:
        """Return the full cell window occupied by the GT raster.

        Older code tried to place the off-origin scene bounds into
        :class:`BevMap`'s origin-centred coordinate system, which clipped most
        GRScenes maps to a small corner and left the rest of the planner grid
        known-free.  A GT BEV map is already a complete raster over its own
        bounds, so it must occupy the full internal grid.
        """
        n = self._cells_per_axis
        return 0, 0, n, n

    # ------------------------------------------------------------------
    # Coordinate overrides (honour real scene bounds, not world origin)
    # ------------------------------------------------------------------

    def world_to_cell(self, xy: np.ndarray) -> np.ndarray:
        xy = np.asarray(xy, dtype=float)
        if xy.size == 0:
            return np.zeros((0, 2), dtype=int)
        xy = xy.reshape(-1, 2)
        # world (x, y) -> raster pixel (col, row) using native ppm.
        col = np.rint((xy[..., 0] - self._bounds_min[0]) * self._ppm_native).astype(int)
        row = np.rint(
            (self._bounds_min[1] + self._span - xy[..., 1]) * self._ppm_native
        ).astype(int)
        cells = np.stack([row, col], axis=-1)
        return self._clip_cells(cells)

    def cell_to_world(self, row: int, col: int) -> tuple[float, float]:
        rr = int(row)
        cc = int(col)
        x = (cc / self._ppm_native) + self._bounds_min[0]
        y = self._bounds_min[1] + self._span - (rr / self._ppm_native)
        return float(x), float(y)

    def cell_to_world_vec(self, rows: np.ndarray, cols: np.ndarray):
        rr = rows.astype(float)
        cc = cols.astype(float)
        x = (cc / self._ppm_native) + self._bounds_min[0]
        y = self._bounds_min[1] + self._span - (rr / self._ppm_native)
        return x, y

    def world_bounds_m(self) -> tuple[float, float, float, float]:
        return (
            float(self._bounds_min[0]),
            float(self._bounds_min[1]),
            float(self._bounds_min[0] + self._span),
            float(self._bounds_min[1] + self._span),
        )

    # ------------------------------------------------------------------
    # The map is static GT — never wipe it on reset.
    # ------------------------------------------------------------------

    def reset(self) -> None:  # noqa: D401 - intentional no-op
        """No-op: the GT occupancy layer must survive :meth:`EpisodicMemory.reset`."""
        return

    # Live ego-depth integration is disabled: the map is oracle-baked.
    def update(self, *args, **kwargs) -> None:  # noqa: D401
        """No-op: occupancy comes from the GT raster, not from ego points."""
        # Still record the pose so the planner and snapshot can use it.
        pose = kwargs.get("pose_xy")
        if pose is None and args:
            pose = args[1] if len(args) > 1 else None
        if pose is not None:
            try:
                self._last_pose_xy = np.asarray(pose, dtype=float).reshape(2)
            except (ValueError, TypeError):
                pass
        return

    def integrate_depth(self, *args, **kwargs) -> None:  # noqa: D401
        """No-op: GT map is static; only the last pose is tracked."""
        pose = kwargs.get("pose")
        if pose is None and args:
            pose = args[1] if len(args) > 1 else None
        if isinstance(pose, dict):
            position = pose.get("position")
            if position and len(position) >= 2:
                self._last_pose_xy = np.asarray(
                    [float(position[0]), float(position[1])], dtype=float
                )
        return


# ---------------------------------------------------------------------------
# Backend: locate the per-scene GT assets and build a GTBevMap
# ---------------------------------------------------------------------------


class GTBevBackend:
    """Locate and load the per-scene GT BEV assets into a static BEV map.

    Construction is cheap (one PNG read + one JSON read) and Isaac-free.
    The backend exposes :meth:`build_map`, called once by the agent factory
    to seed :class:`EpisodicMemory`. It does not perform per-step inference
    — there is nothing to infer, the map is static.

    Asset resolution order:
        1. Official ``occupancy.npy`` from ``TARSPHERE_GT_OCCUPANCY_PATH``.
        2. Official ``benchmark/meta/<scene_id>/occupancy.npy`` from TARSPHERE's
           configured TARSphere assets root.
        3. Legacy ``bev.png`` + ``bev_meta.json`` next to the scene USD, kept
           only for tests / scenes without official metadata.
    """

    def __init__(
        self,
        *,
        bev_dir: str | os.PathLike | None = None,
        scene_asset_path: str | None = None,
        agent_radius_m: float = G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    ) -> None:
        self._explicit_dir = Path(bev_dir) if bev_dir else None
        self._scene_asset_path = scene_asset_path
        self._agent_radius_m = float(agent_radius_m)
        self._map: Optional[BevMap] = None

    # ------------------------------------------------------------------
    # Asset discovery
    # ------------------------------------------------------------------

    def _resolve_dir(self) -> Path:
        if self._explicit_dir is not None:
            return self._explicit_dir
        env_dir = os.environ.get("TARSPHERE_GT_BEV_DIR", "").strip()
        if env_dir:
            return Path(env_dir)
        if self._scene_asset_path:
            return Path(self._scene_asset_path).resolve().parent
        raise RuntimeError(
            "GTBevBackend could not locate the BEV asset directory: pass "
            "bev_dir=, set TARSPHERE_GT_BEV_DIR, or construct with scene_asset_path."
        )

    def _load_meta(self, meta_path: Path) -> dict[str, Any]:
        with open(meta_path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build_map(self) -> BevMap:
        """Build (and cache) the static BEV map for this scene."""
        if self._map is not None:
            return self._map
        bev_dir = self._resolve_dir()
        official = self._resolve_official_occupancy(bev_dir)
        if official is not None:
            self._map = OfficialOccupancyBevMap(occupancy_path=official)
            return self._map

        png_path = bev_dir / "bev.png"
        meta_path = bev_dir / "bev_meta.json"
        if not png_path.is_file() or not meta_path.is_file():
            raise FileNotFoundError(
                f"GT BEV assets not found for scene directory {bev_dir}. Expected "
                "official benchmark/meta/<scene_id>/occupancy.npy, or legacy "
                "bev.png + bev_meta.json."
            )
        meta = self._load_meta(meta_path)
        bounds = meta.get("bev_bounds_min_xy")
        span = meta.get("bev_span_m")
        resolution = meta.get("resolution")
        if not bounds or not span or not resolution:
            raise ValueError(
                f"bev_meta.json at {meta_path} is missing bev_bounds_min_xy / "
                f"bev_span_m / resolution."
            )
        self._map = GTBevMap(
            bev_png_path=png_path,
            bev_bounds_min_xy=(float(bounds[0]), float(bounds[1])),
            bev_span_m=float(span),
            resolution=int(resolution),
            agent_radius_m=self._agent_radius_m,
        )
        return self._map

    def reset(self) -> None:
        """Drop the cached map so the next :meth:`build_map` re-reads disk."""
        self._map = None

    def _scene_id(self, bev_dir: Path) -> str | None:
        if self._scene_asset_path:
            parent = Path(self._scene_asset_path).expanduser().resolve().parent
            if parent.name.endswith("_usd"):
                return parent.name
        if bev_dir.name.endswith("_usd"):
            return bev_dir.name
        return None

    def _resolve_official_occupancy(self, bev_dir: Path) -> Path | None:
        explicit = os.environ.get("TARSPHERE_GT_OCCUPANCY_PATH", "").strip()
        if explicit:
            path = Path(explicit).expanduser().resolve()
            if not path.is_file():
                raise FileNotFoundError(
                    f"TARSPHERE_GT_OCCUPANCY_PATH does not exist: {path}"
                )
            return path

        meta_dir = os.environ.get("TARSPHERE_GT_META_DIR", "").strip()
        if meta_dir:
            path = Path(meta_dir).expanduser().resolve() / "occupancy.npy"
            if not path.is_file():
                raise FileNotFoundError(
                    f"TARSPHERE_GT_META_DIR has no occupancy.npy: {path}"
                )
            return path

        scene_id = self._scene_id(bev_dir)
        if not scene_id:
            return None
        try:
            root = tarsphere_assets_root(require=False)
        except Exception:
            return None
        path = root / "benchmark" / "meta" / scene_id / "occupancy.npy"
        return path if path.is_file() else None


__all__ = ["GTBevBackend", "GTBevMap", "OfficialOccupancyBevMap"]
