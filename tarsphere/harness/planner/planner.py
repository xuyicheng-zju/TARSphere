"""A* navigation planner over the BEV navigable mask.

The planner is a leaf module: it reads the live :class:`BevMap` and a planning
request (:class:`PlanInput`) and produces a collision-checked waypoint path in
world metres (:class:`PlanOutput`).  It never touches the simulator, the VLM,
or any hidden/evaluator state.

Path search is numpy A* with an admissible octile heuristic and finite
clearance-weighted edge lengths. One implementation serves every host and goal
type. Search is wall-clock guarded by ``time_limit_s``. Simplification preserves
both supercover collision checks and the length-integrated clearance objective.
"""

from __future__ import annotations

import heapq
import math
import time
from typing import Any, Optional, Sequence, TYPE_CHECKING

import numpy as np

from tarsphere.harness.planner.contracts import PlanInput, PlanOutput

if TYPE_CHECKING:  # pragma: no cover - typing only
    from tarsphere.harness.memory.bev_map import BevMap


# Segment re-validation epsilon (metres) for path simplification.
_SIMPLIFY_EPSILON_M = 0.3
# Diagonal step length for octile-distance A*.
_SQRT2 = math.sqrt(2.0)
# Finite route preference outside the hard embodiment mask. These global
# costs favour room for gait error without closing any formerly legal passage.
_PREFERRED_CLEARANCE_M = 0.15
_CLEARANCE_COST_WEIGHT = 4.0


class Planner:
    """A* planner over a :class:`BevMap`'s navigable mask.

    Args:
        bev: the live bird's-eye-view map to plan over.
        diagonal: allow 8-connected moves when ``True``, else 4-connected.
        time_limit_s: wall-clock budget for a single A* search.
        simplify: apply Ramer-Douglas-Peucker simplification (re-validated
            against collisions) to the raw cell path.
        clearance_radius_m: optional embodiment clearance applied on top of
            the map's navigation layer.  ``0`` preserves the map verbatim;
            positive values erode free space before search and simplification.
        world_exclusion_zones: optional evaluator-authored ``(x, y, radius)``
            disks, used for actor-aware Golden route certification.
        prepared_navigable_mask: optional embodiment-safe mask prepared by
            another Planner over the same BEV and clearance radius. This
            avoids repeating full-map erosion when only exclusion disks vary.
        backend: optional explicit selection of the sole ``'numpy'`` backend.
            ``None`` uses that same default; dependency installation never
            changes the search objective or implementation.
        cache_static_nav: reuse the prepared navigation mask between plans.
            Enable only when the underlying BEV and exclusion zones are
            immutable; dynamic policy maps keep the default ``False``.
    """

    name = "planner"

    def __init__(
        self,
        bev: "BevMap",
        *,
        diagonal: bool = True,
        time_limit_s: float = 5.0,
        simplify: bool = True,
        clearance_radius_m: float = 0.0,
        world_exclusion_zones: Optional[list[tuple[float, float, float]]] = None,
        prepared_navigable_mask: Optional[np.ndarray] = None,
        backend: Optional[str] = None,
        cache_static_nav: bool = False,
    ):
        if bev is None:
            raise ValueError("Planner requires a BevMap instance.")
        self._bev = bev
        self.diagonal = bool(diagonal)
        self.time_limit_s = float(time_limit_s)
        self.simplify = bool(simplify)
        self.clearance_radius_m = float(clearance_radius_m)
        if not math.isfinite(self.clearance_radius_m) or self.clearance_radius_m < 0:
            raise ValueError("clearance_radius_m must be a finite non-negative value.")
        self.world_exclusion_zones = tuple(
            (float(x), float(y), float(radius))
            for x, y, radius in (world_exclusion_zones or [])
        )
        if any(
            not all(math.isfinite(value) for value in zone) or zone[2] <= 0.0
            for zone in self.world_exclusion_zones
        ):
            raise ValueError(
                "world_exclusion_zones must contain finite (x, y, radius>0) tuples."
            )
        if backend is not None and backend != "numpy":
            raise ValueError(f"Unsupported backend {backend!r}; use 'numpy' or None.")
        self._prepared_navigable_mask = _validate_prepared_navigable_mask(
            prepared_navigable_mask,
            bev=bev,
        )
        self.cache_static_nav = bool(cache_static_nav)
        # Cached last plan metadata (cleared by reset).
        self._last_metadata: dict[str, Any] = {}
        self._active_nav: Optional[np.ndarray] = None
        self._active_cost: Optional[np.ndarray] = None
        self._embodiment_nav_cache: Optional[np.ndarray] = None
        self._static_nav_cache: Optional[np.ndarray] = None

    # ------------------------------------------------------------------
    # Public attributes / module protocol
    # ------------------------------------------------------------------

    @property
    def bev(self) -> "BevMap":
        """The live map (read-only access for the navigation controller)."""
        return self._bev

    def reset(self, context: Any = None) -> None:
        """Clear any cached plan/metadata."""
        self._last_metadata = {}
        self._active_nav = None
        self._active_cost = None

    def tools(self) -> dict:
        """No registered tools (consistency with the module protocol)."""
        return {}

    def is_line_collision_free(
        self,
        start_xy: Sequence[float],
        end_xy: Sequence[float],
    ) -> bool:
        """Check one world-space segment against the current planner mask."""

        return self._is_line_collision_free(
            [float(start_xy[0]), float(start_xy[1])],
            [float(end_xy[0]), float(end_xy[1])],
        )

    def shortcut_preserves_route_cost(self, points: Sequence[Sequence[float]]) -> bool:
        """Accept only collision-free shortcuts that preserve the route objective.

        Used by both path simplification and measured-start prefix extraction.
        A shortcut must not undo the search's preference for obstacle clearance.
        """
        if len(points) < 2 or not self.is_line_collision_free(points[0], points[-1]):
            return False
        original = sum(self._segment_cost(a, b) for a, b in zip(points, points[1:]))
        return (
            math.isfinite(original)
            and self._segment_cost(points[0], points[-1]) <= original + 1e-8
        )

    def shortcut_preserves_obstacle_cost(
        self, points: Sequence[Sequence[float]]
    ) -> bool:
        """Check a local execution chord against the prefix's worst soft cost."""
        if len(points) < 2 or not self.is_line_collision_free(points[0], points[-1]):
            return False
        nav = self.active_navigable_mask()
        cost = self._traversal_cost(nav)
        original_cells: set[tuple[int, int]] = set()
        for start, end in zip(points, points[1:]):
            if not self.is_line_collision_free(start, end):
                return False
            original_cells.update(
                _supercover_world_segment_cells(self._bev, start, end, nav.shape)
            )
        direct_cells = _supercover_world_segment_cells(
            self._bev, points[0], points[-1], nav.shape
        )
        return (
            max(cost[cell] for cell in direct_cells)
            <= max(cost[cell] for cell in original_cells) + 1e-12
        )

    def executable_prefix(
        self, points: Sequence[Sequence[float]], minimum_m: float
    ) -> np.ndarray:
        """Join sub-resolution corners without reducing worst-case clearance.

        Execution cannot resolve arbitrarily short A* corners. Select the first
        executable waypoint whose direct supercover is free and whose maximum
        soft obstacle cost is no greater than the original prefix's maximum.
        This preserves its minimum clearance, not its exact integrated cost.
        """
        if len(points) < 2 or not math.isfinite(minimum_m) or minimum_m <= 0:
            raise ValueError(
                "Executable prefix requires a path and positive finite minimum."
            )
        start = np.asarray(points[0], dtype=float)
        for index in range(1, len(points)):
            if not self.is_line_collision_free(points[index - 1], points[index]):
                raise ValueError("Executable prefix input crosses blocked geometry.")
            end = np.asarray(points[index], dtype=float)
            if np.linalg.norm(end - start) < minimum_m:
                continue
            if self.shortcut_preserves_obstacle_cost(points[: index + 1]):
                return end
            # Do not search farther along the route to justify a clearance loss
            # using an unrelated, less-clear portion of the path.
            break
        raise RuntimeError("No executable A* prefix preserves obstacle clearance.")

    def _traversal_cost(self, nav: np.ndarray) -> np.ndarray:
        from scipy.ndimage import distance_transform_edt

        if self._active_cost is None:
            # Treat the canvas exterior like any other blocked region. The
            # distance is additional clearance beyond the already inflated mask.
            distance = distance_transform_edt(np.pad(nav, 1))[1:-1, 1:-1]
            distance /= float(self._bev.pixels_per_meter)
            self._active_cost = np.where(
                nav,
                1.0
                + _CLEARANCE_COST_WEIGHT
                * np.maximum(0.0, 1.0 - distance / _PREFERRED_CLEARANCE_M) ** 2,
                0.0,
            )
        return self._active_cost

    def _segment_cost(self, start: Sequence[float], end: Sequence[float]) -> float:
        nav = self._active_nav
        if nav is None:
            nav = self.active_navigable_mask()
        cells = _supercover_world_segment_cells(self._bev, start, end, nav.shape)
        if not cells:
            return math.inf
        rows, cols = np.asarray(list(cells), dtype=int).T
        if (
            np.any(rows < 0)
            or np.any(cols < 0)
            or np.any(rows >= nav.shape[0])
            or np.any(cols >= nav.shape[1])
            or not np.all(nav[rows, cols])
        ):
            return math.inf
        cost = self._traversal_cost(nav)
        cuts, indices = world_segment_cell_intervals(self._bev, start, end, nav.shape)
        integral = np.dot(np.diff(cuts), cost[indices[:, 0], indices[:, 1]])
        return math.dist(start, end) * float(integral)

    # ------------------------------------------------------------------
    # Planning
    # ------------------------------------------------------------------

    def plan(self, data: PlanInput, context: Any = None) -> PlanOutput:
        """Turn a request into a collision-checked world-space waypoint path."""
        self._validate_input(data)

        nav = self._static_nav_cache
        if nav is None:
            nav = self.active_navigable_mask()
            if self.cache_static_nav:
                # Golden certification uses an immutable official occupancy
                # layer.  Retain its prepared embodiment-safe mask so each
                # short closed-loop segment does not allocate another large
                # SciPy erosion workspace. Dynamic BEV planners keep the
                # default False and rebuild from their latest map every call.
                self._static_nav_cache = nav
        self._active_nav = nav
        self._active_cost = None
        if not nav.any():
            return PlanOutput(
                status="no_map",
                metadata={
                    "backend": None,
                    "status": "no_map",
                    "clearance_radius_m": self.clearance_radius_m,
                },
            )

        # Resolve the start cell, snapping to the nearest free cell if needed.
        start_cell = self._resolve_start(data.start_xy, nav)
        if start_cell is None:
            return PlanOutput(
                status="no_map", metadata={"backend": None, "status": "no_map"}
            )
        resolved_start_xy = list(
            self._bev.cell_to_world(int(start_cell[0]), int(start_cell[1]))
        )
        start_snapped_m = math.dist(
            [float(data.start_xy[0]), float(data.start_xy[1])],
            resolved_start_xy,
        )

        candidates = data.goal_candidates_xy
        goals: dict[tuple[int, int], list[float]] = {}
        if candidates is not None:
            for point in candidates:
                r, c = (
                    int(v)
                    for v in self._bev.world_to_cell(np.asarray(point).reshape(1, 2))[0]
                )
                if 0 <= r < nav.shape[0] and 0 <= c < nav.shape[1] and nav[r, c]:
                    goals.setdefault((r, c), list(point))
            goal_cell = next(iter(goals), None)
            goal_xy = goals.get(goal_cell)
        else:
            goal_cell, goal_xy = self._resolve_goal(data.goal_xy, nav)
        if goal_cell is None:
            return PlanOutput(
                status="no_map",
                goal_xy=None,
                metadata={"backend": None, "status": "no_map"},
            )

        # Start and goal coincide -> nothing to plan.
        already_there = (
            start_cell in goals
            if candidates is not None
            else self._cells_close(start_cell, goal_cell)
        )
        if already_there:
            return PlanOutput(
                waypoints=[],
                goal_xy=goals[start_cell] if candidates is not None else goal_xy,
                status="already_there",
                metadata={"backend": None, "status": "already_there"},
            )

        if candidates is not None:
            cell_path = self._astar_numpy(nav, start_cell, frozenset(goals))
            backend = "numpy"
            if cell_path:
                goal_xy = goals[cell_path[-1]]
        else:
            cell_path, backend = self._astar(nav, start_cell, goal_cell)
        if cell_path is None:
            return PlanOutput(
                waypoints=[],
                goal_xy=None if candidates is not None else goal_xy,
                status="unreachable",
                metadata={
                    "backend": backend,
                    "status": "unreachable",
                    "num_waypoints": 0,
                    "path_length_m": 0.0,
                },
            )

        # Convert cells (row, col) -> world [x, y], dropping the start cell.
        world_pts: list[list[float]] = [
            list(self._bev.cell_to_world(int(r), int(c))) for r, c in cell_path
        ]
        if candidates is not None:
            # Preserve the allowed endpoint, including an exact sub-cell goal.
            world_pts[-1] = list(goal_xy)
        # Include the measured start in cost-aware simplification. Otherwise
        # its sub-cell offset can create a spurious centimetre-sized first
        # corner when the runtime checks a shortcut from the actual pose.
        measured_cell = tuple(
            int(v)
            for v in self._bev.world_to_cell(np.asarray(data.start_xy).reshape(1, 2))[0]
        )
        if (
            len(world_pts) >= 2
            and measured_cell == start_cell
            and self._is_line_collision_free(list(data.start_xy), world_pts[1])
        ):
            world_pts[0] = list(data.start_xy)
            resolved_start_xy = list(data.start_xy)
            start_snapped_m = 0.0

        if self.simplify and len(world_pts) >= 3:
            world_pts = self._simplify(world_pts)

        path_length_m = self._path_length(world_pts)
        if world_pts and math.dist(world_pts[0], data.start_xy) <= 1e-9:
            world_pts = world_pts[1:]
        metadata = {
            "backend": backend,
            "status": "ok",
            "num_waypoints": len(world_pts),
            "path_length_m": path_length_m,
            "route_objective": "length_with_soft_clearance_cost",
            "requested_start_xy": [float(data.start_xy[0]), float(data.start_xy[1])],
            "resolved_start_xy": [
                float(resolved_start_xy[0]),
                float(resolved_start_xy[1]),
            ],
            "start_snapped_m": float(start_snapped_m),
            "clearance_radius_m": self.clearance_radius_m,
            "world_exclusion_zones": [
                list(zone) for zone in self.world_exclusion_zones
            ],
        }
        self._last_metadata = dict(metadata)
        return PlanOutput(
            waypoints=world_pts,
            goal_xy=goal_xy,
            status="ok",
            metadata=metadata,
        )

    def embodiment_navigable_mask(self) -> np.ndarray:
        """Return the clearance-safe mask before actor exclusion disks."""
        if self._prepared_navigable_mask is not None:
            return self._prepared_navigable_mask
        if self._embodiment_nav_cache is not None:
            return self._embodiment_nav_cache

        nav = self._bev.navigable_mask()
        if self.clearance_radius_m > 0.0:
            pixels_per_meter = float(getattr(self._bev, "pixels_per_meter", 0.0))
            if not math.isfinite(pixels_per_meter) or pixels_per_meter <= 0.0:
                raise ValueError(
                    "A positive clearance_radius_m requires bev.pixels_per_meter."
                )
            radius_cells = int(math.ceil(self.clearance_radius_m * pixels_per_meter))
            nav = erode_navigable_mask(nav, radius_cells)
        if self.cache_static_nav:
            self._embodiment_nav_cache = nav
        return nav

    def active_navigable_mask(self) -> np.ndarray:
        """Return the embodiment-safe mask with actor exclusions applied."""
        nav = self.embodiment_navigable_mask()
        if self.world_exclusion_zones:
            nav = self._apply_world_exclusion_zones(nav)
        return nav

    def _apply_world_exclusion_zones(self, nav: np.ndarray) -> np.ndarray:
        """Remove evaluator-authored actor safety disks from a navigation mask."""
        pixels_per_meter = float(getattr(self._bev, "pixels_per_meter", 0.0))
        if not math.isfinite(pixels_per_meter) or pixels_per_meter <= 0.0:
            raise ValueError("world_exclusion_zones require bev.pixels_per_meter.")
        blocked = np.array(nav, dtype=bool, copy=True)
        for x, y, radius_m in self.world_exclusion_zones:
            center = self._bev.world_to_cell(np.asarray([[x, y]], dtype=float))[0]
            center_row, center_col = int(center[0]), int(center[1])
            radius_cells = int(math.ceil(radius_m * pixels_per_meter))
            row_min = max(0, center_row - radius_cells)
            row_max = min(blocked.shape[0], center_row + radius_cells + 1)
            col_min = max(0, center_col - radius_cells)
            col_max = min(blocked.shape[1], center_col + radius_cells + 1)
            rows, cols = np.ogrid[row_min:row_max, col_min:col_max]
            disk = (rows - center_row) ** 2 + (
                cols - center_col
            ) ** 2 <= radius_cells**2
            blocked[row_min:row_max, col_min:col_max][disk] = False
        return blocked

    # ------------------------------------------------------------------
    # Input validation
    # ------------------------------------------------------------------

    @staticmethod
    def _validate_input(data: PlanInput) -> None:
        if not isinstance(data, PlanInput):
            raise ValueError(f"Planner.plan expects PlanInput, got {type(data)!r}.")
        Planner._require_finite_xy(data.start_xy, "start_xy")
        if data.goal_xy is not None:
            Planner._require_finite_xy(data.goal_xy, "goal_xy")
        if (data.goal_xy is None) == (data.goal_candidates_xy is None):
            raise ValueError(
                "PlanInput needs exactly one goal_xy or goal_candidates_xy."
            )
        if data.goal_candidates_xy is not None:
            for point in data.goal_candidates_xy:
                Planner._require_finite_xy(point, "goal_candidates_xy")

    @staticmethod
    def _require_finite_xy(xy: Any, name: str) -> None:
        arr = np.asarray(xy, dtype=float).reshape(-1)
        if arr.shape[0] < 2 or not np.all(np.isfinite(arr[:2])):
            raise ValueError(f"PlanInput.{name} must be finite (x, y), got {xy!r}.")

    # ------------------------------------------------------------------
    # Start / goal resolution
    # ------------------------------------------------------------------

    def _resolve_start(
        self, start_xy: tuple[float, float], nav: np.ndarray
    ) -> Optional[tuple[int, int]]:
        cell = self._bev.world_to_cell(np.asarray(start_xy, dtype=float).reshape(1, 2))[
            0
        ]
        r, c = int(cell[0]), int(cell[1])
        if 0 <= r < nav.shape[0] and 0 <= c < nav.shape[1] and nav[r, c]:
            return r, c
        snap = self._nearest_free_cell(start_xy, nav)
        return None if snap is None else (int(snap[0]), int(snap[1]))

    def _resolve_goal(
        self, goal_xy: Optional[tuple[float, float]], nav: np.ndarray
    ) -> tuple[Optional[tuple[int, int]], Optional[list[float]]]:
        if goal_xy is None:
            return None, None
        cell = self._bev.world_to_cell(np.asarray(goal_xy, dtype=float).reshape(1, 2))[
            0
        ]
        r, c = int(cell[0]), int(cell[1])
        if 0 <= r < nav.shape[0] and 0 <= c < nav.shape[1] and nav[r, c]:
            return (r, c), [float(goal_xy[0]), float(goal_xy[1])]
        snap = self._nearest_free_cell(goal_xy, nav)
        if snap is None:
            return None, None
        world = list(self._bev.cell_to_world(int(snap[0]), int(snap[1])))
        return (int(snap[0]), int(snap[1])), world

    # ------------------------------------------------------------------
    # Map helper passthrough (graceful if sibling helpers are absent)
    # ------------------------------------------------------------------

    def _nearest_free_cell(
        self, xy: tuple[float, float], nav: np.ndarray
    ) -> Optional[tuple[int, int]]:
        """Nearest navigable CELL (row, col) to ``xy`` via a numpy BFS.

        Note: :meth:`BevMap.nearest_free_cell` returns WORLD ``(x, y)`` and uses
        its own navigable mask, whereas the planner works in cell space over the
        passed mask — so we use the cell-space BFS directly here.
        """
        return self._nearest_free_cell_numpy(xy, nav)

    def _nearest_free_cell_numpy(
        self, xy: tuple[float, float], nav: np.ndarray
    ) -> Optional[tuple[int, int]]:
        target = self._bev.world_to_cell(np.asarray(xy, dtype=float).reshape(1, 2))[0]
        tr, tc = int(target[0]), int(target[1])
        rows, cols = nav.shape
        if not nav.any():
            return None
        # BFS outward from the target cell until a free cell is found.
        seen = np.zeros_like(nav, dtype=bool)
        queue: list[tuple[int, int]] = []
        if 0 <= tr < rows and 0 <= tc < cols:
            queue.append((tr, tc))
            seen[tr, tc] = True
        head = 0
        found: Optional[tuple[int, int]] = None
        best_d2 = math.inf
        # BFS stays in-grid; ``found`` remains None when no navigable cell exists.
        while head < len(queue):
            r, c = queue[head]
            head += 1
            if nav[r, c]:
                d2 = (r - tr) ** 2 + (c - tc) ** 2
                if d2 < best_d2:
                    best_d2 = d2
                    found = (r, c)
                    break  # BFS guarantees nearest on first hit
            for dr in (-1, 0, 1):
                for dc in (-1, 0, 1):
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols and not seen[nr, nc]:
                        seen[nr, nc] = True
                        queue.append((nr, nc))
        if found is not None:
            return found
        # Exhausted reachable region without a free cell: brute-force nearest.
        free = np.argwhere(nav)
        d2 = (free[:, 0] - tr) ** 2 + (free[:, 1] - tc) ** 2
        idx = int(np.argmin(d2))
        return int(free[idx, 0]), int(free[idx, 1])

    def _is_line_collision_free(self, p0: list[float], p1: list[float]) -> bool:
        """Validate a shortcut against the exact mask used by this plan.

        Supercover traversal is required even when ``clearance_radius_m`` is
        zero: the raw occupancy can contain one-cell walls that fixed-rate
        line sampling misses. Before a plan, use the same prepared embodiment
        mask and actor exclusions that planning would install.
        """
        nav = self._active_nav
        if nav is None:
            nav = self.active_navigable_mask()
        cells = _supercover_world_segment_cells(self._bev, p0[:2], p1[:2], nav.shape)
        if cells is None:
            return False
        # Every touched cell counts, including both sides at a grid corner.
        for row, col in cells:
            row, col = int(row), int(col)
            if not (0 <= row < nav.shape[0] and 0 <= col < nav.shape[1]):
                return False
            if not bool(nav[row, col]):
                return False
        return True

    # ------------------------------------------------------------------
    # Shared clearance-weighted A* search
    # ------------------------------------------------------------------

    def _astar(
        self,
        nav: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> tuple[Optional[list[tuple[int, int]]], Optional[str]]:
        """Run A*; return (cell_path incl. start and goal, backend_name)."""
        return self._astar_numpy(nav, start, goal), "numpy"

    def _astar_numpy(
        self,
        nav: np.ndarray,
        start: tuple[int, int],
        goal: tuple[int, int] | frozenset[tuple[int, int]],
    ) -> Optional[list[tuple[int, int]]]:
        """8/4-connected A* integrating costs along centre-to-centre edges."""
        rows, cols = nav.shape
        cost = self._traversal_cost(nav)
        sr, sc = start
        goals = goal if isinstance(goal, frozenset) else frozenset((goal,))
        if not (0 <= sr < rows and 0 <= sc < cols and nav[sr, sc]):
            return None
        if not goals or any(
            not (0 <= r < rows and 0 <= c < cols and nav[r, c]) for r, c in goals
        ):
            return None
        min_r = min(r for r, c in goals)
        max_r = max(r for r, c in goals)
        min_c = min(c for r, c in goals)
        max_c = max(c for r, c in goals)

        if self.diagonal:
            moves = (
                (-1, -1),
                (-1, 0),
                (-1, 1),
                (0, -1),
                (0, 1),
                (1, -1),
                (1, 0),
                (1, 1),
            )
        else:
            moves = ((-1, 0), (1, 0), (0, -1), (0, 1))

        def heuristic(r: int, c: int) -> float:
            # Distance to the goal-set bounding box is an admissible,
            # consistent lower bound; membership, not the box, ends search.
            dr = max(min_r - r, 0, r - max_r)
            dc = max(min_c - c, 0, c - max_c)
            if self.diagonal:
                # Octile distance.
                return (_SQRT2 - 1.0) * min(dr, dc) + max(dr, dc)
            return float(dr + dc)

        start_h = heuristic(sr, sc)
        # heap entries: (f, counter, g, r, c)
        counter = 0
        open_heap: list[tuple[float, int, float, int, int]] = [
            (start_h, counter, 0.0, sr, sc)
        ]
        came_from: dict[tuple[int, int], tuple[int, int]] = {}
        best_g: dict[tuple[int, int], float] = {(sr, sc): 0.0}
        deadline = time.monotonic() + self.time_limit_s

        while open_heap:
            if time.monotonic() > deadline:
                return None
            f, _, g, r, c = heapq.heappop(open_heap)
            if (r, c) in goals:
                return self._reconstruct(came_from, (sr, sc), (r, c))
            if g > best_g.get((r, c), math.inf):
                continue
            for dr, dc in moves:
                nr, nc = r + dr, c + dc
                if not (0 <= nr < rows and 0 <= nc < cols) or not nav[nr, nc]:
                    continue
                # Prevent corner-cutting on diagonals.
                if dr != 0 and dc != 0:
                    if not nav[r + dr, c] or not nav[r, c + dc]:
                        continue
                step = _SQRT2 if (dr != 0 and dc != 0) else 1.0
                ng = g + step * float((cost[r, c] + cost[nr, nc]) * 0.5)
                if ng < best_g.get((nr, nc), math.inf):
                    best_g[(nr, nc)] = ng
                    came_from[(nr, nc)] = (r, c)
                    counter += 1
                    heapq.heappush(
                        open_heap, (ng + heuristic(nr, nc), counter, ng, nr, nc)
                    )
        return None

    @staticmethod
    def _reconstruct(
        came_from: dict[tuple[int, int], tuple[int, int]],
        start: tuple[int, int],
        goal: tuple[int, int],
    ) -> list[tuple[int, int]]:
        path = [goal]
        node = goal
        while node != start:
            node = came_from.get(node)
            if node is None:
                break
            path.append(node)
        path.reverse()
        return path

    # ------------------------------------------------------------------
    # Path post-processing
    # ------------------------------------------------------------------

    def _simplify(self, pts: list[list[float]]) -> list[list[float]]:
        """Ramer-Douglas-Peucker with per-segment collision re-validation.

        Any kept segment that fails ``is_line_collision_free`` re-inserts the
        dropped intermediate waypoint that bisects the offending run, so a
        shortcut can never cut through an obstacle.
        """
        simplified = _ramer_douglas_peucker(pts, _SIMPLIFY_EPSILON_M)
        return self._revalidate(simplified, pts)

    def _revalidate(
        self, simplified: list[list[float]], original: list[list[float]]
    ) -> list[list[float]]:
        if len(simplified) < 2:
            return simplified
        prefix_cost = np.cumsum(
            [0.0, *(self._segment_cost(a, b) for a, b in zip(original, original[1:]))]
        )

        def preserves_cost(first: int, second: int) -> bool:
            return (
                self._is_line_collision_free(original[first], original[second])
                and self._segment_cost(original[first], original[second])
                <= prefix_cost[second] - prefix_cost[first] + 1e-8
            )

        offsets = [original.index(point) for point in simplified]
        if all(
            preserves_cost(first, second) for first, second in zip(offsets, offsets[1:])
        ):
            return simplified

        # RDP can cut across a concave clearance boundary.  Rather than
        # restoring every grid-scale A* turn (which makes the physics
        # controller oscillate), perform conservative line-of-sight string
        # pulling over the original path.  Each step keeps the farthest
        # reachable, non-cost-increasing original vertex. Every emitted segment
        # retains the same hard mask and soft objective used for this plan.
        result: list[list[float]] = [original[0]]
        current = 0
        final = len(original) - 1
        while current < final:
            candidate = final
            while candidate > current + 1:
                if preserves_cost(current, candidate):
                    break
                candidate -= 1
            result.append(original[candidate])
            current = candidate
        return result

    # ------------------------------------------------------------------
    # Small geometry helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cells_close(a: tuple[int, int], b: tuple[int, int]) -> bool:
        return abs(a[0] - b[0]) <= 1 and abs(a[1] - b[1]) <= 1

    @staticmethod
    def _path_length(pts: list[list[float]]) -> float:
        if len(pts) < 2:
            return 0.0
        total = 0.0
        for i in range(1, len(pts)):
            dx = pts[i][0] - pts[i - 1][0]
            dy = pts[i][1] - pts[i - 1][1]
            total += math.hypot(dx, dy)
        return float(total)


# ----------------------------------------------------------------------
# Pure-numpy Ramer-Douglas-Peucker (no scipy/cv2 dependency)
# ----------------------------------------------------------------------


def _validate_prepared_navigable_mask(
    mask: Optional[np.ndarray],
    *,
    bev: "BevMap",
) -> Optional[np.ndarray]:
    if mask is None:
        return None
    prepared = np.asarray(mask, dtype=bool)
    if prepared.ndim != 2 or not all(size > 0 for size in prepared.shape):
        raise ValueError("prepared_navigable_mask must be a non-empty 2D boolean mask")
    expected_shape = np.asarray(
        bev.navigable_mask(),
        dtype=bool,
    ).shape
    if prepared.shape != expected_shape:
        raise ValueError(
            "prepared_navigable_mask shape must match bev.navigable_mask(): "
            f"{prepared.shape} != {expected_shape}"
        )
    read_only = prepared.view()
    read_only.flags.writeable = False
    return read_only


def erode_navigable_mask(nav: np.ndarray, radius_cells: int) -> np.ndarray:
    """Remove free cells within ``radius_cells`` of blocked space.

    Use one dependency-independent Euclidean disk on every host.  The former
    SciPy path used a disk while its minimal-environment fallback repeatedly
    applied a 3x3 square.  That silently changed the G1-clear navigable mask,
    route, and certification hash according to whether SciPy was installed.
    Out-of-map space is always treated as blocked.
    """
    free = np.asarray(nav, dtype=bool)
    radius = int(radius_cells)
    if radius <= 0:
        return free.copy()
    yy, xx = np.ogrid[-radius : radius + 1, -radius : radius + 1]
    footprint = (xx * xx + yy * yy) <= radius * radius
    rows, columns = free.shape
    padded = np.pad(
        free,
        radius,
        mode="constant",
        constant_values=False,
    )
    eroded = np.ones_like(free, dtype=bool)
    for offset_row, offset_column in np.argwhere(footprint) - radius:
        row_start = radius + int(offset_row)
        column_start = radius + int(offset_column)
        eroded &= padded[
            row_start : row_start + rows,
            column_start : column_start + columns,
        ]
    return eroded


def _continuous_world_cell(bev, point, shape) -> Optional[tuple[float, float]]:
    cell = bev.world_to_cell(np.asarray([point], dtype=float))[0]
    row, col = int(cell[0]), int(cell[1])
    rows, cols = int(shape[0]), int(shape[1])
    row_neighbor = row + 1 if row + 1 < rows else row - 1
    col_neighbor = col + 1 if col + 1 < cols else col - 1
    if row_neighbor < 0 or col_neighbor < 0:
        return None
    center = np.asarray(bev.cell_to_world(row, col), dtype=float)
    row_basis = (
        np.asarray(bev.cell_to_world(row_neighbor, col), dtype=float) - center
    ) / float(row_neighbor - row)
    col_basis = (
        np.asarray(bev.cell_to_world(row, col_neighbor), dtype=float) - center
    ) / float(col_neighbor - col)
    basis = np.column_stack([row_basis, col_basis])
    try:
        offset = np.linalg.solve(basis, np.asarray(point, dtype=float) - center)
    except np.linalg.LinAlgError:
        return None
    result = (float(row + offset[0]), float(col + offset[1]))
    if not all(math.isfinite(value) for value in result):
        return None
    return result


def world_segment_cell_intervals(
    bev, start, end, shape
) -> tuple[np.ndarray, np.ndarray]:
    """Partition a world segment at exact cell boundaries.

    Returns increasing fractions from 0 to 1 and one (row, column) per open
    interval. This measures lengths; use supercover separately for collisions
    at zero-length corner touches. Invalid lattice transforms fail explicitly.
    """
    first = _continuous_world_cell(bev, start, shape)
    last = _continuous_world_cell(bev, end, shape)
    if first is None or last is None:
        raise ValueError("Cannot resolve segment in the map coordinate system")
    first, last = np.asarray(first), np.asarray(last)
    delta = last - first
    cuts = [0.0, 1.0]
    for origin, destination, change in zip(first, last, delta):
        if abs(change) <= 1e-12:
            continue
        lo, hi = sorted((origin, destination))
        boundaries = np.arange(math.floor(lo - 0.5) + 1, math.ceil(hi - 0.5)) + 0.5
        cuts.extend(float(t) for t in (boundaries - origin) / change if 0.0 < t < 1.0)
    cuts = np.unique(cuts)
    midpoints = (cuts[:-1] + cuts[1:]) * 0.5
    indices = np.floor(first + midpoints[:, None] * delta + 0.5).astype(int)
    return cuts, indices


def _supercover_world_segment_cells(
    bev: "BevMap",
    p0: list[float],
    p1: list[float],
    shape: tuple[int, int],
) -> Optional[set[tuple[int, int]]]:
    """Return every grid cell intersected by a continuous world segment.

    ``BevMap.world_to_cell`` deliberately returns nearest integer cells.  A
    supercover ray needs the corresponding fractional grid coordinates, so
    they are reconstructed from the local row/column world-space basis.  All
    benchmark BEV backends are regular axis-aligned grids, while this local
    construction also preserves their distinct origins and axis directions.
    """

    start = _continuous_world_cell(bev, p0, shape)
    end = _continuous_world_cell(bev, p1, shape)
    if start is None or end is None:
        return None
    row0, col0 = start
    row1, col1 = end
    row = int(math.floor(row0 + 0.5))
    col = int(math.floor(col0 + 0.5))
    end_row = int(math.floor(row1 + 0.5))
    end_col = int(math.floor(col1 + 0.5))
    cells = {(row, col)}

    delta_row = row1 - row0
    delta_col = col1 - col0
    step_row = 1 if delta_row > 0 else (-1 if delta_row < 0 else 0)
    step_col = 1 if delta_col > 0 else (-1 if delta_col < 0 else 0)
    t_delta_row = 1.0 / abs(delta_row) if step_row else math.inf
    t_delta_col = 1.0 / abs(delta_col) if step_col else math.inf
    next_row_boundary = row + 0.5 if step_row > 0 else row - 0.5
    next_col_boundary = col + 0.5 if step_col > 0 else col - 0.5
    t_max_row = (next_row_boundary - row0) / delta_row if step_row else math.inf
    t_max_col = (next_col_boundary - col0) / delta_col if step_col else math.inf

    while (row, col) != (end_row, end_col):
        if t_max_row < t_max_col - 1e-12:
            row += step_row
            t_max_row += t_delta_row
            cells.add((row, col))
        elif t_max_col < t_max_row - 1e-12:
            col += step_col
            t_max_col += t_delta_col
            cells.add((row, col))
        else:
            # At an exact corner crossing, all four touching cells belong to
            # the conservative supercover, not only the diagonal successor.
            next_row = row + step_row
            next_col = col + step_col
            cells.add((next_row, col))
            cells.add((row, next_col))
            row, col = next_row, next_col
            t_max_row += t_delta_row
            t_max_col += t_delta_col
            cells.add((row, col))
    return cells


def _ramer_douglas_peucker(pts: list[list[float]], epsilon: float) -> list[list[float]]:
    """Iterative RDP simplification of a 2D polyline."""
    if len(pts) < 3:
        return [list(p) for p in pts]
    keep = [False] * len(pts)
    keep[0] = keep[-1] = True
    stack = [(0, len(pts) - 1)]
    while stack:
        i0, i1 = stack.pop()
        if i1 - i0 < 2:
            continue
        dmax = -1.0
        idx = i0
        ax, ay = pts[i0]
        bx, by = pts[i1]
        seg_len = math.hypot(bx - ax, by - ay)
        for j in range(i0 + 1, i1):
            if seg_len < 1e-12:
                d = math.hypot(pts[j][0] - ax, pts[j][1] - ay)
            else:
                # Perpendicular distance from point to segment.
                t = ((pts[j][0] - ax) * (bx - ax) + (pts[j][1] - ay) * (by - ay)) / (
                    seg_len * seg_len
                )
                t = max(0.0, min(1.0, t))
                px = ax + t * (bx - ax)
                py = ay + t * (by - ay)
                d = math.hypot(pts[j][0] - px, pts[j][1] - py)
            if d > dmax:
                dmax = d
                idx = j
        if dmax > epsilon:
            keep[idx] = True
            stack.append((i0, idx))
            stack.append((idx, i1))
    return [list(pts[i]) for i in range(len(pts)) if keep[i]]
