"""Typed access to live PhysX scene geometry."""

from __future__ import annotations

import math
from typing import Callable, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator


Point3 = Tuple[float, float, float]


class StrictModel(BaseModel):
    """Immutable current-release scene-query schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class SurfaceRayHit(StrictModel):
    """One live scene-query intersection used by physical certification."""

    position_xyz: Point3 = Field(
        description="World-space PhysX ray intersection in metres.",
    )
    normal_xyz: Point3 = Field(
        description="World-space unit surface normal returned by PhysX.",
    )
    prim_path: str = Field(
        min_length=1,
        description="USD collision or rigid-body prim hit by the ray.",
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "SurfaceRayHit":
        values = (*self.position_xyz, *self.normal_xyz)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("surface ray hit geometry must be finite")
        normal_length = math.sqrt(sum(value * value for value in self.normal_xyz))
        if not math.isclose(normal_length, 1.0, abs_tol=1e-3):
            raise ValueError("surface ray hit normal must be unit length")
        return self


SurfaceRaycast = Callable[
    [Point3, Point3, float],
    Optional[SurfaceRayHit],
]
SurfaceRaycastAll = Callable[
    [Point3, Point3, float],
    Tuple[SurfaceRayHit, ...],
]


class PhysXSupportClearance:
    """Conservative metric separation from a target AABB to actual colliders.

    Uniform AABB expansion measures an L-infinity gap, a lower bound on
    Euclidean separation. Query the cooked PhysX shapes, preserving shelves,
    holes and collision approximations; an owner's filled bounding box is not
    a collision shape. Bisection returns the collision-free lower endpoint
    within 10 micrometres, never a rounded-up success at the 5 mm hold gate.
    """

    def __init__(self, query: object | None = None) -> None:
        if query is None:
            import omni.physx

            query = omni.physx.get_physx_scene_query_interface()
        if not callable(getattr(query, "overlap_box", None)):
            raise TypeError("PhysX support clearance requires overlap_box")
        self._query = query

    def __call__(
        self,
        *,
        target_bounds_xyz,
        support_bounds_xyz,
        solid_meshes,
        support_prim_path: str,
    ) -> float:
        from tarsphere.infra.simulation.usd_geometry import _validate_bounds
        from tarsphere.infra.simulation.support_surface import inside_support_solid

        lo, hi = target_bounds_xyz
        source_lo, source_hi = support_bounds_xyz
        _validate_bounds(lo, hi)
        _validate_bounds(source_lo, source_hi)
        if any(hi[i] <= lo[i] for i in range(3)):
            raise ValueError("Support clearance requires positive target dimensions")
        if not support_prim_path.startswith("/") or support_prim_path == "/":
            raise ValueError("Support clearance requires an exact non-root prim path")
        center = tuple((lo[i] + hi[i]) * 0.5 for i in range(3))
        half = tuple((hi[i] - lo[i]) * 0.5 for i in range(3))
        owner = support_prim_path.rstrip("/")

        def overlaps(expansion: float) -> bool:
            found = False

            def collect(hit: object) -> bool:
                nonlocal found
                paths = [
                    str(_raw_field(hit, key) or "")
                    for key in ("collision", "rigidBody", "rigid_body")
                ]
                if not any(paths):
                    raise RuntimeError("PhysX overlap hit omitted collider/body paths")
                if any(p == owner or p.startswith(owner + "/") for p in paths):
                    found = True
                    return False
                return True

            self._query.overlap_box(
                tuple(v + expansion for v in half),
                center,
                (0.0, 0.0, 0.0, 1.0),
                collect,
                False,
            )
            return found

        # This box strictly encloses the measured support. Absence of a hit
        # means stale/missing scene-query geometry, not infinite clearance.
        upper = max(
            max(abs(source_lo[i] - center[i]), abs(source_hi[i] - center[i])) - half[i]
            for i in range(3)
        )
        upper = max(0.0, upper) + 0.001
        if not overlaps(upper):
            raise RuntimeError(
                f"PhysX support collider absent from live query: {owner}"
            )
        if overlaps(0.0):
            return 0.0
        # A triangle mesh's native overlap can miss a box wholly inside it.
        # With no surface intersection, the connected target box is entirely
        # inside or outside; its centre resolves containment without sampling.
        if inside_support_solid(center, solid_meshes):
            return 0.0
        lower = 0.0
        while upper - lower > 1e-5:
            middle = (lower + upper) * 0.5
            if overlaps(middle):
                upper = middle
            else:
                lower = middle
        return lower


class PhysXSurfaceRaycast:
    """Expose live PhysX raycasts in the benchmark's metre-based world."""

    def __init__(self, query: object | None = None) -> None:
        if query is None:
            try:
                import omni.physx
            except Exception as exc:
                raise RuntimeError(
                    "Isaac PhysX is required for live surface raycasts"
                ) from exc
            query = omni.physx.get_physx_scene_query_interface()
        if query is None or not callable(getattr(query, "raycast_closest", None)):
            raise TypeError("PhysX surface raycast requires raycast_closest")
        self._query = query

    def __call__(
        self,
        origin_xyz: Point3,
        direction_xyz: Point3,
        distance_m: float,
    ) -> SurfaceRayHit | None:
        raw = self._query.raycast_closest(
            origin_xyz,
            direction_xyz,
            float(distance_m),
        )
        if not isinstance(raw, dict):
            raise RuntimeError("PhysX raycast returned a non-mapping result")
        if not bool(raw.get("hit", False)):
            return None
        return self._surface_hit(raw)

    def all_hits(
        self,
        origin_xyz: Point3,
        direction_xyz: Point3,
        distance_m: float,
    ) -> Tuple[SurfaceRayHit, ...]:
        """Return every scene hit, sorted from the metric ray origin."""
        raycast_all = getattr(self._query, "raycast_all", None)
        if not callable(raycast_all):
            raise TypeError("PhysX surface raycast requires raycast_all")
        hits = []

        def collect(raw: object) -> bool:
            hits.append(self._surface_hit(raw))
            return True

        raycast_all(
            origin_xyz,
            direction_xyz,
            float(distance_m),
            collect,
        )
        return tuple(
            sorted(
                hits,
                key=lambda hit: math.dist(origin_xyz, hit.position_xyz),
            )
        )

    def _surface_hit(self, raw: object) -> SurfaceRayHit:
        position = _point3(_raw_field(raw, "position"))
        normal = _point3(_raw_field(raw, "normal"))
        if position is None or normal is None:
            raise RuntimeError("PhysX raycast hit omitted position or normal")
        paths = tuple(
            str(_raw_field(raw, field) or "").strip()
            for field in ("collision", "rigidBody", "rigid_body")
            if str(_raw_field(raw, field) or "").strip()
        )
        if not paths:
            raise RuntimeError(
                "PhysX raycast hit omitted collision and rigid-body paths"
            )
        return SurfaceRayHit(
            position_xyz=position,
            normal_xyz=normal,
            prim_path=paths[0],
        )


def _raw_field(value: object, name: str) -> object:
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def _point3(value: object) -> Point3 | None:
    """Read tuple/list, Gf vector, or Carb vector values without guessing."""
    try:
        point = tuple(float(value[index]) for index in range(3))  # type: ignore[index]
    except (IndexError, KeyError, TypeError, ValueError):
        point = ()
    if len(point) == 3:
        return point  # type: ignore[return-value]

    components = tuple(getattr(value, name, None) for name in ("x", "y", "z"))
    if any(component is None for component in components):
        return None
    try:
        return tuple(float(component) for component in components)  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None


def belongs_to_prim_tree(prim_path: str, root_prim_path: str) -> bool:
    """Return whether ``prim_path`` is ``root_prim_path`` or its descendant."""
    normalized_root = root_prim_path.rstrip("/")
    return prim_path == normalized_root or prim_path.startswith(normalized_root + "/")


__all__ = [
    "PhysXSupportClearance",
    "PhysXSurfaceRaycast",
    "SurfaceRayHit",
    "SurfaceRaycast",
    "SurfaceRaycastAll",
    "belongs_to_prim_tree",
]
