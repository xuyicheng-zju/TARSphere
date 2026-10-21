"""Read finite world geometry from a live USD stage."""

from __future__ import annotations

import math
from typing import Any


WorldBounds = tuple[list[float], list[float]]


def collision_enabled(prim: Any) -> bool:
    """Resolve live collision participation, including instance ancestors."""
    from pxr import UsdPhysics

    while prim and not prim.IsPseudoRoot():
        attr = prim.GetAttribute("physics:collisionEnabled")
        if attr and attr.HasAuthoredValueOpinion():
            return bool(attr.Get())
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return bool(UsdPhysics.CollisionAPI(prim).GetCollisionEnabledAttr().Get())
        prim = prim.GetParent()
    return False


def rigid_local_geometry_points(stage: Any, prim_path: str) -> list[list[float]]:
    """Cache actual rigid Mesh/Cube vertices in the object's authored frame.

    Rotating the eight corners of a previously aligned bounding box invents
    empty volume, especially below round targets. Cache real vertices once;
    their live rigid transform supplies bounds for support and grasp checks.
    This path requires Mesh/Cube geometry, not an extent-only approximation.
    """
    import numpy as np
    from pxr import Gf, Usd, UsdGeom

    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        raise RuntimeError(f"Rigid geometry requires an existing prim: {prim_path}")
    transforms = UsdGeom.XformCache(Usd.TimeCode.Default())
    inverse = transforms.GetLocalToWorldTransform(root).GetInverse()
    result = []
    for prim in Usd.PrimRange(root, Usd.TraverseInstanceProxies()):
        if not prim.IsA(UsdGeom.Gprim):
            continue
        if UsdGeom.Imageable(
            prim
        ).ComputeVisibility() == UsdGeom.Tokens.invisible and not collision_enabled(
            prim
        ):
            continue
        if prim.IsA(UsdGeom.Mesh):
            mesh = UsdGeom.Mesh(prim)
            points = mesh.GetPointsAttr().Get()
            counts = mesh.GetFaceVertexCountsAttr().Get()
            indices = mesh.GetFaceVertexIndicesAttr().Get()
            if all(v is None or len(v) == 0 for v in (points, counts, indices)):
                if any(
                    p != prim and p.IsA(UsdGeom.Gprim)
                    for p in Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
                ):
                    continue
                raise ValueError(f"Empty rigid mesh: {prim.GetPath()}")
            if (
                points is None
                or counts is None
                or indices is None
                or sum(counts) != len(indices)
            ):
                raise ValueError(f"Malformed rigid mesh: {prim.GetPath()}")
            holes = set(mesh.GetHoleIndicesAttr().Get() or ())
            used, offset = set(), 0
            for face, count in enumerate(counts):
                if count < 3:
                    raise ValueError(f"Invalid rigid mesh face: {prim.GetPath()}")
                if face not in holes:
                    used.update(indices[offset : offset + count])
                offset += count
            if not used or min(used) < 0 or max(used) >= len(points):
                raise ValueError(f"Invalid rigid mesh indices: {prim.GetPath()}")
            vertices = [points[i] for i in sorted(used)]
        elif prim.IsA(UsdGeom.Cube):
            half = float(UsdGeom.Cube(prim).GetSizeAttr().Get()) * 0.5
            vertices = [
                (x, y, z)
                for x in (-half, half)
                for y in (-half, half)
                for z in (-half, half)
            ]
        else:
            raise ValueError(
                f"Rigid bounds require actual Mesh/Cube vertices: {prim.GetPath()}"
            )
        relative = transforms.GetLocalToWorldTransform(prim) * inverse
        result.extend(relative.Transform(Gf.Vec3d(*map(float, p))) for p in vertices)
    points = np.asarray(result, dtype=float)
    if points.ndim != 2 or points.shape[1] != 3 or not np.all(np.isfinite(points)):
        raise ValueError(f"Rigid geometry has no finite vertices: {prim_path}")
    return np.unique(points, axis=0).tolist()


def unrotated_world_size(stage: Any, prim_path: str) -> list[float]:
    """Measure asset-axis dimensions at live world scale, without body rotation.

    A world-aligned bounding box grows when a rigid body rotates. Catalog
    dimensions are asset-axis measurements and must not be compared with that
    rotated envelope. Descendant geometry and the live scale remain measured.
    """
    from pxr import Gf, Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Live USD size requires an existing prim: {prim_path!r}")
    time = Usd.TimeCode.Default()
    transforms = UsdGeom.XformCache(time)
    transform = transforms.GetLocalToWorldTransform(prim)
    inverse = transform.GetInverse()
    lower = [math.inf] * 3
    upper = [-math.inf] * 3
    # Hierarchical BBoxCache can align descendants in world space before
    # transforming back, retaining rotation-induced inflation. Measure each
    # geometric extent directly in the body frame instead.
    for child in Usd.PrimRange(prim, Usd.TraverseInstanceProxies()):
        if not child.IsA(UsdGeom.Boundable):
            continue
        if UsdGeom.Imageable(child).ComputeVisibility(time) == UsdGeom.Tokens.invisible:
            continue
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
            UsdGeom.Boundable(child), time
        )
        if extent is None or len(extent) != 2:
            raise ValueError(f"Missing live geometry extent: {child.GetPath()}")
        relative = transforms.GetLocalToWorldTransform(child) * inverse
        for x in (extent[0][0], extent[1][0]):
            for y in (extent[0][1], extent[1][1]):
                for z in (extent[0][2], extent[1][2]):
                    point = relative.Transform(Gf.Vec3d(x, y, z))
                    lower = [min(lower[i], float(point[i])) for i in range(3)]
                    upper = [max(upper[i], float(point[i])) for i in range(3)]
    _validate_bounds(lower, upper)
    scale = Gf.Transform(transform).GetScale()
    size = [(upper[axis] - lower[axis]) * abs(float(scale[axis])) for axis in range(3)]
    if any(not math.isfinite(value) or value <= 0 for value in size):
        raise ValueError(
            f"Live USD size must have positive finite dimensions: {prim_path!r}"
        )
    return size


def world_bounds(stage: Any, prim_path: str) -> WorldBounds:
    """Return one live USD prim hierarchy's finite world-aligned bounds."""
    from pxr import Usd, UsdGeom

    prim = stage.GetPrimAtPath(prim_path)
    if not prim or not prim.IsValid():
        raise RuntimeError(f"Live USD bounds require an existing prim: {prim_path!r}")
    cache = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [
            UsdGeom.Tokens.default_,
            UsdGeom.Tokens.render,
            UsdGeom.Tokens.proxy,
            UsdGeom.Tokens.guide,
        ],
        useExtentsHint=False,
    )
    direct = _computed_bounds(cache, prim)
    if direct is not None:
        return direct

    try:
        descendants = Usd.PrimRange(prim, Usd.TraverseInstanceProxies())
    except TypeError:
        descendants = Usd.PrimRange(prim)
    valid = [
        bounds
        for candidate in descendants
        if candidate != prim
        for bounds in [_computed_bounds(cache, candidate)]
        if bounds is not None
    ]
    if not valid:
        raise RuntimeError(f"No finite live USD bounds found under {prim_path!r}")
    return (
        [min(bounds[0][axis] for bounds in valid) for axis in range(3)],
        [max(bounds[1][axis] for bounds in valid) for axis in range(3)],
    )


def bounds_in_meters(
    bounds: WorldBounds,
    *,
    authored_unit_in_meters: float,
) -> WorldBounds:
    """Convert one scene-asset bound from authored units to metres."""
    scale = float(authored_unit_in_meters)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("authored_unit_in_meters must be a finite positive number")
    lower, upper = bounds
    _validate_bounds(lower, upper)
    return (
        [float(value) * scale for value in lower],
        [float(value) * scale for value in upper],
    )


def _computed_bounds(cache: Any, prim: Any) -> WorldBounds | None:
    box = cache.ComputeWorldBound(prim).ComputeAlignedBox()
    minimum = box.GetMin()
    maximum = box.GetMax()
    lower = [float(minimum[axis]) for axis in range(3)]
    upper = [float(maximum[axis]) for axis in range(3)]
    try:
        _validate_bounds(lower, upper)
    except ValueError:
        return None
    return lower, upper


def _validate_bounds(lower: list[float], upper: list[float]) -> None:
    if (
        len(lower) != 3
        or len(upper) != 3
        or not all(
            math.isfinite(float(value)) and abs(float(value)) < 1e6
            for value in (*lower, *upper)
        )
        or any(float(lower[axis]) > float(upper[axis]) for axis in range(3))
    ):
        raise ValueError("USD bounds must contain finite ordered 3-D values")


__all__ = [
    "WorldBounds",
    "bounds_in_meters",
    "collision_enabled",
    "rigid_local_geometry_points",
    "unrotated_world_size",
    "world_bounds",
]
