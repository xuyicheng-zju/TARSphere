"""Canonical live geometry checks for scene-bound Human assets.

Production layout geometry records tight bounds over the transformed points of
visible render meshes.  A prim-level ``BBoxCache`` bound is intentionally not
equivalent: after rotation it expands the mesh's local AABB and can be wider
than the authored tight bound even when the source USD is unchanged.
"""

from __future__ import annotations

from typing import Any

import numpy as np


_RENDER_ONLY_PATH_TOKENS = ("/__default_setting/", "/hdr_sphere/", "/sky/")


def tight_visible_mesh_bounds(prim: Any) -> tuple[np.ndarray, np.ndarray]:
    """Return tight world-space bounds using the authoring mesh contract.

    The live scene mount already applies the scene-to-metre scale, so the
    resulting world coordinates are in the same metre frame as episode data.
    """

    from pxr import Usd, UsdGeom

    xforms = UsdGeom.XformCache()
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    for mesh_prim in Usd.PrimRange(prim):
        path = str(mesh_prim.GetPath()).lower()
        if not mesh_prim.IsA(UsdGeom.Mesh):
            continue
        if any(token in path for token in _RENDER_ONLY_PATH_TOKENS):
            continue
        if UsdGeom.Imageable(mesh_prim).ComputeVisibility() == UsdGeom.Tokens.invisible:
            continue
        points = UsdGeom.Mesh(mesh_prim).GetPointsAttr().Get()
        if points is None:
            continue
        local = np.asarray(points, dtype=np.float64)
        if local.ndim != 2 or local.shape[1] != 3 or len(local) == 0:
            continue
        matrix = np.asarray(
            xforms.GetLocalToWorldTransform(mesh_prim),
            dtype=np.float64,
        )
        homogeneous = np.concatenate(
            (local, np.ones((len(local), 1), dtype=np.float64)),
            axis=1,
        )
        world_h = homogeneous @ matrix
        world = world_h[:, :3] / world_h[:, 3:4]
        if not np.all(np.isfinite(world)):
            raise RuntimeError(
                f"Scene-bound Human mesh has non-finite vertices: {mesh_prim.GetPath()}"
            )
        minima.append(world.min(axis=0))
        maxima.append(world.max(axis=0))

    if not minima:
        raise RuntimeError(
            f"Scene-bound Human has no visible mesh geometry: {prim.GetPath()}"
        )
    return np.min(minima, axis=0), np.max(maxima, axis=0)
