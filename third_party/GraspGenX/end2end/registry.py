"""Asset and collision factories for environment YAMLs.

Adding a new asset type or collision representation = drop a function here
and add it to the dispatch tables. The demo script never branches on env type.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import trimesh
import trimesh.transformations as tra

import scene_synthesizer as synth
from scene_synthesizer import procedural_assets as pa


@dataclass
class CollisionObstacle:
    """One cuboid/sphere/mesh obstacle for cuRobo's collision world."""

    name: str
    type: str  # "cuboid" | "sphere" | "mesh"
    # Cuboid fields
    dims: Optional[List[float]] = None
    pose: Optional[List[float]] = None  # [x, y, z, qw, qx, qy, qz] — cuRobo convention
    # Sphere fields
    radius: Optional[float] = None
    # Mesh fields
    mesh_file: Optional[str] = None
    scale: Optional[List[float]] = None

    def to_curobo_dict(self) -> Dict[str, Any]:
        if self.type == "cuboid":
            return {"dims": self.dims, "pose": self.pose}
        if self.type == "sphere":
            return {"radius": self.radius, "pose": self.pose[:3]}
        if self.type == "mesh":
            return {
                "file_path": self.mesh_file,
                "pose": self.pose,
                "scale": self.scale or [1, 1, 1],
            }
        raise ValueError(f"Unknown obstacle type: {self.type}")


# ---------------------------------------------------------------------------
# Asset factories: type-string -> callable(params) -> scene_synthesizer.Asset
# ---------------------------------------------------------------------------


def _make_table_asset(params: Dict[str, Any]):
    """Procedural table from scene_synthesizer."""
    return pa.TableAsset(
        width=params.get("width", 0.8),
        depth=params.get("depth", 0.8),
        height=params.get("height", 0.7),
        thickness=params.get("thickness", 0.04),
        leg_thickness=params.get("leg_thickness", 0.05),
    )


def _make_bin_asset(params: Dict[str, Any]):
    """Procedural bin (open box) from scene_synthesizer.

    `angle` slants the side walls (positive = flared/outward). Note that
    BinAsset silently ignores `angle` when `use_primitives=True`, so we
    default `use_primitives=False` whenever a non-zero angle is requested.
    """
    angle = float(params.get("angle", 0.0))
    use_primitives = params.get("use_primitives", angle == 0.0)
    if angle != 0.0 and use_primitives:
        # Avoid the silent "angle ignored" warning from BinAsset.
        use_primitives = False
    return pa.BinAsset(
        width=params.get("width", 0.4),
        depth=params.get("depth", 0.3),
        height=params.get("height", 0.18),
        thickness=params.get("thickness", 0.005),
        angle=angle,
        wired=params.get("wired", False),
        use_primitives=use_primitives,
    )


def _make_shelf_asset(params: Dict[str, Any]):
    """Procedural shelf from scene_synthesizer."""
    return pa.ShelfAsset(
        width=params.get("width", 0.7),
        depth=params.get("depth", 0.4),
        height=params.get("height", 1.0),
        num_boards=params.get("num_boards", 3),
        board_thickness=params.get("board_thickness", 0.025),
        backboard_thickness=params.get("backboard_thickness", 0.02),
        num_side_columns=params.get("num_side_columns", 2),
        column_thickness=params.get("column_thickness", 0.04),
        bottom_board=params.get("bottom_board", True),
        cylindrical_columns=params.get("cylindrical_columns", False),
        num_vertical_boards=params.get("num_vertical_boards", 0),
    )


def _make_cabinet_asset(params: Dict[str, Any]):
    """Procedural cabinet from scene_synthesizer (URDFAsset)."""
    return pa.CabinetAsset(
        width=params.get("width", 0.6),
        depth=params.get("depth", 0.4),
        height=params.get("height", 0.6),
    )


def _make_mesh_asset(params: Dict[str, Any]):
    """Static mesh (.obj/.stl/.ply) wrapped as a scene_synthesizer Asset."""
    return synth.MeshAsset(
        params["mesh_file"],
        scale=params.get("scale", 1.0),
    )


def _make_urdf_asset(params: Dict[str, Any]):
    """URDF (articulated or rigid) loaded as a scene_synthesizer Asset."""
    return synth.URDFAsset(
        params["urdf_path"],
        configuration=params.get("configuration", None),
    )


ASSET_FACTORIES: Dict[str, Callable[[Dict[str, Any]], Any]] = {
    "procedural_table": _make_table_asset,
    "procedural_bin": _make_bin_asset,
    "procedural_shelf": _make_shelf_asset,
    "procedural_cabinet": _make_cabinet_asset,
    "mesh_asset": _make_mesh_asset,
    "urdf_asset": _make_urdf_asset,
}


# ---------------------------------------------------------------------------
# Collision factories: collision-string -> callable(asset_id, asset_pose, scene)
#   -> list[CollisionObstacle]
# ---------------------------------------------------------------------------


def _xyzw_to_wxyz(q_xyzw: List[float]) -> List[float]:
    """[x, y, z, w] -> [w, x, y, z] (cuRobo convention)."""
    if (
        q_xyzw is None
        or all(abs(v) < 1e-9 for v in q_xyzw[:3])
        and abs(q_xyzw[3] - 1.0) < 1e-9
    ):
        return [1.0, 0.0, 0.0, 0.0]
    x, y, z, w = q_xyzw
    return [w, x, y, z]


def _aabb_to_cuboid(
    asset_id: str, asset_pose: Dict[str, Any], asset, scene
) -> List[CollisionObstacle]:
    """Snap a cuboid around the asset's axis-aligned bounding box.

    Computed from the asset's *local-frame* mesh bounds and then transformed
    into the world frame using `asset_pose`. We deliberately skip
    scene.get_geometries() because it can return mixed frames depending on
    the asset's joint type.
    """
    m = asset.mesh()
    if not isinstance(m, trimesh.Trimesh):
        # scene_synthesizer returns a trimesh.Scene for multi-mesh assets
        # (TableAsset has 5 sub-geoms). Concatenate so we get one AABB.
        pieces = [g for g in m.geometry.values() if isinstance(g, trimesh.Trimesh)]
        if not pieces:
            return []
        m = trimesh.util.concatenate(pieces)
    bounds = m.bounds  # (2, 3) in asset-local frame

    extents = (bounds[1] - bounds[0]).tolist()
    center_local = (bounds[0] + bounds[1]) / 2.0  # in asset-local frame

    # Asset world pose
    T_world = np.eye(4)
    T_world[:3, 3] = asset_pose["translation"]
    q_xyzw = asset_pose.get("quaternion_xyzw", [0, 0, 0, 1])
    if not (
        abs(q_xyzw[0]) < 1e-9
        and abs(q_xyzw[1]) < 1e-9
        and abs(q_xyzw[2]) < 1e-9
        and abs(q_xyzw[3] - 1) < 1e-9
    ):
        T_world[:3, :3] = tra.quaternion_matrix(
            [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]]
        )[:3, :3]

    # Center of the AABB transformed into the world frame.
    center_world = (T_world @ np.array([*center_local, 1.0]))[:3]

    pose_curobo = [
        float(center_world[0]),
        float(center_world[1]),
        float(center_world[2]),
        1.0,
        0.0,
        0.0,
        0.0,
    ]

    return [
        CollisionObstacle(
            name=asset_id,
            type="cuboid",
            dims=[float(x) for x in extents],
            pose=pose_curobo,
        )
    ]


def _asset_to_mesh_obstacle(
    asset_id: str, asset_pose: Dict[str, Any], asset, scene
) -> List[CollisionObstacle]:
    """Use the asset's mesh as a collision obstacle. Heavier than a cuboid
    but accurate for shelves/bins."""
    try:
        m = asset.mesh()
    except Exception:
        return []

    # Dump to a temp file scene_synthesizer can load; cuRobo wants a path.
    import tempfile

    f = tempfile.NamedTemporaryFile(suffix=".obj", delete=False)
    f.close()
    m.export(f.name)

    pose_curobo = [
        float(asset_pose["translation"][0]),
        float(asset_pose["translation"][1]),
        float(asset_pose["translation"][2]),
        *_xyzw_to_wxyz(asset_pose.get("quaternion_xyzw", [0, 0, 0, 1])),
    ]

    return [
        CollisionObstacle(
            name=asset_id,
            type="mesh",
            mesh_file=f.name,
            pose=pose_curobo,
            scale=[1.0, 1.0, 1.0],
        )
    ]


COLLISION_FACTORIES: Dict[
    str,
    Callable[[str, Dict[str, Any], Any, Any], List[CollisionObstacle]],
] = {
    "cuboid_from_extents": _aabb_to_cuboid,
    "mesh": _asset_to_mesh_obstacle,
}


def make_asset(asset_cfg: Dict[str, Any]):
    """Dispatch an asset config dict through the asset factory."""
    t = asset_cfg["type"]
    if t not in ASSET_FACTORIES:
        raise KeyError(
            f"Unknown asset type '{t}'. Registered: {list(ASSET_FACTORIES)}. "
            f"Add a factory in registry.py."
        )
    return ASSET_FACTORIES[t](asset_cfg.get("params", {}))


def make_collision(asset_cfg: Dict[str, Any], asset, scene) -> List[CollisionObstacle]:
    """Dispatch an asset's collision spec through the collision factory."""
    spec = asset_cfg.get("collision")
    if spec is None or spec == "skip":
        # "skip" lets an env include an asset for visualization only — used
        # for bins/cabinets where the simple AABB cuboid would block the
        # gripper from reaching inside.
        return []
    if spec not in COLLISION_FACTORIES:
        raise KeyError(
            f"Unknown collision spec '{spec}'. Registered: {list(COLLISION_FACTORIES)}. "
            f"Add a factory in registry.py."
        )
    return COLLISION_FACTORIES[spec](asset_cfg["id"], asset_cfg["pose"], asset, scene)
