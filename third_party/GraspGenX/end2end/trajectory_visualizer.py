"""Forward-kinematics helpers shared by the viser playback and MP4 export.

Given a joint configuration, return the world-frame pose of every visual link
in the URDF. Both consumers turn around and paint meshes at those poses, so
they stay in lockstep.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import trimesh
import trimesh.transformations as tra
import yourdfpy


@dataclass
class LinkVisual:
    """One renderable mesh attached to a URDF link."""

    link_name: str  # parent link
    mesh: trimesh.Trimesh  # mesh in mesh-local frame
    mesh_rel: str  # path to original file (relative to asset root for JSON export)
    visual_offset: (
        np.ndarray
    )  # 4x4 — link frame -> mesh frame (origin xyz/rpy in <visual>)


class URDFFK:
    """Forward kinematics over a URDF, returning world-frame poses for every
    visual mesh given a joint configuration.

    Powered by yourdfpy (already a GraspGen dep). cuRobo *also* exposes FK and
    is more accurate, but yourdfpy parses the visual block including mesh
    origin offsets, which cuRobo doesn't track for visualization meshes.

    Joint values are passed as a dict {joint_name: value} or a list aligned
    with the actuated joint order returned by `actuated_joint_names()`.
    """

    def __init__(self, urdf_path: str, asset_root: str | None = None):
        self.urdf_path = str(urdf_path)
        # yourdfpy resolves mesh paths relative to the URDF file by default.
        self._urdf = yourdfpy.URDF.load(
            self.urdf_path, build_collision_scene_graph=False, load_meshes=True
        )
        self.asset_root = asset_root or str(Path(self.urdf_path).parent)
        self._visuals: List[LinkVisual] = self._collect_visuals()

    # -- public API --------------------------------------------------------

    def actuated_joint_names(self) -> List[str]:
        return list(self._urdf.actuated_joint_names)

    @property
    def visuals(self) -> List[LinkVisual]:
        return list(self._visuals)

    def link_names(self) -> List[str]:
        """All link names in the URDF (including links without visuals)."""
        return list(self._urdf.link_map.keys())

    def fk(
        self,
        joint_values,
        base_T: np.ndarray | None = None,
        link_names: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """Return {link_name: T_world}.

        By default returns transforms for every link that has a visual mesh
        (the original behavior — used by the renderer). Pass `link_names` to
        request specific links, including ones without any visual (e.g.
        Robotiq's finger-pad collision-only links).
        """
        if isinstance(joint_values, dict):
            cfg = {k: float(v) for k, v in joint_values.items()}
        else:
            names = self.actuated_joint_names()
            if len(joint_values) != len(names):
                raise ValueError(
                    f"joint_values length {len(joint_values)} does not match "
                    f"actuated joints {len(names)}"
                )
            cfg = {n: float(v) for n, v in zip(names, joint_values)}

        self._urdf.update_cfg(cfg)
        out: Dict[str, np.ndarray] = {}
        if link_names is None:
            link_iter = (vis.link_name for vis in self._visuals)
        else:
            link_iter = iter(link_names)
        for ln in link_iter:
            T_link = self._urdf.get_transform(frame_to=ln)
            T_world_link = T_link if base_T is None else (base_T @ T_link)
            out[ln] = T_world_link
        return out

    def link_poses_with_visual_offset(
        self, joint_values, base_T: np.ndarray | None = None
    ) -> List[Tuple[LinkVisual, np.ndarray]]:
        """Convenience: return [(LinkVisual, T_world_for_mesh)] pairs.

        The world transform here is `T_world_link @ visual_offset` so the mesh
        is painted at its visual origin (not the link origin).
        """
        link_to_world = self.fk(joint_values, base_T=base_T)
        out = []
        for vis in self._visuals:
            T_world_mesh = link_to_world[vis.link_name] @ vis.visual_offset
            out.append((vis, T_world_mesh))
        return out

    # -- internals ---------------------------------------------------------

    def _collect_visuals(self) -> List[LinkVisual]:
        out: List[LinkVisual] = []
        urdf_dir = Path(self.urdf_path).parent
        asset_root = Path(self.asset_root).resolve()

        for link_name, link in self._urdf.link_map.items():
            for visual in link.visuals or []:
                geom = visual.geometry
                # yourdfpy.Geometry has either .mesh (a yourdfpy.Mesh metadata
                # object with filename + scale), or .box/.cylinder/.sphere
                # primitives. We only handle mesh visuals here.
                yp_mesh = getattr(geom, "mesh", None)
                if yp_mesh is None:
                    continue
                fname = getattr(yp_mesh, "filename", None)
                if not fname:
                    continue

                fpath = Path(fname)
                if not fpath.is_absolute():
                    fpath = (urdf_dir / fpath).resolve()
                if not fpath.is_file():
                    continue

                # Load the mesh directly via trimesh.
                try:
                    tm = trimesh.load(fpath, force="mesh")
                except Exception:
                    continue
                if not isinstance(tm, trimesh.Trimesh):
                    if hasattr(tm, "geometry"):
                        pieces = [
                            g
                            for g in tm.geometry.values()
                            if isinstance(g, trimesh.Trimesh)
                        ]
                        if not pieces:
                            continue
                        tm = trimesh.util.concatenate(pieces)
                    else:
                        continue

                # URDF Mesh-level scale (e.g. unit conversion). Apply to mesh.
                yp_scale = getattr(yp_mesh, "scale", None)
                if yp_scale is not None:
                    s = np.asarray(yp_scale, dtype=float)
                    if s.ndim == 0:
                        s = np.array([float(s)] * 3)
                    if not np.allclose(s, 1.0):
                        tm = tm.copy()
                        tm.apply_transform(np.diag([s[0], s[1], s[2], 1.0]))

                # Visual origin (xyz + rpy) is the link->mesh transform.
                visual_offset = np.eye(4)
                if visual.origin is not None:
                    visual_offset = np.asarray(visual.origin, dtype=float)

                # Path stored in JSON: absolute (renderer can always resolve it).
                mesh_rel = str(fpath)

                out.append(
                    LinkVisual(
                        link_name=link_name,
                        mesh=tm,
                        mesh_rel=mesh_rel,
                        visual_offset=visual_offset,
                    )
                )
        return out


def matrix_to_xyz_quat_wxyz(T: np.ndarray) -> Tuple[List[float], List[float]]:
    """4x4 -> ([x,y,z], [w,x,y,z])."""
    quat = tra.quaternion_from_matrix(T)  # already wxyz
    pos = T[:3, 3].tolist()
    return [float(p) for p in pos], [float(q) for q in quat]
