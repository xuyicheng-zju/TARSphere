"""Viser overlay of VISUAL meshes vs COLLISION shapes for a clutter scene.

Why: the procedural bin's visual mesh is an OPEN container (hollow, walls +
floor), but its collision is registered as a single SOLID cuboid
(``collision: cuboid_from_extents``) spanning the bin's full AABB up to the
rim. So a dropped object lands on TOP of the solid block at the rim, ~10 cm
above the visible bin floor — a visual/collision mismatch (NOT a USD bug; it
is in the scene setup and the sim trajectory).

This script draws, in viser:
  * each visual mesh (table, bin, objects) as its real geometry (gray; bin
    highlighted blue),
  * each collision cuboid from ``bundle.collision_world`` as a translucent
    RED box at its registered pose/size (bin collision highlighted),
so the solid-vs-hollow discrepancy is obvious.

Run (you forward the port yourself):
  PYOPENGL_PLATFORM=egl uv run python end2end/visualize_collision_vs_visual.py \
      --env_config end2end/runs/franka_hope_v8_subset_gapfix/scene_02/env.yaml \
      --port 8090
Then: ssh -N -L 8090:localhost:8090 <host>  and open http://localhost:8090
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import trimesh

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))


def _wp_pose_to_T(pose):
    # pose = [x, y, z, qw, qx, qy, qz] (cuRobo convention)
    import trimesh.transformations as tra

    x, y, z, qw, qx, qy, qz = pose
    T = tra.quaternion_matrix([qw, qx, qy, qz])  # trimesh wants wxyz
    T[:3, 3] = [x, y, z]
    return T


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env_config", required=True, type=Path)
    ap.add_argument(
        "--robot_config", type=Path, default=_HERE / "robots/franka_panda.yaml"
    )
    ap.add_argument("--port", type=int, default=8090)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    from scene_builder import build_clutter_scene, load_yaml
    from graspgenx.utils.viser_utils import create_visualizer, visualize_mesh

    env_cfg = load_yaml(args.env_config)
    robot_cfg = load_yaml(args.robot_config)
    bundle = build_clutter_scene(env_cfg, robot_cfg, seed=args.seed)

    vis = create_visualizer(port=args.port)
    print(f"viser at http://localhost:{args.port}")

    object_ids = {o.asset_id for o in bundle.objects}

    # --- VISUAL meshes ----------------------------------------------------
    for name, (mesh, T) in bundle.vis_meshes.items():
        is_bin = "bin" in name.lower()
        color = (
            [70, 130, 230]
            if is_bin
            else ([120, 200, 120] if name in object_ids else [185, 185, 185])
        )
        visualize_mesh(vis, f"visual/{name}", mesh, color=color, transform=T)

    # --- COLLISION cuboids ------------------------------------------------
    # Drawn as translucent red boxes at the registered pose/size. The bin's
    # box will visibly fill the open visual bin up to the rim.
    n = 0
    for ob in bundle.collision_world:
        if ob.type != "cuboid" or ob.dims is None or ob.pose is None:
            continue
        box = trimesh.creation.box(extents=[float(d) for d in ob.dims])
        T = _wp_pose_to_T(ob.pose)
        is_bin = "bin" in ob.name.lower()
        color = [230, 40, 40] if is_bin else [230, 140, 40]
        visualize_mesh(vis, f"collision/{ob.name}", box, color=color, transform=T)
        zt = ob.pose[2] + ob.dims[2] / 2.0
        print(
            f"  collision[{ob.name}]: size={[round(d,3) for d in ob.dims]} "
            f"center_z={ob.pose[2]:.3f} z_top={zt:.3f}"
        )
        n += 1
    print(f"Drew {n} collision cuboids + {len(bundle.vis_meshes)} visual meshes.")
    print(
        "RED box = collision; blue = visual bin. Toggle 'collision' / "
        "'visual' folders in the viser scene tree."
    )
    print("Ctrl+C to exit.")
    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        print("bye")


if __name__ == "__main__":
    main()
