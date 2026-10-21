#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Visualize a gripper from the gripper_descriptions repo: animate the URDF from
open to close and overlay the sweep volume box(es) defined in config.json.

Usage:
    python scripts/vis_gripper_desc.py --gripper fetch
    python scripts/vis_gripper_desc.py --gripper franka_panda --port 8081
    python scripts/vis_gripper_desc.py --list

The lookup root resolves to:
    1. $GRASPGENX_GRIPPER_CFG_DIR/gripper_descriptions/assets/x_grippers, if set
    2. <repo>/ext/gripper_descriptions/gripper_descriptions/assets/x_grippers
       (auto-cloned by graspgenx on first import)
Override on the command line with --root <path>.
"""

import argparse
import json
import os
import sys
import time
from typing import Dict, List, Optional

import numpy as np
import trimesh.transformations as tra
import yourdfpy

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from graspgenx import get_gripper_descriptions_assets
from graspgenx.utils.viser_utils import (
    create_visualizer,
    make_frame,
    visualize_bbox,
    visualize_mesh,
)


def _default_root() -> Optional[str]:
    """Resolve the default --root at runtime (env var → ext/ default).

    Returns None on failure so the caller can surface a clear error rather
    than crashing at import time.
    """
    try:
        return str(get_gripper_descriptions_assets())
    except Exception as exc:  # noqa: BLE001
        print(f"[warn] could not resolve default gripper-descriptions root: {exc}")
        return None


def load_urdf(urdf_path: str) -> yourdfpy.URDF:
    return yourdfpy.URDF.load(
        urdf_path,
        build_scene_graph=True,
        load_meshes=True,
        build_collision_scene_graph=False,
        load_collision_meshes=False,
        force_mesh=False,
        force_collision_mesh=False,
    )


def list_grippers(root: str) -> List[str]:
    if not os.path.isdir(root):
        return []
    entries = []
    for d in sorted(os.listdir(root)):
        gpath = os.path.join(root, d)
        if os.path.isdir(gpath) and os.path.exists(os.path.join(gpath, "config.json")):
            entries.append(d)
    return entries


def find_urdf(gripper_dir: str) -> str:
    """Pick a URDF inside the gripper folder.

    Preference order: gripper.urdf at the root, else any *.urdf under urdf/,
    else any *.urdf at the root.
    """
    direct = os.path.join(gripper_dir, "gripper.urdf")
    if os.path.isfile(direct):
        return direct
    urdf_subdir = os.path.join(gripper_dir, "urdf")
    if os.path.isdir(urdf_subdir):
        for f in sorted(os.listdir(urdf_subdir)):
            if f.endswith(".urdf"):
                return os.path.join(urdf_subdir, f)
    for f in sorted(os.listdir(gripper_dir)):
        if f.endswith(".urdf"):
            return os.path.join(gripper_dir, f)
    raise FileNotFoundError(f"No URDF found in {gripper_dir}")


def get_link_colors(num_links: int) -> List[List[int]]:
    base_color = [80, 80, 80]
    finger_color = [50, 180, 50]
    colors = []
    for i in range(num_links):
        colors.append(base_color if i <= 1 else finger_color)
    return colors


def render_gripper(
    vis, robot: yourdfpy.URDF, js_cfg: Dict, name_prefix: str = "gripper", base_T=None
):
    """Render gripper at js_cfg, optionally pre-multiplying every link transform by base_T."""
    robot.update_cfg(js_cfg)
    scene = robot.scene
    geom_names = list(scene.geometry.keys())
    colors = get_link_colors(len(geom_names))
    if base_T is None:
        base_T = np.eye(4)
    for i, gname in enumerate(geom_names):
        mesh = scene.geometry[gname]
        tf = base_T @ scene.graph.get(gname)[0]
        m = mesh.copy()
        m.apply_transform(tf)
        visualize_mesh(vis, f"{name_prefix}/link_{i}", m, color=colors[i])


def interpolate_js(open_js: Dict, close_js: Dict, alpha: float) -> Dict:
    return {k: open_js[k] + (close_js[k] - open_js[k]) * alpha for k in open_js}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Animate a gripper from gripper_descriptions with sweep volumes overlaid",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--gripper", type=str, help="Gripper folder name under --root")
    parser.add_argument(
        "--root",
        type=str,
        default=None,
        help="Root path to gripper folders. Defaults to "
        "$GRASPGENX_GRIPPER_CFG_DIR/gripper_descriptions/assets/x_grippers "
        "if the env var is set, otherwise the auto-cloned "
        "<repo>/ext/gripper_descriptions/gripper_descriptions/assets/x_grippers.",
    )
    parser.add_argument("--port", type=int, default=8080)
    parser.add_argument(
        "--num-steps", type=int, default=30, help="Animation interpolation steps"
    )
    parser.add_argument(
        "--frame-time", type=float, default=0.05, help="Seconds per frame"
    )
    parser.add_argument(
        "--no-half",
        action="store_true",
        help="Hide the half-closed sweep volume (orange box)",
    )
    parser.add_argument("--show-world-frame", action="store_true")
    parser.add_argument(
        "--list", action="store_true", help="List grippers under --root and exit"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Resolve --root lazily so the env-var lookup happens at runtime.
    if args.root is None:
        resolved = _default_root()
        if resolved is None:
            print(
                "Error: could not determine gripper-descriptions root. "
                "Pass --root explicitly or set $GRASPGENX_GRIPPER_CFG_DIR."
            )
            sys.exit(1)
        args.root = resolved

    if args.list or not args.gripper:
        avail = list_grippers(args.root)
        print(f"Grippers under {args.root}:")
        for g in avail:
            print(f"  {g}")
        if args.list:
            return
        if not args.gripper:
            print("\nProvide --gripper <name>.")
            sys.exit(1)

    gripper_dir = os.path.join(args.root, args.gripper)
    if not os.path.isdir(gripper_dir):
        print(f"Error: {gripper_dir} not found.")
        sys.exit(1)

    config_path = os.path.join(gripper_dir, "config.json")
    if not os.path.isfile(config_path):
        print(f"Error: {config_path} not found.")
        sys.exit(1)

    urdf_path = find_urdf(gripper_dir)
    print(f"Loading URDF: {urdf_path}")
    print(f"Loading config: {config_path}")

    robot = load_urdf(urdf_path)
    with open(config_path, "r") as f:
        config = json.load(f)

    open_js = {k: float(v) for k, v in config["open"].items()}
    close_js = {k: float(v) for k, v in config["close"].items()}

    # Optional base rotation (4x4). Older configs without this key default to identity.
    base_T = np.array(config.get("base_rotation", np.eye(4).tolist()), dtype=float)
    if base_T.shape != (4, 4):
        print(
            f"[warn] base_rotation has shape {base_T.shape}, expected (4,4) — using identity."
        )
        base_T = np.eye(4)

    sv = config.get("sweep_volume", {})
    sv_extents = np.array(sv.get("extents", [0.01, 0.01, 0.01]), dtype=float)
    sv_offset = np.array(sv.get("offset", [0.0, 0.0, 0.0]), dtype=float)
    sv2_extents = np.array(sv.get("extents2", sv_extents), dtype=float)
    sv2_offset = np.array(sv.get("offset2", sv_offset), dtype=float)

    vis = create_visualizer(port=args.port)
    if args.show_world_frame:
        make_frame(vis, "world", h=0.10, radius=0.002)

    # Sweep volume boxes (drawn once; the gripper is what animates)
    tf_open = tra.translation_matrix(sv_offset)
    visualize_bbox(vis, "sweep_volume_open", sv_extents, T=tf_open, color=[0, 100, 255])
    if not args.no_half:
        tf_half = tra.translation_matrix(sv2_offset)
        visualize_bbox(
            vis, "sweep_volume_half", sv2_extents, T=tf_half, color=[255, 165, 0]
        )

    print(
        f"\nGripper: {args.gripper} | type: {config.get('type', '?')}"
        f"\nOpen joints:  {open_js}"
        f"\nClose joints: {close_js}"
        f"\nSweep volume (open):  extents={sv_extents.tolist()}, offset={sv_offset.tolist()}"
        f"\nSweep volume (half):  extents={sv2_extents.tolist()}, offset={sv2_offset.tolist()}"
        f"\n\nViser at http://localhost:{args.port} — Ctrl+C to stop.\n"
    )

    traj = [
        interpolate_js(open_js, close_js, s / args.num_steps)
        for s in range(args.num_steps + 1)
    ]
    try:
        while True:
            for js in traj:
                render_gripper(vis, robot, js, base_T=base_T)
                time.sleep(args.frame_time)
            for js in reversed(traj):
                render_gripper(vis, robot, js, base_T=base_T)
                time.sleep(args.frame_time)
    except KeyboardInterrupt:
        print("\nExiting...")


if __name__ == "__main__":
    main()
