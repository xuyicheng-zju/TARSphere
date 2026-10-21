#!/usr/bin/env python3
"""Visualize per-object grasps for a multi-object scene in viser.

For each object in the env YAML's `object_slots`, runs GraspGenX
inference and classifies each grasp as collision-free (GREEN) or
colliding (RED) using the SAME inflated-AABB + reach heuristic that
`clutter_task._count_collision_free_grasps` uses to pick the next
target during the clutter run.

Usage:
    PYOPENGL_PLATFORM=egl uv run python end2end/visualize_scene_grasps.py \\
        --env_config end2end/runs/franka_hope_v8_sweep20/scene_00/env.yaml \\
        --port 8080
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

import numpy as np
import trimesh
import yaml

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

logging.basicConfig(format="%(asctime)s [VIZ] %(message)s", level=logging.INFO)
log = logging.getLogger("viz")

# Reach + AABB thresholds — must mirror clutter_task.py constants.
GRASP_REACH_MIN = 0.20
GRASP_REACH_MAX = 0.85
GRASP_AABB_MARGIN = 0.02


def _object_aabb_world(
    mesh: trimesh.Trimesh, T_world: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """AABB center + half-extents of `mesh` placed at `T_world`."""
    verts_local = np.asarray(mesh.vertices, dtype=np.float64)
    R, t = T_world[:3, :3], T_world[:3, 3]
    verts_world = verts_local @ R.T + t
    lo = verts_world.min(axis=0)
    hi = verts_world.max(axis=0)
    center = 0.5 * (lo + hi)
    half = 0.5 * (hi - lo)
    return center, half


def classify_grasps(
    grasps_world: np.ndarray,
    target_obj,
    all_objects: List,
    bundle_vis_meshes: dict,
    gripper_mesh: trimesh.Trimesh,
    robot_base_xyz: np.ndarray,
    check_reach: bool = True,
    include_target: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Split grasps into (collision_free_idx, colliding_idx).

    A grasp is collision-free iff
      (1) the grasp's tool position is within the robot's reach band, AND
      (2) the gripper mesh placed at the grasp's canonical pose does NOT
          intersect the scene (table + bin + non-target objects, and the
          target object itself when ``include_target`` is True).

    ``include_target`` (default True) reflects the **fully-observed (mesh)**
    setting: we have the target's full mesh, so a grasp whose gripper
    geometry penetrates the target (e.g. palm-into-object, or an object too
    fat for the finger gap) is a real collision and should be rejected. In
    the future **partial-observation (point-cloud)** setting the target is
    only partially seen, so it is excluded (``include_target=False``) — the
    gripper is assumed to wrap whatever it's grasping.

    Uses trimesh.collision.CollisionManager (fcl-backed).
    """
    if len(grasps_world) == 0:
        return np.zeros(0, dtype=int), np.zeros(0, dtype=int)

    # Build a fcl CollisionManager containing every static mesh, every
    # non-target object, and (in fully-observed mode) the target itself.
    #
    # NOTE: ``bundle.vis_meshes`` holds ONLY the static scene (table, bin) —
    # the manipulation objects live in ``all_objects`` (obj.mesh @ obj.world_T)
    # and are NOT in vis_meshes. So we must add the objects from
    # ``all_objects`` explicitly; iterating vis_meshes alone would (and used
    # to) silently skip every object, leaving the gripper checked against the
    # table/bin only.
    obj_ids = {o.asset_id for o in all_objects}
    target_id = getattr(target_obj, "asset_id", None)
    mgr = trimesh.collision.CollisionManager()
    n_obstacles = 0
    # Static scene (table, bin, …) — skip anything that is actually an object.
    for name, (mesh, T_world) in bundle_vis_meshes.items():
        if name in obj_ids:
            continue
        mgr.add_object(name, mesh, transform=np.asarray(T_world, np.float64))
        n_obstacles += 1
    # Manipulation objects: neighbours are always obstacles; the target is
    # included only in fully-observed mode.
    for o in all_objects:
        is_target = (o is target_obj) or (getattr(o, "asset_id", None) == target_id)
        if is_target and not include_target:
            continue
        mgr.add_object(o.asset_id, o.mesh, transform=np.asarray(o.world_T, np.float64))
        n_obstacles += 1

    cf_idx, hit_idx = [], []
    for i, T in enumerate(grasps_world):
        # Reach band (cheap pre-filter; doesn't need fcl).
        if check_reach:
            pos = T[:3, 3]
            d = float(np.linalg.norm(pos - robot_base_xyz))
            if d > GRASP_REACH_MAX or d < GRASP_REACH_MIN:
                hit_idx.append(i)
                continue
        # Gripper-mesh-vs-scene collision.
        hit = mgr.in_collision_single(gripper_mesh, transform=T.astype(np.float64))
        (hit_idx if hit else cf_idx).append(i)
    return np.asarray(cf_idx, dtype=int), np.asarray(hit_idx, dtype=int)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--env_config",
        type=Path,
        required=True,
        help="Path to env.yaml (must have object_slots).",
    )
    ap.add_argument(
        "--robot_config", type=Path, default=_HERE / "robots/franka_panda.yaml"
    )
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--num_grasps", type=int, default=200)
    ap.add_argument("--topk", type=int, default=80)
    ap.add_argument(
        "--threshold",
        type=float,
        default=-1.0,
        help="Confidence threshold (matches viz_grasps.py "
        "default = -1.0 = no thresholding). Pass 0.7 to "
        "match the e2e sweep filter.",
    )
    ap.add_argument(
        "--planner",
        choices=["graspmoe", "diffusion", "topdown"],
        default="graspmoe",
        help="Grasp planner. 'topdown' keeps ONLY the OBB "
        "(top-down) grasps (drops the diffusion branch) — use "
        "it to inspect the top-down grasps the e2e "
        "--planner topdown run uses.",
    )
    ap.add_argument(
        "--moe_obb_density",
        choices=["sparse", "dense", "none"],
        default="sparse",
        help="Matches viz_grasps.py default. 'dense' = grid of "
        "OBB positions x yaws x z-offsets (~1000+ candidates) "
        "is what the e2e sweep uses.",
    )
    ap.add_argument("--num_sample_points", type=int, default=2000)
    ap.add_argument(
        "--max_grasps_per_object",
        type=int,
        default=200,
        help="Cap on grasps drawn per object (after planner).",
    )
    ap.add_argument(
        "--show_top_grasp_mesh",
        type=int,
        default=1,
        help="Render the gripper's vis_mesh.obj at the top-N "
        "highest-confidence grasp per object. 0 = off. "
        "Mesh is drawn at the CANONICAL pose (vis_mesh "
        "lives in the canonical frame, not the tool frame).",
    )
    # viz draws gripper control points in the GraspGenX canonical
    # convention (X = closing axis). Applying T_offset (which rotates
    # canonical -> tool0, e.g. franka's +90 deg about Z) rotates the
    # drawn wireframes away from where the actual gripper geometry
    # sits — making them visually inconsistent with the top-grasp
    # vis_mesh. Default OFF so wireframes line up with the mesh and
    # with the object. Opt in only if you specifically want to see
    # "where cuRobo plans to" (i.e. canonical @ T_offset).
    ap.add_argument(
        "--observation_mode",
        choices=["fully_observed", "partial_observation"],
        default="fully_observed",
        help="fully_observed (default): the target object's full "
        "mesh IS included in the gripper-collision check, so "
        "grasps that penetrate the target are rejected. "
        "partial_observation: the target is EXCLUDED (future "
        "point-cloud setting where the object is only "
        "partially seen and the gripper is assumed to wrap it).",
    )
    ap.add_argument(
        "--apply_grasp_to_tool",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Apply robot_cfg.grasp_to_tool_transform to each "
        "drawn grasp. OFF by default — keeps wireframes at the "
        "canonical pose so they align with the gripper "
        "vis_mesh and the object. Turning it ON rotates the "
        "wireframes by the canonical->tool0 transform (e.g. "
        "Franka's +90 deg about Z), which is the frame cuRobo "
        "plans to, NOT how the gripper geometry sits — so the "
        "wireframes will look 90 deg off the mesh. Opt in only "
        "to inspect the planning target frame.",
    )
    args = ap.parse_args()

    # Lazy imports to keep CLI snappy.
    from scene_builder import build_clutter_scene
    from graspgenx.grasp_server import GraspGenXSampler
    from graspgenx.utils.checkpoint_io import load_model_cfg
    from graspgenx.utils.viser_utils import (
        create_visualizer,
        visualize_mesh,
        visualize_x_grasp,
    )
    from graspgenx.x_grippers import resolve_gripper_info
    from clutter_task import _graspgen_for_object

    env_cfg = yaml.safe_load(args.env_config.read_text())
    robot_cfg = yaml.safe_load(args.robot_config.read_text())

    log.info("Building scene from %s", args.env_config)
    bundle = build_clutter_scene(env_cfg, robot_cfg, seed=args.seed)
    log.info("Scene has %d objects", len(bundle.objects))

    # Robot base pose for reach check.
    base_cfg = robot_cfg.get("robot_base_pose", {})
    base_t = np.asarray(base_cfg.get("translation", [0.0, 0.0, 0.0]), dtype=np.float64)
    log.info(
        "Robot base xyz = %s (reach band [%.2f, %.2f])",
        base_t.tolist(),
        GRASP_REACH_MIN,
        GRASP_REACH_MAX,
    )

    # Load GraspGenX once (mirror e2e_grasp_demo.run_graspgen).
    gg_cfg = robot_cfg["graspgen"]
    gripper_name = gg_cfg["gripper_name"]
    # checkpoints_dir is optional in the robot YAML now (Franka omits it).
    # Fall back to GraspGenX's managed location (same as e2e_grasp_demo's
    # _resolve_checkpoints_dir).
    if gg_cfg.get("checkpoints_dir"):
        ckpts_dir = Path(gg_cfg["checkpoints_dir"]).expanduser().resolve()
    else:
        from graspgenx import get_checkpoints_version_dir

        ckpts_dir = Path(get_checkpoints_version_dir()).resolve()
    gen_dir = ckpts_dir / "gen"
    dis_dir = ckpts_dir / "dis"
    assets_dir = gg_cfg.get("assets_dir") or str(
        _HERE.parent / "ext/gripper_descriptions/gripper_descriptions/assets"
    )
    log.info("Loading GraspGenX from %s (gripper=%s)", ckpts_dir, gripper_name)
    model_cfg = load_model_cfg(
        str(gen_dir), str(dis_dir), gg_cfg.get("gen_pth"), gg_cfg.get("dis_pth")
    )
    sampler = GraspGenXSampler(model_cfg, gripper_name, assets_dir=assets_dir)

    # Resolve the real gripper geometry so visualize_x_grasp draws the
    # ACTUAL Franka/Inspire stroke/depth, not viser_utils.py's generic
    # parallel-jaw stub (the wrong-shape grasps you saw came from
    # visualize_grasp(), the stub). viz_grasps.py:168 uses this exact
    # pattern.
    gripper_info = resolve_gripper_info(gripper_name, assets_dir)
    log.info("Resolved gripper_info for %s from %s", gripper_name, assets_dir)

    # Load the gripper vis_mesh ONCE for the fcl collision check. The
    # mesh is in canonical frame; we'll place it at each grasp's
    # canonical pose during classify_grasps.
    vis_mesh_path = Path(assets_dir) / "x_grippers" / gripper_name / "vis_mesh.obj"
    if not vis_mesh_path.exists():
        raise FileNotFoundError(
            f"Need gripper vis_mesh.obj at {vis_mesh_path} for collision check"
        )
    gripper_mesh = trimesh.load(str(vis_mesh_path), force="mesh")
    log.info(
        "Loaded gripper collision mesh: %d verts, extents=%s",
        len(gripper_mesh.vertices),
        gripper_mesh.extents.round(3).tolist(),
    )

    # Build the canonical->tool0 offset that cuRobo's plan_grasp sees.
    # Without this, viz shows the GraspGenX canonical frame, which can
    # be rotated/translated from the actual gripper tool frame (e.g.
    # franka's panda_hand needs a +90 deg Z rotation; inspire has a
    # -4cm Z retract along the approach axis). Mirrors viz_grasps.py.
    T_offset = np.eye(4)
    if args.apply_grasp_to_tool:
        import trimesh.transformations as tra

        g2t = robot_cfg.get("grasp_to_tool_transform", {})
        tt = g2t.get("translation", [0.0, 0.0, 0.0])
        qq = g2t.get("quaternion_xyzw", [0.0, 0.0, 0.0, 1.0])
        T_offset[:3, 3] = tt
        if not (
            abs(qq[0]) < 1e-9
            and abs(qq[1]) < 1e-9
            and abs(qq[2]) < 1e-9
            and abs(qq[3] - 1.0) < 1e-9
        ):
            T_offset[:3, :3] = tra.quaternion_matrix([qq[3], qq[0], qq[1], qq[2]])[
                :3, :3
            ]
        log.info("Applying grasp_to_tool offset: trans=%s, quat_xyzw=%s", tt, qq)

    vis = create_visualizer(clear=True, port=args.port)
    log.info("viser server running at http://localhost:%d", args.port)

    # Draw static scene meshes (table, bin) in subtle gray. SceneBundle
    # exposes these via `vis_meshes` as {name: (mesh, world_T)}. We
    # skip the manipulation objects here — they're drawn per-object
    # below in white alongside their grasps.
    object_ids = {o.asset_id for o in bundle.objects}
    for name, (mesh, T) in bundle.vis_meshes.items():
        if name in object_ids:
            continue
        visualize_mesh(vis, f"scene/{name}", mesh, color=[180, 180, 180], transform=T)

    # Per-object grasp inference + classification.
    for i, obj in enumerate(bundle.objects):
        label = getattr(obj, "label", f"object_{i}")
        log.info(
            "[%s] running GraspGenX (planner=graspmoe, density=%s)...",
            label,
            args.moe_obb_density,
        )
        grasps_world, conf = _graspgen_for_object(
            sampler,
            obj.mesh,
            obj.world_T,
            num_sample_points=args.num_sample_points,
            num_grasps=args.num_grasps,
            topk=args.topk,
            grasp_threshold=args.threshold,
            planner_name=args.planner,
            moe_obb_density=args.moe_obb_density,
        )
        log.info(
            "[%s] %d grasps returned (conf range [%.3f, %.3f])",
            label,
            len(grasps_world),
            float(conf.min()) if len(conf) else 0.0,
            float(conf.max()) if len(conf) else 0.0,
        )

        # Draw the object mesh itself in white.
        visualize_mesh(
            vis,
            f"scene/{label}/mesh",
            obj.mesh,
            color=[230, 230, 230],
            transform=obj.world_T,
        )

        # Keep canonical poses for the top-grasp mesh draw — vis_mesh.obj
        # is authored in the gripper's canonical frame, not the tool frame.
        grasps_world_canonical = grasps_world.copy()

        # Cap visualized grasp count (highest-confidence first).
        if len(grasps_world) > args.max_grasps_per_object:
            order = np.argsort(-conf)[: args.max_grasps_per_object]
            grasps_world = grasps_world[order]
            grasps_world_canonical = grasps_world_canonical[order]
            conf = conf[order]

        # Real mesh-vs-mesh collision check: place gripper vis_mesh at
        # each grasp's canonical pose and intersect against table + bin
        # + non-target objects (+ the target itself in fully_observed
        # mode) via an fcl-backed CollisionManager.
        cf_idx, hit_idx = classify_grasps(
            grasps_world,
            target_obj=obj,
            all_objects=bundle.objects,
            bundle_vis_meshes=bundle.vis_meshes,
            gripper_mesh=gripper_mesh,
            robot_base_xyz=base_t,
            include_target=(args.observation_mode == "fully_observed"),
        )
        log.info(
            "[%s] %d collision-free (green), %d colliding (red) "
            "[obs_mode=%s, target %s collision set]",
            label,
            len(cf_idx),
            len(hit_idx),
            args.observation_mode,
            "IN" if args.observation_mode == "fully_observed" else "NOT in",
        )

        # Draw at the TOOL pose (canonical @ T_offset) so the gripper
        # appears where cuRobo's plan_grasp actually targets it.
        grasps_drawn = (
            np.einsum("nij,jk->nik", grasps_world, T_offset)
            if args.apply_grasp_to_tool
            else grasps_world
        )
        # Group collision-free / colliding under separate scene-tree
        # folders so viser's GUI lets you toggle each group as a unit.
        for j in cf_idx:
            visualize_x_grasp(
                vis,
                f"grasps/{label}/collision_free/g_{j:04d}",
                grasps_drawn[j],
                color=[0, 200, 0],
                gripper_info=gripper_info,
                linewidth=1.0,
            )
        for j in hit_idx:
            visualize_x_grasp(
                vis,
                f"grasps/{label}/colliding/g_{j:04d}",
                grasps_drawn[j],
                color=[200, 0, 0],
                gripper_info=gripper_info,
                linewidth=0.6,
            )

        # Drop the gripper vis_mesh at the top-N grasps. ALWAYS at the
        # canonical pose — vis_mesh.obj is authored in the canonical
        # frame (X = closing) per viz_grasps.py:232. Drawing at tool
        # pose would visually rotate the mesh 90 deg from where it
        # should appear on the object.
        if args.show_top_grasp_mesh > 0:
            import trimesh as _tm

            vis_mesh_path = (
                Path(assets_dir) / "x_grippers" / gripper_name / "vis_mesh.obj"
            )
            if not vis_mesh_path.exists():
                log.warning(
                    "No vis_mesh.obj at %s — skipping top-grasp mesh.", vis_mesh_path
                )
            else:
                hand_mesh = _tm.load(str(vis_mesh_path), force="mesh")
                # Only consider collision-free grasps for the top-N
                # mesh draw — showing a colliding "best" grasp is
                # misleading. If no CF grasps exist, draw nothing.
                if len(cf_idx) == 0:
                    log.info(
                        "[%s] no collision-free grasps — skipping " "top-grasp mesh",
                        label,
                    )
                    top_order = np.zeros(0, dtype=int)
                else:
                    top_k = min(args.show_top_grasp_mesh, len(cf_idx))
                    cf_conf = conf[cf_idx]
                    top_order = cf_idx[np.argsort(-cf_conf)[:top_k]]
                for rank, idx in enumerate(top_order):
                    # Light blue, getting slightly lighter for lower ranks.
                    fade = min(70, 20 * rank)
                    color = [min(255, 120 + fade), min(255, 190 + fade), 255]
                    visualize_mesh(
                        vis,
                        f"grasps/{label}/top_grasp_mesh/rank_{rank:02d}_g_{idx:04d}",
                        hand_mesh,
                        color=color,
                        transform=grasps_world_canonical[idx],
                    )
                if len(top_order) > 0:
                    log.info(
                        "[%s] drew top-%d collision-free grasp meshes "
                        "at CANONICAL pose (light blue)",
                        label,
                        len(top_order),
                    )

    log.info(
        "All %d objects visualized. Browse http://localhost:%d",
        len(bundle.objects),
        args.port,
    )
    log.info("Ctrl+C to exit (viser continues serving until you do).")
    import time

    try:
        while True:
            time.sleep(60)
    except KeyboardInterrupt:
        log.info("Bye.")


if __name__ == "__main__":
    main()
