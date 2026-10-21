# Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
#
# Headless smoke test for graspgenx.samplers.run_graspmoe.
#
# Loads a checkpoint pair + one sample JSON, runs the union sampler, and
# prints per-branch counts and score ranges. Exits 0 on success.

import argparse
import glob
import json
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import torch
import trimesh.transformations as tra

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_graspmoe
from scripts.demo_object_pc import load_model_cfg


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sample_data_dir", required=True)
    p.add_argument("--gen_dir", required=True)
    p.add_argument("--dis_dir", required=True)
    p.add_argument("--gripper_name", required=True)
    p.add_argument("--assets_dir", default=None)
    p.add_argument("--num_files", type=int, default=1,
                   help="How many JSON files to process (default 1).")
    p.add_argument("--num_grasps", type=int, default=200)
    p.add_argument("--num_yaws", type=int, default=12,
                   help="Smaller default for smoke test speed (full demo uses 36).")
    p.add_argument("--z_offsets_cm", type=str, default="-8,-4,0")
    p.add_argument("--grasp_threshold", type=float, default=-1.0)
    p.add_argument("--topk_num_grasps", type=int, default=100)
    return p.parse_args()


def main():
    args = parse_args()
    if args.assets_dir is None:
        args.assets_dir = str(REPO_ROOT / "assets")

    print(f"[smoke] sample_data_dir = {args.sample_data_dir}")
    print(f"[smoke] gen_dir         = {args.gen_dir}")
    print(f"[smoke] dis_dir         = {args.dis_dir}")
    print(f"[smoke] gripper_name    = {args.gripper_name}")
    print(f"[smoke] assets_dir      = {args.assets_dir}")

    json_files = sorted(glob.glob(os.path.join(args.sample_data_dir, "*.json")))
    if not json_files:
        print(f"[smoke] FAIL: no JSON files under {args.sample_data_dir}", file=sys.stderr)
        sys.exit(2)
    json_files = json_files[: args.num_files]
    print(f"[smoke] processing {len(json_files)} file(s)")

    print("[smoke] loading model config + checkpoints ...")
    cfg = load_model_cfg(args.gen_dir, args.dis_dir, None, None)
    print("[smoke] instantiating GraspGenXSampler ...")
    sampler = GraspGenXSampler(cfg, args.gripper_name, assets_dir=args.assets_dir)
    print(f"[smoke] gripper depth={sampler.gripper.depth:.5f}m  "
          f"width={sampler.gripper.width:.5f}m")

    z_offsets = tuple(float(x) for x in args.z_offsets_cm.split(","))
    n_pass = 0
    n_fail = 0

    for f in json_files:
        print(f"\n[smoke] === {os.path.basename(f)} ===")
        with open(f, "r") as fh:
            data = json.load(fh)
        pc = np.array(data["pc"])
        T_sub = tra.translation_matrix(-pc.mean(axis=0))
        pc_centered = tra.transform_points(pc, T_sub).astype(np.float32)
        print(f"[smoke] pc shape={pc_centered.shape}  "
              f"bounds={pc_centered.min(0).round(3).tolist()}..{pc_centered.max(0).round(3).tolist()}")

        try:
            moe = run_graspmoe(
                pc_centered,
                sampler,
                grasp_threshold=args.grasp_threshold,
                num_grasps=args.num_grasps,
                topk_num_grasps=args.topk_num_grasps,
                num_yaws=args.num_yaws,
                z_offsets_cm=z_offsets,
            )
        except Exception as e:
            print(f"[smoke] FAIL: run_graspmoe raised: {e}", file=sys.stderr)
            traceback.print_exc()
            n_fail += 1
            continue

        nd, no = len(moe["grasps_diff"]), len(moe["grasps_obb"])
        print(f"[smoke] result: diff={nd}  obb={no}  skipped_obb={moe['skipped_obb']}")
        if nd > 0:
            sd = moe["scores_diff"]
            print(f"[smoke]   diff scores: min={sd.min():.3f} med={np.median(sd):.3f} max={sd.max():.3f}")
        if no > 0:
            so = moe["scores_obb"]
            print(f"[smoke]    obb scores: min={so.min():.3f} med={np.median(so):.3f} max={so.max():.3f}")
        if moe["obb"] is not None:
            he = moe["obb"]["half_extent"]
            print(f"[smoke]   OBB extents (full m): {(2.0 * he).round(3).tolist()}")

        ok = (nd + no) > 0 or moe["skipped_obb"]
        # Sanity-check: scores in [0, 1].
        for tag, sc in [("diff", moe["scores_diff"]), ("obb", moe["scores_obb"])]:
            if len(sc) > 0 and ((sc.min() < 0.0) or (sc.max() > 1.0001)):
                print(f"[smoke] FAIL: {tag} scores out of [0,1]: {sc.min()}..{sc.max()}",
                      file=sys.stderr)
                ok = False
        if ok:
            n_pass += 1
            print(f"[smoke] PASS")
        else:
            n_fail += 1
            print(f"[smoke] FAIL")

    print(f"\n[smoke] DONE: pass={n_pass} fail={n_fail}")
    sys.exit(0 if n_fail == 0 else 1)


if __name__ == "__main__":
    main()
