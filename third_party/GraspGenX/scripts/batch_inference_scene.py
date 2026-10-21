# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Headless batch grasp inference over every object in a single scene.

Demonstrates GraspGenX *batch inference over both objects and grasps*: all N
segmented objects in a scene are folded into ONE batched diffusion forward pass
that generates ``num_grasps`` grasps per object (default 500). No collision
checking is performed. Results are summarized to stdout and optionally saved.

Example:
    uv run python scripts/batch_inference_scene.py \
        --sample_data_dir assets/sample_data/real_world \
        --scene 00 --gripper_name robotiq_2f_85 --num_grasps 500
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np

from graspgenx.grasp_server import GraspGenXSampler
from graspgenx.samplers import run_planner_on_batch
from graspgenx.utils.checkpoint_io import load_model_cfg
from graspgenx.utils.scene_loaders import collect_scene_items, load_realworld_scene
from demo_object_pc import _resolve_default_checkpoints


def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sample_data_dir", type=str, default="assets/sample_data/real_world")
    p.add_argument("--scene", type=str, default="00", help="Scene <NN> to process.")
    p.add_argument("--gripper_name", type=str, default="robotiq_2f_85")
    p.add_argument("--assets_dir", type=str, default=None)
    p.add_argument("--checkpoints", type=str, default=None)
    p.add_argument("--gen_pth", type=str, default=None)
    p.add_argument("--dis_pth", type=str, default=None)
    p.add_argument("--num_grasps", type=int, default=500, help="Grasps per object.")
    p.add_argument("--min_obj_points", type=int, default=100)
    p.add_argument(
        "--tensorrt",
        action="store_true",
        help="Accelerate the diffusion/discriminator heads with TensorRT.",
    )
    p.add_argument(
        "--tensorrt_precision",
        type=str,
        default="fp32",
        choices=["fp32", "fp16"],
        help="TensorRT precision when --tensorrt is set (default fp32).",
    )
    p.add_argument(
        "--output_file",
        type=str,
        default=None,
        help="Optional .npz to save per-object grasps + confidences.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    repo_root = Path(__file__).resolve().parent.parent
    if args.assets_dir is None:
        args.assets_dir = str(repo_root / "assets")

    checkpoint_root = args.checkpoints or _resolve_default_checkpoints()
    print(f"Using checkpoints under: {checkpoint_root}")
    model_cfg = load_model_cfg(
        os.path.join(checkpoint_root, "gen"),
        os.path.join(checkpoint_root, "dis"),
        args.gen_pth,
        args.dis_pth,
    )

    # Locate the requested scene.
    items = collect_scene_items(args.sample_data_dir, scene_filter=args.scene)
    if not items:
        raise FileNotFoundError(
            f"No scene matching --scene {args.scene} in {args.sample_data_dir}"
        )
    fmt_tag, path = items[0]
    print(f"[scene] {path} (format={fmt_tag})")

    grasp_sampler = GraspGenXSampler(
        model_cfg,
        args.gripper_name,
        assets_dir=args.assets_dir,
        use_tensorrt=args.tensorrt,
        tensorrt_precision=args.tensorrt_precision,
    )

    scene = load_realworld_scene(path, min_obj_points=args.min_obj_points)
    labels = list(scene["objects"].keys())
    obj_pcs = [scene["objects"][lab]["pc"] for lab in labels]
    print(
        f"[scene {args.scene}] {len(labels)} objects: {labels}\n"
        f"  points/object: {[len(pc) for pc in obj_pcs]}\n"
        f"  -> batched diffusion forward: {len(labels)} objects x "
        f"{args.num_grasps} grasps = {len(labels) * args.num_grasps} total samples"
    )

    # ONE batched diffusion forward pass over all objects (and all grasps).
    # planner="diffusion" => GraspGenXSampler.run_inference_batch, which collates
    # every object into a single (N, num_grasps, ...) forward. No collision check.
    t0 = time.time()
    batch_results = run_planner_on_batch(
        obj_pcs,
        grasp_sampler,
        planner="diffusion",
        grasp_threshold=-1.0,        # keep everything, rank by confidence
        num_grasps=args.num_grasps,
        topk_num_grasps=args.num_grasps,  # keep up to all num_grasps per object
    )
    dt = time.time() - t0

    print(f"\n=== Batch inference done in {dt:.2f}s "
          f"({dt / max(len(labels), 1):.3f}s/object amortized) ===")
    total = 0
    saved = {}
    for label, (grasps, conf, tags, _obb) in zip(labels, batch_results):
        n = len(grasps)
        total += n
        if n == 0:
            print(f"  [{label:8s}] 0 grasps")
            continue
        print(
            f"  [{label:8s}] {n:4d} grasps | conf {conf.min():.3f}..{conf.max():.3f} "
            f"(mean {conf.mean():.3f}) | top-pose t=({grasps[int(conf.argmax())][:3,3]})"
        )
        saved[f"{label}_grasps"] = grasps
        saved[f"{label}_conf"] = conf
    print(f"\nTotal grasps generated across scene: {total}")

    if args.output_file:
        np.savez_compressed(args.output_file, **saved)
        print(f"Saved per-object grasps to {args.output_file}")


if __name__ == "__main__":
    main()
