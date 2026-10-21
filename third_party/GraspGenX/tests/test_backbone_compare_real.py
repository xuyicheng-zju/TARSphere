#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Backbone comparison on realistic grasp data.

Uses a real object mesh (bowl) + real sampled grasps from the pilot dataset,
expanded to 2000 grasps via small SE(3) perturbations.

Usage:
    python tests/test_backbone_compare_real.py [--max-seconds 60] [--batch-size 4]

Backbones tested:
    - ptv3vanilla_small
    - ptv3vanilla_small_compiled  (torch.compile)
"""

import argparse
import gc
import os
import sys
import time

import h5py
import numpy as np
import torch
import torch.nn.functional as F
import trimesh
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graspgenx.models.discriminator import GraspGenDiscriminator


# ---------------------------------------------------------------------------
# Reduced PTv3-Vanilla config (~4.3M encoder params)
# ---------------------------------------------------------------------------
PTV3_VANILLA_SMALL = dict(
    enc_depths=(1, 1, 1, 2, 1),
    enc_channels=(16, 32, 64, 128, 256),
    enc_num_head=(2, 4, 8, 8, 16),
    enc_patch_size=(128, 128, 128, 128, 128),
    drop_path=0.1,
)


# ---------------------------------------------------------------------------
# Dataset from real pilot data
# ---------------------------------------------------------------------------

def load_pilot_grasps(h5_path, object_name, mesh_path, num_grasps=2000,
                      num_points=1024, seed=42):
    """Load real grasps and expand via perturbations."""
    rng = np.random.RandomState(seed)

    # Load mesh and sample point cloud
    mesh = trimesh.load(mesh_path)
    with h5py.File(h5_path, "r") as f:
        obj = f[f"objects/{object_name}"]
        scale = float(obj["asset_scale"][()])
        base_grasps = obj["pred_grasps"][:]  # (N_base, 4, 4)

    # Sample point cloud from mesh surface
    points = mesh.sample(num_points) * scale

    # Expand grasps via small perturbations
    n_base = len(base_grasps)
    all_grasps = [base_grasps]
    while sum(len(g) for g in all_grasps) < num_grasps:
        # Pick random base grasps to perturb
        n_needed = num_grasps - sum(len(g) for g in all_grasps)
        indices = rng.randint(0, n_base, size=min(n_needed, n_base))
        perturbed = base_grasps[indices].copy()

        # Small translation perturbation (1-5mm)
        perturbed[:, :3, 3] += rng.randn(len(perturbed), 3) * 0.003

        # Small rotation perturbation (axis-angle, ~5 degrees)
        for i in range(len(perturbed)):
            axis = rng.randn(3)
            axis /= np.linalg.norm(axis) + 1e-8
            angle = rng.randn() * 0.087  # ~5 degrees std
            K = np.array([
                [0, -axis[2], axis[1]],
                [axis[2], 0, -axis[0]],
                [-axis[1], axis[0], 0],
            ])
            R_perturb = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
            perturbed[i, :3, :3] = R_perturb @ perturbed[i, :3, :3]

        all_grasps.append(perturbed)

    grasps = np.concatenate(all_grasps, axis=0)[:num_grasps]

    # Generate labels: base grasps are "positive", heavily perturbed are "negative"
    # Use distance from nearest base grasp as proxy
    labels = np.zeros(num_grasps, dtype=np.float32)
    labels[:n_base] = 1.0  # Original grasps are positive
    # For perturbed: closer to original = more likely positive
    for i in range(n_base, num_grasps):
        # Distance to nearest base grasp (translation only)
        dists = np.linalg.norm(
            grasps[i, :3, 3] - base_grasps[:, :3, 3], axis=1
        )
        min_dist = dists.min()
        # Sigmoid-like: close = positive, far = negative
        labels[i] = 1.0 / (1.0 + np.exp(min_dist / 0.002 - 2.0))

    # Binarize with threshold
    labels = (labels > 0.5).astype(np.float32)

    return (
        torch.from_numpy(points).float(),
        torch.from_numpy(grasps).float(),
        torch.from_numpy(labels).float(),
    )


class RealGraspDataset(Dataset):
    """Dataset built from a single real object with expanded grasps."""

    def __init__(self, points, grasps, labels, grasps_per_sample=50, seed=42):
        super().__init__()
        self.points = points          # (num_points, 3)
        self.grasps = grasps          # (num_grasps, 4, 4)
        self.labels = labels          # (num_grasps,)
        self.grasps_per_sample = grasps_per_sample
        self.num_grasps = len(grasps)
        self.rng = torch.Generator().manual_seed(seed)

    def __len__(self):
        # One "sample" per chunk of grasps
        return self.num_grasps // self.grasps_per_sample

    def __getitem__(self, idx):
        start = idx * self.grasps_per_sample
        end = start + self.grasps_per_sample
        return {
            "points": self.points,
            "grasps": self.grasps[start:end],
            "labels": self.labels[start:end].unsqueeze(-1),
            "z_offset": torch.tensor([0.05]),  # Fixed z-offset for single gripper
        }


def collate_fn(batch):
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "grasps": [b["grasps"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "z_offset": torch.stack([b["z_offset"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------

def build_model(backbone_name, device):
    common = dict(
        num_object_dim=512,
        num_gripper_dim=512,
        gripper_backbone="z_offset",
        grasp_repr="r3_6d",
        grid_size=0.01,
        sample_embed_dim=256,
        pose_repr="mlp",
        topk_ratio=0.75,
        kappa=1.0,
    )

    # Strip _compiled suffix for model construction
    base_name = backbone_name.replace("_compiled", "")

    if base_name == "ptv3vanilla_small":
        from graspgenx.models.ptv3.ptv3_vanilla import PointTransformerV3Vanilla
        model = GraspGenDiscriminator(object_backbone="ptv3vanilla", **common)
        model.object_encoder = PointTransformerV3Vanilla(
            in_channels=3,
            output_dim=common["num_object_dim"],
            grid_size=common["grid_size"],
            **PTV3_VANILLA_SMALL,
        )
    else:
        model = GraspGenDiscriminator(object_backbone=base_name, **common)

    model = model.to(device)

    if backbone_name.endswith("_compiled"):
        print("  Compiling object_encoder with torch.compile()...")
        model.object_encoder = torch.compile(model.object_encoder)

    return model


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_backbone(backbone_name, dataset, max_seconds, batch_size, device, log_dir):
    print(f"\n{'='*60}")
    print(f"  Training with backbone: {backbone_name}")
    print(f"{'='*60}")

    model = build_model(backbone_name, device)

    num_params = sum(p.numel() for p in model.parameters())
    obj_params = sum(p.numel() for p in model.object_encoder.parameters())
    print(f"  Total params:    {num_params:>12,}")
    print(f"  Encoder params:  {obj_params:>12,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)

    writer = SummaryWriter(log_dir=log_dir)

    losses_history = []
    ap_history = []
    step_times = []
    global_step = 0
    epoch = 0
    compile_warmup_done = False

    torch.cuda.reset_peak_memory_stats(device)
    wall_start = time.time()

    while time.time() - wall_start < max_seconds:
        epoch += 1
        model.train()

        for batch_data in loader:
            if time.time() - wall_start >= max_seconds:
                break

            batch_data["points"] = batch_data["points"].to(device)
            batch_data["grasps"] = [g.to(device) for g in batch_data["grasps"]]
            batch_data["labels"] = [l.to(device) for l in batch_data["labels"]]
            batch_data["z_offset"] = batch_data["z_offset"].to(device)

            step_start = time.time()

            optimizer.zero_grad()
            outputs, losses, stats = model(batch_data)
            loss = sum(w * v for w, v in losses.values())
            loss.backward()
            optimizer.step()

            step_elapsed = time.time() - step_start

            # Skip first few steps for compiled models (compilation overhead)
            if backbone_name.endswith("_compiled") and not compile_warmup_done:
                if global_step < 3:
                    global_step += 1
                    if global_step == 3:
                        compile_warmup_done = True
                        print(f"  [compile warmup done after {time.time()-wall_start:.1f}s]")
                        torch.cuda.reset_peak_memory_stats(device)
                    continue

            step_times.append(step_elapsed)
            loss_val = loss.item()
            losses_history.append(loss_val)
            if "ap" in stats:
                ap_history.append(stats["ap"].item())

            global_step += 1
            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/step_time", step_elapsed, global_step)
            if "ap" in stats:
                writer.add_scalar("train/ap", stats["ap"].item(), global_step)

            if global_step % 20 == 0:
                elapsed = time.time() - wall_start
                ap_str = f"  ap={ap_history[-1]:.3f}" if ap_history else ""
                print(f"  [step {global_step:4d}] loss={loss_val:.4f}  "
                      f"step_time={step_elapsed:.3f}s{ap_str}  elapsed={elapsed:.0f}s")

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    writer.flush()
    writer.close()

    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {
        "backbone": backbone_name,
        "params": num_params,
        "obj_params": obj_params,
        "total_steps": global_step,
        "epochs": epoch,
        "wall_time_s": wall_time,
        "final_loss": losses_history[-1] if losses_history else float("nan"),
        "avg_loss_last_10": (
            sum(losses_history[-10:]) / min(10, len(losses_history))
            if losses_history else float("nan")
        ),
        "final_ap": ap_history[-1] if ap_history else float("nan"),
        "avg_ap_last_10": (
            sum(ap_history[-10:]) / min(10, len(ap_history))
            if ap_history else float("nan")
        ),
        "avg_step_time_ms": (
            sum(step_times) / len(step_times) * 1000
            if step_times else float("nan")
        ),
        "throughput_steps_per_sec": (
            len(step_times) / sum(step_times)
            if step_times else 0
        ),
        "peak_gpu_mem_mb": peak_mem,
    }
    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Backbone comparison on real grasp data")
    parser.add_argument("--max-seconds", type=int, default=60,
                        help="Max training time per backbone (seconds)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (number of objects per step)")
    parser.add_argument("--log-dir", type=str, default="/tmp/backbone_compare_real",
                        help="TensorBoard log directory")
    parser.add_argument("--backbones", nargs="+",
                        default=["ptv3vanilla_small",
                                 "ptv3vanilla_small_compiled"],
                        help="Backbones to compare")
    parser.add_argument("--num-grasps", type=int, default=2000,
                        help="Number of grasps to generate")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # ── Generate dataset from pilot data ──────────────────────────────
    repo_root = os.path.join(os.path.dirname(__file__), "..")
    h5_path = os.path.join(repo_root, "data_generation/pilot_data/eval_input/robotiq_2f_85/pilot_v2.h5")
    mesh_path = os.path.join(repo_root, "data_generation/pilot_data/objects/bowl_0020.stl")

    print(f"Loading pilot data: bowl_0020, gripper=robotiq_2f_85")
    print(f"Expanding to {args.num_grasps} grasps via perturbations...")

    points, grasps, labels = load_pilot_grasps(
        h5_path, "bowl_0020", mesh_path, num_grasps=args.num_grasps
    )
    pos_ratio = labels.mean().item()
    print(f"  Point cloud: {points.shape}")
    print(f"  Grasps: {grasps.shape}")
    print(f"  Labels: {labels.shape} ({pos_ratio:.1%} positive)")

    dataset = RealGraspDataset(points, grasps, labels, grasps_per_sample=50)
    print(f"  Dataset samples: {len(dataset)} (each = 1 object x 50 grasps)")

    # ── Run comparison ────────────────────────────────────────────────
    all_metrics = []
    for bb in args.backbones:
        bb_log_dir = os.path.join(args.log_dir, bb)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        metrics = train_backbone(bb, dataset, args.max_seconds, args.batch_size,
                                 device, bb_log_dir)
        all_metrics.append(metrics)

    # ── Summary table ─────────────────────────────────────────────────
    col_w = 20
    names = [m["backbone"] for m in all_metrics]
    header = f"{'Metric':<30}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = f"{'-'*30}" + "".join(f" {'-'*(col_w-1)}" for _ in names)

    print(f"\n{'='*(30 + col_w * len(names))}")
    print("  COMPARISON SUMMARY (real grasp data: bowl_0020 x robotiq_2f_85)")
    print(f"{'='*(30 + col_w * len(names))}")
    print(header)
    print(sep)

    rows = [
        ("Total params", "params", "{:,}"),
        ("Encoder params", "obj_params", "{:,}"),
        ("Total steps", "total_steps", "{}"),
        ("Epochs completed", "epochs", "{}"),
        ("Wall time (s)", "wall_time_s", "{:.1f}"),
        ("Final loss", "final_loss", "{:.4f}"),
        ("Avg loss (last 10)", "avg_loss_last_10", "{:.4f}"),
        ("Final AP", "final_ap", "{:.4f}"),
        ("Avg AP (last 10)", "avg_ap_last_10", "{:.4f}"),
        ("Avg step time (ms)", "avg_step_time_ms", "{:.1f}"),
        ("Throughput (steps/s)", "throughput_steps_per_sec", "{:.1f}"),
        ("Peak GPU memory (MB)", "peak_gpu_mem_mb", "{:.0f}"),
    ]

    for label, key, fmt in rows:
        vals = [fmt.format(m[key]) for m in all_metrics]
        line = f"{label:<30}" + "".join(f"{v:>{col_w}}" for v in vals)
        print(line)

    print(f"\nTensorBoard logs: {args.log_dir}")
    print(f"  tensorboard --logdir {args.log_dir} --port 6006")


if __name__ == "__main__":
    main()
