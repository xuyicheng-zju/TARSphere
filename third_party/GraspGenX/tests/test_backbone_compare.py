#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PTv3-Vanilla training smoke test.

Trains a GraspGenDiscriminator with the ptv3vanilla backbone on synthetic
data for a fixed wall-clock budget, optionally comparing the small and
standard ptv3vanilla configs for loss convergence, throughput, memory, and
parameter count.

Usage:
    python tests/test_backbone_compare.py [--max-seconds 60] [--batch-size 4]
    python tests/test_backbone_compare.py --backbones ptv3vanilla_small ptv3vanilla

Output:
    - TensorBoard logs under /tmp/backbone_compare/{backbone_name}/
    - Summary table printed to stdout
"""

import argparse
import gc
import os
import sys
import time

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter

# Ensure repo root is on path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from graspgenx.models.discriminator import GraspGenDiscriminator


# ---------------------------------------------------------------------------
# Reduced PTv3-Vanilla config (~4.3M params, comparable to PointNet++ ~4M)
# ---------------------------------------------------------------------------
PTV3_VANILLA_SMALL = dict(
    enc_depths=(1, 1, 1, 2, 1),
    enc_channels=(16, 32, 64, 128, 256),
    enc_num_head=(2, 4, 8, 8, 16),
    enc_patch_size=(128, 128, 128, 128, 128),
    drop_path=0.1,
)


# ---------------------------------------------------------------------------
# Synthetic dataset
# ---------------------------------------------------------------------------

class SyntheticGraspDataset(Dataset):
    """Generates random point clouds, grasps, and binary labels."""

    def __init__(self, num_objects=20, grasps_per_object=50,
                 num_points=1024, seed=42):
        super().__init__()
        rng = torch.Generator().manual_seed(seed)
        self.num_points = num_points
        self.grasps_per_object = grasps_per_object

        self.point_clouds = []
        self.grasps = []
        self.labels = []
        self.z_offsets = []

        for _ in range(num_objects):
            # Random point cloud in unit cube
            pc = torch.rand(num_points, 3, generator=rng) - 0.5
            self.point_clouds.append(pc)

            # Random SE(3) grasps: rotation (orthogonal via QR) + translation
            grasps_obj = []
            for _ in range(grasps_per_object):
                M = torch.randn(3, 3, generator=rng)
                Q, R = torch.linalg.qr(M)
                Q = Q * torch.det(Q).sign()
                T = torch.eye(4)
                T[:3, :3] = Q
                T[:3, 3] = torch.rand(3, generator=rng) * 0.2 - 0.1
                grasps_obj.append(T)
            self.grasps.append(torch.stack(grasps_obj))

            # Binary labels (50% positive)
            labels = (torch.rand(grasps_per_object, 1, generator=rng) > 0.5).float()
            self.labels.append(labels)

            # z_offset: single scalar per object
            self.z_offsets.append(torch.rand(1, generator=rng) * 0.1)

    def __len__(self):
        return len(self.point_clouds)

    def __getitem__(self, idx):
        return {
            "points": self.point_clouds[idx],
            "grasps": self.grasps[idx],
            "labels": self.labels[idx],
            "z_offset": self.z_offsets[idx],
        }


def collate_fn(batch):
    """Collate into the format GraspGenDiscriminator.forward expects."""
    return {
        "points": torch.stack([b["points"] for b in batch]),
        "grasps": [b["grasps"] for b in batch],
        "labels": [b["labels"] for b in batch],
        "z_offset": torch.stack([b["z_offset"] for b in batch]),
    }


# ---------------------------------------------------------------------------
# Model builder — handles backbone-specific constructor args
# ---------------------------------------------------------------------------

def build_model(backbone_name, device):
    """Instantiate GraspGenDiscriminator with the right config per backbone."""
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

    if backbone_name == "ptv3vanilla_small":
        # Build with reduced config directly
        from graspgenx.models.ptv3.ptv3_vanilla import PointTransformerV3Vanilla
        model = GraspGenDiscriminator(object_backbone="ptv3vanilla", **common)
        # Replace the default full-size encoder with the small one
        model.object_encoder = PointTransformerV3Vanilla(
            in_channels=3,
            output_dim=common["num_object_dim"],
            grid_size=common["grid_size"],
            **PTV3_VANILLA_SMALL,
        )
    else:
        model = GraspGenDiscriminator(object_backbone=backbone_name, **common)

    return model.to(device)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_backbone(backbone_name, max_seconds, batch_size, device, log_dir):
    """Train a discriminator and return metrics."""
    display_name = backbone_name
    print(f"\n{'='*60}")
    print(f"  Training with backbone: {display_name}")
    print(f"{'='*60}")

    model = build_model(backbone_name, device)

    num_params = sum(p.numel() for p in model.parameters())
    obj_params = sum(p.numel() for p in model.object_encoder.parameters())
    print(f"  Total params:    {num_params:>12,}")
    print(f"  Encoder params:  {obj_params:>12,}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.05)
    dataset = SyntheticGraspDataset(num_objects=20, grasps_per_object=50, num_points=1024)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                        collate_fn=collate_fn, num_workers=0, drop_last=True)

    writer = SummaryWriter(log_dir=log_dir)

    losses_history = []
    step_times = []
    global_step = 0
    epoch = 0

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
            step_times.append(step_elapsed)
            loss_val = loss.item()
            losses_history.append(loss_val)

            global_step += 1
            writer.add_scalar("train/loss", loss_val, global_step)
            writer.add_scalar("train/step_time", step_elapsed, global_step)
            if "ap" in stats:
                writer.add_scalar("train/ap", stats["ap"].item(), global_step)

            if global_step % 20 == 0:
                elapsed = time.time() - wall_start
                print(f"  [step {global_step:4d}] loss={loss_val:.4f}  "
                      f"step_time={step_elapsed:.3f}s  elapsed={elapsed:.0f}s")

    wall_time = time.time() - wall_start
    peak_mem = torch.cuda.max_memory_allocated(device) / (1024 ** 2)

    writer.flush()
    writer.close()

    # Cleanup to free GPU memory before next backbone
    del model, optimizer
    gc.collect()
    torch.cuda.empty_cache()

    metrics = {
        "backbone": display_name,
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
    parser = argparse.ArgumentParser(description="Backbone comparison test")
    parser.add_argument("--max-seconds", type=int, default=60,
                        help="Max training time per backbone (seconds)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (number of objects)")
    parser.add_argument("--log-dir", type=str, default="/tmp/backbone_compare",
                        help="TensorBoard log directory")
    parser.add_argument("--backbones", nargs="+",
                        default=["ptv3vanilla_small", "ptv3vanilla"],
                        help="Backbones to compare (ptv3vanilla_small or ptv3vanilla)")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        print("WARNING: Running on CPU — will be very slow")

    all_metrics = []
    for bb in args.backbones:
        bb_log_dir = os.path.join(args.log_dir, bb)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        metrics = train_backbone(bb, args.max_seconds, args.batch_size, device, bb_log_dir)
        all_metrics.append(metrics)

    # Print summary table
    col_w = 18
    names = [m["backbone"] for m in all_metrics]
    header = f"{'Metric':<30}" + "".join(f"{n:>{col_w}}" for n in names)
    sep = f"{'-'*30}" + "".join(f" {'-'*(col_w-1)}" for _ in names)

    print(f"\n{'='*(30 + col_w * len(names))}")
    print("  COMPARISON SUMMARY")
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
