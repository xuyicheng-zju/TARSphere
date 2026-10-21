# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""GPU statistical-outlier-removal (OBB) parity tests.

The graspmoe OBB branch runs its kNN-based statistical outlier removal on GPU
(torch.cdist) when CUDA is available. This must produce the *same* OBB as the
scipy cKDTree path — verified here on random clouds. CUDA-gated; no TensorRT
needed.
"""

import numpy as np
import pytest
import torch

import graspgenx.samplers.graspmoe as gm

_requires_cuda = pytest.mark.skipif(
    not torch.cuda.is_available(), reason="GPU SOR requires CUDA"
)


def _random_cloud(seed, n=3500):
    rng = np.random.default_rng(seed)
    # a couple of blobs + a few outliers, like a segmented object
    pts = rng.normal(0, 0.04, (n, 3)).astype(np.float32)
    pts[: n // 10] += np.array([0.15, 0.0, 0.0], dtype=np.float32)
    pts[:5] += rng.normal(0, 0.5, (5, 3)).astype(np.float32)  # outliers
    return pts


@_requires_cuda
def test_sor_keep_mask_matches_scipy():
    """GPU keep-mask equals the scipy cKDTree keep-mask exactly."""
    from scipy.spatial import cKDTree

    pts = _random_cloud(0)
    k, std_ratio = 20, 2.0
    gpu_keep = gm._sor_keep_mask_torch(pts, k, std_ratio)
    tree = cKDTree(pts)
    d, _ = tree.query(pts, k=k + 1)
    mean_d = d[:, 1:].mean(axis=1)
    cpu_keep = mean_d < (mean_d.mean() + std_ratio * mean_d.std())
    assert np.array_equal(gpu_keep, cpu_keep)


@_requires_cuda
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_compute_obb_gpu_matches_cpu(seed):
    """_compute_obb produces an identical OBB with GPU vs CPU SOR."""
    pts = _random_cloud(seed)
    orig = gm._GPU_OBB
    try:
        np.random.seed(7)
        gm._GPU_OBB = False
        c0, h0, R0 = gm._compute_obb(pts)
        np.random.seed(7)
        gm._GPU_OBB = True
        c1, h1, R1 = gm._compute_obb(pts)
    finally:
        gm._GPU_OBB = orig
    assert np.allclose(c0, c1, atol=1e-6)
    assert np.allclose(h0, h1, atol=1e-6)
    assert np.allclose(R0, R1, atol=1e-6)
