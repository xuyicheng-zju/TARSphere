# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Precision-parity tests: fp32 vs fp16 (vs fp8 where the GPU supports it).

Reference is always fp32 **eager** PyTorch. We check that the TensorRT engines
(fp32 and fp16) reproduce it within precision-appropriate tolerances at both the
module level and end-to-end (grasp ranking + poses, using identical seeded
diffusion noise so the two trajectories are comparable).

fp8 requires Ada (sm89) / Hopper (sm90); on older GPUs that test skips. Skipped
entirely unless CUDA and torch_tensorrt are available.
"""

import sys

import numpy as np
import pytest
import torch
import torch.nn as nn

sys.path.insert(0, "tests")
from graspgenx.models.generator import DiffusionNoisePredictionNet
from graspgenx.models.grasp_gen import GraspGen
from graspgenx.models.tensorrt_utils import (
    accelerate_sampler,
    compile_diffusion_head,
    compile_mlp,
    tensorrt_available,
)
from test_inference_installation import (  # noqa: E402
    _make_discriminator_cfg,
    _make_generator_cfg,
    _prepare_batch,
)

pytestmark = pytest.mark.tensorrt

OBS_DIM, SAMPLE_DIM, STEP_DIM = 1024, 6, 512

_requires_trt = pytest.mark.skipif(
    not (torch.cuda.is_available() and tensorrt_available()),
    reason="requires CUDA and torch_tensorrt",
)


def _cos(a, b):
    return torch.nn.functional.cosine_similarity(a.flatten(), b.flatten(), dim=0).item()


def _build_head():
    torch.manual_seed(0)
    return DiffusionNoisePredictionNet(
        STEP_DIM, OBS_DIM, STEP_DIM, SAMPLE_DIM, attention="cat_attn", pose_repr="mlp"
    ).cuda().eval()


# ── module-level parity ──────────────────────────────────────────────────────
@_requires_trt
@pytest.mark.parametrize(
    "precision,atol,cos_min", [("fp32", 1e-2, 0.9999), ("fp16", 5e-2, 0.999)]
)
def test_diffusion_head_precision_parity(precision, atol, cos_min, capsys):
    head = _build_head()
    w = compile_diffusion_head(
        head, obs_dim=OBS_DIM, sample_dim=SAMPLE_DIM, device=torch.device("cuda"),
        opt_batch=500, max_batch=4000, precision=precision, engine_cache_dir=None,
    )
    assert w.is_accelerated
    obs = torch.randn(500, OBS_DIM, device="cuda")
    ts = torch.full((500,), 5.0, device="cuda")
    s = torch.randn(500, SAMPLE_DIM, device="cuda")
    with torch.inference_mode():
        ref, out = head(obs, ts, s), w(obs, ts, s)
    maxd, c = (ref - out).abs().max().item(), _cos(ref, out)
    with capsys.disabled():
        print(f"\n[head {precision}] max_abs={maxd:.3e} cosine={c:.6f}")
    assert maxd < atol and c > cos_min


@_requires_trt
@pytest.mark.parametrize("precision,atol", [("fp32", 1e-2), ("fp16", 5e-2)])
def test_mlp_head_precision_parity(precision, atol):
    torch.manual_seed(0)
    mlp = nn.Sequential(
        nn.Linear(1536, 768), nn.ReLU(), nn.Linear(768, 384), nn.ReLU(), nn.Linear(384, 1)
    ).cuda().eval()
    w = compile_mlp(
        mlp, in_dim=1536, device=torch.device("cuda"), opt_batch=500, max_batch=4000,
        precision=precision, engine_cache_dir=None,
    )
    assert w.is_accelerated
    x = torch.randn(500, 1536, device="cuda")
    with torch.inference_mode():
        ref, out = mlp(x), w(x)
    assert (ref - out).abs().max().item() < atol


# ── end-to-end grasp parity (fp16 vs fp32 eager, identical seeded noise) ──────
@_requires_trt
def test_end_to_end_grasp_parity_fp16(capsys):
    from scipy.stats import spearmanr

    dev = torch.device("cuda")
    num_grasps, seed = 100, 123
    model = GraspGen.from_config(
        _make_generator_cfg("ptv3vanilla"), _make_discriminator_cfg("ptv3vanilla")
    ).to(dev).eval()
    model.grasp_generator.num_grasps_per_object = num_grasps
    pc = torch.randn(2000, 3) * 0.05
    pc -= pc.mean(0)

    def run():
        torch.manual_seed(seed)
        np.random.seed(seed)
        with torch.inference_mode():
            out, _, _ = model.infer(_prepare_batch(pc, dev))
        return out["grasps_pred"][0].float(), out["grasp_confidence"][0][:, 0].float()

    g0, c0 = run()  # fp32 eager reference
    accelerate_sampler(
        type("S", (), {"model": model})(), precision="fp16",
        opt_batch=num_grasps, max_batch=num_grasps * 2, engine_cache_dir=None,
    )
    g1, c1 = run()  # fp16 TRT (same seed -> grasp i corresponds to grasp i)

    tdiff = (g0[:, :3, 3] - g1[:, :3, 3]).norm(dim=-1)
    Rrel = torch.matmul(g0[:, :3, :3].transpose(-1, -2), g1[:, :3, :3])
    ang = (torch.acos(((Rrel.diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2).clamp(-1, 1))
           * 180 / np.pi)
    sp = spearmanr(c0.cpu().numpy(), c1.cpu().numpy()).statistic
    k = 20
    j = len(set(torch.topk(c0, k).indices.tolist()) & set(torch.topk(c1, k).indices.tolist()))
    jac = j / (2 * k - j)
    with capsys.disabled():
        print(f"\n[e2e fp16] conf_maxd={(c0 - c1).abs().max():.4f} spearman={sp:.4f} "
              f"top{k}Jaccard={jac:.3f} trans_med={tdiff.median():.5f}m "
              f"rot_med={ang.median():.4f}deg")
    # grasp ranking + poses must be preserved under fp16
    assert sp > 0.95
    assert jac >= 0.7
    assert tdiff.median().item() < 0.01      # < 1 cm
    assert ang.median().item() < 5.0         # < 5 deg


# ── fp8 (hardware-gated) ─────────────────────────────────────────────────────
@_requires_trt
@pytest.mark.skipif(
    torch.cuda.is_available() and torch.cuda.get_device_capability(0) < (8, 9),
    reason="FP8 requires Ada (sm89) / Hopper (sm90); not available on this GPU",
)
def test_fp8_head_parity(capsys):  # pragma: no cover - no fp8 hardware in CI
    # On supported hardware, fp8 needs explicit quantization (modelopt). This
    # placeholder documents the path; it never runs on sm < 89.
    pytest.skip("fp8 path requires modelopt explicit-quantization wiring (TODO).")
