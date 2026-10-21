# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Latency comparison for the GraspGenX diffusion denoiser:
eager PyTorch vs ``torch.compile`` vs TensorRT.

The denoiser is the module that dominates the inference hot loop (called
``num_diffusion_iters_eval`` times per ``sample()``), so its per-call latency is
what matters. Run explicitly to see the numbers:

    /path/to/trt-venv/bin/python -m pytest tests/test_tensorrt_benchmark.py -s -m tensorrt

Skipped unless CUDA and torch_tensorrt are both available.
"""

import pytest
import torch

from graspgenx.models.generator import DiffusionNoisePredictionNet
from graspgenx.models.tensorrt_utils import compile_diffusion_head, tensorrt_available

pytestmark = pytest.mark.tensorrt

OBS_DIM = 1024
SAMPLE_DIM = 6
STEP_EMBED_DIM = 512
BATCH = 500  # matches the released config's num_grasps_per_object
DIFFUSION_ITERS = 20  # num_diffusion_iters_eval in the released config

_requires_trt = pytest.mark.skipif(
    not (torch.cuda.is_available() and tensorrt_available()),
    reason="requires CUDA and torch_tensorrt (install: uv sync --extra tensorrt)",
)


def _build_head():
    torch.manual_seed(0)
    return DiffusionNoisePredictionNet(
        diffusion_step_embed_dim=STEP_EMBED_DIM,
        observation_embed_dim=OBS_DIM,
        sample_embed_dim=STEP_EMBED_DIM,
        sample_dim=SAMPLE_DIM,
        attention="cat_attn",
        pose_repr="mlp",
    ).cuda().eval()


def _inputs(batch=BATCH):
    obs = torch.randn(batch, OBS_DIM, device="cuda")
    ts = torch.full((batch,), 5.0, device="cuda")
    sample = torch.randn(batch, SAMPLE_DIM, device="cuda")
    return obs, ts, sample


def _bench(fn, inputs, iters=100, warmup=20):
    """Return mean per-call latency in milliseconds (CUDA-event timed)."""
    with torch.inference_mode():
        for _ in range(warmup):
            fn(*inputs)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            fn(*inputs)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


@_requires_trt
def test_benchmark_eager_vs_compile_vs_tensorrt(capsys):
    head = _build_head()
    inputs = _inputs()

    # 1) eager
    eager_ms = _bench(lambda o, t, s: head(o, t, s), inputs)
    with torch.inference_mode():
        eager_out = head(*inputs)

    # 2) torch.compile (inductor)
    compiled = torch.compile(head)
    compile_ms = _bench(lambda o, t, s: compiled(o, t, s), inputs)
    with torch.inference_mode():
        compile_out = compiled(*inputs)

    # 3) tensorrt
    trt = compile_diffusion_head(
        head,
        obs_dim=OBS_DIM,
        sample_dim=SAMPLE_DIM,
        device=torch.device("cuda"),
        min_batch=1,
        opt_batch=BATCH,
        max_batch=2000,
    )
    assert trt.is_accelerated
    trt_ms = _bench(lambda o, t, s: trt(o, t, s), inputs)
    with torch.inference_mode():
        trt_out = trt(*inputs)

    # all three must agree numerically
    assert torch.allclose(compile_out, eager_out, atol=1e-2, rtol=1e-2)
    assert torch.allclose(trt_out, eager_out, atol=1e-2, rtol=1e-2)

    with capsys.disabled():
        print(
            f"\n=== Diffusion denoiser latency (batch={BATCH}, single forward) ===\n"
            f"  1) eager PyTorch : {eager_ms:8.3f} ms   (1.00x)\n"
            f"  2) torch.compile : {compile_ms:8.3f} ms   ({eager_ms/compile_ms:.2f}x)\n"
            f"  3) TensorRT      : {trt_ms:8.3f} ms   ({eager_ms/trt_ms:.2f}x)\n"
            f"--- Extrapolated over {DIFFUSION_ITERS} diffusion steps "
            f"(per sample() call) ---\n"
            f"  1) eager         : {eager_ms*DIFFUSION_ITERS:8.3f} ms\n"
            f"  2) torch.compile : {compile_ms*DIFFUSION_ITERS:8.3f} ms\n"
            f"  3) TensorRT      : {trt_ms*DIFFUSION_ITERS:8.3f} ms\n"
        )

    # TensorRT should not be materially slower than eager.
    assert trt_ms <= eager_ms * 1.5


class _SamplerShim:
    def __init__(self, model):
        self.model = model


@_requires_trt
@pytest.mark.parametrize("num_grasps", [200, 500])
def test_benchmark_end_to_end(num_grasps, capsys):
    """Full-pipeline latency: eager vs TensorRT-accelerated heads, using the
    RELEASED generator head config (cat_attn, 20 diffusion steps)."""
    import sys

    sys.path.insert(0, "tests")
    from graspgenx.models.grasp_gen import GraspGen
    from graspgenx.models.tensorrt_utils import accelerate_sampler
    from test_inference_installation import (
        _make_generator_cfg,
        _make_discriminator_cfg,
        _prepare_batch,
    )

    device = torch.device("cuda")
    torch.manual_seed(0)
    pc = torch.randn(3500, 3) * 0.05
    pc -= pc.mean(0)

    def build():
        g = _make_generator_cfg("ptv3vanilla")
        g["attention"] = "cat_attn"
        g["num_diffusion_iters"] = 20
        g["num_diffusion_iters_eval"] = 20
        m = (
            GraspGen.from_config(g, _make_discriminator_cfg("ptv3vanilla"))
            .to(device)
            .eval()
        )
        m.grasp_generator.num_grasps_per_object = num_grasps
        return m

    def bench(m, iters=20, warmup=5):
        with torch.inference_mode():
            for _ in range(warmup):
                m.infer(_prepare_batch(pc, device))
            torch.cuda.synchronize()
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                m.infer(_prepare_batch(pc, device))
            e.record()
            torch.cuda.synchronize()
        return s.elapsed_time(e) / iters

    eager = bench(build())
    m2 = build()
    accelerate_sampler(_SamplerShim(m2), opt_batch=num_grasps, max_batch=num_grasps * 2)
    trt = bench(m2)

    with capsys.disabled():
        print(
            f"\n=== Full inference (released head: cat_attn, 20 steps, "
            f"num_grasps={num_grasps}) ===\n"
            f"  eager        : {eager:8.2f} ms   (1.00x)\n"
            f"  TensorRT     : {trt:8.2f} ms   ({eager/trt:.2f}x)\n"
        )
    assert trt <= eager * 1.1  # never materially slower
