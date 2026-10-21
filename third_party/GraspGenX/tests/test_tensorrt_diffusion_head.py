# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT conversion tests for the GraspGenX diffusion denoiser.

These verify that the TensorRT-compiled ``DiffusionNoisePredictionNet`` (the
per-timestep denoiser run inside the reverse-diffusion loop) produces outputs
numerically equivalent to the eager PyTorch module, across a range of (dynamic)
batch sizes, and that the wrapper falls back to eager safely outside the
compiled range.

The whole module is skipped unless CUDA *and* ``torch_tensorrt`` are both
available, so CI without the optional 'tensorrt' extra stays green. This mirrors
the skip-if-no-CUDA pattern in ``tests/test_inference_installation.py``.
"""

import pytest
import torch

from graspgenx.models.generator import DiffusionNoisePredictionNet
from graspgenx.models.tensorrt_utils import (
    TensorRTDiffusionHead,
    compile_diffusion_head,
    machine_key,
    tensorrt_available,
)

# Dims matching the released generator checkpoint config:
#   num_object_dim + num_gripper_dim = 512 + 512, grasp_repr=r3_so3 -> sample_dim 6
OBS_DIM = 1024
SAMPLE_DIM = 6
STEP_EMBED_DIM = 512

pytestmark = pytest.mark.tensorrt

_requires_trt = pytest.mark.skipif(
    not (torch.cuda.is_available() and tensorrt_available()),
    reason="TensorRT test requires CUDA and torch_tensorrt (install: uv sync --extra tensorrt)",
)


def _build_head(attention: str = "cat_attn") -> DiffusionNoisePredictionNet:
    """Build a denoiser matching the released config and put it in eval mode."""
    torch.manual_seed(0)
    head = DiffusionNoisePredictionNet(
        diffusion_step_embed_dim=STEP_EMBED_DIM,
        observation_embed_dim=OBS_DIM,
        sample_embed_dim=STEP_EMBED_DIM,
        sample_dim=SAMPLE_DIM,
        attention=attention,
        pose_repr="mlp",
    )
    return head.cuda().eval()


def _inputs(batch_size: int, timestep: int = 5):
    obs = torch.randn(batch_size, OBS_DIM, device="cuda")
    sample = torch.randn(batch_size, SAMPLE_DIM, device="cuda")
    timesteps = torch.tensor(timestep, device="cuda")  # scalar, as in the loop
    return obs, timesteps, sample


def test_tensorrt_available_returns_bool():
    """The availability probe must never raise and always return a bool."""
    assert isinstance(tensorrt_available(), bool)


@_requires_trt
def test_compiled_head_matches_eager():
    """TRT output must match eager output at the optimization batch size."""
    head = _build_head()
    wrapped = compile_diffusion_head(
        head,
        obs_dim=OBS_DIM,
        sample_dim=SAMPLE_DIM,
        device=torch.device("cuda"),
        min_batch=1,
        opt_batch=100,
        max_batch=200,
    )
    assert isinstance(wrapped, TensorRTDiffusionHead)
    assert wrapped.is_accelerated, "Expected TensorRT compilation to succeed"

    obs, timesteps, sample = _inputs(100)
    with torch.inference_mode():
        eager_out = head(obs, timesteps, sample)
        trt_out = wrapped(obs, timesteps, sample)

    assert trt_out.shape == eager_out.shape == (100, SAMPLE_DIM)
    assert torch.allclose(trt_out, eager_out, atol=1e-2, rtol=1e-2), (
        f"max abs diff {(trt_out - eager_out).abs().max().item():.4e}"
    )


@_requires_trt
@pytest.mark.parametrize("batch_size", [1, 50, 100, 200])
def test_dynamic_batch_sizes(batch_size):
    """A single engine compiled with a dynamic batch dim serves many sizes."""
    head = _build_head()
    wrapped = compile_diffusion_head(
        head,
        obs_dim=OBS_DIM,
        sample_dim=SAMPLE_DIM,
        device=torch.device("cuda"),
        min_batch=1,
        opt_batch=100,
        max_batch=200,
    )
    assert wrapped.is_accelerated

    obs, timesteps, sample = _inputs(batch_size)
    with torch.inference_mode():
        eager_out = head(obs, timesteps, sample)
        trt_out = wrapped(obs, timesteps, sample)

    assert trt_out.shape == (batch_size, SAMPLE_DIM)
    assert torch.allclose(trt_out, eager_out, atol=1e-2, rtol=1e-2)


@_requires_trt
def test_fallback_outside_compiled_range():
    """Batches beyond max_batch fall back to eager and stay correct."""
    head = _build_head()
    wrapped = compile_diffusion_head(
        head,
        obs_dim=OBS_DIM,
        sample_dim=SAMPLE_DIM,
        device=torch.device("cuda"),
        min_batch=1,
        opt_batch=50,
        max_batch=100,
    )
    assert wrapped.is_accelerated

    big = 300  # outside [1, 100]
    obs, timesteps, sample = _inputs(big)
    with torch.inference_mode():
        eager_out = head(obs, timesteps, sample)
        out = wrapped(obs, timesteps, sample)  # -> eager fallback path

    assert out.shape == (big, SAMPLE_DIM)
    assert torch.allclose(out, eager_out, atol=1e-4)


@_requires_trt
def test_engine_cache_roundtrip(tmp_path):
    """A machine-keyed engine is written on first compile and loaded on the
    second call (no rebuild), producing identical output."""
    head = _build_head()
    kw = dict(
        obs_dim=OBS_DIM,
        sample_dim=SAMPLE_DIM,
        device=torch.device("cuda"),
        opt_batch=100,
        max_batch=200,
        engine_cache_dir=str(tmp_path),
    )
    w1 = compile_diffusion_head(head, **kw)
    assert w1.is_accelerated
    cached = list(tmp_path.glob("*.engine"))
    assert len(cached) == 1, "expected one cached engine file"
    assert machine_key() in cached[0].name, "filename must embed the machine key"

    w2 = compile_diffusion_head(head, **kw)  # should load from disk
    assert w2.is_accelerated

    obs, ts, sample = _inputs(100)
    with torch.inference_mode():
        assert torch.allclose(w1(obs, ts, sample), w2(obs, ts, sample), atol=1e-2)


@_requires_trt
def test_passthrough_wrapper_when_not_accelerated():
    """A non-accelerated wrapper still returns correct eager results."""
    head = _build_head()
    wrapped = TensorRTDiffusionHead(head, trt_module=None, min_batch=1, max_batch=100)
    assert not wrapped.is_accelerated

    obs, timesteps, sample = _inputs(10)
    with torch.inference_mode():
        eager_out = head(obs, timesteps, sample)
        out = wrapped(obs, timesteps, sample)
    assert torch.allclose(out, eager_out, atol=1e-6)
