# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""End-to-end inference test with the TensorRT-accelerated diffusion head.

This is the TensorRT counterpart of ``test_inference_installation.py``: it
builds the same random-weight GraspGenX model, swaps the diffusion denoiser for
a TensorRT-compiled engine, and verifies the full inference pipeline still
produces the expected 100 valid grasps. The point-cloud backbone (ptv3vanilla)
keeps running in eager PyTorch — only the diffusion head is accelerated — so
this exercises the mixed eager+TRT path the sampler actually uses.

Skipped unless CUDA and torch_tensorrt are both available.
"""

import pytest
import torch

from graspgenx.models.grasp_gen import GraspGen
from graspgenx.models.tensorrt_utils import (
    accelerate_sampler,
    compile_diffusion_head,
    tensorrt_available,
)

# Reuse the exact config/batch helpers from the installation test so the two
# stay in lock-step.
from test_inference_installation import (
    _make_generator_cfg,
    _make_discriminator_cfg,
    _prepare_batch,
    random_point_cloud,  # noqa: F401  (pytest fixture)
)

pytestmark = pytest.mark.tensorrt

_requires_trt = pytest.mark.skipif(
    not (torch.cuda.is_available() and tensorrt_available()),
    reason="requires CUDA and torch_tensorrt (install: uv sync --extra tensorrt)",
)


def _accelerate_diffusion_head(model, num_grasps):
    """Swap the generator's diffusion head for a TensorRT engine in place."""
    gen = model.grasp_generator
    obs_dim = gen.num_object_dim + gen.num_gripper_dim
    wrapped = compile_diffusion_head(
        gen.diffusion_head,
        obs_dim=obs_dim,
        sample_dim=gen.output_dim,
        device=torch.device("cuda"),
        min_batch=1,
        opt_batch=num_grasps,
        max_batch=max(num_grasps * 2, 200),
    )
    gen.diffusion_head = wrapped
    return wrapped


@_requires_trt
@pytest.mark.parametrize("backbone", ["ptv3vanilla"])
def test_inference_100_grasps_tensorrt(backbone, random_point_cloud):
    """Full inference with a TensorRT diffusion head yields 100 valid grasps."""
    device = torch.device("cuda")
    num_grasps = 100

    gen_cfg = _make_generator_cfg(backbone)
    disc_cfg = _make_discriminator_cfg(backbone)
    model = GraspGen.from_config(gen_cfg, disc_cfg).to(device).eval()
    model.grasp_generator.num_grasps_per_object = num_grasps

    wrapped = _accelerate_diffusion_head(model, num_grasps)
    assert wrapped.is_accelerated, "Expected the diffusion head to compile to TensorRT"

    data_batch = _prepare_batch(random_point_cloud, device)
    with torch.inference_mode():
        outputs, _, _ = model.infer(data_batch)

    assert "grasps_pred" in outputs
    grasps = outputs["grasps_pred"][0]
    assert grasps.shape == torch.Size([num_grasps, 4, 4])
    assert torch.isfinite(grasps).all(), "TensorRT inference produced non-finite grasps"
    # Homogeneous bottom row preserved.
    bottom = grasps[:, 3, :]
    expected = torch.tensor([0.0, 0.0, 0.0, 1.0], device=device)
    assert torch.allclose(bottom, expected.expand_as(bottom), atol=1e-4)


class _SamplerShim:
    """Minimal stand-in exposing the .model attribute accelerate_sampler needs."""

    def __init__(self, model):
        self.model = model


@_requires_trt
@pytest.mark.parametrize("backbone", ["ptv3vanilla"])
def test_accelerate_sampler_converts_generator_and_discriminator(
    backbone, random_point_cloud
):
    """accelerate_sampler should TRT-compile the diffusion head AND the two
    discriminator MLP heads, and inference should still yield 100 valid grasps."""
    device = torch.device("cuda")
    num_grasps = 100

    model = GraspGen.from_config(
        _make_generator_cfg(backbone), _make_discriminator_cfg(backbone)
    ).to(device).eval()
    model.grasp_generator.num_grasps_per_object = num_grasps

    accelerated = accelerate_sampler(
        _SamplerShim(model),
        opt_batch=num_grasps,
        max_batch=num_grasps * 2,
        engine_cache_dir=None,  # keep the test hermetic (no disk cache)
    )
    assert accelerated, "Expected at least one module to be TensorRT-accelerated"
    # The diffusion head and both discriminator heads should each be wrapped.
    from graspgenx.models.tensorrt_utils import TensorRTDiffusionHead, TensorRTModule

    assert isinstance(model.grasp_generator.diffusion_head, TensorRTDiffusionHead)
    assert model.grasp_generator.diffusion_head.is_accelerated
    assert isinstance(model.grasp_discriminator.prediction_head, TensorRTModule)
    assert model.grasp_discriminator.prediction_head.is_accelerated
    assert isinstance(model.grasp_discriminator.sample_encoder, TensorRTModule)
    assert model.grasp_discriminator.sample_encoder.is_accelerated

    data_batch = _prepare_batch(random_point_cloud, device)
    with torch.inference_mode():
        outputs, _, _ = model.infer(data_batch)
    grasps = outputs["grasps_pred"][0]
    assert grasps.shape == torch.Size([num_grasps, 4, 4])
    assert torch.isfinite(grasps).all()
