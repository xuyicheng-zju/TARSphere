# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""TensorRT acceleration for the GraspGenX diffusion denoiser.

The reverse-diffusion loop in :class:`GraspGenGenerator` calls
:class:`DiffusionNoisePredictionNet` once per timestep (``num_diffusion_iters_eval``
times, on a batch of ``num_objects * num_grasps_per_object`` rows). That denoiser
is a pure tensor-in / tensor-out MLP (+ optional self-attention), which makes it
a clean target for TensorRT.

Everything here is **strictly opt-in**. If ``torch_tensorrt`` is not installed,
or compilation fails for any reason, the public helpers degrade gracefully to
the original eager PyTorch module so existing behaviour is never broken.

The point-cloud backbones (PTv3 sparse convs, PointNet++ custom CUDA kernels)
are intentionally NOT converted — they use ops with no TensorRT equivalent and
run only once per inference. See ``tensorrt.md`` for the full rationale.
"""

from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn

from graspgenx.utils.logging_config import get_logger

logger = get_logger(__name__)


def tensorrt_available() -> bool:
    """Return True if Torch-TensorRT can be imported in this environment."""
    try:
        import torch_tensorrt  # noqa: F401

        return True
    except Exception:  # pragma: no cover - depends on optional install
        return False


# ── Engine caching ──────────────────────────────────────────────────────────
# Compiled TensorRT engines are GPU/driver/TensorRT-version specific, so the
# cache filename embeds a machine key. This both persists engines across
# processes (skipping the multi-second rebuild) and guarantees a machine never
# loads an engine built for a different GPU.


def machine_key() -> str:
    """A filesystem-safe key identifying this GPU + TensorRT build.

    e.g. ``nvidia-geforce-rtx-3090_sm86_trt10.7.0.post1``.
    """
    try:
        gpu = re.sub(r"[^a-z0-9]+", "-", torch.cuda.get_device_name(0).lower()).strip("-")
        cc = torch.cuda.get_device_capability(0)
        gpu_part = f"{gpu}_sm{cc[0]}{cc[1]}"
    except Exception:  # pragma: no cover - no CUDA
        gpu_part = "nocuda"
    try:
        import tensorrt

        trt = re.sub(r"[^a-z0-9.]+", "", tensorrt.__version__.lower())
    except Exception:
        trt = "na"
    return f"{gpu_part}_trt{trt}"


def default_engine_cache_dir() -> str:
    """Default engine cache location: ``<repo_root>/ext/trt_engines``."""
    return str(Path(__file__).resolve().parents[2] / "ext" / "trt_engines")


def _cfg_signature(component: str, **kw) -> str:
    """Short hash of the compile config so a stale engine is never reused for a
    different architecture/batch-range/precision."""
    payload = component + ";" + ";".join(f"{k}={v}" for k, v in sorted(kw.items()))
    return hashlib.md5(payload.encode()).hexdigest()[:8]


def _engine_path(component: str, cfg_sig: str, engine_cache_dir: Optional[str]):
    """Machine-keyed engine path, or None when caching is disabled."""
    if engine_cache_dir is None:
        return None
    fname = f"graspgenx_{component}_{machine_key()}_{cfg_sig}.engine"
    return os.path.join(engine_cache_dir, fname)


def _load_cached_engine(path: Optional[str]):
    """Return a loaded TensorRT module from ``path`` if present, else None."""
    if path is None or not os.path.exists(path):
        return None
    try:
        import torch_tensorrt

        module = torch_tensorrt.load(path)
        logger.info(f"Loaded cached TensorRT engine from {path}")
        return module
    except Exception as e:  # pragma: no cover - corrupt/stale cache
        logger.warning(f"Could not load cached engine {path} ({e}); will recompile.")
        return None


def _save_engine(trt_module, path: Optional[str], example_inputs: list) -> None:
    """Serialize a compiled TensorRT module to ``path`` (torchscript format)."""
    if path is None:
        return
    try:
        import torch_tensorrt

        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch_tensorrt.save(
            trt_module, path, output_format="torchscript", inputs=example_inputs
        )
        logger.info(f"Cached TensorRT engine to {path}")
    except Exception as e:  # pragma: no cover - disk/serialize failure
        logger.warning(f"Could not cache engine to {path} ({e}).")


def _expand_timesteps(
    timesteps, batch_size: int, device: torch.device
) -> torch.Tensor:
    """Reproduce the scalar -> [B] timestep expansion from the eager forward.

    Mirrors ``DiffusionNoisePredictionNet.forward`` (generator.py) so the TRT
    engine always receives a rank-1 ``[B]`` float tensor, keeping the exported
    graph free of the data-dependent scalar branch.
    """
    if not torch.is_tensor(timesteps):
        timesteps = torch.tensor(timesteps, device=device)
    timesteps = timesteps.to(device)
    if timesteps.dim() == 0:
        timesteps = timesteps[None].expand(batch_size)
    return timesteps.float()


class TensorRTDiffusionHead(nn.Module):
    """Drop-in wrapper around ``DiffusionNoisePredictionNet``.

    Keeps the exact ``forward(observation_embedding, timesteps, sample)``
    signature so it can replace the generator's ``diffusion_head`` attribute
    transparently. Runs the compiled TensorRT engine for batch sizes within the
    compiled ``[min_batch, max_batch]`` range and falls back to the eager module
    otherwise (or on any runtime error).
    """

    def __init__(
        self,
        eager_module: nn.Module,
        trt_module: Optional[nn.Module],
        min_batch: int,
        max_batch: int,
    ):
        super().__init__()
        self.eager_module = eager_module
        self.trt_module = trt_module
        self.min_batch = min_batch
        self.max_batch = max_batch

    @property
    def is_accelerated(self) -> bool:
        return self.trt_module is not None

    def forward(self, observation_embedding, timesteps, sample=None):
        batch_size = observation_embedding.shape[0]

        if self.trt_module is not None and self.min_batch <= batch_size <= self.max_batch:
            try:
                ts = _expand_timesteps(
                    timesteps, batch_size, observation_embedding.device
                )
                return self.trt_module(observation_embedding, ts, sample)
            except Exception as e:  # pragma: no cover - defensive fallback
                logger.warning(
                    f"TensorRT diffusion head failed ({e}); falling back to eager."
                )

        return self.eager_module(observation_embedding, timesteps, sample)


class TensorRTModule(nn.Module):
    """Generic wrapper for a single-tensor-in / single-tensor-out module.

    Used for the discriminator's MLP heads (``sample_encoder``,
    ``prediction_head``). Runs the compiled engine for batch sizes within the
    compiled range and falls back to eager otherwise / on error.
    """

    def __init__(self, eager_module, trt_module, min_batch, max_batch):
        super().__init__()
        self.eager_module = eager_module
        self.trt_module = trt_module
        self.min_batch = min_batch
        self.max_batch = max_batch

    @property
    def is_accelerated(self) -> bool:
        return self.trt_module is not None

    def forward(self, x):
        batch_size = x.shape[0]
        if self.trt_module is not None and self.min_batch <= batch_size <= self.max_batch:
            try:
                return self.trt_module(x)
            except Exception as e:  # pragma: no cover - defensive fallback
                logger.warning(f"TensorRT module failed ({e}); falling back to eager.")
        return self.eager_module(x)


def compile_mlp(
    module: nn.Module,
    *,
    in_dim: int,
    device: torch.device,
    min_batch: int = 1,
    opt_batch: int = 500,
    max_batch: int = 4000,
    precision: str = "fp32",
    label: str = "mlp",
    engine_cache_dir: Optional[str] = None,
) -> TensorRTModule:
    """Compile a single-input ``[B, in_dim] -> [B, out_dim]`` MLP with a
    dynamic batch dimension. Returns a passthrough wrapper on any failure.

    If ``engine_cache_dir`` is set, a machine-keyed engine is loaded from disk
    when present, else built and cached there.
    """
    if not tensorrt_available():
        return TensorRTModule(module, None, min_batch, max_batch)

    import torch_tensorrt

    module = module.to(device).eval()

    cfg_sig = _cfg_signature(
        label, in_dim=in_dim, mn=min_batch, opt=opt_batch, mx=max_batch, prec=precision
    )
    path = _engine_path(label.replace(".", "_"), cfg_sig, engine_cache_dir)
    cached = _load_cached_engine(path)
    if cached is not None:
        return TensorRTModule(module, cached, min_batch, max_batch)

    enabled = {torch.float32}
    if precision == "fp16":
        enabled.add(torch.float16)

    inputs = [
        torch_tensorrt.Input(
            min_shape=[min_batch, in_dim],
            opt_shape=[opt_batch, in_dim],
            max_shape=[max_batch, in_dim],
            dtype=torch.float32,
        )
    ]
    try:
        logger.info(f"Compiling {label} with TensorRT (in_dim={in_dim})...")
        trt_module = torch_tensorrt.compile(
            module,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions=enabled,
            truncate_double=True,
        )
        logger.info(f"{label} compiled successfully.")
        _save_engine(
            trt_module, path, [torch.randn(opt_batch, in_dim, device=device)]
        )
        return TensorRTModule(module, trt_module, min_batch, max_batch)
    except Exception as e:
        logger.warning(f"TensorRT compilation of {label} failed ({e}); using eager.")
        return TensorRTModule(module, None, min_batch, max_batch)


def _first_linear_in_features(seq: nn.Module):
    """Return in_features of the first nn.Linear in a module, or None."""
    for m in seq.modules():
        if isinstance(m, nn.Linear):
            return m.in_features
    return None


def compile_diffusion_head(
    head: nn.Module,
    *,
    obs_dim: int,
    sample_dim: int,
    device: torch.device,
    min_batch: int = 1,
    opt_batch: int = 100,
    max_batch: int = 2000,
    precision: str = "fp32",
    engine_cache_dir: Optional[str] = None,
) -> TensorRTDiffusionHead:
    """Compile a ``DiffusionNoisePredictionNet`` with a dynamic batch dimension.

    Returns a :class:`TensorRTDiffusionHead`. On any failure (missing
    ``torch_tensorrt`` or a compilation error) it returns a passthrough wrapper
    that still runs the original eager module, so callers never need to handle
    the error path. If ``engine_cache_dir`` is set, a machine-keyed engine is
    loaded from disk when present, else built and cached there.
    """
    if not tensorrt_available():
        logger.warning(
            "torch_tensorrt not available; install the 'tensorrt' extra. "
            "Running the diffusion head in eager mode."
        )
        return TensorRTDiffusionHead(head, None, min_batch, max_batch)

    import torch_tensorrt

    head = head.to(device).eval()

    cfg_sig = _cfg_signature(
        "diffusion_head",
        obs_dim=obs_dim,
        sample_dim=sample_dim,
        mn=min_batch,
        opt=opt_batch,
        mx=max_batch,
        prec=precision,
    )
    path = _engine_path("diffusion_head", cfg_sig, engine_cache_dir)
    cached = _load_cached_engine(path)
    if cached is not None:
        return TensorRTDiffusionHead(head, cached, min_batch, max_batch)

    enabled = {torch.float32}
    if precision == "fp16":
        enabled.add(torch.float16)

    def _inp(*trailing):
        return torch_tensorrt.Input(
            min_shape=[min_batch, *trailing],
            opt_shape=[opt_batch, *trailing],
            max_shape=[max_batch, *trailing],
            dtype=torch.float32,
        )

    # observation_embedding [B, obs_dim], timesteps [B], sample [B, sample_dim]
    inputs = [
        _inp(obs_dim),
        torch_tensorrt.Input(
            min_shape=[min_batch],
            opt_shape=[opt_batch],
            max_shape=[max_batch],
            dtype=torch.float32,
        ),
        _inp(sample_dim),
    ]

    try:
        logger.info(
            f"Compiling diffusion head with TensorRT (precision={precision}, "
            f"batch {min_batch}/{opt_batch}/{max_batch}, obs_dim={obs_dim}, "
            f"sample_dim={sample_dim})..."
        )
        trt_module = torch_tensorrt.compile(
            head,
            ir="dynamo",
            inputs=inputs,
            enabled_precisions=enabled,
            truncate_double=True,
        )
        logger.info("TensorRT diffusion head compiled successfully.")
        example = [
            torch.randn(opt_batch, obs_dim, device=device),
            torch.full((opt_batch,), 0.0, device=device),
            torch.randn(opt_batch, sample_dim, device=device),
        ]
        _save_engine(trt_module, path, example)
        return TensorRTDiffusionHead(head, trt_module, min_batch, max_batch)
    except Exception as e:
        logger.warning(
            f"TensorRT compilation failed ({e}); running diffusion head in eager mode."
        )
        return TensorRTDiffusionHead(head, None, min_batch, max_batch)


def compile_encoder_graphbreak_tolerant(encoder: nn.Module):
    """Compile a control-flow-heavy encoder (e.g. ptv3vanilla) with the
    Torch-TensorRT torch.compile backend in graph-break-tolerant mode.

    The encoder cannot be a single static engine (occupancy-dependent grid
    pooling ⇒ data-dependent shapes), so static sub-blocks (qkv/proj/MLP/
    sparse-conv einsum) become TRT engines while serialization/pooling/padding
    run eager. Returns the original module unchanged if torch_tensorrt is
    unavailable. Never raises — suppress_errors keeps unconvertible regions in
    eager PyTorch.
    """
    if not tensorrt_available():
        return encoder
    import torch._dynamo

    torch._dynamo.config.suppress_errors = True
    torch._dynamo.config.cache_size_limit = 256
    try:
        return torch.compile(
            encoder,
            backend="torch_tensorrt",
            options={
                "min_block_size": 1,
                "truncate_long_and_double": True,
                "enabled_precisions": {torch.float32},
            },
        )
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Encoder TensorRT compile setup failed ({e}); using eager.")
        return encoder


def accelerate_sampler(
    sampler,
    *,
    precision: str = "fp32",
    min_batch: int = 1,
    opt_batch: Optional[int] = None,
    max_batch: int = 4000,
    accelerate_encoders: bool = False,
    engine_cache_dir: Optional[str] = "__default__",
) -> bool:
    """Swap a ``GraspGenXSampler``'s diffusion head for a TensorRT-compiled one.

    Mutates ``sampler.model.grasp_generator.diffusion_head`` in place. Returns
    ``True`` if TensorRT acceleration was actually applied, ``False`` otherwise
    (e.g. TensorRT not installed, not on CUDA, or compilation failed). Safe to
    call unconditionally — it never raises.

    Engines are cached under ``engine_cache_dir`` keyed by GPU + TensorRT
    version (see :func:`machine_key`), so subsequent runs load instead of
    rebuilding and different GPUs never collide. The default ``"__default__"``
    resolves to ``<repo_root>/ext/trt_engines``; pass ``None`` to disable
    caching (compile in-memory only).
    """
    if not tensorrt_available():
        logger.warning("TensorRT requested but torch_tensorrt is not installed.")
        return False

    try:
        generator = sampler.model.grasp_generator
        device = next(generator.parameters()).device
    except Exception as e:  # pragma: no cover - defensive
        logger.warning(f"Could not locate generator for TensorRT: {e}")
        return False

    if device.type != "cuda":
        logger.warning("TensorRT requires a CUDA model; skipping acceleration.")
        return False

    if engine_cache_dir == "__default__":
        engine_cache_dir = default_engine_cache_dir()
    if engine_cache_dir is not None:
        logger.info(f"TensorRT engine cache dir: {engine_cache_dir}")

    obs_dim = generator.num_object_dim + generator.num_gripper_dim
    sample_dim = generator.output_dim
    if opt_batch is None:
        opt_batch = int(getattr(generator, "num_grasps_per_object", 100))
        opt_batch = max(min_batch, opt_batch)

    accelerated = {}

    # 1) Generator diffusion head — the 20x-per-inference hot loop.
    wrapped = compile_diffusion_head(
        generator.diffusion_head,
        obs_dim=obs_dim,
        sample_dim=sample_dim,
        device=device,
        min_batch=min_batch,
        opt_batch=opt_batch,
        max_batch=max_batch,
        precision=precision,
        engine_cache_dir=engine_cache_dir,
    )
    generator.diffusion_head = wrapped
    accelerated["diffusion_head"] = wrapped.is_accelerated

    # 2) Discriminator MLP heads — clean single-input MLPs that run over all
    #    grasps (batch = num_objects * num_grasps). The ptv3vanilla object
    #    encoder is intentionally NOT converted: it runs on batch=1 and its
    #    data-dependent control flow (serialization / boolean-mask gather)
    #    does not export. See tensorrt.md.
    discriminator = getattr(sampler.model, "grasp_discriminator", None)
    if discriminator is not None:
        for attr in ("sample_encoder", "prediction_head"):
            mod = getattr(discriminator, attr, None)
            if mod is None:
                continue
            in_dim = _first_linear_in_features(mod)
            if in_dim is None:
                logger.warning(f"Skipping discriminator.{attr}: no Linear layer found.")
                continue
            wrapped_mlp = compile_mlp(
                mod,
                in_dim=in_dim,
                device=device,
                min_batch=min_batch,
                opt_batch=opt_batch,
                max_batch=max_batch,
                precision=precision,
                label=f"discriminator.{attr}",
                engine_cache_dir=engine_cache_dir,
            )
            setattr(discriminator, attr, wrapped_mlp)
            accelerated[f"discriminator.{attr}"] = wrapped_mlp.is_accelerated

    # 3) Object encoders (opt-in) — ptv3vanilla, partial TRT via the
    #    graph-break-tolerant torch.compile backend. See tensorrt.md for why a
    #    clean single engine is not achievable without retraining.
    if accelerate_encoders:
        for owner, name in (
            (generator, "generator"),
            (getattr(sampler.model, "grasp_discriminator", None), "discriminator"),
        ):
            if owner is None:
                continue
            enc = getattr(owner, "object_encoder", None)
            if enc is None:
                continue
            owner.object_encoder = compile_encoder_graphbreak_tolerant(enc)
            accelerated[f"{name}.object_encoder"] = True

    logger.info(f"TensorRT acceleration summary: {accelerated}")
    return any(accelerated.values())
