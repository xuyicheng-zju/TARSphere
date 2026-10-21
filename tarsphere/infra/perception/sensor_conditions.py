"""Episode-wide, deterministic perturbations of the Robot's ego RGB-D sensor."""

from __future__ import annotations

import hashlib
from typing import Literal

import numpy as np
from PIL import Image

PerceptualCondition = Literal["low_light", "reduced_resolution", "depth_noise"]

# A light intervention scales scene-authored and repaired fixtures, never the
# saved RGB array. Exposure remains the shared evaluation setting.
# A single RTX camera-exposure adjustment gives a scene-wide low-light image
# even when GRScenes' local lights are outweighed by emissive materials.
LOW_LIGHT_EXPOSURE_SCALE = 0.22
REDUCED_RESOLUTION_SCALE = 0.40
DEPTH_NOISE_BASE_STD_M = 0.018
DEPTH_NOISE_RANGE_STD_M_PER_M2 = 0.004
DEPTH_NOISE_DROPOUT_PROBABILITY = 0.05


def condition_for_episode(episode: object) -> PerceptualCondition | None:
    """Read the one authoritative task profile without a runtime-side alias."""

    evaluation = getattr(episode, "evaluation", None)
    profile = getattr(evaluation, "task_profile", None)
    capability = getattr(profile, "capability_slice", None)
    if getattr(capability, "value", capability) != "perceptual_robustness":
        return None
    subtype = getattr(profile, "subtype", None)
    if subtype not in {"low_light", "reduced_resolution", "depth_noise"}:
        raise ValueError(f"Unknown perceptual robustness condition: {subtype!r}")
    return subtype


def condition_sensor_frame(
    rgb: np.ndarray,
    depth: np.ndarray | None,
    *,
    condition: PerceptualCondition | None,
    episode_id: str,
    episode_capture_index: int,
    run_seed: int = 0,
    sensor_name: str = "ego_rgbd",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Return one aligned policy-facing RGB-D frame at the native canvas size.

    The untouched depth input remains evaluator-private. All policy consumers
    receive these returned arrays, including continuous mapping captures.
    """

    colour = np.asarray(rgb, dtype=np.uint8)
    if colour.ndim != 3 or colour.shape[2] != 3:
        raise ValueError(f"Ego RGB must be HxWx3 uint8, got {colour.shape!r}")
    metric_depth = None if depth is None else np.asarray(depth, dtype=np.float32)
    if metric_depth is not None and metric_depth.shape != colour.shape[:2]:
        raise ValueError("Ego RGB and depth must have identical spatial shape")
    if condition in {None, "low_light"}:
        return colour, metric_depth
    if condition == "reduced_resolution":
        height, width = colour.shape[:2]
        reduced = (
            max(1, round(width * REDUCED_RESOLUTION_SCALE)),
            max(1, round(height * REDUCED_RESOLUTION_SCALE)),
        )
        image = Image.fromarray(colour)
        restored = image.resize(reduced, Image.Resampling.BILINEAR).resize(
            (width, height),
            Image.Resampling.BILINEAR,
        )
        policy_rgb = np.asarray(restored, dtype=np.uint8)
        if metric_depth is None:
            return policy_rgb, None
        # A lower-resolution depth sensor measures one value per coarse pixel;
        # nearest expansion retains its holes and does not invent sub-pixel
        # geometry when the 640x480 policy canvas is restored.
        source_rows = np.minimum(
            (np.arange(reduced[1]) * height // reduced[1]),
            height - 1,
        )
        source_cols = np.minimum(
            (np.arange(reduced[0]) * width // reduced[0]),
            width - 1,
        )
        coarse_depth = metric_depth[np.ix_(source_rows, source_cols)]
        output_rows = np.minimum(
            (np.arange(height) * reduced[1] // height),
            reduced[1] - 1,
        )
        output_cols = np.minimum(
            (np.arange(width) * reduced[0] // width),
            reduced[0] - 1,
        )
        return policy_rgb, coarse_depth[np.ix_(output_rows, output_cols)].copy()
    if condition == "depth_noise":
        if metric_depth is None:
            raise RuntimeError("depth_noise requires a synchronized ego depth frame")
        # The perturbation is a pure function of formal episode identity and
        # the episode-local sensor sequence. It must not depend on batch order,
        # worker reuse, model name, or process-global RNG state.
        seed_material = (
            f"v1:{int(run_seed)}:{episode_id}:{sensor_name}:"
            f"{int(episode_capture_index)}"
        )
        seed = int.from_bytes(
            hashlib.sha256(seed_material.encode("utf-8")).digest()[:8],
            byteorder="big",
        )
        random = np.random.default_rng(seed)
        valid = np.isfinite(metric_depth) & (metric_depth > 0.0)
        sigma = DEPTH_NOISE_BASE_STD_M + DEPTH_NOISE_RANGE_STD_M_PER_M2 * np.square(
            np.where(valid, metric_depth, 0.0),
            dtype=np.float32,
        )
        noisy = metric_depth.copy()
        perturbation = random.normal(0.0, 1.0, metric_depth.shape).astype(np.float32)
        noisy[valid] = np.maximum(
            0.05,
            metric_depth[valid] + perturbation[valid] * sigma[valid],
        )
        missing = valid & (
            random.random(metric_depth.shape) < DEPTH_NOISE_DROPOUT_PROBABILITY
        )
        noisy[missing] = np.nan
        return colour, noisy
    raise ValueError(f"Unknown perceptual robustness condition: {condition!r}")


__all__ = [
    "LOW_LIGHT_EXPOSURE_SCALE",
    "PerceptualCondition",
    "condition_for_episode",
    "condition_sensor_frame",
]
