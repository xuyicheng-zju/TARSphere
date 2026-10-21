"""FrontierNet RGB-D preprocessing with an explicit crop transform."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
from PIL import Image

from tarsphere.infra.navigation.open_frontier.contracts import CameraIntrinsics


@dataclass(frozen=True, slots=True)
class PreprocessedFrontierFrame:
    rgb: np.ndarray
    depth_m: np.ndarray
    normalized_depth: np.ndarray
    intrinsics: CameraIntrinsics
    scale: float
    crop_left_px: int
    crop_top_px: int


def _validate_rgbd(rgb: Any, depth_m: Any) -> tuple[np.ndarray, np.ndarray]:
    color = np.asarray(rgb)
    depth = np.asarray(depth_m, dtype=np.float32)
    if color.ndim != 3 or color.shape[2] < 3:
        raise ValueError(f"FrontierNet RGB must have shape HxWx3; got {color.shape}.")
    if depth.ndim == 3 and depth.shape[2] == 1:
        depth = depth[:, :, 0]
    if depth.ndim != 2:
        raise ValueError(f"FrontierNet depth must have shape HxW; got {depth.shape}.")
    if color.shape[:2] != depth.shape:
        raise ValueError(
            f"FrontierNet RGB/depth sizes differ: {color.shape[:2]} vs {depth.shape}."
        )
    if not np.issubdtype(color.dtype, np.integer):
        if not np.all(np.isfinite(color)):
            raise ValueError("FrontierNet RGB contains non-finite values.")
        color = np.clip(color, 0.0, 255.0)
    color = np.asarray(color[:, :, :3], dtype=np.uint8)
    valid_depth = np.isfinite(depth) & (depth > 0.0)
    if not valid_depth.any():
        raise ValueError(
            "FrontierNet requires at least one finite positive depth pixel."
        )
    depth = np.where(valid_depth, depth, 0.0).astype(np.float32, copy=False)
    return color, depth


def _resize_depth_for_checkpoint(
    depth: np.ndarray,
    *,
    width: int,
    height: int,
) -> np.ndarray:
    """Apply the released checkpoint's float-depth bicubic resize."""

    return np.asarray(
        Image.fromarray(depth, mode="F").resize(
            (width, height),
            Image.Resampling.BICUBIC,
        ),
        dtype=np.float32,
    )


def preprocess_frontier_rgbd(
    rgb: Any,
    depth_m: Any,
    intrinsics: CameraIntrinsics,
    *,
    output_size: tuple[int, int] = (320, 320),
    resize_margin: float = 0.02,
) -> PreprocessedFrontierFrame:
    """Resize and centre-crop RGB-D exactly once and transform intrinsics."""

    color, depth = _validate_rgbd(rgb, depth_m)
    output_width, output_height = (int(output_size[0]), int(output_size[1]))
    if output_width <= 1 or output_height <= 1:
        raise ValueError("FrontierNet output size must exceed one pixel per axis.")
    if (intrinsics.width, intrinsics.height) != (color.shape[1], color.shape[0]):
        raise ValueError(
            "Camera intrinsics dimensions do not match the RGB-D observation."
        )
    if not math.isfinite(float(resize_margin)) or float(resize_margin) < 0.0:
        raise ValueError("resize_margin must be finite and non-negative.")

    scale = max(
        output_width / color.shape[1],
        output_height / color.shape[0],
    ) + float(resize_margin)
    resized_width = max(output_width, int(color.shape[1] * scale))
    resized_height = max(output_height, int(color.shape[0] * scale))
    crop_left = int(round((resized_width - output_width) / 2.0))
    crop_top = int(round((resized_height - output_height) / 2.0))

    resized_rgb = np.asarray(
        Image.fromarray(color, mode="RGB").resize(
            (resized_width, resized_height),
            Image.Resampling.BICUBIC,
        ),
        dtype=np.uint8,
    )
    resized_depth = _resize_depth_for_checkpoint(
        depth,
        width=resized_width,
        height=resized_height,
    )
    row_slice = slice(crop_top, crop_top + output_height)
    col_slice = slice(crop_left, crop_left + output_width)
    cropped_rgb = resized_rgb[row_slice, col_slice].copy()
    cropped_depth = resized_depth[row_slice, col_slice].copy()

    valid = np.isfinite(cropped_depth) & (cropped_depth > 0.0)
    if not valid.any():
        raise ValueError("FrontierNet crop contains no valid metric depth.")
    depth_min = float(cropped_depth.min())
    depth_max = float(cropped_depth.max())
    normalized_depth = np.zeros_like(cropped_depth, dtype=np.float32)
    if depth_max > depth_min:
        normalized_depth = (
            (cropped_depth - depth_min) / (depth_max - depth_min + 1e-6)
        ).astype(np.float32)

    # PIL operates on the integer resize dimensions above.  Use those exact
    # per-axis scales for projection rather than the pre-rounding nominal
    # scale, otherwise a half-pixel crop rounding error leaks into 3-D lifts.
    scale_x = resized_width / color.shape[1]
    scale_y = resized_height / color.shape[0]
    transformed_intrinsics = CameraIntrinsics(
        width=output_width,
        height=output_height,
        fx=intrinsics.fx * scale_x,
        fy=intrinsics.fy * scale_y,
        cx=intrinsics.cx * scale_x - crop_left,
        cy=intrinsics.cy * scale_y - crop_top,
    )
    return PreprocessedFrontierFrame(
        rgb=cropped_rgb,
        depth_m=cropped_depth,
        normalized_depth=normalized_depth,
        intrinsics=transformed_intrinsics,
        scale=float(scale),
        crop_left_px=int(crop_left),
        crop_top_px=int(crop_top),
    )


__all__ = ["PreprocessedFrontierFrame", "preprocess_frontier_rgbd"]
