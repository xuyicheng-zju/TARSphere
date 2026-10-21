"""Canonical visualization of metric depth and RGB-D evidence."""

from __future__ import annotations

import io

import numpy as np
from PIL import Image


# A fixed indoor range keeps depth colours comparable across policy steps.
# The previous per-frame percentile stretch could map the same metric distance
# to a different colour after every turn and often collapsed a corridor into a
# nearly uniform purple field.  Values outside the range remain valid but are
# clipped to the nearest/farthest endpoint.
DEPTH_VIS_NEAR_M = 0.20
DEPTH_VIS_FAR_M = 8.00


def _depth_rainbow(normalized: np.ndarray) -> np.ndarray:
    """Map [0, 1] depth to red-yellow-green-cyan-blue uint8 colours."""

    values = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    anchors = np.asarray([0.0, 0.25, 0.50, 0.75, 1.0])
    red = np.interp(values, anchors, [1.0, 1.0, 0.0, 0.0, 0.0])
    green = np.interp(values, anchors, [0.0, 1.0, 1.0, 1.0, 0.0])
    blue = np.interp(values, anchors, [0.0, 0.0, 0.0, 1.0, 1.0])
    return np.stack([red, green, blue], axis=-1) * 255.0


def colorize_metric_depth(depth) -> np.ndarray:
    """Return metric depth as red-near/blue-far RGB with black invalid pixels.

    The scale is fixed to :data:`DEPTH_VIS_NEAR_M` through
    :data:`DEPTH_VIS_FAR_M`, so colours retain one metric meaning across an
    episode. Intermediate yellow/green/cyan anchors preserve indoor structure
    better than the old red-blue interpolation, whose midpoint was a broad,
    low-contrast purple field. This function owns the colour contract used by
    both agent prompts and saved RGB-D evidence.
    """

    values = np.asarray(depth, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError(f"depth frame must be 2-D, got {values.shape}")

    valid = np.isfinite(values) & (values > 0.0)
    rgb = np.zeros((*values.shape, 3), dtype=np.uint8)
    if not np.any(valid):
        return rgb

    normalized = (values - DEPTH_VIS_NEAR_M) / (DEPTH_VIS_FAR_M - DEPTH_VIS_NEAR_M)
    colors = _depth_rainbow(normalized).astype(np.uint8)
    rgb[valid] = colors[valid]
    return rgb


def compose_rgbd_image(rgb, depth) -> bytes:
    """Compose a side-by-side RGB | depth-colormap PNG.

    Args:
        rgb: [H, W, 3] uint8 array.
        depth: [H, W] float array (metres).

    Returns:
        PNG bytes.
    """
    if rgb is None or depth is None:
        return b""
    rgb = np.asarray(rgb)
    depth = np.asarray(depth)
    if rgb.dtype != np.uint8:
        rgb = (
            (rgb * 255).clip(0, 255).astype(np.uint8)
            if rgb.max() <= 1.0
            else rgb.astype(np.uint8)
        )
    h, w = rgb.shape[:2]

    depth_vis = colorize_metric_depth(depth)

    combined = np.concatenate([rgb, depth_vis], axis=1)
    img = Image.fromarray(combined)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


__all__ = [
    "DEPTH_VIS_FAR_M",
    "DEPTH_VIS_NEAR_M",
    "colorize_metric_depth",
    "compose_rgbd_image",
]
