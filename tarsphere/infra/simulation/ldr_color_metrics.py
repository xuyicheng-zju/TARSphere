"""Direct LdrColor frame metrics, including content-masked statistics."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image, ImageFilter, ImageOps


def ldr_color_frame_stats(rgb: np.ndarray) -> dict[str, Any]:
    """Return full-frame and content-masked LdrColor statistics for one RGB."""

    rgb = np.asarray(rgb)
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(
            f"LdrColor stats require an HxWx3 RGB array, got {rgb.shape!r}."
        )
    rgb = np.asarray(rgb[..., :3], dtype=np.uint8)
    rgb_float = rgb.astype(np.float32)
    std = float(np.std(rgb_float))
    mean = float(np.mean(rgb_float))
    black_ratio = float(np.mean(np.all(rgb < 10, axis=2)))
    white_ratio = float(np.mean(np.all(rgb > 245, axis=2)))
    channel_range = np.ptp(rgb_float, axis=2)
    pixel_chroma_mean = float(np.mean(channel_range))
    pixel_chroma_gt_8_ratio = float(np.mean(channel_range > 8.0))
    channel_means = np.mean(rgb_float, axis=(0, 1))
    colorfulness = float(np.std(channel_means))
    red = rgb_float[:, :, 0]
    green = rgb_float[:, :, 1]
    blue = rgb_float[:, :, 2]
    saturated_red_count = int(
        np.sum(
            (np.abs(red - 220.0) <= 15.0)
            & (np.abs(green - 58.0) <= 15.0)
            & (np.abs(blue - 58.0) <= 15.0)
        )
    )
    gray = np.asarray(ImageOps.grayscale(Image.fromarray(rgb)), dtype=np.uint8)
    edges = np.asarray(
        Image.fromarray(gray).filter(ImageFilter.FIND_EDGES), dtype=np.uint8
    )
    edge_ratio = float(np.mean(edges > 28))
    height, width = rgb.shape[:2]
    border_width = max(1, min(height, width) // 40)
    border_pixels = np.concatenate(
        (
            rgb[:border_width].reshape(-1, 3),
            rgb[-border_width:].reshape(-1, 3),
            rgb[:, :border_width].reshape(-1, 3),
            rgb[:, -border_width:].reshape(-1, 3),
        ),
        axis=0,
    )
    background_color = np.median(border_pixels, axis=0)
    background_delta = np.max(
        np.abs(rgb.astype(np.int16) - background_color),
        axis=2,
    )
    scene_content_mask = background_delta > 12.0
    scene_content_ratio = float(np.mean(scene_content_mask))
    content_y, content_x = np.nonzero(scene_content_mask)
    if content_x.size:
        content_bbox = [
            int(content_x.min()),
            int(content_y.min()),
            int(content_x.max()),
            int(content_y.max()),
        ]
        content_center_offset_px = [
            float((content_bbox[0] + content_bbox[2]) * 0.5 - (width - 1) * 0.5),
            float((content_bbox[1] + content_bbox[3]) * 0.5 - (height - 1) * 0.5),
        ]
        scene_content_bbox_ratio = float(
            (content_bbox[2] - content_bbox[0] + 1)
            * (content_bbox[3] - content_bbox[1] + 1)
            / float(width * height)
        )
        content_rgb = rgb_float[scene_content_mask]
        content_uint = rgb[scene_content_mask]
        luminance = (
            rgb_float[:, :, 0] * 0.2126
            + rgb_float[:, :, 1] * 0.7152
            + rgb_float[:, :, 2] * 0.0722
        )
        content_luminance = luminance[scene_content_mask]
        content_masked_mean = float(np.mean(content_rgb))
        content_masked_std = float(np.std(content_rgb))
        content_masked_black_ratio = float(np.mean(np.all(content_uint < 10, axis=1)))
        content_masked_white_ratio = float(np.mean(np.all(content_uint > 245, axis=1)))
        content_masked_pixel_count = int(content_uint.shape[0])
        luminance_percentiles = np.quantile(
            content_luminance,
            [0.1, 0.5, 0.9],
        )
        content_luminance_mean = float(np.mean(content_luminance))
        content_luminance_p10 = float(luminance_percentiles[0])
        content_luminance_p50 = float(luminance_percentiles[1])
        content_luminance_p90 = float(luminance_percentiles[2])
        content_luminance_clipped_ratio = float(np.mean(content_luminance >= 250.0))
        content_any_channel_clipped_ratio = float(
            np.mean(np.any(content_uint >= 255, axis=1))
        )
        tile_medians: list[float] = []
        x_min, y_min, x_max, y_max = content_bbox
        for tile_y in range(6):
            y_start = y_min + (y_max - y_min + 1) * tile_y // 6
            y_end = y_min + (y_max - y_min + 1) * (tile_y + 1) // 6
            for tile_x in range(6):
                x_start = x_min + (x_max - x_min + 1) * tile_x // 6
                x_end = x_min + (x_max - x_min + 1) * (tile_x + 1) // 6
                tile_mask = scene_content_mask[y_start:y_end, x_start:x_end]
                if tile_mask.size and float(np.mean(tile_mask)) >= 0.2:
                    tile_medians.append(
                        float(
                            np.median(
                                luminance[y_start:y_end, x_start:x_end][tile_mask]
                            )
                        )
                    )
        if tile_medians:
            tile_p10, tile_p90 = np.quantile(tile_medians, [0.1, 0.9])
            tile_luminance_p10 = float(tile_p10)
            tile_luminance_p90 = float(tile_p90)
            tile_luminance_spread = float(tile_p90 - tile_p10)
        else:
            tile_luminance_p10 = None
            tile_luminance_p90 = None
            tile_luminance_spread = None
    else:
        content_bbox = None
        content_center_offset_px = None
        scene_content_bbox_ratio = 0.0
        content_masked_mean = None
        content_masked_std = None
        content_masked_black_ratio = None
        content_masked_white_ratio = None
        content_masked_pixel_count = 0
        content_luminance_mean = None
        content_luminance_p10 = None
        content_luminance_p50 = None
        content_luminance_p90 = None
        content_luminance_clipped_ratio = None
        content_any_channel_clipped_ratio = None
        tile_medians = []
        tile_luminance_p10 = None
        tile_luminance_p90 = None
        tile_luminance_spread = None
    return {
        "rgb_frame_shape": [int(v) for v in rgb.shape],
        "rgb_frame_mean": round(mean, 3),
        "rgb_frame_std": round(std, 3),
        "rgb_black_ratio": round(black_ratio, 4),
        "rgb_white_ratio": round(white_ratio, 4),
        "rgb_extreme_ratio": round(black_ratio + white_ratio, 4),
        "rgb_pixel_chroma_mean": round(pixel_chroma_mean, 3),
        "rgb_pixel_chroma_gt_8_ratio": round(pixel_chroma_gt_8_ratio, 4),
        "rgb_edge_ratio": round(edge_ratio, 4),
        "rgb_scene_content_ratio": round(scene_content_ratio, 4),
        "rgb_scene_content_bbox_ratio": round(scene_content_bbox_ratio, 4),
        "rgb_scene_content_bbox_xyxy_px": content_bbox,
        "rgb_scene_content_center_offset_px": (
            [round(float(v), 3) for v in content_center_offset_px]
            if content_center_offset_px is not None
            else None
        ),
        "rgb_background_color": [round(float(v), 3) for v in background_color],
        "rgb_channel_mean": [round(float(v), 3) for v in channel_means],
        "rgb_channel_mean_std": round(colorfulness, 3),
        "rgb_saturated_red_count": saturated_red_count,
        "rgb_content_masked_mean": (
            None if content_masked_mean is None else round(content_masked_mean, 3)
        ),
        "rgb_content_masked_std": (
            None if content_masked_std is None else round(content_masked_std, 3)
        ),
        "rgb_content_masked_black_ratio": (
            None
            if content_masked_black_ratio is None
            else round(content_masked_black_ratio, 4)
        ),
        "rgb_content_masked_white_ratio": (
            None
            if content_masked_white_ratio is None
            else round(content_masked_white_ratio, 4)
        ),
        "rgb_content_masked_pixel_count": content_masked_pixel_count,
        "rgb_content_luminance_mean": (
            None if content_luminance_mean is None else round(content_luminance_mean, 3)
        ),
        "rgb_content_luminance_p10": (
            None if content_luminance_p10 is None else round(content_luminance_p10, 3)
        ),
        "rgb_content_luminance_p50": (
            None if content_luminance_p50 is None else round(content_luminance_p50, 3)
        ),
        "rgb_content_luminance_p90": (
            None if content_luminance_p90 is None else round(content_luminance_p90, 3)
        ),
        "rgb_content_luminance_clipped_ratio": (
            None
            if content_luminance_clipped_ratio is None
            else round(content_luminance_clipped_ratio, 4)
        ),
        "rgb_content_any_channel_clipped_ratio": (
            None
            if content_any_channel_clipped_ratio is None
            else round(content_any_channel_clipped_ratio, 4)
        ),
        "rgb_content_tile_count": len(tile_medians),
        "rgb_content_tile_luminance_p10": (
            None if tile_luminance_p10 is None else round(tile_luminance_p10, 3)
        ),
        "rgb_content_tile_luminance_p90": (
            None if tile_luminance_p90 is None else round(tile_luminance_p90, 3)
        ),
        "rgb_content_tile_luminance_spread": (
            None if tile_luminance_spread is None else round(tile_luminance_spread, 3)
        ),
    }


__all__ = ["ldr_color_frame_stats"]
