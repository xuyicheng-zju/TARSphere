"""Readable set-of-marks rendering for current visual frontiers."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from tarsphere.infra.navigation.open_frontier.contracts import DisplayFrontier


def _font(size: int) -> ImageFont.ImageFont:
    for path in (
        "DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
    ):
        try:
            return ImageFont.truetype(path, size)
        except OSError:
            continue
    return ImageFont.load_default()


def render_frontier_overlay(
    rgb: np.ndarray,
    frontier_mask: np.ndarray,
    displayed: tuple[DisplayFrontier, ...],
) -> np.ndarray:
    """Overlay dense frontier support plus grounded A/B/C anchors."""

    color = np.asarray(rgb, dtype=np.uint8)
    mask = np.asarray(frontier_mask, dtype=bool)
    if color.ndim != 3 or color.shape[2] != 3:
        raise ValueError("Frontier overlay RGB must have shape HxWx3.")
    if mask.shape != color.shape[:2]:
        raise ValueError("Frontier overlay mask must align with RGB.")

    image = Image.fromarray(color, mode="RGB").convert("RGBA")
    support = np.zeros((*mask.shape, 4), dtype=np.uint8)
    support[mask] = np.asarray([20, 184, 166, 72], dtype=np.uint8)
    image = Image.alpha_composite(image, Image.fromarray(support, mode="RGBA"))
    draw = ImageDraw.Draw(image, mode="RGBA")
    radius = max(13, int(round(min(image.size) * 0.045)))
    font = _font(max(16, int(radius * 1.15)))

    for candidate in displayed:
        x, y = candidate.pixel_xy
        draw.ellipse(
            (x - radius, y - radius, x + radius, y + radius),
            fill=(15, 23, 42, 225),
            outline=(56, 189, 248, 255),
            width=max(2, radius // 6),
        )
        box = draw.textbbox((0, 0), candidate.label, font=font)
        draw.text(
            (
                x - (box[2] - box[0]) / 2 - box[0],
                y - (box[3] - box[1]) / 2 - box[1],
            ),
            candidate.label,
            font=font,
            fill=(248, 250, 252, 255),
        )

    return np.asarray(image.convert("RGB"), dtype=np.uint8)


__all__ = ["render_frontier_overlay"]
