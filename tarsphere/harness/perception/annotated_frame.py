"""Draw detections on an ego frame as an image the VLM can actually look at.

Perception tools return "here is what the grounder found": ``locate_object``
draws a bounding box; ``grasp_object`` draws the SAM3 instance
mask it will act on. Both are only useful to a vision model if the match is drawn
on the frame — quoted numbers ask the model to re-locate the object in the
picture itself, which is the work it just called the grounder to avoid.

The drawing lives here rather than beside any caller so every tool produces the
same picture: same colour, same caption format, same JPEG encoding. When one of
them owned it privately, the others could only reuse it by reaching into a tool
handler's static method.
"""

from __future__ import annotations

import base64
import io
from typing import Any, Final, Iterable, Sequence

import numpy as np


#: Box outline and caption background. One colour, so a drawn box always means
#: "the detector reported this" regardless of which capability drew it.
_BOX_RGB = (255, 40, 40)
_CAPTION_RGB = (255, 255, 255)
#: Mask fill uses cyan rather than _BOX_RGB so a warm/red target stays visible
#: under the tint; the outline still uses _BOX_RGB for the same semantic cue.
_MASK_FILL_RGB = (0, 210, 255)
_MASK_FILL_ALPHA = 96


def _load_rgb_frame(frame: Any):
    from PIL import Image

    try:
        return Image.fromarray(np.asarray(frame).astype("uint8")).convert("RGB")
    except (TypeError, ValueError):
        return None


def _jpeg_data_url(image: Any) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=88)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode(
        "ascii"
    )


def rgb_frame_to_jpeg_data_url(frame: Any) -> str | None:
    """Encode one RGB frame for model-visible historical inspection."""

    image = _load_rgb_frame(frame)
    if image is None:
        return None
    try:
        return _jpeg_data_url(image)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _draw_caption(draw: Any, x1: float, y1: float, caption: str) -> None:
    text_box = draw.textbbox((x1, y1), caption)
    draw.rectangle(
        [text_box[0] - 2, text_box[1] - 2, text_box[2] + 2, text_box[3] + 2],
        fill=_BOX_RGB,
    )
    draw.text((x1, y1), caption, fill=_CAPTION_RGB)


def annotate_detections(
    frame: Any,
    detections: Iterable[Any],
) -> str | None:
    """Draw labelled boxes on ``frame``; return a JPEG data URL, or None.

    ``None`` when the frame cannot be read as an image — the caller keeps its
    text result rather than failing the call, because the numbers it already
    reported are still true without the picture.

    Each detection needs ``box_xyxy`` in ego-frame pixels, ``label``, and
    ``score``, which is what every perception backend in this package returns.
    """
    from PIL import ImageDraw

    image = _load_rgb_frame(frame)
    if image is None:
        return None
    try:
        draw = ImageDraw.Draw(image)
        for detection in detections:
            box: Sequence[float] = detection.box_xyxy
            x1, y1, x2, y2 = (float(v) for v in box)
            draw.rectangle([x1, y1, x2, y2], outline=_BOX_RGB, width=3)
            caption = f"{detection.label} {float(detection.score):.2f}"
            _draw_caption(draw, x1, y1, caption)
        return _jpeg_data_url(image)
    except (AttributeError, OSError, TypeError, ValueError):
        # Annotation is explanatory evidence, never the semantic result itself.
        # Isaac may supply a mixed Pillow Python/extension build; an encoder or
        # drawing incompatibility must not turn a valid detector result into a
        # failed tool call.
        return None


#: Stable option marks for grounder candidates, mirroring the frontier A/B/C
#: convention: uppercase letters are positions in the confidence ranking, not
#: identities, so the model reads the same vocabulary across both overlays.
GROUNDING_CANDIDATE_LABELS: Final = ("A", "B", "C")


def annotate_grounder_candidates(
    frame: Any,
    candidates: Sequence[Any],
    labels: Sequence[str] = GROUNDING_CANDIDATE_LABELS,
) -> str | None:
    """Draw choice-marked boxes for grounder candidates; JPEG data URL or None.

    Candidate ``i`` is drawn with ``labels[i]`` at its top-left corner, so the
    model can answer with one letter. Ranking order — not pixel position —
    assigns the letters, exactly like the frontier overlay.
    """
    from PIL import ImageDraw

    if len(candidates) > len(labels):
        raise ValueError(f"at most {len(labels)} grounder candidates can be marked")
    image = _load_rgb_frame(frame)
    if image is None:
        return None
    try:
        draw = ImageDraw.Draw(image)
        for index, detection in enumerate(candidates):
            x1, y1, x2, y2 = (float(v) for v in detection.box_xyxy)
            draw.rectangle([x1, y1, x2, y2], outline=_BOX_RGB, width=3)
            caption = f"{labels[index]} {detection.label} {float(detection.score):.2f}"
            _draw_caption(draw, x1, y1, caption)
        return _jpeg_data_url(image)
    except (AttributeError, OSError, TypeError, ValueError):
        return None


def _dilate_one_pixel(mask: np.ndarray) -> np.ndarray:
    """Return a 3x3 binary dilation without Pillow/SciPy runtime code.

    Isaac's extension search path can mix its pre-bundled Pillow C extension
    with the active environment's Python modules. ``ImageFilter.MaxFilter``
    then fails at runtime with an opaque C-API ``TypeError``. Dilation is eight
    NumPy OR operations, so keep this tiny geometry operation inside the array
    boundary we already control.
    """

    padded = np.pad(mask, 1, mode="constant", constant_values=False)
    height, width = mask.shape
    dilated = np.zeros_like(mask, dtype=bool)
    for row_offset in range(3):
        for col_offset in range(3):
            dilated |= padded[
                row_offset : row_offset + height,
                col_offset : col_offset + width,
            ]
    return dilated


def annotate_instance_mask(
    frame: Any,
    mask: Any,
    *,
    label: str,
    score: float,
) -> str | None:
    """Draw a translucent SAM3 mask on ``frame``; return a JPEG data URL, or None.

    ``None`` when the frame or mask cannot be rendered — the caller keeps its text
    result rather than failing the call, because the action metadata it already
    reported is still true without the picture.
    """
    from PIL import Image, ImageDraw

    image = _load_rgb_frame(frame)
    if image is None:
        return None
    try:
        mask_bool = np.asarray(mask).astype(bool, copy=False)
    except (AttributeError, TypeError, ValueError):
        return None
    if mask_bool.ndim != 2:
        return None
    height, width = mask_bool.shape
    if image.size != (width, height):
        return None
    if not mask_bool.any():
        return None

    try:
        base = image.convert("RGBA")
        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        overlay_arr = np.array(overlay)
        overlay_arr[mask_bool] = (*_MASK_FILL_RGB, _MASK_FILL_ALPHA)

        outline = _dilate_one_pixel(mask_bool) & ~mask_bool
        overlay_arr[outline] = (*_BOX_RGB, 255)
        overlay = Image.fromarray(overlay_arr)
        composed = Image.alpha_composite(base, overlay)

        ys, xs = np.nonzero(mask_bool)
        draw = ImageDraw.Draw(composed)
        caption = f"{label} {float(score):.2f}"
        _draw_caption(draw, float(xs.min()), float(ys.min()), caption)
        return _jpeg_data_url(composed.convert("RGB"))
    except (AttributeError, OSError, TypeError, ValueError):
        return None


__all__ = [
    "annotate_detections",
    "annotate_instance_mask",
    "rgb_frame_to_jpeg_data_url",
]
