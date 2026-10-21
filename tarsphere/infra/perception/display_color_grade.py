"""Fixed, offline colour adjustments for the presentation ego video only.

The recorded RGB-D observations, source video, spatial memory, UI colours and
physics clocks remain unchanged. A curve can reduce bright display values;
it cannot recover detail already clipped in the recorded 8-bit source.
"""

from __future__ import annotations


NEUTRAL_TONE_POINTS = (
    (0.0, 0.0),
    (0.25, 0.22),
    (0.5, 0.42),
    (0.75, 0.64),
    (1.0, 0.94),
)

NEUTRAL_FILTER = "curves=master='0/0 0.25/0.22 0.5/0.42 0.75/0.64 1/0.94'"

INDOOR_TONE_POINTS = (
    (0.0, 0.0),
    (0.15, 0.15),
    (0.35, 0.31),
    (0.55, 0.45),
    (0.75, 0.60),
    (0.9, 0.72),
    (1.0, 0.86),
)

INDOOR_FILTER = (
    "curves=master='0/0 0.15/0.15 0.35/0.31 0.55/0.45 0.75/0.60 0.9/0.72 1/0.86',"
    "colorchannelmixer=rr=1.018:gg=1:bb=0.976"
)


def resolve_display_grade(profile: str = "none") -> str:
    """Return a fixed FFmpeg RGB grade, with no time or geometry filters."""
    if profile == "none":
        return ""
    if profile == "neutral":
        return NEUTRAL_FILTER
    if profile == "indoor":
        return INDOOR_FILTER
    raise ValueError(f"Unknown presentation colour profile: {profile!r}")


def display_grade_metadata(profile: str = "none") -> dict:
    """Describe the adjustment separately from immutable recording evidence."""
    filter_expression = resolve_display_grade(profile)
    metadata = {
        "profile": profile,
        "ffmpeg_filter": filter_expression,
        "scope": "presentation ego before HUD composition",
        "fixed_for_entire_clip": True,
        "tone_points": [
            list(point)
            for point in (
                INDOOR_TONE_POINTS if profile == "indoor" else NEUTRAL_TONE_POINTS
            )
        ]
        if filter_expression
        else [],
        "source_video_modified": False,
        "model_observations_modified": False,
        "spatial_memory_or_ui_modified": False,
        "frame_clock_modified": False,
        "clipped_detail_recovered": False,
    }
    if profile == "indoor":
        metadata["rgb_channel_gains"] = {"red": 1.018, "green": 1.0, "blue": 0.976}
    return metadata
