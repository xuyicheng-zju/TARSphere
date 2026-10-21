"""The original full-height ego view, map sidebar, and dark floating overlays."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

from tarsphere.infra.perception.model_output_overlay import (
    INTERACTION_MIN_DISPLAY_SECONDS,
    ffmpeg_subtitles_filter,
    write_model_output_ass,
    write_online_map_video,
)


@dataclass(frozen=True)
class PresentationLayout:
    """Keep the source camera aspect ratio and the original narrow sidebar."""

    height: int = 720
    source_resolution: tuple[int, int] = (640, 480)

    def __post_init__(self):
        if self.height < 720 or self.height % 2:
            raise ValueError("Presentation height must be even and at least 720")
        if min(self.source_resolution) <= 0:
            raise ValueError("Source resolution must be positive")

    @property
    def ego_width(self):
        width, height = self.source_resolution
        return max(2, round(self.height * width / height) // 2 * 2)

    @property
    def side_width(self):
        return max(320, round(self.height * 4 / 9) // 2 * 2)

    @property
    def margin(self):
        return max(16, round(self.height * 16 / 720))

    @property
    def resolution(self):
        return self.ego_width + self.side_width, self.height

    def rect(self, name):
        map_size = self.side_width - 2 * self.margin
        map_height = round(self.height * 248 / 720)
        panel_y = round(self.height * 276 / 720)
        return {
            "ego": (0, 0, self.ego_width, self.height),
            "map": (self.ego_width + self.margin, self.margin, map_size, map_height),
            "decision": (
                self.ego_width,
                panel_y,
                self.side_width,
                self.height - panel_y,
            ),
            "plan": (0, 0, self.ego_width, self.height),
            "interaction": (0, 0, self.ego_width, self.height),
        }[name]


def compose_presentation(
    *,
    ego_path,
    map_path,
    ass_path,
    output_path,
    layout=PresentationLayout(),
    ffmpeg="ffmpeg",
    duration=None,
):
    """Preserve the full camera view and composite the original sidebar layout."""
    width, height = layout.resolution
    mx, my, mw, mh = layout.rect("map")
    tail = INTERACTION_MIN_DISPLAY_SECONDS
    logo = (
        Path(__file__).resolve().parents[3] / "assets/presentation/tarsphere-logo.png"
    )
    logo_width = round(176 * height / 720)
    logo_x = layout.ego_width + layout.margin
    logo_y = round(height - 100 * height / 720)
    filters = (
        f"[0:v]tpad=stop_mode=clone:stop_duration={tail:g},"
        f"scale={layout.ego_width}:{height}:flags=lanczos,setsar=1,"
        f"pad={width}:{height}:0:0:color=0x15171E[canvas];"
        f"[1:v]tpad=stop_mode=clone:stop_duration={tail:g},"
        f"scale={mw}:{mh}:flags=lanczos[map];"
        f"[canvas][map]overlay={mx}:{my}:shortest=1,"
        f"{ffmpeg_subtitles_filter(Path(ass_path))}[hud];"
        f"[hud]drawbox=x={logo_x}:y={logo_y}:"
        f"w={round(192 * height / 720)}:h={round(60 * height / 720)}:color=0xF2F0F5:t=fill[brand];"
        f"[2:v]crop=1805:495:116:149,scale={logo_width}:-1:flags=lanczos[logo];"
        f"[brand][logo]overlay={logo_x + round(8 * height / 720)}:{logo_y + round(6 * height / 720)}:eof_action=repeat[out]"
    )
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(ego_path),
        "-i",
        str(map_path),
        "-i",
        str(logo),
        "-filter_complex",
        filters,
        "-map",
        "[out]",
        "-an",
    ]
    if duration is not None:
        command += ["-t", str(duration)]
    command += [
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(output_path),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode or not Path(output_path).is_file():
        raise RuntimeError("Ego overlay render failed: " + result.stderr[-1600:])
    return Path(output_path)


def _write_missing_map(*, output_path, physics_steps, fps, resolution):
    """Explicit preview placeholder for old footage without archived maps."""
    import numpy as np
    from PIL import Image, ImageDraw, ImageFont
    from tarsphere.infra.perception.continuous_ego_video import _RawRgbVideoWriter

    width, height = resolution
    image = Image.new("RGB", resolution, "#191B23")
    draw = ImageDraw.Draw(image)
    size = max(12, round(width / 23))
    font = ImageFont.load_default(size=size)
    draw.rounded_rectangle(
        (0, 0, width - 1, height - 1),
        radius=round(width / 30),
        outline="#3B3449",
        width=max(1, round(width / 288)),
    )
    draw.text(
        (width / 2, height / 2 - size),
        "SEMANTIC SPATIAL MEMORY",
        anchor="mm",
        font=font,
        fill="#BC9ED3",
    )
    draw.text(
        (width / 2, height / 2 + size),
        "Not archived in this recording",
        anchor="mm",
        font=font,
        fill="#A8A7B2",
    )
    writer = _RawRgbVideoWriter(output_path=output_path, resolution=resolution, fps=fps)
    try:
        pixels = np.asarray(image)
        for _ in physics_steps:
            writer.write(pixels)
    finally:
        writer.close()
    return output_path


def render_ego_presentation(
    *,
    output_dir,
    slug,
    artifacts,
    decision_trace,
    heartbeats,
    online_map_dir=None,
    height=None,
    ffmpeg="ffmpeg",
    duration=None,
    allow_missing_evidence=False,
    task_title=None,
    source_label=None,
    model_input_dir=None,
):
    output_dir = Path(output_dir)
    video_dir = output_dir / "videos"
    video_dir.mkdir(parents=True, exist_ok=True)
    root = Path(online_map_dir) if online_map_dir else output_dir / "online_maps" / slug
    output_height = height if height is not None else max(720, artifacts.resolution[1])
    layout = PresentationLayout(output_height, artifacts.resolution)
    map_path = video_dir / f"{slug}_online_map.mp4"
    rgbd_root = root / "presentation" / "rgbd"
    # Prefer the paper-style two-layer view when actual RGB-D evidence exists.
    if any(rgbd_root.glob("*.npz")):
        from tarsphere.infra.perception.presentation_rgbd import write_layered_map_video

        write_layered_map_video(
            output_path=map_path,
            snapshot_dir=rgbd_root,
            physics_steps=artifacts.physics_steps,
            observer_fps=artifacts.fps,
            resolution=layout.rect("map")[2:],
        )
    elif allow_missing_evidence and not any(
        (root / "open_frontier").glob(
            "observation_*/frontier_step_*_exploration_memory.png"
        )
    ):
        _write_missing_map(
            output_path=map_path,
            physics_steps=artifacts.physics_steps,
            fps=artifacts.fps,
            resolution=layout.rect("map")[2:],
        )
    else:
        write_online_map_video(
            output_path=map_path,
            online_map_dir=root,
            physics_steps=artifacts.physics_steps,
            observer_fps=artifacts.fps,
            trace=decision_trace,
            heartbeats=heartbeats,
            resolution=layout.rect("map")[2:],
        )
    from tarsphere.infra.perception.presentation_radar import load_trace_lidar

    input_root = (
        Path(model_input_dir)
        if model_input_dir
        else output_dir / f"{slug}.model_output.inputs"
    )
    lidar = load_trace_lidar(decision_trace, input_root) if input_root.is_dir() else []
    ass = write_model_output_ass(
        output_path=video_dir / f"{slug}_ego_model_output.ass",
        physics_steps=artifacts.physics_steps,
        observer_fps=artifacts.fps,
        trace=decision_trace,
        heartbeats=heartbeats,
        resolution=layout.resolution,
        panel_region=layout.rect("decision"),
        plan_region=layout.rect("plan"),
        interaction_region=layout.rect("interaction"),
        task_title=task_title,
        source_label=source_label,
        lidar_by_trace=lidar,
    )
    if ass is None:
        raise RuntimeError("Decision-panel video requires model output trace")
    out = compose_presentation(
        ego_path=artifacts.ego_video_path,
        map_path=map_path,
        ass_path=ass,
        output_path=video_dir / f"{slug}_ego_decision_panel.mp4",
        layout=layout,
        ffmpeg=ffmpeg,
        duration=duration,
    )
    return {
        "ego": str(artifacts.ego_video_path),
        "online_map": str(map_path),
        "frame_manifest": str(artifacts.manifest_path),
        "model_output_ass": str(ass),
        "ego_decision_panel": str(out),
    }
