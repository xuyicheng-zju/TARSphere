"""Continuous robot-eye video with a synchronized measured-trajectory BEV.

The recorder consumes synchronized frames supplied by the environment at the
native render cadence. It never requests a render, advances physics, or publishes its BEV
back to the evaluated Agent.  Both video channels and the manifest therefore
describe one shared sequence of simulator instants.
"""

from __future__ import annotations

import json
import math
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np

from tarsphere.assets import resolve_tarsphere_asset
from tarsphere.scene_graph.room_annotations import TopDownBevProjection


CONTINUOUS_EGO_BEV_SIZE_PX = 360
CONTINUOUS_EGO_LATEST_BEV_INTERVAL_FRAMES = 15


@dataclass(frozen=True)
class ContinuousEgoVideoArtifacts:
    """Completed synchronized video files and their physics-time index."""

    ego_video_path: Path
    bev_video_path: Path | None
    manifest_path: Path
    latest_bev_path: Path | None
    physics_steps: tuple[int, ...]
    fps: float
    resolution: tuple[int, int]


class _RawRgbVideoWriter:
    """One bounded-lifecycle ffmpeg raw-RGB encoder."""

    def __init__(
        self,
        *,
        output_path: Path,
        resolution: tuple[int, int],
        fps: float,
        crf: int = 18,
    ) -> None:
        width, height = (int(resolution[0]), int(resolution[1]))
        if width < 1 or height < 1 or fps <= 0.0:
            raise ValueError("Raw RGB video requires positive dimensions and fps.")
        if type(crf) is not int or not 0 <= crf <= 51:
            raise ValueError("Raw RGB video CRF must be an integer in [0, 51].")
        self.output_path = Path(output_path)
        self.resolution = (width, height)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self._log_path = self.output_path.with_suffix(".ffmpeg.log")
        self._log: BinaryIO = self._log_path.open("wb")
        self._process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                f"{width}x{height}",
                "-r",
                f"{float(fps):.8g}",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "4",
                "-preset",
                "veryfast",
                "-crf",
                str(crf),
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(self.output_path),
            ],
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=self._log,
        )
        if self._process.stdin is None:
            self._log.close()
            raise RuntimeError("ffmpeg did not expose its raw-video input pipe.")
        self._closed = False

    def write(self, rgb: np.ndarray) -> None:
        if self._closed:
            raise RuntimeError("Cannot append to a closed RGB video writer.")
        frame = np.asarray(rgb, dtype=np.uint8)
        width, height = self.resolution
        if frame.shape != (height, width, 3):
            raise ValueError(
                f"RGB video frame shape {frame.shape!r} does not match "
                f"{(height, width, 3)!r}."
            )
        try:
            assert self._process.stdin is not None
            self._process.stdin.write(np.ascontiguousarray(frame).tobytes())
        except BrokenPipeError as exc:
            raise RuntimeError(
                f"ffmpeg stopped while encoding {self.output_path}."
            ) from exc

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        assert self._process.stdin is not None
        self._process.stdin.close()
        # Concurrent native captures can take longer to drain and relocate the
        # MP4 index; allow finalization to finish before rejecting the artifact.
        return_code = self._process.wait(timeout=300.0)
        self._log.close()
        if return_code != 0 or not self.output_path.is_file():
            tail = ""
            if self._log_path.is_file():
                tail = self._log_path.read_text(encoding="utf-8", errors="replace")[
                    -600:
                ]
            raise RuntimeError(
                f"ffmpeg exited {return_code} for {self.output_path}: {tail}"
            )


class ContinuousEgoVideoRecorder:
    """Stream native ego RGB and a synchronized live trajectory map to MP4."""

    def __init__(
        self,
        *,
        capture_stride: int,
        physics_hz: float,
        resolution: tuple[int, int],
        include_bev: bool = True,
    ) -> None:
        if capture_stride < 1 or physics_hz <= 0.0:
            raise ValueError(
                "Continuous ego capture stride and physics Hz must be positive."
            )
        self.capture_stride = int(capture_stride)
        self.physics_hz = float(physics_hz)
        self.fps = self.physics_hz / self.capture_stride
        self.resolution = (int(resolution[0]), int(resolution[1]))
        # Diagnostic demos may include a map; Oracle needs only native ego RGB.
        self.include_bev = bool(include_bev)
        if min(self.resolution) < 1:
            raise ValueError("Continuous ego resolution must be positive.")
        self._active = False
        self._last_sample: tuple[int, np.ndarray, tuple[float, float], float] | None = (
            None
        )
        # Sample identity distinguishes a refreshed same-tick terminal view
        # (for example a zero-physics gaze change) from an already written RGB.
        self._last_written_sample: (
            tuple[int, np.ndarray, tuple[float, float], float] | None
        ) = None
        self._next_video_physics_step: int | None = None
        self._physics_steps: list[int] = []
        self._trail_xy: list[tuple[float, float]] = []
        self._ego_writer: _RawRgbVideoWriter | None = None
        self._bev_writer: _RawRgbVideoWriter | None = None
        self._manifest: Any = None
        self._manifest_path: Path | None = None
        self._ego_path: Path | None = None
        self._bev_path: Path | None = None
        self._latest_bev_path: Path | None = None
        self._bev_base: Any = None
        self._projection: TopDownBevProjection | None = None
        self._bev_crop: tuple[float, float, float, float] | None = None
        self._source_image_size: tuple[int, int] | None = None

    @property
    def active(self) -> bool:
        return self._active

    def start_episode(
        self,
        *,
        output_dir: Path,
        slug: str,
        scene_map: Mapping[str, Any],
        route_points_xy: Sequence[Sequence[float]],
    ) -> None:
        if self._active:
            raise RuntimeError(
                "Continuous ego recorder must finish the active episode first."
            )
        rgb_bev = str(scene_map.get("rgb_bev") or "").strip()
        projection_path = str(scene_map.get("projection") or "").strip()
        if self.include_bev and (not rgb_bev or not projection_path):
            raise RuntimeError(
                "Continuous ego trajectory BEV requires scene_map.rgb_bev "
                "and scene_map.projection."
            )
        from PIL import Image

        if self.include_bev:
            base_path = resolve_tarsphere_asset(rgb_bev).resolve()
            meta_path = resolve_tarsphere_asset(projection_path).resolve()
            if not base_path.is_file() or not meta_path.is_file():
                raise FileNotFoundError(
                    f"Continuous ego BEV assets are missing: {base_path}, {meta_path}"
                )
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            self._projection = TopDownBevProjection(
                bounds_min_xy=tuple(float(v) for v in meta["bev_bounds_min_xy"]),
                bounds_max_xy=tuple(float(v) for v in meta["bev_bounds_max_xy"]),
                resolution=int(meta["resolution"]),
            )
            base = Image.open(base_path).convert("RGB")
            self._source_image_size = base.size
            self._bev_crop = self._route_crop_pixels(
                route_points_xy, image_size=base.size
            )
            self._bev_base = base.crop(self._bev_crop).resize(
                (CONTINUOUS_EGO_BEV_SIZE_PX, CONTINUOUS_EGO_BEV_SIZE_PX),
                Image.Resampling.LANCZOS,
            )
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        self._ego_path = output_dir / f"{slug}_ego.mp4"
        self._bev_path = (
            output_dir / f"{slug}_bev_trajectory.mp4" if self.include_bev else None
        )
        self._manifest_path = output_dir / f"{slug}_ego_frames.jsonl"
        self._latest_bev_path = (
            output_dir / f"{slug}_bev_latest.png" if self.include_bev else None
        )
        for path in (
            self._ego_path,
            self._bev_path,
            self._manifest_path,
            self._latest_bev_path,
        ):
            if path is not None and path.exists():
                raise FileExistsError(f"Continuous ego output already exists: {path}")
        self._manifest = self._manifest_path.open("w", encoding="utf-8")
        self._last_sample = None
        self._last_written_sample = None
        self._next_video_physics_step = None
        self._physics_steps = []
        self._trail_xy = []
        self._ego_writer = _RawRgbVideoWriter(
            output_path=self._ego_path,
            resolution=self.resolution,
            fps=self.fps,
        )
        self._bev_writer = (
            _RawRgbVideoWriter(
                output_path=self._bev_path,
                resolution=(CONTINUOUS_EGO_BEV_SIZE_PX, CONTINUOUS_EGO_BEV_SIZE_PX),
                fps=self.fps,
            )
            if self.include_bev
            else None
        )
        self._active = True

    def capture(
        self,
        *,
        physics_step: int,
        rgba: Any,
        position: Sequence[float],
        yaw_rad: float,
    ) -> bool:
        if not self._active:
            return False
        step = int(physics_step)
        if self._last_sample is not None and step < self._last_sample[0]:
            raise ValueError(
                "Continuous ego physics clock moved backwards within an episode."
            )
        frame = np.asarray(rgba.cpu() if hasattr(rgba, "cpu") else rgba)
        if frame.size == 0:
            raise RuntimeError("Continuous ego camera returned an empty RGB frame.")
        if frame.ndim == 4:
            frame = frame[0]
        if frame.ndim != 3 or frame.shape[2] < 3:
            raise RuntimeError(
                f"Continuous ego camera returned invalid RGBA shape {frame.shape!r}."
            )
        rgb = np.asarray(frame[:, :, :3], dtype=np.uint8)
        expected_width, expected_height = self.resolution
        if rgb.shape != (expected_height, expected_width, 3):
            raise RuntimeError(
                "Continuous ego RenderProduct returned "
                f"{rgb.shape!r}, expected "
                f"{(expected_height, expected_width, 3)!r}."
            )
        x, y = float(position[0]), float(position[1])
        yaw = float(yaw_rad)
        if not all(math.isfinite(v) for v in (x, y, yaw)):
            raise RuntimeError(
                "Continuous ego recorder received a non-finite robot pose."
            )
        sample = (step, rgb.copy(), (x, y), yaw)
        if self._next_video_physics_step is None:
            self._next_video_physics_step = step
        written = False
        while self._next_video_physics_step <= step:
            # Irregular decision-boundary renders must not compress simulation
            # time or show a future sample. Hold the last measured frame.
            selected = (
                sample
                if self._next_video_physics_step == step or self._last_sample is None
                else self._last_sample
            )
            self._write_sample(selected, self._next_video_physics_step)
            self._next_video_physics_step += self.capture_stride
            written = True
        self._last_sample = sample
        return written

    def _write_sample(
        self,
        sample: tuple[int, np.ndarray, tuple[float, float], float],
        video_physics_step: int,
        *,
        terminal_hold: bool = False,
    ) -> None:
        step, rgb, (x, y), yaw = sample
        assert self._ego_writer is not None
        self._ego_writer.write(rgb)
        if self._bev_writer is not None:
            if not self._trail_xy or math.dist(self._trail_xy[-1], (x, y)) >= 0.02:
                self._trail_xy.append((x, y))
            bev = self._render_bev((x, y), yaw)
            self._bev_writer.write(bev)
        frame_index = len(self._physics_steps)
        assert self._manifest is not None
        self._manifest.write(
            json.dumps(
                {
                    "frame_index": frame_index,
                    "physics_step": step,
                    "video_physics_step": video_physics_step,
                    "position_xy": [x, y],
                    "yaw_rad": yaw,
                    "terminal_hold": terminal_hold,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self._manifest.flush()
        self._physics_steps.append(step)
        self._last_written_sample = sample
        if (
            self._bev_writer is not None
            and frame_index % CONTINUOUS_EGO_LATEST_BEV_INTERVAL_FRAMES == 0
        ):
            self._write_latest_bev(bev)

    def finish_episode(self) -> ContinuousEgoVideoArtifacts | None:
        if not self._active:
            return None
        errors: list[Exception] = []
        # A final observation can fall between regular output grid points.
        # Hold it on the next point (at most one frame of padding), keeping
        # its actual capture tick rather than pretending physics advanced.
        try:
            if (
                self._last_sample is not None
                and self._last_written_sample is not self._last_sample
            ):
                assert self._next_video_physics_step is not None
                self._write_sample(
                    self._last_sample, self._next_video_physics_step, terminal_hold=True
                )
        except Exception as exc:  # Close all streams even if terminal encoding fails.
            errors.append(exc)
        self._active = False
        for writer in (self._ego_writer, self._bev_writer):
            if writer is not None:
                try:
                    writer.close()
                except Exception as exc:  # noqa: BLE001
                    errors.append(exc)
        if self._manifest is not None:
            try:
                self._manifest.close()
            except Exception as exc:
                errors.append(exc)
        if errors:
            raise RuntimeError(
                "Continuous ego video finalization failed: "
                + "; ".join(str(error) for error in errors)
            ) from errors[0]
        if not self._physics_steps or self._ego_writer is None:
            raise RuntimeError("Continuous ego recording produced no frames.")
        assert self._ego_path is not None
        assert self._manifest_path is not None
        return ContinuousEgoVideoArtifacts(
            ego_video_path=self._ego_path,
            bev_video_path=self._bev_path,
            manifest_path=self._manifest_path,
            latest_bev_path=self._latest_bev_path,
            physics_steps=tuple(self._physics_steps),
            fps=self.fps,
            resolution=self._ego_writer.resolution,
        )

    def _route_crop_pixels(
        self,
        route_points_xy: Sequence[Sequence[float]],
        *,
        image_size: tuple[int, int],
    ) -> tuple[float, float, float, float]:
        assert self._projection is not None
        width, height = image_size
        if not route_points_xy:
            return 0.0, 0.0, float(width), float(height)
        pixels = np.asarray(
            [
                self._projection.world_to_pixel_float(
                    (float(point[0]), float(point[1]))
                )
                for point in route_points_xy
            ],
            dtype=np.float64,
        )
        pixels[:, 0] *= float(width) / float(self._projection.resolution)
        pixels[:, 1] *= float(height) / float(self._projection.resolution)
        lower = pixels.min(axis=0)
        upper = pixels.max(axis=0)
        metres_per_native_pixel = self._projection.span_m / max(
            1, self._projection.resolution - 1
        )
        margin_px = 1.0 / metres_per_native_pixel
        margin_px *= float(width) / float(self._projection.resolution)
        lower -= margin_px
        upper += margin_px
        span = max(float(upper[0] - lower[0]), float(upper[1] - lower[1]), 2.0)
        centre = 0.5 * (lower + upper)
        left = max(0.0, min(float(width) - span, float(centre[0] - span / 2.0)))
        top = max(0.0, min(float(height) - span, float(centre[1] - span / 2.0)))
        right = min(float(width), left + span)
        bottom = min(float(height), top + span)
        return left, top, right, bottom

    def _world_to_panel(self, xy: tuple[float, float]) -> tuple[float, float]:
        assert self._projection is not None
        assert self._bev_crop is not None
        assert self._source_image_size is not None
        native_x, native_y = self._projection.world_to_pixel_float(xy)
        left, top, right, bottom = self._bev_crop
        source_width, source_height = self._source_image_size
        image_x = native_x * source_width / self._projection.resolution
        image_y = native_y * source_height / self._projection.resolution
        return (
            (image_x - left) * CONTINUOUS_EGO_BEV_SIZE_PX / (right - left),
            (image_y - top) * CONTINUOUS_EGO_BEV_SIZE_PX / (bottom - top),
        )

    def _render_bev(self, pose_xy: tuple[float, float], yaw_rad: float) -> np.ndarray:
        from PIL import ImageDraw

        assert self._bev_base is not None
        image = self._bev_base.copy()
        draw = ImageDraw.Draw(image, "RGBA")
        if len(self._trail_xy) >= 2:
            draw.line(
                [self._world_to_panel(point) for point in self._trail_xy],
                fill=(235, 35, 205, 245),
                width=5,
                joint="curve",
            )
        cx, cy = self._world_to_panel(pose_xy)
        heading = np.asarray([math.cos(yaw_rad), -math.sin(yaw_rad)])
        left = np.asarray([-heading[1], heading[0]])
        tip = np.asarray([cx, cy]) + 18.0 * heading
        rear = np.asarray([cx, cy]) - 8.0 * heading
        polygon = [
            tuple(tip),
            tuple(rear + 7.0 * left),
            tuple(rear - 7.0 * left),
        ]
        draw.ellipse(
            [cx - 9, cy - 9, cx + 9, cy + 9],
            fill=(255, 45, 45, 245),
            outline=(255, 255, 255, 255),
            width=3,
        )
        draw.polygon(polygon, fill=(255, 45, 45, 255))
        return np.asarray(image, dtype=np.uint8)

    def _write_latest_bev(self, bev: np.ndarray) -> None:
        from PIL import Image

        assert self._latest_bev_path is not None
        temporary = self._latest_bev_path.with_suffix(".tmp")
        Image.fromarray(bev, mode="RGB").save(temporary, format="PNG")
        temporary.replace(self._latest_bev_path)


__all__ = [
    "CONTINUOUS_EGO_BEV_SIZE_PX",
    "ContinuousEgoVideoArtifacts",
    "ContinuousEgoVideoRecorder",
]
