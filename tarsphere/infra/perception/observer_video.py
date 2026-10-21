"""Encode native observer captures on the simulation clock, preserving gaps."""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from tarsphere.infra.control.physical_contracts import SIMULATION_PHYSICS_HZ
from tarsphere.infra.perception.continuous_ego_video import _RawRgbVideoWriter


class ObserverVideoStream:
    """Encode a live observer with one buffered RGB sample, without a PNG spool."""

    def __init__(self, *, output_path: Path, resolution: tuple[int, int], stride: int):
        if stride < 1:
            raise ValueError("Observer video stride must be positive.")
        self.path = Path(output_path)
        self.manifest_path = self.path.with_suffix(".frames.jsonl")
        # One native first-frame still permits inspection before MP4 finalization.
        self.initial_frame_path = self.path.with_suffix(".first.png")
        if (
            self.path.exists()
            or self.manifest_path.exists()
            or self.initial_frame_path.exists()
        ):
            raise FileExistsError(f"Observer capture already exists: {self.path}")
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.resolution, self.stride = resolution, stride
        self.fps = SIMULATION_PHYSICS_HZ / stride
        self.writer = _RawRgbVideoWriter(
            output_path=self.path, resolution=resolution, fps=self.fps
        )
        self.manifest = self.manifest_path.open("x", encoding="utf-8")
        self.last = None
        self.first_tick = self.next_tick = self.last_written_source = None
        self.frame_count = self.capture_count = 0
        self.closed = False

    def _write(self, sample, tick):
        source_tick, rgb, camera = sample
        self.writer.write(rgb)
        row = {
            "frame_index": self.frame_count,
            "physics_step": source_tick,
            "video_physics_step": tick,
        }
        if camera is not None:
            row["camera"] = camera
        self.manifest.write(json.dumps(row, allow_nan=False) + "\n")
        self.manifest.flush()
        self.frame_count += 1
        self.last_written_source = source_tick

    def capture(
        self, rgb: np.ndarray, physics_step: int, *, camera: dict | None = None
    ) -> None:
        """Camera evidence belongs to the RGB source; holds retain that evidence.

        Native third-person recorders supply authored USD pose in world metres.
        Offline PNG encoding and articulation-attached ego have no such pose.
        """
        if (
            self.closed
            or physics_step < 0
            or (self.last is not None and physics_step <= self.last[0])
        ):
            raise ValueError(
                "Observer stream requires increasing nonnegative capture ticks while open."
            )
        width, height = self.resolution
        if rgb.shape != (height, width, 3) or rgb.dtype != np.uint8:
            raise ValueError("Observer stream requires native-resolution uint8 RGB.")
        sample = (physics_step, rgb.copy(), deepcopy(camera))
        if self.next_tick is None:
            # Publish only a complete still; the remote reviewer may read it
            # immediately while the long video is still being recorded.
            pending_first_frame = self.initial_frame_path.with_suffix(".png.tmp")
            Image.fromarray(rgb).save(pending_first_frame, format="PNG")
            pending_first_frame.replace(self.initial_frame_path)
            self.first_tick = self.next_tick = physics_step
        while self.next_tick <= physics_step:
            self._write(
                sample if self.next_tick == physics_step else self.last, self.next_tick
            )
            self.next_tick += self.stride
        self.last = sample
        self.capture_count += 1

    def close(self) -> dict:
        if self.closed:
            raise RuntimeError("Observer stream was already closed.")
        self.closed = True
        try:
            if self.last is None:
                raise RuntimeError("Observer stream produced no frames.")
            if self.last_written_source != self.last[0]:
                self._write(self.last, self.next_tick)
        finally:
            self.manifest.close()
            self.writer.close()
        return {
            "path": str(self.path),
            "frame_manifest": str(self.manifest_path),
            "initial_frame_path": str(self.initial_frame_path),
            "resolution": list(self.resolution),
            "fps": float(self.fps),
            "frame_count": self.frame_count,
            "capture_count": self.capture_count,
            "first_physics_step": self.first_tick,
            "last_physics_step": self.last[0],
            "policy_visible": False,
        }


def observer_frame_grid(steps: list[int], stride: int) -> list[tuple[int, int]]:
    """Return (source index, video tick); never show a capture before its tick.

    Gaps hold the latest available source. The last sample receives one frame
    on the next grid tick when it falls between ticks, matching the ego writer.
    """
    if not steps or stride < 1 or steps[0] < 0:
        raise ValueError(
            "Observer encoding requires nonnegative ticks and a positive stride."
        )
    if any(b <= a for a, b in zip(steps, steps[1:])):
        raise ValueError("Observer capture ticks must be strictly increasing.")
    end = steps[0] + math.ceil((steps[-1] - steps[0]) / stride) * stride
    source = 0
    grid = []
    for tick in range(steps[0], end + 1, stride):
        while source + 1 < len(steps) and steps[source + 1] <= tick:
            source += 1
        grid.append((source, tick))
    return grid


def encode_room_observer_video(
    *,
    frame_dir: Path,
    output_path: Path,
    fps: float,
    resolution: tuple[int, int],
) -> dict:
    """Encode without changing simulator state; preserve source and video ticks."""
    frames = sorted(Path(frame_dir).glob("step_*.png"))
    if not frames:
        raise RuntimeError(f"No room-observer frames found: {frame_dir}")
    steps = [int(path.stem.rsplit("_", 1)[-1]) for path in frames]
    stride_value = SIMULATION_PHYSICS_HZ / fps if fps > 0 else 0
    stride = round(stride_value)
    if stride < 1 or not math.isclose(stride, stride_value, abs_tol=1e-8):
        raise ValueError("Observer fps must divide the simulator physics rate exactly.")
    observer_frame_grid(
        steps, stride
    )  # Validate the full source before creating outputs.
    writer = ObserverVideoStream(
        output_path=output_path, resolution=resolution, stride=stride
    )
    try:
        for path, step in zip(frames, steps):
            with Image.open(path) as source:
                writer.capture(np.array(source.convert("RGB")), step)
    finally:
        result = writer.close()
    return result
