"""Harness adapter for current-frame visual-frontier selection."""

from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
import os
from pathlib import Path
import time
from typing import Any

import numpy as np
from PIL import Image

from tarsphere.harness.memory.semantic import SemanticEntityRecord
from tarsphere.geometry import fov_to_intrinsics
from tarsphere.infra.navigation.open_frontier.artifacts import (
    save_frontier_frame,
)
from tarsphere.infra.navigation.open_frontier.contracts import (
    CameraIntrinsics,
    DisplayFrontier,
    FrontierFrame,
)
from tarsphere.infra.navigation.open_frontier.frontier_net import (
    FrontierNetConfig,
    FrontierNetPredictor,
)
from tarsphere.infra.navigation.open_frontier.exploration_memory import (
    render_exploration_memory,
)
from tarsphere.infra.navigation.open_frontier.pipeline import (
    OpenFrontierPipeline,
)
from tarsphere.infra.navigation.open_frontier.overlay import (
    render_frontier_overlay,
)


@dataclass(frozen=True, slots=True)
class OpenFrontierHarnessSettings:
    checkpoint_path: str
    device: str = "cuda:0"

    def __post_init__(self) -> None:
        if not str(self.device).startswith("cuda:"):
            raise ValueError(
                "The harness requires FrontierNet GPU inference on an explicit "
                f"CUDA device; got {self.device!r}."
            )

    @classmethod
    def from_environment(cls) -> "OpenFrontierHarnessSettings":
        checkpoint = os.environ.get("TARSPHERE_FRONTIERNET_CHECKPOINT", "").strip()
        if not checkpoint:
            raise RuntimeError(
                "navigation_tool=open_frontier requires "
                "TARSPHERE_FRONTIERNET_CHECKPOINT."
            )
        return cls(
            checkpoint_path=checkpoint,
            device=(
                os.environ.get("TARSPHERE_FRONTIERNET_DEVICE", "cuda:0").strip()
                or "cuda:0"
            ),
        )


class OpenFrontierHarnessBackend:
    """Own one episode-scoped FrontierNet pipeline and latest A/B/C frame."""

    def __init__(self, settings: OpenFrontierHarnessSettings) -> None:
        self.settings = settings
        self.pipeline = OpenFrontierPipeline(
            predictor=FrontierNetPredictor(
                FrontierNetConfig(
                    checkpoint_path=settings.checkpoint_path,
                    device=settings.device,
                )
            )
        )
        print(
            "[OPEN-FRONTIER] FrontierNet inference device="
            f"{self.pipeline.predictor.device}",
            flush=True,
        )
        self.latest_frame: FrontierFrame | None = None
        self._observation_index = 0
        self.last_processing_seconds: float | None = None

    def reset(self) -> None:
        self.pipeline.reset()
        self.latest_frame = None
        self._observation_index = 0
        self.last_processing_seconds = None

    def process_observation(
        self,
        *,
        observation: Any,
        action_step: int,
        occupancy_grid: Any,
        record_dir: str | None,
        record_debug_artifacts: bool = False,
        saved_entities: tuple[SemanticEntityRecord, ...] = (),
    ) -> tuple[FrontierFrame, np.ndarray, np.ndarray, dict[str, str]]:
        started_at = time.perf_counter()
        ego = getattr(observation, "views", {}).get("ego")
        pose = getattr(observation, "pose", None)
        if ego is None or ego.rgb is None or ego.depth is None:
            raise ValueError("OpenFrontier requires synchronized current ego RGB-D.")
        if not isinstance(pose, dict) or pose.get("position") is None:
            raise ValueError("OpenFrontier requires the current robot pose.")
        params = ego.camera_params
        if not isinstance(params, dict):
            raise ValueError("OpenFrontier requires calibrated camera parameters.")
        rgb = np.asarray(ego.rgb)
        depth = np.asarray(ego.depth, dtype=np.float32)
        if rgb.ndim != 3 or depth.shape != rgb.shape[:2]:
            raise ValueError(
                f"OpenFrontier RGB-D shape mismatch: {rgb.shape} vs {depth.shape}."
            )
        world_T_camera = np.asarray(params.get("world_T_cam"), dtype=np.float64)
        if world_T_camera.shape != (4, 4):
            raise ValueError("OpenFrontier world_T_cam must be 4x4.")
        height, width = depth.shape
        if params.get("K") is not None:
            matrix = np.asarray(params["K"], dtype=np.float64)
        else:
            matrix = fov_to_intrinsics(float(params["fov_deg"]), width, height)
        if matrix.shape != (3, 3):
            raise ValueError("OpenFrontier camera intrinsics must be 3x3.")
        focal_x = float(matrix[0, 0])
        if not np.isfinite(focal_x) or focal_x <= 0.0:
            raise ValueError("OpenFrontier camera focal length fx must be positive.")
        horizontal_fov_deg = math.degrees(2.0 * math.atan2(float(width), 2.0 * focal_x))
        intrinsics = CameraIntrinsics(
            width=width,
            height=height,
            fx=float(matrix[0, 0]),
            fy=float(matrix[1, 1]),
            cx=float(matrix[0, 2]),
            cy=float(matrix[1, 2]),
        )
        position = np.asarray(pose["position"], dtype=np.float64).reshape(-1)
        if position.size < 2 or not np.all(np.isfinite(position[:2])):
            raise ValueError("OpenFrontier robot XY must be finite.")
        yaw_rad = pose.get("yaw_rad")
        if yaw_rad is None or not np.isfinite(float(yaw_rad)):
            raise ValueError("OpenFrontier robot yaw_rad must be finite.")
        frame = self.pipeline.observe(
            observation_step=int(action_step),
            rgb=rgb,
            depth_m=depth,
            intrinsics=intrinsics,
            world_T_camera=world_T_camera,
            visible_human_centers_xy=tuple(observation.visible_humans.values()),
        )
        pipeline_seconds = time.perf_counter() - started_at
        self.latest_frame = frame
        ego_overlay = _render_original_frame_overlay(
            original_rgb=rgb,
            original_intrinsics=intrinsics,
            frame=frame,
        )
        exploration_memory = render_exploration_memory(
            occupancy_grid=occupancy_grid,
            robot_world_xy=position[:2],
            robot_yaw_rad=float(yaw_rad),
            camera_horizontal_fov_deg=horizontal_fov_deg,
            displayed_frontiers=frame.displayed_frontiers,
            saved_entities=saved_entities,
        )
        overlay_seconds = time.perf_counter() - started_at - pipeline_seconds
        paths: dict[str, str] = {}
        if record_dir:
            root = (
                Path(record_dir)
                / "open_frontier"
                / f"observation_{self._observation_index:04d}"
            )
            self._observation_index += 1
            root.mkdir(parents=True, exist_ok=True)
            if record_debug_artifacts:
                paths = save_frontier_frame(frame, root)
            ego_overlay_path = root / (
                f"frontier_step_{int(action_step):04d}_ego_overlay.png"
            )
            Image.fromarray(ego_overlay, mode="RGB").save(ego_overlay_path)
            paths["ego_overlay"] = str(ego_overlay_path)
            exploration_memory_path = root / (
                f"frontier_step_{int(action_step):04d}_exploration_memory.png"
            )
            Image.fromarray(exploration_memory, mode="RGB").save(
                exploration_memory_path
            )
            paths["exploration_memory"] = str(exploration_memory_path)
            if record_debug_artifacts:
                total_seconds = time.perf_counter() - started_at
                timing_path = root / (
                    f"frontier_step_{int(action_step):04d}_timing.json"
                )
                timing_path.write_text(
                    json.dumps(
                        {
                            "action_step": int(action_step),
                            "pipeline_seconds": pipeline_seconds,
                            "native_overlay_seconds": overlay_seconds,
                            "artifact_io_seconds": max(
                                0.0,
                                total_seconds - pipeline_seconds - overlay_seconds,
                            ),
                            "total_seconds": total_seconds,
                        },
                        indent=2,
                        sort_keys=True,
                    ),
                    encoding="utf-8",
                )
                paths["timing"] = str(timing_path)
        self.last_processing_seconds = time.perf_counter() - started_at
        return frame, ego_overlay, exploration_memory, paths

    def latest_frontier(
        self,
        *,
        label: str,
        action_step: int,
    ) -> DisplayFrontier:
        """Resolve a label from the unchanged current observation."""

        frame = self.latest_frame
        if frame is None:
            raise ValueError("No current frontier observation is available.")
        if frame.observation_step != int(action_step):
            raise ValueError(
                "The A/B/C frontier labels are stale after physical motion; "
                "use a label from the newly refreshed current RGB frame."
            )
        return frame.frontier_for_label(str(label).strip().upper())


def _render_original_frame_overlay(
    *,
    original_rgb: np.ndarray,
    original_intrinsics: CameraIntrinsics,
    frame: FrontierFrame,
) -> np.ndarray:
    """Inverse-warp only selectable A/B/C marks onto native ego RGB.

    The dense FrontierNet support mask remains a private diagnostic artifact;
    only the three current information-gain candidates are selectable.
    """

    processed = frame.processed_intrinsics
    scale_x = processed.fx / original_intrinsics.fx
    scale_y = processed.fy / original_intrinsics.fy
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise ValueError("Frontier crop has invalid camera scale.")
    crop_left = int(round(original_intrinsics.cx * scale_x - processed.cx))
    crop_top = int(round(original_intrinsics.cy * scale_y - processed.cy))
    remapped = tuple(
        replace(
            candidate,
            pixel_xy=(
                (float(candidate.pixel_xy[0]) + crop_left) / scale_x,
                (float(candidate.pixel_xy[1]) + crop_top) / scale_y,
            ),
        )
        for candidate in frame.displayed_frontiers
    )
    return render_frontier_overlay(
        np.asarray(original_rgb, dtype=np.uint8),
        np.zeros(original_rgb.shape[:2], dtype=bool),
        remapped,
    )


__all__ = [
    "OpenFrontierHarnessBackend",
    "OpenFrontierHarnessSettings",
]
