"""Ground one named object from the robot's current synchronized RGB-D view.

This module is the shared perception boundary for ``locate_object``,
``remember_entity``, and ``grasp_object``. SAM3 first finds an instance in the
current RGB frame; metric geometry uses only its synchronized depth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tarsphere.harness.perception.contracts import (
    Detection2D,
    PerceptionInput,
)
from tarsphere.geometry import (
    backproject_mask,
    fov_to_intrinsics,
    transform_to_world,
)


class ObjectNotVisibleError(RuntimeError):
    """SAM3 found no instance matching the requested phrase in the ego RGB."""


class ObjectGeometryUnavailableError(RuntimeError):
    """The current target mask cannot be lifted with one stamped capture."""


class EgoFrameUnavailableError(ObjectGeometryUnavailableError):
    """The current decision has no usable ego RGB observation."""


@dataclass(frozen=True)
class VisibleObjectGrounding:
    """One SAM3 instance and its same-capture metric camera geometry."""

    # Highest-confidence SAM3 instance for the requested phrase.
    detection: Detection2D
    # Current metric depth raster aligned pixel-for-pixel with the RGB frame.
    depth_m: np.ndarray
    # Boolean SAM3 mask aligned with ``depth_m``.
    target_mask: np.ndarray
    # Pinhole camera intrinsics for the same-capture RGB-D frame.
    camera_intrinsics: np.ndarray
    # Rigid transform from optical camera coordinates to world coordinates.
    world_T_camera: np.ndarray
    # Median target-mask point in optical camera metres: right, down, forward.
    target_center_camera_m: tuple[float, float, float]
    # Median 3-D point of valid target-mask pixels in world metres.
    target_center_world_m: tuple[float, float, float]


@dataclass(frozen=True)
class GroundingCandidates:
    """Ranked SAM3 instances for one phrase in the current frame.

    ``instances`` is ordered by descending confidence; the caller assigns the
    stable A/B/C marks. No geometry is lifted here — measuring every candidate
    would spend depth lifting on instances the model may reject.
    """

    # The phrase the instances were grounded with.
    target: str
    # SAM3 instances for ``target``, highest confidence first.
    instances: tuple[Detection2D, ...]
    # Current metric depth raster aligned pixel-for-pixel with the RGB frame.
    depth_m: np.ndarray
    # Pinhole camera intrinsics for the same-capture RGB-D frame.
    camera_intrinsics: np.ndarray
    # Rigid transform from optical camera coordinates to world coordinates.
    world_T_camera: np.ndarray

    def top(self, count: int) -> tuple[Detection2D, ...]:
        """The first ``count`` ranked instances."""
        return self.instances[:count]


def ground_visible_candidates(
    *,
    semantic_backend: Any,
    observation: Any,
    target: str,
) -> GroundingCandidates:
    """Run SAM3 for ``target`` and rank every instance without measuring.

    Raises ``ObjectNotVisibleError`` when SAM3 returns no instance for the
    phrase — a missed detection, never proof of absence. Camera-calibration
    failures still raise ``ObjectGeometryUnavailableError`` because marks on
    an uncalibrated frame cannot be measured after selection either.
    """

    ego = observation.views.get("ego") if observation is not None else None
    if ego is None or ego.rgb is None:
        raise EgoFrameUnavailableError("the current decision has no ego RGB frame")

    perception = semantic_backend.process(
        PerceptionInput(observation=observation, text_prompts=[target])
    )
    detections = tuple(perception.detections or ())
    if not detections:
        raise ObjectNotVisibleError(
            f'"{target}" is not visible in the current ego RGB frame'
        )
    ranked = tuple(
        sorted(
            detections,
            key=lambda item: -float(item.score),
        )
    )

    depth = np.asarray(ego.depth) if ego.depth is not None else None
    if depth is None:
        raise ObjectGeometryUnavailableError(
            f'"{target}" is visible but synchronized metric depth is unavailable'
        )

    camera = dict(ego.camera_params or {})
    extrinsics_id = camera.get("capture_id")
    depth_id = camera.get("depth_capture_id")
    if extrinsics_id is None or depth_id is None:
        raise ObjectGeometryUnavailableError(
            "the current ego depth and world_T_cam are missing capture_id; "
            f"depth_capture_id={depth_id!r}, capture_id={extrinsics_id!r}"
        )
    # Coerced outside the calibration guard below so a malformed stamp is not
    # reported as missing intrinsics.
    try:
        depth_capture = int(depth_id)
        extrinsics_capture = int(extrinsics_id)
    except (TypeError, ValueError) as exc:
        raise ObjectGeometryUnavailableError(
            "the current ego capture ids are not integers: "
            f"depth_capture_id={depth_capture!r}, capture_id={extrinsics_id!r}"
        ) from exc
    if extrinsics_capture != depth_capture:
        raise ObjectGeometryUnavailableError(
            "the current ego depth and world_T_cam come from different "
            f"captures: depth_capture_id={depth_capture}, "
            f"world_T_cam capture_id={extrinsics_capture}"
        )
    try:
        intrinsics = fov_to_intrinsics(
            float(camera["fov_deg"]),
            int(camera.get("width") or depth.shape[1]),
            int(camera.get("height") or depth.shape[0]),
        )
        world_T_camera = np.asarray(
            camera["world_T_cam"],
            dtype=np.float64,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ObjectGeometryUnavailableError(
            f"the current ego camera calibration is incomplete: {exc}"
        ) from exc
    if world_T_camera.shape != (4, 4) or not np.all(np.isfinite(world_T_camera)):
        raise ObjectGeometryUnavailableError(
            "world_T_cam must be a finite 4x4 transform"
        )

    return GroundingCandidates(
        target=target,
        instances=ranked,
        depth_m=np.asarray(depth, dtype=np.float32),
        camera_intrinsics=intrinsics,
        world_T_camera=world_T_camera,
    )


def measure_selected_instance(
    candidates: GroundingCandidates,
    detection: Detection2D,
) -> VisibleObjectGrounding:
    """Lift one selected candidate instance into metric world geometry.

    ``detection`` must come from ``candidates.instances`` of the same capture,
    so the mask and the depth raster are pixel-aligned by construction. The
    same per-instance guards as ``ground_visible_object`` apply: a maskless or
    depth-poor instance is a loud ``ObjectGeometryUnavailableError``, never a
    guessed coordinate.
    """

    if detection.mask is None:
        raise ObjectGeometryUnavailableError(
            f'SAM3 detected "{candidates.target}" but returned no instance mask'
        )
    mask = np.asarray(detection.mask, dtype=bool)
    depth = candidates.depth_m
    if depth.ndim != 2 or mask.shape != depth.shape:
        raise ObjectGeometryUnavailableError(
            "the target mask and metric depth must be aligned HxW rasters: "
            f"mask={mask.shape!r}, depth={depth.shape!r}"
        )

    target_points_camera = backproject_mask(
        mask,
        depth,
        candidates.camera_intrinsics,
    )
    if target_points_camera.shape[0] < 32:
        raise ObjectGeometryUnavailableError(
            f'"{candidates.target}" has fewer than 32 valid target depth '
            "pixels; move closer or improve the view before measuring"
        )
    center_camera = np.median(target_points_camera, axis=0)
    center_world = transform_to_world(
        center_camera.reshape(1, 3),
        candidates.world_T_camera,
    )[0]
    return VisibleObjectGrounding(
        detection=detection,
        depth_m=depth,
        target_mask=mask,
        camera_intrinsics=candidates.camera_intrinsics,
        world_T_camera=candidates.world_T_camera,
        target_center_camera_m=tuple(float(value) for value in center_camera),
        target_center_world_m=tuple(float(value) for value in center_world),
    )


def ground_visible_object(
    *,
    semantic_backend: Any,
    observation: Any,
    target: str,
) -> VisibleObjectGrounding:
    """Run the same SAM3 and measurement path as the public object tools."""

    candidates = ground_visible_candidates(
        semantic_backend=semantic_backend,
        observation=observation,
        target=target,
    )
    return measure_selected_instance(candidates, candidates.instances[0])


__all__ = [
    "EgoFrameUnavailableError",
    "GroundingCandidates",
    "ObjectGeometryUnavailableError",
    "ObjectNotVisibleError",
    "VisibleObjectGrounding",
    "ground_visible_candidates",
    "ground_visible_object",
    "measure_selected_instance",
]
