"""Lift dense FrontierNet predictions into sparse 3-D frontier candidates."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Protocol

import numpy as np
from scipy import ndimage

from tarsphere.infra.navigation.open_frontier.contracts import (
    CameraIntrinsics,
    FrontierNetPrediction,
    ObservedFrontier,
)


class FrontierClusterer(Protocol):
    def labels(self, weighted_features: np.ndarray) -> np.ndarray:
        """Return one integer cluster label per feature; ``-1`` means noise."""


class HdbscanFrontierClusterer:
    """The density clustering used for production visual frontiers."""

    def __init__(
        self,
        *,
        min_cluster_size: int = 20,
        min_samples: int = 25,
        cluster_selection_epsilon: float = 0.5,
    ) -> None:
        if min_cluster_size < 2 or min_samples < 1:
            raise ValueError("HDBSCAN cluster sizes must be positive.")
        if cluster_selection_epsilon < 0.0:
            raise ValueError("cluster_selection_epsilon cannot be negative.")
        self.min_cluster_size = int(min_cluster_size)
        self.min_samples = int(min_samples)
        self.cluster_selection_epsilon = float(cluster_selection_epsilon)

    def labels(self, weighted_features: np.ndarray) -> np.ndarray:
        try:
            import hdbscan
        except ImportError as exc:
            raise RuntimeError(
                "Visual-frontier extraction requires hdbscan; install the "
                "project open-frontier extra."
            ) from exc
        features = np.asarray(weighted_features, dtype=np.float64)
        if features.ndim != 2:
            raise ValueError("HDBSCAN frontier features must be a 2-D matrix.")
        if features.shape[0] < self.min_cluster_size:
            return np.full(features.shape[0], -1, dtype=np.int32)
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=self.min_cluster_size,
            min_samples=self.min_samples,
            cluster_selection_epsilon=self.cluster_selection_epsilon,
            metric="euclidean",
        )
        return np.asarray(clusterer.fit_predict(features), dtype=np.int32)


@dataclass(frozen=True, slots=True)
class FrontierExtractionConfig:
    minimum_depth_m: float = 0.30
    maximum_depth_m: float = 10.0
    minimum_depth_discontinuity_m: float = 0.05
    maximum_depth_discontinuity_m: float = 10.0
    minimum_information_gain_m3: float = 0.50
    gain_volume_scale: float = 0.01
    maximum_absolute_view_z: float | None = 0.65
    feature_weights: tuple[float, float, float, float, float] = (
        10.0,
        10.0,
        1.0,
        1.0,
        0.3,
    )

    def __post_init__(self) -> None:
        positive = (
            self.minimum_depth_m,
            self.maximum_depth_m,
            self.minimum_depth_discontinuity_m,
            self.maximum_depth_discontinuity_m,
            self.gain_volume_scale,
        )
        if not all(
            math.isfinite(float(value)) and float(value) > 0 for value in positive
        ):
            raise ValueError(
                "Frontier extraction distances and scale must be positive."
            )
        if self.maximum_depth_m <= self.minimum_depth_m:
            raise ValueError("maximum_depth_m must exceed minimum_depth_m.")
        if self.maximum_depth_discontinuity_m <= self.minimum_depth_discontinuity_m:
            raise ValueError("maximum_depth_discontinuity_m must exceed its minimum.")
        if self.minimum_information_gain_m3 < 0.0:
            raise ValueError("minimum_information_gain_m3 cannot be negative.")
        if len(self.feature_weights) != 5 or not all(
            math.isfinite(float(value)) and float(value) > 0
            for value in self.feature_weights
        ):
            raise ValueError("feature_weights must contain five positive values.")


@dataclass(frozen=True, slots=True)
class _PixelFeatures:
    pixel_x: np.ndarray
    pixel_y: np.ndarray
    direction_x: np.ndarray
    direction_y: np.ndarray
    gain: np.ndarray
    anchor_depth_m: np.ndarray

    def matrix(self, *, width: int, height: int) -> np.ndarray:
        return np.column_stack(
            [
                self.pixel_x / float(width),
                self.pixel_y / float(height),
                self.direction_x,
                self.direction_y,
                self.gain,
                self.anchor_depth_m,
            ]
        )


def _validate_world_transform(world_T_camera: Any) -> np.ndarray:
    transform = np.asarray(world_T_camera, dtype=np.float64)
    if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
        raise ValueError("world_T_camera must be one finite 4x4 transform.")
    if not np.allclose(transform[3], [0.0, 0.0, 0.0, 1.0], atol=1e-6):
        raise ValueError("world_T_camera must have homogeneous final row [0,0,0,1].")
    rotation = transform[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3):
        raise ValueError("world_T_camera rotation must be orthonormal.")
    return transform


def _pixel_features(
    prediction: FrontierNetPrediction,
    depth_m: np.ndarray,
    config: FrontierExtractionConfig,
) -> _PixelFeatures:
    mask = np.asarray(prediction.frontier_mask, dtype=bool)
    depth = np.asarray(depth_m, dtype=np.float64)
    if mask.shape != depth.shape:
        raise ValueError("Frontier mask and metric depth must have the same shape.")
    if not mask.any():
        empty = np.zeros(0, dtype=np.float64)
        return _PixelFeatures(empty, empty, empty, empty, empty, empty)

    finite_depth = np.where(np.isfinite(depth) & (depth > 0.0), depth, 0.0)
    gradient_x = ndimage.sobel(finite_depth, axis=1, mode="nearest")
    gradient_y = ndimage.sobel(finite_depth, axis=0, mode="nearest")
    magnitude = np.hypot(gradient_x, gradient_y)
    direction = np.arctan2(gradient_y, gradient_x)

    rows, cols = np.nonzero(mask)
    centre_depth = np.clip(
        finite_depth[rows, cols],
        0.1,
        config.maximum_depth_m,
    )
    step = np.clip(1.0 / np.maximum(centre_depth, 1e-6), 2.5, 4.0)
    delta_row = np.rint(np.sin(direction[rows, cols]) * step).astype(np.int32)
    delta_col = np.rint(np.cos(direction[rows, cols]) * step).astype(np.int32)
    positive_rows = np.clip(rows + delta_row, 0, depth.shape[0] - 1)
    positive_cols = np.clip(cols + delta_col, 0, depth.shape[1] - 1)
    negative_rows = np.clip(rows - delta_row, 0, depth.shape[0] - 1)
    negative_cols = np.clip(cols - delta_col, 0, depth.shape[1] - 1)
    foreground_depth = finite_depth[negative_rows, negative_cols]
    background_depth = finite_depth[positive_rows, positive_cols]
    depth_difference = np.abs(foreground_depth - background_depth)
    valid = (
        np.isfinite(magnitude[rows, cols])
        & (magnitude[rows, cols] <= 1e5)
        & (foreground_depth >= config.minimum_depth_m)
        & (background_depth > 0.0)
        & (depth_difference > config.minimum_depth_discontinuity_m)
        & (depth_difference < config.maximum_depth_discontinuity_m)
    )
    rows = rows[valid]
    cols = cols[valid]
    angles = direction[rows, cols]
    anchor_depth = 0.5 * (foreground_depth[valid] + background_depth[valid])
    return _PixelFeatures(
        pixel_x=cols.astype(np.float64),
        pixel_y=rows.astype(np.float64),
        direction_x=np.cos(angles),
        direction_y=np.sin(angles),
        gain=np.asarray(prediction.information_gain[rows, cols], dtype=np.float64),
        anchor_depth_m=anchor_depth,
    )


class FrontierCandidateExtractor:
    """Cluster FrontierNet pixels and anchor each region in world metres."""

    def __init__(
        self,
        config: FrontierExtractionConfig | None = None,
        *,
        clusterer: FrontierClusterer | None = None,
    ) -> None:
        self.config = config or FrontierExtractionConfig()
        self.clusterer = clusterer or HdbscanFrontierClusterer()

    def extract(
        self,
        *,
        prediction: FrontierNetPrediction,
        depth_m: np.ndarray,
        intrinsics: CameraIntrinsics,
        world_T_camera: Any,
    ) -> tuple[ObservedFrontier, ...]:
        transform = _validate_world_transform(world_T_camera)
        depth = np.asarray(depth_m, dtype=np.float64)
        if depth.shape != (intrinsics.height, intrinsics.width):
            raise ValueError("Processed depth dimensions do not match intrinsics.")
        pixels = _pixel_features(prediction, depth, self.config)
        features = pixels.matrix(width=intrinsics.width, height=intrinsics.height)
        if not features.shape[0]:
            return ()
        selected = features[:, [0, 1, 2, 3, 5]]
        weighted = selected * np.asarray(self.config.feature_weights)[None, :]
        labels = np.asarray(self.clusterer.labels(weighted), dtype=np.int32)
        if labels.shape != (features.shape[0],):
            raise ValueError("Frontier clusterer returned one label per feature.")

        candidates: list[ObservedFrontier] = []
        for cluster_label in sorted(
            int(value) for value in np.unique(labels) if value >= 0
        ):
            cluster = features[labels == cluster_label]
            if cluster.shape[0] == 0:
                continue
            centroid = cluster[:, :2].mean(axis=0)
            centre_index = int(
                np.argmin(
                    np.linalg.norm(
                        cluster[:, :2] - centroid[None, :],
                        axis=1,
                    )
                )
            )
            centre = cluster[centre_index]
            direction_xy = cluster[:, 2:4].sum(axis=0)
            direction_norm = float(np.linalg.norm(direction_xy))
            if direction_norm <= 1e-8:
                continue
            direction_xy /= direction_norm
            gains = np.sort(cluster[:, 4])
            if gains.size >= 10:
                gains = gains[
                    gains.size // 2 : max(gains.size // 2 + 1, int(0.9 * gains.size))
                ]
            information_gain_m3 = (
                float(np.median(gains)) * self.config.gain_volume_scale
            )
            if information_gain_m3 < self.config.minimum_information_gain_m3:
                continue
            depths = np.sort(cluster[:, 5][cluster[:, 5] > 0.0])
            if not depths.size:
                continue
            low = int(0.05 * depths.size)
            high = max(low + 1, int(0.9 * depths.size))
            anchor_depth = float(np.median(depths[low:high]))
            if not (
                self.config.minimum_depth_m
                <= anchor_depth
                <= self.config.maximum_depth_m
            ):
                continue

            pixel_x = float(centre[0] * intrinsics.width)
            pixel_y = float(centre[1] * intrinsics.height)
            camera_point = np.asarray(
                [
                    (pixel_x - intrinsics.cx) * anchor_depth / intrinsics.fx,
                    (pixel_y - intrinsics.cy) * anchor_depth / intrinsics.fy,
                    anchor_depth,
                    1.0,
                ],
                dtype=np.float64,
            )
            world_point = transform @ camera_point
            image_direction_camera = np.asarray(
                [
                    direction_xy[0] / intrinsics.fx,
                    direction_xy[1] / intrinsics.fy,
                    0.0,
                ],
                dtype=np.float64,
            )
            image_direction_camera /= np.linalg.norm(image_direction_camera)
            world_direction = -(transform[:3, :3] @ image_direction_camera)
            world_direction /= np.linalg.norm(world_direction)
            max_view_z = self.config.maximum_absolute_view_z
            if max_view_z is not None and abs(float(world_direction[2])) > max_view_z:
                continue
            candidates.append(
                ObservedFrontier(
                    observation_index=len(candidates),
                    pixel_xy=(pixel_x, pixel_y),
                    world_xyz=tuple(float(value) for value in world_point[:3]),
                    view_direction_world=tuple(
                        float(value) for value in world_direction
                    ),
                    information_gain_m3=information_gain_m3,
                    support_pixel_count=int(cluster.shape[0]),
                )
            )
        return tuple(candidates)


__all__ = [
    "FrontierCandidateExtractor",
    "FrontierClusterer",
    "FrontierExtractionConfig",
    "HdbscanFrontierClusterer",
]
