"""Standalone RGB-D visual-frontier perception and ranking pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np

from tarsphere.infra.navigation.open_frontier.contracts import (
    CameraIntrinsics,
    DisplayFrontier,
    FrontierFrame,
    FrontierNetPrediction,
)
from tarsphere.infra.navigation.open_frontier.extraction import (
    FrontierCandidateExtractor,
)
from tarsphere.infra.navigation.open_frontier.graph import FrontierGraph
from tarsphere.infra.navigation.open_frontier.overlay import (
    render_frontier_overlay,
)
from tarsphere.infra.navigation.open_frontier.preprocessing import (
    PreprocessedFrontierFrame,
)
from tarsphere.geometry import HUMAN_BODY_RADIUS_M
from tarsphere.infra.control.physical_contracts import ROBOT_HUMAN_CENTER_CLEARANCE_M


def frontier_overlaps_visible_human(
    world_xyz: Any,
    camera_xyz: Any,
    visible_human_centers_xy: tuple[tuple[float, float], ...],
) -> bool:
    """Reject body goals and bearings occluded by currently perceived Humans.

    Only current RGB-D-grounded centres are admissible. Saved entities and
    unseen simulator actors must never suppress exploration proposals.
    """
    goal = np.asarray(world_xyz, dtype=float)[:2]
    camera = np.asarray(camera_xyz, dtype=float)[:2]
    ray = goal - camera
    length = float(np.linalg.norm(ray))
    for center in visible_human_centers_xy:
        human = np.asarray(center, dtype=float)
        if human.shape != (2,) or not np.all(np.isfinite(human)):
            raise ValueError("Visible Human centre must contain finite world XY.")
        if np.linalg.norm(goal - human) < ROBOT_HUMAN_CENTER_CLEARANCE_M:
            return True
        if length > 1e-8:
            along = float(np.dot(human - camera, ray / length))
            lateral = float(np.linalg.norm(human - camera - along * ray / length))
            if 0.0 < along <= length and lateral <= HUMAN_BODY_RADIUS_M:
                return True
    return False


class DenseFrontierPredictor(Protocol):
    def predict(
        self,
        *,
        rgb: Any,
        depth_m: Any,
        intrinsics: CameraIntrinsics,
    ) -> tuple[PreprocessedFrontierFrame, FrontierNetPrediction]:
        """Return one aligned RGB-D crop and its dense frontier prediction."""


@dataclass(frozen=True, slots=True)
class OpenFrontierConfig:
    maximum_displayed_frontiers: int = 3

    def __post_init__(self) -> None:
        if not 1 <= int(self.maximum_displayed_frontiers) <= 3:
            raise ValueError("maximum_displayed_frontiers must be in [1, 3].")


class OpenFrontierPipeline:
    """Detect, anchor, track, and rank current visual frontiers by model gain."""

    def __init__(
        self,
        *,
        predictor: DenseFrontierPredictor,
        extractor: FrontierCandidateExtractor | None = None,
        graph: FrontierGraph | None = None,
        config: OpenFrontierConfig | None = None,
    ) -> None:
        self.predictor = predictor
        self.extractor = extractor or FrontierCandidateExtractor()
        self.graph = graph or FrontierGraph()
        self.config = config or OpenFrontierConfig()

    def reset(self) -> None:
        self.graph.reset()

    def observe(
        self,
        *,
        observation_step: int,
        rgb: Any,
        depth_m: Any,
        intrinsics: CameraIntrinsics,
        world_T_camera: Any,
        visible_human_centers_xy: tuple[tuple[float, float], ...] = (),
    ) -> FrontierFrame:
        """Expose the three highest-gain current marks before route planning."""

        transform = np.asarray(world_T_camera, dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError("world_T_camera must be one finite 4x4 transform.")
        processed, prediction = self.predictor.predict(
            rgb=rgb,
            depth_m=depth_m,
            intrinsics=intrinsics,
        )
        observed = self.extractor.extract(
            prediction=prediction,
            depth_m=processed.depth_m,
            intrinsics=processed.intrinsics,
            world_T_camera=transform,
        )
        observed = tuple(
            candidate
            for candidate in observed
            if not frontier_overlaps_visible_human(
                candidate.world_xyz, transform[:3, 3], visible_human_centers_xy
            )
        )
        assignments = self.graph.add_observation(
            observation_step=int(observation_step),
            world_T_camera=transform,
            frontiers=observed,
        )

        # OpenFrontier filters once on the detector's base gain and again
        # after gain adjustment. Extraction owns the one gain threshold; use
        # that same authority here instead of introducing a display cutoff.
        minimum_gain_m3 = float(self.extractor.config.minimum_information_gain_m3)
        ranked = []
        for candidate in observed:
            frontier_id = assignments[candidate.observation_index]
            tracked = self.graph.get(frontier_id)
            if tracked.updated_information_gain_m3 < minimum_gain_m3:
                continue
            ranked.append(
                (
                    float(tracked.updated_information_gain_m3),
                    int(candidate.support_pixel_count),
                    frontier_id,
                    candidate,
                )
            )
        ranked.sort(key=lambda item: (-item[0], -item[1], item[2]))

        displayed = tuple(
            DisplayFrontier(
                label=chr(ord("A") + display_index),
                frontier_id=frontier_id,
                pixel_xy=candidate.pixel_xy,
                world_xyz=candidate.world_xyz,
                information_gain_m3=float(adjusted_gain),
            )
            for display_index, (adjusted_gain, _, frontier_id, candidate) in enumerate(
                ranked[: self.config.maximum_displayed_frontiers]
            )
        )
        overlay = render_frontier_overlay(
            processed.rgb,
            prediction.frontier_mask,
            displayed,
        )
        return FrontierFrame(
            observation_step=int(observation_step),
            processed_rgb=processed.rgb,
            processed_depth_m=processed.depth_m,
            processed_intrinsics=processed.intrinsics,
            prediction=prediction,
            observed_frontiers=observed,
            displayed_frontiers=displayed,
            overlay_rgb=overlay,
            graph_snapshot=self.graph.snapshot(),
        )


__all__ = ["DenseFrontierPredictor", "OpenFrontierConfig", "OpenFrontierPipeline"]
