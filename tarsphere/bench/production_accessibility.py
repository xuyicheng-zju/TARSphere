"""Navigation calibration gates for the initial production benchmark."""

from __future__ import annotations

import math
from dataclasses import dataclass

from tarsphere.harness.perception.backends.gt_bev import OfficialOccupancyBevMap
from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.bench.authoring import CandidateGeometryEvidence
from tarsphere.bench.certification.scene import official_runtime_navigation_mask
from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.bench.spatial.physical_task.navigation_evidence import (
    route_reaches_exact_goal,
)
from tarsphere.geometry import HUMAN_BODY_RADIUS_M
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)

# Routes above this calibration length need review, not automatic rejection.
PRODUCTION_ROUTE_LENGTH_REVIEW_THRESHOLD_M = 18.0
PRODUCTION_NAVIGATION_CLEARANCE_RADIUS_M = G1_NAVIGATION_FOOTPRINT_RADIUS_M


@dataclass(frozen=True)
class ProductionAccessibilityReport:
    """Measured evidence that one candidate avoids avoidable navigation load."""

    optimal_route_length_m: float
    route_length_review_threshold_m: float
    navigation_clearance_radius_m: float
    navigation_checked_segment_count: int

    @property
    def route_length_review_required(self) -> bool:
        """Whether semantic review should inspect the route for needless detours."""
        return self.optimal_route_length_m > self.route_length_review_threshold_m


def validate_production_navigation_accessibility(
    episode: TARSphereEpisode,
    geometry: CandidateGeometryEvidence,
) -> ProductionAccessibilityReport:
    """Validate reachability and flag long routes for review.

    The ordinary publication gate and this initial-batch accessibility gate
    use the same authoritative G1 navigation radius. The production-only
    addition is the route-length review reminder. Passage clearance contributes
    to difficulty but remains admissible whenever the authoritative planner
    proves the route feasible for the G1 footprint.
    """

    route_length = float(geometry.route_length_m)
    if not math.isfinite(route_length) or route_length <= 0:
        raise ValueError(
            f"{episode.episode_id}: route length must be finite and positive"
        )
    planner = Planner(
        OfficialOccupancyBevMap(occupancy_path=geometry.occupancy_path),
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=PRODUCTION_NAVIGATION_CLEARANCE_RADIUS_M,
        prepared_navigable_mask=official_runtime_navigation_mask(
            geometry.occupancy_path,
            clearance_radius_m=PRODUCTION_NAVIGATION_CLEARANCE_RADIUS_M,
            static_obstacles=tuple(
                (
                    float(human.start_position[0]),
                    float(human.start_position[1]),
                    HUMAN_BODY_RADIUS_M,
                )
                for human in episode.humans
                if human.static
            ),
        ),
        cache_static_nav=True,
    )
    checked = 0
    for leg_name, leg in (
        ("navigate_to_target", geometry.route.navigate_to_target),
        ("transport_to_destination", geometry.route.transport_to_destination),
    ):
        for start, goal in zip(leg.points, leg.points[1:]):
            output = planner.plan(
                PlanInput(
                    start_xy=start.position_xy,
                    start_yaw_rad=0.0,
                    goal_xy=goal.position_xy,
                )
            )
            if not route_reaches_exact_goal(
                output,
                start_xy=start.position_xy,
                goal_xy=goal.position_xy,
            ):
                raise ValueError(
                    f"{episode.episode_id}: {leg_name} segment "
                    f"{start.point_id} -> {goal.point_id} has no exact route "
                    f"at the production navigation clearance "
                    f"{PRODUCTION_NAVIGATION_CLEARANCE_RADIUS_M:.3f} m"
                )
            checked += 1

    return ProductionAccessibilityReport(
        optimal_route_length_m=route_length,
        route_length_review_threshold_m=PRODUCTION_ROUTE_LENGTH_REVIEW_THRESHOLD_M,
        navigation_clearance_radius_m=(PRODUCTION_NAVIGATION_CLEARANCE_RADIUS_M),
        navigation_checked_segment_count=checked,
    )


__all__ = [
    "PRODUCTION_NAVIGATION_CLEARANCE_RADIUS_M",
    "PRODUCTION_ROUTE_LENGTH_REVIEW_THRESHOLD_M",
    "ProductionAccessibilityReport",
    "validate_production_navigation_accessibility",
]
