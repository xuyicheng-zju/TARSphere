"""Continuous, machine-scored difficulty for Base benchmark tasks.

Difficulty belongs to the ``(scene, task mainline)`` Base task. Capability
variants inherit it unchanged, so paired comparisons differ only in the
authored interference. The score is a transparent sum of geometry-backed
components; it is not a route-length alias and it never affects task success.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from itertools import pairwise
from pathlib import Path
from types import MappingProxyType
from typing import Any


class DifficultyDimension(str, Enum):
    """Independent, scene-grounded sources of Base-task difficulty."""

    ROUTE_SCALE = "route_scale"
    SPATIAL_TOPOLOGY = "spatial_topology"
    VISUAL_ACCESS = "visual_access"
    ROUTE_GEOMETRY = "route_geometry"
    PASSAGE_CLEARANCE = "passage_clearance"


class DifficultyTier(str, Enum):
    """Three display bands over the continuous score."""

    EASY = "easy"
    MEDIUM = "medium"
    HARD = "hard"


class DifficultyFactor(str, Enum):
    """Named components shown in reports and authoring evidence."""

    ROUTE_SCALE = "route_scale"
    SPATIAL_PARTITION = "spatial_partition"
    TARGET_OCCLUSION = "target_occlusion"
    ROUTE_TORTUOSITY = "route_tortuosity"
    NARROW_PASSAGE = "narrow_passage"


@dataclass(frozen=True)
class DifficultyFactorSpec:
    """One score component, its formula, and its evidence contract."""

    factor: DifficultyFactor
    dimension: DifficultyDimension
    maximum_points: float
    description: str
    formula: str
    evidence_rule: str
    automatic: bool


@dataclass(frozen=True)
class BaseDifficultyMeasurements:
    """Auditable measurements behind one Base score."""

    navigate_to_target_length_m: float
    transport_to_destination_length_m: float
    second_leg_overlap_fraction: float
    spatial_partition_crossed: bool
    target_occluded: bool
    target_occlusion_fraction: float
    target_occlusion_evidence_available: bool
    navigate_to_target_clearance_p10_m: float
    transport_to_destination_clearance_p10_m: float
    navigate_to_target_turn_count: int
    transport_to_destination_turn_count: int
    navigate_to_target_turn_rad: float
    transport_to_destination_turn_rad: float

    @property
    def route_length_m(self) -> float:
        """Physical distance travelled across both task legs."""

        return self.navigate_to_target_length_m + self.transport_to_destination_length_m

    @property
    def second_leg_weight(self) -> float:
        """Second-leg weight retaining execution cost on a fully reused route."""

        return 1.0 - 0.5 * self.second_leg_overlap_fraction

    @property
    def significant_turn_count(self) -> int:
        """Total auditable turn count before overlap weighting."""

        return (
            self.navigate_to_target_turn_count
            + self.transport_to_destination_turn_count
        )

    @property
    def cumulative_turn_rad(self) -> float:
        """Total auditable turning before overlap weighting."""

        return self.navigate_to_target_turn_rad + self.transport_to_destination_turn_rad


@dataclass(frozen=True)
class DifficultyScore:
    """Continuous score with an exact component decomposition."""

    factor_counts: Mapping[DifficultyFactor, int]
    factor_points: Mapping[DifficultyFactor, float]
    dimension_points: Mapping[DifficultyDimension, float]
    total_points: float
    tier: DifficultyTier
    measurements: BaseDifficultyMeasurements


@dataclass(frozen=True)
class DifficultyAssignment:
    """One episode's inherited or directly derived Base difficulty."""

    base_episode_id: str
    score: DifficultyScore
    inherited: bool


@dataclass(frozen=True)
class DifficultyCalibration:
    """Central, physically interpretable constants for the Base score."""

    route_reference_m: float = 9.0
    route_cap: float = 2.5
    binary_context_points: float = 0.5
    significant_turn_rad: float = math.pi / 6.0
    turn_density_reference_rad_per_m: float = math.pi / 6.0
    tortuosity_cap: float = 1.5
    comfortable_clearance_m: float = 0.35
    feasible_clearance_m: float = 0.25
    route_overlap_radius_m: float = 0.25


BASE_DIFFICULTY_CALIBRATION = DifficultyCalibration()


DIFFICULTY_CATALOG: Mapping[DifficultyFactor, DifficultyFactorSpec] = MappingProxyType(
    {
        DifficultyFactor.ROUTE_SCALE: DifficultyFactorSpec(
            factor=DifficultyFactor.ROUTE_SCALE,
            dimension=DifficultyDimension.ROUTE_SCALE,
            maximum_points=BASE_DIFFICULTY_CALIBRATION.route_cap,
            description=(
                "Sublinear route scale with the second leg weighted by spatial novelty."
            ),
            formula=(
                "min(2.5, sqrt(L1 / 9 m) + (1 - 0.5 * overlap_2_to_1) * sqrt(L2 / 9 m))"
            ),
            evidence_rule=(
                "Pinned occupancy-aware optimal length for each leg. "
                "overlap_2_to_1 is the fraction of the second leg within "
                "0.25 m of the first leg."
            ),
            automatic=True,
        ),
        DifficultyFactor.SPATIAL_PARTITION: DifficultyFactorSpec(
            factor=DifficultyFactor.SPATIAL_PARTITION,
            dimension=DifficultyDimension.SPATIAL_TOPOLOGY,
            maximum_points=BASE_DIFFICULTY_CALIBRATION.binary_context_points,
            description="The outbound task leg crosses a spatial partition.",
            formula="0.5 if the outbound chord intersects wall/door geometry else 0",
            evidence_rule=(
                "Test only the start-to-objective chord against pinned wall/door "
                "geometry. Do not count the same partition again on return."
            ),
            automatic=True,
        ),
        DifficultyFactor.TARGET_OCCLUSION: DifficultyFactorSpec(
            factor=DifficultyFactor.TARGET_OCCLUSION,
            dimension=DifficultyDimension.VISUAL_ACCESS,
            maximum_points=BASE_DIFFICULTY_CALIBRATION.binary_context_points,
            description="The objective is occluded within its spatial region.",
            formula=(
                "0.5 if the objective ray is obstructed and no region partition "
                "is crossed else 0"
            ),
            evidence_rule="Pinned eye-height obstruction mask and projection metadata.",
            automatic=True,
        ),
        DifficultyFactor.ROUTE_TORTUOSITY: DifficultyFactorSpec(
            factor=DifficultyFactor.ROUTE_TORTUOSITY,
            dimension=DifficultyDimension.ROUTE_GEOMETRY,
            maximum_points=BASE_DIFFICULTY_CALIBRATION.tortuosity_cap,
            description="Route complexity expressed by cumulative meaningful turning.",
            formula=(
                "min(1.5, tortuosity(L1) + (1 - 0.5 * overlap_2_to_1) * tortuosity(L2))"
            ),
            evidence_rule=(
                "Compute meaningful turning per metre on each task leg, then "
                "weight the second by its route novelty. Exclude the intentional "
                "phase-transition reversal. One point means one 90-degree turn "
                "per 3 route metres."
            ),
            automatic=True,
        ),
        DifficultyFactor.NARROW_PASSAGE: DifficultyFactorSpec(
            factor=DifficultyFactor.NARROW_PASSAGE,
            dimension=DifficultyDimension.PASSAGE_CLEARANCE,
            maximum_points=1.0,
            description="Continuous penalty for sustained low route clearance.",
            formula=("min(1, narrow(L1) + (1 - 0.5 * overlap_2_to_1) * narrow(L2))"),
            evidence_rule=(
                "Compute 10th-percentile raw obstacle clearance per leg, then "
                "weight the second by route novelty; G1 remains feasible at "
                "its 0.25 m planning footprint."
            ),
            automatic=True,
        ),
    }
)

AUTOMATIC_DIFFICULTY_FACTORS = frozenset(
    factor for factor, spec in DIFFICULTY_CATALOG.items() if spec.automatic
)

# Half-open display bands calibrated independently within each 30-Base
# mainline. The continuous formula remains identical across mainlines.
DIFFICULTY_TIER_RANGES_BY_MAINLINE: Mapping[
    str, Mapping[DifficultyTier, tuple[float, float | None]]
] = MappingProxyType(
    {
        "navigate_and_report": MappingProxyType(
            {
                DifficultyTier.EASY: (0.0, 1.51),
                DifficultyTier.MEDIUM: (1.51, 3.08),
                DifficultyTier.HARD: (3.08, None),
            }
        ),
        "fetch_and_deliver": MappingProxyType(
            {
                DifficultyTier.EASY: (0.0, 1.60),
                DifficultyTier.MEDIUM: (1.60, 3.44),
                DifficultyTier.HARD: (3.44, None),
            }
        ),
    }
)


def _clip(value: float, lower: float, upper: float) -> float:
    return min(upper, max(lower, value))


def route_extent_factor(optimal_navigation_length_m: float) -> DifficultyFactor:
    """Compatibility name for the continuous route-scale component."""

    if (
        not math.isfinite(optimal_navigation_length_m)
        or optimal_navigation_length_m < 0
    ):
        raise ValueError("optimal_navigation_length_m must be finite and non-negative")
    return DifficultyFactor.ROUTE_SCALE


def difficulty_tier(total_points: float, *, mainline: str) -> DifficultyTier:
    """Derive a display tier within one task-mainline coordinate system."""

    if not math.isfinite(total_points) or total_points < 0:
        raise ValueError("total_points must be finite and non-negative")
    mainline_value = str(getattr(mainline, "value", mainline))
    ranges = DIFFICULTY_TIER_RANGES_BY_MAINLINE.get(mainline_value)
    if ranges is None:
        raise ValueError(
            f"unsupported task mainline for difficulty: {mainline_value!r}"
        )
    easy_upper = ranges[DifficultyTier.EASY][1]
    medium_upper = ranges[DifficultyTier.MEDIUM][1]
    assert easy_upper is not None and medium_upper is not None
    if total_points < easy_upper:
        return DifficultyTier.EASY
    if total_points < medium_upper:
        return DifficultyTier.MEDIUM
    return DifficultyTier.HARD


def score_difficulty(
    measurements: BaseDifficultyMeasurements,
    *,
    mainline: str,
    authored_factors: Iterable[DifficultyFactor | str] = (),
) -> DifficultyScore:
    """Compute continuous difficulty from machine-verifiable measurements."""

    calibration = BASE_DIFFICULTY_CALIBRATION
    leg_lengths = (
        measurements.navigate_to_target_length_m,
        measurements.transport_to_destination_length_m,
    )
    if any(not math.isfinite(length) or length <= 0 for length in leg_lengths):
        raise ValueError("both route-leg lengths must be finite and positive")
    if not 0.0 <= measurements.second_leg_overlap_fraction <= 1.0:
        raise ValueError("second_leg_overlap_fraction must be within [0, 1]")
    leg_clearances = (
        measurements.navigate_to_target_clearance_p10_m,
        measurements.transport_to_destination_clearance_p10_m,
    )
    if any(not math.isfinite(value) or value < 0 for value in leg_clearances):
        raise ValueError("route-leg clearance measurements must be non-negative")
    leg_turn_counts = (
        measurements.navigate_to_target_turn_count,
        measurements.transport_to_destination_turn_count,
    )
    leg_turn_radians = (
        measurements.navigate_to_target_turn_rad,
        measurements.transport_to_destination_turn_rad,
    )
    if any(value < 0 for value in (*leg_turn_counts, *leg_turn_radians)):
        raise ValueError("route-leg turn measurements cannot be negative")
    if (
        measurements.target_occluded
        and not measurements.target_occlusion_evidence_available
    ):
        raise ValueError("target occlusion requires pinned visual-access evidence")

    authored = tuple(DifficultyFactor(value) for value in authored_factors)
    if len(authored) != len(set(authored)):
        raise ValueError("difficulty factors cannot contain duplicates")
    automatic = sorted(
        factor.value for factor in authored if factor in AUTOMATIC_DIFFICULTY_FACTORS
    )
    if automatic:
        raise ValueError(
            f"automatic difficulty factors cannot be authored: {automatic}"
        )

    second_leg_weight = measurements.second_leg_weight

    def route_scale(length_m: float) -> float:
        return math.sqrt(length_m / calibration.route_reference_m)

    def route_tortuosity(length_m: float, cumulative_turn_rad: float) -> float:
        return _clip(
            (cumulative_turn_rad / length_m)
            / calibration.turn_density_reference_rad_per_m,
            0.0,
            calibration.tortuosity_cap,
        )

    def narrow_passage(clearance_p10_m: float) -> float:
        return _clip(
            (calibration.comfortable_clearance_m - clearance_p10_m)
            / (calibration.comfortable_clearance_m - calibration.feasible_clearance_m),
            0.0,
            1.0,
        )

    factor_points = {
        DifficultyFactor.ROUTE_SCALE: _clip(
            route_scale(measurements.navigate_to_target_length_m)
            + second_leg_weight
            * route_scale(measurements.transport_to_destination_length_m),
            0.0,
            calibration.route_cap,
        ),
        DifficultyFactor.SPATIAL_PARTITION: (
            calibration.binary_context_points
            if measurements.spatial_partition_crossed
            else 0.0
        ),
        DifficultyFactor.TARGET_OCCLUSION: (
            calibration.binary_context_points
            if measurements.target_occluded
            and not measurements.spatial_partition_crossed
            else 0.0
        ),
        DifficultyFactor.ROUTE_TORTUOSITY: _clip(
            route_tortuosity(
                measurements.navigate_to_target_length_m,
                measurements.navigate_to_target_turn_rad,
            )
            + second_leg_weight
            * route_tortuosity(
                measurements.transport_to_destination_length_m,
                measurements.transport_to_destination_turn_rad,
            ),
            0.0,
            calibration.tortuosity_cap,
        ),
        DifficultyFactor.NARROW_PASSAGE: _clip(
            narrow_passage(measurements.navigate_to_target_clearance_p10_m)
            + second_leg_weight
            * narrow_passage(measurements.transport_to_destination_clearance_p10_m),
            0.0,
            1.0,
        ),
    }
    dimensions = {dimension: 0.0 for dimension in DifficultyDimension}
    for factor, points in factor_points.items():
        dimensions[DIFFICULTY_CATALOG[factor].dimension] += points
    total = sum(factor_points.values())
    return DifficultyScore(
        factor_counts=MappingProxyType(
            {factor: 1 for factor, points in factor_points.items() if points > 0}
        ),
        factor_points=MappingProxyType(factor_points),
        dimension_points=MappingProxyType(dimensions),
        total_points=total,
        tier=difficulty_tier(total, mainline=mainline),
        measurements=measurements,
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _route_endpoints(
    route: Mapping[str, Any],
) -> tuple[tuple[tuple[float, float], tuple[float, float]], ...]:
    legs = []
    for name in ("navigate_to_target", "transport_to_destination"):
        points = (route.get(name) or {}).get("points") or ()
        if len(points) < 2:
            raise ValueError(f"difficulty derivation requires two points on {name}")
        start = tuple(float(value) for value in points[0]["position_xy"][:2])
        goal = tuple(float(value) for value in points[-1]["position_xy"][:2])
        legs.append((start, goal))
    return tuple(legs)


def _indexed_route_polyline(
    episode: Any,
    route: Mapping[str, Any],
) -> tuple[tuple[tuple[float, float], ...], int, str | None] | None:
    """Load the frozen reviewed route when the episode declares its index."""

    from tarsphere.assets import TARSPHERE_REPOSITORY_ROOT

    sources = (episode.world_knowledge or {}).get("scene_sources") or {}
    raw_index = str(sources.get("route_index") or "").strip()
    if not raw_index:
        return None
    path = Path(raw_index).expanduser()
    if not path.is_absolute():
        path = TARSPHERE_REPOSITORY_ROOT / path
    payload = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for scene in payload.get("scenes") or ():
        for task in (scene.get("tasks") or {}).values():
            if task.get("task_id") == episode.episode_id:
                matches.append(task)
    if len(matches) != 1:
        raise ValueError(
            f"{episode.episode_id}: route index must contain exactly one matching task"
        )
    record = matches[0]
    if str(record.get("occupancy_sha256")) != str(route.get("occupancy_sha256")):
        raise ValueError(f"{episode.episode_id}: stale route-index occupancy_sha256")
    if not math.isclose(
        float(record["length_m"]),
        float(route["optimal_navigation_length_m"]),
        abs_tol=0.05,
    ):
        raise ValueError(f"{episode.episode_id}: stale route-index length")
    points = tuple(
        tuple(float(value) for value in point[:2])
        for point in record["points_world_xy"]
    )
    objective_index = int(record["objective_point_index"])
    if not 0 < objective_index < len(points) - 1:
        raise ValueError(f"{episode.episode_id}: invalid objective point index")
    endpoints = _route_endpoints(route)
    indexed_endpoints = (
        (points[0], points[objective_index]),
        (points[objective_index], points[-1]),
    )
    if any(
        math.dist(expected, observed) > 0.02
        for expected_leg, observed_leg in zip(endpoints, indexed_endpoints)
        for expected, observed in zip(expected_leg, observed_leg)
    ):
        raise ValueError(f"{episode.episode_id}: stale route-index endpoints")
    return (
        points,
        objective_index,
        str(record.get("layout_geometry_sha256") or "") or None,
    )


def _planned_route_polyline(
    episode: Any,
    occupancy: Any,
    endpoints: Sequence[tuple[tuple[float, float], tuple[float, float]]],
) -> tuple[tuple[tuple[float, float], ...], int, None]:
    """Replan legacy Base routes that predate a frozen production route index."""

    from tarsphere.harness.planner.contracts import PlanInput
    from tarsphere.harness.planner.planner import Planner
    from tarsphere.bench.certification.scene import official_runtime_navigation_mask
    from tarsphere.geometry import HUMAN_BODY_RADIUS_M
    from tarsphere.infra.control.physical_contracts import (
        G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    )

    obstacles = tuple(
        (
            float(human.start_position[0]),
            float(human.start_position[1]),
            HUMAN_BODY_RADIUS_M,
        )
        for human in episode.humans
        if human.static and not getattr(human, "scene_person_prim_path", None)
    )
    mask = official_runtime_navigation_mask(
        occupancy.occupancy_path,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        static_obstacles=obstacles,
    )
    planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        prepared_navigable_mask=mask,
        cache_static_nav=True,
    )
    legs: list[tuple[tuple[float, float], ...]] = []
    for start, goal in endpoints:
        output = planner.plan(
            PlanInput(start_xy=start, start_yaw_rad=0.0, goal_xy=goal)
        )
        if output.status not in {"ok", "already_there"}:
            raise ValueError(f"{episode.episode_id}: Base route is not replannable")
        points = (
            start,
            *(tuple(float(value) for value in point[:2]) for point in output.waypoints),
        )
        if math.dist(points[-1], goal) > 0.02:
            points = (*points, goal)
        legs.append(points)
    return (*legs[0], *legs[1][1:]), len(legs[0]) - 1, None


def _sample_polyline(
    points: Sequence[tuple[float, float]], spacing_m: float = 0.05
) -> tuple[tuple[float, float], ...]:
    samples: list[tuple[float, float]] = []
    for start, end in pairwise(points):
        distance = math.dist(start, end)
        count = max(2, math.ceil(distance / spacing_m) + 1)
        for index in range(count):
            fraction = index / (count - 1)
            samples.append(
                (
                    start[0] + fraction * (end[0] - start[0]),
                    start[1] + fraction * (end[1] - start[1]),
                )
            )
    return tuple(samples or points)


def _second_leg_overlap_fraction(
    first_leg: Sequence[tuple[float, float]],
    second_leg: Sequence[tuple[float, float]],
    *,
    radius_m: float,
) -> float:
    """Measure how much of the second leg reuses the first leg's corridor."""

    import numpy as np
    from scipy.spatial import cKDTree

    first_samples = np.asarray(_sample_polyline(first_leg), dtype=float)
    second_samples = np.asarray(_sample_polyline(second_leg), dtype=float)
    distances, _ = cKDTree(first_samples).query(second_samples, k=1)
    return float(np.mean(distances <= radius_m))


def _segment_intersects_aabb(
    start: tuple[float, float],
    goal: tuple[float, float],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    margin_m: float = 0.03,
) -> bool:
    distance = math.dist(start, goal)
    if distance <= 0.6:
        return False
    x0, y0 = start
    dx, dy = goal[0] - x0, goal[1] - y0
    entry, exit = 0.0, 1.0
    for direction, offset in (
        (-dx, x0 - (float(lower[0]) - margin_m)),
        (dx, (float(upper[0]) + margin_m) - x0),
        (-dy, y0 - (float(lower[1]) - margin_m)),
        (dy, (float(upper[1]) + margin_m) - y0),
    ):
        if abs(direction) < 1e-12:
            if offset < 0:
                return False
            continue
        ratio = offset / direction
        if direction < 0:
            entry = max(entry, ratio)
        else:
            exit = min(exit, ratio)
        if entry > exit:
            return False
    # Ignore geometry touching only the local 0.30 m endpoint neighborhoods.
    return exit * distance > 0.30 and (1.0 - entry) * distance > 0.30


def _scene_geometry_measurements(
    episode: Any,
    endpoints: Sequence[tuple[tuple[float, float], tuple[float, float]]],
    *,
    expected_geometry_sha256: str | None,
) -> tuple[bool, bool, float, bool]:
    """Measure region partitions and visual occlusion from one geometry source."""

    import numpy as np
    from PIL import Image

    from tarsphere.assets import TARSPHERE_REPOSITORY_ROOT, tarsphere_assets_root

    world = episode.world_knowledge or {}
    sources = world.get("scene_sources") or {}
    scene_map = world.get("scene_map") or {}
    raw_geometry = str(sources.get("layout_geometry") or "").strip()
    raw_projection = str(scene_map.get("projection") or "").strip()
    if not raw_geometry or not raw_projection:
        return 0, False, 0.0, False
    geometry_path = Path(raw_geometry).expanduser()
    if not geometry_path.is_absolute():
        geometry_path = TARSPHERE_REPOSITORY_ROOT / geometry_path
    projection_path = Path(raw_projection).expanduser()
    if not projection_path.is_absolute():
        projection_path = tarsphere_assets_root() / projection_path
    geometry = json.loads(geometry_path.read_text(encoding="utf-8"))
    if expected_geometry_sha256 and _sha256(geometry_path) != expected_geometry_sha256:
        raise ValueError(f"{episode.episode_id}: stale layout geometry")
    sight_path = geometry_path.with_suffix(".sight.png")
    if _sha256(sight_path) != geometry["sight_mask_sha256"]:
        raise ValueError(f"{episode.episode_id}: stale sight obstruction mask")
    if _sha256(projection_path) != geometry["projection_sha256"]:
        raise ValueError(f"{episode.episode_id}: stale sight projection metadata")
    meta = json.loads(projection_path.read_text(encoding="utf-8"))
    mask = np.asarray(Image.open(sight_path)) > 0
    partitions = tuple(
        instance
        for instance in geometry.get("instances") or ()
        if instance.get("category") in {"wall", "door"}
    )
    start, objective = endpoints[0]
    spatial_partition_crossed = any(
        _segment_intersects_aabb(
            start,
            objective,
            instance["bounds_min_m"],
            instance["bounds_max_m"],
        )
        for instance in partitions
    )
    samples = _sample_polyline((start, objective), spacing_m=0.02)
    endpoint_samples = max(1, int(0.30 / 0.02))
    if len(samples) > 2 * endpoint_samples:
        samples = samples[endpoint_samples:-endpoint_samples]
    xmin, ymin = (float(value) for value in meta["bev_bounds_min_xy"])
    span = float(meta["bev_span_m"])
    resolution = int(meta["resolution"])
    xy = np.asarray(samples, dtype=float)
    columns = np.rint((xy[:, 0] - xmin) / span * (resolution - 1)).astype(int)
    rows = np.rint((ymin + span - xy[:, 1]) / span * (resolution - 1)).astype(int)
    values = mask[
        np.clip(rows, 0, resolution - 1),
        np.clip(columns, 0, resolution - 1),
    ]
    fraction = float(values.mean()) if len(values) else 0.0
    return spatial_partition_crossed, bool(values.any()), fraction, True


def _turn_measurements(
    points: Sequence[tuple[float, float]],
) -> tuple[int, float]:
    count = 0
    cumulative = 0.0
    for first, pivot, last in zip(points, points[1:], points[2:]):
        incoming = (pivot[0] - first[0], pivot[1] - first[1])
        outgoing = (last[0] - pivot[0], last[1] - pivot[1])
        norm_in = math.hypot(*incoming)
        norm_out = math.hypot(*outgoing)
        if norm_in < 0.10 or norm_out < 0.10:
            continue
        cosine = _clip(
            (incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            / (norm_in * norm_out),
            -1.0,
            1.0,
        )
        angle = math.acos(cosine)
        if angle >= BASE_DIFFICULTY_CALIBRATION.significant_turn_rad:
            count += 1
            cumulative += angle
    return count, cumulative


def measure_base_difficulty(episode: Any) -> BaseDifficultyMeasurements:
    """Measure all automatic Base difficulty components from frozen evidence."""

    import numpy as np
    from scipy.ndimage import distance_transform_edt

    from tarsphere.harness.perception.backends.gt_bev import OfficialOccupancyBevMap
    from tarsphere.assets import resolve_tarsphere_asset

    route = dict((episode.world_knowledge or {}).get("episode_route") or {})
    if "optimal_navigation_length_m" not in route:
        raise ValueError("difficulty derivation requires measured route length")
    route_length = float(route["optimal_navigation_length_m"])
    if not math.isfinite(route_length) or route_length <= 0:
        raise ValueError("optimal_navigation_length_m must be finite and positive")
    endpoints = _route_endpoints(route)
    scene_map = (episode.world_knowledge or {}).get("scene_map") or {}
    raw_occupancy = str(scene_map.get("occupancy") or "").strip()
    if not raw_occupancy:
        raise ValueError("difficulty derivation requires scene_map.occupancy")
    occupancy_path = resolve_tarsphere_asset(raw_occupancy).resolve()
    if _sha256(occupancy_path) != str(route.get("occupancy_sha256") or ""):
        raise ValueError(f"{episode.episode_id}: stale difficulty occupancy")
    occupancy = OfficialOccupancyBevMap(occupancy_path=occupancy_path)
    indexed = _indexed_route_polyline(episode, route)
    points, objective_index, expected_geometry_sha256 = (
        indexed
        if indexed is not None
        else _planned_route_polyline(episode, occupancy, endpoints)
    )

    free = occupancy.navigable_mask()
    clearance = distance_transform_edt(free) / float(occupancy.pixels_per_meter)
    first_leg = points[: objective_index + 1]
    second_leg = points[objective_index:]

    def clearance_p10(route_points: Sequence[tuple[float, float]]) -> float:
        cells = occupancy.world_to_cell(
            np.asarray(_sample_polyline(route_points), dtype=float)
        )
        values = np.asarray([clearance[int(row), int(column)] for row, column in cells])
        return float(np.quantile(values, 0.10))

    navigate_length = float(route["navigate_to_target"]["optimal_navigation_length_m"])
    transport_length = float(
        route["transport_to_destination"]["optimal_navigation_length_m"]
    )
    if not math.isclose(
        navigate_length + transport_length,
        route_length,
        abs_tol=0.05,
    ):
        raise ValueError(f"{episode.episode_id}: route-leg lengths do not match total")
    (
        spatial_partition_crossed,
        target_occluded,
        occlusion_fraction,
        occlusion_available,
    ) = _scene_geometry_measurements(
        episode,
        endpoints,
        expected_geometry_sha256=expected_geometry_sha256,
    )
    first_turn_count, first_cumulative_turn = _turn_measurements(first_leg)
    second_turn_count, second_cumulative_turn = _turn_measurements(second_leg)
    # The intentional task-phase reversal at the objective is not an obstacle
    # turn. Measuring each leg separately excludes that structural U-turn.
    return BaseDifficultyMeasurements(
        navigate_to_target_length_m=navigate_length,
        transport_to_destination_length_m=transport_length,
        second_leg_overlap_fraction=_second_leg_overlap_fraction(
            first_leg,
            second_leg,
            radius_m=BASE_DIFFICULTY_CALIBRATION.route_overlap_radius_m,
        ),
        spatial_partition_crossed=spatial_partition_crossed,
        target_occluded=target_occluded,
        target_occlusion_fraction=occlusion_fraction,
        target_occlusion_evidence_available=occlusion_available,
        navigate_to_target_clearance_p10_m=clearance_p10(first_leg),
        transport_to_destination_clearance_p10_m=clearance_p10(second_leg),
        navigate_to_target_turn_count=first_turn_count,
        transport_to_destination_turn_count=second_turn_count,
        navigate_to_target_turn_rad=first_cumulative_turn,
        transport_to_destination_turn_rad=second_cumulative_turn,
    )


def derive_episode_difficulty(episode: Any) -> DifficultyScore:
    """Derive continuous difficulty for one Base episode."""

    from tarsphere.contracts.task_profile import CapabilitySlice

    if episode.evaluation is None:
        raise ValueError("difficulty derivation requires an evaluation graph")
    if episode.evaluation.task_profile.capability_slice is not CapabilitySlice.BASE:
        raise ValueError(
            "difficulty is defined only on Base episodes; capability variants "
            "must inherit the matching (scene, task mainline) Base score"
        )
    if episode.authoring is None:
        raise ValueError("difficulty derivation requires authoring evidence")
    return score_difficulty(
        measure_base_difficulty(episode),
        mainline=episode.evaluation.task_profile.mainline.value,
        authored_factors=episode.authoring.difficulty_factors,
    )


def base_difficulty_key(episode: Any) -> tuple[str, str]:
    """Return the comparison-unit key shared by one Base and its variants."""

    if episode.evaluation is None:
        raise ValueError("difficulty assignment requires an evaluation graph")
    scene_path = str(episode.scene.asset_path).strip()
    if not scene_path:
        raise ValueError("difficulty assignment requires a scene asset path")
    return scene_path, episode.evaluation.task_profile.mainline.value


def assign_batch_difficulty(
    episodes: Iterable[Any],
) -> Mapping[str, DifficultyAssignment]:
    """Assign each unique Base score to its same-scene/mainline variants."""

    from tarsphere.contracts.task_profile import CapabilitySlice

    values = tuple(episodes)
    bases: dict[tuple[str, str], Any] = {}
    episode_ids: set[str] = set()
    for episode in values:
        if episode.episode_id in episode_ids:
            raise ValueError(f"duplicate episode id: {episode.episode_id}")
        episode_ids.add(episode.episode_id)
        if episode.evaluation is None:
            raise ValueError(f"{episode.episode_id}: missing evaluation graph")
        profile = episode.evaluation.task_profile
        if profile.capability_slice is not CapabilitySlice.BASE:
            if episode.authoring and episode.authoring.difficulty_factors:
                raise ValueError(
                    f"{episode.episode_id}: variants cannot author difficulty factors"
                )
            continue
        key = base_difficulty_key(episode)
        if key in bases:
            raise ValueError(
                "multiple Base episodes share one scene/mainline difficulty unit: "
                f"{bases[key].episode_id!r}, {episode.episode_id!r}"
            )
        bases[key] = episode

    base_scores = {key: derive_episode_difficulty(base) for key, base in bases.items()}
    assignments: dict[str, DifficultyAssignment] = {}
    for episode in values:
        key = base_difficulty_key(episode)
        base = bases.get(key)
        if base is None:
            raise ValueError(
                f"{episode.episode_id}: no matching Base episode for scene "
                f"{key[0]!r} and mainline {key[1]!r}"
            )
        assignments[episode.episode_id] = DifficultyAssignment(
            base_episode_id=base.episode_id,
            score=base_scores[key],
            inherited=episode.episode_id != base.episode_id,
        )
    return MappingProxyType(assignments)


__all__ = [
    "AUTOMATIC_DIFFICULTY_FACTORS",
    "BASE_DIFFICULTY_CALIBRATION",
    "DIFFICULTY_CATALOG",
    "DIFFICULTY_TIER_RANGES_BY_MAINLINE",
    "BaseDifficultyMeasurements",
    "DifficultyAssignment",
    "DifficultyCalibration",
    "DifficultyDimension",
    "DifficultyFactor",
    "DifficultyFactorSpec",
    "DifficultyScore",
    "DifficultyTier",
    "assign_batch_difficulty",
    "base_difficulty_key",
    "derive_episode_difficulty",
    "difficulty_tier",
    "measure_base_difficulty",
    "route_extent_factor",
    "score_difficulty",
]
