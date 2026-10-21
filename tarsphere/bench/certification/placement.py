"""Find physically supported placement patches through live scene raycasts."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.infra.simulation.scene_query import (
    SurfaceRaycast,
    belongs_to_prim_tree,
)
from tarsphere.bench.grounding.schema import (
    GroundedObject,
    GroundedScene,
    ObjectApproachPath,
)
from tarsphere.bench.spatial.physical_task.navigation_evidence import (
    complete_path,
    path_length,
    reachable_radial_occupancy_points,
    require_exact_occupancy_point,
    route_reaches_exact_goal,
)
from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.assets import tarsphere_assets_root
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


Point2 = Tuple[float, float]
Point3 = Tuple[float, float, float]


class StrictModel(BaseModel):
    """Immutable current-release placement-probe schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class PlacementSurfaceSearch(StrictModel):
    """Physical sampling resolution used before G1 stance and IK testing."""

    edge_margin_m: float = Field(
        ge=0.0,
        description=(
            "Additional supported width required outside every side of the "
            "target footprint, in metres."
        ),
    )
    sample_spacing_m: float = Field(
        gt=0.0,
        description=(
            "Maximum world-XY spacing between candidate patch centres, in metres."
        ),
    )
    max_height_variation_m: float = Field(
        ge=0.0,
        description=(
            "Largest allowed Z difference among the centre and four corner "
            "ray hits of one supported patch, in metres."
        ),
    )
    minimum_normal_z: float = Field(
        gt=0.0,
        le=1.0,
        description=(
            "Minimum upward world-Z component required from every sampled "
            "surface normal."
        ),
    )
    ray_origin_offset_m: float = Field(
        gt=0.0,
        description=(
            "Vertical distance above destination bounds from which each "
            "downward PhysX ray begins, in metres."
        ),
    )


PLACEMENT_SURFACE_SEARCH = PlacementSurfaceSearch(
    edge_margin_m=0.03,
    sample_spacing_m=0.10,
    max_height_variation_m=0.01,
    minimum_normal_z=math.cos(math.radians(15.0)),
    ray_origin_offset_m=0.20,
)


class PlacementStanceSearch(StrictModel):
    """Bounded occupancy search before live placement IK certification."""

    minimum_surface_distance_m: float = Field(
        ge=0.0,
        description=(
            "Smallest horizontal distance sampled between the Robot base "
            "centre and support-patch centre, in metres. This is a search "
            "boundary rather than an IK or collision threshold."
        ),
    )
    maximum_surface_distance_m: float = Field(
        gt=0.0,
        description=(
            "Largest horizontal distance sampled between the Robot base "
            "centre and support-patch centre, in metres. Candidates still "
            "require live arm reachability."
        ),
    )
    radial_spacing_m: float = Field(
        gt=0.0,
        description=("Maximum radial spacing between sampled stance rings, in metres."),
    )
    angular_samples: int = Field(
        ge=8,
        description=("Number of evenly spaced base positions sampled on every ring."),
    )

    @model_validator(mode="after")
    def validate_range(self) -> "PlacementStanceSearch":
        if self.minimum_surface_distance_m >= self.maximum_surface_distance_m:
            raise ValueError(
                "placement stance distance range must be strictly increasing"
            )
        return self


PLACEMENT_STANCE_SEARCH = PlacementStanceSearch(
    minimum_surface_distance_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    maximum_surface_distance_m=1.0,
    radial_spacing_m=0.05,
    angular_samples=64,
)


class PlacementSurfacePatch(StrictModel):
    """One live collision patch large and flat enough for the target."""

    target_object_id: str = Field(
        min_length=1,
        description=("Grounded pickable whose live footprint was proven supportable."),
    )
    destination_object_id: str = Field(
        min_length=1,
        description="Grounded destination object supporting every ray hit.",
    )
    center_xyz: Point3 = Field(
        description=(
            "World-space centre of the sampled support patch; this becomes a "
            "candidate support_surface_xyz only after stance and IK testing."
        ),
    )
    supported_footprint_xy_m: Point2 = Field(
        description=(
            "Target footprint plus the configured edge margin proven by the "
            "five downward ray samples, in metres."
        ),
    )
    height_variation_m: float = Field(
        ge=0.0,
        description="Maximum sampled surface-height difference in metres.",
    )
    minimum_normal_z: float = Field(
        gt=0.0,
        le=1.0,
        description="Smallest upward normal component among all ray samples.",
    )
    sampled_prim_paths: Tuple[str, ...] = Field(
        min_length=5,
        max_length=5,
        description=(
            "Destination collision prim hit at patch centre and four corners."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "PlacementSurfacePatch":
        if self.target_object_id == self.destination_object_id:
            raise ValueError("placement target and destination must be distinct")
        values = (
            *self.center_xyz,
            *self.supported_footprint_xy_m,
            self.height_variation_m,
            self.minimum_normal_z,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("placement surface patch geometry must be finite")
        if any(value <= 0.0 for value in self.supported_footprint_xy_m):
            raise ValueError("supported footprint dimensions must be positive")
        return self


class PlacementStanceCandidate(StrictModel):
    """One occupancy-reachable G1 base pose facing a placement patch."""

    robot_stance_xy: Point2 = Field(
        description=(
            "Exact official-occupancy cell centre proposed for the G1 base, "
            "in world metres."
        ),
    )
    robot_yaw_rad: float = Field(
        ge=-math.pi,
        le=math.pi,
        description="World yaw that points the G1 torso toward the patch centre.",
    )
    surface_distance_xy_m: float = Field(
        ge=0.0,
        description=(
            "Horizontal straight-line distance from the base centre to the "
            "support-patch centre, in metres."
        ),
    )
    near_field_path_length_m: float = Field(
        ge=0.0,
        description=(
            "Collision-free occupancy path length from the selected official "
            "destination approach endpoint to this stance, in metres."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "PlacementStanceCandidate":
        values = (
            *self.robot_stance_xy,
            self.robot_yaw_rad,
            self.surface_distance_xy_m,
            self.near_field_path_length_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("placement stance geometry must be finite")
        return self


class PlacementStanceSet(StrictModel):
    """Stance candidates tied to one destination patch and approach path."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene containing the destination and occupancy.",
    )
    destination_object_id: str = Field(
        min_length=1,
        description="Grounded object supporting the selected surface patch.",
    )
    target_object_id: str = Field(
        min_length=1,
        description=("Grounded pickable whose live footprint conditioned the patch."),
    )
    surface_center_xyz: Point3 = Field(
        description="Live-probed placement patch centre in world metres.",
    )
    destination_approach_path_index: int = Field(
        ge=0,
        description=(
            "Official destination-conditioned path whose endpoint begins the "
            "near-field stance route."
        ),
    )
    destination_approach_xy: Point2 = Field(
        description=(
            "Official approach-path endpoint used as the stance-search start."
        ),
    )
    robot_clearance_radius_m: float = Field(
        gt=0.0,
        description=(
            "G1 base-footprint radius applied to official occupancy during "
            "candidate reachability planning."
        ),
    )
    candidates: Tuple[PlacementStanceCandidate, ...] = Field(
        description=(
            "Deterministically ordered occupancy-reachable base poses; an "
            "empty tuple means this patch/path pair has no static candidate."
        ),
    )

    @model_validator(mode="after")
    def validate_candidates(self) -> "PlacementStanceSet":
        if not all(
            math.isfinite(value)
            for value in (*self.surface_center_xyz, *self.destination_approach_xy)
        ):
            raise ValueError("placement stance set geometry must be finite")
        positions = tuple(candidate.robot_stance_xy for candidate in self.candidates)
        if len(set(positions)) != len(positions):
            raise ValueError("placement stance candidates must be unique")
        return self


class PlacementStanceResolutionError(ValueError):
    """Raised when one official destination path cannot seed stance search."""


def find_placement_surface_patches(
    target: GroundedObject,
    destination: GroundedObject,
    *,
    target_bounds_xyz: tuple[Point3, Point3],
    raycast: SurfaceRaycast,
    search: PlacementSurfaceSearch = PLACEMENT_SURFACE_SEARCH,
) -> tuple[PlacementSurfacePatch, ...]:
    """Find flat destination collision patches supporting one target.

    A patch is accepted only when its centre and four footprint corners all
    hit the destination prim hierarchy, all normals point sufficiently upward,
    and the sampled heights agree. The result does not claim G1 navigation,
    stance clearance, IK reachability, release, or settled support.
    """
    if not destination.runtime_prim:
        raise ValueError(
            "placement surface probing requires a verified destination runtime prim"
        )
    target_bounds_min_xyz, target_bounds_max_xyz = _validated_live_bounds(
        target_bounds_xyz
    )
    target_width = target_bounds_max_xyz[0] - target_bounds_min_xyz[0]
    target_depth = target_bounds_max_xyz[1] - target_bounds_min_xyz[1]
    required_half_width = target_width / 2.0 + search.edge_margin_m
    required_half_depth = target_depth / 2.0 + search.edge_margin_m
    nominal_centers = sample_nominal_placement_surface_centers(
        (target_width, target_depth),
        destination,
        search=search,
    )

    lower = destination.bounds_min_xyz
    upper = destination.bounds_max_xyz
    ray_origin_z = upper[2] + search.ray_origin_offset_m
    ray_length = upper[2] - lower[2] + 2.0 * search.ray_origin_offset_m
    patches = []
    for x, y in nominal_centers:
        samples = (
            (x, y),
            (x - required_half_width, y - required_half_depth),
            (x - required_half_width, y + required_half_depth),
            (x + required_half_width, y - required_half_depth),
            (x + required_half_width, y + required_half_depth),
        )
        hits = []
        for sample_x, sample_y in samples:
            hit = raycast(
                (sample_x, sample_y, ray_origin_z),
                (0.0, 0.0, -1.0),
                ray_length,
            )
            if (
                hit is None
                or not _belongs_to_destination(
                    hit.prim_path,
                    destination.runtime_prim,
                )
                or hit.normal_xyz[2] < search.minimum_normal_z
            ):
                hits = []
                break
            hits.append(hit)
        if not hits:
            continue
        heights = tuple(hit.position_xyz[2] for hit in hits)
        height_variation = max(heights) - min(heights)
        if height_variation > search.max_height_variation_m:
            continue
        patches.append(
            PlacementSurfacePatch(
                target_object_id=target.object_id,
                destination_object_id=destination.object_id,
                center_xyz=(x, y, sum(heights) / len(heights)),
                supported_footprint_xy_m=(
                    2.0 * required_half_width,
                    2.0 * required_half_depth,
                ),
                height_variation_m=height_variation,
                minimum_normal_z=min(hit.normal_xyz[2] for hit in hits),
                sampled_prim_paths=tuple(hit.prim_path for hit in hits),
            )
        )
    return tuple(
        sorted(
            patches,
            key=lambda patch: (
                math.dist(
                    patch.center_xyz[:2],
                    destination.position_xyz[:2],
                ),
                patch.center_xyz,
            ),
        )
    )


def sample_nominal_placement_surface_centers(
    target_footprint_xy_m: Point2,
    destination: GroundedObject,
    *,
    search: PlacementSurfaceSearch = PLACEMENT_SURFACE_SEARCH,
) -> tuple[Point2, ...]:
    """Sample AABB-contained patch centres without claiming a real surface.

    The destination centre is not treated as a manipulation target. Instead,
    the full inset footprint is sampled at the same resolution as the live
    five-ray surface probe. Callers may use these points to plan which real
    patches should be tested; only ``find_placement_surface_patches`` can
    certify that geometry exists at any sampled point.
    """
    target_width, target_depth = target_footprint_xy_m
    if (
        not all(math.isfinite(value) for value in target_footprint_xy_m)
        or target_width <= 0.0
        or target_depth <= 0.0
    ):
        raise ValueError("placement target footprint must be positive finite")
    required_half_width = target_width / 2.0 + search.edge_margin_m
    required_half_depth = target_depth / 2.0 + search.edge_margin_m
    lower = destination.bounds_min_xyz
    upper = destination.bounds_max_xyz
    minimum_x = lower[0] + required_half_width
    maximum_x = upper[0] - required_half_width
    minimum_y = lower[1] + required_half_depth
    maximum_y = upper[1] - required_half_depth
    if minimum_x > maximum_x or minimum_y > maximum_y:
        return ()
    return tuple(
        (x, y)
        for x in _sample_axis(
            minimum_x,
            maximum_x,
            maximum_spacing=search.sample_spacing_m,
        )
        for y in _sample_axis(
            minimum_y,
            maximum_y,
            maximum_spacing=search.sample_spacing_m,
        )
    )


def find_placement_stance_candidates(
    scene: GroundedScene,
    destination: GroundedObject,
    patch: PlacementSurfacePatch,
    *,
    approach_path_index: int,
    search: PlacementStanceSearch = PLACEMENT_STANCE_SEARCH,
    assets_root: Optional[Path | str] = None,
) -> PlacementStanceSet:
    """Find G1-clear base poses around one live-probed placement patch.

    Each returned stance is an exact official-occupancy cell centre reachable
    from the selected destination approach endpoint with the G1 footprint
    radius applied. Its yaw faces the patch. The result does not establish
    near-field collider clearance, arm IK, attached-object geometry, release,
    or settled support; those require a live post-grasp certification.
    """
    if destination.object_id != patch.destination_object_id:
        raise ValueError(
            "placement patch and destination must identify the same object"
        )
    scene_destination = _scene_object(scene, destination.object_id)
    if scene_destination != destination:
        raise ValueError(
            "placement stance search requires the unmodified destination "
            "object from the grounded scene"
        )
    approach = _destination_approach(
        scene,
        destination_object_id=destination.object_id,
        path_index=approach_path_index,
    )
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root()
    )
    occupancy_path = (root / scene.map.occupancy).resolve()
    if not occupancy_path.is_file():
        raise FileNotFoundError(
            f"official scene occupancy is missing: {occupancy_path}"
        )
    occupancy = OfficialOccupancyBevMap(
        occupancy_path=occupancy_path,
    )
    planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        cache_static_nav=True,
    )
    try:
        require_exact_occupancy_point(
            planner,
            approach.approach_xy,
            label="destination approach endpoint",
        )
    except ValueError as exc:
        raise PlacementStanceResolutionError(str(exc)) from exc

    candidates = []
    for position_xy in reachable_radial_occupancy_points(
        occupancy,
        center_xy=patch.center_xyz[:2],
        reference_xy=approach.approach_xy,
        minimum_distance_m=search.minimum_surface_distance_m,
        maximum_distance_m=search.maximum_surface_distance_m,
        radial_spacing_m=search.radial_spacing_m,
        angular_samples=search.angular_samples,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    ):
        route = planner.plan(
            PlanInput(
                start_xy=approach.approach_xy,
                start_yaw_rad=0.0,
                goal_xy=position_xy,
            )
        )
        if not route_reaches_exact_goal(
            route,
            start_xy=approach.approach_xy,
            goal_xy=position_xy,
        ):
            continue
        candidates.append(
            PlacementStanceCandidate(
                robot_stance_xy=position_xy,
                robot_yaw_rad=math.atan2(
                    patch.center_xyz[1] - position_xy[1],
                    patch.center_xyz[0] - position_xy[0],
                ),
                surface_distance_xy_m=math.dist(
                    position_xy,
                    patch.center_xyz[:2],
                ),
                near_field_path_length_m=path_length(
                    complete_path(
                        route,
                        start_xy=approach.approach_xy,
                        goal_xy=position_xy,
                    )
                ),
            )
        )
    return PlacementStanceSet(
        scene_id=scene.scene_id,
        destination_object_id=destination.object_id,
        target_object_id=patch.target_object_id,
        surface_center_xyz=patch.center_xyz,
        destination_approach_path_index=approach.path_index,
        destination_approach_xy=approach.approach_xy,
        robot_clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        candidates=tuple(
            sorted(
                candidates,
                key=lambda candidate: (
                    candidate.surface_distance_xy_m,
                    candidate.near_field_path_length_m,
                    candidate.robot_stance_xy,
                ),
            )
        ),
    )


def _sample_axis(
    minimum: float,
    maximum: float,
    *,
    maximum_spacing: float,
) -> tuple[float, ...]:
    if math.isclose(minimum, maximum, abs_tol=1e-12):
        return ((minimum + maximum) / 2.0,)
    intervals = max(
        1,
        math.ceil((maximum - minimum) / maximum_spacing),
    )
    return tuple(
        minimum + (maximum - minimum) * index / intervals
        for index in range(intervals + 1)
    )


def _scene_object(
    scene: GroundedScene,
    object_id: str,
) -> GroundedObject:
    for obj in scene.objects:
        if obj.object_id == object_id:
            return obj
    raise ValueError(f"placement stance destination is absent from scene: {object_id}")


def _destination_approach(
    scene: GroundedScene,
    *,
    destination_object_id: str,
    path_index: int,
) -> ObjectApproachPath:
    for path in scene.object_approach_paths:
        if path.object_id == destination_object_id and path.path_index == path_index:
            return path
    raise ValueError(
        "placement stance search requires an official approach path for "
        f"{destination_object_id} at index {path_index}"
    )


def _validated_live_bounds(
    bounds_xyz: tuple[Point3, Point3],
) -> tuple[Point3, Point3]:
    try:
        lower, upper = bounds_xyz
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "target_bounds_xyz must contain minimum and maximum 3-D points"
        ) from exc
    values = (*lower, *upper)
    if (
        len(lower) != 3
        or len(upper) != 3
        or not all(math.isfinite(float(value)) for value in values)
        or any(float(lower[axis]) > float(upper[axis]) for axis in range(3))
    ):
        raise ValueError("target_bounds_xyz must contain finite ordered 3-D points")
    return (
        tuple(float(value) for value in lower),
        tuple(float(value) for value in upper),
    )


def _belongs_to_destination(
    hit_prim_path: str,
    destination_prim_path: str,
) -> bool:
    return belongs_to_prim_tree(hit_prim_path, destination_prim_path)


__all__ = [
    "PLACEMENT_STANCE_SEARCH",
    "PLACEMENT_SURFACE_SEARCH",
    "PlacementStanceCandidate",
    "PlacementStanceResolutionError",
    "PlacementStanceSearch",
    "PlacementStanceSet",
    "PlacementSurfacePatch",
    "PlacementSurfaceSearch",
    "find_placement_stance_candidates",
    "find_placement_surface_patches",
    "sample_nominal_placement_surface_centers",
]
