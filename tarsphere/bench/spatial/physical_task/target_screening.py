"""Screen grounded manipulation targets before live Isaac certification."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Literal, Optional, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.grounding.schema import (
    GroundedScene,
    ObjectApproachPath,
)
from tarsphere.bench.inputs.sources import resolve_active_grscene
from tarsphere.bench.spatial.physical_task.navigation_evidence import (
    reachable_radial_occupancy_points,
)
from tarsphere.bench.spatial.physical_task.objects import (
    ManipulationTargetCandidate,
    discover_manipulation_targets,
)
from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.assets import tarsphere_assets_root
from tarsphere.infra.simulation.scene_preparation_contract import (
    PhysicalScenePreparation,
)
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    TARGET_SCREENING_MAX_GRASP_SPAN_M,
    TARGET_SCREENING_MIN_AUTHORED_BASE_COLOUR_LUMINANCE,
)


Point2 = Tuple[float, float]
ScreeningRule = Literal[
    "grasp_height",
    "grasp_width",
    "material_visibility",
    "reachable_robot_stance",
]


class StrictModel(BaseModel):
    """Immutable current-release target-screening schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ManipulationTargetScreeningRules(StrictModel):
    """Cheap G1 geometry gates applied before starting an Isaac trial."""

    grasp_height_fraction: float = Field(
        default=0.60,
        ge=0.0,
        le=1.0,
        description=(
            "Fraction of object AABB height used for the prospective gripper "
            "closing centre."
        ),
    )
    minimum_grasp_height_m: float = Field(
        default=0.35,
        ge=0.0,
        description=(
            "Lowest prospective closing-centre height admitted to live G1 "
            "certification, in scene metres."
        ),
    )
    maximum_grasp_height_m: float = Field(
        default=1.25,
        gt=0.0,
        description=(
            "Highest prospective closing-centre height admitted to live G1 "
            "certification, in scene metres."
        ),
    )
    maximum_grasp_width_m: float = Field(
        default=TARGET_SCREENING_MAX_GRASP_SPAN_M,
        gt=0.0,
        description=(
            "Largest horizontal AABB span along the better of the two "
            "target-facing grasp axes admitted by offline screening."
        ),
    )
    minimum_robot_target_distance_m: float = Field(
        default=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        ge=0.0,
        description=(
            "Smallest horizontal distance sampled between G1 base centre "
            "and target centre, in metres."
        ),
    )
    maximum_robot_target_distance_m: float = Field(
        default=0.70,
        gt=0.0,
        description=(
            "Largest coarse horizontal base-to-target reach admitted before "
            "the live right-hand IK gate."
        ),
    )
    radial_spacing_m: float = Field(
        default=0.05,
        gt=0.0,
        description="Maximum spacing between sampled Robot stance rings.",
    )
    angular_samples: int = Field(
        default=64,
        ge=8,
        description="Number of evenly spaced Robot poses sampled per ring.",
    )
    robot_clearance_radius_m: float = Field(
        default=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        gt=0.0,
        description=(
            "G1 body radius used to erode occupancy before connectivity and "
            "table-edge access are evaluated."
        ),
    )
    min_authored_base_colour_luminance: float = Field(
        default=TARGET_SCREENING_MIN_AUTHORED_BASE_COLOUR_LUMINANCE,
        ge=0.0,
        le=1.0,
        description=(
            "Lowest area-weighted authored Rec.709 base-colour luminance "
            "admitted after pickable constant-branch repair. Textured "
            "surfaces count as fully visible; a small dark trim cannot sink "
            "a brighter body."
        ),
    )

    @model_validator(mode="after")
    def validate_ranges(self) -> "ManipulationTargetScreeningRules":
        if self.minimum_grasp_height_m >= self.maximum_grasp_height_m:
            raise ValueError("target-screening grasp-height range must increase")
        if self.minimum_robot_target_distance_m >= self.maximum_robot_target_distance_m:
            raise ValueError("target-screening Robot-distance range must increase")
        return self


MANIPULATION_TARGET_SCREENING_RULES = ManipulationTargetScreeningRules()


class ScreenedManipulationTarget(StrictModel):
    """One target worth testing with live IK, contact, attach, and carry."""

    target: ManipulationTargetCandidate = Field(
        description=("Grounded semantic target and its support/path evidence."),
    )
    grasp_height_m: float = Field(
        ge=0.0,
        description=(
            "Prospective gripper closing-centre height derived from object "
            "bounds, in metres."
        ),
    )
    grasp_width_m: float = Field(
        ge=0.0,
        description=(
            "Smaller horizontal object AABB span offered to the hand, in metres."
        ),
    )
    approach_path_index: int = Field(
        ge=0,
        description=(
            "Official target path whose start is occupancy-connected to the "
            "selected Robot stance."
        ),
    )
    robot_stance_xy: Point2 = Field(
        description=(
            "Nearest sampled G1-clear base position within the coarse arm "
            "reach envelope."
        ),
    )
    robot_target_distance_m: float = Field(
        ge=0.0,
        description=(
            "Horizontal distance from selected base position to target AABB "
            "centre, in metres."
        ),
    )

    @model_validator(mode="after")
    def validate_geometry(self) -> "ScreenedManipulationTarget":
        values = (
            self.grasp_height_m,
            self.grasp_width_m,
            *self.robot_stance_xy,
            self.robot_target_distance_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("screened target geometry must be finite")
        if self.approach_path_index not in self.target.approach_path_indexes:
            raise ValueError("screened target stance is bound to an unknown path")
        return self

    @property
    def target_object_id(self) -> str:
        """Return the grounded object admitted by this screening result."""
        return self.target.target_object_id


class ManipulationTargetScreeningFailure(StrictModel):
    """One explicit rule that rejected a grounded target before Isaac."""

    rule: ScreeningRule = Field(
        description="Named geometry or occupancy gate that failed.",
    )
    explanation: str = Field(
        min_length=1,
        description="Human-readable reason the rule rejected this target.",
    )
    measured_value_m: Optional[float] = Field(
        default=None,
        ge=0.0,
        description=(
            "Measured scalar that failed the rule, or null when failure "
            "means that no valid stance exists."
        ),
    )
    required_range_m: Optional[Tuple[float, float]] = Field(
        default=None,
        description=(
            "Inclusive admitted scalar range, or null for a discrete "
            "occupancy/connectivity failure."
        ),
    )

    @model_validator(mode="after")
    def validate_measurement(self) -> "ManipulationTargetScreeningFailure":
        if self.measured_value_m is not None and not math.isfinite(
            self.measured_value_m
        ):
            raise ValueError("screening failure measurement must be finite")
        if self.required_range_m is not None:
            lower, upper = self.required_range_m
            if (
                not all(math.isfinite(value) for value in (lower, upper))
                or lower > upper
            ):
                raise ValueError("screening failure required range must be ordered")
        return self


class RejectedManipulationTarget(StrictModel):
    """One grounded target excluded before any live Robot motion."""

    target: ManipulationTargetCandidate = Field(
        description="Grounded semantic target that failed static screening.",
    )
    failure: ManipulationTargetScreeningFailure = Field(
        description=(
            "First failed rule in the fixed height, width, then stance screening order."
        ),
    )

    @property
    def target_object_id(self) -> str:
        """Return the grounded object excluded by this screening result."""
        return self.target.target_object_id


class ManipulationTargetScreening(StrictModel):
    """Authoritative offline admission for grounded targets in one scene."""

    scene_id: str = Field(
        min_length=1,
        description="Grounded scene whose targets were screened.",
    )
    rules: ManipulationTargetScreeningRules = Field(
        description="Exact geometry and occupancy rules applied.",
    )
    accepted_targets: Tuple[ScreenedManipulationTarget, ...] = Field(
        description=(
            "Only targets permitted to enter semantic task authoring and "
            "later live grasp certification."
        ),
    )
    rejected_targets: Tuple[RejectedManipulationTarget, ...] = Field(
        description=("Grounded targets excluded with explicit failed rules."),
    )

    @model_validator(mode="after")
    def validate_coverage(self) -> "ManipulationTargetScreening":
        accepted = tuple(item.target_object_id for item in self.accepted_targets)
        rejected = tuple(item.target_object_id for item in self.rejected_targets)
        if (
            len(set(accepted)) != len(accepted)
            or len(set(rejected)) != len(rejected)
            or set(accepted).intersection(rejected)
        ):
            raise ValueError("target screening must partition unique grounded targets")
        return self

    def require(self, target_object_id: str) -> ScreenedManipulationTarget:
        """Return an admitted target or fail with its screening reasons."""
        accepted = next(
            (
                item
                for item in self.accepted_targets
                if item.target_object_id == target_object_id
            ),
            None,
        )
        if accepted is not None:
            return accepted
        rejected = next(
            (
                item
                for item in self.rejected_targets
                if item.target_object_id == target_object_id
            ),
            None,
        )
        if rejected is None:
            raise ValueError(
                f"Object is not a grounded manipulation target: {target_object_id!r}"
            )
        raise ValueError(
            f"Object failed target screening: "
            f"{target_object_id!r}: {rejected.failure.explanation}"
        )

    @property
    def object_ids(self) -> Tuple[str, ...]:
        """Return admitted target ids in deterministic screening order."""
        return tuple(target.target_object_id for target in self.accepted_targets)


def screen_manipulation_targets(
    scene: GroundedScene,
    *,
    rules: ManipulationTargetScreeningRules = (MANIPULATION_TARGET_SCREENING_RULES),
    assets_root: Optional[Path | str] = None,
) -> ManipulationTargetScreening:
    """Apply height, hand-width, body-clearance, and connectivity gates."""
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
    objects = {item.object_id: item for item in scene.objects}
    paths_by_object = {}
    for path in scene.object_approach_paths:
        paths_by_object.setdefault(path.object_id, []).append(path)

    accepted = []
    rejected = []
    candidates = discover_manipulation_targets(scene).targets
    for target in candidates:
        obj = objects[target.target_object_id]
        width = obj.bounds_max_xyz[0] - obj.bounds_min_xyz[0]
        depth = obj.bounds_max_xyz[1] - obj.bounds_min_xyz[1]
        height = obj.bounds_max_xyz[2] - obj.bounds_min_xyz[2]
        grasp_height = obj.bounds_min_xyz[2] + rules.grasp_height_fraction * height
        grasp_width = min(width, depth)
        if not (
            rules.minimum_grasp_height_m <= grasp_height <= rules.maximum_grasp_height_m
        ):
            rejected.append(
                RejectedManipulationTarget(
                    target=target,
                    failure=ManipulationTargetScreeningFailure(
                        rule="grasp_height",
                        explanation=(
                            f"prospective grasp height {grasp_height:.3f} m lies "
                            f"outside "
                            f"[{rules.minimum_grasp_height_m:.3f}, "
                            f"{rules.maximum_grasp_height_m:.3f}] m"
                        ),
                        measured_value_m=grasp_height,
                        required_range_m=(
                            rules.minimum_grasp_height_m,
                            rules.maximum_grasp_height_m,
                        ),
                    ),
                )
            )
            continue
        if grasp_width > rules.maximum_grasp_width_m:
            rejected.append(
                RejectedManipulationTarget(
                    target=target,
                    failure=ManipulationTargetScreeningFailure(
                        rule="grasp_width",
                        explanation=(
                            f"best horizontal grasp width {grasp_width:.3f} m "
                            f"exceeds {rules.maximum_grasp_width_m:.3f} m"
                        ),
                        measured_value_m=grasp_width,
                        required_range_m=(
                            0.0,
                            rules.maximum_grasp_width_m,
                        ),
                    ),
                )
            )
            continue

        target_center_xy = (
            0.5 * (obj.bounds_min_xyz[0] + obj.bounds_max_xyz[0]),
            0.5 * (obj.bounds_min_xyz[1] + obj.bounds_max_xyz[1]),
        )
        stances = _reachable_stances(
            occupancy,
            target_center_xy=target_center_xy,
            paths=tuple(paths_by_object[target.target_object_id]),
            rules=rules,
        )
        if not stances:
            rejected.append(
                RejectedManipulationTarget(
                    target=target,
                    failure=ManipulationTargetScreeningFailure(
                        rule="reachable_robot_stance",
                        explanation=(
                            "no occupancy-connected G1 stance remains after body "
                            "clearance and coarse arm-reach constraints"
                        ),
                    ),
                )
            )
            continue
        path_index, stance_xy = min(
            stances,
            key=lambda item: (
                math.dist(item[1], target_center_xy),
                item[0],
                item[1],
            ),
        )
        accepted.append(
            ScreenedManipulationTarget(
                target=target,
                grasp_height_m=grasp_height,
                grasp_width_m=grasp_width,
                approach_path_index=path_index,
                robot_stance_xy=stance_xy,
                robot_target_distance_m=math.dist(
                    stance_xy,
                    target_center_xy,
                ),
            )
        )
    return ManipulationTargetScreening(
        scene_id=scene.scene_id,
        rules=rules,
        accepted_targets=tuple(accepted),
        rejected_targets=tuple(rejected),
    )


def _reachable_stances(
    occupancy: OfficialOccupancyBevMap,
    *,
    target_center_xy: Point2,
    paths: Tuple[ObjectApproachPath, ...],
    rules: ManipulationTargetScreeningRules,
) -> tuple[tuple[int, Point2], ...]:
    stances = set()
    for path in paths:
        try:
            positions = reachable_radial_occupancy_points(
                occupancy,
                center_xy=target_center_xy,
                reference_xy=path.start_xy,
                minimum_distance_m=(rules.minimum_robot_target_distance_m),
                maximum_distance_m=(rules.maximum_robot_target_distance_m),
                radial_spacing_m=rules.radial_spacing_m,
                angular_samples=rules.angular_samples,
                clearance_radius_m=rules.robot_clearance_radius_m,
            )
        except ValueError:
            continue
        stances.update((path.path_index, position) for position in positions)
    return tuple(sorted(stances))


def _shader_scalar(shader: Any, name: str) -> Optional[float]:
    port = shader.GetInput(name)
    if port is None:
        return None
    value = port.Get()
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value[0])
    except (TypeError, IndexError, ValueError):
        return None


def has_base_colour_texture(shader: Any) -> Optional[bool]:
    """Whether the MDL shader samples a base-colour texture.

    The MDL base-colour branch picks either a texture sample or a constant.
    Pickable configuration repairs the constant branch; textured surfaces
    already keep their authored colour in headless RTX.
    """
    flag = _shader_scalar(shader, "IsBaseColorTex")
    if flag is None:
        return None
    return flag >= 0.5


def _shader_rgb(
    shader: Any,
    name: str,
) -> Optional[Tuple[float, float, float]]:
    port = shader.GetInput(name)
    if port is None:
        return None
    value = port.Get()
    if value is None:
        return None
    if isinstance(value, (int, float)):
        channel = float(value)
        if not math.isfinite(channel):
            return None
        return (channel, channel, channel)
    try:
        red, green, blue = (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, IndexError, ValueError):
        return None
    if not all(math.isfinite(channel) for channel in (red, green, blue)):
        return None
    return (red, green, blue)


def _rec709_luminance(rgb: Tuple[float, float, float]) -> float:
    red, green, blue = rgb
    return 0.2126 * red + 0.7152 * green + 0.0722 * blue


def authored_base_colour_luminance(
    usd_path: Path | str,
    *,
    mdl_render_context: bool = True,
) -> Optional[float]:
    """Area-weighted authored Rec.709 base-colour luminance of one asset.

    Textured shaders count as fully visible (1.0) because that branch already
    survives in headless RTX. Constant-branch shaders contribute the authored
    ``BaseColor_Color`` luminance, which is what remains after pickable
    metallic/opacity repair. Area-weighting keeps a small dark trim from
    sinking the object.
    """
    from pxr import Usd, UsdGeom, UsdShade

    path = Path(usd_path)
    try:
        stage = Usd.Stage.Open(str(path))
    except Exception:
        return None
    if stage is None:
        return None
    bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), [UsdGeom.Tokens.default_])
    total_area = 0.0
    weighted_luminance = 0.0
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Mesh):
            continue
        material, _ = UsdShade.MaterialBindingAPI(prim).ComputeBoundMaterial()
        if not material:
            continue
        if mdl_render_context:
            # GRScenes authors surfaces as MDL, so the universal context is empty.
            source = (
                material.ComputeSurfaceSource("mdl") or material.ComputeSurfaceSource()
            )
        else:
            source = material.ComputeSurfaceSource()
        shader = UsdShade.Shader(source[0]) if source and source[0] else None
        if shader is None or not shader:
            continue
        textured = has_base_colour_texture(shader)
        if textured is None:
            continue
        if textured:
            luminance = 1.0
        else:
            rgb = _shader_rgb(shader, "BaseColor_Color")
            if rgb is None:
                continue
            luminance = _rec709_luminance(rgb)
        extent = bbox_cache.ComputeWorldBound(prim).ComputeAlignedRange()
        if extent.IsEmpty():
            continue
        sx, sy, sz = (extent.GetMax()[i] - extent.GetMin()[i] for i in range(3))
        area = 2.0 * (sx * sy + sy * sz + sx * sz)
        if not math.isfinite(area) or area <= 0.0:
            continue
        if not math.isfinite(luminance):
            continue
        total_area += area
        weighted_luminance += area * luminance
    if total_area <= 0.0:
        return None
    return weighted_luminance / total_area


def resolve_scene_pickable_instance_usd(
    scene_id: str,
    object_id: str,
    *,
    assets_root: Optional[Path | str] = None,
) -> Path:
    """Return the scene-local instance USD for one pickable object id."""
    category, model_part = object_id.split("/", 1)
    prefix = "model_"
    suffix = "_0"
    if not model_part.startswith(prefix) or not model_part.endswith(suffix):
        raise ValueError(
            f"object id does not name a scene-local pickable instance: {object_id!r}"
        )
    model_hash = model_part[len(prefix) : -len(suffix)]
    root = (
        Path(assets_root).expanduser().resolve()
        if assets_root is not None
        else tarsphere_assets_root()
    )
    paths = resolve_active_grscene(scene_id, assets_root=root)
    usd_path = (
        paths.scene_dir
        / "models/object/others"
        / category
        / model_hash
        / "instance.usd"
    )
    if not usd_path.is_file():
        raise FileNotFoundError(
            f"scene-local pickable USD is missing for {object_id!r}: {usd_path}"
        )
    return usd_path


def validate_relocated_pickables_material_visibility(
    scene_id: str,
    preparation: PhysicalScenePreparation,
    *,
    rules: ManipulationTargetScreeningRules = (MANIPULATION_TARGET_SCREENING_RULES),
    assets_root: Optional[Path | str] = None,
) -> None:
    """Reject relocated pickables whose materials cannot show their colour."""
    for item in preparation.relocated_pickables:
        usd_path = resolve_scene_pickable_instance_usd(
            scene_id,
            item.object_id,
            assets_root=assets_root,
        )
        luminance = authored_base_colour_luminance(usd_path)
        if luminance is None:
            raise ValueError(
                f"relocated pickable material is unreadable: {item.object_id!r}"
            )
        if luminance < rules.min_authored_base_colour_luminance:
            raise ValueError(
                f"relocated pickable failed material-visibility screening: "
                f"{item.object_id!r} authored_base_colour_luminance="
                f"{luminance:.4f} is below threshold "
                f"{rules.min_authored_base_colour_luminance:.4f}"
            )


__all__ = [
    "MANIPULATION_TARGET_SCREENING_RULES",
    "ManipulationTargetScreening",
    "ManipulationTargetScreeningFailure",
    "ManipulationTargetScreeningRules",
    "RejectedManipulationTarget",
    "ScreenedManipulationTarget",
    "has_base_colour_texture",
    "resolve_scene_pickable_instance_usd",
    "screen_manipulation_targets",
    "authored_base_colour_luminance",
    "validate_relocated_pickables_material_visibility",
]
