"""Scene-specific spatial lighting repairs for shipped simulation assets.

This module is deliberately narrow.  A repair is keyed by an exact scene
asset id and authors fixed world-space fixtures; it never changes camera
exposure, follows the robot, or alters scenes that are not listed here.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import PurePath
from typing import Any


SCENE_LIGHTING_ROOT = "/World/TARSPHERE_SceneSpatialLighting"
_DARK_STUDY_SCENE_ID = "MVUCSQAKTKJ5EAABAAAAABY8_usd"
_DARK_HOME_ADQ8_SCENE_ID = "MVUCSQAKTKJ5EAABAAAAADQ8_usd"
_OUTPATIENT_CLINIC_SCENE_ID = "MV5M25QKTKJZ2AABAAAAAAY8_usd"
_FULL_SUPERMARKET_SCENE_ID = "MWF4WLIKTIFZIAABAAAAACY8_usd"
_CHILDREN_ACTIVITY_CAFE_SCENE_ID = "MWF4WLIKTIFZIAABAAAAADI8_usd"
_FULL_SERVICE_RESTAURANT_SCENE_ID = "MWHLEPQKTIFZIAABAAAAAAA8_usd"
_SHARED_OFFICE_LOUNGE_SCENE_ID = "MVJWVGYKTLDAYAABAAAAAAQ8_usd"
_CONNECTED_ACTIVITY_CENTER_SCENE_ID = "MV5M25QKTKJZ2AABAAAAAAA8_usd"
_ORIGINAL_INTENSITY_ATTR = "tarsphere:sourceIntensity"


def is_scene_authored_light_path(path: str) -> bool:
    """Recognize source USD lights and their TARSphere runtime scene path."""

    if path.startswith("/Root/lights/"):
        return True
    parts = path.split("/")
    return (
        len(parts) > 5
        and parts[1] == "World"
        and parts[2].startswith("env_")
        and parts[2][4:].isdigit()
        and parts[3:5] == ["scene", "lights"]
    )


@dataclass(frozen=True)
class RectLightSpec:
    """One fixed, downward-facing ceiling fixture in metric world space."""

    name: str
    position_m: tuple[float, float, float]
    width_m: float
    height_m: float
    intensity: float


@dataclass(frozen=True)
class SceneLightingRepair:
    """Exact-scene source-light correction plus optional fixed fixtures."""

    authored_intensity_scale: float = 1.0
    authored_intensity_cap: float | None = None
    fixtures: tuple[RectLightSpec, ...] = ()


def _ceiling_grid(
    *,
    prefix: str,
    x_positions_m: tuple[float, ...],
    y_positions_m: tuple[float, ...],
    z_m: float,
    width_m: float,
    height_m: float,
    intensity: float,
) -> tuple[RectLightSpec, ...]:
    """Build one deterministic row-major fixed ceiling array."""

    return tuple(
        RectLightSpec(
            name=f"{prefix}R{row + 1}C{column + 1}",
            position_m=(x_m, y_m, z_m),
            width_m=width_m,
            height_m=height_m,
            intensity=intensity,
        )
        for row, y_m in enumerate(y_positions_m)
        for column, x_m in enumerate(x_positions_m)
    )


# The study spans roughly x=[-0.49, 2.72], y=[-5.24, -0.25].
# The shipped scene has window and cabinet-interior lights but no ceiling-area
# illumination over the room's navigable centre. The adjoining dining
# room has only a low pendant/fill at z=1.55 m, leaving its Human handover area
# in darkness when viewed through the study doorway.  Fixed ceiling fixtures
# repair each room independently so walls retain their natural occlusion.
_ABY8_SPATIAL_LIGHTS = (
    RectLightSpec(
        name="StudyNorthCeiling",
        position_m=(0.65, -1.55, 2.30),
        width_m=0.75,
        height_m=1.10,
        intensity=32_000.0,
    ),
    RectLightSpec(
        name="StudySouthCeiling",
        position_m=(1.05, -3.65, 2.30),
        width_m=0.75,
        height_m=1.10,
        intensity=32_000.0,
    ),
    RectLightSpec(
        name="DiningEastCeiling",
        position_m=(-1.75, -2.15, 2.30),
        width_m=0.75,
        height_m=1.10,
        intensity=22_500.0,
    ),
)

# The reported overexposed commercial scenes contain pathological source
# fixtures: many 107k lights, duplicated light sets, or one 18M outlier.  The
# exact-scene repairs below attenuate those peaks, then use low-output arrays
# to restore uniform visibility under the same exposure used by evaluation.
_AAY8_SPATIAL_LIGHTS = _ceiling_grid(
    prefix="Clinic",
    x_positions_m=(-4.0, 1.0, 6.0),
    y_positions_m=(-6.5, -1.8),
    z_m=2.55,
    width_m=4.50,
    height_m=4.50,
    intensity=800.0,
)

# Full supermarket ACY8 has a highly non-uniform authored layout: seventeen
# RectLights exceed 100k intensity around checkout while the western aisles
# have no comparable ceiling coverage.  A low-output 4x2 ceiling array covers
# the complete floor instead of concentrating three strong fixtures in one
# aisle.  Source lights are attenuated and capped by the repair below.
_ACY8_SPATIAL_LIGHTS = _ceiling_grid(
    prefix="Supermarket",
    x_positions_m=(-5.5, 0.5, 6.5, 12.5),
    y_positions_m=(-3.8, 2.8),
    z_m=3.35,
    width_m=5.50,
    height_m=5.20,
    intensity=800.0,
)

_ADI8_SPATIAL_LIGHTS = _ceiling_grid(
    prefix="ActivityCafe",
    x_positions_m=(-2.7, 1.2),
    y_positions_m=(-5.5, -1.3, 2.9),
    z_m=2.55,
    width_m=3.60,
    height_m=3.80,
    intensity=800.0,
)

_AAA8_SPATIAL_LIGHTS = _ceiling_grid(
    prefix="Restaurant",
    x_positions_m=(-2.4, 0.3),
    y_positions_m=(-5.5, -2.7, 0.1),
    z_m=2.55,
    width_m=2.50,
    height_m=2.50,
    intensity=800.0,
)

_AAQ8_SPATIAL_LIGHTS = _ceiling_grid(
    prefix="Office",
    x_positions_m=(-3.1, 0.1),
    y_positions_m=(-9.2, -5.9, -2.6, 0.7),
    z_m=3.20,
    width_m=3.00,
    height_m=3.00,
    intensity=800.0,
)

# The composed evaluation scene has 43 lights, concentrated in the western
# lounge and meeting room. Its eastern coworking desks, connecting corridor,
# and southern activity room have no ceiling fixtures. The measured ceiling
# underside is z=2.70 m; these fixed room-local panels sit below it. Preserve
# existing illumination in the already lit western rooms.
_CONNECTED_ACTIVITY_SPATIAL_LIGHTS = (
    *_ceiling_grid(
        prefix="Coworking",
        x_positions_m=(-10.0, -7.0),
        y_positions_m=(11.1,),
        z_m=2.55,
        width_m=2.50,
        height_m=2.50,
        intensity=1_600.0,
    ),
    *_ceiling_grid(
        prefix="ConnectingCorridor",
        x_positions_m=(-6.65,),
        y_positions_m=(4.2, 1.6),
        z_m=2.55,
        width_m=1.20,
        height_m=2.20,
        intensity=3_000.0,
    ),
    *_ceiling_grid(
        prefix="SouthernActivity",
        x_positions_m=(-4.4, -1.5),
        y_positions_m=(-1.0,),
        z_m=2.55,
        width_m=2.40,
        height_m=2.40,
        intensity=1_600.0,
    ),
)

# ADQ8 raw USD authors zero UsdLux lights. Fixed room-local fixtures must
# cover the bedrooms and balconies as well as the central navigation spine;
# a light across a wall does not illuminate the neighbouring room.
_ADQ8_SPATIAL_LIGHTS = (
    RectLightSpec(
        name="LivingSouthCeiling",
        position_m=(1.68, -2.92, 2.25),
        width_m=1.40,
        height_m=1.80,
        intensity=28_000.0,
    ),
    RectLightSpec(
        name="LivingNorthCeiling",
        position_m=(1.30, 1.75, 2.25),
        width_m=1.20,
        height_m=1.40,
        intensity=24_000.0,
    ),
    RectLightSpec(
        name="CorridorCeiling",
        position_m=(3.51, 0.37, 2.25),
        width_m=0.80,
        height_m=2.00,
        intensity=20_000.0,
    ),
    RectLightSpec(
        name="WestStandCeiling",
        position_m=(4.59, -1.56, 2.25),
        width_m=1.20,
        height_m=1.40,
        intensity=24_000.0,
    ),
    RectLightSpec(
        name="EastBedroomCeiling",
        position_m=(-1.25, -2.17, 2.25),
        width_m=1.20,
        height_m=1.40,
        intensity=24_000.0,
    ),
    RectLightSpec(
        name="SouthWindowCeiling",
        position_m=(2.16, 0.88, 2.25),
        width_m=1.20,
        height_m=1.40,
        intensity=22_000.0,
    ),
    RectLightSpec(
        name="NorthwestBedroomCeiling",
        position_m=(-2.25, 2.65, 2.25),
        width_m=1.60,
        height_m=1.80,
        intensity=24_000.0,
    ),
    RectLightSpec(
        name="NortheastBedroomCeiling",
        position_m=(7.0, 2.65, 2.25),
        width_m=1.60,
        height_m=1.80,
        intensity=24_000.0,
    ),
    RectLightSpec(
        name="NorthBathroomCeiling",
        position_m=(4.85, 3.05, 2.25),
        width_m=1.0,
        height_m=1.20,
        intensity=20_000.0,
    ),
    RectLightSpec(
        name="SouthBalconyCeiling",
        position_m=(1.35, -7.25, 2.25),
        width_m=1.40,
        height_m=1.20,
        intensity=20_000.0,
    ),
)


def scene_id_from_asset_path(asset_path: str) -> str:
    """Return the exact GRScenes directory id, or an empty string."""

    for part in PurePath(str(asset_path)).parts:
        if part.endswith("_usd"):
            return part
    return ""


def spatial_lighting_for_asset(asset_path: str) -> tuple[RectLightSpec, ...]:
    """Resolve the one authoritative lighting repair for ``asset_path``."""

    return scene_lighting_repair_for_asset(asset_path).fixtures


def scene_lighting_repair_for_asset(asset_path: str) -> SceneLightingRepair:
    """Return the exact current lighting contract for one scene asset."""

    scene_id = scene_id_from_asset_path(asset_path)
    if scene_id == _DARK_STUDY_SCENE_ID:
        return SceneLightingRepair(fixtures=_ABY8_SPATIAL_LIGHTS)
    if scene_id == _DARK_HOME_ADQ8_SCENE_ID:
        return SceneLightingRepair(fixtures=_ADQ8_SPATIAL_LIGHTS)
    if scene_id == _CONNECTED_ACTIVITY_CENTER_SCENE_ID:
        return SceneLightingRepair(fixtures=_CONNECTED_ACTIVITY_SPATIAL_LIGHTS)
    if scene_id == _OUTPATIENT_CLINIC_SCENE_ID:
        return SceneLightingRepair(
            authored_intensity_scale=0.08,
            authored_intensity_cap=12_000.0,
            fixtures=_AAY8_SPATIAL_LIGHTS,
        )
    if scene_id == _FULL_SUPERMARKET_SCENE_ID:
        return SceneLightingRepair(
            authored_intensity_scale=0.08,
            authored_intensity_cap=12_000.0,
            fixtures=_ACY8_SPATIAL_LIGHTS,
        )
    if scene_id == _CHILDREN_ACTIVITY_CAFE_SCENE_ID:
        return SceneLightingRepair(
            authored_intensity_scale=0.06,
            authored_intensity_cap=12_000.0,
            fixtures=_ADI8_SPATIAL_LIGHTS,
        )
    if scene_id == _FULL_SERVICE_RESTAURANT_SCENE_ID:
        return SceneLightingRepair(
            authored_intensity_scale=0.06,
            authored_intensity_cap=12_000.0,
            fixtures=_AAA8_SPATIAL_LIGHTS,
        )
    if scene_id == _SHARED_OFFICE_LOUNGE_SCENE_ID:
        return SceneLightingRepair(
            authored_intensity_scale=0.05,
            authored_intensity_cap=12_000.0,
            fixtures=_AAQ8_SPATIAL_LIGHTS,
        )
    return SceneLightingRepair()


def apply_scene_spatial_lighting(stage: Any, asset_path: str) -> tuple[str, ...]:
    """Author idempotent fixed fixtures for a known defective scene.

    Positions and dimensions in the manifest are metres.  They are converted
    to the live stage's declared units so the contract stays correct if the
    runtime root layer changes its ``metersPerUnit`` metadata.
    """

    repair = scene_lighting_repair_for_asset(asset_path)
    specs = repair.fixtures
    needs_source_adjustment = (
        repair.authored_intensity_scale != 1.0
        or repair.authored_intensity_cap is not None
    )
    if not specs and not needs_source_adjustment:
        return ()
    if stage is None:
        raise RuntimeError("Cannot apply scene spatial lighting without a USD stage.")

    from pxr import Gf, Sdf, UsdGeom, UsdLux

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    if not meters_per_unit > 0.0:
        raise RuntimeError(
            "Scene spatial lighting requires a positive stage metersPerUnit, "
            f"got {meters_per_unit!r}."
        )
    world = stage.GetPrimAtPath("/World")
    if not world.IsValid():
        # GRScenes BEV opens the authored layer whose default prim is /Root.
        # Runtime TARSphere stages already have /World.  Define the same
        # sibling root so BEV and episode use one fixture path.
        UsdGeom.Xform.Define(stage, "/World")

    if needs_source_adjustment:
        for prim in stage.Traverse():
            prim_path = str(prim.GetPath())
            if not is_scene_authored_light_path(
                prim_path
            ) or not prim.GetTypeName().endswith("Light"):
                continue
            intensity_attr = prim.GetAttribute("inputs:intensity")
            if not intensity_attr or intensity_attr.Get() is None:
                continue
            source_attr = prim.GetAttribute(_ORIGINAL_INTENSITY_ATTR)
            if not source_attr:
                source_attr = prim.CreateAttribute(
                    _ORIGINAL_INTENSITY_ATTR,
                    Sdf.ValueTypeNames.Double,
                    custom=True,
                )
            if source_attr.Get() is None:
                source_attr.Set(float(intensity_attr.Get()))
            source_intensity = float(source_attr.Get())
            repaired_intensity = source_intensity * repair.authored_intensity_scale
            if repair.authored_intensity_cap is not None:
                repaired_intensity = min(
                    repaired_intensity,
                    repair.authored_intensity_cap,
                )
            intensity_attr.Set(float(repaired_intensity))

    UsdGeom.Xform.Define(stage, SCENE_LIGHTING_ROOT)
    paths: list[str] = []
    for spec in specs:
        light_path = f"{SCENE_LIGHTING_ROOT}/{spec.name}"
        light = UsdLux.RectLight.Define(stage, light_path)
        light.CreateIntensityAttr(float(spec.intensity))
        light.CreateExposureAttr(0.0)
        light.CreateNormalizeAttr(False)
        light.CreateWidthAttr(float(spec.width_m / meters_per_unit))
        light.CreateHeightAttr(float(spec.height_m / meters_per_unit))
        prim = light.GetPrim()
        xform = UsdGeom.Xformable(prim)
        translate = tuple(value / meters_per_unit for value in spec.position_m)
        existing_translate = next(
            (
                op
                for op in xform.GetOrderedXformOps()
                if op.GetOpType() == UsdGeom.XformOp.TypeTranslate
            ),
            None,
        )
        translate_op = existing_translate or xform.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*translate))
        paths.append(light_path)
    return tuple(paths)


__all__ = [
    "RectLightSpec",
    "SCENE_LIGHTING_ROOT",
    "SceneLightingRepair",
    "apply_scene_spatial_lighting",
    "scene_lighting_repair_for_asset",
    "scene_id_from_asset_path",
    "spatial_lighting_for_asset",
]
