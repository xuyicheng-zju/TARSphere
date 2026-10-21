"""Isaac-free authored-light and floor-material inventory for one USD stage."""

from __future__ import annotations

from typing import Any


_FLOOR_NAME_MARKERS = ("floor", "ground", "tile", "terrazzo", "carpet")
_ALBEDO_INPUT_MARKERS = (
    "diffuse",
    "albedo",
    "basecolor",
    "base_color",
    "diffuse_color",
    "diffuse_tint",
)


def _vec_values(value: Any) -> list[float] | None:
    if value is None:
        return None
    if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
        try:
            return [float(item) for item in value]
        except (TypeError, ValueError):
            return None
    try:
        return [float(value)]
    except (TypeError, ValueError):
        return None


def inventory_stage_lights(stage: Any) -> list[dict[str, Any]]:
    """Return every UsdLux light prim with enablement, intensity, and pose."""

    from pxr import Usd, UsdGeom, UsdLux

    light_types = (
        UsdLux.DomeLight,
        UsdLux.RectLight,
        UsdLux.DistantLight,
        UsdLux.SphereLight,
        UsdLux.DiskLight,
        UsdLux.CylinderLight,
    )
    lights: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        is_light = prim.HasAPI(UsdLux.LightAPI) or any(
            prim.IsA(light_type) for light_type in light_types
        )
        if not is_light:
            continue
        api = UsdLux.LightAPI(prim)
        intensity_attr = (
            api.GetIntensityAttr() if api else prim.GetAttribute("inputs:intensity")
        )
        exposure_attr = (
            api.GetExposureAttr() if api else prim.GetAttribute("inputs:exposure")
        )
        xform = UsdGeom.Xformable(prim)
        translation = None
        if xform:
            matrix = xform.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
            translation = [float(v) for v in matrix.ExtractTranslation()]
        lights.append(
            {
                "path": str(prim.GetPath()),
                "type": prim.GetTypeName(),
                "active": bool(prim.IsActive()),
                "intensity": float(intensity_attr.Get() or 0.0)
                if intensity_attr
                else None,
                "exposure": float(exposure_attr.Get() or 0.0)
                if exposure_attr
                else None,
                "translation": translation,
            }
        )
    return lights


def _looks_like_floor(prim: Any, bmin: Any, bmax: Any) -> bool:
    name = str(prim.GetPath()).lower()
    if any(marker in name for marker in _FLOOR_NAME_MARKERS):
        return True
    if bmin is None or bmax is None:
        return False
    z_span = float(bmax[2] - bmin[2])
    max_z = float(bmax[2])
    xy_span = max(float(bmax[0] - bmin[0]), float(bmax[1] - bmin[1]))
    return z_span <= 0.20 and max_z <= 0.35 and xy_span >= 2.0


def _bound_material_record(prim: Any) -> dict[str, Any]:
    from pxr import UsdShade

    binding = UsdShade.MaterialBindingAPI(prim)
    bound = binding.ComputeBoundMaterial()
    material = bound[0] if isinstance(bound, tuple) else bound
    if material is None or not material:
        return {"material_path": None, "albedo_inputs": []}
    shader = material.ComputeSurfaceSource()
    if isinstance(shader, tuple):
        shader = shader[0]
    albedo_inputs: list[dict[str, Any]] = []
    if shader:
        for input_api in shader.GetInputs():
            input_name = str(input_api.GetBaseName()).lower()
            if not any(marker in input_name for marker in _ALBEDO_INPUT_MARKERS):
                continue
            albedo_inputs.append(
                {
                    "name": input_api.GetBaseName(),
                    "value": _vec_values(input_api.Get()),
                }
            )
    return {
        "material_path": str(material.GetPath()),
        "albedo_inputs": albedo_inputs,
    }


def inventory_floor_materials(
    stage: Any, *, meters_per_unit: float
) -> list[dict[str, Any]]:
    """Return floor-like meshes and any bound diffuse/albedo inputs.

    A dark floor albedo is a description, not automatically a defect.
    """

    from pxr import Usd, UsdGeom

    if meters_per_unit <= 0:
        raise ValueError(f"meters_per_unit must be positive, got {meters_per_unit!r}.")
    floors: list[dict[str, Any]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Boundable):
            continue
        boundable = UsdGeom.Boundable(prim)
        extent = UsdGeom.Boundable.ComputeExtentFromPlugins(
            boundable, Usd.TimeCode.Default()
        )
        if not extent:
            continue
        bmin = [float(v) * meters_per_unit for v in extent[0]]
        bmax = [float(v) * meters_per_unit for v in extent[1]]
        if not _looks_like_floor(prim, bmin, bmax):
            continue
        record = {
            "path": str(prim.GetPath()),
            "type": prim.GetTypeName(),
            "bounds_min_m": [round(v, 4) for v in bmin],
            "bounds_max_m": [round(v, 4) for v in bmax],
        }
        record.update(_bound_material_record(prim))
        floors.append(record)
    return floors


def summarize_scene_lighting(stage: Any) -> dict[str, Any]:
    """Return lights, floor materials, and metersPerUnit for one opened stage."""

    from pxr import UsdGeom

    meters_per_unit = float(UsdGeom.GetStageMetersPerUnit(stage))
    lights = inventory_stage_lights(stage)
    floors = inventory_floor_materials(stage, meters_per_unit=meters_per_unit)
    active_lights = [light for light in lights if light["active"]]
    return {
        "meters_per_unit": meters_per_unit,
        "light_count": len(lights),
        "active_light_count": len(active_lights),
        "lights": lights,
        "floor_prim_count": len(floors),
        "floors": floors,
    }


__all__ = [
    "inventory_floor_materials",
    "inventory_stage_lights",
    "summarize_scene_lighting",
]
