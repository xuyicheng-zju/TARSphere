"""Public contracts for shared Robot navigation and manipulation tasks.

The package exports concrete contracts without importing the entire route and
certification graph at package initialization. This matters because live
certification imports individual physical-task modules while those modules, in
turn, refer to certification evidence. Loading only the requested module keeps
that dependency graph acyclic.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    # Object discovery and support candidates.
    **{
        name: ".objects"
        for name in (
            "STATIC_PLACEMENT_SUPPORT_CATEGORIES",
            "ManipulationTargetCandidate",
            "ManipulationTargetSet",
            "PlacementDestinationCandidate",
            "PlacementDestinationSet",
            "discover_manipulation_targets",
            "discover_object_placement_destinations",
            "discover_placement_destinations",
            "discover_source_supports",
        )
    },
    # Static route measurement.
    **{
        name: ".navigation"
        for name in (
            "resolve_static_navigation",
            "validate_route_against_task",
        )
    },
    # Stable physical-task schemas.
    **{
        name: ".schema"
        for name in (
            "PHYSICAL_TASK_ROUTE_SCHEMA_VERSION",
            "GraspAction",
            "NavigationPoint",
            "NavigationRoute",
            "PhysicalTaskRoute",
            "PlaceAction",
            "StaticNavigationEvidence",
            "StaticNavigationSegment",
        )
    },
    # Target-entry route resolution.
    **{
        name: ".target_approach"
        for name in (
            "TargetApproachResolutionError",
            "TargetApproachRoute",
            "resolve_target_approach_route",
        )
    },
    # Manipulation target screening.
    **{
        name: ".target_screening"
        for name in (
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
        )
    },
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load only the concrete module that defines the requested contract."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the stable public API to introspection without eager imports."""

    return sorted((*globals(), *__all__))
