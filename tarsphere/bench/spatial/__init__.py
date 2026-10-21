"""Spatial contracts for turning semantic episodes into physical tasks.

Spatial solvers are exposed through this package but loaded from their concrete
modules only when requested. Keeping package initialization lightweight avoids
pulling live certification back into a partially imported route compiler.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


_EXPORT_MODULES = {
    # Stored episode-route contracts.
    **{
        name: ".episode_route"
        for name in (
            "MilestoneRouteStop",
            "EpisodeRoute",
        )
    },
    # Robot navigation, manipulation, screening, and route evidence.
    **{
        name: ".physical_task"
        for name in (
            "MANIPULATION_TARGET_SCREENING_RULES",
            "PHYSICAL_TASK_ROUTE_SCHEMA_VERSION",
            "GraspAction",
            "ManipulationTargetCandidate",
            "ManipulationTargetScreening",
            "ManipulationTargetScreeningFailure",
            "ManipulationTargetScreeningRules",
            "ManipulationTargetSet",
            "NavigationPoint",
            "NavigationRoute",
            "PlacementDestinationCandidate",
            "PlacementDestinationSet",
            "RejectedManipulationTarget",
            "ScreenedManipulationTarget",
            "STATIC_PLACEMENT_SUPPORT_CATEGORIES",
            "PhysicalTaskRoute",
            "PlaceAction",
            "StaticNavigationEvidence",
            "StaticNavigationSegment",
            "TargetApproachResolutionError",
            "TargetApproachRoute",
            "has_base_colour_texture",
            "discover_manipulation_targets",
            "discover_object_placement_destinations",
            "discover_placement_destinations",
            "discover_source_supports",
            "resolve_scene_pickable_instance_usd",
            "resolve_static_navigation",
            "resolve_target_approach_route",
            "screen_manipulation_targets",
            "authored_base_colour_luminance",
            "validate_relocated_pickables_material_visibility",
            "validate_route_against_task",
        )
    },
}

__all__ = sorted(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    """Load the module that owns one requested spatial contract."""

    module_name = _EXPORT_MODULES.get(name)
    if module_name is None:
        raise AttributeError(name)
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    """Expose the stable public API to introspection without eager imports."""

    return sorted((*globals(), *__all__))
