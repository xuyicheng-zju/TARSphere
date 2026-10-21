"""Freeze the model-visible tool surface for one episode.

Tool presence is a run contract, not a representation of mutable world state.
Holding an object, crouching, losing sight of a target, or moving away from a
Human may make a call inappropriate or unsuccessful, but none of those states
may make its schema disappear from the next model input.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from tarsphere.harness.core.types import ToolSpec


LOCOMOTION_TOOLS = frozenset(
    {
        "move_body",
        "navigate_to_pointgoal",
        "navigate_to_frontier",
        "navigate_to_entity",
        "move_forward",
        "move_backward",
        "move_left",
        "move_right",
        "turn_left",
        "turn_right",
    }
)


@dataclass(frozen=True)
class AvailableTools:
    """The ordered, episode-frozen tool surface used by prompt and dispatch."""

    specs: tuple[ToolSpec, ...]
    names: frozenset[str]


def resolve_available_tools(
    *,
    registered_specs: Iterable[ToolSpec],
    episode_actions: Iterable[str] | None,
) -> AvailableTools:
    """Resolve the fixed surface from run configuration only.

    A backend failure later in the rollout is execution feedback and never a
    reason to hide a registered tool.
    """

    specs = tuple(registered_specs)
    registered_names = {spec.name for spec in specs}
    if episode_actions is None:
        admitted = set(registered_names)
    else:
        admitted = {str(name) for name in episode_actions}
        unknown = admitted - registered_names
        if unknown:
            raise ValueError(
                f"Episode action space contains unregistered tools: {sorted(unknown)}."
            )

    available_specs = tuple(spec for spec in specs if spec.name in admitted)
    available_names = frozenset(spec.name for spec in available_specs)
    return AvailableTools(specs=available_specs, names=available_names)


__all__ = [
    "AvailableTools",
    "LOCOMOTION_TOOLS",
    "resolve_available_tools",
]
