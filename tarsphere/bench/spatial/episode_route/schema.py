"""Typed contract for one complete benchmark Robot route."""

from __future__ import annotations

from collections import Counter
from typing import Annotated, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.spatial.physical_task.schema import (
    NavigationRoute,
)


Identifier = Annotated[
    str,
    Field(
        pattern=r"^[a-z][a-z0-9_]*$",
        description="Stable lowercase milestone or route-point identifier.",
    ),
]


class StrictModel(BaseModel):
    """Immutable current-release episode-route schema base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class MilestoneRouteStop(StrictModel):
    """Identify the Robot route point that realizes one causal milestone."""

    milestone_id: Identifier = Field(
        description="Semantic milestone that requires the Robot stop.",
    )
    route_point_id: Identifier = Field(
        description=(
            "Point in this EpisodeRoute where the milestone becomes "
            "spatially reachable."
        ),
    )


class EpisodeRoute(StrictModel):
    """Complete Robot route through one episode's required task stops."""

    navigate_to_target: NavigationRoute = Field(
        description=(
            "Unladen route from the shared Robot start through every required "
            "pre-grasp social stop to the grasp approach."
        ),
    )
    transport_to_destination: NavigationRoute = Field(
        description=(
            "Loaded route from the grasp retreat through every required "
            "transport-stage social stop to the placement approach."
        ),
    )
    milestone_stops: Tuple[MilestoneRouteStop, ...] = Field(
        default=(),
        description=(
            "Causally ordered route bindings for authored milestones. A "
            "Human-free physical task has no milestone stops; active Human "
            "interactions may add encounter route points."
        ),
    )
    optimal_navigation_length_m: float = Field(
        gt=0.0,
        description=(
            "Shortest collision-free length through all physical and social "
            "route points in their compiled causal order."
        ),
    )

    @model_validator(mode="after")
    def validate_milestone_stops(self) -> "EpisodeRoute":
        milestone_ids = tuple(item.milestone_id for item in self.milestone_stops)
        duplicates = sorted(
            value for value, count in Counter(milestone_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate milestone route bindings: {duplicates}")
        route_point_ids = {
            point.point_id
            for route in (
                self.navigate_to_target,
                self.transport_to_destination,
            )
            for point in route.points
        }
        unknown = sorted(
            item.route_point_id
            for item in self.milestone_stops
            if item.route_point_id not in route_point_ids
        )
        if unknown:
            raise ValueError(
                f"milestone bindings reference unknown route points: {unknown}"
            )
        return self


__all__ = [
    "MilestoneRouteStop",
    "EpisodeRoute",
]
