"""Compile visualization trajectories from one current runtime episode."""

from __future__ import annotations

import math
from typing import Any, Literal, Mapping, Tuple

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.bench.certification.scene import (
    load_episode_occupancy,
    official_runtime_navigation_mask,
)
from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import Planner
from tarsphere.bench.schema import TARSphereEpisode
from tarsphere.evaluation.schema import MilestoneCondition
from tarsphere.geometry import HUMAN_BODY_RADIUS_M
from tarsphere.infra.episode.human_actor_runtime import (
    resolve_human_action_point,
)
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
)


Point2 = Tuple[float, float]


class StrictModel(BaseModel):
    """Immutable current-release trajectory evidence base."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class ActorTrajectory(StrictModel):
    """One occupancy-resolved Robot or Human trajectory."""

    actor: str = Field(
        min_length=1,
        description="Human-readable Robot or Human name.",
    )
    actor_type: Literal["robot", "human"] = Field(
        description="Embodied actor that follows the trajectory.",
    )
    mode: Literal["task", "static", "dynamic"] = Field(
        description="Whether the actor performs the task, stays still, or moves.",
    )
    points_xy: Tuple[Point2, ...] = Field(
        min_length=1,
        description="Occupancy-resolved polyline in scene world metres.",
    )
    control_points_xy: Tuple[Point2, ...] = Field(
        min_length=1,
        description="Sparse authored points that the resolved path must visit.",
    )
    route_labels: Tuple[str, ...] = Field(
        min_length=1,
        description="Readable label for every authored control point.",
    )
    path_length_m: float = Field(
        ge=0.0,
        description="Length of the resolved polyline in metres.",
    )

    @model_validator(mode="after")
    def validate_control_labels(self) -> "ActorTrajectory":
        if len(self.control_points_xy) != len(self.route_labels):
            raise ValueError("Every trajectory control point requires one route label.")
        if self.mode == "static" and len(self.points_xy) != 1:
            raise ValueError("Static Human trajectory must contain one point.")
        return self


class MilestoneHumanPosition(StrictModel):
    """Human position when one milestone opens, before its actions run."""

    human_id: str = Field(
        min_length=1,
        description="Episode-local Human identifier referenced by the event.",
    )
    human_name: str = Field(
        min_length=1,
        description="Human-readable actor name shown on the BEV.",
    )
    position_xy: Point2 = Field(
        description="Human world position when the milestone opens.",
    )


class MilestoneEventMarker(StrictModel):
    """One authored milestone bound to its Robot and Human BEV positions."""

    sequence_index: int = Field(
        ge=1,
        description="One-based causal order of the milestone in the episode.",
    )
    milestone_id: str = Field(
        min_length=1,
        description="Release-stable semantic milestone identifier.",
    )
    milestone_name: str = Field(
        min_length=1,
        description="Readable milestone event name shown in the side panel.",
    )
    route_point_id: str = Field(
        min_length=1,
        description="Episode-route point that spatially realizes the event.",
    )
    robot_position_xy: Point2 = Field(
        description="Robot world position when the milestone becomes reachable.",
    )
    human_positions: Tuple[MilestoneHumanPosition, ...] = Field(
        description=(
            "Referenced Humans at their pre-action positions when the event "
            "opens; empty for Robot-only milestones."
        ),
    )
    event_actions: Tuple[str, ...] = Field(
        description=(
            "Ordered Human commands and emitted semantic events authored at "
            "this milestone."
        ),
    )


class EpisodeTrajectories(StrictModel):
    """Visualization evidence for every embodied actor in one episode."""

    scene_id: str = Field(
        min_length=1,
        description="Compiled scene containing the trajectories.",
    )
    trajectories: Tuple[ActorTrajectory, ...] = Field(
        min_length=2,
        description="Robot trajectory followed by every episode Human.",
    )
    milestone_events: Tuple[MilestoneEventMarker, ...] = Field(
        min_length=1,
        description=(
            "Causally ordered milestone events projected onto exact Robot "
            "and Human world positions."
        ),
    )


def compile_episode_trajectories(
    episode: TARSphereEpisode,
) -> EpisodeTrajectories:
    """Resolve the episode-specific Robot and Human routes over occupancy."""
    world = episode.world_knowledge or {}
    scene_id = str(world.get("scene_id") or "").strip()
    episode_route = world.get("episode_route")
    if not scene_id:
        raise ValueError("Compiled episode is missing world_knowledge.scene_id.")
    if not isinstance(episode_route, dict):
        raise ValueError("Compiled episode is missing world_knowledge.episode_route.")

    occupancy = load_episode_occupancy(episode)
    robot_controls, robot_labels = _robot_controls(episode_route)
    robot_points = _resolve_path(
        occupancy,
        robot_controls,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        prepared_navigable_mask=official_runtime_navigation_mask(
            occupancy.occupancy_path,
            clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
            static_obstacles=tuple(
                (
                    float(h.start_position[0]),
                    float(h.start_position[1]),
                    HUMAN_BODY_RADIUS_M,
                )
                for h in episode.humans
                if h.static
            ),
        ),
    )
    trajectories = [
        ActorTrajectory(
            actor=episode.robot.entity_name,
            actor_type="robot",
            mode="task",
            points_xy=robot_points,
            control_points_xy=robot_controls,
            route_labels=robot_labels,
            path_length_m=_path_length(robot_points),
        )
    ]
    trajectories.extend(_human_trajectories(episode, occupancy))
    return EpisodeTrajectories(
        scene_id=scene_id,
        trajectories=tuple(trajectories),
        milestone_events=_milestone_event_markers(episode, episode_route),
    )


def _milestone_event_markers(
    episode: TARSphereEpisode,
    episode_route: Mapping[str, Any],
) -> tuple[MilestoneEventMarker, ...]:
    """Bind every milestone to the exact positions visible on the BEV."""
    route_points = {
        str(point["point_id"]): tuple(
            float(value) for value in point["position_xy"][:2]
        )
        for phase in ("navigate_to_target", "transport_to_destination")
        for point in episode_route[phase]["points"]
    }
    milestone_stops = {
        str(stop["milestone_id"]): str(stop["route_point_id"])
        for stop in episode_route.get("milestone_stops", ())
    }
    humans = {str(human.id): human for human in episode.humans}
    current_positions = {
        human_id: tuple(float(value) for value in human.start_position[:2])
        for human_id, human in humans.items()
    }
    markers = []
    evaluation = episode.evaluation
    if evaluation is None:
        raise ValueError("Trajectory rendering requires evaluation.milestones.")
    for sequence_index, milestone in enumerate(evaluation.milestones, 1):
        milestone_id = str(milestone.milestone_id)
        route_point_id = milestone_stops.get(milestone_id)
        if route_point_id is None:
            raise ValueError(
                f"Milestone {milestone_id!r} has no episode-route binding."
            )
        if route_point_id not in route_points:
            raise ValueError(
                f"Milestone {milestone_id!r} references unknown route point "
                f"{route_point_id!r}."
            )
        conditions = milestone.preconditions
        human_ids = {
            str(condition.arguments["human_id"])
            for condition in conditions
            if isinstance(condition, MilestoneCondition)
            if condition.arguments.get("human_id") not in (None, "")
        }
        human_ids.update(
            str(action.get("human"))
            for action in milestone.event.human_actions
            if action.get("human") not in (None, "")
        )
        reaction_human = milestone.outcome.ground_truth.get("human_id")
        if reaction_human not in (None, ""):
            human_ids.add(str(reaction_human))
        unknown_humans = sorted(human_ids - humans.keys())
        if unknown_humans:
            raise ValueError(
                f"Milestone {milestone_id!r} references unknown Humans: "
                f"{unknown_humans}"
            )
        human_positions = tuple(
            MilestoneHumanPosition(
                human_id=human_id,
                human_name=str(humans[human_id].name or human_id),
                position_xy=current_positions[human_id],
            )
            for human_id in sorted(human_ids)
        )
        actions = tuple(
            str(action.get("command") or "unknown")
            for action in milestone.event.human_actions
        ) + tuple(
            f"task_update:{update.field}" for update in milestone.event.task_updates
        )
        markers.append(
            MilestoneEventMarker(
                sequence_index=sequence_index,
                milestone_id=milestone_id,
                milestone_name=milestone.event.description,
                route_point_id=route_point_id,
                robot_position_xy=route_points[route_point_id],
                human_positions=human_positions,
                event_actions=actions,
            )
        )
        for action in milestone.event.human_actions:
            if str(action.get("command") or "") != "goto":
                continue
            human_id = str(action["human"])
            current_positions[human_id] = tuple(
                float(value)
                for value in resolve_human_action_point(
                    episode,
                    action.get("target"),
                )[:2]
            )
    return tuple(markers)


def _robot_controls(
    episode_route: dict[str, Any],
) -> tuple[tuple[Point2, ...], tuple[str, ...]]:
    raw_points = [
        *episode_route["navigate_to_target"]["points"],
        *episode_route["transport_to_destination"]["points"],
    ]
    controls: list[Point2] = []
    labels: list[str] = []
    for item in raw_points:
        point = tuple(float(value) for value in item["position_xy"][:2])
        if controls and math.dist(controls[-1], point) <= 1e-9:
            continue
        controls.append(point)
        labels.append(str(item["point_id"]).replace("_", " "))
    if len(controls) < 2:
        raise ValueError("Compiled Robot task requires at least two route points.")
    return tuple(controls), tuple(labels)


def _human_trajectories(
    episode: TARSphereEpisode,
    occupancy: Any,
) -> list[ActorTrajectory]:
    humans_by_id = {str(human.id): human for human in episode.humans}
    current = {
        human_id: tuple(float(value) for value in human.start_position[:2])
        for human_id, human in humans_by_id.items()
    }
    controls = {human_id: [position] for human_id, position in current.items()}
    labels = {human_id: ["start"] for human_id in humans_by_id}
    paths = {human_id: [position] for human_id, position in current.items()}

    evaluation = episode.evaluation
    if evaluation is None:
        raise ValueError("Trajectory rendering requires evaluation.milestones.")
    for milestone in evaluation.milestones:
        for action in milestone.event.human_actions:
            if str(action.get("command") or "") != "goto":
                continue
            human_id = str(action.get("human") or "")
            if human_id not in humans_by_id:
                raise ValueError(f"Human GoTo references unknown Human {human_id!r}.")
            target = tuple(
                resolve_human_action_point(
                    episode,
                    action.get("target"),
                )[:2]
            )
            exclusions = [
                (
                    position[0],
                    position[1],
                    2.0 * HUMAN_BODY_RADIUS_M,
                )
                for other_id, position in current.items()
                if other_id != human_id
            ]
            segment = _resolve_path(
                occupancy,
                (current[human_id], target),
                # This is occupancy-based review evidence for the physical
                # Human capsule, not a substitute for the later live NavMesh
                # gate.  Inflating it to the NavMesh bake-agent radius here
                # can disconnect ordinary doorways and fabricate an
                # unreachable overlay even though the 0.20 m body fits.
                clearance_radius_m=HUMAN_BODY_RADIUS_M,
                world_exclusion_zones=exclusions,
            )
            paths[human_id].extend(segment[1:])
            controls[human_id].append(target)
            labels[human_id].append(str(milestone.milestone_id))
            current[human_id] = target

    result = []
    for human_id, human in humans_by_id.items():
        is_dynamic = len(controls[human_id]) > 1
        points = tuple(paths[human_id])
        result.append(
            ActorTrajectory(
                actor=str(human.name),
                actor_type="human",
                mode="dynamic" if is_dynamic else "static",
                points_xy=points,
                control_points_xy=tuple(controls[human_id]),
                route_labels=tuple(labels[human_id]),
                path_length_m=_path_length(points),
            )
        )
    return result


def _resolve_path(
    occupancy: Any,
    controls: tuple[Point2, ...],
    *,
    clearance_radius_m: float,
    world_exclusion_zones: list[tuple[float, float, float]] | None = None,
    prepared_navigable_mask: Any = None,
) -> tuple[Point2, ...]:
    if len(controls) == 1:
        return controls
    planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=clearance_radius_m,
        world_exclusion_zones=world_exclusion_zones,
        prepared_navigable_mask=prepared_navigable_mask,
        cache_static_nav=not world_exclusion_zones,
    )
    path: list[Point2] = [controls[0]]
    for start, goal in zip(controls, controls[1:]):
        result = planner.plan(
            PlanInput(
                start_xy=start,
                start_yaw_rad=0.0,
                goal_xy=goal,
            )
        )
        if result.status != "ok":
            raise ValueError(
                f"Occupancy planning failed for {start} -> {goal}: {result.status}"
            )
        for raw_point in (*result.waypoints, goal):
            point = tuple(float(value) for value in raw_point[:2])
            if math.dist(path[-1], point) > 1e-9:
                path.append(point)
    return tuple(path)


def _path_length(points: tuple[Point2, ...]) -> float:
    return sum(math.dist(first, second) for first, second in zip(points, points[1:]))


__all__ = [
    "ActorTrajectory",
    "MilestoneEventMarker",
    "MilestoneHumanPosition",
    "EpisodeTrajectories",
    "compile_episode_trajectories",
]
