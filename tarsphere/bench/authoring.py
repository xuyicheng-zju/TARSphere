"""Strict evidence gates for publishable benchmark candidates.

The base episode schema also serves component diagnostics. This module is the
smaller publication profile used by coding agents and CI for benchmark data.
It validates evidence; it never invents a route, story, or task binding.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from tarsphere.harness.perception.backends.gt_bev import (
    OfficialOccupancyBevMap,
)
from tarsphere.harness.planner.contracts import PlanInput
from tarsphere.harness.planner.planner import (
    Planner,
    _supercover_world_segment_cells,
)
from tarsphere.bench.certification.scene import official_runtime_navigation_mask
from tarsphere.bench.difficulty import derive_episode_difficulty
from tarsphere.bench.schema import AuthoringGate, TARSphereEpisode
from tarsphere.bench.spatial.episode_route.schema import MilestoneRouteStop
from tarsphere.bench.spatial.physical_task.navigation_evidence import (
    complete_path,
    path_length,
    route_reaches_exact_goal,
)
from tarsphere.contracts.human_interaction import HUMAN_COMMUNICATION_RADIUS_M
from tarsphere.contracts.task_profile import CapabilitySlice, TaskMainline
from tarsphere.evaluation.schema import MilestoneReference
from tarsphere.evaluation.task_bindings import final_outcome_ground_truth
from tarsphere.infra.control.physical_contracts import (
    G1_NAVIGATION_FOOTPRINT_RADIUS_M,
    HANDOVER_MAX_DISTANCE_M,
    HANDOVER_MIN_DISTANCE_M,
    HUMAN_BODY_RADIUS_M,
    ROBOT_HUMAN_CENTER_CLEARANCE_M,
)

DEPRECATED_WORLD_KEYS = frozenset(
    {
        "family_evidence_contract",
        "physical_task_candidate",
        "representative_route",
        "semantic_task",
    }
)

AUTHORITATIVE_STANCE_TOLERANCE_M = 0.002
MAX_SEMANTIC_OBSERVATION_STANCE_DISTANCE_M = 4.0


class CandidateSceneMap(BaseModel):
    """Portable map inputs required by the candidate geometry gate."""

    model_config = ConfigDict(extra="allow")

    rgb_bev: str
    projection: str
    occupancy: str


class CandidateRoutePoint(BaseModel):
    model_config = ConfigDict(extra="forbid")

    point_id: str = Field(pattern=r"^[a-z][a-z0-9_]*$")
    position_xy: tuple[float, float]
    semantic_anchor: str | None = None
    anchor_kind: (
        Literal[
            "robot_start",
            "grasp_stance",
            "human_position",
            "detector_position",
            "object_position",
            "scene_anchor",
        ]
        | None
    ) = None
    anchor_id: str | None = None
    authored_anchor_xy: tuple[float, float] | None = None
    snap_residual_m: float | None = Field(default=None, ge=0.0)

    @model_validator(mode="after")
    def validate_anchor_evidence(self) -> "CandidateRoutePoint":
        if not all(math.isfinite(value) for value in self.position_xy):
            raise ValueError("route point position must be finite")
        fields = (
            self.semantic_anchor,
            self.anchor_kind,
            self.anchor_id,
            self.authored_anchor_xy,
            self.snap_residual_m,
        )
        if any(value is not None for value in fields) and not all(
            value is not None for value in fields
        ):
            raise ValueError(
                "semantic_anchor, authored_anchor_xy, and snap_residual_m "
                "must be authored together"
            )
        if self.semantic_anchor is None:
            return self
        if not str(self.semantic_anchor).strip():
            raise ValueError("semantic_anchor must be non-empty")
        assert self.authored_anchor_xy is not None
        assert self.snap_residual_m is not None
        if not all(math.isfinite(value) for value in self.authored_anchor_xy):
            raise ValueError("authored anchor position must be finite")
        measured = math.dist(self.authored_anchor_xy, self.position_xy)
        if not math.isclose(measured, self.snap_residual_m, abs_tol=0.002):
            raise ValueError(
                "route point snap_residual_m is stale; "
                f"declared {self.snap_residual_m:.3f} m, measured {measured:.3f} m"
            )
        return self


class CandidateRouteLeg(BaseModel):
    model_config = ConfigDict(extra="forbid")

    points: tuple[CandidateRoutePoint, ...] = Field(min_length=2)
    optimal_navigation_length_m: float | None = Field(
        default=None,
        gt=0.0,
        description=(
            "Measured collision-free A* length through this leg. Current "
            "builders persist it so difficulty never substitutes endpoint "
            "distance for a route around walls."
        ),
    )

    @model_validator(mode="after")
    def validate_point_ids(self) -> "CandidateRouteLeg":
        point_ids = tuple(point.point_id for point in self.points)
        if len(set(point_ids)) != len(point_ids):
            raise ValueError("route leg point ids must be unique")
        return self


class CandidateEpisodeRoute(BaseModel):
    """Typed evidence for the current two-leg benchmark route contract."""

    model_config = ConfigDict(extra="forbid")

    planning_clearance_radius_m: float
    occupancy_sha256: str
    planner_path_sha256: str
    collision_cell_count: int = Field(ge=0)
    navigate_to_target: CandidateRouteLeg
    transport_to_destination: CandidateRouteLeg
    milestone_stops: tuple[MilestoneRouteStop, ...] = ()
    optimal_navigation_length_m: float = Field(gt=0.0)

    @model_validator(mode="after")
    def validate_route_evidence_shape(self) -> "CandidateEpisodeRoute":
        for field_name in ("occupancy_sha256", "planner_path_sha256"):
            digest = str(getattr(self, field_name)).strip().lower()
            if len(digest) != 64 or any(
                character not in "0123456789abcdef" for character in digest
            ):
                raise ValueError(f"{field_name} must be lowercase SHA-256")
        all_ids = [
            point.point_id
            for leg in (self.navigate_to_target, self.transport_to_destination)
            for point in leg.points
        ]
        if len(set(all_ids)) != len(all_ids):
            raise ValueError("route point ids must be unique across both legs")
        return self


@dataclass(frozen=True)
class CandidateGeometryEvidence:
    scene_map: CandidateSceneMap
    route: CandidateEpisodeRoute
    occupancy_path: Path
    route_length_m: float
    navigate_path_xy: tuple[tuple[float, float], ...]
    transport_path_xy: tuple[tuple[float, float], ...]
    task_reference_paths_xy: dict[str, tuple[tuple[float, float], ...]] | None = None


def validate_candidate_payload(payload: object, *, owner: Path) -> None:
    """Reject incomplete or runtime-derived candidate fields before parsing."""

    if not isinstance(payload, dict):
        raise ValueError(f"{owner}: episode YAML must be a mapping")
    if "action_space" in payload:
        raise ValueError(f"{owner}: action_space was removed; use agent_action_space")
    robot = payload.get("robot")
    if isinstance(robot, dict):
        derived = sorted({"entity_name", "robot_type"} & set(robot))
        if derived:
            raise ValueError(
                f"{owner}: Robot runtime-derived fields are forbidden: {derived}"
            )
        if "start_orientation" not in robot:
            raise ValueError(
                f"{owner}: robot.start_orientation must be explicitly authored "
                "so spawn heading can be reviewed against task semantics"
            )
    for index, human in enumerate(payload.get("humans") or []):
        if not isinstance(human, dict):
            continue
        derived = sorted({"name", "character_name"} & set(human))
        if derived:
            raise ValueError(
                f"{owner}: humans[{index}] runtime-derived fields are forbidden: "
                f"{derived}"
            )
        if "start_yaw" not in human:
            raise ValueError(
                f"{owner}: humans[{index}].start_yaw must be explicitly authored "
                "so spawn heading can be reviewed against task semantics"
            )


def repository_file(
    root: Path,
    raw: object,
    *,
    owner: Path,
    field: str,
) -> Path:
    """Resolve a repository-relative artifact without permitting escape."""

    if not isinstance(raw, str) or not raw.strip():
        raise ValueError(f"{owner}: {field} must be a repository-relative path")
    path = (root / raw).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"{owner}: {field} leaves the repository: {raw!r}")
    if not path.is_file():
        raise ValueError(f"{owner}: {field} does not exist: {raw!r}")
    return path


def asset_file(
    root: Path,
    raw: object,
    *,
    owner: Path,
    field: str,
) -> Path:
    """Resolve an asset-root-relative path from episode world knowledge."""

    return repository_file(
        root,
        (Path("assets") / str(raw)).as_posix(),
        owner=owner,
        field=field,
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def route_points(route: CandidateEpisodeRoute) -> tuple[tuple[float, float], ...]:
    return tuple(
        point.position_xy
        for leg in (route.navigate_to_target, route.transport_to_destination)
        for point in leg.points
    )


def _resolved_leg_path(
    planner: Planner,
    leg: CandidateRouteLeg,
    *,
    owner: Path,
    leg_name: str,
) -> tuple[tuple[float, float], ...]:
    """Re-run frozen A* between every authored control point."""

    resolved: list[tuple[float, float]] = []
    for start, goal in zip(leg.points, leg.points[1:]):
        output = planner.plan(
            PlanInput(
                start_xy=start.position_xy,
                start_yaw_rad=0.0,
                goal_xy=goal.position_xy,
            )
        )
        if not route_reaches_exact_goal(
            output,
            start_xy=start.position_xy,
            goal_xy=goal.position_xy,
        ):
            raise ValueError(
                f"{owner}: A* failed exact G1 route for {leg_name}: "
                f"{start.point_id} -> {goal.point_id}; status={output.status}"
            )
        segment = complete_path(
            output,
            start_xy=start.position_xy,
            goal_xy=goal.position_xy,
        )
        for point in segment:
            if not resolved or math.dist(resolved[-1], point) > 1e-9:
                resolved.append(point)
    return tuple(resolved)


def _planner_path_sha256(
    navigate_path_xy: tuple[tuple[float, float], ...],
    transport_path_xy: tuple[tuple[float, float], ...],
) -> str:
    payload = {
        "navigate_to_target": [list(point) for point in navigate_path_xy],
        "transport_to_destination": [list(point) for point in transport_path_xy],
    }
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_anchor(
    point: CandidateRoutePoint,
    *,
    owner: Path,
    label: str,
) -> tuple[float, float]:
    if (
        point.semantic_anchor is None
        or point.anchor_kind is None
        or point.anchor_id is None
        or point.authored_anchor_xy is None
        or point.snap_residual_m is None
    ):
        raise ValueError(
            f"{owner}: {label} must record semantic anchor, resolved stance, "
            "and snap residual"
        )
    return point.authored_anchor_xy


def _human_anchor_at_end(
    episode: TARSphereEpisode,
    human_id: str,
    *,
    owner: Path,
) -> tuple[float, float]:
    humans = {human.id: human for human in episode.humans}
    if human_id not in humans:
        raise ValueError(f"{owner}: route anchor Human {human_id!r} is unknown")
    anchor = tuple(float(value) for value in humans[human_id].start_position[:2])
    if episode.evaluation is None:
        return anchor
    for milestone in episode.evaluation.milestones:
        for action in milestone.event.human_actions:
            if (
                str(action.get("human") or "") == human_id
                and str(action.get("command") or "") == "goto"
            ):
                target = action.get("target")
                if not isinstance(target, (list, tuple)) or len(target) < 2:
                    raise ValueError(
                        f"{owner}: Human goto target must contain world XY"
                    )
                anchor = (float(target[0]), float(target[1]))
    return anchor


def _resolved_semantic_anchor(
    episode: TARSphereEpisode,
    point: CandidateRoutePoint,
    *,
    owner: Path,
) -> tuple[float, float]:
    """Resolve a typed route anchor from current episode authority."""

    assert point.anchor_kind is not None
    assert point.anchor_id is not None
    world = episode.world_knowledge
    if point.anchor_kind == "robot_start":
        if point.anchor_id != episode.robot.entity_name:
            raise ValueError(f"{owner}: robot_start anchor_id is not the Robot")
        return tuple(float(value) for value in episode.robot.start_position[:2])
    if point.anchor_kind == "grasp_stance":
        physical_task = world.get("physical_task")
        grasp = physical_task.get("grasp") if isinstance(physical_task, dict) else None
        if not isinstance(grasp, dict):
            raise ValueError(
                f"{owner}: grasp_stance anchor requires physical_task.grasp"
            )
        if point.anchor_id != str(grasp.get("target_object_id") or ""):
            raise ValueError(f"{owner}: grasp_stance anchor_id is not the grasp target")
        stance = grasp.get("robot_stance_xy")
        if not isinstance(stance, (list, tuple)) or len(stance) != 2:
            raise ValueError(f"{owner}: grasp robot_stance_xy is invalid")
        return (float(stance[0]), float(stance[1]))
    if point.anchor_kind == "human_position":
        return _human_anchor_at_end(episode, point.anchor_id, owner=owner)
    if point.anchor_kind == "object_position":
        objects = {
            str(item.get("object_id") or ""): item
            for item in (world.get("objects") or [])
            if isinstance(item, dict)
        }
        item = objects.get(point.anchor_id)
        position = item.get("position_xyz") if isinstance(item, dict) else None
        if not isinstance(position, (list, tuple)) or len(position) < 2:
            raise ValueError(
                f"{owner}: object_position anchor {point.anchor_id!r} is unknown"
            )
        return (float(position[0]), float(position[1]))
    if point.anchor_kind == "detector_position":
        if episode.evaluation is None:
            raise ValueError(f"{owner}: detector_position requires evaluation")
        condition_matches = [
            condition
            for condition in episode.evaluation.conditions()
            if condition.condition_id == point.anchor_id
            and condition.detector == "robot_in_place"
        ]
        reaction_matches = [
            milestone.outcome
            for milestone in episode.evaluation.milestones
            if milestone.milestone_id == point.anchor_id
            and milestone.outcome.evaluator == "robot_in_place"
        ]
        matches = condition_matches + reaction_matches
        if len(matches) != 1:
            raise ValueError(
                f"{owner}: detector_position anchor {point.anchor_id!r} "
                "must identify one robot_in_place condition or reaction"
            )
        binding = matches[0]
        arguments = getattr(binding, "arguments", None)
        if arguments is None:
            arguments = binding.ground_truth
        position = arguments["position"]
        return (float(position[0]), float(position[1]))
    anchors = world.get("spatial_anchors")
    anchor = anchors.get(point.anchor_id) if isinstance(anchors, dict) else None
    if not isinstance(anchor, (list, tuple)) or len(anchor) < 2:
        raise ValueError(
            f"{owner}: scene_anchor {point.anchor_id!r} is not declared in "
            "world_knowledge.spatial_anchors"
        )
    return (float(anchor[0]), float(anchor[1]))


def _final_recipient_anchor(
    episode: TARSphereEpisode,
    *,
    owner: Path,
) -> tuple[str, tuple[float, float]]:
    """Resolve the final handover Human and their last public destination."""

    assert episode.evaluation is not None
    terminal_reactions = [
        milestone.outcome
        for milestone in episode.evaluation.milestones
        if milestone.outcome.evaluator == "handover_accepted"
    ]
    if len(terminal_reactions) != 1:
        raise ValueError(
            f"{owner}: fetch/handover candidate requires exactly one "
            "handover_accepted reaction"
        )
    recipient_id = str(
        final_outcome_ground_truth(episode.evaluation, "task_completed")["human_id"]
    )
    return recipient_id, _human_anchor_at_end(episode, recipient_id, owner=owner)


def _semantic_objective_binding(
    episode: TARSphereEpisode,
    *,
    owner: Path,
) -> tuple[str, str]:
    """Derive the outbound information source from scored milestone truth."""

    assert episode.evaluation is not None
    candidates: list[tuple[str, str]] = []
    for milestone in episode.evaluation.milestones:
        if milestone.kind != "scored":
            continue
        reaction = milestone.outcome
        truth = final_outcome_ground_truth(episode.evaluation, milestone.milestone_id)
        if reaction.evaluator == "robot_in_place":
            candidates.append(("detector_position", milestone.milestone_id))
        elif reaction.evaluator in {
            "agent_interaction_selected",
            "information_received",
        }:
            candidates.append(("human_position", str(truth["human_id"])))
        elif reaction.evaluator == "visual_targets_observed":
            candidates.append(("scene_anchor", str(truth["scene_anchor_id"])))
    if not candidates:
        raise ValueError(
            f"{owner}: semantic acquisition requires a scored onsite "
            "visual_targets_observed, delivered Human information, Human ask, "
            "or robot_in_place reaction before "
            "the information report"
        )
    revised = next(
        (
            milestone
            for milestone in episode.evaluation.milestones
            if milestone.milestone_id == "revised_target_observed"
        ),
        None,
    )
    if revised is not None:
        truth = revised.outcome.ground_truth
        if revised.outcome.evaluator == "visual_targets_observed":
            return "scene_anchor", str(truth["scene_anchor_id"])
        if revised.outcome.evaluator == "human_visible":
            return "human_position", str(truth["human_id"])
        raise ValueError(f"{owner}: revised_target_observed has no routable target")
    return candidates[-1]


def _validate_endpoint_residual(
    point: CandidateRoutePoint,
    *,
    owner: Path,
    label: str,
) -> None:
    """Bound route stance displacement according to its typed authority."""

    assert point.anchor_kind is not None
    assert point.snap_residual_m is not None
    residual = point.snap_residual_m
    if point.anchor_kind in {
        "robot_start",
        "grasp_stance",
        "detector_position",
        "scene_anchor",
    }:
        if residual > AUTHORITATIVE_STANCE_TOLERANCE_M:
            raise ValueError(
                f"{owner}: {label} must coincide with its authoritative "
                f"{point.anchor_kind} (residual {residual:.3f} m)"
            )
        return
    if point.anchor_kind == "human_position":
        minimum = ROBOT_HUMAN_CENTER_CLEARANCE_M
        maximum = HUMAN_COMMUNICATION_RADIUS_M
    elif point.anchor_kind == "object_position":
        minimum = G1_NAVIGATION_FOOTPRINT_RADIUS_M
        maximum = MAX_SEMANTIC_OBSERVATION_STANCE_DISTANCE_M
    else:  # pragma: no cover - Literal exhaustiveness
        raise AssertionError(point.anchor_kind)
    if residual < minimum or residual > maximum:
        raise ValueError(
            f"{owner}: {label} {point.anchor_kind} stance residual must be in "
            f"[{minimum:.3f}, {maximum:.3f}] m; got {residual:.3f} m"
        )


def _validate_task_route_bindings(
    episode: TARSphereEpisode,
    route: CandidateEpisodeRoute,
    *,
    owner: Path,
) -> None:
    """Bind the route to actual task GT and a valid DAG linearization."""

    if episode.evaluation is None:
        raise ValueError(f"{owner}: route binding requires official evaluation")
    start = route.navigate_to_target.points[0]
    objective = route.navigate_to_target.points[-1]
    terminal = route.transport_to_destination.points[-1]
    start_anchor = _require_anchor(start, owner=owner, label="route start")
    objective_anchor = _require_anchor(objective, owner=owner, label="route objective")
    terminal_anchor = _require_anchor(terminal, owner=owner, label="route terminal")
    for label, point, declared_anchor in (
        ("route start", start, start_anchor),
        ("route objective", objective, objective_anchor),
        ("route terminal", terminal, terminal_anchor),
    ):
        resolved_anchor = _resolved_semantic_anchor(episode, point, owner=owner)
        if math.dist(declared_anchor, resolved_anchor) > 0.002:
            raise ValueError(f"{owner}: {label} anchor drifted from its typed binding")
        _validate_endpoint_residual(point, owner=owner, label=label)

    mainline = episode.evaluation.task_profile.mainline
    if mainline is TaskMainline.FETCH_AND_DELIVER:
        if objective.anchor_kind != "grasp_stance":
            raise ValueError(f"{owner}: fetch objective must bind grasp_stance")
        recipient_id, _ = _final_recipient_anchor(episode, owner=owner)
        if (
            terminal.anchor_kind != "human_position"
            or terminal.anchor_id != recipient_id
        ):
            raise ValueError(
                f"{owner}: handover-offer terminal must bind final recipient "
                f"{recipient_id!r}"
            )
        assert terminal.snap_residual_m is not None
        if not (
            HANDOVER_MIN_DISTANCE_M
            <= terminal.snap_residual_m
            <= HANDOVER_MAX_DISTANCE_M
        ):
            raise ValueError(
                f"{owner}: handover terminal stance must be inside the "
                f"runtime handover band [{HANDOVER_MIN_DISTANCE_M:.3f}, "
                f"{HANDOVER_MAX_DISTANCE_M:.3f}] m; got "
                f"{terminal.snap_residual_m:.3f} m"
            )
    elif mainline is TaskMainline.NAVIGATE_AND_REPORT:
        reports = [
            milestone.outcome
            for milestone in episode.evaluation.milestones
            if milestone.outcome.evaluator == "information_report_score"
        ]
        expected_kind, expected_id = _semantic_objective_binding(episode, owner=owner)
        if objective.anchor_kind != expected_kind or objective.anchor_id != expected_id:
            raise ValueError(
                f"{owner}: navigation objective must bind the scored onsite "
                f"target {expected_kind}:{expected_id}"
            )
        if reports:
            recipient_id = str(
                final_outcome_ground_truth(episode.evaluation, "task_completed")[
                    "human_id"
                ]
            )
            if (
                terminal.anchor_kind != "human_position"
                or terminal.anchor_id != recipient_id
            ):
                raise ValueError(
                    f"{owner}: navigation report terminal must bind recipient "
                    f"{recipient_id!r}"
                )
        elif (
            terminal.anchor_kind != objective.anchor_kind
            or terminal.anchor_id != objective.anchor_id
        ):
            raise ValueError(
                f"{owner}: navigation without a report must complete at its "
                "scored onsite objective"
            )
    else:  # pragma: no cover - enum exhaustiveness
        raise AssertionError(mainline)

    milestone_ids = tuple(
        milestone.milestone_id for milestone in episode.evaluation.milestones
    )
    bound_ids = tuple(stop.milestone_id for stop in route.milestone_stops)
    if bound_ids != milestone_ids:
        raise ValueError(
            f"{owner}: milestone_stops must bind every milestone in the "
            "declared topological order"
        )
    route_point_ids = {
        point.point_id
        for leg in (route.navigate_to_target, route.transport_to_destination)
        for point in leg.points
    }
    dangling = sorted(
        stop.route_point_id
        for stop in route.milestone_stops
        if stop.route_point_id not in route_point_ids
    )
    if dangling:
        raise ValueError(
            f"{owner}: milestone_stops reference unknown route points {dangling}"
        )


def _validate_human_routes(episode, occupancy, *, owner: Path) -> None:
    """Validate exact actor starts and destinations using the runtime route contract."""
    from tarsphere.infra.control.kinematic_human_route import resolve_human_route

    planner = Planner(occupancy, clearance_radius_m=HUMAN_BODY_RADIUS_M)
    humans = {human.id: human for human in episode.humans}
    poses = {human.id: list(human.start_position) for human in episode.humans}
    for human_id, pose in poses.items():
        if humans[human_id].scene_person_prim_path:
            # The native person's exact mesh is deliberately part of the
            # pinned occupancy. Its center is therefore not expected to be a
            # free spawn cell, and schema/runtime prohibit moving it.
            continue
        if not planner.is_line_collision_free(pose[:2], pose[:2]):
            raise ValueError(
                f"{owner}: Human {human_id} spawn intersects scene occupancy"
            )
    for node in episode.evaluation.milestones:
        for action in node.event.human_actions:
            if action["command"] != "goto":
                continue
            human_id = action["human"]
            try:
                resolve_human_route(
                    planner,
                    start=poses[human_id],
                    target=action["target"],
                    waypoints=action.get("waypoints"),
                )
            except (RuntimeError, ValueError) as exc:
                raise ValueError(
                    f"{owner}: Human {human_id} exact authored route: {exc}"
                ) from exc
            poses[human_id] = list(action["target"])


def _validate_required_blocker_milestones(
    episode: TARSphereEpisode,
    route: CandidateEpisodeRoute,
    *,
    occupancy: OfficialOccupancyBevMap,
    owner: Path,
) -> None:
    """Allow a required move-aside node only for a geometric cut blocker.

    Optional social tactics stay outside the required milestone DAG. An Agent
    question followed by the same Human moving is therefore a
    required blocker claim: removing that Human disk must be necessary for any
    route from deployment to the authored objective to exist.
    """

    evaluation = episode.evaluation
    if evaluation is None:
        return
    profile = evaluation.task_profile
    if (
        profile.capability_slice is CapabilitySlice.DISRUPTION_ADAPTATION
        and profile.subtype == "spatial_blocking"
    ):
        by_id = {node.milestone_id: node for node in evaluation.milestones}
        block = by_id[profile.blocking_milestone_id]
        clear = by_id[profile.clear_milestone_id]

        def human_destination(node: Any) -> tuple[float, float]:
            motion = next(
                action
                for action in node.event.human_actions
                if action.get("command") == "goto"
                and action.get("human") == profile.blocking_human_id
            )
            return float(motion["target"][0]), float(motion["target"][1])

        other_humans = tuple(
            (
                float(human.start_position[0]),
                float(human.start_position[1]),
                HUMAN_BODY_RADIUS_M,
            )
            for human in episode.humans
            if human.static and not human.scene_person_prim_path
        )

        def route_status(
            blocker_xy: tuple[float, float],
            goal_xy: tuple[float, float],
        ) -> str:
            mask = official_runtime_navigation_mask(
                occupancy.occupancy_path,
                clearance_radius_m=route.planning_clearance_radius_m,
                static_obstacles=(*other_humans, (*blocker_xy, HUMAN_BODY_RADIUS_M)),
            )
            planner = Planner(
                occupancy,
                backend="numpy",
                simplify=True,
                time_limit_s=30.0,
                clearance_radius_m=route.planning_clearance_radius_m,
                prepared_navigable_mask=mask,
            )
            result = planner.plan(
                PlanInput(
                    start_xy=episode.robot.start_position[:2],
                    start_yaw_rad=float(
                        (episode.world_knowledge.get("physical_task") or {}).get(
                            "robot_start_yaw_rad",
                            0.0,
                        )
                    ),
                    goal_xy=goal_xy,
                )
            )
            return str(result.status)

        target_xy = route.navigate_to_target.points[-1].position_xy
        blocked_xy = human_destination(block)
        clear_xy = human_destination(clear)
        if route_status(blocked_xy, target_xy) != "unreachable":
            raise ValueError(
                f"{owner}: spatial_blocking Human does not cut every G1-clear "
                "route to the authored objective"
            )
        if route_status(clear_xy, target_xy) != "ok":
            raise ValueError(
                f"{owner}: spatial_blocking clear pose does not restore "
                "a G1-clear route to the authored objective"
            )
        route_points_by_id = {
            point.point_id: point.position_xy
            for leg in (route.navigate_to_target, route.transport_to_destination)
            for point in leg.points
        }
        request_stop = next(
            stop
            for stop in route.milestone_stops
            if stop.milestone_id == profile.clearance_request_milestone_id
        )
        contact_xy = route_points_by_id[request_stop.route_point_id]
        if route_status(blocked_xy, contact_xy) not in {"ok", "already_there"}:
            raise ValueError(
                f"{owner}: spatial_blocking request stance is not reachable "
                "from the approach side while the route is blocked"
            )
        contact_distance_m = math.dist(blocked_xy, contact_xy)
        minimum_nonoverlap_m = HUMAN_BODY_RADIUS_M + route.planning_clearance_radius_m
        if not (
            minimum_nonoverlap_m < contact_distance_m < HUMAN_COMMUNICATION_RADIUS_M
        ):
            raise ValueError(
                f"{owner}: spatial_blocking request stance must avoid body "
                "overlap and remain within communication range"
            )
    humans = {str(human.id): human for human in episode.humans}
    children: dict[str, list[Any]] = {
        milestone.milestone_id: [] for milestone in evaluation.milestones
    }
    for milestone in evaluation.milestones:
        for precondition in milestone.preconditions:
            if isinstance(precondition, MilestoneReference):
                children[precondition.milestone_ref].append(milestone)

    for milestone in evaluation.milestones:
        truth = milestone.outcome.ground_truth
        if not (milestone.outcome.evaluator == "agent_interaction_selected"):
            continue
        human_id = str(truth.get("human_id") or "")
        releases = [
            child
            for child in children[milestone.milestone_id]
            if any(
                str(action.get("human") or "") == human_id
                and str(action.get("command") or "") == "goto"
                for action in child.event.human_actions
            )
        ]
        if not releases:
            continue
        human = humans.get(human_id)
        if human is None:  # Base schema reports the clearer binding error.
            continue
        blocker_planner = Planner(
            occupancy,
            backend="numpy",
            simplify=True,
            time_limit_s=30.0,
            clearance_radius_m=route.planning_clearance_radius_m,
            prepared_navigable_mask=official_runtime_navigation_mask(
                occupancy.occupancy_path,
                clearance_radius_m=route.planning_clearance_radius_m,
                static_obstacles=tuple(
                    (
                        float(actor.start_position[0]),
                        float(actor.start_position[1]),
                        HUMAN_BODY_RADIUS_M,
                    )
                    for actor in episode.humans
                    if actor.static or actor.id == human_id
                ),
            ),
        )
        blocked = blocker_planner.plan(
            PlanInput(
                start_xy=episode.robot.start_position[:2],
                start_yaw_rad=float(
                    (episode.world_knowledge.get("physical_task") or {}).get(
                        "robot_start_yaw_rad",
                        0.0,
                    )
                ),
                goal_xy=route.navigate_to_target.points[-1].position_xy,
            )
        )
        if blocked.status != "unreachable":
            raise ValueError(
                f"{owner}: milestone {milestone.milestone_id!r} makes asking "
                f"{human_id!r} to move a required progress point, but a G1 "
                "route to the objective remains feasible around that Human. "
                "Keep the request as an optional tactic outside milestones."
            )


def validate_candidate_geometry(
    episode: TARSphereEpisode,
    *,
    root: Path,
    owner: Path,
) -> CandidateGeometryEvidence:
    """Verify authored segments against the actual G1-clear occupancy mask."""

    world = episode.world_knowledge
    deprecated = sorted(DEPRECATED_WORLD_KEYS & set(world))
    if deprecated:
        raise ValueError(
            f"{owner}: old authoring-dialect keys are forbidden: {deprecated}"
        )
    scene_map = CandidateSceneMap.model_validate(world.get("scene_map"))
    route = CandidateEpisodeRoute.model_validate(world.get("episode_route"))
    if episode.evaluation.task_profile.capability_slice.value == "human_coordination":
        from tarsphere.bench.spatial.coordination_route import (
            validate_coordination_contacts,
        )

        validate_coordination_contacts(episode.model_dump(mode="json"), root=root)
    for field in ("rgb_bev", "projection"):
        asset_file(
            root,
            getattr(scene_map, field),
            owner=owner,
            field=f"scene_map.{field}",
        )
    occupancy_path = asset_file(
        root,
        scene_map.occupancy,
        owner=owner,
        field="scene_map.occupancy",
    )
    observed_sha = sha256_file(occupancy_path)
    if str(episode.scene.asset_path).endswith("/start_result_navigation.usd"):
        from tarsphere.bench.certification.scene import (
            validate_scene_occupancy_provenance,
        )

        validate_scene_occupancy_provenance(
            occupancy_path,
            root=root,
            scene_usd=root / "assets" / episode.scene.asset_path,
            verify_scene_files=False,
        )
    if route.occupancy_sha256 != observed_sha:
        raise ValueError(
            f"{owner}: episode_route.occupancy_sha256 is stale; observed {observed_sha}"
        )
    if not math.isclose(
        route.planning_clearance_radius_m,
        G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        abs_tol=1e-9,
    ):
        raise ValueError(
            f"{owner}: route clearance must equal the G1 radius "
            f"{G1_NAVIGATION_FOOTPRINT_RADIUS_M:.3f} m"
        )

    occupancy = OfficialOccupancyBevMap(occupancy_path=occupancy_path)
    _validate_human_routes(episode, occupancy, owner=owner)
    planner = Planner(
        occupancy,
        backend="numpy",
        simplify=True,
        time_limit_s=30.0,
        clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
        prepared_navigable_mask=official_runtime_navigation_mask(
            occupancy_path,
            clearance_radius_m=G1_NAVIGATION_FOOTPRINT_RADIUS_M,
            static_obstacles=tuple(
                (
                    float(human.start_position[0]),
                    float(human.start_position[1]),
                    HUMAN_BODY_RADIUS_M,
                )
                for human in episode.humans
                if human.static and not human.scene_person_prim_path
            ),
        ),
        cache_static_nav=True,
    )
    navigable = planner.active_navigable_mask()
    _validate_required_blocker_milestones(
        episode,
        route,
        occupancy=occupancy,
        owner=owner,
    )
    navigate_path = _resolved_leg_path(
        planner,
        route.navigate_to_target,
        owner=owner,
        leg_name="navigate_to_target",
    )
    transport_path = _resolved_leg_path(
        planner,
        route.transport_to_destination,
        owner=owner,
        leg_name="transport_to_destination",
    )
    observed_planner_sha = _planner_path_sha256(
        navigate_path,
        transport_path,
    )
    if route.planner_path_sha256 != observed_planner_sha:
        raise ValueError(
            f"{owner}: episode_route.planner_path_sha256 is stale; "
            f"observed {observed_planner_sha}"
        )
    leg_paths = {
        "navigate_to_target": navigate_path,
        "transport_to_destination": transport_path,
    }
    for leg_name, leg in (
        ("navigate_to_target", route.navigate_to_target),
        ("transport_to_destination", route.transport_to_destination),
    ):
        if leg.optimal_navigation_length_m is None:
            continue
        measured_leg_length = path_length(leg_paths[leg_name])
        if not math.isclose(
            leg.optimal_navigation_length_m,
            measured_leg_length,
            abs_tol=0.002,
        ):
            raise ValueError(
                f"{owner}: {leg_name}.optimal_navigation_length_m is stale; "
                f"declared {leg.optimal_navigation_length_m:.3f} m, measured "
                f"{measured_leg_length:.3f} m"
            )
    blocked_cells = 0
    route_length = path_length(navigate_path) + path_length(transport_path)
    for points in (navigate_path, transport_path):
        for start, end in zip(points, points[1:]):
            cells = _supercover_world_segment_cells(
                occupancy,
                start,
                end,
                navigable.shape,
            )
            if cells is None:
                blocked_cells += 1
                continue
            for row, column in cells:
                row, column = int(row), int(column)
                if not (
                    0 <= row < navigable.shape[0] and 0 <= column < navigable.shape[1]
                ) or not bool(navigable[row, column]):
                    blocked_cells += 1
    if route.collision_cell_count != blocked_cells:
        raise ValueError(
            f"{owner}: declared collision_cell_count={route.collision_cell_count} "
            f"but measured {blocked_cells}"
        )
    if blocked_cells:
        raise ValueError(f"{owner}: route intersects {blocked_cells} G1-blocked cells")
    if not math.isclose(
        route.optimal_navigation_length_m,
        route_length,
        abs_tol=0.002,
    ):
        raise ValueError(
            f"{owner}: route length is stale; declared "
            f"{route.optimal_navigation_length_m:.3f} m, measured "
            f"{route_length:.3f} m"
        )
    _validate_task_route_bindings(episode, route, owner=owner)
    if (
        math.dist(
            route.navigate_to_target.points[-1].position_xy,
            route.transport_to_destination.points[0].position_xy,
        )
        > 0.002
    ):
        raise ValueError(f"{owner}: route legs are not continuous at the objective")
    reference_paths = None
    comparison = world.get("task_update_route_comparison")
    is_ta = (
        episode.evaluation is not None
        and episode.evaluation.task_profile.capability_slice
        is CapabilitySlice.TASK_UPDATE
    )
    if is_ta:
        if not isinstance(comparison, dict) or set(comparison) != {
            "initial_task_reference_xy",
            "revised_task_reference_xy",
        }:
            raise ValueError(
                f"{owner}: TA requires both authoring-BEV reference routes"
            )
        reference_paths = {}
        for name, waypoints in comparison.items():
            if not isinstance(waypoints, list) or len(waypoints) < 2:
                raise ValueError(f"{owner}: {name} needs at least two waypoints")
            leg = CandidateRouteLeg.model_validate(
                {
                    "points": [
                        {"point_id": f"reference_{index}", "position_xy": point}
                        for index, point in enumerate(waypoints)
                    ],
                }
            )
            reference_paths[name] = _resolved_leg_path(
                planner,
                leg,
                owner=owner,
                leg_name=name,
            )
    elif comparison is not None:
        raise ValueError(f"{owner}: task route comparison belongs only to TA")
    return CandidateGeometryEvidence(
        scene_map=scene_map,
        route=route,
        occupancy_path=occupancy_path,
        route_length_m=route_length,
        navigate_path_xy=navigate_path,
        transport_path_xy=transport_path,
        task_reference_paths_xy=reference_paths,
    )


def validate_candidate_publication(
    episode: TARSphereEpisode,
    *,
    root: Path,
    owner: Path,
) -> CandidateGeometryEvidence:
    """Require all non-runtime evidence needed to publish a candidate."""

    if episode.evaluation is None:
        raise ValueError(f"{owner}: candidate omits official evaluation")
    if episode.runtime.max_physics_steps is not None:
        raise ValueError(
            f"{owner}: published evaluation candidates must not define "
            "runtime.max_physics_steps"
        )
    if episode.authoring is None:
        raise ValueError(f"{owner}: candidate omits authoring evidence")
    required_metadata = ("scene_context",)
    missing = [name for name in required_metadata if not episode.metadata.get(name)]
    if missing:
        raise ValueError(f"{owner}: candidate metadata is missing {missing!r}")
    geometry = validate_candidate_geometry(episode, root=root, owner=owner)
    profile = episode.evaluation.task_profile
    if profile.capability_slice is CapabilitySlice.BASE:
        derive_episode_difficulty(episode)
    elif episode.authoring.difficulty_factors:
        raise ValueError(
            f"{owner}: capability variants inherit difficulty from their "
            "matching Base and cannot author difficulty_factors"
        )
    if AuthoringGate.OVERLAY_REVIEWED not in episode.authoring.achieved_gates:
        raise ValueError(f"{owner}: candidate overlay has not been reviewed")
    overlay = repository_file(
        root,
        episode.authoring.route_overlay,
        owner=owner,
        field="authoring.route_overlay",
    )
    for gate, record in episode.authoring.external_evidence.items():
        artifact = repository_file(
            root,
            record.artifact_path,
            owner=owner,
            field=f"authoring.external_evidence.{gate.value}",
        )
        observed_artifact_sha = sha256_file(artifact)
        if observed_artifact_sha != record.artifact_sha256:
            raise ValueError(
                f"{owner}: {gate.value} artifact digest is stale; "
                f"observed {observed_artifact_sha}"
            )
        try:
            result = json.loads(artifact.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError) as exc:
            raise ValueError(f"{owner}: {gate.value} artifact must be JSON") from exc
        if not isinstance(result, dict):
            raise ValueError(
                f"{owner}: {gate.value} artifact must contain a JSON object"
            )
        _validate_external_result(
            episode,
            gate=gate,
            record=record,
            result=result,
            owner=owner,
        )
    return geometry


def _validate_external_result(
    episode: TARSphereEpisode,
    *,
    gate: AuthoringGate,
    record: Any,
    result: dict[str, Any],
    owner: Path,
) -> None:
    """Confirm a promotion record against the referenced result payload."""

    expected = {
        "episode_id": record.episode_id,
        "scene_id": record.scene_id,
        "scene_server_git_sha": record.git_sha,
        "loaded_scene_asset_path": episode.scene.asset_path,
    }
    mismatched = {
        key: (result.get(key), value)
        for key, value in expected.items()
        if str(result.get(key) or "") != str(value)
    }
    if mismatched:
        raise ValueError(
            f"{owner}: {gate.value} artifact identity mismatch: {mismatched}"
        )
    if gate is AuthoringGate.ORACLE_VALID:
        passed = bool(
            result.get("agent_mode") == "golden-oracle"
            and result.get("certification_stage") == "C"
            and result.get("certification_passed") is True
        )
    elif gate is AuthoringGate.RUNTIME_VALID:
        task_progress = (
            (result.get("evaluation_report") or {}).get("task_progress")
            if isinstance(result.get("evaluation_report"), dict)
            else None
        )
        passed = bool(
            result.get("agent_mode") == "model"
            and result.get("success") is True
            and isinstance(task_progress, dict)
            and task_progress.get("success") is True
        )
    elif gate is AuthoringGate.GOLDEN:
        passed = bool(
            result.get("authoring_gate") == "golden"
            and result.get("passed") is True
            and result.get("oracle_artifact_sha256")
            == episode.authoring.external_evidence[
                AuthoringGate.ORACLE_VALID
            ].artifact_sha256
            and result.get("runtime_artifact_sha256")
            == episode.authoring.external_evidence[
                AuthoringGate.RUNTIME_VALID
            ].artifact_sha256
        )
    else:
        raise ValueError(
            f"{owner}: external artifact is not valid for gate {gate.value}"
        )
    if not passed:
        raise ValueError(
            f"{owner}: {gate.value} artifact does not prove a passing gate"
        )


__all__ = [
    "CandidateEpisodeRoute",
    "CandidateGeometryEvidence",
    "CandidateRouteLeg",
    "CandidateRoutePoint",
    "CandidateSceneMap",
    "asset_file",
    "repository_file",
    "route_points",
    "sha256_file",
    "validate_candidate_geometry",
    "validate_candidate_payload",
    "validate_candidate_publication",
]
